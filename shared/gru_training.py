"""Reusable dataset reader and temporal-safe augmentation for GRU notebook."""
from __future__ import annotations

import io
import json
from collections import defaultdict
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from shared.gru_model import causal_windows
from shared.pose_training import FLIP13


def read_index(archive):
    with ZipFile(archive) as z:
        names = [p[:-len('index.csv')] for p in z.namelist() if p.endswith('/index.csv')]
        if len(names) != 1:
            raise ValueError('Ambiguous processed index.csv')
        prefix = names[0]
        table = pd.read_csv(io.BytesIO(z.read(prefix + 'index.csv')))
        contract = json.loads(z.read(prefix + 'feature_contract.json'))
        split = json.loads(z.read(prefix + 'split_manifest.json'))
    if (contract.get('shape_per_frame') != [13,4] or contract.get('flatten_dim') != 52
            or split.get('strategy') != 'subject_wise_frozen_v2'):
        raise ValueError('Unexpected processed V2 feature contract or split')
    if not table.clip_id.is_unique or set(table.split) != {'train','val','test'}:
        raise ValueError('Incomplete/duplicate frozen split')
    for name, label in [('train','train_people'),('val','validation_people'),('test','test_people')]:
        if set(table.loc[table.split == name, 'person_id']) != set(split[label]):
            raise ValueError('Subject-wise split leakage/mismatch')
    if table.groupby('source_group_id').split.nunique().max() != 1:
        raise ValueError('Source take leaks across splits')
    return table, contract, split, prefix


def load_clips(archive, table, prefix, split_names, window):
    """Only read requested split. Never load held-out test during model selection."""
    result = []
    with ZipFile(archive) as z:
        for row in table[table.split.isin(split_names)].itertuples(index=False):
            if row.feature_path != f'features/{row.clip_id}.npz':
                raise ValueError('Unexpected feature_path in processed archive')
            with np.load(io.BytesIO(z.read(prefix + row.feature_path)), allow_pickle=False) as content:
                x = np.asarray(content['x'], dtype=np.float32)
                times = np.asarray(content['relative_ms']) if 'relative_ms' in content else None
            if x.shape != (int(row.frames),13,4) or not np.isfinite(x).all():
                raise ValueError('Invalid clip data')
            if times is not None and (len(times) != len(x) or np.any(np.diff(times) < 0)):
                raise ValueError('Invalid clip timestamps')
            windows, valid = causal_windows(x, window)
            result.append(dict(clip_id=str(row.clip_id),person=str(row.person_id),
                camera=str(row.camera_id),source=str(row.source_group_id),label=int(row.label),
                split=str(row.split),windows=windows,valid=valid,timestamps=times))
    return result


def quota_anchors(clips, epoch, seed, max_per_source_camera=48):
    """Cap correlated windows from same original source take and camera."""
    rng = np.random.default_rng(seed * 1000003 + epoch)
    groups = defaultdict(list)
    for cidx,c in enumerate(clips):
        groups[(clips[cidx]['source'],clips[cidx]['camera'])].extend(
            (cidx,int(i)) for i in np.flatnonzero(c['valid']))
    indices = []
    for key in sorted(groups):
        candidates = groups[key]
        pick = rng.choice(len(candidates),size=min(len(candidates),max_per_source_camera),replace=False)
        indices.extend(candidates[int(i)] for i in pick)
    rng.shuffle(indices)
    return indices


def augment_windows(batch, generator=None, flip_prob=0.5, jitter_std=0.008):
    """One spatial transform per entire window; preserve causal/missing-data masks."""
    if batch.ndim != 3 or batch.shape[2] != 52:
        raise ValueError('Expected [B,L,52]')
    B,L,_=batch.shape
    x=batch.reshape(B,L,13,4).clone()
    device=x.device
    if flip_prob:
        flips=torch.rand(B,device=device,generator=generator)<flip_prob
        if flips.any():
            swapped=x[flips][:,:,list(FLIP13),:].clone()
            swapped[:,:,:,0] *= -1.
            x[flips]=swapped
    if jitter_std:
        jitter=torch.randn((B,1,13,2),device=device,generator=generator)*jitter_std
        x[:,:,:,:2] += jitter*x[:,:,:,3:4]
    return x.reshape(B,L,52)
