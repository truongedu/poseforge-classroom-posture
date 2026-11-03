"""Clean-pose training: lineage, frozen split, sampling, augmentation, review packaging.

Only pure functions, independent from Collector/Qt/Ultralytics. No dataset writes.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import os
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

FLIP13 = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11)
LABELS = tuple(range(6))


def source_group_id(person, session, label, source_take_id):
    """Do not group simply by `take_001`: that ID repeats across people/labels."""
    return f"{person}/{session}/label_{int(label)}/{source_take_id}"


def read_clean_lineage(zip_path, catalog):
    """Validate exact clip-camera lineage from Phase 1.6A clean export."""
    with ZipFile(zip_path) as z:
        roots = [p[:-len('curation_manifest.json')] for p in z.namelist()
                 if p.endswith('Real_Posture_Dataset/curation_manifest.json')]
        if len(roots) != 1:
            raise ValueError('Clean dataset curation_manifest.json missing/ambiguous')
        manifest = json.loads(z.read(roots[0] + 'curation_manifest.json'))
    records = manifest.get('recordings', [])
    by_clip = {}
    for r in records:
        cid = str(r['clip_id'])
        if cid in by_clip:
            raise ValueError('Duplicate lineage: ' + cid)
        by_clip[cid] = r
    if set(by_clip) != set(catalog.clip_id):
        raise ValueError('Clean lineage != labels.csv clip IDs')
    rows = []
    ranges = defaultdict(list)
    for item in catalog.itertuples(index=False):
        r = by_clip[item.clip_id]
        sid = str(r['source_take_id'])
        start, end = int(r['source_frame_start']), int(r['source_frame_end_exclusive'])
        if start < 0 or end - start != int(item.frames) or start >= end:
            raise ValueError('Invalid source frame range: ' + item.clip_id)
        if (r['take_id'] != item.take_id or r['camera_id'] != item.camera_id or
                int(r['frames']) != int(item.frames)):
            raise ValueError('Lineage/clip mismatch: ' + item.clip_id)
        group = source_group_id(item.person_id, item.session_id, item.label, sid)
        ranges[(group, item.camera_id)].append((start, end, item.clip_id))
        rows.append(dict(clip_id=item.clip_id, source_take_id=sid,
                         source_group_id=group, source_clip_id=r['source_clip_id'],
                         source_frame_start=start, source_frame_end_exclusive=end,
                         source_segment=int(r['segment'])))
    for key, pieces in ranges.items():
        for a, b in zip(sorted(pieces), sorted(pieces)[1:]):
            if a[1] > b[0]:
                raise ValueError(f'Overlapping source frames: {key} {a[2]} / {b[2]}')
    return pd.DataFrame(rows), manifest


def frozen_split(catalog, validation_people=('TDUY', 'DPHI'), test_people=('TRUONG',), seed=302):
    """Set split once at NB01; fail on missing label/pose per evaluation person."""
    people = set(catalog.person_id)
    validation, test = set(validation_people), set(test_people)
    if not validation or not test or validation & test or not validation | test <= people:
        raise ValueError('Validation/test people must exist and be distinct')
    train = people - validation - test
    if len(train) < 2:
        raise ValueError('Need at least 2 train people')
    for person in sorted(validation | test):
        x = catalog[catalog.person_id == person]
        if set(map(int, x.label)) != set(LABELS):
            raise ValueError(f'{person} lacks class coverage')
        if (x.groupby('label').pose17_frames.sum().reindex(LABELS, fill_value=0) == 0).any():
            raise ValueError(f'{person} has a class with no YOLO pose')
    out = catalog[['clip_id', 'person_id', 'session_id', 'take_id', 'camera_id', 'label']].copy()
    out['split'] = np.select([out.person_id.isin(test), out.person_id.isin(validation)],
                             ['test', 'val'], default='train')
    info = dict(strategy='subject_wise_frozen_v2', seed=int(seed),
                train_people=sorted(train), validation_people=sorted(validation),
                test_people=sorted(test), selection_metric='mean_subject_macro_f1',
                test_policy='evaluate only once after final retrain')
    return out, info


def usable_mask(x, minimum_joints=4):
    """True only for frames with enough measured joints; no synthetic pose filling."""
    a = np.asarray(x)
    if a.ndim != 3 or a.shape[1:] != (13, 4):
        raise ValueError('Expected [T,13,4] pose features')
    return (a[:, :, 3] > .5).sum(axis=1) >= int(minimum_joints)


def group_sample_epoch(clips, usable, max_per_source_camera=48, seed=302, epoch=1,
                       balance='none'):
    """At most 48 distinct frames per ORIGINAL source take x camera x epoch.
    A clip specifies start/end offsets into the flattened X array and source_group_id.
    Splits/segments share one quota. Optional balance only *downsamples* majorities,
    never oversamples the same few source frames or violates the source quota.
    """

    if balance not in {'none', 'sqrt_inverse', 'inverse'}:
        raise ValueError('Unknown balance mode: ' + str(balance))
    if int(max_per_source_camera) < 1:
        raise ValueError('Sampling quota must be > 0')

    usable = np.asarray(usable, dtype=bool)
    grouped = defaultdict(list)
    labels = {}
    for c in clips:
        key = (c['source_group_id'], c['camera'])
        y = int(c['label'])
        if key in labels and labels[key] != y:
            raise ValueError('Conflicting label for source take/camera')
        labels[key] = y
        ids = np.arange(int(c['start']), int(c['end']), dtype=np.int64)
        grouped[key].extend(ids[usable[ids]].tolist())

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), 101]))
    by_label = defaultdict(list)
    for key in sorted(grouped):
        ids = np.array(sorted(set(grouped[key])), dtype=np.int64)
        n = min(len(ids), int(max_per_source_camera))
        if n == 0:
            continue
        if n < len(ids):
            # Time-stratified random sample, spread across the original recording.
            edges = np.linspace(0, len(ids), n + 1, dtype=np.int64)
            ids = ids[np.array([rng.integers(edges[j], edges[j+1]) for j in range(n)])]
        by_label[labels[key]].extend(ids.tolist())

    if balance != 'none' and by_label:
        counts = {y: len(v) for y, v in by_label.items()}
        reference = min(counts.values())
        exponent = 0.5 if balance == 'sqrt_inverse' else 1.0
        for y, ids in by_label.items():
            target = max(1, min(len(ids), int(round(len(ids)*(reference / len(ids))**exponent))))
            if target < len(ids):
                by_label[y] = rng.choice(ids, size=target, replace=False).tolist()
    result = np.asarray([i for y in sorted(by_label) for i in by_label[y]], dtype=np.int64)
    rng.shuffle(result)

    return result



def augment_pose13_torch(x, cfg, generator):
    """Train-only augmentation preserving confidence/mask semantics.

    Row-wise transforms; max 2 dropped joints and never <4 valid joints.
    Optional one-side forearm occlusion and confidence decay. Not called at eval.
    """
    import torch

    # A. Skip if there is no augmentation
    if x.numel() == 0 or not any(
        float(cfg.get(k, 0)) > 0
        for k in (
            'flip_prob',
            'jitter_prob',
            'keypoint_dropout_prob',
            'occlusion_prob',
            'confidence_perturb_prob',
        )
    ):
        return x

    a = x.reshape(-1, 13, 4).clone()
    n, device = a.shape[0], a.device

    def rand(*shape):
        return torch.rand(
            shape,
            generator=generator,
            device=device,
        )

    # B. Horizontal flip
    flip_p = float(cfg.get('flip_prob', 0))

    if flip_p:
        flip = rand(n) < flip_p

        if flip.any():
            a[flip] = a[flip][:, list(FLIP13), :].clone()
            a[flip, :, 0] *= -1

    # C. Keypoint jitter
    jitter_p = float(cfg.get('jitter_prob', 0))

    if jitter_p:
        noise = torch.randn(
            (n, 13, 2),
            generator=generator,
            device=device,
        ) * float(cfg.get('jitter_sigma', 0.01))

        mask = (
            (rand(n, 1, 1) < jitter_p)
            & (a[:, :, 3:4] > .5)
        )

        a[:, :, :2] += noise * mask

    # D. Confidence perturbation
    conf_p = float(cfg.get('confidence_perturb_prob', 0))

    if conf_p:
        gate = (
            (rand(n, 1) < conf_p)
            & (a[:, :, 3] > .5)
        )

        # Keep the validity mask unchanged.
        a[:, :, 2] = torch.where(
            gate,
            torch.clamp(
                a[:, :, 2] * (0.85 + .15 * rand(n, 13)),
                min=.35,
            ),
            a[:, :, 2],
        )

    # E. Joint dropout and occlusion
    drop_p = float(cfg.get('keypoint_dropout_prob', 0))
    occ_p = float(cfg.get('occlusion_prob', 0))
    max_drop = int(cfg.get('max_dropout_joints', 2))

    if (drop_p or occ_p) and max_drop:
        valid = a[:, :, 3] > .5

        capacity = (
            valid.sum(dim=1) - 4
        ).clamp(min=0, max=max_drop)

        drop = torch.zeros(
            (n, 13),
            dtype=torch.bool,
            device=device,
        )

        # E1. One-side forearm occlusion
        if occ_p:
            active = rand(n) < occ_p
            left = rand(n) < .5

            for joint, side in (
                (7, True),
                (9, True),
                (8, False),
                (10, False),
            ):
                choose = (
                    active
                    & (left == side)
                    & valid[:, joint]
                    & (drop.sum(dim=1) < capacity)
                )

                drop[:, joint] |= choose

        # E2. Random joint dropout
        if drop_p:
            active = (
                (rand(n) < drop_p)
                & (drop.sum(dim=1) < capacity)
            )

            scores = rand(n, 13).masked_fill(
                ~(valid & ~drop),
                float('inf'),
            )

            selected = scores.argmin(dim=1)
            rows = torch.arange(n, device=device)

            drop[rows, selected] |= (
                active
                & torch.isfinite(scores[rows, selected])
            )

        a[drop] = 0

    # F. Validate and restore MLP shape
    assert (
        a[:, :, 3].sum(dim=1)
        >= (
            x.reshape(-1, 13, 4)[:, :, 3] > .5
        ).sum(dim=1).clamp(max=4)
    ).all()

    return a.reshape(-1, 52)



def dataset_output_root(root, source_sha):
    """One immutable dataset-ID namespace for all Phase 2 V2 outputs."""
    return Path(root) / 'outputs' / 'phase2_v2' / str(source_sha)[:12]


def refresh_review_zip(dataset_root, latest_run=None):
    """Single compact shareable review: analyses + metrics/curves, no weights or features.

    Rewrites one review.zip after each notebook; latest training run only.
    """
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / 'latest_run.json'
    if latest_run is not None:
        relative = Path(latest_run).relative_to(root)
        marker.write_text(json.dumps({'run': relative.as_posix()}, indent=2), encoding='utf-8')
    run = None
    if marker.exists():
        relative = json.loads(marker.read_text(encoding='utf-8'))['run']
        candidate = (root / relative).resolve()
        if candidate.is_relative_to(root.resolve()) and candidate.is_dir():
            run = candidate
    dst = root / 'review.zip'
    tmp = dst.with_suffix('.zip.tmp')
    with ZipFile(tmp, 'w', compression=ZIP_DEFLATED, compresslevel=6) as z:
        for folder in (root / 'raw', root / 'processed', run):
            if folder is None or not folder.exists():
                continue
            for f in sorted(folder.rglob('*')):
                if not f.is_file() or f.suffix in {'.pt', '.npz', '.zip'}:
                    continue
                z.write(f, f.relative_to(root).as_posix())
        if marker.exists():
            z.write(marker, marker.relative_to(root).as_posix())
    with ZipFile(tmp) as z:
        assert z.testzip() is None
    os.replace(tmp, dst)
    return dst
