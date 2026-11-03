"""Read-only split planner. Export manifests only, never mutate collected data."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from uuid import uuid4

from .dataset_audit import REQUIRED_COLUMNS, _inside

OUT_FIELDS = ('clip_id', 'take_id', 'person_id', 'session_id', 'label', 'camera_id',
              'camera_view', 'video_path', 'split', 'start_frame', 'end_frame_exclusive',
              'start_ns', 'end_ns')


def _rows(dataset):
    root = Path(dataset).expanduser().resolve(strict=True)
    with (root / 'labels.csv').open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError('Unexpected labels.csv schema; abort split')
        result = list(reader)
    ids = [r['clip_id'] for r in result]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate clip_id; abort split')
    return root, result


def _time_frames(root, row):
    video = _inside(root, row['video_path'])
    path = video.with_name(row['camera_id'] + '_timestamps.csv')
    if not path.is_file(): raise FileNotFoundError(path)
    with path.open(encoding='utf-8-sig',newline='') as f:
        records = list(csv.DictReader(f))
    times = [int(r['host_recv_monotonic_ns']) for r in records]
    indices = [int(r['writer_frame_index']) for r in records]
    if len(times) != int(row['frames']) or indices != list(range(len(times))):
        raise ValueError(f'Timestamp/frame index mismatch: {row["clip_id"]}')
    if len(times) < 3 or times != sorted(times):
        raise ValueError(f'Insufficient or nonmonotonic capture timestamps: {row["clip_id"]}')
    return times


def make_split_plan(dataset, *, mode='subject', assignments=None,
                    train_pct=70, val_pct=15, gap_ms=0):
    """Return frame ranges linked to original clip IDs, no file creation.

    subject mode: all frames from each person assigned to one split.
    within mode: simultaneous cameras from each take get time-aligned segments.
    """
    root, rows = _rows(dataset)
    if mode not in ('subject', 'within'):
        raise ValueError('mode must be subject or within')
    if train_pct <= 0 or val_pct <= 0 or train_pct + val_pct >= 100:
        raise ValueError('Expected positive train/val/test percentages totaling 100')
    if not 0 <= gap_ms <= 2000:
        raise ValueError('Invalid gap milliseconds')
    entries, warnings = [], []
    if mode == 'subject':
        people = {r['person_id'] for r in rows}
        if not assignments or set(assignments) != people:
            raise ValueError('Every person must be assigned explicitly to train/val/test')
        if set(assignments.values()) - {'train', 'val', 'test'}:
            raise ValueError('Unknown split assignment')
        if not {'train','test'} <= set(assignments.values()):
            raise ValueError('Require at least one train and one test person')
        for r in rows:
            entries.append({k:r[k] for k in ('clip_id','take_id','person_id','session_id','label',
                                              'camera_id','camera_view','video_path')} |
                           {'split':assignments[r['person_id']], 'start_frame':0,
                            'end_frame_exclusive':int(r['frames']),'start_ns':'','end_ns':''})
    else:
        takes = defaultdict(list)
        for row in rows: takes[row['take_id']].append(row)
        for take_id, cameras in sorted(takes.items()):
            times_by_camera = {r['camera_id']:_time_frames(root, r) for r in cameras}
            start = max(t[0] for t in times_by_camera.values())
            stop = min(t[-1] for t in times_by_camera.values())
            if stop <= start:
                warnings.append(f'{take_id}: camera timelines do not overlap')
                continue
            train_cut = start + (stop-start)*train_pct/100
            val_cut = start + (stop-start)*(train_pct+val_pct)/100
            segments = [('train',start,train_cut),('val',train_cut,val_cut),('test',val_cut,stop+1)]
            for r in cameras:
                times = times_by_camera[r['camera_id']]
                for split, a, b in segments:
                    padding = int(gap_ms * 500_000)  # Half the gap on either side of a boundary
                    lo = a + (padding if split != 'train' else 0)
                    hi = b - (padding if split != 'test' else 0)
                    indices = [j for j, t in enumerate(times) if lo <= t < hi]
                    if not indices:
                        warnings.append(f'{r["clip_id"]}: empty {split} time segment')
                        continue
                    entry = {k:r[k] for k in ('clip_id','take_id','person_id','session_id','label',
                                               'camera_id','camera_view','video_path')}
                    entry.update(split=split, start_frame=indices[0],
                                 end_frame_exclusive=indices[-1]+1,
                                 start_ns=int(lo), end_ns=int(hi))
                    entries.append(entry)
                    if len(indices) < 16:
                        warnings.append(f'{r["clip_id"]}: {split} has {len(indices)} frames (<16)')
    grouped = Counter(e['split'] for e in entries)
    covered = defaultdict(set)
    for e in entries: covered[(e['person_id'], e['label'])].add(e['split'])
    missing = {f'{person}/L{label}':sorted({'train','val','test'} - values)
               for (person,label),values in covered.items() if mode=='within' and len(values)<3}
    return {'source':str(root), 'mode':mode, 'assignments':assignments if mode=='subject' else None,
            'fractions':({'train':train_pct,'val':val_pct,'test':100-train_pct-val_pct}
                         if mode == 'within' else None),
            'gap_ms':gap_ms, 'counts':dict(grouped), 'person_class_missing':missing,
            'warnings':warnings, 'entries':entries}


def export_plan(plan, output_root):
    """Write one immutable, timestamped split run outside the raw dataset."""
    out = Path(output_root).expanduser().resolve()
    source = Path(plan['source']).resolve()
    if out == source or out.is_relative_to(source):
        raise ValueError('Split outputs must be outside source dataset')
    out.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    with tempfile.TemporaryDirectory(prefix='.split-staging-', dir=out) as temp:
        staging = Path(temp)
        for split in ('train', 'val', 'test'):
            with (staging / f'{split}.csv').open('w', newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
                w.writeheader(); w.writerows(e for e in plan['entries'] if e['split'] == split)
        (staging/'split_config.json').write_text(json.dumps({k:v for k,v in plan.items()
            if k != 'entries'}, ensure_ascii=False, indent=2),encoding='utf-8')
        target = out / f'{plan["mode"]}_{tag}'
        staging.rename(target)
    return target
