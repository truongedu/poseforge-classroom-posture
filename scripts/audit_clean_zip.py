#!/usr/bin/env python3
"""Read-only Phase 1.6A quality audit directly from clean_raw_vNNN.zip.

Run from repo root: python -B scripts/audit_clean_zip.py [ZIP]
Outputs compact reports under artifacts/audits/; never changes source ZIP/data.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.dataset_pipeline import scan_raw_zip  # noqa: E402

PREFIX = 'Real_Posture_Dataset/'


def get_zip(user_path):
    if user_path:
        return Path(user_path).expanduser().resolve(strict=True)
    choices = list((ROOT / 'artifacts/dataset_exports').glob('clean_raw_v*.zip'))
    if not choices:
        raise FileNotFoundError('No clean_raw_v*.zip under artifacts/dataset_exports')
    return max(choices, key=lambda p: int(re.search(r'_v(\d+)\.zip$', p.name).group(1)))


def new_report_dir(zip_file):
    base = ROOT / 'artifacts/audits' / zip_file.stem
    for i in range(1, 10000):
        path = base.parent / f'{base.name}_audit_v{i:03d}'
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            continue
    raise RuntimeError('Exhausted audit version numbers')


def write_csv(path, data):
    if not data:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def audit(zip_file, out):
    catalog, frames, _joints, _examples, diagnostics = scan_raw_zip(zip_file)
    issues = list(diagnostics['issues'])
    if catalog.empty:
        raise ValueError('Clean dataset catalog is empty')
    with ZipFile(zip_file) as z:
        manifest = json.loads(z.read(PREFIX + 'curation_manifest.json'))
        rows = list(csv.DictReader(io.StringIO(z.read(PREFIX + 'labels.csv').decode('utf-8-sig'))))
        if len(rows) != len(catalog) or len(rows) != len(manifest.get('recordings', [])):
            issues.append('labels/catalog/lineage count mismatch')
        clip_ids = set(catalog.clip_id)
        segment_map = defaultdict(list)
        by_clip = {r['clip_id']: r for r in rows}
        for entry in manifest['recordings']:
            clip = entry['clip_id']
            if clip not in clip_ids:
                issues.append(f'{clip}: lineage not in catalog')
            if int(entry['source_frame_end_exclusive']) - int(entry['source_frame_start']) != int(entry['frames']):
                issues.append(f'{clip}: source-frame interval != exported frames')
            if clip in by_clip and int(by_clip[clip]['frames']) != int(entry['frames']):
                issues.append(f'{clip}: lineage and labels frames disagree')
            segment_map[entry['source_clip_id']].append((int(entry['source_frame_start']), int(entry['source_frame_end_exclusive']), clip))
        for source, segments in segment_map.items():
            last_end = -1
            for a,b,clip in sorted(segments):
                if a < last_end:
                    issues.append(f'{source}: overlapping exported segments near {clip}')
                last_end = b
        clock_rows = []
        for row in rows:
            cam = row['camera_id']
            member = PREFIX + row['video_path'].replace('\\','/').rsplit('/',1)[0] + f'/{cam}_timestamps.csv'
            ticks = list(csv.DictReader(io.StringIO(z.read(member).decode('utf-8-sig'))))
            diffs = [(int(b['host_recv_monotonic_ns']) - int(a['host_recv_monotonic_ns']),
                      a['source_frame_index'], b['source_frame_index']) for a,b in zip(ticks,ticks[1:])]
            duplicate = [p for p in diffs if p[0] == 0]
            backward = [p for p in diffs if p[0] < 0]
            clock_rows.append({
                'clip_id': row['clip_id'], 'person': row['person_id'], 'label': int(row['label']),
                'camera': cam, 'frames': len(ticks), 'duplicate_adjacent': len(duplicate),
                'duplicate_same_source_frame': sum(a == b for _,a,b in duplicate),
                'duplicate_different_source_frame': sum(a != b for _,a,b in duplicate),
                'backward_adjacent': len(backward),
                'max_backward_ms': round(-min((p[0] for p in backward), default=0)/1e6,3),
                'duplicates_pct': round(100*len(duplicate)/max(1,len(ticks)-1),2),
            })

    ratios = frames.groupby('clip_id', sort=False).agg(
        usable4=('valid13_count', lambda s: int((s >= 4).sum())),
        usable8=('valid13_count', lambda s: int((s >= 8).sum())),
        valid13_mean=('valid13_count', 'mean'),
        shoulder_frames=('shoulders_valid', 'sum'),
    )
    detailed = catalog.merge(ratios, left_on='clip_id', right_index=True, how='left')
    flags = []
    fatal = []
    for row in detailed.itertuples(index=False):
        n = int(row.frames)
        checks = []
        if row.pose17_frames == 0:
            checks.append('ZERO_POSE')
            fatal.append(row.clip_id)
        if int(row.usable4) == 0:
            checks.append('ZERO_USABLE_4_JOINTS')
            if row.clip_id not in fatal: fatal.append(row.clip_id)
        if row.pose17_rate < .80:
            checks.append('LOW_POSE_COVERAGE_LT80')
        if int(row.usable4) / max(1,n) < .50:
            checks.append('LOW_4JOINT_COVERAGE_LT50')
        if int(row.shoulder_frames)/max(1,n) < .15:
            checks.append('LOW_BOTH_SHOULDERS_LT15')
        if n < 12:
            checks.append('SHORT_LT12_FRAMES')
        if checks:
            flags.append({'clip_id':row.clip_id,'person':row.person_id,'label':int(row.label),
                          'camera':row.camera_id,'frames':n,'pose_frames':int(row.pose17_frames),
                          'full13_frames':int(row.full13_frames),'usable4_frames':int(row.usable4),
                          'mean_valid13':round(float(row.valid13_mean),2), 'flags':';'.join(checks)})
    detailed['usable4_rate'] = detailed.usable4/detailed.frames
    detailed['usable8_rate'] = detailed.usable8/detailed.frames
    by_person_label = detailed.groupby(['person_id','label']).agg(
        clips=('clip_id','count'),frames=('frames','sum'),pose_frames=('pose17_frames','sum'),
        usable4=('usable4','sum'),usable8=('usable8','sum'),full13=('full13_frames','sum'),
    ).reset_index()
    by_person_label['pose_pct'] = (100*by_person_label.pose_frames/by_person_label.frames).round(2)
    by_person_label['usable4_pct'] = (100*by_person_label.usable4/by_person_label.frames).round(2)
    people = sorted(detailed.person_id.unique())
    missing_person_labels = {p: sorted(set(range(6))-set(detailed.loc[detailed.person_id==p, 'label']))
                             for p in people}
    missing_person_labels = {p:labels for p,labels in missing_person_labels.items() if labels}
    for p in ['TRUONG','TDUY','DPHI']:
        if p not in people:
            issues.append('Required TEST/VAL person absent: ' + p)
    write_csv(out/'clips.csv', detailed.round(4).to_dict('records'))
    write_csv(out/'review_clips.csv', flags)
    write_csv(out/'person_label.csv', by_person_label.to_dict('records'))
    write_csv(out/'timestamps.csv', clock_rows)
    report = {
        'dataset_zip':zip_file.name,
        'clips':len(detailed), 'takes':int(detailed.take_id.nunique()),
        'people':people, 'frames':int(detailed.frames.sum()),
        'pose_frames':int(detailed.pose17_frames.sum()),
        'full13_frames':int(detailed.full13_frames.sum()),
        'usable_4joint_frames':int(detailed.usable4.sum()),
        'zero_pose_or_zero_4joint_clips':fatal,
        'review_flagged_clips':len(flags),
        'missing_person_labels':missing_person_labels,
        'timestamp_duplicates':sum(r['duplicate_adjacent'] for r in clock_rows),
        'timestamp_backward':sum(r['backward_adjacent'] for r in clock_rows),
        'clock_high_duplicates':sorted((r for r in clock_rows if r['duplicates_pct'] >= 5),
                                       key=lambda r: -r['duplicates_pct'])[:20],
        'issues':issues,
        'guidance':'Review flags are NOT automatic exclusion rules. Posture labels require visual inspection.',
    }
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with ZipFile(out/'audit_review.zip','w',ZIP_DEFLATED) as z:
        for name in ('summary.json','clips.csv','review_clips.csv','person_label.csv','timestamps.csv'):
            file = out/name
            if file.exists(): z.write(file,name)
    print('\n=== CLEAN DATASET QUALITY ===')
    print(f'ZIP: {zip_file.name} | people={len(people)} takes={report["takes"]} clips={len(detailed)} frames={report["frames"]}')
    print(f'Selected poses: {report["pose_frames"]}/{report["frames"]} ({100*report["pose_frames"]/report["frames"]:.2f}%)')
    print(f'Frames with >=4 valid keypoints: {report["usable_4joint_frames"]}/{report["frames"]}')
    print(f'Fatal feature-absent clips: {len(fatal)} | Review-flagged clips: {len(flags)} | Structure issues: {len(issues)}')
    print(f'Timestamps duplicate={report["timestamp_duplicates"]}, backward={report["timestamp_backward"]}')
    print('Per-person missing labels:',missing_person_labels or 'none')
    print('Top review clips:', [(r['clip_id'],r['flags']) for r in flags[:8]])
    if issues: print('First structural issues:', issues[:6])
    print('Report:',out/'audit_review.zip')
    return 2 if issues or fatal else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zip', nargs='?',help='Clean ZIP, default latest artifacts/dataset_exports/clean_raw_vNNN.zip')
    args = parser.parse_args()
    zip_file = get_zip(args.zip)
    out = new_report_dir(zip_file)
    return audit(zip_file, out)


if __name__ == '__main__':
    raise SystemExit(main())
