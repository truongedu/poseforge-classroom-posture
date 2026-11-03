#!/usr/bin/env python3
"""Read-only PoseForge V1 dataset integrity and provenance preflight.

Checks the original raw schema. Does not normalize, rewrite, or repair data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

COLUMNS = (
    'clip_id', 'take_id', 'person_id', 'session_id', 'camera_id', 'camera_view',
    'label', 'video_path', 'metadata_path', 'duration_s', 'frames', 'status',
)


def inside(root: Path, path: str) -> Path:
    rel = Path(str(path).replace('\\', '/'))
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'unsafe relative path: {path!r}')
    dest = (root / rel).resolve()
    if not dest.is_relative_to(root):
        raise ValueError(f'path escapes dataset: {path!r}')
    return dest


def inspect_keypoints(path: Path, expected: int, issues: list[str], clip_id: str):
    rows = 0
    valid = 0
    status = Counter()
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if int(item['frame_index']) != rows:
                    issues.append(f'{clip_id}: raw frame index out of order at line {rows + 1}')
                kp = item.get('keypoints')
                if kp is not None:
                    if len(kp) != 17 or any(len(p) != 3 for p in kp):
                        issues.append(f'{clip_id}: invalid raw keypoint shape at frame {rows}')
                    else:
                        valid += 1
                status[str(item.get('status', 'unknown'))] += 1
            except (ValueError, TypeError, KeyError) as exc:
                issues.append(f'{clip_id}: invalid raw JSONL row {rows}: {exc}')
            rows += 1
    if rows != expected:
        issues.append(f'{clip_id}: raw frame count {rows} != manifest {expected}')
    return rows, valid, status


def run(root: Path, decode: bool = False):
    root = root.expanduser().resolve()
    manifest = root / 'labels.csv'
    if not manifest.is_file():
        raise FileNotFoundError(f'No labels.csv in {root}')
    with manifest.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f'Unexpected labels.csv schema: {reader.fieldnames}')
        records = list(reader)
    cv2 = None
    if decode:
        import cv2 as cv2_module
        cv2 = cv2_module

    errors = []
    warnings = []
    samples = Counter()
    status_by_scope = Counter()
    models = Counter()
    missing = []
    seen_ids = set()
    unique_takes = set()
    timescale_alerts = []
    valid_frames = 0
    inspected_frames = 0
    raw_status = Counter()
    for idx, row in enumerate(records, 1):
        clip = row['clip_id']
        try:
            if not clip or clip in seen_ids:
                raise ValueError('empty or duplicate clip_id')
            seen_ids.add(clip)
            samples[(row['person_id'], row['label'])] += 1
            unique_takes.add(row['take_id'])
            if row['label'] not in {str(i) for i in range(6)}:
                errors.append(f'{clip}: label outside 0..5')
            video = inside(root, row['video_path'])
            metadata_file = inside(root, row['metadata_path'])
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError(f'video missing or empty: {video}')
            if not metadata_file.is_file():
                raise ValueError(f'metadata missing: {metadata_file}')
            meta = json.loads(metadata_file.read_text(encoding='utf-8'))
            cam = meta.get('cameras', {}).get(row['camera_id'])
            if not isinstance(cam, dict):
                raise ValueError('metadata missing camera entry')
            if (str(meta.get('person_id')) != row['person_id'] or
                str(meta.get('label')) != row['label'] or
                str(meta.get('take_id')) != row['take_id']):
                errors.append(f'{clip}: person/label/take mismatch between CSV and metadata')
            if str(cam.get('status')) == row['status']:
                status_by_scope['camera_status_matched'] += 1
            else:
                status_by_scope['camera_status_mismatch'] += 1
                warnings.append(f'{clip}: manifest camera status differs from metadata camera status')
            if str(meta.get('status')) != row['status']:
                status_by_scope['take_status_differs'] += 1
            expected = int(row['frames'])
            if expected <= 0:
                errors.append(f'{clip}: nonpositive frame count')
            timestamp = video.parent / f"{row['camera_id']}_timestamps.csv"
            if not timestamp.is_file():
                errors.append(f'{clip}: timestamps missing')
            else:
                with timestamp.open(encoding='utf-8-sig', newline='') as stream:
                    times = list(csv.DictReader(stream))
                if len(times) != expected:
                    errors.append(f'{clip}: {len(times)} timestamp rows != {expected} manifest frames')
                if times:
                    ids = [int(t['writer_frame_index']) for t in times]
                    if ids != list(range(len(times))):
                        errors.append(f'{clip}: timestamp writer indices not continuous')
                    if len(times) > 1:
                        span = (int(times[-1]['host_recv_monotonic_ns']) - int(times[0]['host_recv_monotonic_ns'])) / 1e9
                        declared_fps = float(cam.get('output_fps') or 0)
                        if span > 0 and declared_fps > 0:
                            ratio = (expected / declared_fps) / span
                            if not 0.85 <= ratio <= 1.15:
                                timescale_alerts.append({'clip': clip, 'encoded_duration_to_capture_span_ratio': round(ratio, 3)})
            kp = video.parent / 'keypoints' / f"{row['camera_id']}_yolo17_raw.txt"
            summary = kp.with_suffix('.summary.json')
            if not kp.is_file():
                missing.append(clip)
            else:
                frames, valid, state = inspect_keypoints(kp, expected, errors, clip)
                inspected_frames += frames
                valid_frames += valid
                raw_status.update(state)
            if summary.is_file():
                details = json.loads(summary.read_text(encoding='utf-8'))
                model = str(details.get('model_path') or 'unknown').replace('\\', '/').rsplit('/', 1)[-1]
                models[model] += 1
                if kp.is_file() and int(details.get('frames', -1)) != expected:
                    errors.append(f'{clip}: extraction summary frames mismatch')
            elif kp.is_file():
                warnings.append(f'{clip}: raw exists but summary missing')
            if decode:
                cap = cv2.VideoCapture(str(video))
                try:
                    if not cap.isOpened():
                        errors.append(f'{clip}: MP4 cannot be opened')
                    else:
                        decoded = 0
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            decoded += 1
                        if decoded != expected:
                            errors.append(f'{clip}: {decoded} decoded frames != {expected} manifest frames')
                finally:
                    cap.release()
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(f'{clip or "row " + str(idx)}: {exc}')
        if idx % 20 == 0 or idx == len(records):
            print(f'Checked {idx}/{len(records)} clip-camera records', file=sys.stderr)
    return {
        'dataset': str(root),
        'clip_camera_records': len(records),
        'take_count': len(unique_takes),
        'people': len(set(p for p, _ in samples)),
        'person_class_combinations': len(samples),
        'raw_keypoints_present': len(records) - len(missing),
        'raw_keypoints_missing': missing,
        'pose_models_from_raw_summaries': dict(sorted(models.items())),
        'valid_raw_frames': valid_frames,
        'inspected_raw_frames': inspected_frames,
        'raw_frame_status': dict(raw_status),
        'manifest_status_vs_metadata': dict(status_by_scope),
        'video_capture_timing_mismatches': len(timescale_alerts),
        'timing_examples': timescale_alerts[:8],
        'errors': errors,
        'warnings': warnings,
        'decoded_videos_checked': decode,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--decode-videos', action='store_true', help='Decode every MP4 and verify number of frames (slower)')
    parser.add_argument('--json', type=Path, help='Optional report file (recommend artifacts/audits/...)')
    args = parser.parse_args()
    result = run(args.data, decode=args.decode_videos)
    print(f'Clips: {result["clip_camera_records"]}, takes: {result["take_count"]}, people: {result["people"]}')
    print(f'Raw YOLO17: {result["raw_keypoints_present"]} present, {len(result["raw_keypoints_missing"])} missing')
    print('Model provenance:', result['pose_models_from_raw_summaries'])
    print('Status scopes:', result['manifest_status_vs_metadata'])
    print('Valid keypoint frames:', f'{result["valid_raw_frames"]}/{result["inspected_raw_frames"]}')
    print('Video timebase discrepancies:', result['video_capture_timing_mismatches'])
    print('ERRORS:', len(result['errors']), 'WARNINGS:', len(result['warnings']))
    for item in result['errors'][:15]:
        print('[ERROR]', item)
    for item in result['warnings'][:8]:
        print('[WARN]', item)
    if args.json:
        out = args.json.expanduser().resolve()
        if out.exists():
            parser.error(f'Report already exists (refusing overwrite): {out}')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('Report:', out)
    return 2 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
