"""Read-only audit of a PoseForge V1 dataset; does not create or mutate data."""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

REQUIRED_COLUMNS = [
    'clip_id', 'take_id', 'person_id', 'session_id', 'camera_id', 'camera_view',
    'label', 'video_path', 'metadata_path', 'duration_s', 'frames', 'status',
]


def _inside(root: Path, candidate: str) -> Path:
    relative = Path(candidate.replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe relative path: ' + candidate)
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes dataset: ' + candidate)
    return path


def audit_dataset(root, *, deep=False, progress=None, person=None):
    """Report coverage and inconsistencies without rewriting labels/metadata.

    Windows separators and `status=needs_review` are supported. Warnings
    indicate quality issues; errors indicate missing/broken references.
    """
    root = Path(root).expanduser().resolve()
    manifest = root / 'labels.csv'
    if not manifest.is_file():
        raise FileNotFoundError(f'labels.csv missing: {manifest}')
    errors, warnings, rows = [], [], []
    counts = Counter()
    with manifest.open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != REQUIRED_COLUMNS:
            raise ValueError(f'Unexpected labels.csv columns: {reader.fieldnames}')
        rows = [r for r in reader if person is None or r['person_id'] == person]
    seen = set()
    extractor_counts = Counter()
    for index, record in enumerate(rows, 1):
        clip_id = record['clip_id']
        if not clip_id or clip_id in seen:
            errors.append(f'Row {index}: empty or duplicate clip_id {clip_id!r}')
        seen.add(clip_id)
        counts[f'label_{record["label"]}'] += 1
        counts['clip_camera_records'] += 1
        if not record['person_id']:
            errors.append(f'{clip_id}: missing person_id')
        try:
            video = _inside(root, record['video_path'])
            meta_file = _inside(root, record['metadata_path'])
            if not video.is_file() or video.stat().st_size == 0:
                errors.append(f'{clip_id}: video missing/empty: {record["video_path"]}')
            if not meta_file.is_file():
                errors.append(f'{clip_id}: metadata missing: {record["metadata_path"]}')
                continue
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
            if str(meta.get('person_id')) != record['person_id']:
                errors.append(f'{clip_id}: metadata person mismatch')
            if str(meta.get('label')) != record['label']:
                errors.append(f'{clip_id}: metadata label mismatch')
            if meta.get('status') != record['status']:
                counts['metadata_manifest_status_differ'] += 1
            kp_dir = video.parent / 'keypoints'
            raw = kp_dir / f'{record["camera_id"]}_yolo17_raw.txt'
            summary = raw.with_suffix('.summary.json')
            if summary.is_file():
                try:
                    details = json.loads(summary.read_text(encoding='utf-8'))
                    extractor_counts[str(details.get('model_path') or 'unknown').replace('\\', '/').rsplit('/', 1)[-1]] += 1
                except (OSError, ValueError):
                    warnings.append(f'{clip_id}: summary unreadable')
            else:
                extractor_counts['unknown'] += 1
            if not raw.is_file():
                counts['missing_yolo17_raw'] += 1
                warnings.append(f'{clip_id}: no YOLO17 raw keypoints')
            elif deep:
                with raw.open(encoding='utf-8') as f:
                    for line_no, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            if item.get('keypoints') is not None and len(item['keypoints']) != 17:
                                errors.append(f'{clip_id}: non-17 keypoints, line {line_no}')
                        except (ValueError, TypeError) as exc:
                            errors.append(f'{clip_id}: invalid keypoints line {line_no}: {exc}')
            if progress and index % 20 == 0:
                progress(f'Audit: {index}/{len(rows)} camera clips')
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f'{clip_id}: {exc}')
    return {
        'root': str(root),
        'clip_camera_records': len(rows),
        'people': sorted(set(r['person_id'] for r in rows)),
        'per_label': {str(i): counts[f'label_{i}'] for i in range(6)},
        'metadata_manifest_status_differ': counts['metadata_manifest_status_differ'],
        'missing_yolo17_raw': counts['missing_yolo17_raw'],
        'pose_model_sources': dict(sorted(extractor_counts.items())),
        'errors': errors, 'warnings': warnings,
    }


def snapshot_files(root, *, progress=None):
    """Full SHA-256 inventory for before/after verification, read-only."""
    root = Path(root).expanduser().resolve()
    result = {}
    files = sorted(p for p in root.rglob('*') if p.is_file())
    for index, path in enumerate(files, 1):
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        result[path.relative_to(root).as_posix()] = {
            'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
        if progress and (index % 25 == 0 or index == len(files)):
            progress(f'Hashing: {index}/{len(files)} files')
    return result


def compare_snapshots(previous, current):
    old = previous['files']; new = current['files']
    return {
        'missing': sorted(set(old) - set(new)),
        'changed': sorted(k for k in old.keys() & new.keys() if old[k] != new[k]),
        'added': sorted(set(new) - set(old)),
    }
