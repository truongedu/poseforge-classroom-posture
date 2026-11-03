"""Selective, append-safe re-extraction of YOLO17 from existing raw MP4 clips.

No changes to video, timestamps, clip IDs or labels.csv. Every overwrite
requires an explicit opt-in, and previous files are backed up outside the
source dataset before atomic replacement. This module has no Qt dependency.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

from .process import extract_video, read_timestamps


def _digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest()


def list_targets(dataset, *, person=None, session=None, label=None, take=None,
                 camera=None, mode='missing'):
    """Return deterministic camera targets without initializing or writing DataStore.

    mode='missing': only camera videos without raw keypoints.
    mode='selected': all or filtered camera videos belonging to one Person.
    mode='all': every recorded camera, explicit UI confirmation required.
    mode='zero': raw files with zero selected poses, explicit overwrite required.
    """
    root = Path(dataset).expanduser().resolve(strict=True)
    if mode not in ('missing', 'selected', 'all', 'zero'):
        raise ValueError('mode must be missing, selected, zero or all')
    if mode in ('all','zero') and any(x is not None for x in (person, session, label, take, camera)):
        raise ValueError('Full-dataset mode does not accept selection filters')
    if mode == 'selected' and not person:
        raise ValueError('Selected reprocess requires Person ID')
    targets = []
    for meta_file in sorted((root / 'recordings').glob('*/*/label_*/take_*/metadata.json')):
        folder = meta_file.parent
        if not folder.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe take path: {folder}')
        parts = folder.relative_to(root / 'recordings').parts
        if len(parts) != 4:
            continue
        pid, sid, label_dir, tid = parts
        if person is not None and pid != person: continue
        if session is not None and sid != session: continue
        if label is not None and label_dir != f'label_{int(label)}': continue
        if take is not None and tid != take: continue
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
        for cid in sorted(meta.get('cameras', {})):
            if camera is not None and cid != camera:
                continue
            video = folder / f'{cid}_raw.mp4'
            raw = folder / 'keypoints' / f'{cid}_yolo17_raw.txt'
            summary = raw.with_suffix('.summary.json')
            if not video.is_file():
                continue
            if mode == 'missing' and raw.is_file():
                continue
            if mode == 'zero' and raw.is_file():
                try:
                    with raw.open(encoding='utf-8') as f:
                        if any((r := json.loads(line)).get('status') == 'ok' and r.get('keypoints') is not None
                               for line in f if line.strip()):
                            continue
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # corrupted file also needs re-extraction
            targets.append({'person': pid, 'session': sid, 'label': int(label_dir[6:]),
                            'take': tid, 'camera': cid, 'video': video,
                            'raw': raw, 'summary': summary, 'metadata': meta_file})
    return targets


def _replace_from_bytes(destination, data):
    """Write a complete file next to its destination, then atomically rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.poseforge-', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, destination)
    finally:
        if os.path.exists(name): os.unlink(name)


def _validate_staged(raw, summary, expected_frames):
    if not raw.is_file() or not summary.is_file():
        raise RuntimeError('Extractor did not produce both raw TXT and summary JSON')
    details = json.loads(summary.read_text(encoding='utf-8'))
    if details.get('backend') != 'yolo' or details.get('frames') != expected_frames:
        raise ValueError('Unexpected extractor metadata or incomplete frame count')
    lines = 0
    with raw.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row.get('frame_index') != lines or row.get('backend') != 'yolo':
                raise ValueError('Non-contiguous frame indices or wrong backend')
            kp = row.get('keypoints')
            if kp is not None and (len(kp) != 17 or any(len(p) != 3 for p in kp)):
                raise ValueError(f'Invalid YOLO17 keypoints at frame {lines}')
            lines += 1
    if lines != expected_frames or lines == 0:
        raise ValueError(f'Raw TXT frame count mismatch: {lines} != {expected_frames}')
    return lines


def reextract_targets(dataset, targets, model, *, backup_root, overwrite=False,
                      progress=None, device='auto'):
    """Process a preselected batch; rollback changed files on per-camera failure.

    Re-extraction is independent of train preprocessing. The original data
    contract is maintained: original JSONL and summary schema are generated
    by process.extract_video, not duplicated here.
    """
    root = Path(dataset).expanduser().resolve(strict=True)
    weights = Path(model).expanduser().resolve(strict=True)
    backup_base = Path(backup_root).expanduser().resolve()
    if backup_base.is_relative_to(root):
        raise ValueError('Backup must be outside the source dataset')
    if not weights.is_file() or weights.suffix.lower() not in ('.pt', '.onnx'):
        raise ValueError('Select a local YOLO-Pose .pt or .onnx checkpoint')
    if not targets:
        return {'selected': 0, 'succeeded': 0, 'failed': 0, 'skipped': 0, 'results': []}
    if not overwrite and any(t['raw'].exists() for t in targets):
        raise FileExistsError('Raw file already exists. Explicit overwrite confirmation required')
    if not overwrite:
        # Missing-only must not mix two extractor models silently in one dataset.
        provenance = set()
        for summary in root.glob('recordings/*/*/label_*/take_*/keypoints/*_yolo17_raw.summary.json'):
            try:
                model_name = str(json.loads(summary.read_text(encoding='utf-8')).get('model_path', '')).replace('\\','/').split('/')[-1]
            except (OSError, ValueError):
                model_name = 'unknown'
            provenance.add(model_name or 'unknown')
        if provenance and provenance != {weights.name}:
            raise ValueError('Missing-only would mix pose checkpoints: existing=' +
                             ', '.join(sorted(provenance)) + ', chosen=' + weights.name)
    batch_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    backup = backup_base / batch_id
    backup.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        for i, t in enumerate(targets, 1):
            prefix = f"{i}/{len(targets)} {t['person']}/{t['session']}/label_{t['label']}/{t['take']}/{t['camera']}"
            if progress: progress('Extract ' + prefix)
            originals = (t['raw'], t['summary'], t['metadata'])
            if not overwrite and t['raw'].exists():
                results.append({'target': prefix, 'status': 'skipped'})
                continue
            try:
                # Validation before touching existing files, staging on same local disk.
                with tempfile.TemporaryDirectory(prefix='staging-', dir=backup) as staging:
                    report = extract_video(t['video'], 'yolo', str(weights), progress,
                                           device=device, output_dir=staging)
                    staged_raw = Path(report['raw_keypoints_txt'])
                    staged_summary = staged_raw.with_suffix('.summary.json')
                    _validate_staged(staged_raw, staged_summary, report['frames'])
                    if report['frames'] <= 0:
                        raise ValueError('Empty decoded video')
                    # Never silently replace a formerly usable clip with fewer selected poses.
                    if overwrite and t['raw'].is_file():
                        def count_selected(path):
                            with path.open(encoding='utf-8') as f:
                                return sum(1 for ln in f if ln.strip() and
                                    (row := json.loads(ln)).get('status') == 'ok' and row.get('keypoints') is not None)
                        try:
                            previous = count_selected(t['raw'])
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            previous = 0  # malformed legacy file; replacement is validated above
                        current = count_selected(staged_raw)
                        if current < previous:
                            raise ValueError(f'Quality regression {current} < {previous} selected frames; original preserved')
                    timestamps = read_timestamps(t['video'].with_name(t['camera'] + '_timestamps.csv'))
                    if timestamps and len(timestamps) != report['frames']:
                        raise ValueError('Video decode frame count differs from timestamps; refusing write')
                    # Capture original bytes before any mutation (including metadata).
                    original_bytes = {p: p.read_bytes() if p.is_file() else None for p in originals}
                    saved = backup / t['video'].parent.relative_to(root)
                    saved.mkdir(parents=True, exist_ok=True)
                    backup_meta = []
                    for p, data in original_bytes.items():
                        if data is not None:
                            dst = saved / p.name
                            # Metadata is shared by all cameras of one take.
                            # Preserve the FIRST (pre-batch) copy rather than
                            # overwriting it after cam01 changes metadata.
                            if not dst.exists():
                                _replace_from_bytes(dst, data)
                                if _digest(dst) != hashlib.sha256(data).hexdigest():
                                    raise IOError(f'Backup verification failed: {dst}')
                            backup_meta.append({'original':str(p.relative_to(root)),
                                                'backup':str(dst.relative_to(backup)),
                                                'sha256':_digest(dst)})
                    meta = json.loads(t['metadata'].read_text(encoding='utf-8'))
                    old_entries = meta.setdefault('keypoint_extractions', {})
                    old_entries[f"{t['camera']}:yolo"] = {
                        'status': 'complete', 'frames': report['frames'],
                        'relative_path': f"keypoints/{t['raw'].name}",
                        'detected_single': report['selected_frames'],
                        'ambiguous_multi_person': report['multiple_people'],
                        'extracted_at': report['extracted_at']}
                    summary_obj = json.loads(staged_summary.read_text(encoding='utf-8'))
                    summary_obj['raw_keypoints_txt'] = str(t['raw'])
                    proposed = {t['raw']: staged_raw.read_bytes(),
                                t['summary']: (json.dumps(summary_obj, indent=2, ensure_ascii=False)+'\n').encode('utf-8'),
                                t['metadata']: (json.dumps(meta, indent=2, ensure_ascii=False)+'\n').encode('utf-8')}
                    try:
                        for path, data in proposed.items(): _replace_from_bytes(path, data)
                        if _digest(t['raw']) != hashlib.sha256(proposed[t['raw']]).hexdigest():
                            raise IOError('Post-write hash verification failed')
                    except Exception:
                        for path, data in original_bytes.items():
                            if data is None: path.unlink(missing_ok=True)
                            else: _replace_from_bytes(path, data)
                        raise
                    entry = {'target':prefix,'status':'ok','frames':report['frames'],
                             'backup_files':backup_meta, 'model':weights.name}
                    results.append(entry)
            except Exception as exc:
                results.append({'target':prefix,'status':'error','message':str(exc)})
                if progress: progress(f'ERROR {prefix}: {exc}')
            if progress: progress(f'Progress: {i}/{len(targets)} camera clips')
    finally:
        # The directory is a durable rollback location, not scratch data.
        manifest = {'dataset':str(root), 'checkpoint':str(weights), 'checkpoint_sha256':_digest(weights),
                    'created_utc':batch_id, 'overwrite':bool(overwrite), 'results':results}
        _replace_from_bytes(backup/'manifest.json', json.dumps(manifest,indent=2,ensure_ascii=False).encode('utf-8'))
    return {'selected':len(targets), 'succeeded':sum(x['status']=='ok' for x in results),
            'failed':sum(x['status']=='error' for x in results),
            'skipped':sum(x['status']=='skipped' for x in results),
            'backup':str(backup), 'results':results}
