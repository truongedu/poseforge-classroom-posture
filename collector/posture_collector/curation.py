"""Non-destructive, track-ID-independent take curation and clean ZIP export.

Raw recordings and dataset_backup are NEVER modified. A single JSON stores edits;
only approved frames/cameras are exported. All time intervals are half-open in
host monotonic nanoseconds to align independently captured cameras.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import json
import os
import re
import shutil
import uuid
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2

from .dataset_audit import REQUIRED_COLUMNS, _inside
from .storage import load_json, save_json, now_iso

VERSION = 1
PREFIX = 'Real_Posture_Dataset/'


def project_artifacts(root):
    """`root` is artifacts/dataset_working/Real_Posture_Dataset."""
    root = Path(root).resolve()
    if root.name != 'Real_Posture_Dataset' or root.parent.name != 'dataset_working':
        raise ValueError('Curation only supports artifacts/dataset_working/Real_Posture_Dataset')
    return root.parent / 'curation.json', root.parent.parent / 'dataset_exports'


def take_key(root, take):
    root = Path(root).resolve()
    take = Path(take).resolve()
    parts = take.relative_to(root / 'recordings').parts
    if (len(parts) != 4 or not re.fullmatch(r'[A-Za-z0-9_-]+', parts[0])
            or not parts[2].startswith('label_') or not re.fullmatch(r'take_\d+', parts[3])
            or not (take / 'metadata.json').is_file()):
        raise ValueError('Not a valid recorded take under working dataset')
    return '/'.join(parts)


def read_edits(root):
    path, _ = project_artifacts(root)
    data = load_json(path, {'version': VERSION, 'takes': {}})
    if not isinstance(data, dict) or data.get('version') != VERSION or not isinstance(data.get('takes'), dict):
        raise ValueError('Unsupported/malformed curation.json; refusing to change it')
    return data


def save_edit(root, take, *, ranges=None, exclude_take=False, exclude_cameras=()):
    """ranges is optional [[start_ns,end_ns],...] ; None = entire take.

    UI stores only UTC-independent host monotonic times. Restoring a take removes
    its entry completely, keeping working recordings byte-identical.
    """
    key = take_key(root, take)
    raw = json.loads((Path(root) / 'recordings' / key / 'metadata.json').read_text())
    cameras = set(raw.get('cameras', {}))
    excluded = set(exclude_cameras)
    if not excluded <= cameras:
        raise ValueError(f'Unknown camera(s): {sorted(excluded - cameras)}')
    if ranges is not None:
        if not isinstance(ranges, list) or not ranges:
            raise ValueError('At least one keep range required; use Restore for entire take')
        cleaned = []
        for pair in ranges:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('Each range must have start/end timestamps')
            a, b = int(pair[0]), int(pair[1])
            if a >= b or a < 0 or (cleaned and a < cleaned[-1][1]):
                raise ValueError('Ranges must be ordered, disjoint, and end-exclusive')
            cleaned.append([a, b])
        ranges = cleaned
    data = read_edits(root)
    if not exclude_take and ranges is None and not excluded:
        data['takes'].pop(key, None)
    else:
        data['takes'][key] = {'exclude_take': bool(exclude_take),
                              'exclude_cameras': sorted(excluded), 'keep_ranges_ns': ranges,
                              'updated_utc': now_iso()}
    path, _ = project_artifacts(root)
    save_json(path, data)
    return key


def capture_bounds(root, take):
    """Full take coverage on the common capture monotonic clock (all cameras)."""
    folder = Path(root) / 'recordings' / take_key(root, take)
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    clocks = [_times(folder / f'{camera}_timestamps.csv')[2]
              for camera in metadata['cameras']]
    if not clocks:
        raise ValueError('Take has no cameras')
    return min(min(ns) for ns in clocks), max(max(ns) for ns in clocks) + 1


def subtract_interval(kept, removed, full):
    """Remove [start, end) from currently kept capture-time intervals."""
    if kept is None:
        kept = [list(full)]
    start, end = map(int, removed)
    if start >= end or start < full[0] or end > full[1]:
        raise ValueError('Invalid remove interval')
    output = []
    for a, b in kept:
        if b <= start or a >= end:
            output.append([a, b])
        else:
            if a < start:
                output.append([a, start])
            if end < b:
                output.append([end, b])
    return output


def timestamp_ns(root, take, camera, frame):
    """Map a replay frame directly to its captured monotonic timestamp."""
    key = take_key(root, take)
    csv_path = Path(root) / 'recordings' / key / f'{camera}_timestamps.csv'
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index == frame:
                if int(row['writer_frame_index']) != index:
                    raise ValueError('Timestamp/frame index mismatch')
                return int(row['host_recv_monotonic_ns'])
    raise ValueError(f'Frame {frame} out of range for {camera}')


def _times(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        if not {'writer_frame_index', 'host_recv_monotonic_ns', 'host_recv_utc'} <= set(cols):
            raise ValueError(f'Invalid timestamps schema: {path}')
        rows = list(reader)
    ns = [int(row['host_recv_monotonic_ns']) for row in rows]
    if not rows or any(int(row['writer_frame_index']) != i for i, row in enumerate(rows)):
        raise ValueError(f'Non-contiguous timestamps: {path}')
    # Capture timestamps describe when frames were received, not the MP4 order.
    # Older recordings can contain repeated/out-of-order ticks. Never sort,
    # interpolate, drop or rewrite rows: frame i must remain aligned with pose i.
    # Export reports these anomalies, and rejects trims that would splice
    # non-adjacent source frames into one artificial segment.
    return cols, rows, ns


def _timestamp_anomalies(ns):
    gaps = [b - a for a, b in zip(ns, ns[1:])]
    backward = [-gap for gap in gaps if gap < 0]
    return {
        'duplicate_adjacent': sum(gap == 0 for gap in gaps),
        'backward_adjacent': len(backward),
        'max_backward_ms': round(max(backward, default=0) / 1_000_000, 3),
    }


def _read_pose(path, expected):
    if not path.is_file():
        raise ValueError(f'Missing raw YOLO17: {path}')
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                raise ValueError(f'Blank JSONL row: {path}')
            item = json.loads(line)
            points = item.get('keypoints')
            if (item.get('backend') != 'yolo' or int(item['frame_index']) != len(rows)
                    or (points is not None and (
                        len(points) != 17 or any(
                            len(point) != 3 or not all(math.isfinite(float(v)) for v in point)
                            or not (0 <= float(point[2]) <= 1)
                            for point in points
                        )))):
                raise ValueError(f'Invalid raw YOLO17/frame index: {path} at {len(rows)}')
            rows.append(item)
    if len(rows) != expected:
        raise ValueError(f'Raw rows {len(rows)} != timestamps {expected}: {path}')
    return rows


def _validate_video(src, expected):
    """Read to EOF: container frame metadata alone can hide truncated MP4s."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError('Cannot decode MP4: ' + str(src))
    count = 0
    try:
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    if count != expected:
        raise ValueError(f'MP4 decoded {count} frames but expected {expected}: {src}')


def _render_video(src, indices, dest, fps, expected):
    """Decode each source frame once, output exactly selected frames, no resizing."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f'Cannot open MP4: {src}')
    writer = None
    selected = set(indices)
    written = 0
    decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if decoded in selected:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    if not writer.isOpened():
                        raise ValueError(f'Cannot create MP4: {dest}')
                writer.write(frame)
                written += 1
            decoded += 1
        if decoded != expected or written != len(indices):
            raise ValueError(f'Decode/write mismatch: {src} decoded={decoded}/{expected} written={written}/{len(indices)}')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if written == 0:
        raise ValueError(f'No selected frames: {src}')
    # Strict check on final encoded MP4; important for frame-to-keypoint mapping.
    check = cv2.VideoCapture(str(dest))
    count = 0
    try:
        if not check.isOpened():
            raise ValueError(f'Encoded MP4 cannot be reopened: {dest}')
        while check.read()[0]:
            count += 1
    finally:
        check.release()
    if count != written:
        raise ValueError(f'Encoded MP4 frame count {count} != {written}: {dest}')


def _json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def _csv_bytes(columns, rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode('utf-8-sig')


def _safe_camera_meta(camera):
    result = dict(camera)
    result.pop('url', None)  # Never upload RTSP/HTTP URLs and credentials.
    return result


def _archive_path(z, name, path):
    if PREFIX + name in z.namelist():
        raise ValueError('Duplicate ZIP path: ' + name)
    z.write(path, PREFIX + name, compress_type=(zipfile.ZIP_STORED if name.endswith('.mp4')
                                                else zipfile.ZIP_DEFLATED))


def export_clean_dataset(root, *, progress=None):
    """One-command export; atomic ZIP, no changes to raw/backup or edit manifest.

    Unmodified MP4 is streamed straight from working dataset into ZIP. Modified
    MP4 is temporarily transcoded only once. No second full dataset copy.
    """
    root = Path(root).resolve()
    _, export_dir = project_artifacts(root)
    decisions = read_edits(root)['takes']
    with (root / 'labels.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if list(reader.fieldnames or []) != REQUIRED_COLUMNS:
            raise ValueError('labels.csv schema mismatch')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty labels.csv')
    grouped = defaultdict(list)
    clip_ids = set()
    for r in rows:
        if r['clip_id'] in clip_ids:
            raise ValueError('Duplicate source clip ID: ' + r['clip_id'])
        clip_ids.add(r['clip_id'])
        meta = _inside(root, r['metadata_path'])
        if not meta.is_file() or not meta.parent.is_relative_to(root / 'recordings'):
            raise ValueError('Invalid metadata location: ' + str(meta))
        grouped[take_key(root, meta.parent)].append(r)
    unknown = set(decisions) - set(grouped)
    if unknown:
        raise ValueError('Stale curation entries (take not in labels.csv): ' + ', '.join(sorted(unknown)[:3]))
    # Catch newly recorded takes that have not yet been appended to labels.csv.
    # Silent omission here is dangerous: V2 notebooks would never see DPHI/new CHANH.
    on_disk = {take_key(root, p.parent) for p in
               (root / 'recordings').glob('*/*/label_*/take_*/metadata.json')}
    orphaned = on_disk - set(grouped)
    if orphaned:
        raise ValueError('Recorded takes absent from labels.csv: '
                         + ', '.join(sorted(orphaned)[:3]) + '. Update source manifest before export.')

    if next(root.rglob('*_raw.pending.mp4'), None) is not None or next(root.rglob('*.writing.txt'), None) is not None:
        raise RuntimeError('Recording or pose extraction still writing; finish before export')
    export_dir.mkdir(parents=True, exist_ok=True)
    # Produce a hidden .partial first, and publish the verified ZIP atomically.
    staging = export_dir / f'.clean_{uuid.uuid4().hex}.partial'
    exported, lineage, used_names = [], [], set()
    clock_warnings = []
    try:
        with tempfile.TemporaryDirectory(prefix='.curation_', dir=export_dir) as tmp, \
                zipfile.ZipFile(staging, 'x', allowZip64=True) as z:
            temp = Path(tmp)
            for n, (key, clips) in enumerate(sorted(grouped.items()), 1):
                change = decisions.get(key, {})
                if change.get('exclude_take'):
                    continue
                folder = root / 'recordings' / key
                metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
                parts = key.split('/')
                if (metadata.get('person_id') != parts[0]
                        or metadata.get('session_id') != parts[1]
                        or str(metadata.get('label')) != parts[2].removeprefix('label_')
                        or any(r['person_id'] != parts[0] or r['session_id'] != parts[1]
                               or r['label'] != parts[2].removeprefix('label_')
                               or r['take_id'] != metadata.get('take_id') for r in clips)):
                    raise ValueError('Inconsistent take/person/label metadata: ' + key)
                ranges = change.get('keep_ranges_ns')
                if ranges is not None:
                    if not ranges or any(int(a) >= int(b) for a, b in ranges):
                        raise ValueError('Invalid range for ' + key)
                chosen = [r for r in clips if r['camera_id'] not in change.get('exclude_cameras', [])]
                if not chosen:
                    continue  # Both cameras excluded: effectively exclude take.
                # For untrimmed takes, retain original key/path and avoid re-encoding.
                for seg, interval in enumerate(ranges if ranges is not None else [None], 1):
                    changed = interval is not None
                    src_take_id = metadata['take_id']
                    new_take_id = f'{src_take_id}_seg{seg:02d}' if changed else src_take_id
                    rel_dir = '/'.join([*key.split('/')[:-1],
                                        key.split('/')[-1] + f'_seg{seg:02d}' if changed else key.split('/')[-1]])
                    if new_take_id in used_names:
                        raise ValueError('Duplicate output take ID: ' + new_take_id)
                    used_names.add(new_take_id)
                    new_meta = dict(metadata)
                    new_meta['take_id'] = new_take_id
                    new_meta['status'] = 'ok'
                    new_meta['cameras'] = {}
                    new_meta['keypoint_extractions'] = {}
                    new_meta['source_take_id'] = src_take_id
                    new_meta['curation_segment_index'] = seg if changed else 0
                    segment_has_camera = False
                    segment_starts = []
                    segment_ends = []
                    for r in chosen:
                        cam = r['camera_id']
                        source_video = _inside(root, r['video_path'])
                        if source_video.parent != folder:
                            raise ValueError('Clip path not in its take: ' + r['clip_id'])
                        timestamps = folder / f'{cam}_timestamps.csv'
                        raw = folder / 'keypoints' / f'{cam}_yolo17_raw.txt'
                        summary = raw.with_suffix('.summary.json')
                        if not source_video.is_file() or not summary.is_file():
                            raise ValueError(f'Incomplete camera {key} {cam}')
                        cols, ts_rows, ns = _times(timestamps)
                        anomalies = _timestamp_anomalies(ns)
                        if anomalies['duplicate_adjacent'] or anomalies['backward_adjacent']:
                            warning = {'source_clip_id': r['clip_id'], **anomalies}
                            # One warning per source clip, not per output segment.
                            if warning not in clock_warnings:
                                clock_warnings.append(warning)
                                if progress:
                                    progress(f'[WARN] Non-monotonic capture clock: {r["clip_id"]} '
                                             f'(duplicates={anomalies["duplicate_adjacent"]}, '
                                             f'backward={anomalies["backward_adjacent"]}, '
                                             f'max_backward_ms={anomalies["max_backward_ms"]}); '
                                             'original frame order retained')
                        pose_rows = _read_pose(raw, len(ts_rows))
                        if int(r['frames']) != len(ns):
                            raise ValueError(f'Frame count mismatch for {r["clip_id"]}')
                        cam_meta = metadata.get('cameras', {}).get(cam)
                        if not cam_meta or int(cam_meta['frames']) != len(ns):
                            raise ValueError(f'Camera metadata mismatch: {r["clip_id"]}')
                        if (r['status'] != 'ok' or cam_meta.get('status') != 'ok') and ranges is None:
                            raise ValueError(f'Camera needs review: {r["clip_id"]}. '
                                             'Mark a valid keep range or exclude this camera/take.')
                        if interval is None:
                            ids = list(range(len(ns)))
                        else:
                            a, b = map(int, interval)
                            ids = [i for i, t in enumerate(ns) if a <= t < b]
                            if not ids:
                                raise ValueError(f'Range has no frames for {cam} in {key}; adjust or exclude camera')
                            if ids[-1] - ids[0] + 1 != len(ids):
                                raise ValueError(f'Trim selects non-contiguous video frames for {key}/{cam} '
                                                 f'due to non-monotonic timestamps; adjust range or exclude camera. '
                                                 'Refusing to join non-adjacent frames.')
                        segment_has_camera = True
                        # Preserve the capture extrema even if the source clock stepped backward.
                        first_index = min(ids, key=ns.__getitem__)
                        last_index = max(ids, key=ns.__getitem__)
                        segment_starts.append((ns[first_index], ts_rows[first_index]['host_recv_utc']))
                        segment_ends.append((ns[last_index], ts_rows[last_index]['host_recv_utc']))
                        video_name = f'{rel_dir}/{cam}_raw.mp4'
                        if changed:
                            prepared = temp / f'{n}_{seg}_{cam}.mp4'
                            fps = float(cam_meta.get('output_fps') or 20)
                            if not 1 <= fps <= 120:
                                raise ValueError('Invalid output fps for ' + r['clip_id'])
                            _render_video(source_video, ids, prepared, fps, len(ns))
                            _archive_path(z, video_name, prepared)
                        else:
                            _validate_video(source_video, len(ns))
                            _archive_path(z, video_name, source_video)
                        selected_times = []
                        selected_pose = []
                        origin_ns = ns[ids[0]]
                        last_ms = -1
                        for pos, index in enumerate(ids):
                            t = dict(ts_rows[index])
                            t['writer_frame_index'] = str(pos)
                            selected_times.append(t)
                            k = dict(pose_rows[index])
                            k['frame_index'] = pos
                            ms = max(last_ms + 1, int((ns[index] - origin_ns) / 1_000_000))
                            k['relative_ms'] = ms
                            last_ms = ms
                            # Retain source track fields only as provenance; never use to select/relabel.
                            selected_pose.append(json.dumps(k, ensure_ascii=False, separators=(',', ':')))
                        z.writestr(PREFIX + f'{rel_dir}/{cam}_timestamps.csv', _csv_bytes(cols, selected_times))
                        z.writestr(PREFIX + f'{rel_dir}/keypoints/{cam}_yolo17_raw.txt',
                                   ('\n'.join(selected_pose) + '\n').encode('utf-8'))
                        summary_data = json.loads(summary.read_text(encoding='utf-8'))
                        summary_data['frames'] = len(ids)
                        summary_data['selected_frames'] = sum(k.get('keypoints') is not None for k in (pose_rows[i] for i in ids))
                        summary_data['missing'] = len(ids) - summary_data['selected_frames']
                        summary_data['model_path'] = str(summary_data.get('model_path', '')).replace('\\', '/').rsplit('/', 1)[-1]
                        summary_data['multiple_people'] = sum(len(pose_rows[i].get('persons') or []) > 1 for i in ids)
                        summary_data['detected_single'] = summary_data['selected_frames']
                        # Remove machine-private absolute paths and stale full-take statistics.
                        summary_data['video'] = video_name
                        summary_data['raw_keypoints_txt'] = f'{rel_dir}/keypoints/{cam}_yolo17_raw.txt'
                        summary_data.pop('selection_reasons', None)
                        summary_data['curation_source_take_id'] = src_take_id
                        z.writestr(PREFIX + f'{rel_dir}/keypoints/{cam}_yolo17_raw.summary.json', _json_bytes(summary_data))
                        cm = _safe_camera_meta(cam_meta)
                        cm['frames'] = len(ids)
                        cm['capture_span_s'] = round((max(ns[i] for i in ids) - min(ns[i] for i in ids)) / 1e9, 6)
                        cm['first_host_received_utc'] = ts_rows[ids[0]]['host_recv_utc']
                        cm['last_host_received_utc'] = ts_rows[ids[-1]]['host_recv_utc']
                        cm['status'] = 'ok'
                        new_meta['cameras'][cam] = cm
                        source_extract = metadata.get('keypoint_extractions', {}).get(f'{cam}:yolo', {})
                        new_meta['keypoint_extractions'][f'{cam}:yolo'] = {
                            **source_extract, 'frames': len(ids), 'status': 'complete',
                            'relative_path': f'keypoints/{cam}_yolo17_raw.txt',
                            'detected_single': summary_data['selected_frames'],
                        }
                        output_id = f'{new_take_id}_{cam}'
                        exported.append({'clip_id': output_id, 'take_id': new_take_id,
                                         'person_id': r['person_id'], 'session_id': r['session_id'],
                                         'camera_id': cam, 'camera_view': r['camera_view'],
                                         'label': r['label'], 'video_path': video_name,
                                         'metadata_path': f'{rel_dir}/metadata.json',
                                         'duration_s': cm['capture_span_s'], 'frames': len(ids), 'status': 'ok'})
                        lineage.append({'clip_id': output_id, 'take_id': new_take_id,
                                        'source_take_id': src_take_id, 'source_clip_id': r['clip_id'],
                                        'camera_id': cam, 'segment': seg if changed else 0,
                                        'source_frame_start': ids[0], 'source_frame_end_exclusive': ids[-1] + 1,
                                        'frames': len(ids), 'capture_start_ns': ns[ids[0]],
                                        'capture_end_ns': ns[ids[-1]], 'reencoded': changed})
                    if segment_has_camera:
                        first_ns, first_utc = min(segment_starts)
                        last_ns, last_utc = max(segment_ends)
                        new_meta['start_host_monotonic_ns'] = first_ns
                        new_meta['end_host_monotonic_ns'] = last_ns
                        new_meta['record_start_utc'] = first_utc
                        new_meta['record_end_utc'] = last_utc
                        new_meta['record_command_duration_s'] = round((last_ns - first_ns) / 1e9, 6)
                        z.writestr(PREFIX + f'{rel_dir}/metadata.json', _json_bytes(new_meta))
                if progress and (n % 10 == 0 or n == len(grouped)):
                    progress(f'Clean export {n}/{len(grouped)} takes')
            if not exported:
                raise ValueError('No clips remain after curation')
            if len(exported) != len({r['clip_id'] for r in exported}):
                raise ValueError('Duplicate exported clip IDs')
            z.writestr(PREFIX + 'labels.csv', _csv_bytes(REQUIRED_COLUMNS, exported))
            labels_file = root / 'config' / 'labels.json'
            if not labels_file.is_file():
                raise ValueError('Missing label schema')
            _archive_path(z, 'config/labels.json', labels_file)
            z.writestr(PREFIX + 'curation_manifest.json', _json_bytes({
                'version': VERSION, 'created_utc': now_iso(),
                'timebase': 'host_recv_monotonic_ns', 'range_rule': '[start,end)',
                'recordings': lineage, 'capture_timestamp_anomalies': clock_warnings,
                'excluded_source_takes': sorted(
                    k for k, v in decisions.items() if v.get('exclude_take')),
                'clip_camera_records': len(exported),
            }))
        with zipfile.ZipFile(staging) as check:
            bad = check.testzip()
            if bad:
                raise ValueError('Corrupted ZIP member: ' + bad)
            names = set(check.namelist())
            for r in exported:
                for col in ('video_path', 'metadata_path'):
                    if PREFIX + r[col] not in names:
                        raise ValueError('Missing output member: ' + r[col])
        version = 1
        while True:
            dest = export_dir / f'clean_raw_v{version:03d}.zip'
            try:
                os.link(staging, dest)  # Atomic, refuses overwrite.
                break
            except FileExistsError:
                version += 1
        return {'zip': str(dest), 'clips': len(exported),
                'takes': len({r['take_id'] for r in exported}),
                'frames': sum(int(r['frames']) for r in exported),
                'bytes': dest.stat().st_size, 'capture_clock_warnings': len(clock_warnings)}
    finally:
        staging.unlink(missing_ok=True)
