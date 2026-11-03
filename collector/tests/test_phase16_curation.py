"""Real MP4 roundtrip with two different capture clocks and raw JSONL."""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from posture_collector.curation import (
    export_clean_dataset, read_edits, save_edit, timestamp_ns, take_key,
    capture_bounds, subtract_interval,
)
from posture_collector.storage import save_json


def dataset(tmp_path):
    root = tmp_path / 'artifacts/dataset_working/Real_Posture_Dataset'
    take = root / 'recordings/DPHI/session_20261029_1142/label_5/take_002'
    (take / 'keypoints').mkdir(parents=True)
    (root / 'config').mkdir(parents=True)
    save_json(root / 'config/labels.json', {'version': 1, 'labels': [{'id': 5, 'name': 'Raised hand', 'key': 'raise_hand'}]})
    source_id = 'DPHI_session_20261029_1142_L5_take_002'
    cameras = {}
    rows = []
    for cam in ('cam01', 'cam02'):
        # Capture timestamps deliberately differ per camera; they are not video FPS timestamps.
        offset = 0 if cam == 'cam01' else 10_000_000
        n = 12
        fps = 20.0
        video = take / f'{cam}_raw.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (64, 48))
        assert writer.isOpened()
        for i in range(n):
            writer.write(np.full((48, 64, 3), (i * 17) % 255, dtype=np.uint8))
        writer.release()
        with (take / f'{cam}_timestamps.csv').open('w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=['writer_frame_index', 'source_frame_index',
                                               'host_recv_monotonic_ns', 'host_recv_utc'])
            wr.writeheader()
            for i in range(n):
                wr.writerow({'writer_frame_index': i, 'source_frame_index': i + 100,
                             'host_recv_monotonic_ns': 1_000_000_000 + offset + i * 50_000_000,
                             'host_recv_utc': f'2026-10-29T05:00:{i:02d}Z'})
        raw = take / 'keypoints' / f'{cam}_yolo17_raw.txt'
        with raw.open('w') as f:
            for i in range(n):
                f.write(json.dumps({'frame_index': i, 'backend': 'yolo', 'status': 'ok',
                                    'keypoints': [[float(i), float(i), 0.9] for _ in range(17)],
                                    'track_id': None, 'persons': []}) + '\n')
        save_json(raw.with_suffix('.summary.json'), {'frames': n, 'selected_frames': n,
                                                        'model_path': '/home/private/yolo11n-pose.pt'})
        cameras[cam] = {'camera_id': cam, 'view': cam, 'url': 'rtsp://secret:password@192.168.1.1',
                        'output_fps': fps, 'frames': n, 'status': 'ok'}
        rows.append({'clip_id': source_id + '_' + cam, 'take_id': source_id, 'person_id': 'DPHI',
                     'session_id': 'session_20261029_1142', 'camera_id': cam, 'camera_view': cam,
                     'label': 5, 'video_path': str(video.relative_to(root)).replace('/', '\\'),
                     'metadata_path': str((take / 'metadata.json').relative_to(root)).replace('/', '\\'),
                     'duration_s': 0.55, 'frames': n, 'status': 'ok'})
    save_json(take / 'metadata.json', {'take_id': source_id, 'person_id': 'DPHI',
                                       'session_id': 'session_20261029_1142', 'label': 5,
                                       'status': 'ok', 'cameras': cameras})
    with (root / 'labels.csv').open('w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return root, take


def read_export(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        root = 'Real_Posture_Dataset/'
        labels = list(csv.DictReader(io.StringIO(z.read(root+'labels.csv').decode('utf-8-sig'))))
        return z, labels


def test_keep_two_disjoint_ranges_two_cameras_and_source_unchanged(tmp_path):
    root, take = dataset(tmp_path)
    originals = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in root.rglob('*') if p.is_file()}
    start = timestamp_ns(root, take, 'cam01', 2)
    end = timestamp_ns(root, take, 'cam01', 5)
    a = timestamp_ns(root, take, 'cam01', 7)
    b = timestamp_ns(root, take, 'cam01', 11)
    save_edit(root, take, ranges=[[start, end], [a, b]])
    result = export_clean_dataset(root)
    assert result['clips'] == 4 and result['takes'] == 2
    with zipfile.ZipFile(result['zip']) as z:
        assert z.testzip() is None
        prefix = 'Real_Posture_Dataset/'
        rows = list(csv.DictReader(io.StringIO(z.read(prefix+'labels.csv').decode('utf-8-sig'))))
        assert {r['take_id'] for r in rows} == {
            'DPHI_session_20261029_1142_L5_take_002_seg01',
            'DPHI_session_20261029_1142_L5_take_002_seg02'}
        assert {r['frames'] for r in rows} == {'3', '4'}
        for row in rows:
            base = str(Path(row['video_path']).parent)
            timestamp_rows = list(csv.DictReader(io.StringIO(
                z.read(prefix+base+'/'+row['camera_id']+'_timestamps.csv').decode('utf-8-sig'))))
            pose_rows = [json.loads(s) for s in z.read(
                prefix+base+'/keypoints/'+row['camera_id']+'_yolo17_raw.txt').decode().splitlines()]
            assert len(timestamp_rows) == len(pose_rows) == int(row['frames'])
            assert [int(p['frame_index']) for p in pose_rows] == list(range(len(pose_rows)))
            assert [int(p['relative_ms']) for p in pose_rows][0] == 0
            assert [int(t['writer_frame_index']) for t in timestamp_rows] == list(range(len(pose_rows)))
            assert 'password' not in z.read(prefix+base+'/metadata.json').decode()
            assert json.loads(z.read(prefix+base+'/metadata.json'))['source_take_id'].endswith('take_002')
            # Decode selected MP4 to verify video/pose alignment.
            video_path = tmp_path / ('check_'+row['clip_id']+'.mp4')
            video_path.write_bytes(z.read(prefix+row['video_path']))
            cap = cv2.VideoCapture(str(video_path)); count = 0
            while cap.read()[0]: count += 1
            cap.release()
            assert count == len(pose_rows)
        provenance = json.loads(z.read(prefix+'curation_manifest.json'))
        assert len(provenance['recordings']) == 4
        assert {r['source_take_id'] for r in provenance['recordings']} == {'DPHI_session_20261029_1142_L5_take_002'}
    assert {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()} == originals
    assert export_clean_dataset(root)['zip'].endswith('clean_raw_v002.zip')


def test_exclude_camera_take_and_restore(tmp_path):
    root, take = dataset(tmp_path)
    save_edit(root, take, exclude_cameras=['cam01'])
    out = export_clean_dataset(root)
    assert out['clips'] == 1 and out['frames'] == 12
    with zipfile.ZipFile(out['zip']) as z:
        assert not any(n.endswith('cam01_raw.mp4') for n in z.namelist())
    save_edit(root, take, exclude_take=True)
    with pytest.raises(ValueError, match='No clips remain'):
        export_clean_dataset(root)
    assert not (root.parents[1] / 'dataset_exports/clean_raw_v002.zip').exists()
    save_edit(root, take)
    assert read_edits(root)['takes'] == {}
    assert export_clean_dataset(root)['clips'] == 2


def test_corruption_aborts_export_no_partial_zip(tmp_path):
    root, take = dataset(tmp_path)
    raw = take / 'keypoints/cam01_yolo17_raw.txt'
    raw.write_text(raw.read_text().replace('"frame_index": 0', '"frame_index": 5', 1))
    with pytest.raises(ValueError, match='Invalid raw'):
        export_clean_dataset(root)
    _, outputs = __import__('posture_collector.curation', fromlist=['project_artifacts']).project_artifacts(root)
    assert not list(outputs.glob('*.zip'))


def test_invalid_ranges_rejected(tmp_path):
    root, take = dataset(tmp_path)
    with pytest.raises(ValueError, match='ordered'):
        save_edit(root, take, ranges=[[5, 8], [6, 9]])
    with pytest.raises(ValueError, match='start/end'):
        save_edit(root, take, ranges=[[1]])
    with pytest.raises(ValueError, match='Unknown camera'):
        save_edit(root, take, exclude_cameras=['cam09'])


def test_trim_excludes_source_bad_frames(tmp_path):
    root, take = dataset(tmp_path)
    save_edit(root, take, ranges=[[
        timestamp_ns(root, take, 'cam01', 3), timestamp_ns(root, take, 'cam01', 7)
    ]])
    out = export_clean_dataset(root)
    with zipfile.ZipFile(out['zip']) as z:
        manifest = json.loads(z.read('Real_Posture_Dataset/curation_manifest.json'))
        assert all(item['source_frame_start'] >= 3 for item in manifest['recordings'])
        assert all(item['source_frame_end_exclusive'] <= 7 for item in manifest['recordings'])


def test_unlisted_new_take_cannot_disappear_silently(tmp_path):
    root, take = dataset(tmp_path)
    orphan = take.parent / 'take_003'
    orphan.mkdir()
    meta = json.loads((take / 'metadata.json').read_text())
    meta['take_id'] = 'DPHI_session_20261029_1142_L5_take_003'
    save_json(orphan / 'metadata.json', meta)
    with pytest.raises(ValueError, match='absent from labels.csv'):
        export_clean_dataset(root)


def test_bad_status_must_be_reviewed_or_excluded(tmp_path):
    root, take = dataset(tmp_path)
    metadata = json.loads((take / 'metadata.json').read_text())
    metadata['cameras']['cam01']['status'] = 'needs_review'
    save_json(take / 'metadata.json', metadata)
    with pytest.raises(ValueError, match='needs review'):
        export_clean_dataset(root)
    save_edit(root, take, exclude_cameras=['cam01'])
    assert export_clean_dataset(root)['clips'] == 1


def test_pending_recorder_does_not_export(tmp_path):
    root, take = dataset(tmp_path)
    (take / 'cam01_raw.pending.mp4').write_bytes(b'dummy')
    with pytest.raises(RuntimeError, match='still writing'):
        export_clean_dataset(root)


def test_remove_middle_range_keeps_two_good_segments(tmp_path):
    root, take = dataset(tmp_path)
    bounds = capture_bounds(root, take)
    a = timestamp_ns(root, take, 'cam01', 4)
    b = timestamp_ns(root, take, 'cam01', 7)
    remaining = subtract_interval(None, (a, b), bounds)
    assert len(remaining) == 2
    save_edit(root, take, ranges=remaining)
    out = export_clean_dataset(root)
    assert out['clips'] == 4 and out['takes'] == 2
    with zipfile.ZipFile(out['zip']) as z:
        manifest = json.loads(z.read('Real_Posture_Dataset/curation_manifest.json'))
        assert all(not (4 <= r['source_frame_start'] < 7) for r in manifest['recordings'])


def test_remove_first_and_last_and_repeated_subtraction(tmp_path):
    root, take = dataset(tmp_path)
    full = capture_bounds(root, take)
    front = timestamp_ns(root, take, 'cam01', 2)
    tail = timestamp_ns(root, take, 'cam01', 9)
    only_middle = subtract_interval(None, (full[0], front), full)
    only_middle = subtract_interval(only_middle, (tail, full[1]), full)
    assert only_middle == [[front, tail]]
    assert subtract_interval(only_middle, (front, tail), full) == []
    assert subtract_interval(only_middle, (front - 1, front), full) == only_middle


def test_timestamp_duplicates_or_backsteps_preserved_not_reordered(tmp_path):
    root, take = dataset(tmp_path)
    from posture_collector.curation import _times, _timestamp_anomalies, capture_bounds
    path = take / 'cam02_timestamps.csv'
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    ns = [int(r['host_recv_monotonic_ns']) for r in rows]
    rows[3]['host_recv_monotonic_ns'] = str(ns[2])       # same receipt tick
    rows[7]['host_recv_monotonic_ns'] = str(ns[6] - 5_000_000)  # 5 ms backward
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cols, read_rows, read_ns = _times(path)
    assert len(read_ns) == 12
    assert _timestamp_anomalies(read_ns) == {
        'duplicate_adjacent': 1, 'backward_adjacent': 1, 'max_backward_ms': 5.0}
    assert capture_bounds(root, take) == (1_000_000_000, max(ns) + 1)
    messages = []
    result = export_clean_dataset(root, progress=messages.append)
    assert result['capture_clock_warnings'] == 1
    assert any('Non-monotonic capture clock' in m for m in messages)
    with zipfile.ZipFile(result['zip']) as z:
        manifest = json.loads(z.read('Real_Posture_Dataset/curation_manifest.json'))
        assert len(manifest['capture_timestamp_anomalies']) == 1
        timestamp_rows = list(csv.DictReader(io.StringIO(z.read(
            next(name for name in z.namelist() if name.endswith('/cam02_timestamps.csv'))
        ).decode('utf-8-sig'))))
        assert [r['host_recv_monotonic_ns'] for r in timestamp_rows] == [
            r['host_recv_monotonic_ns'] for r in rows]


def test_trimming_noncontiguous_clock_selection_must_fail_safely(tmp_path):
    root, take = dataset(tmp_path)
    path = take / 'cam01_timestamps.csv'
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    rows[5]['host_recv_monotonic_ns'] = rows[1]['host_recv_monotonic_ns']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    a = timestamp_ns(root, take, 'cam01', 3)
    b = timestamp_ns(root, take, 'cam01', 8)
    save_edit(root, take, ranges=[[a, b]])
    with pytest.raises(ValueError, match='non-contiguous video frames'):
        export_clean_dataset(root)
    assert not list((root.parents[1] / 'dataset_exports').glob('clean_raw_*.zip'))


def test_trim_stays_valid_when_clock_anomaly_is_outside_kept_interval(tmp_path):
    root, take = dataset(tmp_path)
    path = take / 'cam02_timestamps.csv'
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    rows[3]['host_recv_monotonic_ns'] = rows[1]['host_recv_monotonic_ns']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    a = timestamp_ns(root, take, 'cam01', 8)
    b = timestamp_ns(root, take, 'cam01', 11)
    save_edit(root, take, ranges=[[a, b]])
    result = export_clean_dataset(root)
    assert result['clips'] == 2 and result['frames'] == 6
    with zipfile.ZipFile(result['zip']) as z:
        rows = list(csv.DictReader(io.StringIO(z.read('Real_Posture_Dataset/labels.csv').decode('utf-8-sig'))))
        assert all(int(r['frames']) == 3 for r in rows)
        manifest = json.loads(z.read('Real_Posture_Dataset/curation_manifest.json'))
        assert len(manifest['capture_timestamp_anomalies']) == 1


def test_timestamp_frame_index_still_required_contiguous(tmp_path):
    root, take = dataset(tmp_path)
    path = take / 'cam01_timestamps.csv'
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    rows[2]['writer_frame_index'] = '9'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError, match='Non-contiguous timestamps'):
        export_clean_dataset(root)
