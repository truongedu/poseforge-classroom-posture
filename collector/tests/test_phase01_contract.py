"""Additional regression checks for the existing collector format.

Tests intentionally use synthetic keypoints and do not need any private video.
"""
import csv
import json

import numpy as np

from posture_collector.process import normalize_rows
from posture_collector.storage import DataStore, save_json


def test_legacy_yolo17_to_normalized13_contract(tmp_path):
    raw_file = tmp_path / 'cam01_yolo17_raw.txt'
    joints = [[0, 0, 0.0] for _ in range(17)]
    joints[0] = [50, 20, 0.99]  # nose
    joints[5] = [40, 40, 0.99]  # left shoulder
    joints[6] = [60, 40, 0.99]  # right shoulder
    joints[11] = [40, 80, 0.99] # left hip
    joints[12] = [60, 80, 0.99] # right hip
    # Clip-wide shoulder scale=20 and torso proxy=30 -> median=25.
    rows = [
        dict(frame_index=0, relative_ms=0, width=100, height=100, keypoints=joints),
        dict(frame_index=1, relative_ms=50, width=100, height=100, keypoints=joints),
        dict(frame_index=2, relative_ms=100, width=100, height=100, keypoints=None),
    ]
    raw_file.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    out = normalize_rows(raw_file, 'yolo')
    assert out['keypoints'].shape == (3, 13, 4)
    assert out['keypoints'].dtype == np.float32
    assert np.isclose(out['scales_px'][0], 25.0)
    assert out['feature_version'] == 'pose13_xy_conf_mask_causal_v2'
    np.testing.assert_allclose(out['keypoints'][0, 0], [0, -.8, .99, 1], atol=1e-6)
    np.testing.assert_allclose(out['keypoints'][0, 5], [-.4, 0, .99, 1], atol=1e-6)
    np.testing.assert_array_equal(out['keypoints'][2], np.zeros((13, 4), np.float32))
    np.testing.assert_array_equal(out['relative_ms'], [0, 50, 100])


def test_original_manifest_columns_and_relative_video_path(tmp_path):
    db = DataStore(tmp_path)
    db.add_subject('TQB', True)
    folder = db.create_take('TQB', 0, 'session_test')
    (folder / 'cam01_raw.mp4').touch()
    save_json(folder / 'metadata.json', {
        'take_id': 'TQB_session_test_L0_take_001',
        'person_id': 'TQB', 'session_id': 'session_test',
        'label': 0, 'cameras': {'cam01': {'view': 'front', 'status': 'ok',
                                        'capture_span_s': 5.0, 'frames': 100}},
    })
    path, n = db.export_manifest()
    assert n == 1
    with path.open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    assert list(records[0]) == [
        'clip_id', 'take_id', 'person_id', 'session_id', 'camera_id', 'camera_view',
        'label', 'video_path', 'metadata_path', 'duration_s', 'frames', 'status',
    ]
    assert records[0]['video_path'] == 'recordings/TQB/session_test/label_0/take_001/cam01_raw.mp4'
