"""Causal feature parity and missing-point handling, with synthetic inputs."""
import json
from pathlib import Path
import numpy as np
import pytest
from posture_collector.features import CausalPoseNormalizer, FEATURE_VERSION
from posture_collector.process import normalize_rows, build_processed


def points(width=100, confidence=.99, displacement=0):
    x = np.zeros((17, 3), np.float32)
    x[0] = (50 + displacement, 20, confidence)
    x[5] = (40 + displacement, 40, confidence)
    x[6] = (60 + displacement, 40, confidence)
    x[11] = (40 + displacement, 80, confidence)
    x[12] = (60 + displacement, 80, confidence)
    return x.tolist()


def raw_file(path, frames):
    path.write_text('\n'.join(json.dumps({
        'frame_index': i, 'relative_ms': i * 100, 'width': 100, 'height': 100,
        'keypoints': joints}) for i, joints in enumerate(frames)), encoding='utf-8')


def test_prefix_is_invariant_to_future_frames(tmp_path):
    baseline = [points(), points(displacement=2), None]
    short = tmp_path/'short.txt'; long = tmp_path/'long.txt'
    raw_file(short, baseline)
    raw_file(long, baseline + [points(displacement=24)] * 20)
    a = normalize_rows(short, 'yolo'); b = normalize_rows(long, 'yolo')
    np.testing.assert_array_equal(a['keypoints'], b['keypoints'][:len(baseline)])
    np.testing.assert_array_equal(a['scales_px'], b['scales_px'][:len(baseline)])
    assert a['feature_version'] == FEATURE_VERSION
    assert a['keypoints'].shape == (3, 13, 4)
    assert (a['keypoints'][-1] == 0).all()


def test_live_frame_by_frame_equals_offline(tmp_path):
    frames = [points(), None, points(displacement=3), points(confidence=.10)]
    raw = tmp_path/'clip.txt'; raw_file(raw, frames)
    offline = normalize_rows(raw, 'yolo')['keypoints']
    normalizer = CausalPoseNormalizer()
    live = np.stack([normalizer.transform(kp, 100, 100)[0] for kp in frames])
    np.testing.assert_array_equal(live, offline)
    assert live[-1, :, 3].sum() == 0
    assert (live[-1, :, :2] == 0).all()


def test_output_must_be_separate_and_non_overwriting(tmp_path):
    dataset = tmp_path/'dataset'; dataset.mkdir()
    with pytest.raises(ValueError):
        build_processed(dataset, 'yolo')
    with pytest.raises(ValueError):
        build_processed(dataset, 'yolo', output_root=dataset/'processed')
    output = tmp_path/'output'; output.mkdir(); (output/'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        build_processed(dataset, 'yolo', output_root=output)
    assert (output/'sentinel').read_text() == 'keep'
