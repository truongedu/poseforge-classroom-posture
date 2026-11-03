"""Phase 3.2 correctness, throughput-off path and V2 feature compatibility."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

from inference import pipeline, run


def frame():
    return np.zeros((120, 160, 3), dtype=np.uint8)


class FakeClassifier:
    conf_thresh = .35
    ema_alpha = .25
    labels = [f'class_{i}' for i in range(6)]

    def __init__(self):
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        assert np.asarray(features).shape == (13, 4)
        return np.asarray([.94, .02, .01, .01, .01, .01], np.float32)


class SequencePose:
    observations = []

    def __init__(self, *args, **kwargs):
        self.observations = list(type(self).observations)
        self.idx = 0
        self.resets = 0

    def track_frame(self, _frame):
        count, track = self.observations[min(self.idx, len(self.observations) - 1)]
        self.idx += 1
        if count is None:
            return []
        kp = np.zeros((17, 3), np.float32)
        for i in range(count):
            kp[i] = [25 + 2 * i, 30 + 2 * i, .95]
        return [{'bbox': [10, 10, 130, 110], 'keypoints': kp.tolist(),
                 'track_id': track}]

    def reset_tracking(self):
        self.resets += 1

    def close(self):
        pass


def test_no_stable_result_for_repeated_low_confidence():
    s = pipeline.StableClass(window=3, min_conf=.55)
    weak = np.asarray([.40, .30, .10, .08, .07, .05])
    for _ in range(8):
        assert s.update(weak) is None
    strong = np.asarray([.95, .01, .01, .01, .01, .01])
    assert s.update(strong) is None
    assert s.update(strong) is None
    assert s.update(strong) == 0
    assert s.update(weak) is None


def test_sparse_keypoints_no_prediction_and_recovery(monkeypatch):
    SequencePose.observations = [(13, 1)] * 3 + [(3, 1), (0, 1), (None, None)] + [(13, 1)] * 3
    monkeypatch.setattr(pipeline, 'PoseRunner', SequencePose)
    classifier = FakeClassifier()
    p = pipeline.PosePipeline('fake.pt', classifier, stable_frames=3, overlay=False)
    try:
        ids = []
        for index in range(9):
            input_frame = frame()
            info, output_frame = p.process(input_frame)
            assert output_frame is input_frame  # no copy when overlay disabled
            ids.append(info['stable_id'])
            if 3 <= index < 6:
                assert info['class_id'] is None
                assert info['probs'] is None
                assert info['label'] is None
        assert ids == [None, None, 0, None, None, None, None, None, 0]
        assert classifier.calls == 6
    finally:
        p.close()


def test_extended_sparse_or_missing_sequence_resets_state(monkeypatch):
    SequencePose.observations = [(13, 7)] + [(2, 7)] * 9 + [(13, 8)]
    monkeypatch.setattr(pipeline, 'PoseRunner', SequencePose)
    p = pipeline.PosePipeline('fake.pt', FakeClassifier(), stable_frames=2, overlay=False)
    try:
        p.process(frame())
        for _ in range(9):
            info, _ = p.process(frame())
            assert info['keypoints_valid'] == 2
            assert info['class_id'] is None
        assert p.last_track_id is None
        assert p.normalizer.center is None
        last, _ = p.process(frame())
        assert last['class_id'] == 0 and last['stable_id'] is None
    finally:
        p.close()


def test_track_id_switch_clears_stability_without_model_reload(monkeypatch):
    SequencePose.observations = [(13, 4)] * 3 + [(13, 9)] * 3
    monkeypatch.setattr(pipeline, 'PoseRunner', SequencePose)
    p = pipeline.PosePipeline('fake.pt', FakeClassifier(), stable_frames=3, overlay=False)
    try:
        expected = [None, None, 0, None, None, 0]
        assert [p.process(frame())[0]['stable_id'] for _ in expected] == expected
        assert p.last_track_id == 9
        p.reset()
        assert p.last_track_id is None
        assert p.pose.resets == 1
    finally:
        p.close()


def test_overlay_disabled_skips_rendering_for_valid_and_missing_pose(monkeypatch):
    SequencePose.observations = [(13, 1), (None, None)]
    monkeypatch.setattr(pipeline, 'PoseRunner', SequencePose)
    monkeypatch.setattr(pipeline, 'draw_overlay',
                        lambda *_: (_ for _ in ()).throw(AssertionError('overlay invoked')))
    p = pipeline.PosePipeline('fake.pt', FakeClassifier(), overlay=False)
    try:
        for _ in range(2):
            input_frame = frame()
            _, output_frame = p.process(input_frame)
            assert output_frame is input_frame
    finally:
        p.close()


def test_collector_legacy_feature_api_is_exact_shared_class():
    from shared.pose_features import CausalPoseNormalizer
    from posture_collector.features import CausalPoseNormalizer as CollectorNormalizer
    assert CollectorNormalizer is CausalPoseNormalizer
    # Collector-only scripts have collector/ but not project root on sys.path.
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(root / 'collector')
    result = subprocess.run([sys.executable, '-B', '-c',
                             'from posture_collector.features import CausalPoseNormalizer; '
                             'assert CausalPoseNormalizer.__module__ == "shared.pose_features"'],
                            cwd=root.parent, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_camera_headless_without_events_does_not_draw_or_write_video(monkeypatch, tmp_path):
    SequencePose.observations = [(13, 1)]
    monkeypatch.setattr(pipeline, 'PoseRunner', SequencePose)
    monkeypatch.setattr(run, 'label_on_frame',
                        lambda *_: (_ for _ in ()).throw(AssertionError('frame copy for recording')))
    monkeypatch.setattr(run, 'EventRecorder',
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('recorder created')))

    class Reader:
        def __init__(self, *_args):
            self.seq = 0
        def snapshot(self):
            import time
            self.seq += 1
            return 1, self.seq, time.monotonic(), frame()
        def close(self):
            pass

    monkeypatch.setattr(run, 'CameraReader', Reader)
    args = Namespace(pose='fake.pt', device='auto', stable_frames=3, min_conf=.55,
                     overlay=False, record_events=False, display=False, name='',
                     record_seconds=.2, event_fps=10, open_timeout=100, read_timeout=100,
                     stale_seconds=8., freeze_seconds=15., max_seconds=.05)
    run.run_cameras(['camera-1', 'camera-2'], args, FakeClassifier(), tmp_path)
    for name in ('cam01', 'cam02'):
        performance = json.loads((tmp_path / name / 'performance.json').read_text())
        assert performance['processed_frames'] > 0
        assert performance['saved_classes'] == []
        assert not list((tmp_path / name).glob('*.mp4'))


def test_pt_default_ignores_preexisting_onnx_file(tmp_path):
    import torch
    from shared.mlp_model import PostureMLP
    from inference.pipeline import PostureClassifier
    contract = {
        'version': 'pose13_xy_conf_mask_causal_v2',
        'flatten_dim': 52,
        'shape_per_frame': [13, 4],
        'channels': ['x_norm', 'y_norm', 'confidence', 'valid_mask'],
        'joints': pipeline.REQUIRED_JOINTS,
        'confidence_threshold': .35,
        'ema_alpha': .25,
    }
    (tmp_path / 'feature_contract.json').write_text(json.dumps(contract))
    (tmp_path / 'model.onnx').write_bytes(b'not a real onnx file')
    model = PostureMLP('mlp_wide', .0)
    torch.save({'stage': 'final', 'feature_contract': contract,
                'label_names': [f'cls{i}' for i in range(6)],
                'model_name': 'mlp_wide', 'model_config': {'dropout': .0},
                'model_state_dict': model.state_dict()}, tmp_path / 'model_final.pt')
    cls = PostureClassifier(tmp_path)
    assert cls.backend == 'pt'
    assert cls.predict(np.zeros((13, 4), np.float32)).shape == (6,)


def test_long_sparse_pose_clears_selector_history_without_changing_thresholds(monkeypatch):
    SequencePose.observations = [(13, 7)] + [(2, 7)] * 9 + [(13, 8)]
    monkeypatch.setattr(pipeline, 'PoseRunner', SequencePose)
    p = pipeline.PosePipeline('fake.pt', FakeClassifier(), overlay=False)
    try:
        p.process(frame())
        selector = p.selector
        selector.temporal_iou = .71
        assert selector.previous is not None
        for _ in range(8):
            p.process(frame())
            assert p.selector is selector
            assert p.selector.previous is not None
        p.process(frame())
        assert p.selector is selector
        assert p.selector.previous is None
        assert p.selector.temporal_iou == .71
        assert p.selector.missed == 0
        assert p.last_track_id is None
        assert p.normalizer.center is None
        p.process(frame())
        p.reset()
        assert p.selector is selector
        assert p.selector.previous is None
        assert p.selector.temporal_iou == .71
    finally:
        p.close()


def test_camera_setup_failure_releases_every_initialized_resource(monkeypatch, tmp_path):
    import pytest

    for failing_stage in ('pipeline', 'reader', 'recorder'):
        created = {'pipeline': [], 'reader': [], 'recorder': []}

        class Resource:
            def __init__(self):
                self.closed = False
            def close(self):
                self.closed = True

        def factory(stage):
            def construct(*args, **kwargs):
                if stage == failing_stage and len(created[stage]) == 1:
                    raise RuntimeError(f'{stage} setup failed')
                resource = Resource()
                created[stage].append(resource)
                return resource
            return construct

        monkeypatch.setattr(run, 'PosePipeline', factory('pipeline'))
        monkeypatch.setattr(run, 'CameraReader', factory('reader'))
        monkeypatch.setattr(run, 'EventRecorder', factory('recorder'))
        options = Namespace(pose='fake.pt', device='auto', stable_frames=5,
                            min_conf=.55, overlay=False, record_events=True,
                            record_seconds=3., event_fps=10., open_timeout=100,
                            read_timeout=100, freeze_seconds=15.)
        with pytest.raises(RuntimeError, match=f'{failing_stage} setup failed'):
            run.run_cameras(['cam1', 'cam2'], options, FakeClassifier(), tmp_path)
        assert all(r.closed for group in created.values() for r in group)
        assert len(created['pipeline']) == (1 if failing_stage == 'pipeline' else 2)
        assert len(created['reader']) == (0 if failing_stage == 'pipeline' else
                                          1 if failing_stage == 'reader' else 2)
        assert len(created['recorder']) == (1 if failing_stage == 'recorder' else 0)
