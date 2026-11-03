"""Phase 3.1: baseline instrumentation does not change live inference behavior."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from scripts import benchmark_inference_pipeline as benchmark
from inference import pipeline


class FakeRunner:
    def __init__(self, *args, **kwargs):
        self.predictor = SimpleNamespace(results=[SimpleNamespace(speed={
            'preprocess': 1.1, 'inference': 2.2, 'postprocess': .9})], trackers=[])
        self.engine = SimpleNamespace(predictor=self.predictor)
        self.reset_count = 0

    def track_frame(self, frame):
        points = np.zeros((17, 3), np.float32)
        for idx in range(17):
            points[idx] = [30 + idx, 40 + idx, .95]
        return [{'bbox': [10, 10, 110, 115],
                 'keypoints': points.tolist(), 'track_id': 1}]

    def reset_tracking(self):
        self.reset_count += 1

    def close(self):
        pass


class FakeClassifier:
    conf_thresh = .35
    ema_alpha = .25
    labels = [f'C{i}' for i in range(6)]

    def __init__(self, *args, **kwargs):
        pass

    def predict(self, x):
        return np.array([.9,.02,.02,.02,.02,.02], np.float32)


def frame():
    return np.zeros((120, 160, 3), np.uint8)


def test_metrics_missing_stage_not_zero_and_percentiles():
    assert benchmark.metrics([None, float('nan')])['fps'] is None
    result = benchmark.metrics([1, 2, 3, 4])
    assert result['count'] == 4
    assert result['p95_ms'] > result['p50_ms']
    assert result['fps'] == 400.0


def test_instrumentation_wrap_restores_and_preserves_predictions(monkeypatch):
    monkeypatch.setattr(pipeline, 'PoseRunner', FakeRunner)
    worker = pipeline.PosePipeline('fake.pt', FakeClassifier(), stable_frames=3)
    before = worker.process(frame())[0]['class_id']
    worker.reset()
    old_draw = pipeline.draw_overlay
    with benchmark.Instrument(worker, overlay=False) as meter:
        measured = meter.process(frame())
        assert measured['class_id'] == before
        assert measured['selected_pose'] == 1
        assert measured['classified'] == 1
        assert measured['yolo_forward_ms'] == 2.2
        assert measured['overlay_ms'] is not None
        assert measured['mlp_predict_ms'] >= 0
        assert measured['normalizer_ms'] >= 0
        assert measured['selector_ms'] >= 0
        assert measured['pose_runner_ms'] >= 0
    assert pipeline.draw_overlay is old_draw
    assert worker.process(frame())[0]['class_id'] == before
    worker.close()


def test_instrument_on_second_clip_after_selector_reset(monkeypatch):
    """A clip reset must preserve both temporal state isolation and profiler hooks."""
    monkeypatch.setattr(pipeline, 'PoseRunner', FakeRunner)
    worker = pipeline.PosePipeline('fake.pt', FakeClassifier(), overlay=False)
    selector = worker.selector
    try:
        with benchmark.Instrument(worker, overlay=False) as meter:
            first = meter.process(frame())
            worker.reset()
            second = meter.process(frame())
            assert worker.selector is selector
            assert first['selector_ms'] >= 0
            assert second['selector_ms'] >= 0
            assert second['normalizer_ms'] >= 0
            assert first['class_id'] == second['class_id']
    finally:
        worker.close()


def test_cli_fake_video_generates_versioned_output(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, 'PoseRunner', FakeRunner)
    monkeypatch.setattr(benchmark, 'PostureClassifier', FakeClassifier)
    pose = tmp_path / 'yolo11n-pose.pt'
    pose.write_bytes(b'dummy')
    model = tmp_path / 'mlp_wide' / 'v002'
    model.mkdir(parents=True)
    (model / 'model_final.pt').write_bytes(b'dummy')
    video = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 15, (160, 120))
    assert writer.isOpened()
    for _ in range(7):
        writer.write(frame())
    writer.release()
    args = ['--video',str(video),'--pose',str(pose),'--classifier',str(model),
            '--device','cpu','--cls-device','cpu','--frames','5',
            '--rounds','2','--warmup','1','--output',str(tmp_path/'out')]
    output = benchmark.main(args)
    assert output.name == 'v001'
    config = json.loads((output/'config.json').read_text())
    assert config['frames_requested'] == 5
    with (output/'summary.csv').open() as f:
        result = next(csv.DictReader(f))
    assert int(result['frames']) == 10
    assert float(result['yolo_fps']) > 0
    assert not result['mlp_fps'] or float(result['mlp_fps']) > 0
    assert (output/'stage_latency.csv').exists()
    assert (output/'runs.csv').exists()
    assert not (output/'INCOMPLETE.txt').exists()
    output2 = benchmark.main(args)
    assert output2.name == 'v002'


def test_pure_mlp_forward_separated_from_predict(monkeypatch):
    import torch
    monkeypatch.setattr(pipeline, 'PoseRunner', FakeRunner)

    class TorchClassifier(FakeClassifier):
        backend = 'pt'
        device = 'cpu'
        def __init__(self):
            self.model = torch.nn.Linear(52, 6).eval()
        def predict(self, x):
            with torch.inference_mode():
                scores = self.model(torch.from_numpy(np.asarray(x, np.float32).reshape(1, 52)))
                return torch.softmax(scores, dim=-1).numpy()[0]

    worker = pipeline.PosePipeline('fake.pt', TorchClassifier())
    with benchmark.Instrument(worker, overlay=False) as meter:
        result = meter.process(frame())
    assert result['mlp_forward_ms'] >= 0
    assert result['mlp_predict_ms'] >= result['mlp_forward_ms']
    worker.close()


def test_cli_help_survives_ros_scripts_shadowing(tmp_path):
    """Exercise both invocation styles with a hostile ROS-like scripts package."""
    root = Path(__file__).resolve().parents[1]
    fake_ros = tmp_path / 'ros'
    ros_package = fake_ros / 'scripts'
    ros_package.mkdir(parents=True)
    (ros_package / '__init__.py').write_text(
        "raise RuntimeError('Imported the wrong ROS scripts package')\n")
    env = os.environ.copy()

    # Direct file invocation: ROS wins qualified `scripts.*` imports when its
    # path precedes project root. Local sibling import must still succeed.
    env['PYTHONPATH'] = os.pathsep.join([str(fake_ros), str(root),
                                        str(root / 'collector')])
    direct = subprocess.run(
        [sys.executable, '-B', str(root / 'scripts' / 'benchmark_inference_pipeline.py'), '--help'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert direct.returncode == 0, direct.stderr
    assert '--cls-backend' in direct.stdout

    # Package invocation: the explicit project package must beat the ROS one
    # when project root appears earlier on Python's import path.
    env['PYTHONPATH'] = os.pathsep.join([str(root), str(fake_ros),
                                        str(root / 'collector')])
    package = subprocess.run(
        [sys.executable, '-B', '-m', 'scripts.benchmark_inference_pipeline', '--help'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert package.returncode == 0, package.stderr
    assert '--cls-backend' in package.stdout
