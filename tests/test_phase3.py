"""Fast local tests; real YOLO camera and ONNX export need Ubuntu smoke."""
import csv
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from inference import pipeline, run


def _frame():
    return np.zeros((120, 160, 3), np.uint8)


def test_version_and_secret_redaction(tmp_path):
    assert run.make_run_dir(tmp_path, 'cam01').name == 'cam01_v001'
    assert run.make_run_dir(tmp_path, 'cam01').name == 'cam01_v002'
    assert 'PASS' not in run.clean_source('rtsp://USER:PASS@192.168.1.5:554/live?token=secret')
    assert 'token' not in run.clean_source('rtsp://USER:PASS@192.168.1.5:554/live?token=secret')


def test_stable_class_needs_consistency():
    smooth = pipeline.StableClass(window=3, min_conf=.55)
    probs = np.array([.8, .02, .02, .02, .02, .12], np.float32)
    assert smooth.update(probs) is None
    assert smooth.update(probs) is None
    assert smooth.update(probs) == 0
    smooth.reset()
    assert smooth.update(probs) is None


def test_event_recorder_bounded(tmp_path):
    rec = run.EventRecorder(tmp_path, [f'label_{i}' for i in range(6)], duration_s=.4, fps=10)
    for cls in range(6):
        for _ in range(9):
            rec.add(_frame(), {'stable_id': cls, 'score': .9})
    rec.close()
    videos = list(tmp_path.glob('class_*.mp4'))
    assert len(videos) == 6
    with (tmp_path / 'events.csv').open() as file:
        assert len(list(csv.DictReader(file))) == 6
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 4
        cap.release()


class FakePoseRunner:
    def __init__(self, *args, **kwargs):
        self.counter = 0
        self.reset_count = 0

    def track_frame(self, frame):
        self.counter += 1
        kp = np.zeros((17, 3), np.float32)
        for j in range(17):
            kp[j] = [60 + j, 30 + j * 2, .96]
        return [{'bbox': [25, 10, 140, 118], 'keypoints': kp.tolist(), 'track_id': 1}]

    def reset_tracking(self):
        self.reset_count += 1

    def close(self):
        pass


class FakeClassifier:
    labels = [f'label_{i}' for i in range(6)]
    conf_thresh = .35
    ema_alpha = .25

    def predict(self, features):
        assert np.asarray(features).shape == (13, 4)
        return np.array([.99, .002, .002, .002, .002, .002], np.float32)


def test_inference_normalizer_selector_and_offline_video(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'PoseRunner', FakePoseRunner)
    model = FakeClassifier()
    p = pipeline.PosePipeline('fake.pt', model, stable_frames=3)
    labels = []
    for _ in range(5):
        info, img = p.process(_frame())
        assert img.shape == _frame().shape
        assert info['keypoints_valid'] == 13
        labels.append(info['stable_id'])
    assert labels == [None, None, 0, 0, 0]
    p.reset()
    assert p.pose.reset_count == 1
    assert p.process(_frame())[0]['stable_id'] is None
    p.close()

    video = tmp_path / 'input.mp4'
    writer = run.video_writer(video, 160, 120, 12.)
    for _ in range(6):
        writer.write(_frame())
    writer.release()
    out = tmp_path / 'video_demo'
    out.mkdir()
    options = Namespace(pose='fake.pt', device='auto', stable_frames=3,
                        min_conf=.55, display=False, max_frames=0)
    run.run_file(video, 'video', options, model, out)
    with (out / 'frames.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert sum(row['stable_id'] == '0' for row in rows) == 4
    assert (out / 'annotated.mp4').stat().st_size > 100


def test_real_selector_accepts_nested_duplicate_boxes():
    from collector.posture_collector.single_person import SinglePersonSelector
    points = [[70, 50, .9]] * 17
    persons = [{'bbox': [10, 10, 200, 200], 'keypoints': points},
               {'bbox': [20, 20, 195, 195], 'keypoints': points}]
    chosen, reason = SinglePersonSelector().select(persons)
    assert chosen is persons[0]
    assert reason == 'duplicate_boxes'


def test_camera_event_and_metrics_with_two_sources(tmp_path, monkeypatch):
    import time
    monkeypatch.setattr(pipeline, 'PoseRunner', FakePoseRunner)

    class Reader:
        def __init__(self, source, name, *args):
            self.name = name
            self.seq = 0
        def snapshot(self):
            self.seq += 1
            return (1, self.seq, time.monotonic(), _frame())
        def close(self):
            pass

    monkeypatch.setattr(run, 'CameraReader', Reader)
    opts = Namespace(pose='fake.pt', device='auto', stable_frames=3,
                     min_conf=.55, display=False, name='', record_seconds=.2,
                     event_fps=10, record_events=True, open_timeout=100, read_timeout=100,
                     stale_seconds=8., freeze_seconds=15., max_seconds=.12)
    run.run_cameras(['rtsp://test1', 'rtsp://test2'], opts, FakeClassifier(), tmp_path)
    for cam in ('cam01', 'cam02'):
        folder = tmp_path / cam
        assert (folder / 'class_0.mp4').is_file()
        assert (folder / 'events.csv').is_file()
        data = __import__('json').loads((folder / 'performance.json').read_text())
        assert data['processed_frames'] > 3
        assert data['saved_classes'] == [0]
        assert data['mean_inference_ms'] >= 0


def test_reader_reconnect_increments_generation(monkeypatch):
    import time
    openings = []

    class FakeCapture:
        def __init__(self, *args):
            self.calls = 0
            openings.append(self)
        def isOpened(self):
            return True
        def read(self):
            self.calls += 1
            return (True, _frame()) if self.calls == 1 else (False, None)
        def release(self):
            pass

    monkeypatch.setattr(run.cv2, 'VideoCapture', FakeCapture)
    reader = run.CameraReader('0', 'fake')
    try:
        deadline = time.monotonic() + 1.4
        while reader.generation < 2 and time.monotonic() < deadline:
            time.sleep(.02)
        assert reader.generation >= 2
        assert len(openings) >= 2
    finally:
        reader.close()


def test_probability_dashboard_has_six_colored_bars_and_live_metrics():
    info = {'class_id': 3, 'label': 'Leaning back', 'score': .68,
            'stable_id': None, 'probs': [.02, .02, .03, .68, .15, .10]}
    img = pipeline.draw_dashboard(_frame(), info, [f'Class {i}' for i in range(6)], fps=24.8)
    assert img.shape == (450, 520, 3)
    colors = [tuple(x) for x in pipeline.CLASS_COLORS]
    for color in colors:
        assert np.any(np.all(img == np.array(color, dtype=np.uint8), axis=-1)), color
    empty = pipeline.draw_dashboard(_frame(), {}, [f'Class {i}' for i in range(6)], status='RECONNECTING')
    assert empty.shape == img.shape


def test_csv_has_six_probabilities_without_unknown_keys():
    from inference.run import log_row, FIELDS
    info = {'class_id': 2, 'label': 'Desk', 'score': .80,
            'stable_id': None, 'probs': [.01, .01, .80, .07, .05, .06],
            'keypoints_valid': 13, 'selection_reason': 'single_detection',
            'track_id': 1, 'inference_ms': 4.}
    row = log_row(5, .5, info)
    assert set(row).issubset(set(FIELDS))
    assert row['p2'] == .80
    assert abs(sum(row[f'p{i}'] for i in range(6)) - 1.0) < 1e-6
    assert all(not k.startswith('p') or k[1:].isdigit() for k in row if k.startswith('p'))


def test_auto_backend_falls_back_to_pt_and_pt_matches_checkpoint(tmp_path):
    import zipfile, io, json, torch
    from inference.pipeline import PostureClassifier
    from shared.mlp_model import PostureMLP
    import os
    source = Path(os.environ.get('POSTURE_TEST_BUNDLE', 'final_model_bundle.zip'))
    if not source.exists():
        import pytest
        pytest.skip('Real training bundle not available')
    with zipfile.ZipFile(source) as z:
        for filename in ('model_final.pt', 'feature_contract.json'):
            (tmp_path / filename).write_bytes(z.read(filename))
    cls = PostureClassifier(tmp_path, backend='auto', device='cpu')
    assert cls.backend == 'pt' and len(cls.labels) == 6
    a = np.arange(52, dtype=np.float32).reshape(13, 4) * .01
    got = cls.predict(a)
    assert got.shape == (6,)
    assert abs(float(got.sum()) - 1.) < 1e-5
    ckpt = torch.load(tmp_path / 'model_final.pt', map_location='cpu', weights_only=True)
    expected = PostureMLP(ckpt['model_name'], ckpt['model_config']['dropout'])
    expected.load_state_dict(ckpt['model_state_dict'], strict=True)
    with torch.inference_mode():
        p = torch.softmax(expected.eval()(torch.from_numpy(a.reshape(1, -1))), -1).numpy()[0]
    np.testing.assert_allclose(got, p, rtol=1e-6, atol=1e-7)


def test_exporter_builder_from_real_checkpoint():
    import zipfile, io, torch
    from scripts.export_onnx import build_model
    from shared.mlp_model import PostureMLP
    import os
    z = Path(os.environ.get('POSTURE_TEST_BUNDLE', 'final_model_bundle.zip'))
    if not z.exists():
        import pytest
        pytest.skip('Real training bundle not available')
    with zipfile.ZipFile(z) as archive:
        ckpt = torch.load(io.BytesIO(archive.read('model_final.pt')), weights_only=True, map_location='cpu')
    model = build_model(ckpt)
    assert isinstance(model, PostureMLP)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    with torch.inference_mode():
        assert model.eval()(torch.zeros(2, 52)).shape == (2, 6)
