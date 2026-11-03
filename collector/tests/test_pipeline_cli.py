"""Offline pipeline compatibility. Inference is a deterministic test stub."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from posture_collector.storage import DataStore, save_json
from posture_collector.process import extract_all
from posture_collector.pose import person_record


@pytest.mark.parametrize('backend,npoints', [('yolo', 17), ('mediapipe', 33)])
def test_extraction_then_existing_cli_commands(tmp_path, monkeypatch, backend, npoints):
    import posture_collector.process as process
    db = DataStore(tmp_path)
    for pid in ('TQB', 'DTT', 'NVA', 'ABC', 'DEF', 'GHI'): db.add_subject(pid, True)
    folder = db.create_take('TQB', 0, 'session_pipeline')
    configs = {}
    digests = {}
    for cid in ('cam01', 'cam02'):
        video = folder/f'{cid}_raw.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 20, (128, 96))
        assert writer.isOpened()
        for _ in range(120): writer.write(np.full((96, 128, 3), 60, np.uint8))
        writer.release(); digests[cid] = hashlib.sha256(video.read_bytes()).hexdigest()
        configs[cid] = dict(frames=120, view='front', status='ok', capture_span_s=5.95)
    save_json(folder/'metadata.json', dict(take_id='TQB_session_pipeline_L0_take_001',
        person_id='TQB', session_id='session_pipeline', label=0, status='ok', cameras=configs))
    class StubPose:
        last_world = []
        tracking_session = 'test-session'
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def detect(self, image, ms):
            if backend == 'yolo':
                points = [[40+i*2, 30+i, .99] for i in range(17)]
                points[5] = [40, 40, .99]; points[6] = [70, 40, .99]
                self.last_tracks=[person_record(0,points,[20,10,90,90],.99,1)]
            else:
                points = [[.3+i*.01, .3+i*.01, 0, .99, .99] for i in range(33)]
            return [points]
    monkeypatch.setattr(process, 'PoseRunner', StubPose)
    extracted = extract_all(tmp_path, backend, 'TEST STUB, NOT REAL MODEL')
    assert len(extracted) == 2 and all(r['frames']==120 for r in extracted)
    for cid in configs:
        raw = folder/'keypoints'/f'{cid}_{backend}{npoints}_raw.txt'
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
        assert len(rows)==120 and len(rows[0]['keypoints'])==npoints
    db.export_manifest()  # Create labels.csv for the read-only audit command
    project = Path(__file__).resolve().parents[1]
    output = tmp_path.parent / ('processed_' + backend)
    for command in (['normalize', '--backend', backend, '--output', str(output)], ['audit']):
        result = subprocess.run([sys.executable, str(project/'process_dataset.py'), '--data', str(tmp_path), *command],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        assert result.returncode == 0, result.stderr
    assert list(output.rglob('*_normalized13.npz'))
    assert not list((tmp_path/'processed').rglob('*.npz'))
    for cid in configs:
        assert hashlib.sha256((folder/f'{cid}_raw.mp4').read_bytes()).hexdigest()==digests[cid]
