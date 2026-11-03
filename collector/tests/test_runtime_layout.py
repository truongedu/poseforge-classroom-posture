"""Regression checks for active camera configuration and default runtime path."""
import json
from pathlib import Path

from posture_collector.storage import DataStore, DEFAULT_COLLECTOR_DATA_ROOT

REPO = Path(__file__).resolve().parents[2]


def test_camera_url_persists_across_app_restarts(tmp_path):
    root = tmp_path / 'artifacts' / 'collector_runtime' / 'Real_Posture_Dataset'
    store = DataStore(root)
    cams = store.cameras()
    cams[0]['url'] = 'http://192.0.2.10:4747/video'
    cams[1]['url'] = 'http://192.0.2.11:4747/video'
    store.set_cameras(cams)
    reopened = DataStore(root)
    assert reopened.cameras() == cams
    assert json.loads((root / 'config' / 'cameras.json').read_text()) == cams


def test_default_runtime_root_points_outside_collector():
    assert DEFAULT_COLLECTOR_DATA_ROOT == REPO / 'artifacts' / 'collector_runtime' / 'Real_Posture_Dataset'
