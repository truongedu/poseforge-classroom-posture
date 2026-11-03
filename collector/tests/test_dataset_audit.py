import csv
import json
from pathlib import Path
from posture_collector.dataset_audit import audit_dataset, snapshot_files, compare_snapshots
from posture_collector.storage import DataStore, save_json


def test_audit_read_only_and_snapshots_detect_modification(tmp_path):
    db = DataStore(tmp_path)
    db.add_subject('CHANH', True)
    folder = db.create_take('CHANH', 0, 'session_20261027_1124')
    raw = folder/'cam01_raw.mp4'; raw.write_bytes(b'MP4 TEST')
    save_json(folder/'metadata.json', {'take_id':'CHANH_session_20261027_1124_L0_take_001',
        'person_id':'CHANH', 'session_id':'session_20261027_1124', 'label':0,
        'status':'needs_review', 'cameras': {'cam01': {'view':'front','status':'ok',
        'capture_span_s':2.3, 'frames':46}}})
    manifest,_ = db.export_manifest()
    old = manifest.read_bytes()
    before = {'files': snapshot_files(tmp_path)}
    report = audit_dataset(tmp_path)
    assert report['clip_camera_records'] == 1
    assert report['metadata_manifest_status_differ'] == 1
    assert not report['errors']
    assert manifest.read_bytes() == old
    assert compare_snapshots(before, {'files': snapshot_files(tmp_path)}) == {
        'missing':[], 'changed':[], 'added':[]}
    raw.write_bytes(b'CORRUPTED')
    assert len(compare_snapshots(before, {'files': snapshot_files(tmp_path)})['changed']) == 1
