
"""Existing datasets must remain intact when adding subjects or takes."""
import csv
import json
from pathlib import Path
from posture_collector.storage import DataStore, save_json


def record(store, person, session, label, cam='cam01', status='ok'):
    folder = store.create_take(person, label, session)
    (folder / f'{cam}_raw.mp4').write_bytes(b'raw video must not change')
    save_json(folder / 'metadata.json', {
        'take_id': f'{person}_{session}_L{label}_{folder.name}',
        'person_id': person, 'session_id': session, 'label': label,
        'status': 'needs_review',
        'cameras': {cam: {'view': 'front', 'status': status,
                         'frames': 60, 'capture_span_s': 2.7}},
    })
    return folder


def test_existing_manifest_is_byte_preserved_when_new_samples_added(tmp_path):
    store = DataStore(tmp_path)
    store.add_subject('CHANH', True)
    first = record(store, 'CHANH', 'session_20261027_1124', 0)
    path, count = store.export_manifest()
    assert count == 1
    historical = path.read_bytes().replace(b'"ok"', b'"ok"')
    # Represent the original Windows CSV: \ paths and an earlier curated status.
    old = historical.decode('utf-8-sig').replace('recordings/', 'recordings\\')
    old = old.replace('/session_', '\\session_').replace('/label_', '\\label_')
    old = old.replace('/take_', '\\take_').replace('/metadata.json', '\\metadata.json')
    old = old.replace('/cam01_raw.mp4', '\\cam01_raw.mp4')
    old = old.replace(',2.7,60,ok', ',2.7,60,curated')
    path.write_bytes(b'\xef\xbb\xbf' + old.encode())
    before = path.read_bytes()
    first_bytes = (first / 'cam01_raw.mp4').read_bytes()
    another = record(store, 'CHANH', 'session_20261027_1124', 0)
    assert another.name == 'take_002'
    store.add_subject('TQB', True)
    new_person = record(store, 'TQB', 'session_20261028_1441', 5)
    assert store.export_manifest()[1] == 3
    after = path.read_bytes()
    assert after.startswith(before)
    assert after.count(b'clip_id') == 1
    assert store.export_manifest()[1] == 3
    assert path.read_bytes() == after  # idempotent
    with path.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3
    assert rows[0]['status'] == 'curated'
    assert '\\' in rows[1]['video_path']
    assert [r['clip_id'] for r in rows] == list(dict.fromkeys(r['clip_id'] for r in rows))
    assert (first / 'cam01_raw.mp4').read_bytes() == first_bytes
    assert (new_person / 'cam01_raw.mp4').exists()


def test_schema_mismatch_never_modifies_legacy_manifest(tmp_path):
    store = DataStore(tmp_path)
    store.add_subject('CHANH', True)
    record(store, 'CHANH', 's_1', 0)
    bad = store.root / 'labels.csv'
    bad.write_text('clip_id,unknown\nold,value\n')
    before = bad.read_bytes()
    try:
        store.export_manifest()
    except ValueError as e:
        assert 'schema mismatch' in str(e)
    else:
        raise AssertionError('schema mismatch was not blocked')
    assert bad.read_bytes() == before


def test_old_take_folders_are_not_overwritten(tmp_path):
    db = DataStore(tmp_path)
    db.add_subject('ABC', True)
    first = db.create_take('ABC', 2, 'same_session')
    (first / 'metadata.json').write_bytes(b'original metadata')
    second = db.create_take('ABC', 2, 'same_session')
    assert second.name == 'take_002'
    assert (first / 'metadata.json').read_bytes() == b'original metadata'
