"""V3 safety tests: preview, append, overwrite backup and split source integrity."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from posture_collector.storage import DataStore, save_json
from posture_collector import reextract
from posture_collector.split_tools import make_split_plan, export_plan


@pytest.fixture
def existing(tmp_path):
    root = tmp_path/'dataset'
    store = DataStore(root)
    for person in ('CHANH','DGD'):
        store.add_subject(person, True)
        for label in (0,1):
            take = store.create_take(person, label, 'session_20261028_1234')
            cams={}
            for camera in ('cam01','cam02'):
                video = take/f'{camera}_raw.mp4'
                video.write_bytes(b'ORIGINAL_VIDEO_' + camera.encode())
                stamps = take/f'{camera}_timestamps.csv'
                with stamps.open('w',newline='') as file:
                    w=csv.DictWriter(file,fieldnames=['writer_frame_index','host_recv_monotonic_ns'])
                    w.writeheader()
                    for i in range(100):
                        w.writerow({'writer_frame_index':i,
                                    'host_recv_monotonic_ns':10000000000 + i*100000000})
                cams[camera]={'view':'front' if camera=='cam01' else 'oblique'}
            save_json(take/'metadata.json', {'person_id':person,'session_id':'session_20261028_1234',
                'label':label,'take_id':f'{person}_session_20261028_1234_L{label}_{take.name}',
                'cameras':cams,'status':'ok'})
    rows=[]
    for folder, meta in store.iter_takes():
        for camera in meta['cameras']:
            video=folder/f'{camera}_raw.mp4'
            rows.append(dict(clip_id=meta['take_id']+'_'+camera,take_id=meta['take_id'],
                person_id=meta['person_id'],session_id=meta['session_id'],
                camera_id=camera,camera_view=meta['cameras'][camera]['view'], label=str(meta['label']),
                video_path=video.relative_to(root).as_posix(),
                metadata_path=(folder/'metadata.json').relative_to(root).as_posix(),
                duration_s='5',frames='100',status='ok'))
    from posture_collector.dataset_audit import REQUIRED_COLUMNS
    with (root/'labels.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=REQUIRED_COLUMNS)
        writer.writeheader();writer.writerows(rows)
    weights=tmp_path/'test-pose.pt'
    weights.write_bytes(b'FAKE WEIGHTS')
    return root, weights


def fake_extract(video, backend, model, progress=None, *, device='auto', output_dir=None):
    camera=Path(video).name.split('_raw.mp4')[0]
    raw=Path(output_dir)/f'{camera}_yolo17_raw.txt'
    raw.parent.mkdir(parents=True,exist_ok=True)
    with raw.open('w') as file:
        for i in range(100):
            file.write(json.dumps({'frame_index':i,'backend':'yolo',
                'keypoints':[[float(i),float(i),0.9] for _ in range(17)]})+'\n')
    raw.with_suffix('.summary.json').write_text(json.dumps({'backend':'yolo','frames':100,
        'model_path':str(model),'raw_keypoints_txt':str(raw)}))
    return {'raw_keypoints_txt':str(raw),'frames':100,'selected_frames':100,
            'multiple_people':0,'extracted_at':'2026-10-28T00:00:00Z'}


def test_missing_only_and_safe_backup(existing,tmp_path,monkeypatch):
    root, weights=existing
    monkeypatch.setattr(reextract,'extract_video',fake_extract)
    src=(root/'labels.csv').read_bytes()
    video_hashes={p:p.read_bytes() for p in root.rglob('*_raw.mp4')}
    targets=reextract.list_targets(root, mode='missing')
    assert len(targets)==8
    assert not list((tmp_path/'backups').glob('*')) if (tmp_path/'backups').exists() else True
    result=reextract.reextract_targets(root,targets,weights,
        backup_root=tmp_path/'backups')
    assert result['succeeded']==8 and result['failed']==0
    assert not reextract.list_targets(root,mode='missing')
    assert (root/'labels.csv').read_bytes()==src
    assert all(p.read_bytes()==data for p,data in video_hashes.items())
    first=targets[0]
    old=first['raw'].read_bytes()
    scoped=reextract.list_targets(root,mode='selected',person='CHANH',label=0)
    assert len(scoped)==2
    with pytest.raises(FileExistsError):
        reextract.reextract_targets(root,scoped,weights,backup_root=tmp_path/'backups')
    result2=reextract.reextract_targets(root,scoped,weights,
        backup_root=tmp_path/'backups',overwrite=True)
    assert result2['succeeded']==2
    assert first['raw'].read_bytes()==old
    assert (root/'labels.csv').read_bytes()==src
    assert any(f.name==first['raw'].name for f in Path(result2['backup']).rglob('*'))
    summary=json.loads(first['summary'].read_text())
    assert summary['raw_keypoints_txt']==str(first['raw'])


def test_selected_scope_requires_person_but_allows_all_takes(existing):
    root,_=existing
    with pytest.raises(ValueError):reextract.list_targets(root,mode='selected')
    all_person = reextract.list_targets(root, mode='selected', person='CHANH')
    assert len(all_person) == 4  # all labels/takes, both cameras for this person
    assert {t['person'] for t in all_person} == {'CHANH'}
    assert len(reextract.list_targets(root, mode='selected', person='CHANH', camera='cam02')) == 2
    assert len(reextract.list_targets(root,mode='selected',person='DGD',label=1,camera='cam02'))==1



def test_selected_whole_person_overwrite_is_isolated(existing, tmp_path, monkeypatch):
    root, weights = existing
    monkeypatch.setattr(reextract, 'extract_video', fake_extract)
    targets = reextract.list_targets(root, mode='selected', person='CHANH')
    assert len(targets) == 4
    # One existing extraction must be backed up before replacement.
    old_raw = targets[0]['raw']
    old_raw.parent.mkdir(parents=True, exist_ok=True)
    old_raw.write_bytes(b'previous-keypoints')
    labels_before = (root/'labels.csv').read_bytes()
    videos_before = {p: p.read_bytes() for p in root.rglob('*_raw.mp4')}
    other_person_raw_before = tuple((root/'recordings'/'DGD').rglob('*_yolo17_raw.txt'))
    assert not other_person_raw_before

    report = reextract.reextract_targets(
        root, targets, weights, backup_root=tmp_path/'safe-backups', overwrite=True)
    assert report['selected'] == 4 and report['succeeded'] == 4 and report['failed'] == 0
    assert len(list((root/'recordings'/'CHANH').rglob('*_yolo17_raw.txt'))) == 4
    assert not list((root/'recordings'/'DGD').rglob('*_yolo17_raw.txt'))
    saved_raw = Path(report['backup']) / targets[0]['video'].parent.relative_to(root) / old_raw.name
    assert saved_raw.read_bytes() == b'previous-keypoints'
    assert (root/'labels.csv').read_bytes() == labels_before
    assert all(path.read_bytes() == content for path, content in videos_before.items())


def test_failed_extraction_does_not_mutate(existing,tmp_path,monkeypatch):
    root,weights=existing
    target=reextract.list_targets(root,person='CHANH',label=1,mode='missing')[:1]
    original_meta=target[0]['metadata'].read_bytes()
    before=(root/'labels.csv').read_bytes()
    def broken(*args,**kwargs):
        raise RuntimeError('simulated crash')
    monkeypatch.setattr(reextract,'extract_video',broken)
    result=reextract.reextract_targets(root,target,weights,backup_root=tmp_path/'backups')
    assert result['failed']==1
    assert not target[0]['raw'].exists()
    assert target[0]['metadata'].read_bytes()==original_meta
    assert (root/'labels.csv').read_bytes()==before
    assert not list(root.rglob('*.writing.txt'))


def test_subject_split_and_time_aligned_within_person(existing,tmp_path):
    root,_=existing
    before=(root/'labels.csv').read_bytes()
    mappings={'CHANH':'train','DGD':'test'}
    plan=make_split_plan(root,mode='subject',assignments=mappings)
    assert plan['counts']=={'train':4,'test':4}
    assert all(x['end_frame_exclusive']==100 for x in plan['entries'])
    output=export_plan(plan,tmp_path/'outputs')
    assert (output/'train.csv').exists() and (output/'test.csv').exists()
    with pytest.raises(ValueError):export_plan(plan,root/'splits')
    within=make_split_plan(root,mode='within',train_pct=70,val_pct=15)
    assert within['counts']=={'train':8,'val':8,'test':8}
    # Each camera in one take uses a consistent time partition.
    example=[x for x in within['entries'] if x['take_id']==plan['entries'][0]['take_id']]
    for split in ('train','val','test'):
        group=[x for x in example if x['split']==split]
        assert len(group)==2
        assert group[0]['start_ns']==group[1]['start_ns']
        assert group[0]['end_ns']==group[1]['end_ns']
    assert (root/'labels.csv').read_bytes()==before


def test_missing_mode_refuses_mixed_model_checkpoint(existing,tmp_path,monkeypatch):
    root, _ = existing
    first = reextract.list_targets(root,mode='missing')[0]
    first['summary'].write_text(json.dumps({'model_path': 'C:\\old\\yolo11n-pose.pt'}))
    weights = tmp_path/'yolo11m-pose.pt'
    weights.write_bytes(b'FAKE M WEIGHTS')
    monkeypatch.setattr(reextract,'extract_video',fake_extract)
    with pytest.raises(ValueError, match='mix pose checkpoints'):
        reextract.reextract_targets(root, reextract.list_targets(root, mode='missing'),
            weights, backup_root=tmp_path/'backups')
    assert not first['raw'].exists()


def test_full_reextract_preserves_original_shared_metadata_backup(existing, tmp_path, monkeypatch):
    root, weights = existing
    monkeypatch.setattr(reextract, 'extract_video', fake_extract)
    full = reextract.list_targets(root, mode='all')
    assert len(full) == 8
    before_meta = {p.relative_to(root): p.read_bytes() for p in root.glob(
        'recordings/*/*/label_*/take_*/metadata.json')}
    labels = (root/'labels.csv').read_bytes()
    videos = {p.relative_to(root): p.read_bytes() for p in root.rglob('*_raw.mp4')}
    result = reextract.reextract_targets(root, full, weights, backup_root=tmp_path/'backups',
                                         overwrite=True)
    assert result['selected'] == 8 and result['succeeded'] == 8 and result['failed'] == 0
    assert (root/'labels.csv').read_bytes() == labels
    assert all((root/p).read_bytes() == data for p, data in videos.items())
    for relative, original in before_meta.items():
        backed_up = Path(result['backup'])/relative
        assert backed_up.read_bytes() == original, f'Shared camera metadata backup changed: {relative}'
