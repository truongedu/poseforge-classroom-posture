from __future__ import annotations
import csv
import json
import time
from pathlib import Path
import numpy as np
import cv2
import pytest
from posture_collector.storage import DataStore
from posture_collector.capture import CollectorCore
from posture_collector.process import normalize_rows, build_processed


def test_store_schema_and_person_safety(tmp_path):
    db=DataStore(tmp_path)
    assert [x['id'] for x in db.get_labels()['labels']]==list(range(6))
    with pytest.raises(ValueError):db.add_subject('../bad',True)
    with pytest.raises(ValueError):db.add_subject('TQB',False)
    db.add_subject('TQB',True,'consent_form_001')
    with pytest.raises(ValueError):db.add_subject('TQB',True)
    assert db.add_label('Cầm điện thoại','holding_phone')==6
    assert db.get_labels()['labels'][4]['id']==4
    p=db.create_take('TQB',4,'session_test')
    assert p.relative_to(tmp_path).parts==('recordings','TQB','session_test','label_4','take_001')
    p2=db.create_take('TQB',4,'session_test')
    assert p2.name=='take_002'


def test_recording_two_cameras_raw_and_timestamps(tmp_path,monkeypatch):
    db=DataStore(tmp_path);db.add_subject('DTT',True)
    core=CollectorCore(db,fps=20)
    settings=[dict(id='cam01',url='synthetic_source_1',view='front',enabled=True),
              dict(id='cam02',url='synthetic_source_2',view='oblique',enabled=True)]
    import posture_collector.capture as m
    t0=time.monotonic_ns()
    monkeypatch.setattr(m.time,'monotonic_ns',lambda:t0)
    folder=core.start('DTT','session_test',0,settings,require_all_ready=False)
    for frame in range(120):
        stamp=t0+frame*50_000_000
        a=np.full((96,128,3),(10,40,80),dtype=np.uint8)
        b=np.full((96,128,3),(120,20,10),dtype=np.uint8)
        core.ingest('cam01',a,stamp,'2030-01-01T00:00:00.000Z',frame)
        core.ingest('cam02',b,stamp+1_000_000,'2030-01-01T00:00:00.001Z',frame)
    monkeypatch.setattr(m.time,'monotonic_ns',lambda:t0+6_000_000_000)
    core.stop();core.finish_thread.join(timeout=20)
    assert not core.finalizing
    meta=json.loads((folder/'metadata.json').read_text())
    assert meta['status']=='ok',meta
    assert meta['record_command_duration_s']==6
    for cam,expect_bgr in [('cam01',(10,40,80)),('cam02',(120,20,10))]:
        r=meta['cameras'][cam];assert r['frames']==120,r
        assert r['status']=='ok'
        video=folder/f'{cam}_raw.mp4'
        assert video.exists() and video.stat().st_size>1000
        cap=cv2.VideoCapture(str(video));n=0;first=None
        while True:
            ok,img=cap.read()
            if not ok:break
            if n==0:first=img
            n+=1
        cap.release();assert n==120
        assert np.mean(np.abs(first.astype(float)-np.array(expect_bgr)))<12
        with (folder/f'{cam}_timestamps.csv').open() as f:
            times=list(csv.DictReader(f))
        assert len(times)==120
        assert [int(row['writer_frame_index']) for row in times]==list(range(120))
    assert len(list(csv.DictReader((tmp_path/'labels.csv').open(encoding='utf-8-sig'))))==2


def test_normalization_and_missing_frames(tmp_path):
    kpfile=tmp_path/'cam01_yolo17_raw.txt'
    pts=np.zeros((17,3),dtype=float)
    pts[0]=[30,20,.95];pts[5]=[20,50,.98];pts[6]=[40,50,.9]
    pts[7]=[13,70,.95];pts[8]=[47,70,.88]
    data=[]
    for i in range(5):
        kp=pts.tolist() if i!=3 else None
        row=dict(frame_index=i,relative_ms=i*50,width=100,height=100,
                 status='ok' if kp else 'not_detected',keypoints=kp)
        data.append(row)
    kpfile.write_text('\n'.join(json.dumps(r) for r in data)+'\n')
    norm=normalize_rows(kpfile,'yolo')
    assert norm['keypoints'].shape==(5,13,4)
    assert np.all(norm['keypoints'][3]==0)
    assert np.allclose(norm['keypoints'][0,5,:2],[-.5,0.],atol=.01)
    assert np.allclose(norm['keypoints'][0,6,:2],[.5,0.],atol=.01)
    assert norm['scales_px'][0] == 20.
