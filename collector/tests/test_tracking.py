"""Official ByteTrack association with deterministic detections + real YOLO11.

FakeYOLO substitutes detector output only. All ID assignment/association uses
Ultralytics' unmodified BYTETracker and its actual postprocess callback.
"""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest

ultralytics=pytest.importorskip('ultralytics')
import torch
from ultralytics.engine.results import Results
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.track import on_predict_postprocess_end
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from posture_collector.pose import PoseRunner
from posture_collector.tracking import TrackSelection,LatestFrames,TrackingJournal,parse_track_choices
from posture_collector.process import extract_video,normalize_rows


def detection(x=20,y=20,confidence=.95):
    return [x,y,x+50,y+100,confidence,0]


def detection_keypoints(box,index):
    # Nonintegral detector values expose any accidental coordinate changes.
    return [[box[0]+j+.125,box[1]+j+.375,.875-j*.001] for j in range(17)]


class FakeYOLO:
    def __init__(self,path):
        self.task='pose';self.callbacks=[];self.predictor=None;self.calls=[]
        self.script=None;self.index=0

    def add_callback(self,event,callback):self.callbacks.append(callback)

    def track(self,source,**kwargs):
        assert kwargs['persist'] is True and kwargs['tracker']=='bytetrack.yaml'
        assert kwargs['conf']==.25 and kwargs['verbose'] is False
        self.calls.append(kwargs)
        if self.predictor is None:
            tracker=BYTETracker(IterableSimpleNamespace(**YAML.load(check_yaml('bytetrack.yaml'))))
            self.predictor=SimpleNamespace(trackers=[tracker],args=SimpleNamespace(mode='track',task='pose'),
                dataset=SimpleNamespace(mode='image'),vid_path=[None])
        boxes=self.script[self.index] if self.script and self.index<len(self.script) else [detection()]
        self.index+=1
        array=np.asarray(boxes,np.float32).reshape(-1,6)
        kps=np.asarray([detection_keypoints(box,i) for i,box in enumerate(boxes)],np.float32).reshape(-1,17,3)
        result=Results(source,path='frame.jpg',names={0:'person'},boxes=torch.from_numpy(array),keypoints=torch.from_numpy(kps))
        self.predictor.results=[result]
        for callback in self.callbacks:callback(self.predictor)
        on_predict_postprocess_end(self.predictor,persist=True)
        return self.predictor.results


@pytest.fixture
def runner_factory(tmp_path,monkeypatch):
    monkeypatch.setattr(ultralytics,'YOLO',FakeYOLO)
    model=tmp_path/'yolo11n-pose.pt';model.touch();runners=[]
    def make(script=None):
        runner=PoseRunner('yolo',model,device='cpu');runner.engine.script=script
        runners.append(runner);return runner
    yield make
    for runner in runners:runner.close()


FRAME=np.zeros((200,300,3),np.uint8)


def test_moving_person_and_exact_detector_coordinates(runner_factory):
    script=[[detection(x=20+i*2)] for i in range(10)]
    runner=runner_factory(script);ids=[]
    for boxes in script:
        person=runner.track_frame(FRAME)[0];ids.append(person['track_id'])
        assert person['bbox']==np.asarray(boxes[0][:4],np.float32).tolist()
        assert person['keypoints']==np.asarray(detection_keypoints(boxes[0],0),np.float32).tolist()
        assert person['keypoint_confidences']==[p[2] for p in person['keypoints']]
    assert ids==[1]*10


def test_multiple_people_and_per_camera_counter_isolation(runner_factory):
    a=runner_factory([[detection(),detection(170)]]*8)
    b=runner_factory([[detection(170)]]*8)
    assert [p['track_id'] for p in a.track_frame(FRAME)]==[1,2]
    assert [p['track_id'] for p in b.track_frame(FRAME)]==[1]
    assert a.engine is not b.engine and a.engine.predictor.trackers[0] is not b.engine.predictor.trackers[0]
    for _ in range(3):
        assert [p['track_id'] for p in a.track_frame(FRAME)]==[1,2]
        assert [p['track_id'] for p in b.track_frame(FRAME)]==[1]
    old_session=b.tracking_session; b.reset_tracking()
    assert b.tracking_session!=old_session
    assert not b.engine.predictor.trackers[0].tracked_stracks
    assert b.track_frame(FRAME)[0]['track_id']==1
    assert [p['track_id'] for p in a.track_frame(FRAME)]==[1,2]
    assert a.engine.predictor.trackers[0].frame_id==5


def test_unassigned_detections_are_retained_and_missing_never_switches(runner_factory):
    # An object first appearing after frame 1 needs confirmation. Ultralytics
    # normally hides that row; our pre-tracker snapshot must still export it.
    runner=runner_factory([[],[detection()], [detection()], [], [detection(170)], [detection(170)], [detection()]])
    assert runner.track_frame(FRAME)==[]
    first=runner.track_frame(FRAME);assert len(first)==1 and first[0]['track_id'] is None
    assert first[0]['tracking_state']=='unassigned'
    persons=runner.track_frame(FRAME);assert persons[0]['track_id']==1
    selection=TrackSelection();assert selection.observe(persons)['track_id']==1
    assert selection.observe(runner.track_frame(FRAME)) is None and selection.state=='missing'
    runner.track_frame(FRAME)
    other=runner.track_frame(FRAME);assert other[0]['track_id']==2
    assert selection.observe(other) is None and selection.track_id==1
    # ByteTrack can recover its own retained ID after a short gap.
    recovered=runner.track_frame(FRAME)
    assert selection.observe(recovered)['track_id']==1


def test_multiple_people_require_explicit_selection(runner_factory):
    people=runner_factory([[detection(),detection(170)]]).track_frame(FRAME)
    selection=TrackSelection();assert selection.observe(people) is None
    assert selection.state=='selection_required'
    assert selection.observe(people[:1]) is None  # do not choose a remaining bystander
    selection.select(2);assert selection.observe(people)['track_id']==2
    selection.select(None);assert selection.observe(people[:1]) is None


def test_unassigned_row_keeps_original_index_when_other_person_tracked(runner_factory):
    runner=runner_factory([[detection()],[detection(170),detection()],[detection(170),detection()]])
    assert runner.track_frame(FRAME)[0]['track_id']==1
    rows=runner.track_frame(FRAME)
    assert rows[0]['track_id'] is None and rows[1]['track_id']==1
    assert rows[0]['bbox'][0]==170 and rows[1]['bbox'][0]==20
    rows=runner.track_frame(FRAME)
    assert rows[0]['track_id']==2 and rows[1]['track_id']==1


def test_reset_of_other_camera_cannot_reuse_an_active_id(runner_factory):
    two=[detection(10),detection(110)]
    three=two+[detection(210)]
    a=runner_factory([two,three,three]);b=runner_factory([[detection()]]*3)
    assert [p['track_id'] for p in a.track_frame(FRAME)]==[1,2]
    b.track_frame(FRAME);b.reset_tracking();b.track_frame(FRAME)
    a.track_frame(FRAME)
    assert [p['track_id'] for p in a.track_frame(FRAME)]==[1,2,3]


def test_offline_v2_export_missing_states_and_legacy_normalization(tmp_path,runner_factory,monkeypatch):
    import posture_collector.process as processing
    script=[[detection(),detection(170)], [detection(),detection(170)], [], [detection()]]
    model=tmp_path/'yolo11n-pose.pt'
    # Use the real PoseRunner adapter with the deterministic official tracker.
    monkeypatch.setattr(processing,'PoseRunner',lambda *a,**kw:runner_factory(script))
    folder=tmp_path/'take_001';folder.mkdir();video=folder/'cam01_raw.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(300,200))
    for _ in script:writer.write(FRAME)
    writer.release();digest=hashlib.sha256(video.read_bytes()).hexdigest()
    with (folder/'cam01_timestamps.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['host_recv_monotonic_ns','host_recv_utc'])
        for i in range(4):writer.writerow([1_000_000_000+i*50_000_000,f'2026-10-27T00:00:00.{i*50:03d}Z'])
    result=extract_video(video,'yolo',model,selected_track_id=2)
    rows=[json.loads(line) for line in Path(result['raw_keypoints_txt']).read_text().splitlines()]
    assert len(rows)==4 and rows[0]['track_id']==2 and rows[0]['selected_track_id']==2
    assert len(rows[0]['persons'])==2 and rows[0]['camera_id']=='cam01'
    assert rows[0]['host_recv_monotonic_ns']==1_000_000_000
    assert rows[2]['status']=='selected_track_missing' and rows[2]['keypoints'] is None
    assert rows[3]['selected_track_id']==2 and rows[3]['track_id'] is None
    assert rows[3]['persons'][0]['track_id']==1
    norm=normalize_rows(result['raw_keypoints_txt'],'yolo')
    assert norm['keypoints'].shape==(4,13,4) and np.all(norm['keypoints'][2:]==0)
    assert hashlib.sha256(video.read_bytes()).hexdigest()==digest


def test_camera_generations_and_old_snapshot_api(tmp_path):
    from posture_collector.capture import CameraStream
    video=tmp_path/'loop.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(100,100))
    for _ in range(3):writer.write(np.zeros((100,100,3),np.uint8))
    writer.release()
    core=SimpleNamespace(ingest=lambda *a,**kw:None)
    camera=CameraStream(dict(id='cam01',url=str(video)),core);camera.start()
    try:
        deadline=time.monotonic()+5
        while not camera.ready():
            assert time.monotonic()<deadline;time.sleep(.02)
        assert len(camera.snapshot())==5
        first=camera.snapshot(include_generation=True);assert first[-1] is not None
        while camera.snapshot(include_generation=True)[-2]==first[-2]:
            assert time.monotonic()<deadline;time.sleep(.02)
        assert camera.connection_generation>=2
        replacement=CameraStream(dict(id='cam01',url=str(video)),core)
        assert replacement.stream_id!=camera.stream_id
    finally:camera.stop();camera.thread.join(2)


def test_bounded_fair_queue_and_reject_inflight_result(monkeypatch):
    import threading
    import posture_collector.ui as ui
    started=threading.Event();release=threading.Event();instances=[]
    class Runner:
        tracking_session='initial'
        def __init__(self,*a,**kw):self.calls=0;self.resets=0;instances.append(self)
        def track_frame(self,frame):
            self.calls+=1
            if self.calls==1:started.set();release.wait(3)
            return []
        def reset_tracking(self):self.resets+=1;self.tracking_session='reset'
        def close(self):pass
    monkeypatch.setattr(ui,'PoseRunner',Runner)
    worker=ui.PosePreview('yolo','test-only-stub')
    try:
        worker.submit('cam01',FRAME,1,'old');assert started.wait(2)
        worker.reset_camera('cam01','new')
        assert worker.get('cam01') is None
        worker.submit('cam01',FRAME,2,'new')
        for _ in range(50):worker.submit('cam02',FRAME,3,'camera2')
        assert worker.queue.qsize()==2
        release.set();deadline=time.monotonic()+5
        while worker.get('cam01') is None or worker.get('cam02') is None:
            assert time.monotonic()<deadline;time.sleep(.02)
        assert worker.get('cam01')['stream_token']=='new'
        assert worker.get('cam01')['timestamp_ns']==2
        assert len(instances)==2 and instances[0].resets==1
        assert worker.get('cam02')['stream_token']=='camera2'
    finally:release.set();worker.stop();worker.thread.join(3)


def test_live_journal_keeps_selection_per_camera(tmp_path):
    (tmp_path/'keypoints').mkdir()
    journal=TrackingJournal(tmp_path,dict(person_id='TQB',session_id='session_test',label=4))
    journal.put('cam01',dict(track_session='a',selected_track_id=1,selection_state='selected',persons=[]))
    journal.put('cam02',dict(track_session='b',selected_track_id=2,selection_state='missing',persons=[]))
    journal.finish()
    report=json.loads((tmp_path/'tracking_selection.json').read_text())
    assert report['person_id']=='TQB' and report['label']==4
    assert report['cameras']['cam01']['selected_track_id']==1
    assert report['cameras']['cam02']['selected_track_id']==2
    assert report['cameras']['cam02']['selection_state']=='missing'


def test_offline_choice_validation():
    assert parse_track_choices('cam01=1, cam02=4')=={'cam01':1,'cam02':4}
    for text in ('cam01=0','cam01=no','cam01=2 cam01=3','bad..=2'):
        with pytest.raises(ValueError):parse_track_choices(text)


@pytest.mark.parametrize('weights',['yolo11n-pose.pt','yolo11m-pose.pt'])
def test_real_yolo11_pose_bytetrack(weights):
    project=Path(__file__).resolve().parents[1];model=project/'models'/weights
    image=cv2.imread(str(Path(__file__).parent/'tracking_assets'/'bus.jpg'))
    if not model.exists() or image is None:pytest.skip('Download official local weights and test image to run real inference')
    with PoseRunner('yolo',model,device='cpu') as runner:
        reference=runner.engine.predict(image,conf=.25,classes=[0],verbose=False)[0]
        original_kp=reference.keypoints.data.cpu().numpy().tolist()
        first=runner.track_frame(image)
        assert len(first)>=2 and all(p['track_id'] is not None for p in first)
        assert [p['keypoints'] for p in first]==original_kp
        ids=[p['track_id'] for p in first]
        moved=cv2.warpAffine(image,np.float32([[1,0,2],[0,1,0]]),(image.shape[1],image.shape[0]))
        second=runner.track_frame(moved)
        assert set(ids).intersection(p['track_id'] for p in second)
        assert len({p['track_id'] for p in second})==len(second)
