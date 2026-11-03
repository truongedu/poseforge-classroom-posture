"""Phase 3.3 worker/ROI/metrics regression; no GPU, camera or Qt required."""
from __future__ import annotations

import time

import cv2
import numpy as np

from inference import pipeline
from inference.ui_runtime import (InferenceSession, LiveFPS, RollingFPS, Settings,
                                  Source, roi_contains, roi_filter)


def frame():
    return np.zeros((80, 120, 3), dtype=np.uint8)


def person(x, track=1):
    kp = [[float(x), 35.0+i, .95] for i in range(17)]
    return {'bbox': [x-10, 15, x+10, 70], 'keypoints': kp, 'track_id': track}


class FakePose:
    frames = 0
    instances = []

    def __init__(self, *args, **kwargs):
        self.resets = 0
        self.last_input = None
        self.closed = False
        FakePose.instances.append(self)

    def track_frame(self, data):
        self.last_input = data
        return [person(26), person(90,2)]

    def reset_tracking(self):
        self.resets += 1

    def close(self):
        self.closed = True


class FakeClassifier:
    labels = [f'label_{i}' for i in range(6)]
    conf_thresh = .35
    ema_alpha = .25
    backend = 'stub'

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        return np.array([.98,.004,.004,.004,.004,.004],np.float32)


def test_roi_filters_people_after_detection_not_before(monkeypatch):
    monkeypatch.setattr(pipeline,'PoseRunner',FakePose)
    p = pipeline.PosePipeline('fake.pt',FakeClassifier(),overlay=False)
    left_roi = [(0,0),(.45,0),(.45,1),(0,1)]
    try:
        original = frame()
        info, output = p.process(original,detection_filter=lambda people, shape: roi_filter(people,left_roi,shape))
        assert output is original
        assert p.pose.last_input is original
        assert info['track_id'] == 1
        assert info['class_id'] == 0
        assert len(roi_filter([person(26),person(90)], left_roi, original.shape)) == 1
        outside_roi = [(.48,0),(1,0),(1,1),(.48,1)]
        p.reset()
        info, _ = p.process(original,detection_filter=lambda people, shape: roi_filter(people,outside_roi,shape))
        assert info['track_id'] == 2
        assert roi_contains(left_roi,person(26),120,80)
        assert not roi_contains(left_roi,person(90),120,80)
    finally:
        p.close()


def test_roi_ambiguous_and_empty_abstain(monkeypatch):
    monkeypatch.setattr(pipeline,'PoseRunner',FakePose)
    cls=FakeClassifier()
    p=pipeline.PosePipeline('fake.pt',cls,overlay=False)
    try:
        info,_=p.process(frame())
        assert info['probs'] is None  # two separate people, no prior identity
        info,_=p.process(frame(),detection_filter=lambda *_: [])
        assert info['class_id'] is None
        assert cls.calls == 0
        assert not roi_contains([(0,0),(.1,0)],person(26),120,80)
    finally:
        p.close()


def test_fps_four_metrics_are_not_conflated():
    stage=RollingFPS()
    stage.add(5.,10.)
    stage.add(10.,11.)
    assert abs(stage.fps(11.)-133.3333333)<.01
    assert stage.fps(20.) is None
    live=LiveFPS()
    assert live.get(10.) is None
    for t in (10.,10.5,11.):
        live.tick(t)
    assert live.get(11.)==2.0
    assert live.get(15.) is None
    Settings().validate()
    for invalid in (Settings(stable_frames=0),Settings(render_fps=0),Settings(behavior='unknown')):
        import pytest
        with pytest.raises(ValueError):
            invalid.validate()


def test_image_testcase_session_clean_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline,'PoseRunner',FakePose)
    path=tmp_path/'test.jpg'
    cv2.imwrite(str(path),frame())
    events=[]
    obj=InferenceSession([Source('image',str(path))], 'fake.pt','fake_model',
                         emit=lambda *parts:events.append(parts),
                         classifier_factory=FakeClassifier)
    obj.start(testcase=True)
    assert obj.close(timeout=5)  # stop can race with quick processing; see below
    # Run independently of cancellation to guarantee completed frame.
    events.clear()
    obj=InferenceSession([Source('image',str(path))], 'fake.pt','fake_model',
                         emit=lambda *parts:events.append(parts),
                         classifier_factory=FakeClassifier)
    obj.start(testcase=True)
    obj.thread.join(5.)
    assert not obj.thread.is_alive()
    frames=[e for e in events if e[0]=='frame']
    assert len(frames)==1
    image,info,m=frames[0][2]
    assert image.shape == frame().shape
    assert info['class_id'] is None  # multiple people => abstain
    assert m['pipeline'] is not None and m['yolo'] is None and m['mlp'] is None
    assert m['live'] is None
    assert events[-1]==('state',-1,'READY')
    assert all(p.closed for p in FakePose.instances)


def test_two_sources_separate_pose_state_and_roi_updates(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'PoseRunner',FakePose)
    path=tmp_path/'test.jpg'
    cv2.imwrite(str(path),frame())
    left=[(0,0),(.45,0),(.45,1),(0,1)]
    right=[(.52,0),(1,0),(1,1),(.52,1)]
    events=[]
    obj=InferenceSession([Source('image',str(path)),Source('image',str(path))],
                         'fake.pt','fake_model',rois={0:left,1:right},
                         classifier_factory=FakeClassifier,emit=lambda *parts:events.append(parts))
    obj.start(testcase=True)
    obj.thread.join(5.)
    assert not obj.thread.is_alive()
    frames={i: payload for kind,i,payload in events if kind=='frame'}
    assert frames[0][1]['track_id']==1
    assert frames[1][1]['track_id']==2
    assert obj._states[0].pipeline is not obj._states[1].pipeline
    assert obj._states[0].pipeline.normalizer is not obj._states[1].pipeline.normalizer
    obj.update_roi(0,[])  # accepted, does not mutate YOLO input
    obj.update_roi(0,[(.1,.1)])  # incomplete editor polygon ignored
    assert obj._pending_rois[0] == ()


def test_testcase_video_is_continuous_until_eof(tmp_path, monkeypatch):
    """A testcase is a continuous stream, not one inference/frame."""
    monkeypatch.setattr(pipeline, 'PoseRunner', FakePose)
    path = tmp_path / 'case.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 15, (120, 80))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(frame())
    writer.release()
    events = []
    session = InferenceSession([Source('video', str(path))], 'fake.pt', 'fake_model',
                               classifier_factory=FakeClassifier,
                               emit=lambda *parts: events.append(parts))
    session.start(testcase=True)
    session.thread.join(5)
    assert not session.thread.is_alive()
    assert session._states[0].processed == 8
    assert events[-1] == ('state', -1, 'READY')
    assert ('case', -1, ('complete', 1)) in events


def test_live_testcase_boundary_pauses_resets_and_resumes(monkeypatch):
    import threading
    monkeypatch.setattr(pipeline, 'PoseRunner', FakePose)

    class ManualReader:
        instance = None

        def __init__(self, *_args, **_kwargs):
            self.seq = 0
            self.lock = threading.Lock()
            self.latest = None
            self.closed = False
            ManualReader.instance = self

        def publish(self):
            with self.lock:
                self.seq += 1
                self.latest = (1, self.seq, time.monotonic(), frame())

        def snapshot(self):
            with self.lock:
                return self.latest

        def close(self):
            self.closed = True

    events = []
    lock = threading.Lock()

    def emit(*event):
        with lock:
            events.append(event)

    def wait_for(predicate, timeout=3.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with lock:
                if predicate(events):
                    return
            time.sleep(.005)
        raise AssertionError(f'Condition not reached: {events!r}')

    session = InferenceSession([Source('camera', '0')], 'fake.pt', 'fake_model',
                               classifier_factory=FakeClassifier, reader_factory=ManualReader,
                               emit=emit)
    try:
        session.start(testcase=True)
        wait_for(lambda e: ('case', -1, ('running', 1)) in e)
        reader = ManualReader.instance
        for target in (1, 2, 3):
            reader.publish()
            wait_for(lambda e: session._states[0].processed >= target)
        assert session._states[0].processed == 3
        session.finish_case()
        wait_for(lambda e: ('case', -1, ('ready', 1)) in e)
        assert session._states[0].case_processed == 0
        assert session._states[0].pipeline.missing == 0
        # New frames arrive, but none are consumed while waiting for Next Case.
        reader.publish()
        time.sleep(.05)
        assert session._states[0].processed == 3
        assert session.next_case()
        wait_for(lambda e: ('case', -1, ('running', 2)) in e)
        reader.publish()
        wait_for(lambda e: session._states[0].processed >= 4)
        assert session._states[0].case_processed == 1
        assert session._states[0].pipeline.missing == 1  # only current frame since reset
    finally:
        assert session.close(timeout=3.)
    assert reader.closed


def test_guarded_uses_stable_class_not_raw_label():
    from inference.ui_runtime import display_decision
    labels = ['good', 'bad']
    assert display_decision({'probs': [.1, .9], 'class_id': 1, 'label': 'bad',
                             'stable_id': None}, 'guarded', labels) == 'WAITING / NO DECISION'
    assert display_decision({'probs': [.8, .2], 'class_id': 0, 'label': 'good',
                             'stable_id': 1}, 'guarded', labels) == 'bad'
    assert display_decision({'probs': [.8, .2], 'class_id': 0, 'label': 'good',
                             'stable_id': 1}, 'fast', labels) == 'good'
    assert display_decision({'probs': None}, 'guarded', labels) == 'NO POSE'


def test_overlay_work_only_on_preview_frames(monkeypatch):
    monkeypatch.setattr(pipeline, 'PoseRunner', FakePose)
    p = pipeline.PosePipeline('fake.pt', FakeClassifier(), overlay=True)
    try:
        original = frame()
        _, no_preview = p.process(original, render_preview=False)
        _, preview = p.process(original, render_preview=True)
        assert no_preview is original
        assert preview is not original
        # Original command-line API keeps its prior drawing behavior.
        _, prior_default = p.process(original)
        assert prior_default is not original
    finally:
        p.close()


def test_four_source_session_separate_temporal_states(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'PoseRunner', FakePose)
    path = tmp_path/'image.jpg'
    cv2.imwrite(str(path), frame())
    events = []
    session = InferenceSession([Source('image',str(path)) for _ in range(4)],
                               'fake_pose.pt','fake_model',classifier_factory=FakeClassifier,
                               emit=lambda *event: events.append(event))
    session.start(testcase=True)
    session.thread.join(5)
    assert not session.thread.is_alive()
    assert len([e for e in events if e[0]=='frame']) == 4
    assert len({id(s.pipeline.pose) for s in session._states}) == 4
    assert len({id(s.pipeline.normalizer) for s in session._states}) == 4


def test_preview_is_raw_by_default_and_pose_only_on_selected_stream(tmp_path):
    from inference.ui_runtime import PreviewSession, draw_pose_preview
    import threading
    src = tmp_path/'a.png'
    cv2.imwrite(str(src), frame())
    ev = []
    class FakePreviewPose:
        instances = []
        def __init__(self, *args, **kwargs):
            self.calls = 0
            self.resets = 0
            self.closed = False
            self.__class__.instances.append(self)
        def track_frame(self, frame):
            self.calls += 1
            return [person(26)]
        def reset_tracking(self):
            self.resets += 1
        def close(self):
            self.closed = True
    session = PreviewSession([Source('image',str(src)),Source('image',str(src))],
                             'pose.pt',emit=lambda *e: ev.append(e),
                             pose_factory=FakePreviewPose, fps=30)
    session.start()
    import time
    deadline = time.monotonic()+2
    while time.monotonic()<deadline and len([e for e in ev if e[0]=='preview_frame'])<4:
        time.sleep(.01)
    assert not FakePreviewPose.instances  # no detector/model loaded for raw preview
    session.set_pose_index(1)
    deadline = time.monotonic()+2
    while time.monotonic()<deadline and not FakePreviewPose.instances:
        time.sleep(.01)
    assert len(FakePreviewPose.instances)==1
    deadline = time.monotonic()+2
    while time.monotonic()<deadline and FakePreviewPose.instances[0].calls<1:
        time.sleep(.01)
    assert FakePreviewPose.instances[0].calls > 0
    assert session.close(timeout=3)
    assert FakePreviewPose.instances[0].closed
    assert draw_pose_preview(frame(),[person(26)]).shape == frame().shape


def test_production_runtime_default_uses_cuda_classifier():
    session = InferenceSession([Source('camera', '0')], 'pose.pt', 'mlp_wide/v002')
    assert session.device == '0'
    assert session.cls_device == 'cuda:0'
