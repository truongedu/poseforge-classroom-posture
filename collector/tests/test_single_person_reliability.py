"""Regression for single-person selection without dependence on ByteTrack IDs."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from posture_collector.single_person import SinglePersonSelector
from posture_collector.tracking import TrackSelection
from posture_collector.process import extract_video
from posture_collector.pose_quality import quality_rows, aggregate, save_reports
from posture_collector.reextract import list_targets


def candidate(box, tid=None, conf=.8):
    return {'bbox':box, 'bbox_confidence':conf, 'track_id':tid,
            'keypoints':[[float(i+15),float(i+40),.8] for i in range(17)],
            'keypoints_17':[[float(i+15),float(i+40)] for i in range(17)],
            'keypoint_confidences':[.8]*17}


def test_duplicate_boxes_and_id_changes_are_causal():
    detector=SinglePersonSelector()
    full=candidate([100,90,260,520],tid=1)
    upper=candidate([102,160,256,330],tid=2)
    selected,reason=detector.select([full,upper])
    assert selected is full and reason=='duplicate_boxes'
    # The same person is unassigned on the next frame; never latch the old ID.
    unassigned=candidate([102,92,262,522],tid=None)
    selected,reason=detector.select([unassigned])
    assert selected is unassigned and reason=='single_detection'
    # Distinct spurious candidate: choose only if unique close temporal match.
    same=candidate([104,95,264,525],tid=43)
    separate=candidate([400,200,550,540],tid=6)
    assert detector.select([same,separate]) == (same,'temporal_reacquire')


def test_ambiguous_no_hallucination_and_recovery():
    detector=SinglePersonSelector(max_gap=2)
    a=candidate([10,30,110,420],tid=1)
    b=candidate([370,30,475,420],tid=2)
    assert detector.select([a,b])[0] is None
    assert detector.select([])==(None,'not_detected')
    assert detector.select([a])[0] is a
    assert detector.select([a,b])[0] is a


def test_live_selection_auto_recovers_duplicate_and_manual_remains_explicit():
    live=TrackSelection(single_person=True)
    a=candidate([10,10,160,420],tid=2)
    b=candidate([11,30,156,200],tid=9)
    assert live.observe([a,b]) is a and live.auto
    assert live.observe([candidate([10,10,160,420],tid=None)]) is not None
    live.select(9)
    assert live.observe([a,b]) is b


def test_offline_raw_uses_same_selector_and_preserves_all_candidates(tmp_path,monkeypatch):
    from posture_collector import process
    rows=[
        [candidate([10,10,160,420],tid=1), candidate([11,30,155,200],tid=2)],
        [candidate([10,10,160,420],tid=None)],
        [],
        [candidate([10,10,160,420],tid=17)],
    ]
    class FakeRunner:
        def __init__(self,*args,**kwargs):
            self.tracking_session='test';self.frame=0;self.last_tracks=[]
        def __enter__(self): return self
        def __exit__(self,*args):pass
        def detect(self,image,ms):
            self.last_tracks=rows[self.frame];self.frame+=1
            return [p['keypoints'] for p in self.last_tracks]
    monkeypatch.setattr(process,'PoseRunner',FakeRunner)
    take=tmp_path/'take_001';take.mkdir()
    video=take/'cam01_raw.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(160,120))
    for _ in rows:writer.write(np.zeros((120,160,3),np.uint8))
    writer.release()
    report=extract_video(video,'yolo','unused.pt',output_dir=tmp_path/'staged')
    raw=Path(report['raw_keypoints_txt'])
    out=[json.loads(line) for line in raw.read_text().splitlines()]
    assert [row['status'] for row in out]==['ok','ok','not_detected','ok']
    assert [row['track_id'] for row in out]==[1,None,None,17]
    assert len(out[0]['persons'])==2
    assert report['selected_frames']==3
    assert report['selection_policy']=='single_person_causal_v1'
    assert out[1]['keypoints'] is not None


def test_quality_scanner_and_zero_scope_are_readonly(tmp_path):
    root=tmp_path/'dataset';take=root/'recordings'/'TDUY'/'session_x'/'label_5'/'take_001'
    keypoints=take/'keypoints';keypoints.mkdir(parents=True)
    (take/'metadata.json').write_text(json.dumps({'cameras':{'cam01':{},'cam02':{}}}))
    for cam in ('cam01','cam02'):
        (take/f'{cam}_raw.mp4').write_bytes(b'mp4')
    (keypoints/'cam01_yolo17_raw.txt').write_text('\n'.join(json.dumps({
        'frame_index':i,'status':'ambiguous_multiple_people','keypoints':None,
        'persons':[candidate([10,10,160,420]),candidate([10,30,158,190])]
    }) for i in range(3))+'\n')
    (keypoints/'cam02_yolo17_raw.txt').write_text(json.dumps({
        'frame_index':0,'status':'ok','keypoints':candidate([10,10,160,420])['keypoints'],
        'persons':[candidate([10,10,160,420])]})+'\n')
    targets=list_targets(root,mode='zero')
    assert len(targets)==1 and targets[0]['camera']=='cam01'
    rows=quality_rows(root)
    a=next(row for row in rows if row['camera']=='cam01')
    assert a['selected']==0 and a['projected']==3 and a['recoverable']==3
    result=save_reports(rows,tmp_path/'report')
    assert result['total']['frames']==4
    assert (tmp_path/'report'/'person_label_camera.csv').is_file()
    assert (keypoints/'cam01_yolo17_raw.txt').read_text().count('ambiguous_multiple_people')==3


def test_overwrite_blocks_quality_regression(tmp_path, monkeypatch):
    from posture_collector import reextract
    root=tmp_path/'dataset'
    take=root/'recordings'/'A'/'session_1'/'label_0'/'take_001'
    kp_dir=take/'keypoints';kp_dir.mkdir(parents=True)
    (take/'metadata.json').write_text(json.dumps({'cameras':{'cam01':{}}}))
    (take/'cam01_raw.mp4').write_bytes(b'ORIGINAL VIDEO')
    raw=kp_dir/'cam01_yolo17_raw.txt'
    before=''.join(json.dumps({'backend':'yolo','frame_index':i,'status':'ok',
                     'keypoints':candidate([10,10,110,420])['keypoints']})+'\n' for i in range(2))
    raw.write_text(before)
    model=tmp_path/'yolo11n-pose.pt';model.write_bytes(b'FAKE MODEL')
    def fail_pose(video,backend,model,progress=None,*,device='auto',output_dir=None):
        staged=Path(output_dir)/'cam01_yolo17_raw.txt'
        staged.write_text(''.join(json.dumps({'backend':'yolo','frame_index':i,
                         'status':'not_detected','keypoints':None})+'\n' for i in range(2)))
        staged.with_suffix('.summary.json').write_text(json.dumps({'backend':'yolo','frames':2}))
        return {'raw_keypoints_txt':str(staged),'frames':2,'selected_frames':0,
                'multiple_people':0,'extracted_at':'2026-10-28T00:00:00Z'}
    monkeypatch.setattr(reextract,'extract_video',fail_pose)
    targets=reextract.list_targets(root,person='A',mode='selected')
    result=reextract.reextract_targets(root,targets,model,backup_root=tmp_path/'backup',overwrite=True)
    assert result['failed']==1
    assert 'Quality regression' in result['results'][0]['message']
    assert raw.read_text()==before
    assert (take/'cam01_raw.mp4').read_bytes()==b'ORIGINAL VIDEO'
