"""Fast, CPU-only checks; no Ultralytics models or private dataset required."""
from pathlib import Path
import sys
import json

import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_sustained_pose import build_plan, iter_images, compare_outputs, GPUMonitor, make_dir


def _fake_dataset(root):
    data=root/'dataset'
    for label in range(6):
        for camera in ('cam01','cam02'):
            folder=data/'recordings'/f'P{label:02d}'/'session_1'/f'label_{label}'/'take_001'
            folder.mkdir(parents=True,exist_ok=True)
            metadata=folder/'metadata.json'
            metadata.write_text(json.dumps({'person_id':f'P{label:02d}', 'label':label,
                'cameras':{'cam01':{}, 'cam02':{}}}))
            path=folder/f'{camera}_raw.mp4'
            w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,64))
            for i in range(14):
                w.write(np.full((64,64,3),i*14,np.uint8))
            w.release()
    return data


def test_balanced_plan_reproducible_contiguous(tmp_path):
    dataset=_fake_dataset(tmp_path)
    plan=build_plan(dataset,120,302)
    assert plan==build_plan(dataset,120,302)
    assert sum(r['frames'] for r in plan)==120
    assert len({(r['label'],r['camera']) for r in plan})==12
    assert all(r['frames']>0 and r['start']>=0 for r in plan)
    images=list(iter_images(dataset,plan))
    assert len(images)==120
    keys=[(r['clip'],idx) for r,idx,_,_ in images]
    assert len(set(keys))==120
    assert sum(first for _,_,_,first in images)==len(plan)


def test_curation_exclude_take_camera(tmp_path):
    dataset=_fake_dataset(tmp_path)
    (dataset.parent/'curation.json').write_text(json.dumps({'takes':{
        'P00/session_1/label_0/take_001':{'exclude_take':True},
        'P01/session_1/label_1/take_001':{'exclude_cameras':['cam02']}}}))
    plan=build_plan(dataset,30,302)
    assert all(r['label']!=0 for r in plan)
    assert all(not (r['label']==1 and r['camera']=='cam02') for r in plan)


def test_comparison_mlp_and_pose_disagreement():
    features=[0.0]*52
    a={'clip#0':{'present':True,'keypoints':[[1.0,2.0,.8]]*17,
                  'features':features,'label':2,'person':'X','valid_joints':9,
                  'mlp_class':2,'mlp_probs':[0,0,1,0,0,0], 'mlp_confidence':1.0}}
    b={'clip#0':dict(a['clip#0'],mlp_class=3,mlp_probs=[0,0,.1,.9,0,0],mlp_confidence=.9)}
    report, review=compare_outputs(a,b)
    assert report['mlp_compared_frames']==1
    assert report['mlp_class_disagree_frames']==1
    assert report['mlp_take_label_match_delta_pp']==-100.0
    assert len(review)==1


def test_versioning_and_optional_gpu_monitor(tmp_path):
    assert make_dir(tmp_path).name=='pose_sustained_v001'
    assert make_dir(tmp_path).name=='pose_sustained_v002'
    with GPUMonitor('cpu') as m:
        pass
    assert m.summary()['gpu_telemetry']=='unavailable'


def test_cli_requires_models_and_no_source_mutation(tmp_path):
    dataset=_fake_dataset(tmp_path)
    names=sorted(str(p.relative_to(dataset)) for p in dataset.rglob('*') if p.is_file())
    plan=build_plan(dataset,80,302)
    list(iter_images(dataset,plan))
    assert names==sorted(str(p.relative_to(dataset)) for p in dataset.rglob('*') if p.is_file())


def test_respects_trimmed_segments(tmp_path):
    dataset=_fake_dataset(tmp_path)
    folder=dataset/'recordings'/'P02'/'session_1'/'label_2'/'take_001'
    with (folder/'cam01_timestamps.csv').open('w') as f:
        f.write('writer_frame_index,host_recv_monotonic_ns\n')
        for i in range(14):
            f.write(f'{i},{1000+i*100}\n')
    (dataset.parent/'curation.json').write_text(json.dumps({'takes':{
        'P02/session_1/label_2/take_001': {
          'keep_ranges_ns':[[1200,1500]],'exclude_cameras':['cam02']}}}))
    from benchmark_sustained_pose import source_clips
    groups=source_clips(dataset,302)
    rows=groups[(2,'cam01')]
    assert len(rows)==1
    assert rows[0]['source_start']==2 and rows[0]['count']==3
    assert (2,'cam02') not in groups


def test_runner_smoke_no_ultralytics(tmp_path, monkeypatch):
    data=_fake_dataset(tmp_path)
    plan=build_plan(data,24,302)
    import benchmark_sustained_pose as benchmark

    class FakeRunner:
        def __init__(self,*args,**kwargs):
            self.engine=None
        def track_frame(self,frame):
            points=np.zeros((17,3),dtype=float)
            points[:,:2]=24
            points[:,2]=.9
            return [{'bbox':[1,1,62,62],'keypoints':points.tolist(),'track_id':None}]
        def reset_tracking(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(benchmark,'PoseRunner',FakeRunner)
    monkeypatch.setattr(benchmark,'ensure_gpu',lambda *a:None)
    dummy=tmp_path/'model.pt'
    dummy.write_bytes(b'x')
    row,results=benchmark.run_pass(dummy,data,plan,'cpu',warmup=1,model_classifier=None)
    assert row['frames']==24
    assert row['selected_poses']==24
    assert all(rec['present'] and rec['valid_joints']==13 for rec in results.values())
    assert len(results)==24
