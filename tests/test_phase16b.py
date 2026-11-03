"""CPU-only regression: file safety + model selection; no Ultralytics dependency."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'collector'))
from posture_collector.reextract import list_targets
from posture_collector.storage import DataStore, save_json
from benchmark_yolo_pose import samples_from_dataset, compare
from reextract_pose_candidate import new_candidate, copy_raw_without_old_pose


def test_clone_isolated_and_keeps_curation(tmp_path):
    src=tmp_path/'working'/'Real_Posture_Dataset'
    take=src/'recordings'/'DPHI'/'session_1'/'label_2'/'take_001'
    kp=take/'keypoints';kp.mkdir(parents=True)
    (take/'metadata.json').write_text(json.dumps({'person_id':'DPHI','label':2,'cameras':{'cam01':{}}}))
    (take/'cam01_raw.mp4').write_bytes(b'ORIGINAL')
    (take/'cam01_timestamps.csv').write_bytes(b'frames,timestamp\n')
    (kp/'cam01_yolo17_raw.txt').write_bytes(b'old')
    (kp/'cam01_yolo17_raw.summary.json').write_bytes(b'oldmeta')
    (kp/'cam01_mediapipe33_raw.txt').write_bytes(b'mp')
    (src/'labels.csv').write_bytes(b'clip_id\nDPHI\n')
    dest=new_candidate(tmp_path/'candidates','yolo11s-pose')
    clone=dest/'Real_Posture_Dataset'
    copy_raw_without_old_pose(src,clone)
    assert (clone/'labels.csv').read_bytes()==(src/'labels.csv').read_bytes()
    assert (clone/'recordings/DPHI/session_1/label_2/take_001/cam01_raw.mp4').read_bytes()==b'ORIGINAL'
    assert not (clone/'recordings/DPHI/session_1/label_2/take_001/keypoints/cam01_yolo17_raw.txt').exists()
    assert (clone/'recordings/DPHI/session_1/label_2/take_001/keypoints/cam01_mediapipe33_raw.txt').read_bytes()==b'mp'
    targets=list_targets(clone,mode='missing')
    assert len(targets)==1
    assert len(list_targets(src,mode='selected',person='DPHI'))==1
    assert new_candidate(tmp_path/'candidates','yolo11s-pose').name=='yolo11s-pose_v002'


def test_benchmark_sampling_and_comparison(tmp_path):
    src=tmp_path/'dataset'
    for label in range(6):
        take=src/'recordings'/'P01'/'session_1'/f'label_{label}'/'take_001'
        take.mkdir(parents=True)
        (take/'metadata.json').write_text(json.dumps({'person_id':'P01','label':label,'cameras':{'cam01':{}}}))
        video=take/'cam01_raw.mp4'
        w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,64))
        for i in range(9):w.write(np.full((64,64,3),i*25,np.uint8))
        w.release()
    rows=samples_from_dataset(src,18,302)
    assert rows and sum(len(r['frames']) for r in rows)<=18
    assert samples_from_dataset(src,18,302)[0]['clip']==rows[0]['clip']
    val={'a':{'present':True,'keypoints':[[1,1,.8]]*17,'features':[0]*52}}
    val['a']['features'][3]=1
    assert compare(val,val)['selected_disagreement_frames']==0
    assert compare(val,val)['matched_keypoint_mean_px']==0.0


def test_gui_model_discovery_includes_onnx():
    source=(ROOT/'collector/posture_collector/ui.py').read_text()
    assert "'-pose.onnx'" in source
    source=(ROOT/'collector/posture_collector/reextract.py').read_text()
    assert "'.onnx'" in source
