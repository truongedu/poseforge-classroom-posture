"""Read-only replay and deliberate full-dataset re-extraction contracts."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from posture_collector.replay_data import take_catalog, read_pose_jsonl, draw_pose_overlay
from posture_collector.reextract import list_targets


def test_replay_catalog_and_overlay(tmp_path):
    # Synthetic layout matches the collector without depending on the laptop dataset.
    assert tmp_path is not None
    root = tmp_path/'dataset'
    folder = root/'recordings'/'TQB'/'session_a'/'label_0'/'take_001'
    folder.mkdir(parents=True)
    (folder/'metadata.json').write_text(json.dumps({'cameras':{'cam01':{'view':'front'}},'status':'ok'}))
    video=folder/'cam01_raw.mp4';video.write_bytes(b'video')
    raw=folder/'keypoints'/'cam01_yolo17_raw.txt';raw.parent.mkdir()
    pose=[[20.,20.,.8] for _ in range(17)]
    raw.write_text(json.dumps({'backend':'yolo','frame_index':0,'keypoints':pose,
                               'bbox':[10,10,40,40],'status':'ok'})+'\n'+
                   json.dumps({'backend':'yolo','frame_index':1,'keypoints':None,
                               'status':'not_detected'})+'\n')
    clip=take_catalog(root)
    assert len(clip)==1 and clip[0]['person']=='TQB'
    rows,message=read_pose_jsonl(raw)
    assert len(rows)==2 and rows[1]['status']=='not_detected'
    pixels=np.zeros((60,60,3),dtype=np.uint8)
    rendered=draw_pose_overlay(pixels,rows[0])
    assert np.any(rendered!=pixels) and not np.any(pixels)
    assert not np.any(draw_pose_overlay(pixels,rows[1]))
    raw.unlink()
    missing,warning=read_pose_jsonl(raw)
    assert missing=={} and 'Missing' in warning


def test_full_scope_requires_no_filters(tmp_path):
    root=tmp_path/'dataset'
    for pid in ('A','B'):
        folder=root/'recordings'/pid/'session_x'/'label_0'/'take_001'
        folder.mkdir(parents=True)
        (folder/'metadata.json').write_text(json.dumps({'cameras':{'cam01':{}}}))
        (folder/'cam01_raw.mp4').write_bytes(b'mp4')
    assert len(list_targets(root, mode='all'))==2
    with pytest.raises(ValueError,match='does not accept'):
        list_targets(root, mode='all', person='A')
    with pytest.raises(ValueError,match='requires Person'):
        list_targets(root, mode='selected')
