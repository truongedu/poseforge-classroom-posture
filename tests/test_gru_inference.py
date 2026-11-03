"""GRU checkpoint, causal sequence, reset and MLP back-compat tests."""
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from shared.gru_model import PostureGRU, causal_windows
from shared.gru_training import augment_windows, quota_anchors
from inference import pipeline
from inference.pipeline import PostureClassifier


CONTRACT = {
    'version':'pose13_xy_conf_mask_causal_v2',
    'flatten_dim':52, 'shape_per_frame':[13,4],
    'channels':['x_norm','y_norm','confidence','valid_mask'],
    'joints':pipeline.REQUIRED_JOINTS,
    'confidence_threshold':.35, 'ema_alpha':.25,
}


def checkpoint(folder, family='gru'):
    folder.mkdir(parents=True)
    (folder/'feature_contract.json').write_text(json.dumps(CONTRACT))
    if family == 'gru':
        model = PostureGRU(hidden_size=8,num_layers=1,dropout=.2)
        cfg = dict(hidden_size=8,num_layers=1,dropout=.2,sequence_length=4)
        name = 'gru_temporal'
    else:
        from shared.mlp_model import PostureMLP
        model=PostureMLP('mlp_small',dropout=.15)
        cfg=dict(dropout=.15)
        name='mlp_small'
    torch.save(dict(stage='final',model_family=family,model_name=name,
        feature_contract=CONTRACT,label_names=[f'c{i}' for i in range(6)],
        model_config=cfg,model_state_dict=model.state_dict()),folder/'model_final.pt')
    return folder


def good_xy():
    arr=np.zeros((13,4),np.float32)
    arr[:,3]=1
    arr[:,2]=.9
    return arr


def test_causal_windows_reset_and_left_pad():
    a=np.stack([good_xy() for _ in range(5)])
    a[2]=0  # invalid frame MUST break sequence
    w,valid=causal_windows(a,4)
    assert valid.tolist()==[True,True,False,True,True]
    assert not w[0,:-1].any() and not w[2].any()
    assert not w[3,:-1].any() and not w[4,:-2].any()
    assert np.array_equal(w[4,-1],a[4].reshape(52))
    assert np.array_equal(w[4,-2],a[3].reshape(52))


def test_gru_classifier_and_mlp_backward_compatibility(tmp_path):
    gru=PostureClassifier(checkpoint(tmp_path/'gru'),backend='pt',device='cpu')
    assert gru.model_family=='gru' and gru.sequence_length==4
    probs=gru.predict(np.zeros((4,52),np.float32))
    assert probs.shape==(6,) and np.isclose(probs.sum(),1)
    with pytest.raises(ValueError,match='causal'):
        gru.predict(np.zeros(52,np.float32))
    mlp=PostureClassifier(checkpoint(tmp_path/'mlp','mlp'),backend='pt',device='cpu')
    assert mlp.model_family=='mlp' and mlp.sequence_length==1
    assert np.isclose(mlp.predict(good_xy()).sum(),1)


def test_pipeline_sequence_per_instance_and_loss(tmp_path,monkeypatch):
    class FakePose:
        def __init__(self,*args,**kwargs):self.frame=0
        def track_frame(self,frame):
            self.frame += 1
            if self.frame in (3,7): return []
            p=np.ones((17,3),np.float32)
            p[:,:2]=10.
            return [dict(bbox=[5,5,30,30],keypoints=p,track_id=1 if self.frame<6 else 2)]
        def reset_tracking(self): pass
        def close(self): pass
    monkeypatch.setattr(pipeline,'PoseRunner',FakePose)
    cls=PostureClassifier(checkpoint(tmp_path/'gru'),backend='pt',device='cpu')
    captured=[]
    old=cls.predict
    def predict(f):captured.append(f.copy());return old(f)
    cls.predict=predict
    p=pipeline.PosePipeline('fake.pt',cls,overlay=False)
    try:
        frame=np.zeros((80,80,3),np.uint8)
        p.process(frame);p.process(frame)
        assert not captured[0][:-1].any()
        assert np.any(captured[1][-2])
        p.process(frame) # NO POSE
        p.process(frame) # start window again
        assert not captured[-1][:-1].any()
        p.reset();p.process(frame)
        assert not captured[-1][:-1].any()
    finally:p.close()


def test_augment_coherent_across_time_and_absent_joints():
    v=np.zeros((1,5,13,4),np.float32)
    v[:,:,:,3]=1
    v[:,:,:,0]=.7
    v[:,:,7,:]=0 # absent joint preserved
    x=torch.from_numpy(v.reshape(1,5,52))
    out=augment_windows(x,torch.Generator().manual_seed(302),flip_prob=1.,jitter_std=0).reshape(1,5,13,4)
    assert torch.all(out[:,:,:,0][out[:,:,:,3] > 0]==-.7)
    assert not out[:,:,8,:].any() # left/right swap


def test_quota_caps_overlapping_windows_per_source_camera():
    fake=[dict(source='s1',camera='cam01',valid=np.ones(100,dtype=bool)),
          dict(source='s1',camera='cam01',valid=np.ones(70,dtype=bool)),
          dict(source='s1',camera='cam02',valid=np.ones(100,dtype=bool))]
    a=quota_anchors(fake,1,302,48)
    assert len(a)==96
    assert a==quota_anchors(fake,1,302,48)
