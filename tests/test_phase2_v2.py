"""Phase 2 V2 pure contract tests. Run PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_phase2_v2.py"""
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import torch

from shared.pose_training import (FLIP13, augment_pose13_torch, dataset_output_root, refresh_review_zip, frozen_split,
                             group_sample_epoch, read_clean_lineage, source_group_id,
                             usable_mask)


def test_source_group_key_avoids_collisions():
    assert source_group_id('P', 'S', 1, 'take_001') != source_group_id('P', 'S', 2, 'take_001')
    assert source_group_id('P', 'S', 1, 'take_001') != source_group_id('Q', 'S', 1, 'take_001')


def test_frozen_split_keeps_people_separate():
    persons = ['A', 'B', 'C', 'D']
    catalog = pd.DataFrame([dict(clip_id=f'{p}_{label}', person_id=p, label=label,
                                 pose17_frames=2, take_id=f'{p}_{label}', camera_id='cam01', session_id='s')
                            for p in persons for label in range(6)])
    split, info = frozen_split(catalog, ['B'], ['A'])
    assert info['train_people'] == ['C', 'D']
    assert info['validation_people'] == ['B'] and info['test_people'] == ['A']
    assert set(split[split.person_id == 'A'].split) == {'test'}
    assert set(split[split.person_id == 'B'].split) == {'val'}
    assert set(split[split.person_id == 'C'].split) == {'train'}


def test_frozen_split_cannot_use_missing_labels():
    catalog = pd.DataFrame([dict(clip_id=f'{p}_{label}', person_id=p, label=label,
                                 pose17_frames=2, take_id=f'{p}_{label}', camera_id='cam01', session_id='s')
                            for p in ['A','B','C','D'] for label in range(6) if not (p=='A' and label==3)])
    with pytest.raises(ValueError, match='lacks class'):
        frozen_split(catalog, ['B'], ['A'])


def clips_for_sampling():
    # Two segmented clips are one source take on cam01, plus separate cam02.
    return [dict(start=0,end=60,label=0,source_group_id='P/S/label_0/take_001',camera='cam01'),
            dict(start=60,end=120,label=0,source_group_id='P/S/label_0/take_001',camera='cam01'),
            dict(start=120,end=200,label=0,source_group_id='P/S/label_0/take_001',camera='cam02'),
            dict(start=200,end=260,label=1,source_group_id='P/S/label_1/take_001',camera='cam01')]


def test_sample_caps_original_source_take_camera():
    usable=np.ones(260,dtype=bool)
    clips=clips_for_sampling()
    ids=group_sample_epoch(clips,usable,48,seed=302,epoch=1)
    assert len(ids)==144
    assert len(set(ids))==144
    assert (ids<120).sum()==48
    assert ((ids>=120)&(ids<200)).sum()==48
    assert (ids>=200).sum()==48
    assert not np.array_equal(ids,group_sample_epoch(clips,usable,48,seed=302,epoch=2))
    assert np.array_equal(ids,group_sample_epoch(clips,usable,48,seed=302,epoch=1))


def test_sample_never_selects_unusable_or_exceeds_cap_with_balance():
    clips=clips_for_sampling()
    usable=np.ones(260,dtype=bool);usable[:40]=False;usable[200:220]=False
    for balance in ('none','sqrt_inverse','inverse'):
        ids=group_sample_epoch(clips,usable,48,302,1,balance)
        assert usable[ids].all() and len(ids)==len(set(ids))
        assert (ids<120).sum()<=48 and ((ids>=120)&(ids<200)).sum()<=48
        assert (ids>=200).sum()<=48


def test_sample_short_clips_are_preserved():
    clips=[dict(start=0,end=3,label=0,source_group_id='a',camera='cam01')]
    assert set(group_sample_epoch(clips,np.ones(3,bool),48))=={0,1,2}


def test_usable_mask_rejects_missing_pose():
    x=np.zeros((4,13,4),np.float32)
    x[1,:3,3]=1;x[2,:4,3]=1;x[3,:,3]=1
    assert usable_mask(x).tolist()==[False,False,True,True]


def test_aug_preserves_missing_joint_semantics_and_nonzero_eligibility():
    a=torch.zeros((8,13,4))
    a[:,:,:2]=.7;a[:,:,2]=.8;a[:,:,3]=1
    a[:,3,:]=0
    cfg=dict(flip_prob=.5,jitter_prob=1.,jitter_sigma=.03,
             keypoint_dropout_prob=1.,occlusion_prob=1.,max_dropout_joints=2,
             confidence_perturb_prob=1.)
    b=augment_pose13_torch(a.reshape(-1,52),cfg,torch.Generator().manual_seed(302)).reshape(-1,13,4)
    assert torch.isfinite(b).all()
    assert (b[:,:,3].sum(1)>=10).all()
    assert (b[:,:,3].sum(1)<=12).all()
    assert (b[:,:,0:2][b[:,:,3]==0]==0).all()
    assert (b[:,:,2][b[:,:,3]==0]==0).all()
    assert torch.equal(b[:,:,3],b[:,:,3].round())


def test_flip_is_coherent_and_involutive():
    a=torch.zeros((1,13,4))
    a[0,:,0]=torch.arange(13).float();a[0,:,1]=42
    a[0,:,2]=.9;a[0,:,3]=1
    cfg=dict(flip_prob=1.,jitter_prob=0.,keypoint_dropout_prob=0.)
    gen=torch.Generator().manual_seed(1)
    b=augment_pose13_torch(a.reshape(-1,52),cfg,gen).reshape(-1,13,4)
    assert torch.equal(b[0,:,0], -a[0,list(FLIP13),0])
    assert torch.equal(b[0,:,1:], a[0,list(FLIP13),1:])
    c=augment_pose13_torch(b.reshape(-1,52),cfg,gen).reshape(-1,13,4)
    assert torch.equal(a,c)


def make_lineage_zip(tmp_path, overlap=False):
    catalog=pd.DataFrame([
      dict(clip_id=f'P_s_L0_take_001_seg{i:02d}_cam01',take_id=f'take_001_seg{i:02d}',
           person_id='P',session_id='s',camera_id='cam01',label=0,frames=3)
      for i in [1,2]])
    entries=[]
    for i,clip in enumerate(catalog.itertuples(index=False)):
        start=(2 if overlap else 3)*i
        entries.append(dict(clip_id=clip.clip_id,take_id=clip.take_id,source_take_id='take_001',
                            source_clip_id='P_s_L0_take_001_cam01',camera_id='cam01',segment=i+1,
                            source_frame_start=start,source_frame_end_exclusive=start+3,frames=3))
    path=tmp_path/'test.zip'
    with ZipFile(path,'w') as z:
        z.writestr('Real_Posture_Dataset/curation_manifest.json',json.dumps({'recordings':entries}))
    return path,catalog


def test_lineage_groups_segments_and_rejects_overlap(tmp_path):
    path,catalog=make_lineage_zip(tmp_path)
    frame,manifest=read_clean_lineage(path,catalog)
    assert frame.source_group_id.nunique()==1
    assert frame.source_frame_start.tolist()==[0,3]
    path,catalog=make_lineage_zip(tmp_path,overlap=True)
    with pytest.raises(ValueError,match='Overlapping'):
        read_clean_lineage(path,catalog)


def test_notebook_cells_compile():
    for name in ('01_raw_eda_split_v2.ipynb','02_process_eda_export_v2.ipynb','03_train_mlp_v2.ipynb'):
        nb=json.loads((Path(__file__).resolve().parent.parent/'notebooks'/name).read_text())
        for i,c in enumerate(nb['cells']):
            if c['cell_type']=='code':
                compile(''.join(c['source']),f'{name}:cell{i}','exec')


def test_single_review_zip_excludes_weights_features_and_old_runs(tmp_path):
    root = dataset_output_root(tmp_path, 'a' * 64)
    (root / 'raw' / 'charts').mkdir(parents=True)
    (root / 'raw' / 'charts' / 'chart.png').write_bytes(b'png')
    (root / 'processed').mkdir()
    (root / 'processed' / 'pose13_features.zip').write_bytes(b'big')
    run1 = root / 'training' / 'mlp_clean14_v001'
    run2 = root / 'training' / 'mlp_clean14_v002'
    run1.mkdir(parents=True); run2.mkdir(parents=True)
    (run1 / 'curve.png').write_bytes(b'old')
    (run2 / 'curve.png').write_bytes(b'new')
    (run2 / 'best.pt').write_bytes(b'weights')
    result=refresh_review_zip(root,run2)
    with ZipFile(result) as z:
        names=set(z.namelist())
        assert 'raw/charts/chart.png' in names
        assert 'training/mlp_clean14_v002/curve.png' in names
        assert 'training/mlp_clean14_v001/curve.png' not in names
        assert not any(n.endswith(('.pt','.zip','.npz')) for n in names)
    assert result == root / 'review.zip'


def test_notebook_03_has_full_sweep_and_curves():
    path=Path(__file__).resolve().parent.parent/'notebooks/03_train_mlp_v2.ipynb'
    cells=json.loads(path.read_text())['cells']
    config=''.join(cells[2]['source'])
    training=''.join(cells[10]['source'])
    ranking=''.join(cells[12]['source'])
    assert config.count("{'name':'c") >= 14
    assert 'CONFIRM_TOP_K=3' in config
    assert 'val_nll' in training and 'val_ece' in training
    assert 'learning_curves.png' in training
    assert 'seed_confirmation.csv' in ranking and 'confirmed_ranking.csv' in ranking


def test_notebook_02_keeps_no_duplicate_feature_directory():
    path=Path(__file__).resolve().parent.parent/'notebooks/02_process_eda_export_v2.ipynb'
    code='\n'.join(''.join(c['source']) for c in json.loads(path.read_text())['cells'] if c['cell_type']=='code')
    assert 'tempfile.mkdtemp' in code
    assert 'shutil.rmtree(STAGING_ROOT)' in code
    # Notebook downloaded from Colab can keep equivalent Python spacing.
    import re
    assert re.search(r"processed_zip\s*=\s*PROCESSED_REPORT\s*/\s*['\"]pose13_features\.zip['\"]", code)
