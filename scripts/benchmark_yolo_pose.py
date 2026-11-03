#!/usr/bin/env python3
"""Benchmark n/s × PT/ONNX using identical real frames and PoseForge tracking.

Reports *relative pose agreement*, not accuracy: the dataset has no manual
17-keypoint ground truth. No input dataset is modified.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'collector'))
sys.path.insert(0, str(ROOT))
from posture_collector.pose import PoseRunner
from posture_collector.single_person import SinglePersonSelector
from shared.pose_features import CausalPoseNormalizer

MODELS = ['yolo11n-pose.pt', 'yolo11n-pose.onnx',
          'yolo11s-pose.pt', 'yolo11s-pose.onnx']


def samples_from_dataset(dataset, limit, seed):
    """Choose videos across all six labels and both cameras; decode a few frames each."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for meta_file in sorted(dataset.glob('recordings/*/*/label_*/take_*/metadata.json')):
        folder = meta_file.parent
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
        for cam in meta.get('cameras', {}):
            path = folder / f'{cam}_raw.mp4'
            if path.is_file():
                groups[(int(meta['label']), cam)].append((path, str(meta.get('person_id', folder.parents[2].name))))
    if not groups:
        raise ValueError('No camera videos found')
    for value in groups.values():
        rng.shuffle(value)
    # Round-robin groups gives fair class/camera coverage, then different takes.
    pools = {k: iter(v) for k, v in sorted(groups.items())}
    chosen = []
    while len(chosen) < max(1, limit // 3) and pools:
        for k in list(pools):
            try:
                path, person = next(pools[k])
                chosen.append((path, person, k))
                if len(chosen) >= max(1, limit // 3):
                    break
            except StopIteration:
                del pools[k]
    samples = []
    for path, person, (label, cam) in chosen:
        cap = cv2.VideoCapture(str(path))
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or count < 1:
                continue
            middle = count // 2
            indices = sorted(set(min(count-1, max(0, middle+d)) for d in (-1, 0, 1)))
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, image = cap.read()
                if ok:
                    frames.append((idx,image))
            if frames:
                samples.append({'clip': str(path.relative_to(dataset)), 'person': person,
                                'label': label, 'camera': cam, 'frames': frames})
        finally:
            cap.release()
    if not samples:
        raise ValueError('No decodable frames in selected videos')
    print(f'[SAMPLE] {len(samples)} clips, {sum(len(c["frames"]) for c in samples)} frames, '
          f'{len({c["person"] for c in samples})} people', flush=True)
    return samples


def ensure_gpu(device, model_path):
    if str(device) not in ('0','cuda:0'):
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but PyTorch cannot access NVIDIA GPU')
    if model_path.suffix == '.onnx':
        import onnxruntime as ort
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('ONNX CUDAExecutionProvider absent. Fix onnxruntime-gpu/CUDA, '
                               'or explicitly benchmark all four models with --device cpu.')


def run_model(path, samples, device, warmup):
    ensure_gpu(device, path)
    # Use the SAME track_frame/selector/normalizer as offline extraction.
    runner = PoseRunner('yolo', str(path), video=True, device=device)
    latencies = []
    results = {}
    try:
        warmup_frame = samples[0]['frames'][0][1]
        for _ in range(warmup):
            runner.track_frame(warmup_frame)
        if path.suffix.lower() == '.onnx' and str(device) in ('0', 'cuda:0'):
            backend = getattr(getattr(runner.engine, 'predictor', None), 'model', None)
            session = getattr(backend, 'session', None)
            providers = session.get_providers() if session is not None else []
            if 'CUDAExecutionProvider' not in providers:
                raise RuntimeError(f'ONNX session did not confirm CUDAExecutionProvider: {providers}; '
                                   'refusing a misleading GPU benchmark')
        runner.reset_tracking()
        for clip in samples:
            runner.reset_tracking()
            selector = SinglePersonSelector()
            normalizer = CausalPoseNormalizer()
            for idx, frame in clip['frames']:
                if str(device) in ('0','cuda:0'):
                    import torch
                    torch.cuda.synchronize()
                start = time.perf_counter()
                people = runner.track_frame(frame)
                selected, _ = selector.select(people)
                kps = selected.get('keypoints') if selected is not None else None
                x, _, _ = normalizer.transform(kps, frame.shape[1], frame.shape[0], backend='yolo')
                if str(device) in ('0','cuda:0'):
                    torch.cuda.synchronize()
                latencies.append(1000*(time.perf_counter()-start))
                key = clip['clip']+'#'+str(idx)
                results[key] = {'person':clip['person'], 'label':clip['label'], 'camera':clip['camera'],
                    'present': kps is not None, 'keypoints':kps,
                    'valid_joints': int(x[:,3].sum()), 'features': x.reshape(-1).tolist()}
    finally:
        runner.close()
    times = sorted(latencies)
    mean_ms = statistics.mean(times)
    group_counts = defaultdict(lambda: {'frames':0,'selected':0})
    for row in results.values():
        key=f"L{row['label']}_{row['camera']}"
        group_counts[key]['frames'] += 1
        group_counts[key]['selected'] += bool(row['present'])
    summary = {'model':path.name, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
               'frames':len(times), 'fps':round(1000/mean_ms,2),
               'mean_ms':round(mean_ms,3),
               'p95_ms':round(times[min(len(times)-1,int(len(times)*.95))],3),
               'selected_poses':sum(v['present'] for v in results.values()),
               'valid_joints':sum(v['valid_joints'] for v in results.values()),
               'per_group':dict(group_counts)}
    return summary,results


def compare(reference, output):
    both = []
    feature_diffs = []
    mask_changes = 0
    selections_differ = 0
    for key, row in output.items():
        base = reference[key]
        selections_differ += base['present'] != row['present']
        if base['present'] and row['present']:
            a = np.asarray(base['keypoints'], dtype=np.float32)
            b = np.asarray(row['keypoints'], dtype=np.float32)
            good = np.minimum(a[:,2],b[:,2]) >= .35
            if np.any(good):
                both.extend(np.linalg.norm(a[good,:2]-b[good,:2],axis=1).tolist())
            af = np.asarray(base['features'], dtype=np.float32).reshape(13,4)
            bf = np.asarray(row['features'], dtype=np.float32).reshape(13,4)
            mask_changes += int(np.count_nonzero(af[:,3]!=bf[:,3]))
            shared = (af[:,3]>0)&(bf[:,3]>0)
            if np.any(shared):
                feature_diffs.extend(np.linalg.norm(af[shared,:2]-bf[shared,:2],axis=1).tolist())
    return {'selected_disagreement_frames':selections_differ,
            'matched_keypoint_mean_px':round(float(np.mean(both)),3) if both else None,
            'matched_normalized_xy_error':round(float(np.mean(feature_diffs)),5) if feature_diffs else None,
            'valid_mask_disagreements':mask_changes}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'artifacts/dataset_working/Real_Posture_Dataset')
    p.add_argument('--pretrained',type=Path,default=ROOT/'artifacts/pretrained')
    p.add_argument('--device',default='0',help='GPU device 0, or cpu')
    p.add_argument('--frames',type=int,default=96)
    p.add_argument('--seed',type=int,default=302)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/benchmarks')
    a=p.parse_args()
    if a.frames<12:
        p.error('--frames must be >=12 for varied sampling')
    paths=[a.pretrained/name for name in MODELS]
    for path in paths:
        if not path.is_file():p.error(f'Missing checkpoint {path}; prepare models first')
    samples=samples_from_dataset(a.data,a.frames,a.seed)
    a.output.mkdir(parents=True,exist_ok=True)
    n=1
    while True:
        dest=a.output/f'pose_n_vs_s_v{n:03d}'
        try:dest.mkdir();break
        except FileExistsError:n+=1
    summary=[];cache={}
    try:
        for path in paths:
            print(f'[BENCHMARK] {path.name}',flush=True)
            row,pred=run_model(path,samples,a.device,a.warmup)
            summary.append(row);cache[path.name]=pred
            print(f'  {row["fps"]} FPS | selected {row["selected_poses"]}/{row["frames"]}',flush=True)
        ref=cache[MODELS[0]]
        for row in summary:
            row.update(compare(ref,cache[row['model']]))
        (dest/'results.json').write_text(json.dumps({'device':a.device,'seed':a.seed,
            'sample_clips':[{'clip':c['clip'],'person':c['person'], 'label':c['label'],
                             'camera':c['camera'],'indices':[i for i,_ in c['frames']]}
                            for c in samples], 'models':summary},indent=2)+'\n')
        with (dest/'comparison.csv').open('w',newline='') as file:
            fields = [k for k in summary[0].keys() if k != 'per_group']
            writer=csv.DictWriter(file,fieldnames=fields)
            writer.writeheader();writer.writerows([{k:r[k] for k in fields} for r in summary])
        print(f'[OK] {dest}/comparison.csv\n[NOTE] Differences are relative to n.pt, '
              'NOT supervised pose accuracy. Review representative frames before selection.')
    except BaseException:
        (dest/'INCOMPLETE.txt').write_text('Benchmark failed. No result should be used.\n')
        raise

if __name__=='__main__':
    main()
