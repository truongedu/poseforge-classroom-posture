#!/usr/bin/env python3
"""Order-balanced sustained YOLO-Pose benchmark (PT, optionally ONNX).

Uses the real PoseForge PoseRunner/ByteTrack/selector/causal normalizer and
optionally measures downstream MLP predictions. Never changes source data.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'collector'))
sys.path.insert(0, str(ROOT))
from posture_collector.pose import PoseRunner
from posture_collector.single_person import SinglePersonSelector
from shared.pose_features import CausalPoseNormalizer, FEATURE_VERSION
from benchmark_yolo_pose import compare, ensure_gpu

MODEL_SETS = {
    'pt': ['yolo11n-pose.pt', 'yolo11s-pose.pt'],
    'all': ['yolo11n-pose.pt', 'yolo11n-pose.onnx',
            'yolo11s-pose.pt', 'yolo11s-pose.onnx'],
}


def _allowed_segments(timestamps: Path, ranges, total: int):
    """Continuous source frame runs matching [start_ns, end_ns), without editing video."""
    if ranges is None:
        return [(0, total)]
    with timestamps.open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    if len(records) != total:
        raise ValueError(f'Timestamp frame mismatch: {timestamps}')
    accepted = []
    for i, record in enumerate(records):
        if int(record['writer_frame_index']) != i:
            raise ValueError(f'Invalid timestamp frame index: {timestamps}')
        stamp = int(record['host_recv_monotonic_ns'])
        if any(int(a) <= stamp < int(b) for a,b in ranges):
            accepted.append(i)
    segments = []
    for i in accepted:
        if segments and segments[-1][1] == i:
            segments[-1] = (segments[-1][0], i+1)
        else:
            segments.append((i,i+1))
    return segments


def source_clips(dataset: Path, seed: int):
    """Index usable videos, respecting excluded takes/cameras and curated keep ranges."""
    curation = dataset.parent / 'curation.json'
    edits = json.loads(curation.read_text()).get('takes', {}) if curation.is_file() else {}
    groups = defaultdict(list)
    for meta_path in sorted((dataset / 'recordings').glob('*/*/label_*/take_*/metadata.json')):
        rel = meta_path.parent.relative_to(dataset / 'recordings').as_posix()
        edit = edits.get(rel, {})
        if edit.get('exclude_take'):
            continue
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        label = int(meta['label'])
        person = str(meta.get('person_id', meta_path.parents[3].name))
        for camera in sorted(meta.get('cameras', {})):
            if camera in edit.get('exclude_cameras', []):
                continue
            video = meta_path.parent / f'{camera}_raw.mp4'
            if not video.is_file():
                continue
            cap = cv2.VideoCapture(str(video))
            try:
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            finally:
                cap.release()
            if frames > 0:
                intervals = _allowed_segments(meta_path.parent / f'{camera}_timestamps.csv',
                    edit.get('keep_ranges_ns'), frames)
                for start, end in intervals:
                    groups[(label, camera)].append({'clip': video.relative_to(dataset).as_posix(),
                        'person': person, 'label': label, 'camera': camera,
                        'source_start': start, 'count': end-start})
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    return groups


def build_plan(dataset: Path, frames: int, seed: int):
    """Take contiguous frames from balanced class/camera groups, bounded RAM."""
    groups = source_clips(dataset, seed)
    if not groups:
        raise ValueError('No readable videos in dataset')
    keys = sorted(groups)
    per_turn = max(1, math.ceil(frames / len(keys)))
    positions = {k: 0 for k in keys}
    consumed = defaultdict(int)
    plan = []
    remaining = frames
    while remaining:
        progress = False
        for key in keys:
            if not remaining:
                break
            rows = groups[key]
            pos = positions[key]
            while pos < len(rows) and consumed[(rows[pos]['clip'], rows[pos]['source_start'])] >= rows[pos]['count']:
                pos += 1
            positions[key] = pos
            if pos == len(rows):
                continue
            row = rows[pos]
            used = consumed[(row['clip'], row['source_start'])]
            take = min(per_turn, row['count'] - used, remaining)
            # Use a single contiguous region of a recording; no temporal jumps.
            if take:
                plan.append({k: row[k] for k in ('clip','person','label','camera')} |
                    {'start': row['source_start'] + used, 'frames': take})
                consumed[(row['clip'], row['source_start'])] += take
                remaining -= take
                progress = True
        if not progress:
            break
    if not plan:
        raise ValueError('Could not build frame plan')
    return plan


def iter_images(dataset: Path, plan):
    """Sequential decode; reading/seek is excluded from inference timing."""
    for segment in plan:
        cap = cv2.VideoCapture(str(dataset / segment['clip']))
        try:
            if not cap.isOpened():
                raise RuntimeError(f'Could not decode {segment["clip"]}')
            if segment['start']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, segment['start'])
            for i in range(segment['frames']):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Missing frame {segment["start"]+i} in {segment["clip"]}')
                yield segment, segment['start'] + i, frame, i == 0
        finally:
            cap.release()


class GPUMonitor:
    """Best-effort background telemetry; a missing nvidia-smi never affects inference."""
    def __init__(self, device):
        self.enabled = str(device).replace('cuda:', '').isdigit()
        self.device = str(device).replace('cuda:', '')
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    def _poll(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '-i', self.device,
                    '--query-gpu=temperature.gpu,clocks.sm,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=4)
                if result.returncode == 0:
                    numbers = [float(v.strip()) for v in result.stdout.splitlines()[0].split(',')]
                    if len(numbers) == 4:
                        self.samples.append(numbers)
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(1.0)

    def __enter__(self):
        if self.enabled:
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)

    def summary(self):
        if not self.samples:
            return {'gpu_telemetry': 'unavailable'}
        a = np.asarray(self.samples, dtype=float)
        return {'gpu_samples': len(a), 'gpu_temp_min_c': float(a[:,0].min()),
                'gpu_temp_max_c':float(a[:,0].max()), 'gpu_clock_mean_mhz':round(float(a[:,1].mean()),1),
                'gpu_clock_min_mhz':float(a[:,1].min()), 'gpu_util_mean_pct':round(float(a[:,2].mean()),1),
                'gpu_power_mean_w':round(float(a[:,3].mean()),1)}


def load_classifier(path: Path | None):
    if path is None:
        return None
    import torch
    from shared.mlp_model import PostureMLP
    path = path / 'model_final.pt' if path.is_dir() else path
    if not path.is_file():
        raise FileNotFoundError(f'MLP checkpoint not found: {path}; use --classifier none to skip')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['feature_contract']['version'] != FEATURE_VERSION:
        raise ValueError('MLP feature contract mismatch')
    model = PostureMLP(model_name=checkpoint['model_name'],
        dropout=checkpoint['model_config']['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model


def classifier_predictions(model, outputs):
    if model is None:
        return
    import torch
    # Classification happens AFTER pose timing: it cannot bias pose FPS.
    keys = [key for key, x in outputs.items() if x['present'] and x['valid_joints'] >= 4]
    if not keys:
        return
    features = np.asarray([outputs[key]['features'] for key in keys], dtype=np.float32)
    with torch.inference_mode():
        probs = torch.softmax(model(torch.from_numpy(features)), dim=1).numpy()
    for key, p in zip(keys, probs):
        outputs[key]['mlp_class'] = int(p.argmax())
        outputs[key]['mlp_confidence'] = float(p.max())
        outputs[key]['mlp_probs'] = p.tolist()


def run_pass(path, dataset, plan, device, warmup, model_classifier=None):
    ensure_gpu(device, path)
    gpu = str(device) in ('0', 'cuda:0')
    if gpu:
        import torch
    runner = PoseRunner('yolo', str(path), video=True, device=device)
    latencies, results = [], {}
    selected_count = valid_count = frames = 0
    try:
        # Warm the model and backend before measuring. Reset tracking after warmup.
        first = next(iter_images(dataset, plan))[2]
        for _ in range(warmup):
            runner.track_frame(first)
        if path.suffix == '.onnx' and gpu:
            backend = getattr(getattr(runner.engine, 'predictor', None), 'model', None)
            session = getattr(backend, 'session', None)
            providers = session.get_providers() if session is not None else []
            if 'CUDAExecutionProvider' not in providers:
                raise RuntimeError(f'ONNX not confirmed on CUDA: {providers}')
        runner.reset_tracking()
        selector, normalizer = SinglePersonSelector(), CausalPoseNormalizer()
        with GPUMonitor(device) as monitor:
            for segment, index, frame, is_start in iter_images(dataset, plan):
                if is_start:
                    runner.reset_tracking()
                    selector, normalizer = SinglePersonSelector(), CausalPoseNormalizer()
                if gpu: torch.cuda.synchronize()
                start = time.perf_counter()
                people = runner.track_frame(frame)
                selected, _ = selector.select(people)
                kp = selected.get('keypoints') if selected is not None else None
                x, _, _ = normalizer.transform(kp, frame.shape[1], frame.shape[0], backend='yolo')
                if gpu: torch.cuda.synchronize()
                latencies.append(1000 * (time.perf_counter() - start))
                key = segment['clip'] + '#' + str(index)
                results[key] = {'person':segment['person'], 'label':segment['label'],
                    'camera':segment['camera'], 'present':kp is not None,
                    'keypoints': kp, 'valid_joints':int(x[:,3].sum()),
                    'features': x.reshape(-1).tolist()}
                frames += 1
                selected_count += (kp is not None)
                valid_count += int(x[:,3].sum())
        telemetry = monitor.summary()
    finally:
        runner.close()
        del runner
        gc.collect()
        if gpu: torch.cuda.empty_cache()
    classifier_predictions(model_classifier, results)
    array = np.asarray(latencies)
    row = {'model':path.name,'frames':frames,
        'fps':round(float(1000/array.mean()),2),
        'mean_ms':round(float(array.mean()),3),
        'p95_ms':round(float(np.percentile(array,95)),3),
        'selected_poses':selected_count,'valid_joints':valid_count,
        **telemetry}
    return row, results


def compare_mlp(a, b):
    keys = [k for k in a.keys() & b.keys() if 'mlp_class' in a[k] and 'mlp_class' in b[k]]
    if not keys:
        return {'mlp_compared_frames':0,'mlp_class_disagree_frames':None,
                'mlp_prob_l1_mean':None, 'mlp_take_label_match_delta_pp':None}
    disagreement = sum(a[k]['mlp_class'] != b[k]['mlp_class'] for k in keys)
    l1 = [sum(abs(x-y) for x,y in zip(a[k]['mlp_probs'], b[k]['mlp_probs'])) for k in keys]
    label_a = sum(a[k]['mlp_class'] == a[k]['label'] for k in keys) / len(keys)
    label_b = sum(b[k]['mlp_class'] == b[k]['label'] for k in keys) / len(keys)
    return {'mlp_compared_frames':len(keys), 'mlp_class_disagree_frames':disagreement,
            'mlp_prob_l1_mean':round(float(np.mean(l1)),4),
            'mlp_take_label_match_delta_pp':round(100*(label_b-label_a),2)}


def compare_outputs(reference, target):
    common = sorted(reference.keys() & target.keys())
    a, b = ({k:reference[k] for k in common}, {k:target[k] for k in common})
    out = compare(a,b) | compare_mlp(a,b)
    # Highest-risk frames for manual Replay: missing poses, class changes, and
    # largest geometric disagreement. This is NOT a ground-truth error ranking.
    review = []
    for key in common:
        r, s = a[key], b[key]
        different_pose = r['present'] != s['present']
        changed_class = ('mlp_class' in r and 'mlp_class' in s and
                         r['mlp_class'] != s['mlp_class'])
        delta = 0.0
        if r['present'] and s['present']:
            ra = np.asarray(r['keypoints'],dtype=np.float32)
            sa = np.asarray(s['keypoints'],dtype=np.float32)
            valid = (ra[:,2]>=.35)&(sa[:,2]>=.35)
            if valid.any():
                delta = float(np.linalg.norm(ra[valid,:2]-sa[valid,:2],axis=1).mean())
        review.append({'frame':key, 'label':r['label'], 'person':r['person'],
            'reference_present':r['present'],'candidate_present':s['present'],
            'reference_class':r.get('mlp_class'), 'candidate_class':s.get('mlp_class'),
            'reference_conf':r.get('mlp_confidence'), 'candidate_conf':s.get('mlp_confidence'),
            'mean_joint_delta_px':round(delta,2),
            'class_changed':bool(changed_class), 'pose_changed':bool(different_pose)})
    review.sort(key=lambda r: (r['pose_changed'],r['class_changed'],r['mean_joint_delta_px']), reverse=True)
    review=review[:40]  # Small, actionable review list; no full-frame dumping.

    return out, review


def make_dir(root):
    root.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        path = root / f'pose_sustained_v{i:03d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            i += 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'artifacts/dataset_working/Real_Posture_Dataset')
    parser.add_argument('--pretrained',type=Path,default=ROOT/'artifacts/pretrained')
    parser.add_argument('--device',default='0',help='GPU device 0 or cpu')
    parser.add_argument('--models',choices=MODEL_SETS,default='pt',help='pt: n/s PT; all: n/s PT+ONNX')
    parser.add_argument('--frames',type=int,default=720,help='target number of contiguous source frames per pass')
    parser.add_argument('--rounds',type=int,default=4,help='4 recommended for balanced order ABBA')
    parser.add_argument('--warmup',type=int,default=8)
    parser.add_argument('--seed',type=int,default=302)
    parser.add_argument('--classifier',default=str(ROOT/'artifacts/models/mlp_wide/v001'),
                        help='MLP final checkpoint directory/path; use none to skip')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/benchmarks')
    args = parser.parse_args(argv)
    if args.frames<24 or args.rounds<2 or args.warmup<0:
        parser.error('Require --frames >=24, --rounds >=2, --warmup >=0')
    paths = [args.pretrained/n for n in MODEL_SETS[args.models]]
    for p in paths:
        if not p.is_file():
            parser.error(f'Missing model {p}')
    plan = build_plan(args.data, args.frames, args.seed)
    total = sum(p['frames'] for p in plan)
    print(f'[SAMPLE] {total} contiguous frames across {len(plan)} segments, '
          f'{len({(r["label"],r["camera"]) for r in plan})} class/camera groups',flush=True)
    print('[NOTE] MP4 decode excluded; no Track ID dependency; curated intervals respected.',flush=True)
    model = load_classifier(None if args.classifier.lower()=='none' else Path(args.classifier))
    dest = make_dir(args.output)
    rows, reference_outputs = [], {}
    try:
        for r in range(args.rounds):
            order = paths if r % 2 == 0 else list(reversed(paths))
            for path in order:
                print(f'[ROUND {r+1}/{args.rounds}] {path.name}', flush=True)
                result, output = run_pass(path, args.data, plan, args.device, args.warmup, model)
                row = {'round':r+1,'device':args.device, **result}
                rows.append(row)
                print(f'   {result["fps"]} FPS | P95 {result["p95_ms"]} ms | '
                      f'pose {result["selected_poses"]}/{result["frames"]} | '
                      f'GPU max {result.get("gpu_temp_max_c", "n/a")} C',flush=True)
                if path.name not in reference_outputs:
                    reference_outputs[path.name] = output
        # n.pt is the reference for agreement, NOT keypoint ground truth.
        ref_name = MODEL_SETS[args.models][0]
        comparisons, review = [], []
        for name, output in reference_outputs.items():
            if name == ref_name:
                continue
            comparison, items = compare_outputs(reference_outputs[ref_name],output)
            comparisons.append({'reference':ref_name,'candidate':name,**comparison})
            review.extend({'candidate':name, **v} for v in items)
        by_model = defaultdict(list)
        for row in rows:
            by_model[row['model']].append(row)
        summary = []
        for name, records in by_model.items():
            speeds = [x['fps'] for x in records]
            summary.append({'model':name,'rounds':len(records),
                'fps_mean':round(statistics.mean(speeds),2),
                'fps_min':round(min(speeds),2),'fps_max':round(max(speeds),2),
                'fps_std':round(statistics.stdev(speeds),2) if len(speeds)>1 else 0,
                'p95_mean_ms':round(statistics.mean(x['p95_ms'] for x in records),2),
                'pose_frames':records[0]['selected_poses'],
                'valid_joints':records[0]['valid_joints'],
                'gpu_temp_max_c':max((x.get('gpu_temp_max_c',-1) for x in records)),
                'gpu_clock_min_mhz':min((x.get('gpu_clock_min_mhz',float('inf')) for x in records))})
        def write_csv(filename, items):
            if not items:
                return
            with (dest/filename).open('w',newline='',encoding='utf-8') as fp:
                writer=csv.DictWriter(fp,fieldnames=list(dict.fromkeys(k for item in items for k in item)))
                writer.writeheader(); writer.writerows(items)
        write_csv('runs.csv',rows)
        write_csv('summary.csv',summary)
        write_csv('pose_mlp_comparison.csv',comparisons)
        write_csv('review_frames.csv',review)
        (dest/'results.json').write_text(json.dumps({'config':vars(args)|{'data':str(args.data),'pretrained':str(args.pretrained),'output':str(args.output)},
            'plan':plan,'models_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'runs':rows,'summary':summary,'comparisons':comparisons,
            'note':'Pose disagreement is relative to n.pt; MLP take-label match is NOT held-out accuracy. '
                   'Timed FPS excludes video decode, classifier, GUI and disk I/O.'},indent=2)+'\n')
        print('\n=== ORDER-BALANCED SUMMARY ===')
        for row in summary:
            print(f'{row["model"]:20} mean {row["fps_mean"]:7.2f} FPS  '
                  f'[{row["fps_min"]:.2f}..{row["fps_max"]:.2f}]',flush=True)
        print('[OK]',dest,flush=True)
    except BaseException:
        (dest/'INCOMPLETE.txt').write_text('Failed benchmark; do not use partial results.\n')
        raise


if __name__=='__main__':
    main()
