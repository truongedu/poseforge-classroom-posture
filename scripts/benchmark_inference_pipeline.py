#!/usr/bin/env python3
"""Read-only Phase 3.1 baseline: profile the *existing* PosePipeline, not a second model path.

MP4 decode/seek, GUI display, file writes and model initialization are excluded.
YOLO internal timing uses the Ultralytics result.speed profiler (GPU synchronized).
Framewise throughput is intentionally measured, not FPS constrained by source playback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'collector'))
sys.path.insert(0, str(ROOT / 'scripts'))
from inference import pipeline as inference_module
from inference.pipeline import PosePipeline, PostureClassifier
# The CLI runs as a file as well as a module; resolve this sibling directly.
# Do not import `scripts.*`: ROS Humble also installs a `scripts` package.
from benchmark_sustained_pose import build_plan, iter_images, GPUMonitor

STAGES = (
    'yolo_preprocess_ms', 'yolo_forward_ms', 'yolo_postprocess_ms',
    'bytetrack_ms', 'pose_runner_ms', 'selector_ms', 'normalizer_ms',
    'mlp_predict_ms', 'mlp_forward_ms', 'smoothing_ms', 'overlay_ms', 'pipeline_ms',
)


def version_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(1, 100000):
        path = root / f'v{i:03d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError('Too many benchmark versions')


def metrics(values):
    """Absent stages are missing, not zero. Avoid false FPS on skipped MLP frames."""
    nums = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not nums:
        return {'count': 0, 'mean_ms': None, 'p50_ms': None,
                'p95_ms': None, 'p99_ms': None, 'fps': None}
    mean = statistics.fmean(nums)
    return {'count': len(nums), 'mean_ms': round(mean, 5),
            'p50_ms': round(float(np.percentile(nums, 50)), 5),
            'p95_ms': round(float(np.percentile(nums, 95)), 5),
            'p99_ms': round(float(np.percentile(nums, 99)), 5),
            'fps': round(1000. / mean, 2) if mean > 0 else None}


def summarize(rows, round_no=None):
    selected = [r for r in rows if round_no is None or r['round'] == round_no]
    total_ms = sum(r['pipeline_ms'] for r in selected)
    result = {
        'round': 'all' if round_no is None else round_no,
        'frames': len(selected),
        'detected_frames': sum(bool(r['selected_pose']) for r in selected),
        'classified_frames': sum(bool(r['classified']) for r in selected),
        'pipeline_fps': round(1000 * len(selected) / total_ms, 2) if total_ms > 0 else None,
        'pipeline_mean_ms': round(total_ms / len(selected), 5) if selected else None,
    }
    for key in STAGES:
        result[key] = metrics([row.get(key) for row in selected])
    return result


class Instrument:
    """Temporary, reversible instrumentation on one existing inference pipeline."""

    def __init__(self, worker: PosePipeline, overlay: bool):
        self.worker = worker
        self.overlay = overlay
        self.frame = {}
        self.originals = []
        self.hooks = []

    def _wrap(self, obj, attribute, field, call=True):
        old = getattr(obj, attribute)

        def measured(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return old(*args, **kwargs) if call else args[0]
            finally:
                self.frame[field] = self.frame.get(field, 0.) + (time.perf_counter_ns() - start) / 1e6

        setattr(obj, attribute, measured)
        self.originals.append((obj, attribute, old))

    def __enter__(self):
        # Pipe objects retain the exact same state and implementations during measurement.
        self._wrap(self.worker.selector, 'select', 'selector_ms')
        self._wrap(self.worker.normalizer, 'transform', 'normalizer_ms')
        self._wrap(self.worker.classifier, 'predict', 'mlp_predict_ms')
        classifier = self.worker.classifier
        if getattr(classifier, 'backend', 'pt') == 'pt' and hasattr(classifier, 'model'):
            # Two hooks measure only the MLP forward, excluding NumPy conversion,
            # host->device transfer, softmax and device->host transfer.
            def synchronize():
                if str(getattr(classifier, 'device', 'cpu')).startswith('cuda'):
                    import torch
                    torch.cuda.synchronize(classifier.device)

            starts = []

            def before(*_):
                synchronize()
                starts.append(time.perf_counter_ns())

            def after(*_):
                synchronize()
                if starts:
                    self.frame['mlp_forward_ms'] = self.frame.get('mlp_forward_ms', 0.) + (
                        time.perf_counter_ns() - starts.pop()) / 1e6

            self.hooks.append(classifier.model.register_forward_pre_hook(before))
            self.hooks.append(classifier.model.register_forward_hook(after))
        elif getattr(classifier, 'backend', None) == 'onnx' and hasattr(classifier, 'session'):
            try:
                self._wrap(classifier.session, 'run', 'mlp_forward_ms')
            except (AttributeError, TypeError):
                # Some ONNX Runtime versions prevent patching run(); do not
                # mislabel the full predict() time as a pure model forward.
                pass
        self._wrap(self.worker.stable, 'update', 'smoothing_ms')
        self._wrap(inference_module, 'draw_overlay', 'overlay_ms', self.overlay)
        original_track = self.worker.pose.track_frame

        def track(frame):
            start = time.perf_counter_ns()
            try:
                return original_track(frame)
            finally:
                self.frame['pose_runner_ms'] = (time.perf_counter_ns() - start) / 1e6
                predictor = getattr(self.worker.pose.engine, 'predictor', None)
                results = getattr(predictor, 'results', None)
                speed = getattr(results[0], 'speed', None) if results else None
                if isinstance(speed, dict):
                    for stage, key in (('preprocess','yolo_preprocess_ms'),
                                       ('inference','yolo_forward_ms'),
                                       ('postprocess','yolo_postprocess_ms')):
                        if isinstance(speed.get(stage), (float, int)):
                            self.frame[key] = float(speed[stage])

        self.worker.pose.track_frame = track
        self.originals.append((self.worker.pose, 'track_frame', original_track))
        # Instrument the real ByteTrack.update, not an approximated residual.
        predictor = getattr(self.worker.pose.engine, 'predictor', None)
        trackers = getattr(predictor, 'trackers', []) if predictor else []
        for tracker in trackers:
            if type(tracker).__name__ == 'BYTETracker' and callable(getattr(tracker, 'update', None)):
                self._wrap(tracker, 'update', 'bytetrack_ms')
        return self

    def __exit__(self, *_):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        for obj, attr, old in reversed(self.originals):
            setattr(obj, attr, old)
        self.originals.clear()

    def process(self, frame):
        self.frame = {}
        start = time.perf_counter_ns()
        info, _ = self.worker.process(frame)
        total = (time.perf_counter_ns() - start) / 1e6
        return {'pipeline_ms': total, **self.frame,
                'selected_pose': int(info.get('track_id') is not None or info.get('keypoints_valid', 0) > 0),
                'classified': int(info.get('probs') is not None),
                'valid_joints': int(info.get('keypoints_valid') or 0),
                'class_id': info.get('class_id')}


def video_frames(path, limit):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError(f'Cannot decode video: {path}')
        for index in range(limit):
            success, frame = cap.read()
            if not success:
                break
            yield {'clip': path.name, 'camera': 'video'}, index, frame, index == 0
    finally:
        cap.release()


def frames_for(args, plan):
    if args.video is not None:
        yield from video_frames(args.video, args.frames)
    else:
        yield from iter_images(args.data, plan)


def hash_file(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_report(path, summaries, classifier_family='mlp'):
    # Visualization is run *after* timing. Not required to run benchmark.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows = summaries['stages']
        names = [x['stage'].replace('_ms','').replace('yolo_','YOLO ') for x in rows
                 if x['stage'] != 'pipeline_ms' and x['mean_ms'] is not None]
        names = [name.replace('mlp predict', f'{classifier_family.upper()} predict').replace('mlp forward', f'{classifier_family.upper()} forward') for name in names]
        vals = [x['mean_ms'] for x in rows if x['stage'] != 'pipeline_ms' and x['mean_ms'] is not None]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.barh(names, vals)
        ax.invert_yaxis()
        ax.set_xlabel('Mean time / executed frame (ms)')
        ax.set_title('PoseForge stage latency — diagnostic, not additive')
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except (ImportError, RuntimeError) as exc:
        print(f'[WARN] report.png unavailable: {exc}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--data', type=Path, help='Working dataset; honors local curation.json')
    input_group.add_argument('--video', type=Path, help='A long continuous MP4 (recommended for maximum FPS)')
    parser.add_argument('--pose', type=Path, default=ROOT/'artifacts/pretrained/yolo11n-pose.pt')
    parser.add_argument('--classifier', type=Path, default=ROOT/'artifacts/models/mlp_wide/v002')
    parser.add_argument('--device', default='0', help='YOLO device: cpu, 0, cuda:0')
    parser.add_argument('--cls-device', default='cuda:0', help='Classifier device: cuda:0 (default), cpu for comparison')
    parser.add_argument('--cls-backend', choices=('pt', 'onnx'), default='pt')
    parser.add_argument('--frames', type=int, default=720)
    parser.add_argument('--warmup', type=int, default=15)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=302)
    parser.add_argument('--overlay', choices=('on','off'), default='on',
                        help='on measures current baseline with overlay; off estimates headless potential')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/benchmarks/inference_pipeline')
    args = parser.parse_args(argv)
    if args.frames < 1 or args.rounds < 1 or args.warmup < 0:
        parser.error('Require --frames>=1, --rounds>=1, --warmup>=0')
    if not args.pose.is_file():
        parser.error(f'Missing pose checkpoint: {args.pose}')
    if not (args.classifier / 'model_final.pt').is_file():
        parser.error(f'Missing final classifier: {args.classifier}/model_final.pt')
    if args.video is not None and not args.video.is_file():
        parser.error(f'Missing video: {args.video}')
    if args.data is not None and not args.data.is_dir():
        parser.error(f'Missing dataset: {args.data}')

    plan = None if args.video else build_plan(args.data, args.frames, args.seed)
    print(f'[INPUT] {args.video or args.data} | frames/round <= {args.frames} | rounds={args.rounds}',flush=True)
    print('[NOTE] Decode/capture, UI, disk write and model loading excluded. No FPS cap.', flush=True)
    classifier = PostureClassifier(args.classifier, backend=args.cls_backend, device=args.cls_device)
    worker = PosePipeline(args.pose, classifier, device=args.device,
                          overlay=args.overlay == 'on')
    dest = None
    rows, round_summaries = [], []
    try:
        # Preheat Ultralytics/torch and then clear causal history for the timed pass.
        first = next(frames_for(args, plan), None)
        if first is None:
            raise RuntimeError('No readable frames')
        for _ in range(args.warmup):
            worker.process(first[2])
        worker.reset()
        with GPUMonitor(args.device) as monitor:
            for r in range(args.rounds):
                worker.reset()
                with Instrument(worker, overlay=args.overlay == 'on') as instrument:
                    count = 0
                    for segment, index, frame, is_start in frames_for(args, plan):
                        if is_start:
                            # Reset state, not the instrumented selector instance.
                            worker.reset()
                        result = instrument.process(frame)
                        rows.append({'round': r+1, 'clip': segment['clip'], 'frame': index,
                                     **result})
                        count += 1
                    if count == 0:
                        raise RuntimeError('No decodable frames in round')
                result = summarize(rows, r+1)
                round_summaries.append({k: result[k] for k in ('round','frames','detected_frames',
                    'classified_frames','pipeline_fps','pipeline_mean_ms')})
                print(f'[ROUND {r+1}] {count} frames | pipeline {result["pipeline_fps"]} FPS | '
                      f'YOLO {result["yolo_forward_ms"]["fps"]} FPS | {str(getattr(classifier, "model_family", "mlp")).upper()} {result["mlp_forward_ms"]["fps"]} FPS',flush=True)
        telemetry = monitor.summary()
        overall = summarize(rows)
        stage_rows = [{'stage': key, **overall[key]} for key in STAGES]
        dest = version_dir(args.output)
        (dest / 'INCOMPLETE.txt').write_text('Writing benchmark outputs\n')
        cfg = {'classifier_family':getattr(classifier, 'model_family', 'mlp'), 'phase': '3.1', 'timestamp_utc':datetime.now(timezone.utc).isoformat(),
               'video': args.video.name if args.video else None,
               'data': str(args.data) if args.data else None,
               'pose': args.pose.name, 'pose_sha256': hash_file(args.pose),
               'classifier': str(args.classifier), 'classifier_sha256': hash_file(args.classifier/'model_final.pt'),
               'device':args.device, 'cls_device':args.cls_device, 'cls_backend':args.cls_backend,
               'frames_requested':args.frames, 'rounds':args.rounds,'warmup':args.warmup,
               'seed':args.seed,'overlay':args.overlay, 'gpu_telemetry':telemetry,
               'timing_notes': 'Stages are nested and must not be summed. Ultralytics speed for YOLO internals; '
                               'ByteTrack.update measured when available. FPS per stage is reciprocal of '
                               'mean latency only for frames executing that stage. Legacy mlp_forward_ms CSV key tracks the selected MLP or GRU classifier forward. Pipeline FPS excludes MP4 decode and UI.'}
        (dest / 'config.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n')
        write_csv(dest / 'stage_latency.csv', stage_rows)
        write_csv(dest / 'runs.csv', round_summaries)
        # Avoid redundant per-frame file. The summary has per-round and per-stage evidence.
        summary = {k: v for k, v in overall.items() if k not in STAGES}
        summary.update({'classifier_family':getattr(classifier, 'model_family', 'mlp'), 'classifier_fps':overall['mlp_forward_ms']['fps'],
                        'yolo_fps':overall['yolo_forward_ms']['fps'],
                        'mlp_fps':overall['mlp_forward_ms']['fps'], 'overlay':args.overlay})
        write_csv(dest / 'summary.csv', [summary])
        draw_report(dest / 'report.png', {'stages':stage_rows}, getattr(classifier, 'model_family', 'mlp'))
        (dest / 'INCOMPLETE.txt').unlink()
        print(f'[OK] {dest} | {overall["frames"]} frames | pipeline {overall["pipeline_fps"]} FPS', flush=True)
        print(f'[FPS] YOLO forward={overall["yolo_forward_ms"]["fps"]} | '
              f'{str(getattr(classifier, "model_family", "mlp")).upper()}={overall["mlp_forward_ms"]["fps"]} | pipeline={overall["pipeline_fps"]}',flush=True)
        if overall['yolo_forward_ms']['count'] == 0:
            print('[WARN] YOLO internal stage timings unavailable; check Ultralytics result.speed', flush=True)
        if overall['mlp_forward_ms']['count'] == 0:
            print('[WARN] Pure MLP forward timing unavailable; see mlp_predict_ms for wrapper timing', flush=True)
    finally:
        worker.close()
    return dest


if __name__ == '__main__':
    main()
