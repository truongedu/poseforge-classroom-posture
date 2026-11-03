#!/usr/bin/env python3
"""PoseForge inference: annotated image/video or reconnecting IP-camera snippets."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import zlib
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.pipeline import PosePipeline, PostureClassifier, draw_dashboard, CLASS_COLORS

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
FIELDS = ['frame', 'elapsed_s', 'class_id', 'label', 'score', 'stable_id',
          'keypoints_valid', 'selection_reason', 'track_id', 'inference_ms'] + [f'p{i}' for i in range(6)]


def log_row(frame_no, elapsed, info):
    row = {k: info.get(k) for k in FIELDS if k not in ('frame', 'elapsed_s') and not k.startswith('p')}
    row.update(frame=frame_no, elapsed_s=round(elapsed, 3))
    if info.get('probs') is not None:
        row.update({f'p{i}': round(float(p), 6) for i, p in enumerate(info['probs'])})
    return row


def label_on_frame(frame, info):
    result = frame.copy()
    idx = info.get('class_id')
    msg = ('NO POSE' if idx is None else
           f"{info['label']}  {info['score']*100:.1f}%")
    color = ((80, 170, 240) if idx is None else CLASS_COLORS[int(idx)])
    cv2.rectangle(result, (0, 0), (min(result.shape[1], 470), 44), (27, 29, 33), -1)
    cv2.putText(result, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2, cv2.LINE_AA)
    return result


def make_run_dir(base, name):
    base.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r'[^a-zA-Z0-9_-]+', '_', name).strip('_')[:60] or 'infer'
    for n in range(1, 100000):
        path = base / f'{tag}_v{n:03d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError('Too many runs')


def choose_model(root, model):
    if model:
        return Path(model).expanduser().resolve()
    matches = list((root / 'artifacts/models').glob('*/v*/model_final.pt'))
    if not matches:
        raise FileNotFoundError('No final model found in artifacts/models/<name>/vNNN/')
    return max(matches, key=lambda f: (f.stat().st_mtime_ns, str(f))).parent


def clean_source(value):
    """Do not store camera credentials in any artifact or terminal output."""
    s = urlsplit(value)
    if s.scheme.lower() in {'rtsp', 'rtsps', 'http', 'https'}:
        host = s.hostname or 'camera'
        return urlunsplit((s.scheme, f'{host}:{s.port}' if s.port else host, '', '', ''))
    return str(Path(value).name)


def video_writer(path, width, height, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'),
                             max(1., float(fps)), (width, height))
    if not writer.isOpened():
        raise OSError(f'Cannot create MP4: {path}')
    return writer


def run_file(source, mode, options, classify, run_dir):
    pipeline = PosePipeline(options.pose, classify, options.device, options.stable_frames,
                            options.min_conf, overlay=getattr(options, 'overlay', False))
    output = run_dir / 'frames.csv'
    try:
        if mode == 'image':
            image = cv2.imread(str(source))
            if image is None:
                raise OSError('Cannot decode image')
            info, annotated = pipeline.process(image)
            annotated = label_on_frame(annotated, info)
            cv2.imwrite(str(run_dir / 'annotated.jpg'), annotated)
            (run_dir / 'prediction.json').write_text(json.dumps(info, indent=2) + '\n')
            print(f'[IMAGE] {info["label"]} score={info["score"]:.3f} -> {run_dir}')
            if options.display:
                cv2.imshow('PoseForge', annotated)
                cv2.waitKey(0)
            return
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise OSError('Cannot open video')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.
        fps = fps if 1. <= fps <= 120. else 25.
        writer, frames = None, 0
        t0 = time.monotonic()
        try:
            with output.open('w', newline='', encoding='utf-8') as handle:
                log = csv.DictWriter(handle, fieldnames=FIELDS)
                log.writeheader()
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = video_writer(run_dir / 'annotated.mp4', w, h, fps)
                    info, annotated = pipeline.process(frame)
                    annotated = label_on_frame(annotated, info)
                    writer.write(annotated)
                    log.writerow(log_row(frames, frames / fps, info))
                    frames += 1
                    if options.display:
                        cv2.imshow('PoseForge', annotated)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    if options.max_frames and frames >= options.max_frames:
                        break
            print(f'[VIDEO] {frames} frames in {time.monotonic()-t0:.1f}s -> {run_dir}')
        finally:
            cap.release()
            if writer is not None:
                writer.release()
    finally:
        pipeline.close()
        if options.display:
            cv2.destroyAllWindows()


class CameraReader:
    """Only latest decoded frame retained. I/O isolated from inference thread."""

    def __init__(self, source, name, open_timeout_ms=5000, read_timeout_ms=5000, freeze_seconds=15.):
        self.source = int(source) if str(source).isdigit() else str(source)
        self.name = name
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.seq = 0
        self.generation = 0
        self.freeze_seconds = freeze_seconds
        self.thread = threading.Thread(target=self._work, daemon=True, name=f'camera-{name}')
        self.thread.start()

    def _open(self):
        if isinstance(self.source, int):
            return cv2.VideoCapture(self.source)
        opts = []
        if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
            opts.extend((cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms))
        if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
            opts.extend((cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms))
        try:
            return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, opts)
        except (TypeError, cv2.error):
            return cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

    def _work(self):
        delay = .5
        while not self.stop.is_set():
            cap = None
            try:
                cap = self._open()
                if not cap.isOpened():
                    raise OSError('open failed')
                with self.lock:
                    self.generation += 1
                    self.latest = None
                print(f'[STREAM] {self.name} connected (session {self.generation})', flush=True)
                delay = .5
                previous_hash, unchanged_since = None, time.monotonic()
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise OSError('frame read failed')
                    # Identical decoded frames for too long may mean a frozen RTSP feed.
                    digest = zlib.crc32(frame[::8, ::8].tobytes())
                    now = time.monotonic()
                    if digest != previous_hash:
                        previous_hash, unchanged_since = digest, now
                    elif self.freeze_seconds > 0 and now - unchanged_since > self.freeze_seconds:
                        raise OSError('frozen feed')
                    with self.lock:
                        self.seq += 1
                        self.latest = (self.generation, self.seq, time.monotonic(), frame)
            except Exception:
                if not self.stop.is_set():
                    print(f'[STREAM] {self.name} disconnected; retry in {delay:.1f}s', flush=True)
            finally:
                if cap is not None:
                    cap.release()
                with self.lock:
                    self.latest = None
            self.stop.wait(delay)
            delay = min(8., delay * 2)

    def snapshot(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)  # FFmpeg read/open timeout protects most backends.


class EventRecorder:
    """One finite MP4 per class and camera; never records continuously."""

    def __init__(self, folder, labels, duration_s=3., fps=10.):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.labels = labels
        self.length = max(1, int(round(duration_s * fps)))
        self.fps = fps
        self.seen = set()
        self.active = {}
        self.events = []

    def add(self, frame, info):
        cls = info['stable_id']
        if cls is not None and cls not in self.seen:
            self.seen.add(cls)
            filename = f'class_{cls}.mp4'
            h, w = frame.shape[:2]
            self.active[cls] = [video_writer(self.folder / filename, w, h, self.fps), 0]
            self.events.append({'class_id': cls, 'label': self.labels[cls], 'file': filename,
                                'score_at_trigger': round(info['score'], 4),
                                'timestamp': datetime.now().isoformat(timespec='seconds')})
            print(f'[EVENT] {self.folder.name}: {self.labels[cls]} -> {filename}', flush=True)
        for cls, (writer, count) in list(self.active.items()):
            writer.write(frame)
            count += 1
            if count >= self.length:
                writer.release()
                del self.active[cls]
            else:
                self.active[cls][1] = count

    def stop_active(self):
        for writer, _ in self.active.values():
            writer.release()
        self.active.clear()

    def close(self):
        self.stop_active()
        with (self.folder / 'events.csv').open('w', newline='', encoding='utf-8') as handle:
            columns = ['class_id', 'label', 'file', 'score_at_trigger', 'timestamp']
            w = csv.DictWriter(handle, fieldnames=columns)
            w.writeheader()
            w.writerows(self.events)


def run_cameras(sources, options, classifier, run_dir):
    names = ([options.name] if len(sources) == 1 and options.name else
             [f'cam{i:02d}' for i in range(1, len(sources) + 1)])
    # Construct inference state first; a failed second model must not leave
    # the first stream's reader thread running or the first model allocated.
    pipelines, readers, recorders = [], [], []
    try:
        for _ in sources:
            pipelines.append(PosePipeline(options.pose, classifier, options.device,
                                          options.stable_frames, options.min_conf,
                                          overlay=getattr(options, 'overlay', False)))
        for name, url in zip(names, sources):
            readers.append(CameraReader(url, name, options.open_timeout,
                                        options.read_timeout, options.freeze_seconds))
        # A None recorder avoids VideoWriter and frame copies in headless mode.
        for name in names:
            recorders.append(EventRecorder(run_dir / name, classifier.labels,
                                            options.record_seconds, options.event_fps)
                             if getattr(options, 'record_events', False) else None)
            (run_dir / name).mkdir(parents=True, exist_ok=True)
    except BaseException:
        for reader in readers:
            reader.close()
        for recorder in recorders:
            if recorder is not None:
                recorder.close()
        for worker in pipelines:
            worker.close()
        raise
    states = [(-1, -1) for _ in sources]
    last_seen = [time.monotonic()] * len(sources)
    no_frame_reset = [False] * len(sources)
    count = [0] * len(sources)
    total_ms = [0.] * len(sources)
    fps_ema = [0.] * len(sources)
    last_processed = [None] * len(sources)
    last_ui = [0.] * len(sources)
    stop = False

    def quit_signal(*_):
        nonlocal stop
        stop = True

    prev_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, quit_signal)
    started = time.monotonic()
    try:
        while not stop and (not options.max_seconds or time.monotonic() - started < options.max_seconds):
            worked = False
            for i, reader in enumerate(readers):
                item = reader.snapshot()
                now = time.monotonic()
                if item is None or now - item[2] > options.stale_seconds:
                    if now - last_seen[i] > options.stale_seconds and not no_frame_reset[i]:
                        pipelines[i].reset()
                        if recorders[i] is not None:
                            recorders[i].stop_active()
                        no_frame_reset[i] = True
                        print(f'[STREAM] {names[i]} stale; state reset', flush=True)
                    if options.display and now - last_ui[i] > .5:
                        blank = np.full((480, 640, 3), (28, 29, 33), dtype=np.uint8)
                        cv2.imshow(names[i], draw_dashboard(blank, {}, classifier.labels, fps=0., status='RECONNECTING'))
                        last_ui[i] = now
                    continue
                gen, seq, ts, frame = item
                if (gen, seq) == states[i]:
                    continue
                if gen != states[i][0]:
                    pipelines[i].reset()  # fresh ByteTrack, selector and EMA
                    if recorders[i] is not None:
                        recorders[i].stop_active()
                    fps_ema[i] = 0.
                    last_processed[i] = None
                    print(f'[STREAM] {names[i]} fresh pose state', flush=True)
                states[i] = (gen, seq)
                no_frame_reset[i] = False
                last_seen[i] = now
                info, annotated = pipelines[i].process(frame)
                if last_processed[i] is not None:
                    delta = max(1e-6, time.monotonic() - last_processed[i])
                    instant = min(300., 1. / delta)
                    fps_ema[i] = instant if fps_ema[i] == 0 else (.25 * instant + .75 * fps_ema[i])
                last_processed[i] = time.monotonic()
                count[i] += 1
                total_ms[i] += info['inference_ms']
                if recorders[i] is not None:
                    recorders[i].add(label_on_frame(annotated, info), info)
                worked = True
                if options.display:
                    cv2.imshow(names[i], draw_dashboard(annotated, info, classifier.labels, fps=fps_ema[i]))
                    last_ui[i] = now
            if options.display and cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if not worked:
                time.sleep(.015)
    finally:
        for reader in readers:
            reader.close()
        for pipe in pipelines:
            pipe.close()
        for recorder in recorders:
            if recorder is not None:
                recorder.close()
        elapsed = max(.001, time.monotonic() - started)
        for i, name in enumerate(names):
            (run_dir / name / 'performance.json').write_text(json.dumps({
                'processed_frames': count[i], 'wall_seconds': round(elapsed, 2),
                'effective_fps': round(count[i] / elapsed, 2),
                'mean_inference_ms': round(total_ms[i] / max(1, count[i]), 2),
                'saved_classes': sorted(recorders[i].seen) if recorders[i] is not None else [],
            }, indent=2) + '\n')
        if options.display:
            cv2.destroyAllWindows()
        signal.signal(signal.SIGINT, prev_int)
        print(f'[STREAM] Complete -> {run_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', action='append', required=True, help='image/video path, RTSP/HTTP URL, or webcam index')
    ap.add_argument('--mode', choices=['auto', 'image', 'video', 'camera'], default='auto')
    ap.add_argument('--model', type=Path, default=ROOT / 'artifacts/models/mlp_wide/v002',
                    help='installed model directory (default: mlp_wide/v002)')
    ap.add_argument('--pose', type=Path, default=ROOT / 'artifacts/pretrained/yolo11n-pose.pt')
    ap.add_argument('--device', default='auto', help='YOLO device: auto, cpu, cuda:0, ...')
    ap.add_argument('--backend', choices=['auto', 'pt', 'onnx'], default='pt',
                    help='PT by default; auto or ONNX must be selected explicitly')
    ap.add_argument('--cls-device', default='cuda:0', help='classifier cuda:0 (production) or cpu (benchmark); YOLO device is separate')
    ap.add_argument('--name', default='', help='run name (camera: single-camera display name)')
    ap.add_argument('--display', action='store_true', help='force OpenCV window; q to stop')
    ap.add_argument('--overlay', action='store_true',
                    help='draw skeleton overlay (off by default for throughput)')
    ap.add_argument('--record-events', action='store_true',
                    help='save class-triggered camera clips (off by default)')
    ap.add_argument('--headless', action='store_true', help='disable GUI (e.g. SSH/server)')
    ap.add_argument('--stable-frames', type=int, default=5)
    ap.add_argument('--min-conf', type=float, default=.55)
    ap.add_argument('--record-seconds', type=float, default=3.)
    ap.add_argument('--event-fps', type=float, default=10.)
    ap.add_argument('--max-seconds', type=float, default=0., help='camera stop after N seconds; 0=infinite')
    ap.add_argument('--max-frames', type=int, default=0, help='offline video limit for testing; 0=all')
    ap.add_argument('--stale-seconds', type=float, default=8.)
    ap.add_argument('--open-timeout', type=int, default=5000, help='FFmpeg connection timeout ms')
    ap.add_argument('--read-timeout', type=int, default=5000, help='FFmpeg read timeout ms')
    ap.add_argument('--freeze-seconds', type=float, default=15., help='reconnect on exact frozen frames; 0=off')
    args = ap.parse_args()
    if args.stable_frames < 1 or not 0 < args.min_conf <= 1 or args.record_seconds <= 0 or args.event_fps <= 0:
        ap.error('Invalid stability/recording settings')
    source = args.source
    if args.mode == 'auto':
        args.mode = ('camera' if len(source) > 1 or source[0].isdigit() or
                     source[0].lower().startswith(('rtsp://', 'rtsps://', 'http://', 'https://'))
                     else 'image' if Path(source[0]).suffix.lower() in IMAGE_SUFFIXES else 'video')
    if args.mode != 'camera' and len(source) != 1:
        ap.error('Multiple sources require --mode camera')
    if args.headless and args.display:
        ap.error('Choose either --display or --headless')
    if args.headless:
        args.display = False
    elif args.mode == 'camera' and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        args.display = True
    if not args.pose.is_file():
        ap.error(f'YOLO pose weights not found: {args.pose}')
    model_dir = choose_model(ROOT, args.model)
    classifier = PostureClassifier(model_dir, args.backend, args.cls_device)
    run_dir = make_run_dir(ROOT / 'artifacts/inference', args.name or args.mode)
    (run_dir / 'run_config.json').write_text(json.dumps({
        'mode': args.mode, 'model': str(model_dir.relative_to(ROOT)) if model_dir.is_relative_to(ROOT) else str(model_dir),
        'sources': [clean_source(s) for s in source], 'device': args.device,
        'classifier_backend': classifier.backend, 'cls_device': args.cls_device,
        'stable_frames': args.stable_frames,
        'min_conf': args.min_conf, 'overlay': args.overlay,
        'record_events': args.record_events,
    }, ensure_ascii=False, indent=2) + '\n')
    print(f'[RUN] {args.mode} | model={model_dir.name} | backend={classifier.backend} | output={run_dir}')
    if args.mode == 'camera':
        run_cameras(source, args, classifier, run_dir)
    else:
        run_file(source[0], args.mode, args, classifier, run_dir)


if __name__ == '__main__':
    main()
