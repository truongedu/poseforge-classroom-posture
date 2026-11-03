"""Phase 3.3 inference sessions. No Qt, no duplicate detector/capture stack.

A session owns one inference thread and a model/pose state per source. The UI
only consumes rate-limited snapshots; it cannot set the detector's FPS.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from inference.pipeline import PosePipeline, PostureClassifier, PoseRunner, SKELETON
from inference.run import CameraReader, IMAGE_SUFFIXES


@dataclass(frozen=True)
class Source:
    kind: str  # image, video, camera
    value: str


@dataclass(frozen=True)
class Settings:
    stable_frames: int = 5
    min_conf: float = .55
    overlay: bool = True
    render_fps: int = 15
    stale_seconds: float = 8.
    behavior: str = 'fast'  # guarded changes PRESENTATION only, never model logic

    def validate(self):
        if not 1 <= self.stable_frames <= 30 or not 0 < self.min_conf <= 1:
            raise ValueError('Invalid stable_frames/min_conf')
        if not 1 <= self.render_fps <= 60 or self.stale_seconds <= 0:
            raise ValueError('Invalid render_fps/stale_seconds')
        if self.behavior not in ('fast', 'guarded'):
            raise ValueError('Unknown behavior')
        return self


def roi_contains(roi, person, width, height):
    """Normalized ROI excludes detections by bbox center; no image crop/rescale."""
    if not roi:
        return True
    if len(roi) < 3:
        return False
    try:
        box = [float(x) for x in person['bbox']]
        if len(box) != 4 or not np.isfinite(box).all():
            return False
        polygon = np.asarray([(float(x)*width, float(y)*height) for x,y in roi], np.float32)
        if polygon.shape != (len(roi), 2) or not np.isfinite(polygon).all():
            return False
        center = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        return cv2.pointPolygonTest(polygon, center, False) >= 0
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def roi_filter(people, roi, shape):
    if not roi:
        return people
    h, w = shape[:2]
    return [p for p in (people or []) if roi_contains(roi, p, w, h)]


def display_decision(info, behavior, labels):
    """Select the presentation decision without changing model probabilities."""
    if info.get('probs') is None:
        return 'NO POSE'
    if behavior == 'guarded':
        stable_id = info.get('stable_id')
        if stable_id is None:
            return 'WAITING / NO DECISION'
        return str(labels[int(stable_id)])
    return info.get('label') or 'NO POSE'


class RollingFPS:
    def __init__(self, seconds=3.):
        self.seconds = seconds
        self.samples = deque()

    def add(self, value, stamp=None):
        now = time.monotonic() if stamp is None else stamp
        self.samples.append((now, float(value)))
        self.trim(now)

    def trim(self, now=None):
        now = time.monotonic() if now is None else now
        while self.samples and self.samples[0][0] < now - self.seconds:
            self.samples.popleft()

    def fps(self, now=None):
        self.trim(now)
        if not self.samples:
            return None
        mean = sum(x[1] for x in self.samples) / len(self.samples)
        return 1000. / mean if mean > 0 else None


class LiveFPS:
    """Processed frames / wall clock; captures source/scheduling stalls, not UI drawing."""
    def __init__(self, seconds=3.):
        self.seconds = seconds
        self.stamps = deque()

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        self.stamps.append(now)
        self.get(now)

    def get(self, now=None):
        now = time.monotonic() if now is None else now
        while self.stamps and self.stamps[0] < now - self.seconds:
            self.stamps.popleft()
        if len(self.stamps) < 2:
            return None
        duration = now - self.stamps[0]
        return (len(self.stamps)-1) / duration if duration > 0 else None


class ClassifierForwardMeter:
    """Torch forward hooks: pure model forward, not softmax or tensor transfer."""
    def __init__(self, classifier):
        self.value = None
        self._begin = None
        self.hooks = []
        if getattr(classifier, 'backend', None) != 'pt':
            return
        model = getattr(classifier, 'model', None)
        if model is None:
            return
        device = str(getattr(classifier, 'device', 'cpu'))

        def sync():
            if device.startswith('cuda'):
                import torch
                torch.cuda.synchronize(device)

        def before(*_):
            sync()
            self._begin = time.perf_counter_ns()

        def after(*_):
            sync()
            if self._begin is not None:
                self.value = (time.perf_counter_ns()-self._begin)/1e6

        self.hooks = [model.register_forward_pre_hook(before), model.register_forward_hook(after)]

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class StreamState:
    def __init__(self, source, pipeline, roi=None):
        self.source = source
        self.pipeline = pipeline
        self.roi = list(roi or [])
        self.reader = None
        self.video = None
        self.last_key = None
        self.last_input = time.monotonic()
        self.stale_reset = False
        self.yolo = RollingFPS()
        self.mlp = RollingFPS()
        self.pipeline_fps = RollingFPS()
        self.live = LiveFPS()
        self.processed = 0
        self.case_processed = 0

    def reset(self, preserve_key=False):
        key = self.last_key if preserve_key else None
        self.pipeline.reset()
        self.yolo = RollingFPS()
        self.mlp = RollingFPS()
        self.pipeline_fps = RollingFPS()
        self.live = LiveFPS()
        self.last_key = key
        self.case_processed = 0
        self.stale_reset = False

    def close(self):
        if self.reader is not None:
            self.reader.close()
        if self.video is not None:
            self.video.release()
        self.pipeline.close()


class InferenceSession:
    """GUI-agnostic session, one sequential inference worker for up to 4 sources.

    emit(kind, camera_index, payload) is called FROM worker thread. Consumers
    must use queued delivery to Qt's GUI thread. No capture credential logging.
    """
    def __init__(self, sources, pose, model, device='0', cls_device='cuda:0',
                 settings=None, rois=None, emit=None, classifier_factory=PostureClassifier,
                 pipeline_factory=PosePipeline, reader_factory=CameraReader):
        if not 1 <= len(sources) <= 4:
            raise ValueError('Choose 1–4 sources')
        self.sources = list(sources)
        if any(s.kind not in ('image','video','camera') for s in self.sources):
            raise ValueError('Unsupported source')
        self.settings = (settings or Settings()).validate()
        self.pose, self.model, self.device, self.cls_device = pose, model, device, cls_device
        self.rois = rois or {}
        self.emit = emit or (lambda *_: None)
        self.classifier_factory = classifier_factory
        self.pipeline_factory = pipeline_factory
        self.reader_factory = reader_factory
        self.stop = threading.Event()
        self.thread = None
        self.testcase = False
        self._states = []
        self._reset = threading.Event()
        self._finish_case = threading.Event()
        self._resume_case = threading.Event()
        self._case_paused = False
        self._case_id = 1
        self._roi_lock = threading.Lock()
        self._pending_rois = {}

    def start(self, testcase=False):
        self.testcase = bool(testcase)
        if self.thread and self.thread.is_alive():
            raise RuntimeError('Session already running')
        self.stop.clear()
        self._reset.clear()
        self._finish_case.clear()
        self._resume_case.clear()
        self._case_paused = False
        self._case_id = 1
        self.thread = threading.Thread(target=self._run, daemon=True, name='poseforge-inference')
        self.thread.start()

    def request_reset(self):
        # Testcase reset is an explicit case boundary, not a single-frame action.
        if self.testcase:
            self.finish_case()
        else:
            self._reset.set()

    def finish_case(self):
        """Finish the live testcase at the next frame boundary; pause + reset."""
        if not self.testcase:
            raise RuntimeError('End Case requires Testcase mode')
        self._finish_case.set()

    def next_case(self):
        """Continue the same live source with fresh temporal state."""
        if not self.testcase:
            raise RuntimeError('Next Case requires Testcase mode')
        if not self._case_paused:
            return False
        self._resume_case.set()
        return True

    def update_roi(self, index, points):
        if not 0 <= index < len(self.sources):
            raise IndexError(index)
        if len(points) not in (0,) and len(points) < 3:
            return  # incomplete polygon only exists in the visual editor
        if any(len(point) != 2 or not all(0. <= float(v) <= 1. for v in point)
               for point in points):
            raise ValueError('ROI coordinates must be normalized')
        with self._roi_lock:
            self._pending_rois[index] = tuple(tuple(map(float, point)) for point in points)

    def close(self, timeout=8.):
        self.stop.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout)
        return self.thread is None or not self.thread.is_alive()

    def _notify(self, kind, i, payload):
        if not self.stop.is_set() or kind == 'state':
            self.emit(kind, i, payload)

    def _open_sources(self, classifier):
        for i, source in enumerate(self.sources):
            pipe = self.pipeline_factory(self.pose, classifier, device=self.device,
                                         stable_frames=self.settings.stable_frames,
                                         min_conf=self.settings.min_conf, overlay=self.settings.overlay)
            state = StreamState(source, pipe, self.rois.get(i))
            self._states.append(state)  # ensure cleanup even if open fails
            if source.kind == 'camera':
                state.reader = self.reader_factory(source.value, f'cam{i+1:02d}')
            elif source.kind == 'video':
                state.video = cv2.VideoCapture(source.value)
                if not state.video.isOpened():
                    raise OSError('Cannot open video source')
            elif not Path(source.value).is_file():
                raise OSError('Image file does not exist')

    def _next_frame(self, state):
        if state.source.kind == 'image':
            if state.processed:
                return 'done', None
            image = cv2.imread(state.source.value)
            if image is None:
                raise OSError('Cannot decode image')
            return 'frame', image
        if state.source.kind == 'video':
            ok, frame = state.video.read()
            return ('frame', frame) if ok else ('done', None)
        snap = state.reader.snapshot()
        now = time.monotonic()
        if snap is None or now - snap[2] > self.settings.stale_seconds:
            if now - state.last_input > self.settings.stale_seconds and not state.stale_reset:
                state.pipeline.reset()
                state.stale_reset = True
                self._notify('status', self._states.index(state), 'RECONNECTING')
            return 'none', None
        generation, seq, stamp, frame = snap
        if (generation,seq) == state.last_key:
            return 'none', None
        if generation != (state.last_key[0] if state.last_key is not None else None):
            state.reset()  # no previous stable class on reconnect
        state.last_key = (generation,seq)
        state.last_input = now
        state.stale_reset = False
        return 'frame', frame

    def _run(self):
        meter = None
        try:
            classifier = self.classifier_factory(self.model, backend='pt', device=self.cls_device)
            meter = ClassifierForwardMeter(classifier)
            self._open_sources(classifier)
            self._notify('status', -1, 'RUNNING')
            if self.testcase:
                self._notify('case', -1, ('running', self._case_id))
            finished = set()
            last_emit = [0.] * len(self._states)
            while not self.stop.is_set() and len(finished) < len(self._states):
                if self.testcase and self._finish_case.is_set() and not self._case_paused:
                    self._finish_case.clear()
                    self._resume_case.clear()
                    for i, state in enumerate(self._states):
                        state.reset(preserve_key=True)
                        last_emit[i] = 0.
                        self._notify('status', i, 'READY')
                    self._case_paused = True
                    last_emit = [0.] * len(self._states)
                    self._notify('case', -1, ('ready', self._case_id))
                if self.testcase and self._case_paused:
                    if not self._resume_case.is_set():
                        self.stop.wait(.01)
                        continue  # No frame consumed between cases.
                    self._resume_case.clear()
                    self._case_paused = False
                    self._case_id += 1
                    for state in self._states:
                        # Frames captured while paused belong to neither case.
                        # Require a NEW camera frame after START instead of
                        # classifying a buffered frame from the pause interval.
                        if state.source.kind == 'camera' and state.reader is not None:
                            snapshot = state.reader.snapshot()
                            if snapshot is not None:
                                state.last_key = (snapshot[0], snapshot[1])
                        state.reset(preserve_key=True)
                        state.last_input = time.monotonic()
                    last_emit = [0.] * len(self._states)
                    self._notify('case', -1, ('running', self._case_id))
                if self._reset.is_set():
                    self._reset.clear()
                    for i, state in enumerate(self._states):
                        state.reset(preserve_key=True)
                        last_emit[i] = 0.
                        self._notify('status', i, 'READY')
                with self._roi_lock:
                    pending, self._pending_rois = self._pending_rois, {}
                for idx, points in pending.items():
                    if idx < len(self._states):
                        state = self._states[idx]
                        state.roi = list(points)
                        state.reset(preserve_key=True)  # new ROI invalidates temporal state and metrics
                        last_emit[idx] = 0.
                        self._notify('status', idx, 'ROI UPDATED / READY')
                worked = False
                for i, state in enumerate(self._states):
                    if i in finished or self.stop.is_set():
                        continue
                    kind, frame = self._next_frame(state)
                    if kind == 'done':
                        finished.add(i)
                        self._notify('status', i, 'DONE')
                        continue
                    if kind != 'frame':
                        continue
                    worked = True
                    # Post-detection ROI filter is injected, never monkeypatched.
                    filter_people = ((lambda people, shape, roi=state.roi: roi_filter(people, roi, shape))
                                     if state.roi else None)
                    now = time.monotonic()
                    preview_due = (now - last_emit[i] >= 1./self.settings.render_fps
                                   or state.source.kind == 'image')
                    meter.value = None
                    begin = time.perf_counter_ns()
                    info, annotated = state.pipeline.process(
                        frame, detection_filter=filter_people, render_preview=preview_due)
                    ms = (time.perf_counter_ns()-begin)/1e6
                    state.pipeline_fps.add(ms)
                    state.live.tick()
                    if info.get('probs') is not None and meter.value is not None:
                        state.mlp.add(meter.value)
                    predictor = getattr(getattr(state.pipeline.pose, 'engine', None), 'predictor', None)
                    results = getattr(predictor, 'results', None) if predictor is not None else None
                    speed = getattr(results[0], 'speed', None) if results else None
                    if isinstance(speed, dict) and isinstance(speed.get('inference'), (int,float)):
                        state.yolo.add(speed['inference'])
                    state.processed += 1
                    state.case_processed += 1
                    if preview_due:
                        last_emit[i] = time.monotonic()
                        info = dict(info)
                        info['display_label'] = display_decision(
                            info, self.settings.behavior, state.pipeline.classifier.labels)
                        info['case_id'] = self._case_id if self.testcase else None
                        metrics = {'classifier_family': classifier.model_family if hasattr(classifier, 'model_family') else 'mlp',
                                   'yolo': state.yolo.fps(), 'mlp': state.mlp.fps(),
                                   'pipeline': state.pipeline_fps.fps(), 'live': state.live.get(),
                                   'processed': state.processed,
                                   'case_processed': state.case_processed}
                        self._notify('frame', i, (annotated, info, metrics))
                if not worked:
                    self.stop.wait(.01)
            if self.testcase and not self.stop.is_set():
                # Finite file EOF is a natural case boundary. Models are closed below.
                for state in self._states:
                    state.reset(preserve_key=True)
                self._notify('case', -1, ('complete', self._case_id))
            self._notify('status', -1, 'DONE' if not self.stop.is_set() else 'STOPPED')
        except Exception as exc:
            # Error messages may include sensitive camera URLs: do not echo them.
            self._notify('error', -1, f'{type(exc).__name__}: inference failed (check models, source and dependencies)')
        finally:
            try:
                for state in self._states:
                    try:
                        state.close()
                    except Exception:
                        pass  # close remaining sources even if one backend fails
            finally:
                if meter:
                    meter.close()
                self._notify('state', -1, 'READY')


# Setup preview is intentionally separate from the classifier session. It uses the
# existing CameraReader and (optionally) PoseRunner, and never updates testcase state.
def draw_pose_preview(frame, people):
    """Render detector boxes + all visible COCO17 joints; do not classify."""
    out = frame.copy()
    for person in people:
        box = np.asarray(person.get('bbox'), dtype=float)
        if box.shape == (4,) and np.isfinite(box).all():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 175, 30), 2)
        kp = person.get('keypoints')
        if kp is None:
            continue
        points = np.asarray(kp, dtype=float)
        if points.shape != (17, 3):
            continue
        good = np.isfinite(points).all(axis=1) & (points[:, 2] >= .35)
        for a, b in SKELETON:
            if good[a] and good[b]:
                cv2.line(out, tuple(points[a, :2].astype(int)),
                         tuple(points[b, :2].astype(int)), (0, 220, 0), 2)
        for i in range(17):
            if good[i]:
                cv2.circle(out, tuple(points[i, :2].astype(int)), 3, (0, 240, 255), -1)
    return out


class PreviewSession:
    """Non-blocking setup preview. Only ONE selected tile runs YOLO pose at a time.

    All sources can show raw previews simultaneously. The pose runner is loaded
    lazily, once, and does not run concurrently with inference. Video preview
    restarts from the beginning when the real session opens its own reader.
    """
    def __init__(self, sources, pose, device='0', emit=None,
                 reader_factory=CameraReader, pose_factory=PoseRunner, fps=8):
        if not 1 <= len(sources) <= 4:
            raise ValueError('Choose 1–4 sources')
        self.sources = list(sources)
        self.pose, self.device = pose, device
        self.emit = emit or (lambda *_: None)
        self.reader_factory, self.pose_factory = reader_factory, pose_factory
        self.interval = 1. / fps
        self.stop = threading.Event()
        self.thread = None
        self._pose_lock = threading.Lock()
        self._pose_index = -1

    def set_pose_index(self, index):
        if index != -1 and not 0 <= index < len(self.sources):
            raise IndexError(index)
        with self._pose_lock:
            self._pose_index = index

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError('Preview already running')
        self.stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name='poseforge-preview')
        self.thread.start()

    def close(self, timeout=0.):
        self.stop.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout)
        return self.thread is None or not self.thread.is_alive()

    def _run(self):
        opened = []
        pose_runner = None
        active_index = -1
        try:
            for i, src in enumerate(self.sources):
                if src.kind == 'camera':
                    cap = self.reader_factory(src.value, f'preview{i+1:02d}')
                elif src.kind == 'video':
                    cap = cv2.VideoCapture(src.value)
                    if not cap.isOpened():
                        cap.release()
                        raise OSError('Video unavailable')
                elif src.kind == 'image':
                    cap = cv2.imread(src.value)
                    if cap is None:
                        raise OSError('Image unavailable')
                else:
                    raise ValueError('Invalid source type')
                opened.append((src.kind, cap))
            seen = [None] * len(opened)
            while not self.stop.is_set():
                begin = time.monotonic()
                with self._pose_lock:
                    desired = self._pose_index
                if desired != active_index:
                    if pose_runner is not None:
                        pose_runner.reset_tracking()
                    active_index = desired
                for i, (kind, cap) in enumerate(opened):
                    if self.stop.is_set():
                        break
                    if kind == 'camera':
                        snap = cap.snapshot()
                        if snap is None:
                            continue
                        key, frame = (snap[0], snap[1]), snap[3]
                        if seen[i] == key or time.monotonic() - snap[2] > 5.:
                            continue
                        seen[i] = key
                    elif kind == 'image':
                        frame = cap
                    else:
                        ok, frame = cap.read()
                        if not ok:
                            continue  # EOF: keep last preview until Stop Preview
                    if i == active_index:
                        if pose_runner is None:
                            pose_runner = self.pose_factory('yolo', self.pose, video=True, device=self.device)
                        frame = draw_pose_preview(frame, pose_runner.track_frame(frame))
                    self.emit('preview_frame', i, frame)
                self.stop.wait(max(0., self.interval - (time.monotonic()-begin)))
        except Exception as exc:
            # Do not leak RTSP passwords or user paths in GUI error messages.
            self.emit('preview_error', -1, f'{type(exc).__name__}: preview failed (check source/model)')
        finally:
            try:
                for kind, cap in opened:
                    try:
                        if kind == 'camera':
                            cap.close()
                        elif kind == 'video':
                            cap.release()
                    except Exception:
                        pass  # close all remaining readers even if one fails
                if pose_runner is not None:
                    try:
                        pose_runner.close()
                    except Exception:
                        pass
            finally:
                self.emit('preview_state', -1, 'READY')
