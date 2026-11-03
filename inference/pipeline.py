"""PoseForge: YOLO+ByteTrack, single-person selection and causal V2 classifier."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from collector.posture_collector.pose import PoseRunner
from collector.posture_collector.single_person import SinglePersonSelector
from shared.pose_features import CausalPoseNormalizer, FEATURE_VERSION

SKELETON = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12))
REQUIRED_JOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                   'left_wrist', 'right_wrist', 'left_hip', 'right_hip']
# Distinct BGR colors, ordered by class ID. OpenCV does the chart drawing; no matplotlib.
CLASS_COLORS = ((225, 159, 70), (72, 183, 239), (175, 118, 241),
                (96, 191, 108), (222, 129, 200), (83, 204, 224))
SHORT_LABELS = ('UP', 'HUNCH', 'DESK', 'BACK', 'SIDE', 'HAND')


class PostureClassifier:
    """Classify causal PoseForge V2 features; MLP default, GRU selectable."""

    def __init__(self, model_dir: Path, backend='pt', device='cpu'):
        model_dir = Path(model_dir)
        contract = json.loads((model_dir / 'feature_contract.json').read_text())
        if (contract.get('version') != FEATURE_VERSION or contract.get('flatten_dim') != 52
                or contract.get('shape_per_frame') != [13, 4]
                or contract.get('channels') != ['x_norm', 'y_norm', 'confidence', 'valid_mask']
                or contract.get('joints') != REQUIRED_JOINTS):
            raise ValueError('Model feature contract incompatible with shared.pose_features')
        self.conf_thresh = float(contract['confidence_threshold'])
        self.ema_alpha = float(contract['ema_alpha'])
        if backend not in ('auto', 'pt', 'onnx'):
            raise ValueError('Backend must be auto, pt, or onnx')
        self.backend = ('onnx' if (model_dir / 'model.onnx').is_file() else 'pt') if backend == 'auto' else backend
        self.device = device
        self.model_family = 'mlp'
        self.sequence_length = 1
        if self.backend == 'onnx':
            import onnxruntime as ort
            labels_path = model_dir / 'inference_labels.json'
            if not labels_path.is_file():
                raise FileNotFoundError('Missing inference_labels.json; run export_onnx.py first')
            self.labels = json.loads(labels_path.read_text())
            available = ort.get_available_providers()
            if device.startswith('cuda') and 'CUDAExecutionProvider' not in available:
                raise RuntimeError('ONNX CUDA provider unavailable; choose --cls-device cpu or install onnxruntime-gpu')
            providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.startswith('cuda')
                         else ['CPUExecutionProvider'])
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(model_dir / 'model.onnx'),
                                                 sess_options=opts, providers=providers)
            if (self.session.get_inputs()[0].name != 'features'
                    or self.session.get_outputs()[0].name != 'logits'):
                raise ValueError('Unexpected ONNX model signature')
        else:
            import torch
            from shared.mlp_model import PostureMLP
            ckpt = torch.load(model_dir / 'model_final.pt', weights_only=True, map_location='cpu')
            if ckpt.get('stage') != 'final' or ckpt['feature_contract'] != contract:
                raise ValueError('PT checkpoint feature contract mismatch')
            self.labels = ckpt['label_names']
            self.model_family = str(ckpt.get('model_family', 'mlp'))
            if self.model_family == 'gru':
                from shared.gru_model import PostureGRU
                settings = ckpt['model_config']
                self.sequence_length = int(settings['sequence_length'])
                if not 1 <= self.sequence_length <= 128:
                    raise ValueError('Invalid GRU sequence_length')
                self.model = PostureGRU(hidden_size=settings['hidden_size'],
                    num_layers=settings['num_layers'],dropout=settings['dropout'])
            elif self.model_family == 'mlp':
                self.model = PostureMLP(ckpt['model_name'], ckpt['model_config']['dropout'])
            else:
                raise ValueError('Unknown classifier family')
            self.model.load_state_dict(ckpt['model_state_dict'], strict=True)
            if device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable for PT classifier; choose --cls-device cpu')
            self.model = self.model.to(device).eval()
        if len(self.labels) != 6:
            raise ValueError('Expected six label names')

    def predict(self, features):
        if self.model_family == 'gru':
            a = np.asarray(features, np.float32)
            if a.shape != (self.sequence_length, 52):
                raise ValueError('GRU input must be a left-padded causal window [length,52]')
            a = a.reshape(1, self.sequence_length, 52)
        else:
            a = np.asarray(features, np.float32).reshape(1, 52)
        if self.backend == 'onnx':
            logits = self.session.run(['logits'], {'features': a})[0][0]
        else:
            import torch
            with torch.inference_mode():
                logits = self.model(torch.from_numpy(a).to(self.device)).cpu().numpy()[0]
        logits = logits - np.max(logits)
        exps = np.exp(logits)
        return exps / exps.sum()


class StableClass:
    def __init__(self, window=5, min_conf=.55):
        self.window = deque(maxlen=window)
        self.min_conf = min_conf

    def reset(self):
        self.window.clear()

    def update(self, probs):
        index = int(np.argmax(probs))
        self.window.append(index if float(probs[index]) >= self.min_conf else -1)
        if (len(self.window) == self.window.maxlen
                and self.window[0] >= 0 and len(set(self.window)) == 1):
            return self.window[0]
        return None


class PosePipeline:
    def __init__(self, pose_file, classifier, device='auto', stable_frames=5, min_conf=.55,
                 overlay=True):
        self.pose = PoseRunner('yolo', pose_file, video=True, device=device)
        self.classifier = classifier
        self.overlay = overlay
        self.normalizer = CausalPoseNormalizer(classifier.conf_thresh, classifier.ema_alpha)
        self.selector = SinglePersonSelector()
        self.stable = StableClass(stable_frames, min_conf)
        self.last_track_id = None
        self.missing = 0
        self.sequence = (deque(maxlen=int(classifier.sequence_length))
                         if getattr(classifier, 'model_family', 'mlp') == 'gru' else None)

    def _lost_pose(self):
        self.missing += 1
        self.stable.reset()
        if self.sequence is not None:
            self.sequence.clear()
        if self.missing == 9:
            # Long loss invalidates temporal identity even when a box exists
            # but too few joints are usable. Preserve the selector instance so
            # benchmark instrumentation and future runtime hooks stay attached.
            self.normalizer.reset()
            self.selector.reset()
            self.last_track_id = None

    def reset(self):
        self.pose.reset_tracking()
        self.selector.reset()
        self.normalizer.reset()
        self.stable.reset()
        self.last_track_id = None
        self.missing = 0
        if self.sequence is not None:
            self.sequence.clear()

    def close(self):
        self.pose.close()

    def process(self, frame, *, detection_filter=None, render_preview=None):
        """Filter after detection; optionally skip drawing on non-preview frames.

        render_preview=None retains the existing CLI/benchmark overlay behavior.
        Rendering never changes detector input or feature/classification logic.
        """
        do_overlay = self.overlay and render_preview is not False
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        people = self.pose.track_frame(frame)
        if detection_filter is not None:
            people = detection_filter(people, frame.shape)
        selected, reason = self.selector.select(people)
        kp = None if selected is None else selected.get('keypoints')
        track_id = None if selected is None else selected.get('track_id')
        info = {'class_id': None, 'label': None, 'score': 0.0, 'stable_id': None,
                'probs': None, 'keypoints_valid': 0, 'selection_reason': reason,
                'track_id': track_id}
        if kp is None:
            self.normalizer.transform(None, w, h)
            self._lost_pose()
            out = draw_overlay(frame, info, None) if do_overlay else frame
            info['inference_ms'] = round((time.perf_counter() - t0) * 1000, 2)
            return info, out

        if (track_id is not None and self.last_track_id is not None
                and track_id != self.last_track_id):
            self.normalizer.reset()
            self.stable.reset()
            if self.sequence is not None:
                self.sequence.clear()
        if track_id is not None:
            self.last_track_id = track_id
        features, _, _ = self.normalizer.transform(kp, w, h)
        info['keypoints_valid'] = int(features[:, 3].sum())
        if info['keypoints_valid'] >= 4:
            self.missing = 0
            if self.sequence is not None:
                self.sequence.append(features.reshape(52).copy())
                # Causal left padding: never read future frames or a previous case.
                samples = np.zeros((self.sequence.maxlen, 52), dtype=np.float32)
                samples[-len(self.sequence):] = np.stack(tuple(self.sequence))
                probs = self.classifier.predict(samples)
            else:
                probs = self.classifier.predict(features)
            index = int(np.argmax(probs))
            info.update(class_id=index, label=self.classifier.labels[index],
                        score=float(probs[index]), stable_id=self.stable.update(probs),
                        probs=[float(p) for p in probs])
        else:
            # Preserve the shared causal transform for feature parity, but do
            # not classify sparse poses or let them keep a stable prediction.
            self._lost_pose()
        out = draw_overlay(frame, info, kp) if do_overlay else frame
        info['inference_ms'] = round((time.perf_counter() - t0) * 1000, 2)
        return info, out


def draw_overlay(frame, info, kp):
    out = frame.copy()
    if kp is not None:
        pts = np.asarray(kp, np.float32)
        if pts.shape == (17, 3):
            ok = np.isfinite(pts).all(axis=1) & (pts[:, 2] >= .35)
            for i, j in SKELETON:
                if ok[i] and ok[j]:
                    cv2.line(out, tuple(pts[i, :2].astype(int)), tuple(pts[j, :2].astype(int)),
                             (0, 220, 0), 2)
            for i in range(13):
                if ok[i]:
                    cv2.circle(out, tuple(pts[i, :2].astype(int)), 3, (0, 240, 255), -1)
    return out


def draw_dashboard(frame, info, labels, fps=0., status='LIVE'):
    """Compact live window: FPS, chosen class/confidence, six distinct probability columns."""
    h, w = frame.shape[:2]
    # Bound GUI width without resizing source data passed to YOLO or saved events.
    factor = min(1., 960. / w, 700. / h)
    if factor < 1.:
        frame = cv2.resize(frame, (max(1, round(w * factor)), max(1, round(h * factor))),
                           interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
    panel_w, canvas_h = 360, max(h, 450)
    canvas = np.full((canvas_h, w + panel_w, 3), (28, 29, 33), np.uint8)
    canvas[:h, :w] = frame
    x = w + 18
    def line(text, y, size=.65, color=(235, 235, 235), thick=2):
        cv2.putText(canvas, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, thick, cv2.LINE_AA)

    line('POSEFORGE  /  LIVE', 38, .69)
    state_color = (88, 210, 118) if status == 'LIVE' else (70, 160, 240)
    line(f'{status}   FPS {fps:5.1f}', 75, .62, state_color)

    probs = info.get('probs')
    cls = info.get('class_id')
    stable = info.get('stable_id')
    if probs is None:
        line('NO POSE / NO PREDICTION', 125, .55, (90, 175, 245))
        line('Confidence: --', 160, .56)
    else:
        decision = int(stable) if stable is not None else int(cls)
        label = str(labels[decision])
        state = 'STABLE' if stable is not None else 'WAITING'
        label = label[:25]
        line(f'{state}: {label}', 125, .55, CLASS_COLORS[decision])
        line(f'Confidence: {float(info["score"])*100:5.1f}%', 160, .58)
    line('CLASS PROBABILITIES', 208, .54, (180, 180, 180), 1)

    chart_left, top, bottom = x + 5, 235, 370
    cv2.line(canvas, (chart_left, bottom), (w + panel_w - 18, bottom), (92, 92, 92), 1)
    values = np.asarray(probs if probs is not None else np.zeros(6), np.float32)
    bar_width, step = 32, 50
    for i in range(6):
        bx = chart_left + i * step + 3
        height = int(np.clip(values[i], 0, 1) * (bottom - top))
        if height:
            cv2.rectangle(canvas, (bx, bottom - height), (bx + bar_width, bottom - 1),
                          CLASS_COLORS[i], -1)
        cv2.putText(canvas, f'{values[i]*100:.0f}', (bx + 1, max(top - 6, bottom - height - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, SHORT_LABELS[i], (bx - 4, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, CLASS_COLORS[i], 1, cv2.LINE_AA)
    line('Press Q to quit', 428, .52, (145, 145, 145), 1)
    return canvas
