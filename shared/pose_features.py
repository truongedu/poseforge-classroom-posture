"""Single causal pose-feature contract for MLP now and temporal inference later.

No frame after the current frame is accessed. Pure NumPy so the same file can
be imported by Colab training and Ubuntu deployment without Qt/Ultralytics.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

FEATURE_VERSION = 'pose13_xy_conf_mask_causal_v2'
FEATURE_SHAPE = (13, 4)
CONFIDENCE_THRESHOLD = 0.35


@dataclass
class CausalPoseNormalizer:
    """State belongs to ONE person/track on ONE camera; reset on reconnect."""

    confidence_threshold: float = CONFIDENCE_THRESHOLD
    ema_alpha: float = 0.25
    center: np.ndarray | None = None
    scale: float | None = None

    def reset(self) -> None:
        self.center = None
        self.scale = None

    def transform(self, keypoints, width: int, height: int, backend: str = 'yolo'):
        if width <= 0 or height <= 0:
            raise ValueError('Frame dimensions must be positive')
        xy = np.zeros((13, 2), dtype=np.float32)
        confidence = np.zeros(13, dtype=np.float32)
        valid = np.zeros(13, dtype=bool)
        if keypoints is not None:
            raw = np.asarray(keypoints, dtype=np.float32)
            if backend == 'yolo':
                if raw.shape != (17, 3):
                    raise ValueError(f'Expected YOLO keypoints (17,3), got {raw.shape}')
                xy = raw[:13, :2].copy()
                confidence = raw[:13, 2].copy()
            elif backend == 'mediapipe':
                # COCO-compatible index order is defined in pose.py; pass already
                # remapped MediaPipe 13x5 observations from the calling adapter.
                if raw.shape != (13, 5):
                    raise ValueError(f'Expected mapped MediaPipe keypoints (13,5), got {raw.shape}')
                xy = raw[:, :2].copy() * (width, height)
                confidence = np.minimum(raw[:, 3], raw[:, 4])
            else:
                raise ValueError('Unsupported backend: ' + str(backend))

            valid = (np.isfinite(confidence) & (confidence >= self.confidence_threshold)
                     & np.isfinite(xy).all(axis=1)
                     & (xy >= 0).all(axis=1)
                     & (xy <= (width, height)).all(axis=1))
        xy[~valid] = 0
        confidence = np.where(valid, np.clip(confidence, 0, 1), 0).astype(np.float32)

        shoulders = [idx for idx in (5, 6) if valid[idx]]
        hips = [idx for idx in (11, 12) if valid[idx]]
        if shoulders:
            current_center = xy[shoulders].mean(axis=0)
        elif self.center is not None:
            current_center = self.center.copy()
        elif valid[0]:
            current_center = xy[0].copy()
        else:
            current_center = np.array([width / 2, height / 2], dtype=np.float32)

        if shoulders or valid[0]:
            self.center = current_center.copy()

        candidates = []
        if len(shoulders) == 2:
            candidates.append(float(np.linalg.norm(xy[5] - xy[6])))
        if shoulders and hips:
            candidates.append(float(np.linalg.norm(xy[shoulders].mean(axis=0) - xy[hips].mean(axis=0))) * .75)

        candidates = [v for v in candidates if np.isfinite(v) and v > 2]
        observed = max(8., float(np.median(candidates))) if candidates else None
        if observed is not None:
            self.scale = (observed if self.scale is None else
                          (1 - self.ema_alpha) * self.scale + self.ema_alpha * observed)
        elif self.scale is None:
            self.scale = max(8., height * .18)

        out = np.zeros(FEATURE_SHAPE, dtype=np.float32)
        out[valid, :2] = (xy[valid] - current_center) / self.scale
        out[:, 2] = confidence
        out[:, 3] = valid.astype(np.float32)
        return out, current_center.astype(np.float32), float(self.scale)
