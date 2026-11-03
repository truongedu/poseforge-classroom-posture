"""Backward-compatible collector API; canonical V2 features live in shared/."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from shared.pose_features import (  # noqa: E402
    CausalPoseNormalizer, CONFIDENCE_THRESHOLD, FEATURE_SHAPE, FEATURE_VERSION,
)

__all__ = ['CausalPoseNormalizer', 'CONFIDENCE_THRESHOLD', 'FEATURE_SHAPE', 'FEATURE_VERSION']
