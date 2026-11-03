"""Check final PT MLP FP32 probabilities on CPU vs CUDA; no YOLO or retraining.

Optional --features takes an existing N x 52 float32 .npy of normalized V2
features; otherwise deterministic synthetic V2-shaped features are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_features(count: int, seed: int) -> np.ndarray:
    if count < 1:
        raise ValueError('samples must be >= 1')
    rng = np.random.default_rng(seed)
    features = np.zeros((count, 13, 4), np.float32)
    features[:, :, :2] = rng.uniform(-1.5, 1.5, (count, 13, 2))
    features[:, :, 2] = rng.uniform(0.3, 1.0, (count, 13))
    features[:, :, 3] = (rng.random((count, 13)) > .1).astype(np.float32)
    features[:, :, :3] *= features[:, :, 3:4]
    return features.reshape(count, 52)


def load_features(path: Path, limit: int) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 52 or not len(array):
        raise ValueError('Expected non-empty N x 52 V2 features .npy')
    if not np.isfinite(array).all():
        raise ValueError('Input features contain NaN or infinity')
    return np.ascontiguousarray(array[:limit], dtype=np.float32)


def compare(cpu, cuda, features, tolerance=1e-4):
    max_error = 0.0
    disagreements = 0
    for row in features:
        a = np.asarray(cpu.predict(row), np.float32)
        b = np.asarray(cuda.predict(row), np.float32)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid probability output')
        max_error = max(max_error, float(np.max(np.abs(a-b))))
        disagreements += int(np.argmax(a) != np.argmax(b))
    return max_error, disagreements, max_error <= tolerance and disagreements == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classifier', type=Path,
                        default=Path('artifacts/models/mlp_wide/v002'))
    parser.add_argument('--features', type=Path,
                        help='Optional real normalized V2 features, N x 52 .npy')
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--seed', type=int, default=302)
    parser.add_argument('--atol', type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.atol <= 0:
        parser.error('--samples and --atol must be positive')
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA unavailable; run in the PoseForge GPU Conda environment')
    from inference.pipeline import PostureClassifier
    features = (load_features(args.features, args.samples) if args.features else
                make_features(args.samples, args.seed))
    cpu = PostureClassifier(args.classifier, backend='pt', device='cpu')
    gpu = PostureClassifier(args.classifier, backend='pt', device='cuda:0')
    error, disagreements, passed = compare(cpu, gpu, features, tolerance=args.atol)
    kind = 'real V2' if args.features else 'synthetic V2-shaped'
    print(f'[{"PASS" if passed else "FAIL"}] MLP PT FP32 CPU vs CUDA | '
          f'{len(features)} {kind} samples | max |prob diff|={error:.8g} | '
          f'top1 disagreements={disagreements} | atol={args.atol:g}')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
