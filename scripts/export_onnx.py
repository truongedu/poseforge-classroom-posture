#!/usr/bin/env python3
"""Export a supported installed FINAL PyTorch model to ONNX with numeric parity.

One CLI for all registered architectures. Extend build_model() for a new family;
unsupported checkpoints fail explicitly (no generic arbitrary .pt assumptions).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.mlp_model import MODEL_REGISTRY, PostureMLP
from shared.pose_features import FEATURE_VERSION


def build_model(ckpt):
    """Architecture dispatch: all registered MLP names share one builder."""
    name = ckpt['model_name']
    if name in MODEL_REGISTRY:
        return PostureMLP(model_name=name, dropout=ckpt['model_config']['dropout'])
    raise ValueError(f'No ONNX exporter registered for model architecture: {name!r}')


def export_yolo(weights: Path, imgsz: int = 640, device: str = 'cpu') -> Path:
    """Export a verified local YOLO-Pose PT using Ultralytics; refuse overwrites."""
    from ultralytics import YOLO
    weights = weights.expanduser().resolve(strict=True)
    if weights.suffix.lower() != '.pt' or not weights.name.endswith('-pose.pt'):
        raise ValueError('Expected a local YOLO-Pose .pt model')
    target = weights.with_suffix('.onnx')
    if target.exists():
        import onnx
        onnx.checker.check_model(str(target))
        print(f'[SKIP] Valid ONNX already exists: {target}')
        return target
    model = YOLO(str(weights))
    if model.task != 'pose':
        raise ValueError('Checkpoint task is not pose')
    exported = Path(model.export(format='onnx', imgsz=imgsz, batch=1,
                                  dynamic=False, simplify=False, nms=False,
                                  device=device)).resolve()
    if exported != target:
        raise RuntimeError(f'Unexpected exported model location: {exported} != {target}')
    if not exported.is_file() or exported.stat().st_size < 1024:
        raise RuntimeError('ONNX export incomplete')
    import onnx
    onnx.checker.check_model(str(exported))
    print(f'[PASS] YOLO-Pose exported: {exported}')
    return exported


def export(model_dir: Path) -> Path:
    import onnx
    import onnxruntime as ort

    folder = model_dir.expanduser().resolve()
    target = folder / 'model.onnx'
    labels_path = folder / 'inference_labels.json'
    report = folder / 'onnx_export.json'
    if any(p.exists() for p in (target, labels_path, report)):
        raise FileExistsError('ONNX export already exists; files will not be overwritten')

    ckpt = torch.load(folder / 'model_final.pt', map_location='cpu', weights_only=True)
    contract = json.loads((folder / 'feature_contract.json').read_text())
    if ckpt.get('stage') != 'final' or ckpt.get('feature_contract') != contract:
        raise ValueError('Expected FINAL checkpoint matching feature_contract.json')
    if (contract.get('version') != FEATURE_VERSION or contract.get('flatten_dim') != 52
            or contract.get('shape_per_frame') != [13, 4]):
        raise ValueError('Unsupported feature contract')
    labels = ckpt['label_names']
    if len(labels) != 6:
        raise ValueError('Expected six posture labels')

    model = build_model(ckpt)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    fd, tmp_name = tempfile.mkstemp(prefix='.onnx-export-', suffix='.onnx', dir=folder)
    os.close(fd)
    tmp = Path(tmp_name)
    created = []
    try:
        torch.onnx.export(
            model, torch.zeros((1, 52), dtype=torch.float32), str(tmp),
            opset_version=17, dynamo=False,
            input_names=['features'], output_names=['logits'],
            dynamic_axes={'features': {0: 'batch'}, 'logits': {0: 'batch'}},
        )
        proto = onnx.load(str(tmp))
        proto.metadata_props.add(key='feature_version', value=FEATURE_VERSION)
        proto.metadata_props.add(key='model_name', value=ckpt['model_name'])
        onnx.save(proto, str(tmp))
        onnx.checker.check_model(str(tmp))
        sess = ort.InferenceSession(str(tmp), providers=['CPUExecutionProvider'])
        assert sess.get_inputs()[0].name == 'features'
        assert sess.get_outputs()[0].name == 'logits'
        max_diff = 0.0
        rng = np.random.default_rng(302)
        with torch.inference_mode():
            for features in (np.zeros((1, 52), np.float32),
                             rng.normal(size=(8, 52)).astype(np.float32)):
                a = model(torch.from_numpy(features)).numpy()
                b = sess.run(['logits'], {'features': features})[0]
                diff = float(np.max(np.abs(a - b)))
                max_diff = max(max_diff, diff)
                if not np.allclose(a, b, rtol=1e-5, atol=1e-5):
                    raise RuntimeError(f'PyTorch/ONNX logits mismatch: {diff:.3g}')
        # Exclusive creation; never overwrite a model version accidentally.
        os.link(tmp, target)
        created.append(target)
        with labels_path.open('x') as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)
            f.write('\n')
        created.append(labels_path)
        with report.open('x') as f:
            json.dump({'model_name': ckpt['model_name'], 'feature_version': FEATURE_VERSION,
                       'onnx_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                       'max_abs_logit_error': max_diff, 'opset': 17,
                       'torch_version': torch.__version__, 'ort_version': ort.__version__}, f, indent=2)
            f.write('\n')
        created.append(report)
        print(f'[PASS] {target}\n[PASS] PT/ONNX parity: max error {max_diff:.2e}')
        return target
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--model', type=Path, help='Trained posture classifier model directory')
    choice.add_argument('--yolo', type=Path, help='YOLO-Pose .pt checkpoint to export')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu', help='Export torch device, e.g. cpu or 0')
    args = parser.parse_args()
    if args.yolo:
        export_yolo(args.yolo, imgsz=args.imgsz, device=args.device)
    else:
        export(args.model)
