#!/usr/bin/env python3
"""Install official YOLO11n/s-Pose checkpoints and optionally export ONNX.

Only writes missing files under artifacts/pretrained. Restart Collector to refresh
its model dropdown; no dataset is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'artifacts' / 'pretrained'
RELEASE = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/'


def install_variant(variant: str, destination: Path = ASSETS) -> Path:
    from ultralytics import YOLO
    destination.mkdir(parents=True, exist_ok=True)
    name = f'yolo11{variant}-pose.pt'
    target = destination / name
    if not target.is_file():
        tmp = target.with_suffix('.pt.part')
        try:
            print(f'[DOWNLOAD] {RELEASE + name}', flush=True)
            with urlopen(RELEASE + name, timeout=45) as source, tmp.open('xb') as f:
                while chunk := source.read(1024 * 1024):
                    f.write(chunk)
            if tmp.stat().st_size < 100000:
                raise ValueError('Downloaded checkpoint is abnormally small')
            os.link(tmp, target)  # no overwrite on racing installs
        except FileExistsError:
            if not target.is_file():
                raise
        except Exception as exc:
            print(f'[DOWNLOAD FAILED] {exc}\nManual: curl -fL "{RELEASE + name}" -o "{target}"')
            raise
        finally:
            tmp.unlink(missing_ok=True)
    if YOLO(str(target)).task != 'pose':
        raise ValueError(f'Not a pose model: {target}')
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    print(f'[OK] {target} | sha256={sha}')
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', nargs='+', choices=['n', 's'], default=['n', 's'])
    p.add_argument('--export-onnx', action='store_true')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--export-device', default='cpu', help='cpu or 0')
    args = p.parse_args()
    for variant in args.variants:
        pt = install_variant(variant)
        if args.export_onnx:
            from export_onnx import export_yolo
            export_yolo(pt, imgsz=args.imgsz, device=args.export_device)
    print('[OK] Restart Collector to refresh the model selector')


if __name__ == '__main__':
    main()
