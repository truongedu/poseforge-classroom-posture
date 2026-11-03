#!/usr/bin/env python3
"""PoseForge Studio: python main.py [--data DATASET_ROOT]."""
import argparse
from pathlib import Path
from posture_collector.storage import DEFAULT_COLLECTOR_DATA_ROOT


def main():
    parser = argparse.ArgumentParser(description='PoseForge multi-camera posture recording GUI')
    # None distinguishes the implicit default from an intentional --data choice.
    parser.add_argument('--data', default=None)
    args = parser.parse_args()
    old_data = Path(__file__).resolve().parent / 'Real_Posture_Dataset'
    if args.data is None and old_data.exists():
        raise SystemExit('Old collector/Real_Posture_Dataset still exists. '
                         'Use --data PATH to open a specific dataset, or migrate the old '
                         'runtime directory before running with defaults.')
    target = args.data if args.data is not None else str(DEFAULT_COLLECTOR_DATA_ROOT)
    try:
        from posture_collector.ui import launch
    except ImportError as e:
        raise SystemExit('Thiếu PySide6 hoặc OpenCV. Cài: pip install -r requirements.txt\n'+str(e))
    return launch(target)


if __name__ == '__main__':
    raise SystemExit(main())
