#!/usr/bin/env python3
"""One-command local clean ZIP export. Does not modify dataset_working or backup."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'collector'))
from posture_collector.curation import export_clean_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Materialize curated poses/videos to an immutable clean raw ZIP')
    parser.add_argument('--data', type=Path,
                        default=ROOT/'artifacts/dataset_working/Real_Posture_Dataset')
    args = parser.parse_args()
    result = export_clean_dataset(args.data, progress=print)
    print(f"[OK] {result['clips']} clips | {result['takes']} takes | {result['frames']} frames")
    print(f"[ZIP] {result['zip']}")


if __name__ == '__main__':
    main()
