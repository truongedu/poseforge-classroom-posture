#!/usr/bin/env python3
"""Read-only dataset validation with optional checksum snapshot comparison."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'collector'))
from posture_collector.dataset_audit import audit_dataset, snapshot_files, compare_snapshots


def main():
    parser = argparse.ArgumentParser(description='Read-only PoseForge dataset validator')
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--deep', action='store_true', help='Check every JSONL keypoint row')
    parser.add_argument('--snapshot', type=Path, help='Write external SHA-256 inventory JSON')
    parser.add_argument('--compare', type=Path, help='Compare against external saved inventory')
    options = parser.parse_args()
    root = options.data.expanduser().resolve()
    report = audit_dataset(root, deep=options.deep, progress=print)
    print(json.dumps({k: v for k, v in report.items() if k != 'warnings'}, indent=2, ensure_ascii=False))
    print(f'[WARN] {len(report["warnings"])} quality warnings (not automatically excluded)')
    current = None
    if options.snapshot or options.compare:
        current = {'dataset_root': str(root), 'files': snapshot_files(root, progress=print)}
    if options.snapshot:
        target = options.snapshot.expanduser().resolve()
        if target == root or target.is_relative_to(root):
            parser.error('Snapshot MUST be outside the input dataset')
        if target.exists():
            parser.error(f'Snapshot exists; refusing overwrite: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(current, indent=2), encoding='utf-8')
        print(f'[OK] Snapshot saved: {target}')
    if options.compare:
        old = json.loads(options.compare.read_text(encoding='utf-8'))
        difference = compare_snapshots(old, current)
        print(json.dumps({'snapshot_diff': difference}, ensure_ascii=False, indent=2))
        if difference['missing'] or difference['changed']:
            print('[BLOCKED] Existing data missing or changed')
            return 2
        print('[OK] Existing files unchanged. New files:', len(difference['added']))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
