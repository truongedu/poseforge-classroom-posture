#!/usr/bin/env python3
"""CLI for safe YOLO17 recovery. Dry-run by default, never changes raw videos."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'collector'))
from posture_collector.reextract import list_targets, reextract_targets


def main():
    ap = argparse.ArgumentParser(description='PoseForge safe YOLO17 re-extraction')
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--mode', choices=['missing','selected','zero','all'], default='missing')
    ap.add_argument('--person')
    ap.add_argument('--session')
    ap.add_argument('--label', type=int)
    ap.add_argument('--take')
    ap.add_argument('--camera')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--apply', action='store_true',help='Run after preview; default is dry-run')
    ap.add_argument('--confirm-overwrite',action='store_true',help='Required for selected mode')
    args = ap.parse_args()
    targets = list_targets(args.data, person=args.person, session=args.session,
        label=args.label, take=args.take, camera=args.camera, mode=args.mode)
    print(f'[PLAN] {len(targets)} camera clips · {args.mode}')
    for target in targets:
        print(' ', target['person'], target['session'], f"L{target['label']}",
              target['take'], target['camera'])
    if not args.apply:
        print('[DRY RUN] Add --apply to execute')
        return 0
    if args.mode in ('selected','zero','all') and not args.confirm_overwrite:
        ap.error('selected mode requires --confirm-overwrite')
    if not targets:
        return 0
    report = reextract_targets(args.data, targets, args.model,
        backup_root=PROJECT/'artifacts'/'reextract_backups',
        overwrite=(args.mode in ('selected','zero','all')), device=args.device, progress=print)
    print(f"[RESULT] {report['succeeded']}/{report['selected']} success, "
          f"{report['failed']} failed · backups {report['backup']}")
    return 1 if report['failed'] else 0


if __name__ == '__main__': raise SystemExit(main())
