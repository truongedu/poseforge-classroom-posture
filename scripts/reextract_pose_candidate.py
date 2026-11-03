#!/usr/bin/env python3
"""Safely clone active raw dataset, then re-extract *all* YOLO17 poses on the clone.

The original dataset, curation decisions, and backup are never edited. Each run
has a new version. On failures the candidate is marked INCOMPLETE and may be
inspected, but not treated as a final clean dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'collector'))
from posture_collector.reextract import list_targets, reextract_targets


def new_candidate(base: Path, name: str) -> Path:
    base.mkdir(parents=True,exist_ok=True)
    i=1
    while True:
        dest=base/f'{name}_v{i:03d}'
        try:
            dest.mkdir()
            return dest
        except FileExistsError:i+=1


def copy_raw_without_old_pose(src, dest):
    """Copy full recordings, labels and configs, excluding only old YOLO17 results.

    Re-extract will produce one uniform model provenance; originals stay intact.
    Other raw files (e.g. MediaPipe), videos and timestamps are preserved.
    """
    def ignored(folder, names):
        relative=Path(folder).relative_to(src)
        skip=set()
        if len(relative.parts)==0:
            skip.update({'processed','splits','reports'})
        if relative.name=='keypoints':
            skip.update(n for n in names if n.endswith('_yolo17_raw.txt') or
                        n.endswith('_yolo17_raw.summary.json') or
                        n.endswith('_yolo17_raw.writing.txt'))
        return skip
    shutil.copytree(src,dest,ignore=ignored,dirs_exist_ok=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'artifacts/dataset_working/Real_Posture_Dataset')
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--output-root',type=Path,default=ROOT/'artifacts/dataset_candidates')
    p.add_argument('--apply',action='store_true',help='Actually clone and re-extract; dry-run otherwise')
    args=p.parse_args()
    src=args.data.expanduser().resolve(strict=True)
    model=args.model.expanduser().resolve(strict=True)
    if model.suffix.lower() not in ('.pt','.onnx') or not model.name.endswith(('-pose.pt','-pose.onnx')):
        p.error('Expected a YOLO-Pose checkpoint (.pt or .onnx)')
    out=args.output_root.expanduser().resolve()
    if out==src or out.is_relative_to(src) or src.is_relative_to(out):
        p.error('Candidate destination must not contain source dataset or vice versa')
    planned=list_targets(src,mode='all')
    print(f'[PLAN] Clone {src}\n[MODEL] {model}\n[TARGETS] {len(planned)} camera clips\n[DEST] {out}',flush=True)
    if not args.apply:
        print('[DRY RUN] Use --apply after reviewing benchmark and model quality')
        return
    candidate=new_candidate(out,model.stem + '_' + model.suffix.lstrip('.'))
    dataset=candidate/'Real_Posture_Dataset'
    status=candidate/'INCOMPLETE.txt'
    status.write_text('Re-extraction incomplete; do NOT export this candidate yet.\n')
    try:
        copy_raw_without_old_pose(src,dataset)
        # Curation settings follow the cloned raw, so Export can reuse decisions.
        curation=src.parent/'curation.json'
        if curation.is_file():
            shutil.copy2(curation,candidate/'curation.json')
        targets=list_targets(dataset,mode='missing')
        if len(targets)!=len(planned):
            raise RuntimeError(f'Target mismatch in clone {len(targets)} != {len(planned)}')
        # No raw YOLO17 files exist in this clone; missing-only is append-safe.
        report=reextract_targets(dataset,targets,model,
            backup_root=candidate/'extraction_backups',overwrite=False,
            device=args.device,progress=lambda s:print(s,flush=True))
        (candidate/'reextract_report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
        if report['failed'] or report['skipped'] or report['succeeded'] != len(targets):
            raise RuntimeError(f'Extraction not complete: {report["succeeded"]}/{len(targets)}, failures={report["failed"]}')
        status.unlink()
        (candidate/'candidate_info.json').write_text(json.dumps({
            'source':str(src),'model':str(model),'created_utc':datetime.now(timezone.utc).isoformat(),
            'camera_clips':len(targets),'review_required':True,
            'note':'Use replay on this candidate; raw working and backup are untouched.'
        },indent=2)+'\n')
        print(f'\n[PASS] candidate dataset: {dataset}\n[REVIEW] Replay --data "{dataset}"')
    except BaseException as exc:
        status.write_text(f'Failed/incomplete: {type(exc).__name__}: {exc}\n')
        print(f'[INCOMPLETE] {candidate} | {exc}',file=sys.stderr)
        raise

if __name__=='__main__':
    main()
