#!/usr/bin/env python3
"""Read-only summary and counterfactual selection quality, without re-running YOLO."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collector'))
from posture_collector.pose_quality import quality_rows,save_reports

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
rows=quality_rows(a.data,progress=print)
report=save_reports(rows,a.output)
t=report.get('total',{})
print(f"Clips: {t.get('clips',0)}, selected: {t.get('selected',0)}/{t.get('frames',0)}, "
      f"projected (causal selection): {t.get('projected',0)}, recoverable: {t.get('recoverable',0)}")
print('Reports:',a.output.resolve())
