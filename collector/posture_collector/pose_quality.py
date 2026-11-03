"""Read-only quality inventory for recorded YOLO17; no video decoding needed."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math

from .single_person import SinglePersonSelector


def quality_rows(dataset, progress=None):
    root = Path(dataset).expanduser().resolve(strict=True)
    files = sorted((root/'recordings').glob('*/*/label_*/take_*/keypoints/*_yolo17_raw.txt'))
    output = []
    for i, raw in enumerate(files, 1):
        pid, session, label_dir, take = raw.relative_to(root/'recordings').parts[:4]
        camera = raw.name.split('_yolo17_raw.txt')[0]
        counts = Counter(); selector = SinglePersonSelector()
        with raw.open(encoding='utf-8') as file:
            for idx, line in enumerate(file):
                if not line.strip(): continue
                obj = json.loads(line)
                if obj.get('frame_index') != counts['frames']:
                    raise ValueError(f'Noncontiguous frame index in {raw} at line {idx+1}')
                counts['frames'] += 1
                existing = obj.get('keypoints')
                if obj.get('status') == 'ok' and isinstance(existing,list) and len(existing)==17:
                    counts['selected'] += 1
                    nvalid = 0
                    for p in existing[:13]:
                        try:
                            if len(p)>=3 and math.isfinite(float(p[0])) and math.isfinite(float(p[1])) and float(p[2])>=.35:
                                nvalid+=1
                        except (ValueError,TypeError): pass
                    counts['full13']+=int(nvalid==13)
                    counts['partial13']+=int(0<nvalid<13)
                persons = obj.get('persons') or []
                counts['duplicate_frames'] += int(len(persons)>1)
                selected, why = selector.select(persons)
                counts['projected'] += int(selected is not None)
                counts['recoverable'] += int(selected is not None and existing is None)
                counts['unresolved'] += int(selected is None)
                counts['reason:'+why] += 1
        n=counts['frames']
        output.append(dict(person=pid,session=session,label=int(label_dir[6:]),take=take,camera=camera,
                frames=n,selected=counts['selected'],full13=counts['full13'],partial13=counts['partial13'],
                duplicate_frames=counts['duplicate_frames'],projected=counts['projected'],
                recoverable=counts['recoverable'],unresolved=counts['unresolved'],
                coverage_pct=round(100*counts['selected']/n,1) if n else 0.0,
                projected_pct=round(100*counts['projected']/n,1) if n else 0.0,
                ambiguity=counts['reason:ambiguous_candidates'],
                raw_path=str(raw.relative_to(root))))
        if progress and (i%10==0 or i==len(files)):
            progress(f'Pose quality {i}/{len(files)} clips')
    return output


def aggregate(rows, columns):
    sums = defaultdict(lambda: Counter())
    totals = ('frames','selected','full13','partial13','duplicate_frames','projected','recoverable','unresolved','ambiguity')
    for row in rows:
        result = sums[tuple(row[k] for k in columns)]
        for metric in totals: result[metric] += int(row[metric])
        result['clips'] += 1
    output=[]
    for key, counts in sorted(sums.items()):
        entry=dict(zip(columns,key))
        entry.update({k:counts[k] for k in totals})
        entry['clips']=counts['clips']
        entry['coverage_pct']=round(100*counts['selected']/counts['frames'],1) if counts['frames'] else 0
        entry['projected_pct']=round(100*counts['projected']/counts['frames'],1) if counts['frames'] else 0
        output.append(entry)
    return output


def save_reports(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    levels={'clip':rows,
            'person':aggregate(rows,('person',)),
            'person_label':aggregate(rows,('person','label')),
            'person_label_camera':aggregate(rows,('person','label','camera'))}
    for name, entries in levels.items():
        with (output_dir/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            if entries:
                w=csv.DictWriter(f,fieldnames=entries[0].keys());w.writeheader();w.writerows(entries)
    totals=aggregate(rows,tuple())
    report={'total': totals[0] if totals else {},'by_person':levels['person'],
            'by_person_label':levels['person_label'],'by_person_label_camera':levels['person_label_camera'],
            'clips':rows}
    (output_dir/'quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report
