#!/usr/bin/env python3
"""Generate recording quality CSV and completeness matrix by human participant."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse,csv,json
from posture_collector.storage import DataStore, DEFAULT_COLLECTOR_DATA_ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=str(DEFAULT_COLLECTOR_DATA_ROOT))
    args=p.parse_args();store=DataStore(args.data)
    results=[];take_count=defaultdict(int);labels=store.get_labels()['labels']
    for folder,m in store.iter_takes():
        take_count[(m['person_id'],int(m['label']))]+=int(m.get('status')=='ok')
        cameras=m.get('cameras',{})
        first=[r.get('first_host_received_utc') for r in cameras.values() if r.get('first_host_received_utc')]
        for cid,r in cameras.items():
            results.append({'person_id':m.get('person_id'),'session_id':m.get('session_id'),
                 'label':m.get('label'),'take':folder.name,'camera':cid,'camera_view':r.get('view'),
                 'frames':r.get('frames'),'duration_seconds':r.get('capture_span_s'),
                 'dropped_or_resized':r.get('dropped_due_to_overflow_or_resize'),
                 'clip_status':m.get('status'), 'camera_status':r.get('status'),
                 'reextract_count':sum(1 for k in m.get('keypoint_extractions',{}) if k.startswith(cid+':'))})
    dst=store.root/'reports'/'recording_quality.csv'
    columns=['person_id','session_id','label','take','camera','camera_view','frames',
             'duration_seconds','dropped_or_resized','clip_status','camera_status','reextract_count']
    with dst.open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=columns);wr.writeheader();wr.writerows(results)
    matrix=store.root/'reports'/'coverage_by_person.csv'
    with matrix.open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.writer(f);wr.writerow(['person_id']+[f'label_{x["id"]}_valid_takes' for x in labels])
        for pid in sorted(store.subjects()):
            wr.writerow([pid]+[take_count[(pid,d['id'])] for d in labels])
    print('Saved:',dst,'and',matrix,'rows:',len(results))

if __name__=='__main__':main()
