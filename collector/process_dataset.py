#!/usr/bin/env python3
"""Offline CLI, separate from GUI. All extraction uses saved raw MP4."""
from pathlib import Path
import argparse
import json
from posture_collector.process import extract_all, build_processed
from posture_collector.storage import DEFAULT_COLLECTOR_DATA_ROOT
from posture_collector.tracking import parse_track_choices

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default=str(DEFAULT_COLLECTOR_DATA_ROOT))
    sub=p.add_subparsers(dest='cmd',required=True)
    e=sub.add_parser('extract');e.add_argument('--backend',choices=['yolo','mediapipe'],required=True)
    e.add_argument('--model',required=True,help='LOCAL .pt or .task model file');e.add_argument('--only-missing',action='store_true')
    e.add_argument('--track-id',action='append',default=[],metavar='CAMERA=ID',help='Select an offline ByteTrack ID per camera; live IDs are independent')
    e.add_argument('--overwrite',action='store_true',help='Explicitly overwrite existing extracted keypoints')
    e.add_argument('--device',default='auto',help='auto, cpu, or CUDA device index (0)')
    n=sub.add_parser('normalize');n.add_argument('--backend',choices=['yolo','mediapipe'],required=True)
    n.add_argument('--output',required=True,help='Non-empty destination outside raw dataset is forbidden')
    sub.add_parser('audit')
    a=p.parse_args()
    if a.cmd=='extract':
        try:
            tracks=parse_track_choices(' '.join(a.track_id))
        except ValueError:p.error('--track-id must be CAMERA=positive_integer, e.g. cam01=1')
        result=extract_all(a.data,a.backend,a.model,print,not a.overwrite,selected_tracks=tracks,device=a.device)
    elif a.cmd=='normalize':result=build_processed(a.data,a.backend,print,output_root=a.output)
    else:
        from posture_collector.dataset_audit import audit_dataset
        result=audit_dataset(a.data)
    print(json.dumps(str(result),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
