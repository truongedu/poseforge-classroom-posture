"""Offline raw-keypoint extraction, normalization and subject-wise export."""
from __future__ import annotations
from pathlib import Path
import csv
import json
import numpy as np
import cv2
from .storage import DataStore, load_json, save_json, now_iso
from .pose import PoseRunner, MEDIAPIPE_TO_COCO13
from .tracking import TrackSelection
from .single_person import SinglePersonSelector

BACKEND_SIZE={'yolo':17,'mediapipe':33}


def read_timestamps(csv_file):
    """Only writer-confirmed frames, not raw camera source indices."""
    if not Path(csv_file).exists():return []
    with open(csv_file,newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def extract_video(video, backend, model, progress=None, *, selected_track_id=None, device='auto', output_dir=None):
    """Re-extract keypoints from unmodified saved video; preserve every frame.
    Runs inference in subprocess/CLI or GUI's separate worker thread.
    """
    video=Path(video);cam=video.name.split('_raw.mp4')[0]
    times=read_timestamps(video.with_name(cam+'_timestamps.csv'))
    destination=(Path(output_dir) if output_dir is not None else video.parent/'keypoints')/f'{cam}_{backend}{BACKEND_SIZE[backend]}_raw.txt'
    destination.parent.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError('Không đọc được video: '+str(video))
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    fps=fps if 1<=fps<=120 else 20.
    frame_count=0;detected=0;multi=0;single=0
    selection_reasons={}
    temp=destination.with_suffix('.writing.txt')
    try:
        with PoseRunner(backend,model,video=True,device=device) as engine, temp.open('w',encoding='utf-8') as file:
            lastms=-1
            selection=TrackSelection()
            causal_selection=SinglePersonSelector()
            if selected_track_id is not None:selection.select(selected_track_id)
            while True:
                ok,frame=cap.read()
                if not ok:break
                h,w=frame.shape[:2]
                if frame_count<len(times):
                    host_ns=int(times[frame_count]['host_recv_monotonic_ns'])
                    baseline=int(times[0]['host_recv_monotonic_ns'])
                    ms=max(0,int((host_ns-baseline)/1e6))
                    utc=times[frame_count]['host_recv_utc']
                else:
                    ms=round(frame_count/fps*1000)
                    utc=None
                    host_ns=None
                # MediaPipe video mode requires monotonically increasing timestamp.
                ms=max(lastms+1,ms)
                lastms=ms
                people=engine.detect(frame,ms)
                tracks=getattr(engine,'last_tracks',[]) if backend=='yolo' else []
                single+=int(len(tracks)==1) if backend=='yolo' else int(len(people)==1)
                if backend=='yolo' and selected_track_id is None:
                    selected, reason = causal_selection.select(tracks)
                    selection_reasons[reason] = selection_reasons.get(reason, 0) + 1
                    state = {'selected_track_id': selected.get('track_id') if selected else None,
                             'selection_state': reason}
                else:
                    selected=selection.observe(tracks) if backend=='yolo' else None
                    reason = None
                    state = selection.snapshot()
                if backend=='yolo':
                    pts=selected.get('keypoints') if selected else None
                    if selected and pts is not None and np.asarray(pts).shape==(17,3):
                        status='ok';detected+=1
                    else:
                        pts=None
                        status=('invalid_keypoints' if selected else
                                'selected_track_missing' if selection.track_id is not None else
                                'ambiguous_multiple_people' if reason == 'ambiguous_candidates' else
                                'invalid_keypoints' if reason == 'invalid_candidates' else
                                'unassigned_track' if tracks else 'not_detected')
                    multi+=int(len(tracks)>1)
                elif len(people)==1:
                    pts=people[0];detected+=1;status='ok'
                elif len(people)>1:
                    pts=None;multi+=1;status='ambiguous_multiple_people'
                else:
                    pts=None;status='not_detected'
                row={'frame_index':frame_count,'relative_ms':ms,'host_utc':utc,
                     'width':w,'height':h,'backend':backend,
                     'person_count':len(tracks) if backend=='yolo' else len(people),'status':status,'keypoints':pts,
                     'world_keypoints':(engine.last_world[0] if backend=='mediapipe' and len(people)==1 and engine.last_world else None)}
                if backend=='yolo':
                    row.update(schema_version=2,camera_id=cam,timestamp=utc if utc is not None else ms,
                        host_recv_monotonic_ns=host_ns,tracker='bytetrack',
                        track_session=engine.tracking_session,persons=tracks,**state,
                        selection_policy='single_person_causal_v1' if selected_track_id is None else 'explicit_track_id',
                        track_id=selected['track_id'] if selected else None,
                        bbox=selected['bbox'] if selected else None,
                        bbox_confidence=selected['bbox_confidence'] if selected else None,
                        keypoints_17=selected['keypoints_17'] if selected else None,
                        keypoint_confidences=selected['keypoint_confidences'] if selected else None)
                file.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                frame_count+=1
                if progress and frame_count%30==0:progress(f'{video.name} · {backend}: {frame_count} frames')
        temp.replace(destination)
    finally:cap.release()
    report={'video':str(video),'backend':backend,'frames':frame_count,'detected_single':single,
            'selected_frames':detected, 'selection_reasons':selection_reasons,
            'selection_policy':'single_person_causal_v1' if selected_track_id is None and backend=='yolo' else 'explicit_or_other',
            'multiple_people':multi,'missing':frame_count-detected if backend=='yolo' else frame_count-detected-multi,
            'raw_keypoints_txt':str(destination),'model_path':str(model),'extracted_at':now_iso()}
    save_json(destination.with_suffix('.summary.json'),report)
    meta_file=video.parent/'metadata.json'
    if output_dir is None and meta_file.exists():
        meta=load_json(meta_file,{})
        entries=meta.setdefault('keypoint_extractions',{})
        entries[f'{cam}:{backend}']={
            'status':'complete','frames':frame_count,
            'relative_path':str(destination.relative_to(video.parent)),
            'detected_single':detected,'ambiguous_multi_person':multi,
            'extracted_at':report['extracted_at']}
        save_json(meta_file,meta)
    return report


def extract_all(root, backend, model, progress=None, only_missing=False, *, selected_tracks=None, device='auto'):
    store=DataStore(root);outputs=[]
    takes=list(store.iter_takes())
    known={cid for _,meta in takes for cid in meta.get('cameras',{})}
    unknown=set(selected_tracks or {})-known
    if unknown:raise ValueError('Offline target IDs reference cameras without recordings: '+', '.join(sorted(unknown)))
    for folder,meta in takes:
        for cam in meta.get('cameras',{}):
            video=folder/f'{cam}_raw.mp4'
            if not video.exists():continue
            txt=folder/'keypoints'/f'{cam}_{backend}{BACKEND_SIZE[backend]}_raw.txt'
            if only_missing and txt.exists():continue
            try:outputs.append(extract_video(video,backend,model,progress,
                selected_track_id=(selected_tracks or {}).get(cam),device=device))
            except Exception as exc:
                if progress:progress(f'ERROR {video}: {exc}')
                outputs.append({'video':str(video),'backend':backend,'error':str(exc)})
    return outputs


def normalize_rows(raw_txt, backend, visibility_threshold=.35):
    """Causal V2 features. Exactly the same class used in live inference.

    Input is immutable raw JSONL; every output depends only on past/current
    frames. This replaces (not aliases) legacy full-clip median normalization.
    """
    from .features import CausalPoseNormalizer, FEATURE_VERSION

    normalizer = CausalPoseNormalizer(confidence_threshold=visibility_threshold)
    features, indices, times, centers, scales = [], [], [], [], []
    with open(raw_txt, encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            kp = row.get('keypoints')
            if backend == 'mediapipe' and kp is not None:
                raw = np.asarray(kp, dtype=np.float32)
                if raw.shape != (33, 5):
                    raise ValueError(f'Expected MediaPipe (33,5), got {raw.shape}')
                kp = raw[MEDIAPIPE_TO_COCO13]
            value, center, scale = normalizer.transform(
                kp, int(row['width']), int(row['height']), backend=backend)
            features.append(value)
            indices.append(int(row['frame_index']))
            times.append(int(row['relative_ms']))
            centers.append(center)
            scales.append(scale)
    if not features:
        raise ValueError('Empty raw keypoints: ' + str(raw_txt))
    return {
        'keypoints': np.stack(features),
        'frame_index': np.asarray(indices, dtype=np.int32),
        'relative_ms': np.asarray(times, dtype=np.int64),
        'centers_px': np.stack(centers),
        'scales_px': np.asarray(scales, dtype=np.float32),
        'feature_version': FEATURE_VERSION,
    }


def build_processed(root, backend, progress=None, *, output_root=None):
    """Write derived V2 features OUTSIDE raw dataset; never replace existing files.

    This is intentionally not exposed in collector UI. Training uses Colab.
    """
    if output_root is None:
        raise ValueError('Explicit --output outside source dataset is required')
    source = Path(root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError('Output must be outside source dataset')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output already exists and is not empty: {output}')
    # Enumerate existing metadata directly: no DataStore initialization and
    # therefore no writes to the input dataset, even if config folders lack.
    def read_only_takes():
        for meta_file in sorted((source / 'recordings').glob('*/*/label_*/take_*/metadata.json')):
            try:
                yield meta_file.parent, load_json(meta_file, {})
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    takes = list(read_only_takes())
    # A single processed output must not silently mix YOLO11n/m extraction.
    provenance = set()
    for folder, meta in takes:
        for cam in meta.get('cameras', {}):
            raw = folder / 'keypoints' / f'{cam}_{backend}{BACKEND_SIZE[backend]}_raw.txt'
            if not raw.is_file():
                continue
            summary = raw.with_suffix('.summary.json')
            if summary.is_file():
                details = load_json(summary, {})
                provenance.add(str(details.get('model_path') or 'unknown').replace('\\', '/').rsplit('/', 1)[-1])
            else:
                provenance.add('unknown')
    if len(provenance) > 1:
        raise ValueError('Mixed pose-extractor checkpoints in raw data: ' + ', '.join(sorted(provenance)))
    done, problems = 0, []
    for folder, meta in takes:
        for cam in meta.get('cameras', {}):
            raw = folder / 'keypoints' / f'{cam}_{backend}{BACKEND_SIZE[backend]}_raw.txt'
            if not raw.is_file():
                continue
            try:
                values = normalize_rows(raw, backend)
                target = output / backend / meta['person_id'] / meta['session_id'] / (
                    f"label_{meta['label']}") / folder.name
                target.mkdir(parents=True, exist_ok=True)
                dest = target / f'{cam}_normalized13.npz'
                if dest.exists():
                    raise FileExistsError(str(dest))
                np.savez_compressed(dest, **values,
                    person_id=meta['person_id'], camera_id=cam, label=int(meta['label']),
                    clip_id=f"{meta['take_id']}_{cam}", backend=backend,
                    raw_path=str(raw.relative_to(source)),
                    pose_extractor=next(iter(provenance), 'unknown'))
                done += 1
                if progress:
                    progress(f'{done} · {meta["person_id"]}/{cam}: {len(values["frame_index"])} frames')
            except Exception as exc:
                problems.append({'file': str(raw), 'error': str(exc)})
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'normalize_report.json', {
        'feature_version': 'pose13_xy_conf_mask_causal_v2',
        'pose_extractor': next(iter(provenance), 'unknown'),
        'processed_clips': done, 'errors': problems})
    return {'processed_clips': done, 'errors': problems}
