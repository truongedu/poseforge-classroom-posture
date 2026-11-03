"""Read-only video/pose lookup for Replay & Inspect. No DataStore writes."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from .pose import COCO_BONES


def take_catalog(root):
    """Build a sorted catalog from existing takes; paths always remain under root."""
    root = Path(root).expanduser().resolve(strict=True)
    result = []
    for meta_file in sorted((root / 'recordings').glob('*/*/label_*/take_*/metadata.json')):
        folder = meta_file.parent.resolve()
        if not folder.is_relative_to(root / 'recordings'):
            continue
        parts = folder.relative_to(root / 'recordings').parts
        if len(parts) != 4:
            continue
        person, session, label_dir, take = parts
        if not label_dir.startswith('label_'):
            continue
        try:
            label = int(label_dir.split('_', 1)[1])
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        for mp4 in sorted(folder.glob('*_raw.mp4')):
            camera = mp4.name[:-len('_raw.mp4')]
            if camera not in meta.get('cameras', {}):
                continue
            raw = folder / 'keypoints' / f'{camera}_yolo17_raw.txt'
            result.append({'person': person, 'session': session, 'label': label,
                           'take': take, 'camera': camera, 'video': mp4,
                           'raw': raw, 'status': meta.get('status', 'unknown')})
    return result


def read_pose_jsonl(raw):
    """Frame-indexed YOLO17 overlay rows, preserving missing/non-ok rows."""
    path = Path(raw)
    if not path.is_file():
        return {}, 'Missing raw keypoints; video can still be replayed'
    rows = {}
    try:
        with path.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get('backend') != 'yolo':
                    raise ValueError('Not a YOLO raw file')
                frame = int(record['frame_index'])
                if frame in rows or frame < 0:
                    raise ValueError(f'Duplicate/invalid frame index at line {number}')
                rows[frame] = record
    except (ValueError, OSError, TypeError, KeyError) as exc:
        return {}, f'Cannot load raw overlay: {exc}'
    return rows, f'{len(rows)} keypoint frames loaded'


def draw_pose_overlay(frame, row, *, draw_bbox=True, draw_skeleton=True, all_people=False,
                      confidence_threshold=0.35):
    """Draw stored poses, not newly inferred poses; returns a copy of BGR frame."""
    import math
    image = frame.copy()
    if not row:
        return image
    people = row.get('persons') or []
    if all_people:
        tracks = people
    else:
        # Draw only the selected person. If selection is ambiguous, avoid falsely
        # presenting another person's detection as the correct labeled subject.
        tracks = [dict(bbox=row.get('bbox'), keypoints=row.get('keypoints'),
                       track_id=row.get('track_id'))] if row.get('keypoints') is not None else []
    for person in tracks:
        pts = person.get('keypoints')
        bbox = person.get('bbox')
        color = (42, 225, 246) if not all_people else (255, 175, 55)
        if draw_bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            if all(isinstance(x, (int, float)) and math.isfinite(x) for x in bbox):
                x1,y1,x2,y2 = (int(x) for x in bbox)
                cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
                tid = person.get('track_id')
                if tid is not None:
                    cv2.putText(image, f'ID {tid}', (x1,max(18,y1-7)),
                                cv2.FONT_HERSHEY_SIMPLEX, .62, color, 2, cv2.LINE_AA)
        if not draw_skeleton or not isinstance(pts, list) or len(pts) != 17:
            continue
        valid = []
        for point in pts:
            try:
                x,y,conf = point[:3]
                valid.append((int(x),int(y)) if all(math.isfinite(v) for v in (x,y,conf))
                             and conf >= confidence_threshold else None)
            except (TypeError, ValueError):
                valid.append(None)
        for a,b in COCO_BONES:
            if valid[a] and valid[b]:
                cv2.line(image, valid[a], valid[b], color, 2, cv2.LINE_AA)
        for point in valid:
            if point: cv2.circle(image, point, 3, (96, 250, 112), -1, cv2.LINE_AA)
    return image
