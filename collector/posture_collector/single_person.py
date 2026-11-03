"""Causal selection of ONE physical person from YOLO-Pose detections.

A tracker ID is metadata, not a prerequisite for a usable pose. Multiple
nested boxes may describe the same person. When candidates are spatially
separate, abstain unless one is uniquely consistent with the previous box.
"""
from __future__ import annotations
import math


def valid_box(person):
    try:
        box = [float(x) for x in person['bbox']]
        points = person['keypoints']
        if (len(box) != 4 or not all(map(math.isfinite, box)) or
                box[2] <= box[0] or box[3] <= box[1]):
            return None
        if (len(points) != 17 or
                any(len(p) != 3 or not all(math.isfinite(float(x)) for x in p) for p in points)):
            return None
        return box
    except (TypeError, KeyError, ValueError, OverflowError):
        return None


def area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def overlap(box1, box2):
    intersection = (max(0., min(box1[2], box2[2])-max(box1[0], box2[0])) *
                    max(0., min(box1[3], box2[3])-max(box1[1], box2[1])))
    return intersection / max(1., area(box1) + area(box2) - intersection), intersection / max(1., min(area(box1), area(box2)))


class SinglePersonSelector:
    """A deterministic per-stream selector; no future frames, no model inference."""
    def __init__(self, containment=0.83, temporal_iou=0.52, max_gap=8):
        self.containment = containment
        self.temporal_iou = temporal_iou
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        """Forget temporal identity while preserving selector thresholds."""
        self.previous = None
        self.missed = 0
        self.last_reason = 'waiting'

    def select(self, persons):
        persons = persons or []
        usable = [(p, valid_box(p)) for p in persons]
        usable = [(p, b) for p, b in usable if b is not None]
        chosen, reason = None, 'not_detected' if not persons else 'invalid_candidates'
        if len(usable) == 1:
            chosen, reason = usable[0][0], 'single_detection'
        elif len(usable) > 1:
            # Duplicated full/partial body boxes must refer to the same area.
            largest = max(usable, key=lambda item: area(item[1]))
            if len(usable) == len(persons) and all(
                    overlap(largest[1], box)[1] >= self.containment
                    for _, box in usable):
                chosen, reason = largest[0], 'duplicate_boxes'
            elif self.previous is not None and self.missed <= self.max_gap:
                matches = sorted(((overlap(self.previous, box)[0], p) for p, box in usable),
                                 key=lambda pair: pair[0], reverse=True)
                if matches[0][0] >= self.temporal_iou and matches[1][0] < self.temporal_iou * 0.55:
                    chosen, reason = matches[0][1], 'temporal_reacquire'
                else:
                    reason = 'ambiguous_candidates'
            else:
                reason = 'ambiguous_candidates'
        if chosen is not None:
            self.previous = valid_box(chosen)
            self.missed = 0
        else:
            self.missed += 1
            if self.missed > self.max_gap:
                self.previous = None
        self.last_reason = reason
        return chosen, reason
