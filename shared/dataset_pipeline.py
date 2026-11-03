"""PoseForge Phase 2A: read-only raw YOLO17 catalog and subject-wise split.

Notebook 01 uses this module. It reads JSONL directly from a ZIP; it never
runs pose inference, rewrites recordings, or normalizes coordinates.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

COCO13 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip"]
COLUMNS = ("clip_id", "take_id", "person_id", "session_id", "camera_id",
           "camera_view", "label", "video_path", "metadata_path",
           "duration_s", "frames", "status")


def _json(zf, member):
    return json.loads(zf.read(member).decode("utf-8-sig"))


def _csv(zf, member):
    with zf.open(member) as source:
        return list(csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig")))


def _relative(path):
    """Normalize recorded Windows separators but reject unsafe ZIP members."""
    raw = str(path).replace("\\", "/")
    p = PurePosixPath(raw)
    if not raw or p.is_absolute() or any(part in ("..", "") for part in p.parts) or ":" in p.parts[0]:
        raise ValueError(f"Unsafe dataset path: {path!r}")
    return str(p)


def _member(prefix, relative):
    return prefix + _relative(relative)


def _prefix(zf):
    matches = [n[:-len("labels.csv")] for n in zf.namelist()
               if n.endswith("Real_Posture_Dataset/labels.csv")]
    if len(matches) != 1:
        raise ValueError(f"Expected one Real_Posture_Dataset/labels.csv in ZIP, found {len(matches)}")
    return matches[0]


def scan_raw_zip(zip_path, threshold=0.35):
    """Return clip catalog, frame quality, joint coverage, example skeletons, diagnostics.

    The pose-shape check and 13-point validity check are deliberately separate.
    All CSV/JSONL fields come from the ZIP, not from machine-specific file paths.
    """
    catalog, frame_rows, examples, issues = [], [], {}, []
    joint_counts = defaultdict(lambda: [0, 0, 0.0])  # total, valid, sum confidence
    status_counts = Counter()
    model_counts = Counter()
    with ZipFile(zip_path) as zf:
        prefix = _prefix(zf)
        members = set(zf.namelist())
        labels = _csv(zf, prefix + "labels.csv")
        if not labels or tuple(labels[0]) != COLUMNS:
            raise ValueError("Unexpected labels.csv schema")
        label_schema = _json(zf, prefix + "config/labels.json")
        label_names = {int(d["id"]): d["name"] for d in label_schema["labels"]}
        seen = set()
        for record in tqdm(labels, desc="Scan YOLO17 clip-camera", unit="clip"):
            cid = record["clip_id"]
            if cid in seen:
                issues.append(f"duplicate clip_id: {cid}")
                continue
            seen.add(cid)
            person, camera = record["person_id"], record["camera_id"]
            label = int(record["label"])
            expected = int(record["frames"])
            video = _member(prefix, record["video_path"])
            meta_path = _member(prefix, record["metadata_path"])
            raw = str(PurePosixPath(video).parent / "keypoints" / f"{camera}_yolo17_raw.txt")
            summary_path = str(PurePosixPath(raw).with_suffix(".summary.json"))
            timestamps = str(PurePosixPath(video).parent / f"{camera}_timestamps.csv")
            for member in (video, raw, summary_path, meta_path, timestamps):
                if member not in members:
                    issues.append(f"{cid}: missing {member.removeprefix(prefix)}")
            if any(p not in members for p in (raw, summary_path, meta_path, timestamps)):
                continue
            meta = _json(zf, meta_path)
            cam_meta = meta.get("cameras", {}).get(camera, {})
            src_info = _json(zf, summary_path)
            model_name = str(src_info.get("model_path", "unknown")).replace("\\", "/").split("/")[-1]
            model_counts[model_name] += 1
            if model_name != "yolo11n-pose.pt":
                issues.append(f"{cid}: unexpected pose model {model_name}")
            if str(meta.get("person_id")) != person or str(meta.get("label")) != str(label) or str(meta.get("take_id")) != record["take_id"]:
                issues.append(f"{cid}: CSV / metadata person-label-take mismatch")
            if str(cam_meta.get("status")) != record["status"]:
                issues.append(f"{cid}: CSV / camera status mismatch")
            if int(src_info.get("frames", -1)) != expected:
                issues.append(f"{cid}: extraction summary frame count mismatch")
            timestamp_rows = _csv(zf, timestamps)
            if len(timestamp_rows) != expected:
                issues.append(f"{cid}: timestamp rows {len(timestamp_rows)} != {expected}")
            if timestamp_rows and [int(t["writer_frame_index"]) for t in timestamp_rows] != list(range(len(timestamp_rows))):
                issues.append(f"{cid}: noncontinuous timestamp frame indices")
            capture_span = np.nan
            encoded_span = np.nan
            ratio = np.nan
            fps = float(cam_meta.get("output_fps") or 0)
            if len(timestamp_rows) > 1:
                capture_span = (int(timestamp_rows[-1]["host_recv_monotonic_ns"]) - int(timestamp_rows[0]["host_recv_monotonic_ns"])) / 1e9
            if fps > 0:
                encoded_span = expected / fps
            if capture_span > 0 and math.isfinite(encoded_span):
                ratio = encoded_span / capture_span
            valid_pose_frames = 0
            frames_full13 = 0
            frames_partial13 = 0
            frames_none13 = 0
            raw_rows = 0
            with zf.open(raw) as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except (ValueError, UnicodeError) as exc:
                        issues.append(f"{cid}: JSONL parse error frame {raw_rows}: {exc}")
                        raw_rows += 1
                        continue
                    frame_index = int(entry.get("frame_index", -1))
                    if frame_index != raw_rows:
                        issues.append(f"{cid}: frame index {frame_index} != {raw_rows}")
                    raw_rows += 1
                    status = str(entry.get("status", "unknown"))
                    status_counts[status] += 1
                    width, height = int(entry.get("width") or 0), int(entry.get("height") or 0)
                    keypoints = entry.get("keypoints")
                    shape_ok = False
                    xy_conf = None
                    if keypoints is not None:
                        try:
                            xy_conf = np.asarray(keypoints, dtype=np.float32)
                            shape_ok = xy_conf.shape == (17, 3) and np.isfinite(xy_conf).all()
                        except (ValueError, TypeError):
                            pass
                        if not shape_ok:
                            issues.append(f"{cid}: invalid YOLO17 array at frame {frame_index}")
                    if shape_ok:
                        valid_pose_frames += 1
                        visible = ((xy_conf[:13, 2] >= threshold)
                                   & (xy_conf[:13, 0] >= 0) & (xy_conf[:13, 0] < width)
                                   & (xy_conf[:13, 1] >= 0) & (xy_conf[:13, 1] < height))
                        confidence = xy_conf[:13, 2]
                    else:
                        visible = np.zeros(13, dtype=bool)
                        confidence = np.zeros(13, dtype=np.float32)
                    n_valid = int(visible.sum())
                    if n_valid == 13:
                        frames_full13 += 1
                    elif n_valid == 0:
                        frames_none13 += 1
                    else:
                        frames_partial13 += 1
                    for joint, name in enumerate(COCO13):
                        # all frames count in denominator; no-pose frames also count
                        for key in ((label, camera, name), ("all", "all", name)):
                            stat = joint_counts[key]
                            stat[0] += 1
                            stat[1] += int(visible[joint])
                            stat[2] += float(confidence[joint]) if visible[joint] else 0.0
                    frame_rows.append({
                        "clip_id": cid, "person_id": person, "camera_id": camera,
                        "label": label, "frame_index": frame_index,
                        "status": status, "pose17_shape_ok": int(shape_ok),
                        "valid13_count": n_valid, "shoulders_valid": int(visible[5] and visible[6]),
                        "hips_valid": int(visible[11] and visible[12]),
                        "wrists_valid": int(visible[9] or visible[10]),
                        "relative_ms": entry.get("relative_ms"),
                    })
                    # representative raw skeletons; keep only one per label
                    if shape_ok and status == "ok" and n_valid >= 9:
                        prev = examples.get(label)
                        if prev is None or n_valid > prev["n_valid"]:
                            examples[label] = {"keypoints": xy_conf[:13].tolist(),
                                               "width": width, "height": height,
                                               "person_id": person, "camera_id": camera,
                                               "frame_index": frame_index, "n_valid": n_valid,
                                               "clip_id": cid}
            if raw_rows != expected:
                issues.append(f"{cid}: raw frames {raw_rows} != {expected}")
            catalog.append({
                "clip_id": cid, "take_id": record["take_id"], "person_id": person,
                "session_id": record["session_id"], "label": label,
                "camera_id": camera, "camera_view": record["camera_view"],
                "frames": expected, "raw_rows": raw_rows, "duration_s": float(record["duration_s"]),
                "video_path": _relative(record["video_path"]),
                "raw_member": raw.removeprefix(prefix),
                "timestamp_member": timestamps.removeprefix(prefix),
                "status_camera": record["status"],
                "status_take": meta.get("status", "unknown"),
                "pose_model": model_name, "pose17_frames": valid_pose_frames,
                "full13_frames": frames_full13, "partial13_frames": frames_partial13,
                "none13_frames": frames_none13,
                "pose17_rate": valid_pose_frames / expected if expected else 0.0,
                "full13_rate": frames_full13 / expected if expected else 0.0,
                "capture_span_s": capture_span, "encoded_span_s": encoded_span,
                "encoded_to_capture_ratio": ratio,
                "timebase_discrepancy": int(math.isfinite(ratio) and not (0.85 <= ratio <= 1.15)),
            })
    joints = [{"label": k[0], "camera_id": k[1], "joint": k[2],
               "frames": v[0], "valid_frames": v[1],
               "visibility_rate": v[1] / v[0] if v[0] else 0,
               "mean_conf_valid": v[2] / v[1] if v[1] else 0}
              for k, v in joint_counts.items()]
    return (pd.DataFrame(catalog), pd.DataFrame(frame_rows), pd.DataFrame(joints),
            examples, {"label_names": label_names, "model_counts": dict(model_counts),
                       "status_counts": dict(status_counts), "issues": issues,
                       "raw_zip_root": prefix})


def subject_split(catalog, test_people=None, seed=302):
    """Subject-wise split: 1+ unseen test people, remaining people in train_pool.

    No frame/clip/take from a test person can appear in train_pool.
    Validation is NOT chosen here; it must be drawn from train_pool later.
    """
    people = sorted(catalog["person_id"].unique().tolist())
    if len(people) < 3:
        raise ValueError("Need at least 3 people: test, train and future validation")
    if test_people is None or test_people == "AUTO" or test_people == ["AUTO"]:
        candidates = [p for p in people if set(catalog.loc[catalog.person_id == p, "label"]) == set(range(6))]
        if not candidates:
            raise ValueError("No person with all 6 labels for AUTO test")
        test_people = [candidates[int(np.random.default_rng(seed).integers(len(candidates)))]]
    elif isinstance(test_people, str):
        test_people = [test_people]
    else:
        test_people = list(test_people)

    if not test_people or len(set(test_people)) != len(test_people):
        raise ValueError("TEST_PEOPLE must contain unique person IDs")
    unknown = sorted(set(test_people) - set(people))
    if unknown:
        raise ValueError(f"Unknown test people: {unknown}; available: {people}")
    test_people = sorted(test_people)
    train_people = sorted(set(people) - set(test_people))
    if len(train_people) < 2:
        raise ValueError("Keep at least 2 people in train_pool for internal validation")
    for person in test_people:
        sub = catalog[catalog.person_id == person]
        if set(sub.label) != set(range(6)):
            raise ValueError(f"Test person {person}: missing labels; need all 6")
        usable = sub.groupby("label").pose17_frames.sum().reindex(range(6), fill_value=0)
        if (usable == 0).any():
            raise ValueError(f"Test person {person}: labels with zero pose frames: {usable[usable == 0].index.tolist()}")

    result = catalog[["clip_id", "take_id", "person_id", "session_id", "camera_id", "label"]].copy()
    result["split"] = np.where(result.person_id.isin(test_people), "test", "train_pool")
    assert not (set(result.loc[result.split == "test", "person_id"]) &
                set(result.loc[result.split == "train_pool", "person_id"]))
    assert not (set(result.loc[result.split == "test", "take_id"]) &
                set(result.loc[result.split == "train_pool", "take_id"]))
    info = {"strategy": "subject_wise_multi_test", "seed": int(seed),
            "test_people": test_people, "train_pool_people": train_people,
            "validation_people": [],
            "note": "Use only train_pool for model selection and validation; do not tune on test."}
    return result, info


def file_sha256(path, buffer_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buffer_size), b""):
            h.update(chunk)
    return h.hexdigest()
