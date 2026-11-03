"""File layout, participant consent bookkeeping, and extensible label schema."""
from __future__ import annotations
from pathlib import Path
import csv
import io
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone

ID_RE = re.compile(r'^[A-Z0-9_-]{2,24}$')
CAM_RE = re.compile(r'^[a-zA-Z0-9_-]{2,24}$')
# One persistent collector runtime location; Git ignores everything under artifacts/.
DEFAULT_COLLECTOR_DATA_ROOT = (Path(__file__).resolve().parents[2] / 'artifacts'
                               / 'collector_runtime' / 'Real_Posture_Dataset')
DEFAULT_LABELS = [
    {'id': 0, 'name': 'Ngồi thẳng', 'key': 'sitting_straight'},
    {'id': 1, 'name': 'Cúi gập / gù lưng', 'key': 'hunched_forward'},
    {'id': 2, 'name': 'Gục xuống bàn', 'key': 'head_on_desk'},
    {'id': 3, 'name': 'Ngả người ra sau', 'key': 'leaning_backward'},
    {'id': 4, 'name': 'Nghiêng / quay sang bên', 'key': 'turning_sideways'},
    {'id': 5, 'name': 'Giơ tay phát biểu', 'key': 'raising_hand'},
]
DEFAULT_CAMERAS = [
    {'id':'cam01','name':'DroidCam 01','view':'front','url':'http://192.168.1.101:4747/video','enabled':True},
    {'id':'cam02','name':'DroidCam 02','view':'oblique','url':'http://192.168.1.102:4747/video','enabled':True},
]

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')

def save_json(path: Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def load_json(path: Path, default):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

class DataStore:
    def __init__(self, root):
        self.root=Path(root).expanduser().resolve()
        for name in ('config','recordings','processed','splits','reports'):
            (self.root/name).mkdir(parents=True, exist_ok=True)
        self.subject_file=self.root/'config'/'subjects.json'
        self.labels_file=self.root/'config'/'labels.json'
        self.camera_file=self.root/'config'/'cameras.json'
        if not self.labels_file.exists():
            save_json(self.labels_file, {'version':1,'labels':DEFAULT_LABELS})
        if not self.camera_file.exists():
            save_json(self.camera_file, DEFAULT_CAMERAS)
        if not self.subject_file.exists():
            save_json(self.subject_file, {})

    def subjects(self):
        return load_json(self.subject_file, {})

    def add_subject(self, pid, consent_confirmed, consent_reference=''):
        pid=pid.strip().upper()
        if not ID_RE.fullmatch(pid):
            raise ValueError('ID gồm 2–24 ký tự A-Z, 0-9, _ hoặc -')
        if not consent_confirmed:
            raise ValueError('Cần xác nhận đã có văn bản đồng ý trước khi thu hình')
        subjects=self.subjects()
        if pid in subjects:
            raise ValueError(f'ID {pid} đã có: không tạo trùng người')
        subjects[pid]={'person_id':pid,'consent_written_confirmed':True,
                       'consent_reference':consent_reference.strip(),'created_at_utc':now_iso()}
        save_json(self.subject_file, subjects)
        return subjects[pid]

    def get_labels(self):
        return load_json(self.labels_file, {'version':1,'labels':DEFAULT_LABELS})

    def add_label(self, name, key):
        name=name.strip();key=key.strip().lower().replace(' ','_')
        if not name or not re.fullmatch(r'[a-z][a-z0-9_]{1,49}', key):
            raise ValueError('Tên hoặc key không hợp lệ; key dùng a-z, 0-9, _')
        d=self.get_labels()
        if any(x['key']==key for x in d['labels']):
            raise ValueError('Label key đã tồn tại')
        nextid=max([x['id'] for x in d['labels']],default=-1)+1
        if nextid < 6: nextid=6
        d['labels'].append({'id':nextid,'name':name,'key':key})
        d['version'] += 1
        save_json(self.labels_file,d)
        return nextid

    def cameras(self):
        return load_json(self.camera_file, DEFAULT_CAMERAS)

    def set_cameras(self, configs):
        used=set()
        for c in configs:
            if not CAM_RE.fullmatch(c['id']) or c['id'] in used:
                raise ValueError('Camera ID trùng hoặc không hợp lệ')
            used.add(c['id'])
            if not str(c['url']).strip():
                raise ValueError('URL camera không được để trống')
        save_json(self.camera_file,configs)

    def create_take(self, pid, label_id, session_id=None):
        if pid not in self.subjects():
            raise ValueError('Person ID chưa đăng ký')
        if not self.subjects()[pid].get('consent_written_confirmed', False):
            raise ValueError('Chưa xác nhận đồng ý bằng văn bản')
        if label_id not in [x['id'] for x in self.get_labels()['labels']]:
            raise ValueError('Nhãn không tồn tại')
        # Sessions are explicitly chosen by the user and stored across takes.
        if not session_id:
            session_id='session_'+datetime.now().strftime('%Y%m%d_%H%M')
        if not re.fullmatch(r'[A-Za-z0-9_-]{2,40}',session_id):
            raise ValueError('Session ID chứa ký tự không hợp lệ')
        target=self.root/'recordings'/pid/session_id/f'label_{label_id}'
        target.mkdir(parents=True,exist_ok=True)
        existing=[]
        for child in target.glob('take_*'):
            if child.is_dir() and child.name[5:].isdigit():
                existing.append(int(child.name[5:]))
        n=max(existing,default=0)+1
        while True:
            folder=target/f'take_{n:03d}'
            try:folder.mkdir(exist_ok=False);break
            except FileExistsError:n+=1
        (folder/'keypoints').mkdir()
        return folder

    def iter_takes(self):
        for file in sorted((self.root/'recordings').glob('*/*/label_*/take_*/metadata.json')):
            try:
                d=load_json(file,{})
                yield file.parent,d
            except (OSError,ValueError,json.JSONDecodeError):
                continue

    def export_manifest(self):
        """Add only missing clip records; never rewrite historical CSV rows.

        Old Windows datasets use backslash-separated relative paths and may
        contain curated status values not equal to metadata.json. Preserve those
        bytes, values and ordering. New records inherit the existing path style.
        """
        dest = self.root / 'labels.csv'
        columns = ['clip_id','take_id','person_id','session_id','camera_id','camera_view',
                   'label','video_path','metadata_path','duration_s','frames','status']
        original = dest.read_bytes() if dest.exists() else b''
        existing_ids = set()
        existing_count = 0
        windows_paths = False
        if original:
            with io.TextIOWrapper(io.BytesIO(original), encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != columns:
                    raise ValueError('labels.csv schema mismatch: refusing to overwrite')
                for row in reader:
                    cid = row['clip_id']
                    if not cid or cid in existing_ids:
                        raise ValueError('Duplicate/empty clip_id in labels.csv: refusing to modify')
                    existing_ids.add(cid)
                    existing_count += 1
                    windows_paths |= '\\' in row['video_path']
        fresh = []
        for folder, meta in self.iter_takes():
            for camera_id, report in meta.get('cameras', {}).items():
                video = folder / f'{camera_id}_raw.mp4'
                if not video.is_file():
                    continue
                clip_id = f"{meta.get('take_id')}_{camera_id}"
                if clip_id in existing_ids:
                    continue
                rel_video = video.relative_to(self.root).as_posix()
                rel_metadata = (folder / 'metadata.json').relative_to(self.root).as_posix()
                if windows_paths:
                    rel_video = rel_video.replace('/', '\\')
                    rel_metadata = rel_metadata.replace('/', '\\')
                fresh.append({
                    'clip_id': clip_id, 'take_id': meta.get('take_id'),
                    'person_id': meta.get('person_id'), 'session_id': meta.get('session_id'),
                    'camera_id': camera_id, 'camera_view': report.get('view', 'unknown'),
                    'label': meta.get('label'), 'video_path': rel_video,
                    'metadata_path': rel_metadata,
                    'duration_s': report.get('capture_span_s', 0),
                    'frames': report.get('frames', 0),
                    'status': report.get('status', 'unknown'),
                })
                existing_ids.add(clip_id)
        if not fresh and original:
            return dest, existing_count
        # Keep the original bytes untouched and atomically add only new rows.
        # Also preserve the Windows CSV CRLF convention for newly written rows.
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=columns)
        if not original:
            writer.writeheader()
        writer.writerows(fresh)
        suffix = buffer.getvalue().encode('utf-8')
        if original:
            needs_newline = not original.endswith((b'\n', b'\r'))
            suffix = (b'\r\n' if needs_newline else b'') + suffix
        content = (original + suffix) if original else (b'\xef\xbb\xbf' + suffix)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile('wb', dir=dest.parent, prefix='.labels_',
                                             suffix='.tmp', delete=False) as tmp:
                temp_path = Path(tmp.name)
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(temp_path, dest)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        return dest, existing_count + len(fresh)
