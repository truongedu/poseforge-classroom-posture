"""Concurrent camera reading and isolated raw-video writers.

Timestamp = host receipt time, NOT verified exposure time. Movie is constant FPS;
sidecar preserves receipt timestamp for VFR/reconstruction/QA. Raw means no GUI
skeleton or labeling burned into frames; video is re-encoded by OpenCV.
"""
from __future__ import annotations
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import csv
import os
import queue
import threading
import time
import traceback
import uuid
import cv2
from .storage import save_json, now_iso

UTC_FMT='%Y-%m-%dT%H:%M:%S.%fZ'

def utc_time():
    return datetime.now(timezone.utc).strftime(UTC_FMT)

class CameraStream:
    def __init__(self, config, core):
        self.config=dict(config)
        self.id=config['id'];self.url=config['url'];self.core=core
        self.stop_event=threading.Event()
        self.thread=None
        self.lock=threading.Lock()
        self.latest=None;self.latest_mono_ns=0;self.latest_utc=None
        self.latest_index=-1;self.fps=0.0;self.error='Disconnected'
        self.last_resolution=None
        self.frame_counter=0
        self.stream_id=uuid.uuid4().hex;self.connection_generation=0

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.stop_event.clear()
        self.thread=threading.Thread(target=self.run,name=f'camera-{self.id}',daemon=True)
        self.thread.start()

    def stop(self):self.stop_event.set()

    def ready(self):
        with self.lock:
            return self.latest is not None and time.monotonic_ns()-self.latest_mono_ns<2_000_000_000

    def snapshot(self,include_generation=False):
        with self.lock:
            result=(None if self.latest is None else self.latest.copy(),self.latest_mono_ns,
                    self.latest_index,self.fps,self.error)
            return result+(f'{self.stream_id}:{self.connection_generation}',self.latest_utc) if include_generation else result

    def run(self):
        while not self.stop_event.is_set():
            source = int(self.url) if str(self.url).isdigit() else str(self.url)
            # Local MP4 path is useful for testing without phones; pace it in
            # real-time rather than reading thousands of frames per second.
            local_file=isinstance(source,str) and Path(source).is_file()
            cap=cv2.VideoCapture(source)
            try:
                if hasattr(cv2,'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,4000)
                if hasattr(cv2,'CAP_PROP_READ_TIMEOUT_MSEC'):
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,3000)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,2)
                if not cap.isOpened():
                    with self.lock:self.error='Cannot connect'
                    self.stop_event.wait(1.7)
                    continue
                with self.lock:
                    self.error='Connected';self.connection_generation+=1
                    self.latest=None;self.latest_mono_ns=0
                last_time=time.monotonic();rolling_fps=None
                file_fps=float(cap.get(cv2.CAP_PROP_FPS)) if local_file else 0
                if not 1<=file_fps<=120:file_fps=20
                next_file_tick=time.monotonic()
                while not self.stop_event.is_set():
                    if local_file:
                        delay=max(0,next_file_tick-time.monotonic())
                        if delay:self.stop_event.wait(delay)
                        next_file_tick=max(next_file_tick+1/file_fps,time.monotonic())
                    ok,frame=cap.read()
                    if not ok or frame is None:
                        with self.lock:self.error='Stream lost; reconnecting'
                        break
                    tick_ns=time.monotonic_ns();stamp=utc_time()
                    tick=tick_ns/1e9
                    inst=1/max(tick-last_time,1e-4)
                    rolling_fps=inst if rolling_fps is None else .85*rolling_fps+.15*inst
                    last_time=tick
                    self.core.ingest(self.id,frame,tick_ns,stamp,self.frame_counter,source=self)
                    with self.lock:
                        self.latest=frame
                        self.latest_mono_ns=tick_ns
                        self.latest_utc=stamp
                        self.latest_index=self.frame_counter
                        self.fps=rolling_fps
                        self.last_resolution=(frame.shape[1],frame.shape[0])
                        self.error='Connected'
                    self.frame_counter+=1
            except Exception as e:
                with self.lock:self.error=f'Capture exception: {e}'
            finally:cap.release()
            if not self.stop_event.is_set():self.stop_event.wait(1.2)
        with self.lock:self.error='Stopped'

class CameraWriter:
    """Writer handles one unmodified camera stream. Limited queue isolates capture.

    On queue overflow, preserve raw files and declare incomplete recording; drops
    are documented rather than silently misrepresented as intact video.
    """
    def __init__(self,folder,cam_config,fps=20.0,max_queue=260):
        self.folder=Path(folder);self.config=dict(cam_config);self.id=cam_config['id']
        self.queue=queue.Queue(maxsize=max_queue)
        self.fps=fps
        self.count=0;self.drops=0;self.size=None;self.error=None
        self.first_ns=None;self.last_ns=None;self.first_utc=None;self.last_utc=None
        self.received=0;self.finalized=False
        self.thread=threading.Thread(target=self.run,name=f'writer-{self.id}',daemon=True)
        self.thread.start()

    def put(self,frame,tick_ns,utc_stamp,source_idx):
        self.received+=1
        try:
            self.queue.put_nowait((frame.copy(),tick_ns,utc_stamp,source_idx))
        except queue.Full:
            self.drops+=1

    def close_and_join(self,timeout=None):
        # Caller must ensure no further producer accesses queue.
        # Do not deadlock if the writer failed while its queue was full.
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.5)
                break
            except queue.Full:
                continue
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise RuntimeError('Video writer did not finish for '+self.id)
        return self.report()

    def run(self):
        pending=self.folder/f'{self.id}_raw.pending.mp4'
        final=self.folder/f'{self.id}_raw.mp4'
        time_path=self.folder/f'{self.id}_timestamps.csv'
        writer=None
        try:
            with time_path.open('w',newline='',encoding='utf-8') as fd:
                csvw=csv.writer(fd)
                csvw.writerow(['writer_frame_index','source_frame_index','host_recv_monotonic_ns','host_recv_utc'])
                while True:
                    item=self.queue.get()
                    if item is None:break
                    frame,tick,stamp,source_idx=item
                    h,w=frame.shape[:2]
                    if writer is None:
                        # MP4 uses fixed output FPS; source timestamps remain in CSV.
                        self.size=(w,h)
                        writer=cv2.VideoWriter(str(pending),cv2.VideoWriter_fourcc(*'mp4v'),self.fps,(w,h))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot open MP4 writer (codec mp4v unavailable?)')
                    if (w,h)!=self.size:
                        self.drops+=1
                        continue
                    writer.write(frame)
                    if self.first_ns is None:
                        self.first_ns=tick;self.first_utc=stamp
                    self.last_ns=tick;self.last_utc=stamp
                    csvw.writerow([self.count,source_idx,tick,stamp])
                    self.count+=1
                    if self.count%60==0:fd.flush()
            if writer is not None:
                writer.release();writer=None
                os.replace(pending,final)
        except Exception as exc:
            self.error=f'{type(exc).__name__}: {exc}'
        finally:
            if writer is not None:writer.release()
            self.finalized=True

    def report(self):
        span=(self.last_ns-self.first_ns)/1e9 if self.first_ns is not None and self.last_ns is not None else 0
        status='ok' if self.count>=2 and not self.error and self.drops==0 else 'needs_review'
        return {'camera_id':self.id,'view':self.config.get('view','unknown'),
                'url':self.config.get('url',''),'frames':self.count,
                'dropped_due_to_overflow_or_resize':self.drops,
                'received_during_recording':self.received,'output_fps':self.fps,
                'resolution':list(self.size) if self.size else None,
                'first_host_received_utc':self.first_utc,'last_host_received_utc':self.last_utc,
                'capture_span_s':round(span,3),'error':self.error,'status':status}

class CollectorCore:
    """Portable backend independent of PySide; deterministic ingest for tests."""
    def __init__(self,datastore,fps=20.0):
        self.store=datastore;self.fps=fps
        self._lock=threading.RLock()
        self.streams={}
        self.active=None
        self.finalizing=False
        self.last_folder=None
        self.last_report=None
        self.finish_thread=None
        self.recording_error=None

    def connect_cameras(self,cameras):
        self.disconnect_cameras()
        for c in cameras:
            if c.get('enabled',True):
                s=CameraStream(c,self)
                self.streams[s.id]=s
                s.start()

    def disconnect_cameras(self):
        if self.active:raise RuntimeError('Dừng ghi trước khi ngắt camera')
        for s in self.streams.values():s.stop()
        for s in self.streams.values():
            if s.thread:s.thread.join(timeout=1.2)
        self.streams={}

    def ingest(self,cid,frame,tick_ns,utc_stamp,source_idx,source=None):
        with self._lock:
            if source is not None and self.streams.get(cid) is not source:
                return
            a=self.active
            if a and cid in a['writers'] and tick_ns>=a['start_ns']:
                a['writers'][cid].put(frame,tick_ns,utc_stamp,source_idx)

    def start(self,pid,session_id,label,camera_configs,require_all_ready=True):
        with self._lock:
            if self.active or self.finalizing:
                raise RuntimeError('Đang ghi hoặc đang hoàn tất take trước')
            selected=[c for c in camera_configs if c.get('enabled',True)]
            if not selected:raise RuntimeError('Chưa có camera đang bật')
            if require_all_ready:
                missing=[c['id'] for c in selected if c['id'] not in self.streams or not self.streams[c['id']].ready()]
                if missing:raise RuntimeError('Camera chưa có frame mới: '+', '.join(missing))
            folder=self.store.create_take(pid,int(label),session_id)
            # Record start is a *single host event*, not hardware shutter sync.
            started_ns=time.monotonic_ns()
            started_utc=now_iso()
            writers={c['id']:CameraWriter(folder,c,self.fps) for c in selected}
            self.active={'folder':folder,'start_ns':started_ns,'start_utc':started_utc,
                         'pid':pid,'session':session_id,'label':int(label),
                         'label_version':self.store.get_labels()['version'],'writers':writers}
            self.recording_error=None
            return folder

    def stop(self):
        with self._lock:
            if not self.active:raise RuntimeError('Chưa ghi hình')
            self.finalizing=True
            a=self.active;self.active=None
            a['end_ns']=time.monotonic_ns()
            a['end_utc']=now_iso()
        self.finish_thread=threading.Thread(target=self._finish,args=(a,),daemon=True)
        self.finish_thread.start()
        return a['folder']

    def _finish(self,a):
        reports={}
        try:
            for cid,w in a['writers'].items():
                reports[cid]=w.close_and_join()
            valid=all(v['status']=='ok' for v in reports.values())
            duration=(a['end_ns']-a['start_ns'])/1e9
            if duration<5 or duration>10:valid=False
            if any(r.get('capture_span_s',0)<4.5 for r in reports.values()):valid=False
            meta={'take_id':f"{a['pid']}_{a['session']}_L{a['label']}_{a['folder'].name}",
                  'person_id':a['pid'],'session_id':a['session'],'label':a['label'],
                  'label_schema_version':a['label_version'],
                  'record_start_utc':a['start_utc'],'record_end_utc':a['end_utc'],
                  'start_host_monotonic_ns':a['start_ns'],'end_host_monotonic_ns':a['end_ns'],
                  'record_command_duration_s':round(duration,3),
                  'synchronization':'host reception timestamps; not exposure-synchronized',
                  'raw_video':'unannotated image frames, transcoded to fixed-fps MP4',
                  'raw_pose':'extract later offline from saved video',
                  'status':'ok' if valid else 'needs_review',
                  'cameras':reports}
            save_json(a['folder']/'metadata.json',meta)
            self.last_report=meta
            self.last_folder=a['folder']
            self.store.export_manifest()
        except Exception as exc:
            self.recording_error=f'{type(exc).__name__}: {exc}'
        finally:
            with self._lock:self.finalizing=False

    def close(self):
        if self.active:self.stop()
        if self.finish_thread and self.finish_thread.is_alive():
            self.finish_thread.join(timeout=30)
            if self.finish_thread.is_alive():
                raise RuntimeError('Video files are still flushing; app will remain open to protect recordings')
        self.disconnect_cameras()
