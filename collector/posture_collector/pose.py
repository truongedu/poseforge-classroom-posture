"""Optional pose backends. No inference runs in the recording thread.

The local model files must already exist; no online API or auto-download during
inference. Dataset annotation comes from operator's selected clip label only.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import threading
import uuid
import numpy as np
import cv2

COCO_BONES=[(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,6),
            (5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16)]
MEDIAPIPE_TO_COCO13=[0,2,5,7,8,11,12,13,14,15,16,23,24]

# Ultralytics BaseTrack uses a process-global ID allocator. Scope that allocator
# to the model owning this continuous stream, including creation/reset. No IDs
# are invented or remapped: ByteTrack still assigns them. All PoseForge YOLO
# calls use this lock, including offline extraction concurrent with preview.
_TRACK_CALL_LOCK=threading.RLock()


def person_record(index, keypoints, bbox=None, confidence=None, track_id=None):
    """JSON-safe original detector values, with explicit absent confidence/IDs."""
    points=np.asarray(keypoints) if keypoints is not None else None
    valid=points is not None and points.ndim==2 and points.shape[0]==17 and points.shape[1]>=2
    return {'detection_index':index, 'track_id':track_id, 'bbox':bbox,
            'bbox_confidence':confidence,
            'keypoints':points.tolist() if valid else None,
            'keypoints_17':points[:,:2].tolist() if valid else None,
            'keypoint_confidences':points[:,2].tolist() if valid and points.shape[1]>=3 else None,
            'tracking_state':'tracked' if track_id is not None else 'unassigned'}

class PoseRunner:
    def __init__(self,backend,model_file,video=False,device='auto'):
        self.backend=backend.lower();self.model_file=str(Path(model_file).expanduser())
        if not Path(self.model_file).is_file():
            raise FileNotFoundError('Cần có model local: '+self.model_file)
        self.video=video;self.engine=None;self.last_world=[];self.last_tracks=[]
        self.device=device;self.tracking_session=uuid.uuid4().hex
        self._id_count=0;self._raw_people=None
        if self.backend=='yolo':
            try:from ultralytics import YOLO
            except ImportError as e:raise RuntimeError('Install tracking dependencies: pip install -r requirements-pose.txt') from e
            self.engine=YOLO(self.model_file)
            if self.engine.task!='pose':raise ValueError('Choose a YOLO-Pose model (.pt/.onnx), not a detection/classification model')
            # Runs BEFORE Ultralytics' tracking callback, retaining unassigned
            # detections and original detector boxes/keypoints, not Kalman boxes.
            self.engine.add_callback('on_predict_postprocess_end',self._capture_detections)
            self.n=17
        elif self.backend=='mediapipe':
            try:import mediapipe as mp
            except ImportError as e:raise RuntimeError('Cài: pip install mediapipe') from e
            self.mp=mp
            opts=mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.model_file),
                running_mode=(mp.tasks.vision.RunningMode.VIDEO if video else mp.tasks.vision.RunningMode.IMAGE),
                num_poses=2)
            self.engine=mp.tasks.vision.PoseLandmarker.create_from_options(opts)
            self.n=33
        else:raise ValueError('Backend phải là yolo hoặc mediapipe')

    @contextmanager
    def _tracking_scope(self):
        from ultralytics.trackers.basetrack import BaseTrack
        with _TRACK_CALL_LOCK:
            outer_count=BaseTrack._count
            BaseTrack._count=self._id_count
            try:yield
            finally:
                self._id_count=BaseTrack._count
                BaseTrack._count=outer_count

    def _capture_detections(self,predictor):
        self._raw_people=[]
        for result in predictor.results:
            boxes=result.boxes
            if boxes is None:continue
            xyxy=boxes.xyxy.cpu().numpy();scores=boxes.conf.cpu().numpy()
            data=result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
            for index,box in enumerate(xyxy):
                kp=data[index] if data is not None and index<len(data) else None
                self._raw_people.append(person_record(index,kp,box.tolist(),float(scores[index])))

    def track_frame(self,frame_bgr):
        """Official ByteTrack only; one runner belongs to one camera/clip.

        Association uses ByteTrack's detection indices (STrack.idx), never a
        coordinate heuristic. Keypoints are copied before the tracking callback.
        """
        if self.backend!='yolo':raise ValueError('ByteTrack requires the YOLO backend')
        with self._tracking_scope():
            self._raw_people=None
            options={} if self.device=='auto' else {'device':self.device}
            self.engine.track(source=frame_bgr,persist=True,tracker='bytetrack.yaml',
                              conf=.25,verbose=False,classes=[0],**options)
            if self._raw_people is None:
                raise RuntimeError('Ultralytics raw detection callback was not invoked; check the supported version')
            tracker=self.engine.predictor.trackers[0]
            if type(tracker).__name__!='BYTETracker':raise RuntimeError('Only official ByteTrack is permitted')
            for track in tracker.tracked_stracks:
                if track.is_activated and track.frame_id==tracker.frame_id:
                    index=int(track.idx)
                    if not 0<=index<len(self._raw_people):
                        raise RuntimeError('Ultralytics returned an invalid detection index')
                    row=self._raw_people[index]
                    row['track_id']=int(track.track_id);row['tracking_state']='tracked'
            self.last_tracks=self._raw_people
            return self.last_tracks

    def reset_tracking(self):
        """Clear one stream's ByteTrack state while keeping its model loaded."""
        if self.backend=='yolo':
            with self._tracking_scope():
                for tracker in getattr(self.engine.predictor,'trackers',[]):tracker.reset()
                self._id_count=0
            self._id_count=0
        self.last_tracks=[];self._raw_people=None
        self.tracking_session=uuid.uuid4().hex

    def detect(self,frame_bgr,ms=0):
        """Returns list of detected persons with preserved original keypoint units.

        yolo keypoints: [[x_px, y_px, score]] (17 rows).
        mediapipe: [[x_norm, y_norm, z_norm, visibility, presence]] (33 rows).
        """
        if self.backend=='yolo':
            return [row['keypoints'] for row in self.track_frame(frame_bgr) if row['keypoints'] is not None]
        rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        mpimg=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
        res=self.engine.detect_for_video(mpimg,int(ms)) if self.video else self.engine.detect(mpimg)
        self.last_world=[]
        for world in (res.pose_world_landmarks or []):
            self.last_world.append([[float(q.x),float(q.y),float(q.z)] for q in world])
        persons=[]
        for landmarks in res.pose_landmarks:
            persons.append([[float(l.x),float(l.y),float(l.z),
                             float(l.visibility or 0),float(l.presence or 0)] for l in landmarks])
        return persons

    def close(self):
        if self.backend=='mediapipe' and self.engine is not None:
            self.engine.close()
        self.engine=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def pose_to_preview_points(poses,backend,width,height):
    if len(poses)!=1:return []
    a=np.asarray(poses[0],dtype=float)
    if backend=='mediapipe':
        a=a[MEDIAPIPE_TO_COCO13]
        return [(float(x*width),float(y*height),float(min(vis,pres)))
                for x,y,z,vis,pres in a]
    return [tuple(map(float,p)) for p in a]
