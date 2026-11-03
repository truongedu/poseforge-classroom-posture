"""Selection and bounded live tracking output. No tracking/stabilization math."""
from collections import OrderedDict
from pathlib import Path
import json
import queue
import threading

from .storage import save_json
from .storage import CAM_RE
from .single_person import SinglePersonSelector


def parse_track_choices(text):
    """Parse optional explicit OFFLINE camera=track choices, never live IDs."""
    choices={}
    for entry in text.replace(',',' ').split():
        try:
            camera,number=entry.split('=',1);number=int(number)
            if not CAM_RE.fullmatch(camera) or number<1 or camera in choices:raise ValueError
        except ValueError as exc:
            raise ValueError('Use camera=positive_ID, e.g. cam01=1, cam02=2; each camera once') from exc
        choices[camera]=number
    return choices


class TrackSelection:
    """Latch a candidate once; never follow a different ID after target loss."""
    def __init__(self,auto=True,single_person=False):
        self.single_person=single_person
        self.causal=SinglePersonSelector() if single_person else None
        self.track_id=None;self.auto=auto;self.state='waiting' if auto else 'reselect_required'

    def select(self,track_id):
        self.track_id=track_id;self.auto=False
        self.state='selected' if track_id is not None else 'unselected'

    def observe(self,persons):
        if self.single_person and self.auto:
            selected, reason = self.causal.select(persons)
            self.track_id = selected.get('track_id') if selected else None
            self.state = reason
            return selected
        if self.track_id is None and self.auto:
            if len(persons)>1:
                self.auto=False;self.state='selection_required'
            elif len(persons)==1 and persons[0]['track_id'] is not None:
                self.track_id=persons[0]['track_id'];self.auto=False
        selected=next((p for p in persons if self.track_id is not None and p['track_id']==self.track_id),None)
        if self.track_id is not None:self.state='selected' if selected else 'missing'
        elif self.auto:self.state='unassigned' if persons else 'waiting'
        return selected

    def snapshot(self):
        return {'selected_track_id':self.track_id,'selection_state':self.state}


class LatestFrames:
    """One newest frame per registered camera, FIFO camera fairness, bounded RAM."""
    def __init__(self):
        self.pending=OrderedDict();self.condition=threading.Condition()

    def put(self,camera,item):
        with self.condition:
            self.pending[camera]=item;self.condition.notify()

    def get(self,timeout=.2):
        with self.condition:
            if not self.pending:self.condition.wait(timeout)
            if not self.pending:raise queue.Empty
            return self.pending.popitem(last=False)

    def discard(self,camera):
        with self.condition:self.pending.pop(camera,None)

    def qsize(self):
        with self.condition:return len(self.pending)


class TrackingJournal:
    """Sampled preview tracks/selection sidecar, never a source of raw video.

    The worker owns disk IO. Dropped samples/errors are explicit in the final
    sidecar. Full-frame chronological extraction remains the offline pipeline.
    """
    def __init__(self,folder,identity):
        self.folder=Path(folder);self.identity=identity;self.queue=queue.Queue(maxsize=256)
        self.dropped=0;self.error=None;self.summary={}
        self.thread=threading.Thread(target=self.run,name='tracking-journal',daemon=True)
        self.thread.start()

    def put(self,camera,row):
        try:self.queue.put_nowait((camera,row))
        except queue.Full:self.dropped+=1

    def finish(self):
        # Called off the Qt thread; raw writers do not depend on this queue.
        while self.thread.is_alive():
            try:self.queue.put(None,timeout=.2);break
            except queue.Full:continue
        self.thread.join()

    def run(self):
        files={}
        try:
            while True:
                item=self.queue.get()
                if item is None:break
                camera,row=item
                if camera not in files:
                    files[camera]=(self.folder/'keypoints'/f'{camera}_yolo17_live_tracks.txt').open('w',encoding='utf-8')
                files[camera].write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                self.summary[camera]=dict(track_session=row.get('track_session'),**{
                    k:row.get(k) for k in ('selected_track_id','selection_state')})
        except Exception as exc:self.error=f'{type(exc).__name__}: {exc}'
        finally:
            for file in files.values():file.close()
            try:
                save_json(self.folder/'tracking_selection.json',dict(
                    version=1,**self.identity,cameras=self.summary,dropped_preview_samples=self.dropped,
                    error=self.error,scope='sampled live preview; offline tracks have independent IDs'))
            except Exception as exc:self.error=f'{type(exc).__name__}: {exc}'
