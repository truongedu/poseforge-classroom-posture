"""Exercise real YOLO11/ByteTrack, two local cameras and the actual Qt window.

Requires models/yolo11n-pose.pt and tests/tracking_assets/bus.jpg. All recordings
go to ui_validation/bytetrack; public test footage must not become training data.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QFont
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from posture_collector.storage import DataStore
from posture_collector.ui import Window,THEME


def pump(seconds):
    # QTest.qWait holds the Python GIL on some PySide builds. Let Python model
    # loading/inference workers run while dispatching the real Qt events.
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        QApplication.instance().processEvents();time.sleep(.01)


def wait_for(predicate,timeout=30):
    deadline=time.monotonic()+timeout
    while not predicate():
        if time.monotonic()>deadline:raise RuntimeError('Timed out during tracking validation')
        pump(.03)


def main():
    output=PROJECT/'ui_validation'/'bytetrack';output.mkdir(parents=True,exist_ok=True)
    image=cv2.imread(str(PROJECT/'tests'/'tracking_assets'/'bus.jpg'))
    assert image is not None
    image=cv2.resize(image,(640,480))
    cameras=[]
    for i in (1,2):
        video=output/f'prerecorded_cam{i:02d}.mp4'
        if not video.exists():
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,480));assert writer.isOpened()
            for j in range(560):
                frame=cv2.warpAffine(image,np.float32([[1,0,(j%20)//4],[0,1,0]]),(640,480))
                writer.write(frame)
            writer.release()
        cameras.append(dict(id=f'cam{i:02d}',name='Public prerecorded TEST ONLY',view='front' if i==1 else 'oblique',url=str(video),enabled=True))
    store=DataStore(output/'dataset')
    if 'TQB' not in store.subjects():store.add_subject('TQB',True,'PUBLIC TEST IMAGE; not consenting participant data, never train on this')
    store.set_cameras(cameras)
    app=QApplication([]);app.setStyle('Fusion');app.setFont(QFont('Segoe UI',10));app.setStyleSheet(THEME)
    w=Window(store.root);errors=[];w.alert=lambda title,message:errors.append((title,str(message)))
    w.showMaximized();w.activateWindow();w.session_edit.setText('session_bytetrack_test');w.auto_extract.setChecked(False)
    w.connect_cameras()
    wait_for(lambda:not w.camera_busy and len(w.core.streams)==2 and all(s.ready() for s in w.core.streams.values()))
    w.preview_combo.setCurrentIndex(1);w.preview_weights.setCurrentIndex(1);w.preview_device.setCurrentIndex(1)
    w.enable_pose_preview();worker=w.preview_worker
    wait_for(lambda:all(worker.get(cid) and len(worker.get(cid)['persons'])>=2 for cid in ('cam01','cam02')))
    w.refresh()
    chosen={}
    for cid in w.cards:
        result=worker.get(cid);assert not result['error']
        chosen[cid]=next(p['track_id'] for p in result['persons'] if p['track_id'] is not None)
        w.select_track(cid,chosen[cid])
    pump(.4);w.grab().save(str(output/'01_selected_tracks.png'))
    w.start_btn.setFocus();w.activateWindow();pump(.08);QTest.keyClick(w,Qt.Key.Key_S)
    assert w.core.active;folder=w.core.active['folder']
    pump(1.6);w.grab().save(str(output/'02_recording_tracks.png'))
    pump(4);w.end_btn.setFocus();QTest.keyClick(w,Qt.Key.Key_E)
    assert w.core.active is None
    wait_for(lambda:not w.core.finalizing and not any(t.is_alive() for t in w.journal_threads))
    report=json.loads((folder/'metadata.json').read_text());selection=json.loads((folder/'tracking_selection.json').read_text())
    assert report['person_id']=='TQB' and report['label']==0 and report['session_id']=='session_bytetrack_test'
    assert report['status']=='ok',report
    assert selection['error'] is None
    for cid in chosen:
        records=[json.loads(line) for line in (folder/'keypoints'/f'{cid}_yolo17_live_tracks.txt').read_text().splitlines()]
        assert any(r['event']=='sample' and r['selected_track_id']==chosen[cid] and r['persons'] for r in records)
        assert selection['cameras'][cid]['selected_track_id']==chosen[cid]
        with (folder/f'{cid}_timestamps.csv').open() as file:stamps=list(csv.DictReader(file))
        assert len(stamps)==report['cameras'][cid]['frames']
        assert report['cameras'][cid]['output_fps']==20
    raw_hashes={cid:hashlib.sha256((folder/f'{cid}_raw.mp4').read_bytes()).hexdigest() for cid in chosen}
    # Exercise the actual background Dataset Tools action and full-frame tracking
    # extraction. Multi-person targets need an explicit OFFLINE ID, not live IDs.
    # The GUI's default extraction is conservative and exports all persons.
    w.backend_combo.setCurrentText('yolo');w.offline_model.setText(str(PROJECT/'models'/'yolo11n-pose.pt'))
    w.extract_clicked();wait_for(lambda:not w.busy,timeout=120)
    for cid in chosen:
        raw=folder/'keypoints'/f'{cid}_yolo17_raw.txt'
        rows=[json.loads(line) for line in raw.read_text().splitlines()]
        assert len(rows)==report['cameras'][cid]['frames']
        assert any(any(p['track_id'] is not None for p in row['persons']) for row in rows)
        assert hashlib.sha256((folder/f'{cid}_raw.mp4').read_bytes()).hexdigest()==raw_hashes[cid]
    w.normalize_clicked();wait_for(lambda:not w.busy)
    w.set_page(2);pump(.1);w.grab().save(str(output/'03_dataset_tools.png'))
    w.set_page(0)
    # Explicitly replace the live connections, then restart tracking. Old target
    # choices must not automatically bind to new streams' numeric IDs.
    old_tokens=dict(w.tracking_tokens)
    w.connect_cameras();wait_for(lambda:not w.camera_busy and all(s.ready() for s in w.core.streams.values()))
    w.enable_pose_preview();new_worker=w.preview_worker
    wait_for(lambda:all(new_worker.get(cid) and new_worker.get(cid)['persons'] for cid in chosen))
    w.refresh()
    assert all(w.tracking_tokens[cid]!=old_tokens[cid] for cid in chosen)
    assert all(w.track_selections[cid].track_id is None for cid in chosen)
    assert all(w.track_selections[cid].state=='reselect_required' for cid in chosen)
    w.grab().save(str(output/'04_reconnected_reselection.png'))
    results=dict(person_id=report['person_id'],label=report['label'],session_id=report['session_id'],
        chosen_live_tracks=chosen,cameras=report['cameras'],tracking_sidecar=selection,
        raw_hashes=raw_hashes,actual_dpr=w.devicePixelRatioF(),errors=errors)
    w.close();wait_for(lambda:not w.isVisible())
    assert not errors,errors
    results['safe_close']=not w.core.streams
    (output/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(output=str(output),recording=str(folder),safe_close=results['safe_close'])))


def entrypoint_smoke():
    """Run main.py's actual app.exec loop with TWO real YOLO11m models."""
    import runpy
    output=PROJECT/'ui_validation'/'bytetrack';store=DataStore(output/'entrypoint_dataset')
    if 'TQB' not in store.subjects():store.add_subject('TQB',True,'PUBLIC TEST FOOTAGE ONLY')
    cameras=[dict(id=f'cam{i:02d}',name='Entry-point test',view='front' if i==1 else 'oblique',
                  url=str(output/f'prerecorded_cam{i:02d}.mp4'),enabled=True) for i in (1,2)]
    assert all(Path(c['url']).exists() for c in cameras),'Run the normal validation first to create feeds'
    store.set_cameras(cameras);app=QApplication([])
    context={'phase':'initial','errors':[],'started':time.monotonic()}
    def tick():
        phase=context['phase']
        if phase=='closing':return
        w=context.get('window')
        try:
            assert time.monotonic()-context['started']<60,'Entry-point validation timed out'
            if phase=='initial':
                w=next(w for w in app.topLevelWidgets() if isinstance(w,Window))
                context['window']=w;w.alert=lambda a,b:context['errors'].append([a,str(b)])
                w.showMaximized();w.session_edit.setText('session_entrypoint_m');w.auto_extract.setChecked(False)
                w.connect_cameras();context['phase']='connected'
            elif phase=='connected' and not w.camera_busy and len(w.core.streams)==2 and all(s.ready() for s in w.core.streams.values()):
                w.preview_combo.setCurrentIndex(1);w.preview_weights.setCurrentIndex(2);w.preview_device.setCurrentIndex(1)
                w.enable_pose_preview();context['phase']='tracking'
            elif phase=='tracking' and all(w.preview_worker.get(cid) and w.preview_worker.get(cid)['persons'] for cid in w.cards):
                for cid in w.cards:
                    result=w.preview_worker.get(cid);assert not result['error']
                    w.select_track(cid,next(p['track_id'] for p in result['persons'] if p['track_id'] is not None))
                w.start_btn.setFocus();w.activateWindow();context['phase']='start'
            elif phase=='start':
                QTest.keyClick(w,Qt.Key.Key_S);assert w.core.active
                context['folder']=w.core.active['folder'];context['recorded_at']=time.monotonic();context['phase']='recording'
            elif phase=='recording':
                elapsed=time.monotonic()-context['recorded_at']
                if elapsed>1 and not context.get('screenshot'):
                    w.grab().save(str(output/'05_entrypoint_yolo11m_recording.png'));context['screenshot']=True
                if elapsed>5.6:
                    w.end_btn.setFocus();QTest.keyClick(w,Qt.Key.Key_E);assert not w.core.active
                    context['phase']='saving'
            elif phase=='saving' and not w.core.finalizing and not any(t.is_alive() for t in w.journal_threads):
                w.close();context['phase']='closing'
        except Exception as exc:
            context['errors'].append([type(exc).__name__,str(exc)])
            if w:
                if w.core.active:w.end_record()
                if not w.core.finalizing:w.close();context['phase']='closing'
                else:context['phase']='saving'
            else:app.quit()
    driver=QTimer();driver.timeout.connect(tick);driver.start(100)
    sys.argv=['main.py','--data',str(store.root)]
    try:runpy.run_path(str(PROJECT/'main.py'),run_name='__main__')
    except SystemExit as exc:assert exc.code in (0,None)
    driver.stop();w=context['window'];assert not context['errors'],context['errors']
    assert not w.core.streams and not w.core.finalizing
    folder=context['folder'];report=json.loads((folder/'metadata.json').read_text())
    assert report['status']=='ok' and report['person_id']=='TQB'
    selection=json.loads((folder/'tracking_selection.json').read_text())
    assert selection['error'] is None and 'yolo11m-pose.pt' in selection['model']
    (output/'entrypoint_results.json').write_text(json.dumps(dict(model=selection['model'],safe_close=True,
        recording=str(folder),metadata=report,selection=selection,errors=context['errors']),indent=2),encoding='utf-8')
    print('Actual main.py event loop, two YOLO11m models, S/E, raw files, sidecars and safe close: PASSED')


if __name__=='__main__':
    entrypoint_smoke() if '--entrypoint' in sys.argv else main()
