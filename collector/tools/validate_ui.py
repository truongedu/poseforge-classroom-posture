"""Render the real dashboard at repeatable physical client sizes and Qt DPI.

Example: python tools/validate_ui.py --width 1920 --height 1080 --scale 1.25
All generated test footage/metadata stays in ui_validation; never uses user data.
The DPI override is per process, and does not change Windows display settings.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--width', type=int, default=1920)
parser.add_argument('--height', type=int, default=1080)
parser.add_argument('--scale', type=float, default=1)
parser.add_argument('--native-maximized', action='store_true')
args = parser.parse_args()
if not args.native_maximized:
    environment = dict(os.environ)
    environment.pop('QT_SCREEN_SCALE_FACTORS', None)
    environment.pop('QT_SCALE_FACTOR', None)
    native_scale = float(subprocess.check_output([sys.executable, '-c',
        'from PySide6.QtWidgets import QApplication; a=QApplication([]); print(a.primaryScreen().devicePixelRatio())'],
        env=environment, text=True).strip())
    # Qt's global scale is a multiplier of the native Windows DPI.
    os.environ.pop('QT_SCREEN_SCALE_FACTORS', None)
    os.environ['QT_SCALE_FACTOR'] = str(args.scale/native_scale)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from posture_collector.ui import Window, THEME
from posture_collector.storage import DataStore


def wait_until(predicate, timeout=15):
    deadline = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() > deadline: raise RuntimeError('Validation timed out')
        QTest.qWait(30)


base = Path(__file__).resolve().parents[1]/'ui_validation'
name = 'native_maximized_125' if args.native_maximized else f'{args.width}x{args.height}_{round(args.scale*100)}pct'
output = base/name; output.mkdir(parents=True, exist_ok=True)
data = output/'synthetic_dataset'; store = DataStore(data)
if 'TQB' not in store.subjects(): store.add_subject('TQB', True, 'SYNTHETIC UI TEST ONLY; no real participant')
feeds = base/'synthetic_feeds'; feeds.mkdir(exist_ok=True)
configs = []
for index, bg in enumerate([(48, 38, 30), (40, 48, 32)], 1):
    path = feeds/f'cam{index:02d}.mp4'
    if not path.exists():
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 360))
        assert writer.isOpened()
        for frame in range(240):
            image = np.full((360, 640, 3), bg, np.uint8)
            for x in range(0, 640, 40): cv2.line(image, (x, 0), (x, 360), (63, 65, 70))
            for y in range(0, 360, 40): cv2.line(image, (0, y), (640, y), (63, 65, 70))
            cv2.rectangle(image, (110, 235), (530, 260), (105, 105, 110), -1)
            cv2.circle(image, (320, 104), 28, (178, 180, 190), -1)
            cv2.line(image, (320, 134), (320, 232), (178, 180, 190), 14)
            cv2.line(image, (320, 155), (253, 229), (178, 180, 190), 12)
            cv2.line(image, (320, 155), (395, 229), (178, 180, 190), 12)
            cv2.putText(image, 'SIMULATED CAMERA / NOT HUMAN DATA', (20, 31), cv2.FONT_HERSHEY_SIMPLEX, .6, (200, 210, 220), 1, cv2.LINE_AA)
            writer.write(image)
        writer.release()
    configs.append(dict(id=f'cam{index:02d}', name=f'Validation camera {index}',
                        view='front' if index == 1 else 'oblique', url=str(path), enabled=True))
store.set_cameras(configs)
app = QApplication([]); app.setStyle('Fusion'); app.setFont(QFont('Segoe UI', 10)); app.setStyleSheet(THEME)
if not args.native_maximized:
    assert abs(app.primaryScreen().devicePixelRatio()-args.scale) < .01
w = Window(data); errors = []; w.alert = lambda title, message: errors.append([title, str(message)])
if args.native_maximized: w.showMaximized()
else:
    w.show(); QTest.qWait(100)
    w.resize(round(args.width/args.scale), round(args.height/args.scale))
w.activateWindow(); QTest.qWait(400)
w.subjects_combo.setCurrentText('TQB'); w.session_edit.setText('session_validation')
w.auto_extract.setChecked(False)
metrics = dict(requested_physical=[args.width, args.height], requested_scale=args.scale,
               native_maximized=args.native_maximized, actual_device_pixel_ratio=w.devicePixelRatioF(),
               screen_logical=list(app.primaryScreen().size().toTuple()),
               available_logical=list(app.primaryScreen().availableGeometry().size().toTuple()), screenshots={})


def capture(state):
    QTest.qWait(220)
    image = w.grab(); image.save(str(output/f'{state}.png'))
    metrics['screenshots'][state] = dict(physical=list(image.size().toTuple()),
        client_logical=list(w.size().toTuple()), camera_columns=w._camera_columns,
        control_scroll_max=w.control_scroll.verticalScrollBar().maximum(),
        control_horizontal_overflow=w.control_scroll.horizontalScrollBar().maximum(),
        preview_horizontal_overflow=w.preview_scroll.horizontalScrollBar().maximum())


capture('01_disconnected')
w.connect_cameras()
wait_until(lambda: not w.camera_busy and len(w.core.streams)==2 and all(s.ready() for s in w.core.streams.values()))
w.refresh(); capture('02_connected')
w.start_btn.setFocus(); w.activateWindow(); QTest.qWait(80); QTest.keyClick(w, Qt.Key.Key_S)
assert w.core.active, 'S shortcut did not start'
folder = w.core.active['folder']; QTest.qWait(1200)
capture('03_recording')
QTest.qWait(4200); w.end_btn.setFocus(); QTest.keyClick(w, Qt.Key.Key_E)
assert w.core.active is None, 'E shortcut did not stop'
wait_until(lambda: not w.core.finalizing); w.refresh()
w.control_scroll.verticalScrollBar().setValue(w.control_scroll.verticalScrollBar().maximum())
capture('04_progress_scrolled')
w.set_page(1); capture('05_camera_sources')
w.set_page(2); capture('06_dataset_tools')
w.set_page(0); w.control_scroll.verticalScrollBar().setValue(0)
w.toggle_log(); capture('07_log_toggled'); w.toggle_log()
# A source whose last frame is stale must lose its live badge and disable Start.
for stream in w.core.streams.values(): stream.stop()
for stream in w.core.streams.values(): stream.thread.join(2)
QTest.qWait(2200); w.refresh(); capture('08_stalled')
assert not w.start_btn.isEnabled()
w.connect_cameras()
wait_until(lambda: not w.camera_busy and all(s.ready() for s in w.core.streams.values()))
w.refresh(); assert w.start_btn.isEnabled(); capture('09_reconnected')
w.refresh_timer.stop()
for cid, card in w.cards.items():
    frame = w.core.streams[cid].snapshot()[0]
    points = [(240+i*10, 110+i*7, 1.0) for i in range(13)]
    card.show_frame(frame, points)
capture('10_simulated_preview_overlay')
w.refresh_timer.start(100)
w.disconnect_cameras(); wait_until(lambda: not w.camera_busy)
# Exercise adaptive grid with additional configured sources, without creating fake connections.
w.rebuild_camera_cards(configs+[dict(c, id=f'cam{i:02d}') for i, c in zip((3, 4), configs)])
capture('11_four_sources')
w.rebuild_camera_cards(configs)
report = json.loads((folder/'metadata.json').read_text())
assert report['person_id']=='TQB' and report['session_id']=='session_validation' and report['label']==0
assert report['status']=='ok', report
for cid in report['cameras']:
    with (folder/f'{cid}_timestamps.csv').open() as file: timestamps = list(csv.DictReader(file))
    video = cv2.VideoCapture(str(folder/f'{cid}_raw.mp4'))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT))==len(timestamps)==report['cameras'][cid]['frames']
    video.release()
metrics['recording_report'] = report
metrics['errors'] = errors
w.close(); wait_until(lambda: not w.isVisible())
assert not errors, errors
metrics['safe_close'] = not w.core.streams and not w.core.finalizing
(output/'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(dict(output=str(output), dpr=metrics['actual_device_pixel_ratio'],
                     screenshots=len(metrics['screenshots']), safe_close=metrics['safe_close'])))
