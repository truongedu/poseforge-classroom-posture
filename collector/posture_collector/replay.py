"""Frame-accurate replay and non-destructive take curation."""
from __future__ import annotations

from pathlib import Path
import json

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PySide6.QtWidgets import (QWidget, QFrame, QHBoxLayout, QVBoxLayout,
    QComboBox, QSlider, QLabel, QCheckBox, QSizePolicy, QSplitter, QMessageBox)

from .replay_data import take_catalog, read_pose_jsonl, draw_pose_overlay
from .curation import (read_edits, save_edit, timestamp_ns, take_key,
                       capture_bounds, subtract_interval, _times)
from .ui_components import button, label, panel


class VideoPane(QLabel):
    def __init__(self):
        super().__init__('Select a take on the right')
        self.setObjectName('replayScreen')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = None

    def set_frame(self, image):
        # QImage wrapping numpy memory requires copying before leaving this scope.
        h,w = image.shape[:2]
        qimg = QImage(image.data, w, h, image.strides[0], QImage.Format.Format_BGR888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._fit()

    def clear_frame(self, text):
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def _fit(self):
        if self._pixmap and not self._pixmap.isNull():
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class ReplayInspector(QWidget):
    """No YOLO calls; curation writes JSON decisions only, never raw video."""
    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = Path(root).resolve()
        self.catalog = []
        self.current = None
        self.cap = None
        self.frames = {}
        self.frame_index = 0
        self.frame_total = 0
        self.fps = 20.0
        self.playing = False
        self.last_frame = None
        self._changing = False
        self._overlay_note = ''
        self.mark_start = None
        self.mark_end = None
        self.keep_ranges = []
        self.last_edit = None  # one-step undo, scoped to a take
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)

        shell = QVBoxLayout(self)
        shell.setContentsMargins(22,20,22,18)
        shell.setSpacing(12)
        shell.addWidget(label('Replay & Inspect', 'headline'))
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        left, left_layout = panel()
        left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.setContentsMargins(12,12,12,12)
        self.screen = VideoPane()
        left_layout.addWidget(self.screen, 1)
        self.frame_info = label('Frame —', 'muted')
        left_layout.addWidget(self.frame_info)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setObjectName('replayTimeline')
        self.timeline.setRange(0,0)
        self.timeline.sliderPressed.connect(self.pause)
        self.timeline.valueChanged.connect(self.seek)
        left_layout.addWidget(self.timeline)
        transport = QHBoxLayout()
        self.first_btn = button('⏮ First', lambda: self.seek(0))
        self.prev_btn = button('◀ Frame', lambda: self.seek(self.frame_index-1))
        self.play_btn = button('▶ Play', self.toggle_play, 'primary')
        self.next_btn = button('Frame ▶', lambda: self.seek(self.frame_index+1))
        self.last_btn = button('Last ⏭', lambda: self.seek(self.frame_total-1))
        for widget in (self.first_btn,self.prev_btn,self.play_btn,self.next_btn,self.last_btn):
            transport.addWidget(widget)
        left_layout.addLayout(transport)
        # Replay seek position is the only input needed to edit a take.
        edits = QHBoxLayout()
        edits.addWidget(button('Mark start', self.set_range_start))
        edits.addWidget(button('Mark end', self.set_range_end))
        edits.addWidget(button('Remove range', self.remove_range, 'primary'))
        edits.addWidget(button('Keep range', self.append_range))
        self.undo_btn = button('Undo last', self.undo_last_change)
        edits.addWidget(self.undo_btn)
        edits.addWidget(button('Reset cuts', self.reset_ranges))
        left_layout.addLayout(edits)
        self.edit_info = label('Select a take to curate.', 'muted')
        self.edit_info.setWordWrap(True)
        left_layout.addWidget(self.edit_info)
        splitter.addWidget(left)

        right, rv = panel()
        right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        right.setMinimumWidth(260)
        right.setMaximumWidth(360)
        rv.addWidget(label('Choose recording', 'h2'))
        self.person = QComboBox(); self.session = QComboBox()
        self.label_id = QComboBox(); self.take = QComboBox(); self.camera = QComboBox()
        for caption, combo in [('Person',self.person),('Session',self.session),
                               ('Label',self.label_id),('Take',self.take),('Camera',self.camera)]:
            rv.addWidget(label(caption,'muted'))
            combo.setMinimumWidth(0)
            rv.addWidget(combo)
        rv.addWidget(button('Refresh recordings', self.refresh_catalog))
        rv.addSpacing(12)
        rv.addWidget(label('Saved overlays', 'h2'))
        self.box_check = QCheckBox('Bounding boxes'); self.box_check.setChecked(True)
        self.kp_check = QCheckBox('17 keypoints / skeleton'); self.kp_check.setChecked(True)
        self.all_check = QCheckBox('Show all detected people'); self.all_check.setChecked(False)
        for check in (self.box_check,self.kp_check,self.all_check):
            rv.addWidget(check); check.toggled.connect(self.repaint_frame)
        self.quality = label('No clip loaded','muted')
        self.quality.setWordWrap(True)
        rv.addWidget(self.quality)
        rv.addSpacing(12)
        rv.addWidget(label('Take curation (source unchanged)', 'h2'))
        self.exclude_camera_btn = button('Exclude this camera', self.exclude_camera)
        self.exclude_take_btn = button('Exclude entire take', self.exclude_take)
        rv.addWidget(self.exclude_camera_btn)
        rv.addWidget(self.exclude_take_btn)
        rv.addWidget(button('Restore original take', self.restore_take))
        rv.addWidget(button('Export clean ZIP', self.export_from_replay, 'primary'))
        rv.addStretch(1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0,4)
        splitter.setStretchFactor(1,1)
        splitter.setSizes([1000,320])
        shell.addWidget(splitter, 1)
        for combo, callback in ((self.person,self.update_sessions),
                                 (self.session,self.update_labels),
                                 (self.label_id,self.update_takes),
                                 (self.take,self.update_cameras),
                                 (self.camera,self.open_selected)):
            combo.currentIndexChanged.connect(callback)
        self.refresh_catalog()
        # Shortcuts apply when this viewer is active, not while recording.
        for key, action in (('Space',self.toggle_play),('Left',lambda: self.seek(self.frame_index-1)),
                            ('Right',lambda: self.seek(self.frame_index+1)),
                            ('Home',lambda: self.seek(0)),('End',lambda: self.seek(self.frame_total-1))):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)

    @staticmethod
    def _fill(combo, values, previous=None):
        combo.blockSignals(True)
        combo.clear()
        for value in values:
            combo.addItem(str(value), value)
        if previous is not None and combo.findData(previous) >= 0:
            combo.setCurrentIndex(combo.findData(previous))
        combo.blockSignals(False)

    def refresh_catalog(self):
        self.pause()
        previous = self.person.currentData()
        self.catalog = take_catalog(self.root)
        self._fill(self.person, sorted({x['person'] for x in self.catalog}), previous)
        self.update_sessions()

    def choose(self, person, session, label_id, take, camera=None):
        """Jump from a selected Dataset take to its video replay."""
        self.refresh_catalog()
        for combo, value, update in ((self.person, person, self.update_sessions),
                                     (self.session, session, self.update_labels),
                                     (self.label_id, label_id, self.update_takes),
                                     (self.take, take, self.update_cameras)):
            pos = combo.findData(value)
            if pos < 0:return False
            combo.setCurrentIndex(pos)
            update()
        if camera is not None:
            pos = self.camera.findData(camera)
            if pos >= 0:self.camera.setCurrentIndex(pos)
        self.open_selected()
        return self.current is not None

    def update_sessions(self, *_):
        person = self.person.currentData()
        current = self.session.currentData()
        self._fill(self.session, sorted({x['session'] for x in self.catalog if x['person']==person}), current)
        self.update_labels()

    def update_labels(self, *_):
        person,session = self.person.currentData(),self.session.currentData()
        current = self.label_id.currentData()
        self._fill(self.label_id, sorted({x['label'] for x in self.catalog
                       if x['person']==person and x['session']==session}), current)
        self.update_takes()

    def update_takes(self, *_):
        person,session,lid=self.person.currentData(),self.session.currentData(),self.label_id.currentData()
        current = self.take.currentData()
        self._fill(self.take,sorted({x['take'] for x in self.catalog
                     if x['person']==person and x['session']==session and x['label']==lid}),current)
        self.update_cameras()

    def update_cameras(self, *_):
        person,session,lid,tid=(self.person.currentData(),self.session.currentData(),
                                 self.label_id.currentData(),self.take.currentData())
        current = self.camera.currentData()
        self._fill(self.camera,sorted({x['camera'] for x in self.catalog
                     if x['person']==person and x['session']==session and x['label']==lid and x['take']==tid}),current)
        self.open_selected()

    def open_selected(self, *_):
        self.pause()
        self.close_capture()
        desired=(self.person.currentData(),self.session.currentData(),self.label_id.currentData(),
                 self.take.currentData(),self.camera.currentData())
        self.current=next((r for r in self.catalog if
                            (r['person'],r['session'],r['label'],r['take'],r['camera'])==desired),None)
        self.load_curation()
        if self.current is None:
            self.screen.clear_frame('No video selected')
            self.quality.setText('No clip loaded')
            return
        cap=cv2.VideoCapture(str(self.current['video']))
        if not cap.isOpened():
            self.screen.clear_frame('Cannot decode MP4')
            self.quality.setText('Video open failed')
            return
        self.cap=cap
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n<=0:
            # Avoid O(n) scans on the GUI thread; malformed videos are flagged.
            self.close_capture()
            self.screen.clear_frame('Video frame count unavailable')
            self.quality.setText('Cannot seek this MP4')
            return
        fps=cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if 1<=fps<=120 else 20.
        self.frame_total=n
        self.frames,self._overlay_note=read_pose_jsonl(self.current['raw'])
        good=sum(record.get('status')=='ok' for record in self.frames.values())
        mismatch = (f' · expected video frames {n}' if self.frames and len(self.frames) != n else '')
        self.quality.setText(f'{self._overlay_note}{mismatch}\nValid pose: {good}/{len(self.frames)} frames\n'
                             'Blue: other people · cyan: selected')
        self.timeline.blockSignals(True)
        self.timeline.setRange(0,max(0,n-1))
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.seek(0)

    def _current_take(self):
        return self.current['video'].parent if self.current else None

    def _safe_to_edit(self, *, needs_video=False):
        host = self.window()  # QStackedWidget is the immediate parent; Window owns CollectorCore.
        if not hasattr(host, 'core'):
            QMessageBox.warning(self, 'Cannot verify recording state',
                                'Replay is not attached to the Collector main window.')
            return False
        if host.core.active or host.core.finalizing or host.busy or getattr(host, 'camera_busy', False):
            QMessageBox.warning(self, 'Recording / processing active',
                                'Finish recording and background processing before editing a take.')
            return False
        if self.current is None:
            return False
        if needs_video and (self.cap is None or self.frame_total <= 0):
            QMessageBox.warning(self, 'No playable video', 'Cannot mark frames without a readable video.')
            return False
        if needs_video:
            key = take_key(self.root, self._current_take())
            if read_edits(self.root)['takes'].get(key, {}).get('exclude_take'):
                QMessageBox.warning(self, 'Take excluded',
                                    'Restore original take before marking its frames.')
                return False
        return True

    def load_curation(self):
        self.mark_start = self.mark_end = None
        self.keep_ranges = []
        if not self.current:
            if hasattr(self, 'edit_info'): self.edit_info.setText('Select a take to curate.')
            return
        try:
            change = read_edits(self.root)['takes'].get(take_key(self.root, self._current_take()), {})
            self.keep_ranges = [list(r) for r in (change.get('keep_ranges_ns') or [])]
            self.update_curation_info(change)
        except (ValueError, OSError) as exc:
            self.edit_info.setText(f'Curation error: {exc}')

    def update_curation_info(self, change=None):
        if change is None and self.current:
            change = read_edits(self.root)['takes'].get(take_key(self.root, self._current_take()), {})
        change = change or {}
        excluded = ', '.join(change.get('exclude_cameras', [])) or 'none'
        ranges = 'ALL'
        if self.keep_ranges:
            try:
                origin = timestamp_ns(self.root, self._current_take(), self.current['camera'], 0)
                ranges = ', '.join(f'{(a-origin)/1e9:.2f}–{(b-origin)/1e9:.2f}s'
                                   for a, b in self.keep_ranges)
            except (ValueError, OSError):
                ranges = f'{len(self.keep_ranges)} intervals'
        take_excluded = bool(change.get('exclude_take'))
        cam_excluded = self.current['camera'] in change.get('exclude_cameras', [])
        self.edit_info.setText(
            ('⛔ TAKE EXCLUDED — not exported' if take_excluded else
             '⛔ CAMERA EXCLUDED — not exported' if cam_excluded else
             '✓ Included in clean export') +
            f' | Kept: {ranges} | Excluded cameras: {excluded}. '
            'Source files unchanged.')
        self.edit_info.setStyleSheet('color: #FFB2A4;' if take_excluded or cam_excluded
                                     else 'color: #A7E7CD;')
        self.exclude_take_btn.setEnabled(not take_excluded)
        self.exclude_take_btn.setText('Take excluded ✓' if take_excluded else 'Exclude entire take')
        self.exclude_camera_btn.setText('Include this camera' if cam_excluded else 'Exclude this camera')
        key = take_key(self.root, self._current_take())
        self.undo_btn.setEnabled(bool(self.last_edit and self.last_edit[0] == key))

    def save_current_edit(self, **kwargs):
        """Write one curation decision and retain previous state for Undo last."""
        take = self._current_take()
        key = take_key(self.root, take)
        previous = read_edits(self.root)['takes'].get(key, {}).copy()
        save_edit(self.root, take, **kwargs)
        self.last_edit = (key, previous)

    def undo_last_change(self):
        if not self._safe_to_edit() or not self.last_edit: return
        key, before = self.last_edit
        if key != take_key(self.root, self._current_take()): return
        try:
            save_edit(self.root, self._current_take(),
                      ranges=before.get('keep_ranges_ns'),
                      exclude_take=before.get('exclude_take', False),
                      exclude_cameras=before.get('exclude_cameras', []))
            self.last_edit = None
            self.load_curation()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Cannot undo', str(exc))

    def set_range_start(self):
        if not self._safe_to_edit(needs_video=True): return
        self.mark_start = self.frame_index
        self.edit_info.setText(f'Start marked: frame {self.mark_start+1}. Mark end, then Remove range.')

    def set_range_end(self):
        if not self._safe_to_edit(needs_video=True): return
        self.mark_end = self.frame_index
        self.edit_info.setText(f'End marked: frame {self.mark_end+1}. Click Remove range (or Keep range).')

    def remove_range(self):
        """Cut an inclusive frame interval without changing any original recording.

        Internally keep_ranges_ns stays the only curation representation, so
        exporter and Notebook V2 need no new schema or Track ID dependency.
        """
        if not self._safe_to_edit(needs_video=True): return
        if self.mark_start is None or self.mark_end is None or self.mark_start > self.mark_end:
            QMessageBox.warning(self, 'Invalid range', 'Mark start frame <= end frame first.')
            return
        try:
            take, cam = self._current_take(), self.current['camera']
            full = capture_bounds(self.root, take)
            start = (full[0] if self.mark_start == 0 else
                     timestamp_ns(self.root, take, cam, self.mark_start))
            end = (full[1] if self.mark_end == self.frame_total - 1 else
                   timestamp_ns(self.root, take, cam, self.mark_end + 1))
            edits = read_edits(self.root)['takes'].get(take_key(self.root, take), {})
            ranges = subtract_interval(edits.get('keep_ranges_ns'), (start, end), full)
            if not ranges:
                QMessageBox.information(self, 'Entire take selected',
                                        'This would remove every frame. Use Exclude entire take instead.')
                return
            # Every remaining segment must have frames in every enabled camera.
            # This catches sub-frame differences in the two capture clocks early.
            meta = json.loads((take / 'metadata.json').read_text())
            enabled = set(meta.get('cameras', {})) - set(edits.get('exclude_cameras', []))
            for camera in enabled:
                _, _, ns = _times(take / f'{camera}_timestamps.csv')
                for a, b in ranges:
                    if not any(a <= t < b for t in ns):
                        raise ValueError(f'Remaining segment has no {camera} frames. '
                                         'Move mark slightly away from clip boundary.')
            self.save_current_edit(ranges=ranges,
                      exclude_take=False, exclude_cameras=edits.get('exclude_cameras', []))
            self.keep_ranges = ranges
            self.mark_start = self.mark_end = None
            self.update_curation_info()
        except (OSError, ValueError, IndexError) as exc:
            QMessageBox.warning(self, 'Cannot remove range', str(exc))

    def append_range(self):
        if not self._safe_to_edit(needs_video=True): return
        if self.mark_start is None or self.mark_end is None or self.mark_start > self.mark_end:
            QMessageBox.warning(self, 'Invalid range', 'Mark start frame <= end frame first.')
            return
        try:
            take, cam = self._current_take(), self.current['camera']
            start = timestamp_ns(self.root, take, cam, self.mark_start)
            # Inclusive last selected frame is represented with an exclusive upper timestamp.
            end = (timestamp_ns(self.root, take, cam, self.mark_end + 1)
                   if self.mark_end + 1 < self.frame_total else
                   timestamp_ns(self.root, take, cam, self.mark_end) + 1)
            edits = read_edits(self.root)['takes'].get(take_key(self.root, take), {})
            candidate = sorted([*self.keep_ranges, [start, end]])
            ranges = []
            for a, b in candidate:
                if ranges and a <= ranges[-1][1]:
                    ranges[-1][1] = max(ranges[-1][1], b)
                else:
                    ranges.append([a, b])
            self.save_current_edit(ranges=ranges,
                      exclude_take=False, exclude_cameras=edits.get('exclude_cameras', []))
            self.keep_ranges = ranges
            self.mark_start = self.mark_end = None
            self.update_curation_info()
        except (OSError, ValueError, IndexError) as exc:
            QMessageBox.warning(self, 'Cannot keep range', str(exc))

    def reset_ranges(self):
        if not self._safe_to_edit(): return
        try:
            take = self._current_take()
            edits = read_edits(self.root)['takes'].get(take_key(self.root, take), {})
            self.save_current_edit(ranges=None, exclude_take=edits.get('exclude_take', False),
                      exclude_cameras=edits.get('exclude_cameras', []))
            self.load_curation()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Cannot reset ranges', str(exc))

    def exclude_camera(self):
        if not self._safe_to_edit(): return
        try:
            take = self._current_take()
            edits = read_edits(self.root)['takes'].get(take_key(self.root, take), {})
            cameras = set(edits.get('exclude_cameras', []))
            cam = self.current['camera']
            if cam in cameras:
                cameras.remove(cam)
            else:
                cameras.add(cam)
            self.save_current_edit(ranges=edits.get('keep_ranges_ns'),
                      exclude_take=edits.get('exclude_take', False), exclude_cameras=cameras)
            self.load_curation()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Cannot exclude camera', str(exc))

    def exclude_take(self):
        if not self._safe_to_edit(): return
        take = self._current_take()
        if QMessageBox.question(self, 'Exclude entire take',
                f'Exclude {take.name} from the next clean dataset export?\nRaw files stay unchanged.') \
                != QMessageBox.StandardButton.Yes:
            return
        try:
            edits = read_edits(self.root)['takes'].get(take_key(self.root, take), {})
            self.save_current_edit(ranges=edits.get('keep_ranges_ns'), exclude_take=True,
                                   exclude_cameras=edits.get('exclude_cameras', []))
            self.load_curation()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Cannot exclude take', str(exc))

    def restore_take(self):
        if not self._safe_to_edit(): return
        self.save_current_edit()
        self.load_curation()

    def export_from_replay(self):
        if not self._safe_to_edit(): return
        host = self.window()
        if hasattr(host, 'export_clean_clicked'):
            host.export_clean_clicked()

    def pause(self):
        self.playing=False
        self.timer.stop()
        if hasattr(self,'play_btn'): self.play_btn.setText('▶ Play')

    def toggle_play(self):
        if self.cap is None or self.frame_total<=0: return
        if self.playing:
            self.pause()
        else:
            if self.frame_index>=self.frame_total-1: self.seek(0)
            self.playing=True
            self.play_btn.setText('⏸ Pause')
            self.timer.start(max(25,round(1000/self.fps)))

    def advance(self):
        if self.frame_index>=self.frame_total-1:
            self.pause()
        else:
            self.seek(self.frame_index+1)

    def seek(self, index):
        if self.cap is None or self.frame_total<=0:return
        index=max(0,min(int(index),self.frame_total-1))
        if self.last_frame is None or index!=self.frame_index:
            # Sequential playback avoids expensive seek on every frame; backward
            # and slider seeks explicitly reposition OpenCV to the frame index.
            if index != self.frame_index+1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok,frame=self.cap.read()
            if not ok:
                self.pause()
                self.frame_info.setText(f'Cannot decode frame {index}')
                return
            self.last_frame=frame
        self.frame_index=index
        self.timeline.blockSignals(True)
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)
        self.repaint_frame()

    def repaint_frame(self, *_):
        if self.last_frame is None or self.current is None:return
        record=self.frames.get(self.frame_index)
        shown=draw_pose_overlay(self.last_frame,record,draw_bbox=self.box_check.isChecked(),
                draw_skeleton=self.kp_check.isChecked(),all_people=self.all_check.isChecked())
        self.screen.set_frame(shown)
        ms=record.get('relative_ms') if record else None
        clock=f'{float(ms)/1000:.2f}s' if isinstance(ms,(int,float)) else f'{self.frame_index/self.fps:.2f}s (video)'
        status=record.get('status','unknown') if record else 'no keypoints'
        self.frame_info.setText(f'Frame {self.frame_index+1}/{self.frame_total}  ·  {clock}  ·  {status}')

    def close_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap=None
        self.last_frame=None
        self.frames={}
        self.frame_total=0

    def shutdown(self):
        self.pause()
        self.close_capture()
