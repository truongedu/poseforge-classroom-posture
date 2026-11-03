"""PoseForge Phase 3.3 inference monitor: python -m inference.gui.

Qt owns presentation only. Preview and inference use the existing capture/model
modules on background threads; Qt widgets are never modified by worker threads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import QObject, Qt, QSettings, QTimer, Signal, QRect, QPointF
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPushButton, QSpinBox, QDoubleSpinBox, QVBoxLayout, QWidget,
)

from inference.ui_runtime import InferenceSession, PreviewSession, Source, Settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POSE = ROOT/'artifacts/pretrained/yolo11n-pose.pt'
DEFAULT_CLASSIFIER = ROOT/'artifacts/models/mlp_wide/v002'
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
NA_METRICS = 'YOLO N/A  ·  MLP N/A  ·  Pipeline N/A  ·  Live N/A'


def available_classifiers(root=ROOT):
    """Offer only real, installed final checkpoints, never invented model aliases."""
    return sorted({p.parent for p in (root / 'artifacts/models').glob('*/v*/model_final.pt')})


def choose_model_paths(pose_arg, classifier_arg, store):
    """Explicit launch flags override saved GUI model choices."""
    pose = pose_arg if pose_arg is not None else store.value('pose_model', str(DEFAULT_POSE))
    classifier = (classifier_arg if classifier_arg is not None else
                  store.value('classifier_model', str(DEFAULT_CLASSIFIER)))
    return Path(pose).expanduser(), Path(classifier).expanduser()


class FrameView(QLabel):
    """Letterboxed frame with normalized polygon ROI on top; never crops YOLO input."""
    roi_changed = Signal(list)

    def __init__(self):
        super().__init__('NO INPUT · Select source, then Preview or START')
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(235, 165)
        self.setStyleSheet('background:#111c2d;color:#9eabbc;border:1px solid #2a3446;')
        self.image = None
        self.roi = []
        self.edit_roi = False

    def set_image(self, frame):
        h, w = frame.shape[:2]
        # Qt presentation only: avoid copying a full 1080p frame for each tile.
        # Aspect ratio and normalized ROI coordinates remain unchanged.
        scale = min(1., max(1,self.width())/w, max(1,self.height())/h)
        if scale < 1.:
            frame = cv2.resize(frame, (max(1,round(w*scale)), max(1,round(h*scale))),
                               interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
        self.setText('')
        self.update()

    def clear_image(self):
        self.image = None
        self.setText('NO INPUT')
        self.update()

    def set_roi(self, points):
        self.roi = list(points)
        self.roi_changed.emit(self.roi[:])
        self.update()

    def _target(self):
        if self.image is None:
            return None
        scale = min(self.width()/self.image.width(), self.height()/self.image.height())
        w, h = max(1, round(self.image.width()*scale)), max(1, round(self.image.height()*scale))
        return QRect((self.width()-w)//2, (self.height()-h)//2, w, h)

    def paintEvent(self, event):
        if self.image is None:
            return super().paintEvent(event)
        rect = self._target()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#111c2d'))
        painter.drawImage(rect, self.image)
        if self.roi:
            points = [QPointF(rect.x()+x*rect.width(), rect.y()+y*rect.height()) for x,y in self.roi]
            painter.setPen(QPen(QColor('#ffca65'), 2))
            for a,b in zip(points, points[1:]):
                painter.drawLine(a,b)
            if len(points) >= 3:
                painter.drawLine(points[-1], points[0])
            painter.setBrush(QColor('#ffca65'))
            for point in points:
                painter.drawEllipse(point, 4, 4)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if not self.edit_roi or self.image is None or event.button() != Qt.MouseButton.LeftButton:
            return
        rect = self._target()
        if rect is None or not rect.contains(event.position().toPoint()):
            return
        nx = (event.position().x()-rect.x())/max(1,rect.width())
        ny = (event.position().y()-rect.y())/max(1,rect.height())
        self.set_roi([*self.roi, (min(1.,max(0.,nx)), min(1.,max(0.,ny)))])


class Bridge(QObject):
    arrived = Signal(str, int, object)


class SourcePanel(QWidget):
    def __init__(self, index, remove_callback, pose_callback):
        super().__init__()
        self.setObjectName('source-panel')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(5)
        top = QHBoxLayout()
        self.title = QLabel(f'Source {index+1}')
        self.title.setStyleSheet('font-weight:600;')
        self.kind = QComboBox()
        self.kind.addItems(['camera', 'video', 'image'])
        self.value = QLineEdit()
        self.value.setPlaceholderText('0 / file / RTSP or IP URL')
        self.browse = QPushButton('Browse')
        self.browse.clicked.connect(self.choose_file)
        self.remove = QPushButton('×')
        self.remove.setToolTip('Remove this source')
        self.remove.setFixedWidth(28)
        self.remove.clicked.connect(lambda: remove_callback(self))
        top.addWidget(self.title)
        top.addWidget(self.kind)
        top.addWidget(self.value, 1)
        top.addWidget(self.browse)
        top.addWidget(self.remove)
        layout.addLayout(top)
        self.view = FrameView()
        layout.addWidget(self.view, 1)
        footer = QHBoxLayout()
        self.decision = QLabel('NO PREDICTION')
        self.decision.setStyleSheet('font-weight:600;')
        footer.addWidget(self.decision, 1)
        self.pose_preview = QCheckBox('Show pose in Preview')
        self.pose_preview.setToolTip('Pose boxes/skeleton for ONE source at a time; no MLP')
        self.pose_preview.toggled.connect(lambda checked: pose_callback(self, checked))
        footer.addWidget(self.pose_preview)
        self.draw_roi = QCheckBox('Draw ROI')
        self.draw_roi.toggled.connect(lambda on: setattr(self.view, 'edit_roi', on))
        footer.addWidget(self.draw_roi)
        clear = QPushButton('Clear ROI')
        clear.clicked.connect(lambda: self.view.set_roi([]))
        footer.addWidget(clear)
        layout.addLayout(footer)
        self.metrics = QLabel(NA_METRICS)
        self.metrics.setStyleSheet('color:#9baec9;')
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select source', str(Path.home()),
                                               'Images / Videos (*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.avi *.mkv *.mov);;All files (*)')
        if path:
            self.value.setText(path)
            self.kind.setCurrentText('image' if Path(path).suffix.lower() in IMAGE_SUFFIXES else 'video')

    def clear_output(self):
        self.view.clear_image()
        self.decision.setText('NO PREDICTION')
        self.metrics.setText(NA_METRICS)

    def set_source_locked(self, locked):
        self.kind.setEnabled(not locked)
        self.value.setReadOnly(locked)
        self.browse.setEnabled(not locked)
        self.remove.setEnabled(not locked)


class AdvancedDialog(QDialog):
    """Settings + discovered installed checkpoints; no RTSP URLs are persisted."""
    def __init__(self, settings, pose, model, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Advanced Settings · Model & Inference')
        self.resize(520, 330)
        outer = QVBoxLayout(self)
        form = QFormLayout()
        self.pose = QComboBox(); self.pose.setEditable(True)
        pose_choices = sorted((ROOT/'artifacts/pretrained').glob('*.pt'))
        self.pose.addItems(str(p) for p in pose_choices)
        self.pose.setCurrentText(str(pose))
        self.pose_browse = QPushButton('Browse Pose…')
        self.pose_browse.clicked.connect(self.choose_pose)
        row = QHBoxLayout(); row.addWidget(self.pose, 1); row.addWidget(self.pose_browse)
        form.addRow('YOLO Pose weights', row)
        self.model = QComboBox(); self.model.setEditable(True)
        self.model.addItems(str(p) for p in available_classifiers())
        self.model.setCurrentText(str(model))
        self.model_browse = QPushButton('Browse Classifier…')
        self.model_browse.clicked.connect(self.choose_model)
        row = QHBoxLayout(); row.addWidget(self.model, 1); row.addWidget(self.model_browse)
        form.addRow('Classifier checkpoint folder (MLP / GRU)', row)
        self.stable = QSpinBox(); self.stable.setRange(1,30)
        self.conf = QDoubleSpinBox(); self.conf.setRange(.01,1.); self.conf.setDecimals(2); self.conf.setSingleStep(.05)
        self.rate = QSpinBox(); self.rate.setRange(1,60)
        self.stale = QDoubleSpinBox(); self.stale.setRange(.5,120); self.stale.setSuffix(' s')
        self.overlay = QCheckBox('Skeleton on inference frames sent to UI')
        for label, widget in (('Stable frames',self.stable), ('Minimum confidence',self.conf),
                              ('UI render FPS (not model limit)',self.rate),
                              ('Camera stale timeout',self.stale), ('Overlay',self.overlay)):
            form.addRow(label, widget)
        outer.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel |
                                   QDialogButtonBox.StandardButton.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self.restore_defaults)
        outer.addWidget(buttons)
        self.populate(settings)

    def restore_defaults(self):
        self.populate(Settings())
        self.pose.setCurrentText(str(DEFAULT_POSE))
        self.model.setCurrentText(str(DEFAULT_CLASSIFIER))

    def choose_pose(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select YOLO Pose weights',str(ROOT/'artifacts/pretrained'), 'Pose weights (*.pt)')
        if path: self.pose.setCurrentText(path)

    def choose_model(self):
        path = QFileDialog.getExistingDirectory(self, 'Select final classifier checkpoint folder', str(ROOT/'artifacts/models'))
        if path: self.model.setCurrentText(path)

    def populate(self, settings):
        self.stable.setValue(settings.stable_frames)
        self.conf.setValue(settings.min_conf)
        self.rate.setValue(settings.render_fps)
        self.stale.setValue(settings.stale_seconds)
        self.overlay.setChecked(settings.overlay)

    def result_settings(self, behavior):
        return Settings(self.stable.value(), self.conf.value(), self.overlay.isChecked(),
                        self.rate.value(), self.stale.value(), behavior).validate()

    def result_models(self):
        return Path(self.pose.currentText().strip()).expanduser(), Path(self.model.currentText().strip()).expanduser()

    def accept(self):
        pose, model = self.result_models()
        if not pose.is_file() or not (model / 'model_final.pt').is_file():
            QMessageBox.warning(self, 'Invalid models',
                                'Select a YOLO Pose .pt file and a classifier folder containing model_final.pt.')
            return
        super().accept()


class MainWindow(QMainWindow):
    def __init__(self, pose, model, device='0', cls_device='cuda:0'):
        super().__init__()
        self.setWindowTitle('PoseForge · Inference Monitor')
        self.resize(1370, 850)
        self.pose, self.model, self.device, self.cls_device = Path(pose), Path(model), device, cls_device
        self.store = QSettings('PoseForge', 'InferenceMonitor')
        self.settings = self._load_settings()
        self.session = None
        self.preview = None
        self.case_paused = False
        self.pending_start = False
        self.last_error = None
        self.bridge = Bridge()
        self.bridge.arrived.connect(self.on_event)

        base = QWidget(); self.setCentralWidget(base)
        outer = QVBoxLayout(base)
        outer.setContentsMargins(9, 8, 9, 8)
        outer.setSpacing(5)
        top = QHBoxLayout()
        self.state = QLabel('READY')
        self.state.setStyleSheet('color:#9bdcae;font-weight:700;')
        self.model_hint = QLabel()
        self._model_hint()
        self.model_hint.setStyleSheet('color:#aebed1;')
        self.add = QPushButton('+ Add Source')
        self.add.clicked.connect(self.add_source)
        self.advanced = QPushButton('Advanced Settings')
        self.advanced.clicked.connect(self.advanced_settings)
        for widget in (self.state, self.model_hint): top.addWidget(widget)
        top.addStretch(1)
        top.addWidget(self.add)
        top.addWidget(self.advanced)
        outer.addLayout(top)

        self.grid = QGridLayout()
        self.grid.setContentsMargins(0,0,0,0)
        self.grid.setSpacing(7)
        outer.addLayout(self.grid, 1)
        self.panels = []
        self.add_source()

        bar = QHBoxLayout()
        self.behavior = QComboBox(); self.behavior.addItems(['fast','guarded'])
        self.behavior.setCurrentText(self.settings.behavior)
        self.mode = QComboBox(); self.mode.addItems(['Testcase','Continuous'])
        self.preview_button = QPushButton('Preview Sources')
        self.preview_button.clicked.connect(self.toggle_preview)
        self.start = QPushButton('START')
        self.start.clicked.connect(self.start_clicked)
        self.end = QPushButton('END')
        self.end.clicked.connect(self.end_clicked)
        for widget in (QLabel('Display'),self.behavior,QLabel('Mode'),self.mode,
                       self.preview_button,self.start,self.end):
            bar.addWidget(widget)
        bar.addStretch(1)
        outer.addLayout(bar)
        self.end.setEnabled(False)
        self.setStyleSheet('QWidget {font-size:12px;} QPushButton {padding:5px;} QLineEdit {padding:5px;}')

    def _model_hint(self):
        self.model_hint.setText(f'YOLO {self.pose.name}  ·  Classifier {self.model.parent.name}/{self.model.name} (PT FP32)')

    def _load_settings(self):
        try:
            return Settings(stable_frames=int(self.store.value('stable_frames',5)),
                            min_conf=float(self.store.value('min_conf',.55)),
                            overlay=str(self.store.value('overlay','true')).lower() == 'true',
                            render_fps=int(self.store.value('render_fps',15)),
                            stale_seconds=float(self.store.value('stale_seconds',8.)),
                            behavior=str(self.store.value('behavior','fast'))).validate()
        except (ValueError, TypeError):
            return Settings()

    def _save_settings(self):
        for key in ('stable_frames','min_conf','overlay','render_fps','stale_seconds','behavior'):
            self.store.setValue(key, getattr(self.settings, key))
        # Persist model selections only. Never store source values (RTSP credentials).
        self.store.setValue('pose_model', str(self.pose))
        self.store.setValue('classifier_model', str(self.model))

    def add_source(self):
        if self.session or self.preview or len(self.panels) >= 4:
            return
        panel = SourcePanel(len(self.panels), self.remove_source, self._on_pose_checkbox)
        panel.view.roi_changed.connect(lambda points, p=panel: self._on_roi(p, points))
        self.panels.append(panel)
        self._rebuild_grid()

    def remove_source(self, panel):
        if self.session or self.preview or len(self.panels) <= 1:
            return
        if panel not in self.panels: return
        self.panels.remove(panel)
        self.grid.removeWidget(panel)
        panel.deleteLater()
        self._rebuild_grid()

    def _rebuild_grid(self):
        for p in self.panels:
            self.grid.removeWidget(p)
        count = len(self.panels)
        cols = 1 if count == 1 else 2
        for i, panel in enumerate(self.panels):
            panel.title.setText(f'Source {i+1}')
            self.grid.addWidget(panel, i // cols, i % cols)
        self.add.setEnabled(count < 4 and self.session is None and self.preview is None)
        for p in self.panels:
            p.remove.setEnabled(count > 1 and self.session is None and self.preview is None)

    def _on_pose_checkbox(self, panel, checked):
        if not checked or panel not in self.panels:
            if self.preview and all(not p.pose_preview.isChecked() for p in self.panels):
                self.preview.set_pose_index(-1)
            return
        for p in self.panels:
            if p is not panel:
                p.pose_preview.blockSignals(True)
                p.pose_preview.setChecked(False)
                p.pose_preview.blockSignals(False)
        if self.preview:
            if not self._validate_models(classifier=False):
                panel.pose_preview.blockSignals(True)
                panel.pose_preview.setChecked(False)
                panel.pose_preview.blockSignals(False)
                self.preview.set_pose_index(-1)
                return
            self.preview.set_pose_index(self.panels.index(panel))

    def _on_roi(self, panel, points):
        if self.session and panel in self.panels:
            self.session.update_roi(self.panels.index(panel), points)

    def _get_sources(self):
        sources = []
        for i, p in enumerate(self.panels):
            value = p.value.text().strip()
            if not value:
                QMessageBox.warning(self,'Missing source',f'Enter the source for tile {i+1}.')
                return None
            kind = p.kind.currentText()
            if kind in ('image','video') and not Path(value).expanduser().is_file():
                QMessageBox.warning(self,'Invalid source',f'Source {i+1}: local file not found.')
                return None
            sources.append(Source(kind, value))
        return sources

    def _validate_models(self, classifier=False):
        if not self.pose.is_file():
            QMessageBox.warning(self,'Model unavailable','YOLO Pose checkpoint does not exist. Select it in Advanced Settings.')
            return False
        if classifier and not (self.model / 'model_final.pt').is_file():
            QMessageBox.warning(self,'Model unavailable','MLP final checkpoint not found. Select an installed version in Advanced Settings.')
            return False
        return True

    def _lock_controls(self, locked):
        self.add.setEnabled(not locked and len(self.panels)<4)
        self.advanced.setEnabled(not locked)
        self.behavior.setEnabled(not locked)
        self.mode.setEnabled(not locked)
        for panel in self.panels:
            panel.set_source_locked(locked)
            panel.pose_preview.setEnabled(self.preview is not None and self.session is None)
        self._rebuild_grid()

    def advanced_settings(self):
        if self.session is not None or self.preview is not None:
            QMessageBox.information(self,'Settings','Close the session/preview before changing the model or settings.')
            return
        dialog = AdvancedDialog(self.settings,self.pose,self.model,self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings = dialog.result_settings(self.behavior.currentText())
            self.pose, self.model = dialog.result_models()
            self._model_hint()
            self._save_settings()

    def toggle_preview(self):
        if self.session is not None or self.pending_start:
            return
        if self.preview is not None:
            self.preview.close(timeout=0.)
            self.preview_button.setEnabled(False)
            self.state.setText('CLOSING PREVIEW...')
            return
        sources = self._get_sources()
        if sources is None: return
        if any(p.pose_preview.isChecked() for p in self.panels) and not self._validate_models(classifier=False):
            return
        self.preview = PreviewSession(sources,self.pose,self.device,emit=self.bridge.arrived.emit)
        selected = next((i for i,p in enumerate(self.panels) if p.pose_preview.isChecked()), -1)
        self.preview.set_pose_index(selected)
        self._lock_controls(True)
        for p in self.panels:
            p.clear_output()
            p.pose_preview.setEnabled(True)
        self.preview_button.setText('Stop Preview')
        self.state.setText('PREVIEW · raw frames (no classification)')
        self.preview.start()

    def start_clicked(self):
        if self.session is not None:
            if self.case_paused and self.session.next_case():
                self.case_paused = False
                self.start.setEnabled(False)
                self.end.setEnabled(True)
                self.state.setText('STARTING NEXT CASE...')
            return
        if self.preview is not None:
            self.pending_start = True
            self.preview.close(timeout=0.)
            self.preview_button.setEnabled(False)
            self.start.setEnabled(False)
            self.state.setText('SWITCHING FROM PREVIEW...')
            QTimer.singleShot(75, self._start_after_preview)
            return
        self._start_new()

    def _start_after_preview(self):
        if self.preview is not None:
            if not self.preview.close(timeout=0.):
                QTimer.singleShot(75,self._start_after_preview)
                return
            self.preview = None
        self.pending_start = False
        self._start_new()

    def _start_new(self):
        sources = self._get_sources()
        if sources is None or not self._validate_models(classifier=True):
            self._set_idle()
            return
        self.settings = Settings(self.settings.stable_frames,self.settings.min_conf,
                                 self.settings.overlay,self.settings.render_fps,
                                 self.settings.stale_seconds,self.behavior.currentText())
        self._save_settings()
        for p in self.panels: p.clear_output()
        self.last_error = None
        self.session = InferenceSession(sources,self.pose,self.model,self.device,self.cls_device,
                                        settings=self.settings,
                                        rois={i:p.view.roi for i,p in enumerate(self.panels)},
                                        emit=self.bridge.arrived.emit)
        self.case_paused = False
        self._lock_controls(True)
        self.preview_button.setEnabled(False)
        self.start.setEnabled(False)
        self.end.setEnabled(True)
        self.state.setText('STARTING...')
        self.session.start(testcase=(self.mode.currentText()=='Testcase'))

    def end_clicked(self):
        if self.session is None:
            return
        self.end.setEnabled(False)
        if self.session.testcase:
            self.session.finish_case()  # pause + reset, START resumes without model reload
            self.state.setText('ENDING CASE...')
        else:
            self.session.stop.set()
            self.state.setText('ENDING SESSION...')

    def _set_idle(self):
        self.case_paused = False
        self.start.setEnabled(True)
        self.end.setEnabled(False)
        self.preview_button.setEnabled(True)
        self.preview_button.setText('Preview Sources')
        self._lock_controls(False)

    def on_event(self, kind, i, payload):
        if kind == 'preview_frame':
            if self.preview is not None and not self.pending_start and 0 <= i < len(self.panels):
                self.panels[i].view.set_image(payload)
            return
        if kind == 'preview_error':
            self.state.setText(payload)
            return
        if kind == 'preview_state':
            if self.preview is not None and not self.pending_start:
                self.preview = None
                self._set_idle()
                if not self.state.text().endswith('failed (check source/model)'):
                    self.state.setText('READY')
            return
        if kind == 'state' and payload == 'READY':
            self.session = None
            self._set_idle()
            self.state.setText(self.last_error or 'READY')
            return
        if kind == 'case':
            phase, case_id = payload
            if phase == 'ready':
                self.case_paused = True
                self.state.setText(f'READY · Case {case_id} ended')
                self.start.setEnabled(True)
                self.end.setEnabled(False)
                for p in self.panels: p.clear_output()
            elif phase == 'running':
                self.case_paused = False
                self.state.setText(f'RUNNING · Case {case_id}')
                self.start.setEnabled(False)
                self.end.setEnabled(True)
            elif phase == 'complete':
                self.state.setText(f'Case {case_id} complete · source EOF')
            return
        if kind == 'error':
            self.last_error = str(payload)
            self.state.setText(self.last_error)
            return
        if kind == 'status':
            if i < 0 and payload in ('RUNNING','STOPPED','DONE'):
                if payload == 'RUNNING' and self.mode.currentText()=='Continuous':
                    self.state.setText('RUNNING · Continuous')
            elif 0 <= i < len(self.panels) and payload in ('READY','RECONNECTING','DONE','ROI UPDATED / READY'):
                self.panels[i].decision.setText('NO PREDICTION')
                self.panels[i].metrics.setText(NA_METRICS)
            return
        if kind == 'frame' and 0 <= i < len(self.panels):
            frame,info,metrics = payload
            p = self.panels[i]
            p.view.set_image(frame)
            p.decision.setText(f"{info.get('display_label','NO POSE')}  ·  raw confidence {info.get('score',0)*100:.1f}%")
            def fmt(key):
                val = metrics.get(key)
                return 'N/A' if val is None else f'{val:.1f}'
            p.metrics.setText(f"YOLO {fmt('yolo')}  ·  {str(metrics.get('classifier_family','mlp')).upper()} {fmt('mlp')}  ·  Pipeline {fmt('pipeline')}  ·  Live {fmt('live')} FPS")

    def closeEvent(self, event):
        active = False
        if self.preview is not None:
            active = not self.preview.close(timeout=0.) or active
        if self.session is not None:
            active = not self.session.close(timeout=0.) or active
        if active:
            event.ignore()
            self.state.setText('CLOSING WORKERS...')
            QTimer.singleShot(150,self.close)
        else:
            event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose',type=Path,default=None,
                        help='Override saved YOLO checkpoint (persisted on START)')
    parser.add_argument('--classifier',type=Path,default=None,
                        help='Override saved MLP/GRU folder (persisted on START)')
    parser.add_argument('--device',default='0')
    parser.add_argument('--cls-device',default='cuda:0')
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    store = QSettings('PoseForge', 'InferenceMonitor')
    # Explicit CLI paths take precedence over previously chosen GUI models.
    pose, model = choose_model_paths(args.pose, args.classifier, store)
    win = MainWindow(pose, model, args.device, args.cls_device)
    win.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
