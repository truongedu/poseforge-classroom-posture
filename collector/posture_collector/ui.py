"""Modern dark PySide6 desktop GUI for multi-DroidCam capture."""
from __future__ import annotations
from datetime import datetime
from collections import deque
from pathlib import Path
import queue
import re
import sys
import threading
import time
from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QFont, QTextCursor, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication,QMainWindow,QWidget,QLineEdit,QComboBox,QVBoxLayout,QHBoxLayout,
    QTableWidget,QTableWidgetItem,QHeaderView,QMessageBox,
    QPlainTextEdit,QTextEdit,QScrollArea,QTreeWidget,QTreeWidgetItem,QSpinBox,
    QListWidget, QAbstractItemView, QStackedWidget, QSizePolicy, QFrame, QInputDialog, QDialog,
)
from .storage import DataStore
from .capture import CollectorCore
from .pose import PoseRunner,pose_to_preview_points
from .tracking import LatestFrames,TrackSelection,TrackingJournal,parse_track_choices
from .storage import now_iso
import uuid
from .process import extract_video
from .dataset_audit import audit_dataset
from .pose_quality import quality_rows, save_reports, aggregate
from .reextract import list_targets, reextract_targets
from .split_tools import make_split_plan, export_plan
from .replay import ReplayInspector
from .curation import export_clean_dataset, read_edits

from .ui_components import THEME, CameraCard, label, panel, button, scrollable
from .ui_layout import DashboardLayout, POSTURE_NAMES
import html

class PosePreview:
    """One inference owner, independent loaded model/tracker per camera.

    Only the newest frame for each camera is queued. Generation checks discard
    in-flight results from disconnected/replaced streams. No worker touches Qt.
    """
    def __init__(self,backend,model_file,device='auto'):
        self.backend=backend;self.model=model_file;self.device=device
        self.queue=LatestFrames();self.stop_flag=threading.Event()
        self.lock=threading.RLock();self.results={};self.tokens={};self.submitted={};self.error=None
        self.thread=threading.Thread(target=self.run,name='pose-tracking',daemon=True)
        self.thread.start()

    def reset_camera(self,cid,token):
        with self.lock:
            self.tokens[cid]=token;self.results.pop(cid,None);self.submitted.pop(cid,None);self.queue.discard(cid)

    def submit(self,cid,frame,recv_ns,token='default',frame_index=-1,utc_stamp=None):
        with self.lock:
            if cid not in self.tokens:self.tokens[cid]=token
            if self.tokens[cid]!=token:return
            key=(token,recv_ns,frame_index)
            if self.submitted.get(cid)==key:return
            self.submitted[cid]=key
            self.queue.put(cid,(frame.copy(),recv_ns,token,frame_index,utc_stamp))

    def run(self):
        runners={};epochs={};ticks={};failed={}
        try:
            while not self.stop_flag.is_set():
                try:cid,item=self.queue.get()
                except queue.Empty:continue
                image,stamp,token,index,utc_stamp=item
                with self.lock:
                    if self.tokens.get(cid)!=token:continue
                if failed.get(cid)==token:continue
                try:
                    if cid not in runners:
                        runners[cid]=PoseRunner(self.backend,self.model,video=False,device=self.device)
                    pose=runners[cid]
                    if epochs.get(cid)!=token:
                        if cid in epochs and hasattr(pose,'reset_tracking'):pose.reset_tracking()
                        epochs[cid]=token;ticks[cid]=deque(maxlen=20)
                    started=time.monotonic()
                    if self.backend=='yolo' and hasattr(pose,'track_frame'):
                        persons=pose.track_frame(image);points=[]
                    else:
                        people=pose.detect(image)
                        points=pose_to_preview_points(people,self.backend,image.shape[1],image.shape[0])
                        persons=[]
                    ended=time.monotonic();ticks[cid].append(ended)
                    span=ticks[cid][-1]-ticks[cid][0]
                    fps=(len(ticks[cid])-1)/span if span>0 else 0.0
                    result={'persons':persons,'points':points,'timestamp_ns':stamp,
                            'frame_index':index,'host_utc':utc_stamp,'stream_token':token,
                            'track_session':getattr(pose,'tracking_session',token),
                            'tracking_fps':fps,'latency_ms':(ended-started)*1000,'error':None}
                except Exception as exc:
                    failed[cid]=token
                    result={'error':f'{type(exc).__name__}: {exc}','stream_token':token,
                            'timestamp_ns':stamp,'persons':[],'points':[],'tracking_fps':0.0}
                with self.lock:
                    if self.tokens.get(cid)==token:self.results[cid]=result
        finally:
            for runner in runners.values():
                if hasattr(runner,'close'):runner.close()

    def get(self,cid):
        with self.lock:return self.results.get(cid)

    def stop(self):self.stop_flag.set()

class Window(DashboardLayout, QMainWindow):
    def __init__(self,root):
        super().__init__();self.setWindowTitle('PoseForge • Multi-Camera Posture Dataset Studio')
        self.setMinimumSize(960, 560)
        self.resize(1600, 900)
        self.store=DataStore(root);self.core=CollectorCore(self.store,fps=20)
        self.cards={};self.preview_worker=None;self.pose_last_sent={};self.last_seen_idx={}
        self.async_events=queue.Queue();self.busy=False
        self.record_started=None;self.last_active=False;self.last_finalizing=False
        self.last_saved_take=None
        self.camera_busy=False; self.closing=False
        self.ui_events=queue.Queue(); self.take_cache=[]
        self.subject_cache={}; self.pose_workers=[]; self.offline_thread=None
        self.fps_samples={}
        self.track_selections={};self.tracking_tokens={};self.last_prediction={}
        self.selection_states={}
        self.tracking_context=None;self.tracking_epoch=uuid.uuid4().hex
        self.tracking_journal=None;self.journal_threads=[];self.journal_errors=queue.Queue()
        self.nav=[]
        self.build()
        self.refresh_subjects();self.refresh_labels();self.populate_cameras();self.refresh_library()
        self.rebuild_camera_cards(self.store.cameras())
        self.subjects_combo.currentTextChanged.connect(self.refresh_progress)
        self.label_combo.currentIndexChanged.connect(self.refresh_take_number)
        self.session_edit.textChanged.connect(self.refresh_take_number)
        self.session_edit.textChanged.connect(self.refresh_controls)
        self.subjects_combo.currentTextChanged.connect(self.refresh_controls)
        QApplication.instance().installEventFilter(self)
        self.refresh_controls()
        self.refresh_timer=QTimer(self);self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start(100)

    def build_pose_controls(self):
        """Pose settings remain beside the recording controls; no second setup page."""
        card, layout = panel()
        layout.addWidget(label('Pose & tracking', 'h2'))
        self.pose_preset = QComboBox()
        self.pose_preset.setAccessibleName('Pose checkpoint')
        self.pose_preset.addItem('Off · Video only', ('none', ''))
        assets = Path(__file__).resolve().parents[2] / 'artifacts' / 'pretrained'
        fallback = Path(__file__).resolve().parents[1] / 'models'
        found = set()
        for directory in (assets, fallback):
            if not directory.is_dir():
                continue
            for file in sorted(directory.iterdir()):
                if not file.is_file() or file.name in found:
                    continue
                if not (file.name.lower().endswith(('-pose.pt', '-pose.onnx')) or
                        file.name.lower().startswith('pose_landmarker') and file.suffix == '.task'):
                    continue
                backend = 'yolo' if file.suffix.lower() in ('.pt', '.onnx') else 'mediapipe'
                self.pose_preset.addItem(file.stem, (backend, str(file.resolve())))
                found.add(file.name)
        layout.addWidget(self.pose_preset)
        settings = QHBoxLayout()
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItem('ByteTrack', 'bytetrack')
        self.tracker_combo.setToolTip('Only tracker currently implemented and validated')
        settings.addWidget(self.tracker_combo)
        self.preview_device = QComboBox()
        self.preview_device.addItems(['Auto device', 'CPU', 'CUDA:0'])
        settings.addWidget(self.preview_device)
        layout.addLayout(settings)
        # Compatibility with V1 auto-extraction and preview pipeline.
        self.preview_combo = QComboBox()
        self.preview_combo.addItems(['Off', 'YOLO-Pose', 'MediaPipe Pose'])
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(['yolo', 'mediapipe'])
        self.offline_model = QLineEdit()
        self.preview_model = QLineEdit()
        self.pose_preset.currentIndexChanged.connect(self.set_pose_preset)
        actions = QHBoxLayout()
        self.enable_tracking_btn = button('Enable pose', self.enable_pose_preview, 'primary')
        self.disable_tracking_btn = button('Disable pose', self.disable_pose_preview)
        actions.addWidget(self.enable_tracking_btn)
        actions.addWidget(self.disable_tracking_btn)
        layout.addLayout(actions)
        self.set_pose_preset(self.pose_preset.currentIndex())
        return card

    def build_cameras(self):
        page = QWidget()
        content = QVBoxLayout(page)
        content.setContentsMargins(24, 20, 24, 20)
        content.setSpacing(12)
        content.addLayout(self.top_header('Cameras', 'Edit URLs and connect streams'))
        source_card, source = panel()
        source.addWidget(label('Camera sources', 'h2'))
        self.cameras_table = QTableWidget(0, 5)
        self.cameras_table.setHorizontalHeaderLabels(
            ['Camera ID', 'Name', 'View', 'Stream URL / local video', 'Enabled'])
        self.cameras_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.cameras_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.cameras_table.setMinimumHeight(145)
        self.cameras_table.setMaximumHeight(215)
        self.cameras_table.verticalHeader().hide()
        self.cameras_table.setWordWrap(False)
        for col, width in ((0, 90), (1, 150), (2, 100), (4, 90)):
            self.cameras_table.setColumnWidth(col, width)
        source.addWidget(self.cameras_table)
        buttons = QHBoxLayout()
        self.add_camera_btn = button('+ Add', self.add_camera_row)
        self.remove_camera_btn = button('Remove', self.remove_camera_row)
        buttons.addWidget(self.add_camera_btn)
        buttons.addWidget(self.remove_camera_btn)
        buttons.addStretch()
        self.disconnect_btn = button('Disconnect', self.disconnect_cameras)
        buttons.addWidget(self.disconnect_btn)
        self.connect_btn = button('Save & Connect', self.connect_cameras, 'primary')
        buttons.addWidget(self.connect_btn)
        source.addLayout(buttons)
        content.addWidget(source_card)
        content.addStretch()
        return scrollable(page)

    def build_tools(self):
        """Dataset browser, not a flat 88-row table."""
        page = QWidget()
        content = QVBoxLayout(page)
        content.setContentsMargins(24, 20, 24, 20)
        content.setSpacing(12)
        content.addLayout(self.top_header('Dataset', 'Browse and validate recordings'))
        self.root_label = QLineEdit(str(self.store.root))
        self.root_label.setReadOnly(True)
        content.addWidget(self.root_label)
        actions = QHBoxLayout()
        actions.addWidget(button('Open folder', self.open_folder))
        actions.addWidget(button('Refresh', self.refresh_library))
        actions.addWidget(button('Validate all', lambda: self.validate_clicked(person_only=False), 'primary'))
        actions.addWidget(button('Validate person', lambda: self.validate_clicked(person_only=True)))
        actions.addWidget(button('Pose quality', self.show_pose_quality))
        actions.addWidget(button('Export clean ZIP', self.export_clean_clicked, 'primary'))
        actions.addStretch()
        content.addLayout(actions)
        self.dataset_summary = label('Select a person to inspect sessions and takes.', 'muted')
        content.addWidget(self.dataset_summary)
        library_panel, library_layout = panel()
        library_layout.addWidget(label('Person → Session → Label → Take', 'h2'))
        self.library = QTreeWidget()
        self.library.setObjectName('datasetTree')
        self.library.setExpandsOnDoubleClick(False)
        self.library.setAnimated(True)
        self.library.setIndentation(23)
        self.library.setUniformRowHeights(True)
        self.library.itemClicked.connect(self.toggle_library_branch)
        self.library.itemDoubleClicked.connect(self.open_dataset_take_in_replay)
        self.library.setColumnCount(2)
        self.library.setHeaderLabels(['Recording', 'Status / count'])
        self.library.setMinimumHeight(350)
        self.library.setAlternatingRowColors(True)
        self.library.itemSelectionChanged.connect(self.on_library_selection)
        library_layout.addWidget(self.library)
        content.addWidget(library_panel, 1)
        self.tool_log = QPlainTextEdit()
        self.tool_log.setReadOnly(True)
        self.tool_log.setMaximumHeight(90)
        content.addWidget(self.tool_log)
        return scrollable(page)

    def build_replay(self):
        self.replay_inspector = ReplayInspector(self.store.root, parent=self)
        return self.replay_inspector

    def build_utilities(self):
        """One focused task at a time; no exposed controls before choosing a task."""
        page = QWidget()
        content = QVBoxLayout(page)
        content.setContentsMargins(24, 20, 24, 20)
        content.setSpacing(14)
        content.addLayout(self.top_header('Tools', ''))

        choices = QHBoxLayout(); choices.setSpacing(14)
        self.reextract_task = button('Re-extract Pose\nMissing / selected', role='toolExtract')
        self.split_task = button('Split Planning\nSubject / within-person', role='toolSplit')
        for task in (self.reextract_task, self.split_task):
            task.setCheckable(True)
            task.setMinimumHeight(100)
            task.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            choices.addWidget(task, 1)
        self.reextract_task.clicked.connect(lambda checked: self.set_tool_task('reextract' if checked else None))
        self.split_task.clicked.connect(lambda checked: self.set_tool_task('split' if checked else None))
        content.addLayout(choices)

        self.tool_task_stack = QStackedWidget()
        self.tool_task_stack.setObjectName('toolTaskStack')
        self.tool_task_stack.setMinimumSize(0, 0)
        self.tool_task_stack.addWidget(self.build_reextract_task())
        self.tool_task_stack.addWidget(self.build_split_task())
        content.addWidget(self.tool_task_stack)
        self.tool_task_stack.hide()
        content.addStretch(1)
        return scrollable(page)

    def set_tool_task(self, task):
        """Exactly one expanded tools pane, or none."""
        chosen = {'reextract': 0, 'split': 1}.get(task)
        self.reextract_task.setChecked(chosen == 0)
        self.split_task.setChecked(chosen == 1)
        self.tool_task_stack.setVisible(chosen is not None)
        if chosen is not None:
            self.tool_task_stack.setCurrentIndex(chosen)
            if chosen == 0:
                self.refresh_reextract_options()

    def build_reextract_task(self):
        card, layout = panel()
        self.reextract_mode = QComboBox()
        self.reextract_mode.addItem('Missing only (safe)', 'missing')
        self.reextract_mode.addItem('Reprocess selection (overwrite)', 'selected')
        self.reextract_mode.addItem('Recover ZERO-pose clips · auto scope', 'zero')
        self.reextract_mode.addItem('FULL DATASET · Re-extract all cameras', 'all')
        self.reextract_mode.currentIndexChanged.connect(self.update_reextract_mode)
        layout.addWidget(self.reextract_mode)

        # MultiSelection: one click toggles each Person independently. Default
        # missing-only = all people; overwrite = no people until explicitly chosen.
        group = QHBoxLayout(); group.setSpacing(14)
        people_box = QFrame(); people_box.setObjectName('personPicker')
        people_box.setMinimumWidth(200)
        people_box.setMaximumWidth(340)
        pv = QVBoxLayout(people_box)
        pv.setContentsMargins(11, 10, 11, 10); pv.setSpacing(7)
        person_top = QHBoxLayout()
        self.re_people_title = label('Persons', 'h2')
        person_top.addWidget(self.re_people_title)
        person_top.addStretch()
        pv.addLayout(person_top)
        self.re_people = QListWidget()
        self.re_people.setObjectName('personMulti')
        self.re_people.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.re_people.setMinimumHeight(165)
        self.re_people.setMaximumHeight(210)
        self.re_people.itemSelectionChanged.connect(self.refresh_reextract_scope)
        pv.addWidget(self.re_people)
        people_actions = QHBoxLayout()
        people_actions.addWidget(button('Select all', lambda: self.set_reextract_people(True)))
        people_actions.addWidget(button('Clear', lambda: self.set_reextract_people(False)))
        pv.addLayout(people_actions)
        group.addWidget(people_box, 1)

        criteria = QFrame(); criteria.setObjectName('personPicker')
        cv = QVBoxLayout(criteria)
        cv.setContentsMargins(12, 10, 12, 10); cv.setSpacing(9)
        self.re_session = QComboBox()
        self.re_label = QComboBox()
        self.re_take = QComboBox()
        self.re_camera = QComboBox()
        for title, combo in (('Session', self.re_session), ('Label', self.re_label),
                             ('Take', self.re_take), ('Camera', self.re_camera)):
            row = QVBoxLayout(); row.setSpacing(4)
            row.addWidget(label(title, 'muted')); row.addWidget(combo)
            cv.addLayout(row)
        group.addWidget(criteria, 2)
        layout.addLayout(group)
        for combo in (self.re_session, self.re_label, self.re_take):
            combo.currentIndexChanged.connect(self.refresh_reextract_scope)
        self.re_model = QComboBox()
        layout.addWidget(label('Extractor checkpoint', 'muted'))
        layout.addWidget(self.re_model)
        action_row = QHBoxLayout()
        action_row.addWidget(button('Preview selection', self.preview_reextract))
        self.run_reextract_btn = button('Recover missing', self.run_reextract, 'primary')
        action_row.addWidget(self.run_reextract_btn)
        action_row.addStretch()
        layout.addLayout(action_row)
        self.reextract_summary = label('', 'muted')
        layout.addWidget(self.reextract_summary)
        self._reextract_last_mode = None
        return card

    def build_split_task(self):
        card, splits = panel()
        self.split_mode = QComboBox()
        self.split_mode.addItems(['Subject-wise: 1 test person, optional val',
                                  'Within-person: time-aligned segments'])
        self.split_mode.currentIndexChanged.connect(self.update_split_mode)
        splits.addWidget(self.split_mode)
        self.split_people_row = QWidget()
        people_row = QHBoxLayout(self.split_people_row)
        people_row.setContentsMargins(0, 0, 0, 0)
        self.split_test_person = QComboBox()
        self.split_val_person = QComboBox()
        for title, control in (('Test person', self.split_test_person),
                               ('Validation person (optional)', self.split_val_person)):
            field = QVBoxLayout(); field.addWidget(label(title, 'muted')); field.addWidget(control)
            people_row.addLayout(field)
        splits.addWidget(self.split_people_row)
        fractions = QHBoxLayout()
        self.split_train_pct = QSpinBox(); self.split_train_pct.setRange(1, 98)
        self.split_train_pct.setValue(70)
        self.split_val_pct = QSpinBox(); self.split_val_pct.setRange(1, 98)
        self.split_val_pct.setValue(15)
        for title, spin in (('Train %', self.split_train_pct), ('Val %', self.split_val_pct)):
            field = QVBoxLayout(); field.addWidget(label(title, 'muted')); field.addWidget(spin)
            fractions.addLayout(field)
        self.split_fractions = QWidget(); self.split_fractions.setLayout(fractions)
        splits.addWidget(self.split_fractions)
        actions = QHBoxLayout()
        actions.addWidget(button('Preview split', self.preview_split))
        actions.addWidget(button('Export manifest', self.export_split, 'primary'))
        actions.addStretch(); splits.addLayout(actions)
        self.split_summary = label('', 'muted')
        splits.addWidget(self.split_summary)
        self.update_split_mode()
        return card

    def set_page(self,index):
        self.stack.setCurrentIndex(index)
        for i,b in enumerate(self.nav):b.setChecked(i==index)
        if index==2:self.refresh_library()
        if index==3:self.refresh_reextract_options()
        if index == 4: self.replay_inspector.refresh_catalog()
        elif hasattr(self, 'replay_inspector'): self.replay_inspector.pause()
        page=self.stack.widget(index)
        if isinstance(page,QScrollArea):
            page.verticalScrollBar().setValue(0)
            QTimer.singleShot(0,lambda:page.verticalScrollBar().setValue(0))

    def log(self,message):
        message=str(message)
        t=datetime.now().strftime('%H:%M:%S')
        lower=message.lower()
        color=('#EF4444' if any(v in lower for v in ('error','exception','lỗi','✖')) else
               '#F59E0B' if any(v in lower for v in ('warning','stalled','needs_review')) else
               '#10B981' if any(v in lower for v in ('saved','✓','connected')) else
               '#22D3EE' if 'start' in lower else '#94A3B8')
        if hasattr(self,'logs'):
            self.logs.append(f'<span style="color:#64748B">{t}</span> &nbsp; '
                             f'<span style="color:{color}">{html.escape(message)}</span>')
            self.logs.moveCursor(QTextCursor.MoveOperation.End)
        if hasattr(self,'tool_log'):self.tool_log.appendPlainText(f'[{t}] {message}')
        self.statusBar().showMessage(message.replace('\n',' ')[:150],10000)

    def alert(self,title,message):
        self.log(f'{title}: {message}')
        QMessageBox.warning(self,title,str(message))

    def refresh_subjects(self):
        old=self.subjects_combo.currentText();self.subjects_combo.clear()
        self.subject_cache=self.store.subjects()
        self.subjects_combo.addItems(sorted(self.subject_cache))
        if old:self.subjects_combo.setCurrentText(old)

    def refresh_labels(self):
        curr=self.label_combo.currentData();self.label_combo.clear()
        for d in self.store.get_labels()['labels']:
            name=POSTURE_NAMES[d['id']] if d['id'] in range(6) else d['name']
            self.label_combo.addItem(f"{d['id']}  ·  {name}",d['id'])
        if curr is not None:
            idx=self.label_combo.findData(curr)
            if idx>=0:self.label_combo.setCurrentIndex(idx)

    def add_subject(self):
        try:
            d=self.store.add_subject(self.new_pid.text(),self.consent_check.isChecked(),self.consent_ref.text())
            self.refresh_subjects();self.subjects_combo.setCurrentText(d['person_id'])
            self.new_pid.clear();self.consent_ref.clear();self.consent_check.setChecked(False)
            self.log('Thêm người '+d['person_id']+' · đã xác nhận có văn bản đồng ý')
        except Exception as e:self.alert('Không thêm được người',e)

    def populate_cameras(self):
        self.cameras_table.setRowCount(0)
        for c in self.store.cameras():self.add_camera_row(c)

    def add_camera_row(self,config=None):
        if isinstance(config,bool) or config is None:
            used={self.cameras_table.item(row,0).text() for row in range(self.cameras_table.rowCount())
                  if self.cameras_table.item(row,0)}
            i=1
            while f'cam{i:02d}' in used:i+=1
            config={'id':f'cam{i:02d}','name':f'DroidCam {i:02d}',
                    'view':'oblique','url':'http://192.168.1.103:4747/video','enabled':True}
        r=self.cameras_table.rowCount();self.cameras_table.insertRow(r)
        for col,value in enumerate([config['id'],config['name'],config['view'],
                                     config['url'],'yes' if config.get('enabled',True) else 'no']):
            self.cameras_table.setItem(r,col,QTableWidgetItem(str(value)))
        self.cameras_table.setRowHeight(r,42)

    def remove_camera_row(self):
        if self.core.active:self.alert('Đang ghi','Không sửa danh sách camera khi đang ghi');return
        self.cameras_table.removeRow(self.cameras_table.currentRow())

    def camera_settings(self):
        rows=[]
        for i in range(self.cameras_table.rowCount()):
            vals=[(self.cameras_table.item(i,j).text().strip() if self.cameras_table.item(i,j) else '') for j in range(5)]
            rows.append(dict(zip(['id','name','view','url'],vals[:4]),enabled=vals[4].lower() in ('1','yes','true','on','y')))
        return rows

    def camera_action(self, action, message):
        """Camera join/open/reconnect may block; no waiting on the Qt thread."""
        if self.core.active or self.core.finalizing or self.camera_busy or self.closing:
            return
        self.camera_busy=True; self.refresh_controls(); self.log(message)
        def run():
            try:
                action()
                self.ui_events.put(('camera_done', None))
            except Exception as exc:
                self.ui_events.put(('camera_done', str(exc)))
        threading.Thread(target=run, name='camera-configuration', daemon=True).start()

    def connect_cameras(self):
        if self.core.active or self.core.finalizing or self.camera_busy:return
        try:
            configs=self.camera_settings(); self.store.set_cameras(configs)
            self.disable_pose_preview()
            self.rebuild_camera_cards(configs)
            self.last_seen_idx.clear()
            self.fps_samples.clear()
            self.camera_action(lambda:self.core.connect_cameras(configs), 'Connecting camera sources…')
        except Exception as exc:self.alert('Camera configuration error',exc)

    def rebuild_camera_cards(self,cameras):
        for card in self.cards.values():
            self.camera_grid.removeWidget(card); card.deleteLater()
        self.cards.clear()
        enabled=[c for c in cameras if c.get('enabled',True)]
        self.empty_camera.setVisible(not enabled)
        for config in enabled:
            cid=config['id']
            self.cards[cid]=CameraCard(cid,config.get('name',cid),config.get('view',''))
            self.cards[cid].track_selected.connect(self.select_track)
            self.track_selections.setdefault(cid,TrackSelection(single_person=True))
        self._camera_columns=None; self.layout_camera_grid()

    def disconnect_cameras(self):
        if self.core.active or self.core.finalizing or self.camera_busy:return
        self.disable_pose_preview()
        self.camera_action(self.core.disconnect_cameras, 'Disconnecting camera sources…')

    def set_pose_preset(self, index):
        backend, checkpoint = self.pose_preset.itemData(index) or ('none', '')
        self.preview_combo.setCurrentIndex({'none':0,'yolo':1,'mediapipe':2}[backend])
        self.backend_combo.setCurrentText('mediapipe' if backend == 'mediapipe' else 'yolo')
        self.preview_model.setText(checkpoint)
        self.offline_model.setText(checkpoint)
        self.tracker_combo.setEnabled(backend == 'yolo')
        if self.preview_worker is not None:
            self.disable_pose_preview()
        if hasattr(self, 'auto_extract'):
            self.auto_extract.setEnabled(bool(checkpoint))

    def select_track(self,cid,track_id):
        selection=self.track_selections.setdefault(cid,TrackSelection(single_person=True))
        selection.select(track_id)
        result=self.preview_worker.get(cid) if self.preview_worker else None
        if result and not result.get('error'):selection.observe(result['persons'])
        if cid in self.cards:self.cards[cid].update_tracking(result,selection,self.preview_worker is not None and self.preview_worker.backend=='yolo')
        self.log(f'{cid.upper()} target '+(f'Track ID {track_id}' if track_id is not None else 'unselected')+' · operator choice, not verified identity')
        self.journal_tracking(cid,result,event='selection')

    def journal_tracking(self,cid,result,event='sample'):
        journal=self.tracking_journal
        if not journal:return
        selection=self.track_selections.get(cid,TrackSelection(auto=False,single_person=True))
        result=result or {}
        stamp=result.get('timestamp_ns') if event=='sample' else time.monotonic_ns()
        if event=='sample' and (stamp is None or stamp<journal.identity['start_host_monotonic_ns']):return
        journal.put(cid,dict(schema_version=2,event=event,camera_id=cid,
            frame_index=result.get('frame_index'),frame_index_kind='camera_source_index',
            host_recv_monotonic_ns=stamp,timestamp=result.get('host_utc') if event=='sample' else now_iso(),
            track_session=result.get('track_session'),stream_token=result.get('stream_token'),
            persons=result.get('persons',[]) if event=='sample' else [],
            error=result.get('error'),**selection.snapshot()))

    def enable_pose_preview(self):
        if self.preview_combo.currentIndex()==0:self.disable_pose_preview();return
        backend=['none','yolo','mediapipe'][self.preview_combo.currentIndex()]
        path=self.preview_model.text().strip()
        if not Path(path).is_file():self.alert('Chưa có model','Chọn YOLO-Pose .pt/.onnx hoặc MediaPipe .task đã có trên máy');return
        self.disable_pose_preview()
        device=['auto','cpu','0'][self.preview_device.currentIndex()]
        self.preview_worker=PosePreview(backend,path,device)
        self.pose_workers=[worker for worker in self.pose_workers if worker.thread.is_alive()]
        self.pose_workers.append(self.preview_worker)
        if not self.offline_model.text():self.offline_model.setText(path)
        self.begin_tracking_journal()
        self.pose_last_sent={};self.log('Đang bật preview '+backend+'; có thể giảm FPS overlay khi model nặng')

    def disable_pose_preview(self):
        if self.preview_worker:self.preview_worker.stop();self.preview_worker=None
        self.pose_last_sent={}
        self.tracking_tokens.clear();self.last_prediction.clear()
        for cid,selection in list(self.track_selections.items()):
            self.track_selections[cid]=TrackSelection(auto=selection.auto,single_person=True)
            if cid in self.cards:self.cards[cid].update_tracking(None,self.track_selections[cid],False)
            self.journal_tracking(cid,None,event='tracking_disabled')

    def start_record(self):
        if not self.start_btn.isEnabled():return
        try:
            pid=self.subjects_combo.currentText()
            if not pid:raise RuntimeError('Thêm Person ID và xác nhận phiếu đồng ý trước khi ghi')
            configs=[c for c in self.store.cameras() if c.get('enabled',True)]
            folder=self.core.start(pid,self.session_edit.text().strip(),self.label_combo.currentData(),configs)
            self.begin_tracking_journal()
            self.record_started=time.monotonic()
            self.log('● START '+str(folder.relative_to(self.store.root)))
            self.refresh_controls()
        except Exception as e:self.alert('Không thể ghi hình',e)

    def end_record(self):
        if self.core.active is None:return
        try:
            folder=self.core.stop()
            self.finish_tracking_journal()
            self.log('■ STOP · Đang hoàn tất MP4/CSV: '+str(folder.relative_to(self.store.root)))
            self.refresh_controls()
        except Exception as e:self.alert('Không thể kết thúc',e)

    def finish_tracking_journal(self):
        journal=self.tracking_journal
        if not journal:return
        for cid in self.cards:self.journal_tracking(cid,self.preview_worker.get(cid) if self.preview_worker else None,event='recording_stop')
        self.tracking_journal=None
        def finish():
            journal.finish()
            if journal.error:self.journal_errors.put(journal.error)
            if journal.dropped:self.journal_errors.put(f'{journal.dropped} live preview samples dropped; raw video unaffected')
        thread=threading.Thread(target=finish,name='finish-tracking-journal',daemon=True)
        self.journal_threads=[t for t in self.journal_threads if t.is_alive()]+[thread];thread.start()

    def begin_tracking_journal(self):
        active=self.core.active
        if self.tracking_journal or not active or not self.preview_worker or self.preview_worker.backend!='yolo':return
        identity=dict(person_id=active['pid'],session_id=active['session'],label=active['label'],
            start_host_monotonic_ns=active['start_ns'],model=self.preview_worker.model,device=self.preview_worker.device)
        self.tracking_journal=TrackingJournal(active['folder'],identity)
        for cid in self.cards:self.journal_tracking(cid,self.preview_worker.get(cid),event='recording_start')

    def refresh_controls(self):
        active=self.core.active is not None
        locked=active or self.core.finalizing or self.camera_busy or self.closing
        pid=self.subjects_combo.currentText()
        session=self.session_edit.text().strip()
        configs=[c for c in self.store.cameras() if c.get('enabled',True)]
        ready=bool(configs) and all(c['id'] in self.core.streams and self.core.streams[c['id']].ready() for c in configs)
        consent=self.subject_cache.get(pid,{}).get('consent_written_confirmed',False)
        valid=bool(pid and consent and re.fullmatch(r'[A-Za-z0-9_-]{2,40}',session) and self.label_combo.currentData() is not None)
        self.start_btn.setEnabled(not locked and not self.busy and ready and valid)
        self.end_btn.setEnabled(active and not self.closing)
        for widget in (self.subjects_combo,self.session_combo,self.label_combo,self.cameras_table,
                       self.connect_btn,self.disconnect_btn,self.add_camera_btn,self.remove_camera_btn,
                       self.add_person_btn,self.add_subject_toggle,self.preview_combo,self.preview_model,
                       self.pose_preset,self.preview_device,self.tracker_combo,
                       self.enable_tracking_btn,self.disable_tracking_btn):
            widget.setEnabled(not locked)
        self.auto_extract.setEnabled(not locked and bool(self.offline_model.text().strip()))
        state='Recording' if active else 'Saving…' if self.core.finalizing else 'Connecting…' if self.camera_busy else 'Idle'
        self.record_badge.setText(state)
        self.record_badge.setStyleSheet('background:#511E2C;color:#FCA5A5;border-color:#EF4444;' if active else '')
        self.app_status.setText(state)
        reason=('Recording raw video · E to stop.' if active else 'Finalizing MP4 and timestamps…' if self.core.finalizing else
                'Waiting for camera connections…' if self.camera_busy else 'Add a consenting participant to begin.' if not pid or not consent else
                'Enter a valid session ID (2–40 letters, digits, _ or -).' if not valid else
                'Connect all enabled cameras to start.' if not ready else 'Offline processing in progress…' if self.busy else
                'Ready · Hold each posture for 5–10 seconds.')
        self.takes_hint.setText(reason)
        self.start_btn.setToolTip(reason)

    def refresh_take_number(self, *args):
        if not hasattr(self,'take_number'):return
        if self.core.active:
            text='Take '+self.core.active['folder'].name.removeprefix('take_')
        else:
            folder=self.store.root/'recordings'/self.subjects_combo.currentText()/self.session_edit.text().strip()/f'label_{self.label_combo.currentData()}'
            numbers=[int(p.name[5:]) for p in folder.glob('take_*') if p.is_dir() and p.name[5:].isdigit()]
            text=f'Next take {max(numbers,default=0)+1:03d}'
        self.take_number.setText(text)

    def refresh_progress(self, *args):
        if not hasattr(self,'progress_rows'):return
        pid=self.subjects_combo.currentText()
        counts={i:0 for i in self.progress_rows}; reviews={i:0 for i in counts}
        for folder,meta in self.take_cache:
            lid=meta.get('label')
            if meta.get('person_id')==pid and lid in counts:
                counts[lid]+=1
                reviews[lid]+=int(meta.get('status')!='ok')
        self.progress_caption.setText(f'{pid} · All sessions' if pid else 'Select a person to view recorded takes.')
        for lid,row in self.progress_rows.items():
            row.setText(f'{counts[lid]} '+('take' if counts[lid]==1 else 'takes'))
            row.setToolTip(f'{reviews[lid]} need review')
            row.setStyleSheet('color:#F59E0B' if reviews[lid] else 'color:#10B981' if counts[lid] else 'color:#94A3B8')
        sessions=sorted({m.get('session_id','') for _,m in self.take_cache if m.get('person_id')==pid})
        current=self.session_edit.text()
        self.session_combo.blockSignals(True)
        self.session_combo.clear();self.session_combo.addItems(sorted(set(sessions+[current])))
        self.session_combo.setCurrentText(current);self.session_combo.blockSignals(False)
        self.refresh_take_number()

    def refresh(self):
        # Thread -> GUI communication: only main UI timer touches Qt widgets.
        while True:
            try:kind,error=self.ui_events.get_nowait()
            except queue.Empty:break
            if kind=='camera_done':
                self.camera_busy=False
                if error:self.alert('Camera connection error',error)
                else:self.log('Camera configuration applied. Streams reconnect automatically.')
            elif kind=='pose_quality_done':
                self.display_pose_quality(*error)
            elif kind=='reextract_done':
                self.reextract_summary.setText(
                    f"Recovered {error['succeeded']}/{error['selected']} clips · failed {error['failed']}")
                self.refresh_library()
                if hasattr(self, 'replay_inspector'): self.replay_inspector.refresh_catalog()
            elif kind=='close_done':
                if error:
                    self.closing=False;self.alert('Shutdown error',error)
                else:
                    self._shutdown_complete=True;self.close();return
        if self.core.finalizing != self.last_finalizing:
            self.last_finalizing=self.core.finalizing
        if self.core.recording_error and not self.core.finalizing:
            self.log('SAVE ERROR: '+self.core.recording_error)
            self.core.recording_error=None
        report=self.core.last_report
        if report and report.get('take_id')!=self.last_saved_take:
            self.last_saved_take=report['take_id']
            self.log('✓ Saved '+report['take_id']+' | '+report['status'])
            self.refresh_library()
            if self.auto_extract.isChecked() and Path(self.offline_model.text().strip()).is_file():
                backend=self.backend_combo.currentText()
                model=self.offline_model.text().strip()
                folder=self.core.last_folder
                if Path(model).is_file():
                    def task():
                        out=[]
                        for cid in report['cameras']:
                            video=folder/f'{cid}_raw.mp4'
                            if video.is_file():out.append(extract_video(video,backend,model,self.async_events.put))
                        return {'raw_keypoint_files':len(out),'backend':backend}
                    self.background_action(task,'Auto-extract '+backend+' for '+report['take_id'])
                else:
                    self.log('Video đã lưu. Chọn pose model trong Camera & Pose để trích keypoints.')
        active=self.core.active is not None
        if active != self.last_active:
            self.refresh_controls();self.last_active=active;self.refresh_take_number()
        if active:
            secs=time.monotonic()-(self.record_started or time.monotonic())
            self.duration.setText(f'{int(secs//60):02d}:{int(secs%60):02d}')
        elif self.core.finalizing:self.duration.setText('Saving…')
        else:self.duration.setText('00:00')
        context=(self.subjects_combo.currentText(),self.session_edit.text().strip())
        if context!=self.tracking_context:
            self.tracking_context=context;self.tracking_epoch=uuid.uuid4().hex
            self.track_selections={cid:TrackSelection(single_person=True) for cid in self.cards}
            self.tracking_tokens.clear();self.last_prediction.clear()
        ready=0; live_fps=[]
        for cid,card in self.cards.items():
            stream=self.core.streams.get(cid)
            if not stream:
                card.set_online(False);card.stats.setText('No frames received · Raw video')
                selection=self.track_selections.setdefault(cid,TrackSelection(single_person=True))
                if selection.track_id is not None and selection.state!='missing':
                    selection.state='missing';self.journal_tracking(cid,None,event='camera_offline')
                card.raw_stats='No frames received · Raw video'
                card.update_tracking(None,selection,False)
                continue
            image,recv_ns,index,_reported_fps,err,camera_token,utc_stamp=stream.snapshot(include_generation=True)
            # Count actual decoded frames over a time window. Averaging inverse
            # inter-frame delays exaggerates FPS when Windows delivers frames in bursts.
            samples=self.fps_samples.setdefault(cid,deque(maxlen=20))
            if not samples or samples[-1][1]!=index:
                samples.append((time.monotonic(),index))
            span=samples[-1][0]-samples[0][0]
            fps=(samples[-1][1]-samples[0][1])/span if len(samples)>1 and span>.1 else 0.0
            ok=stream.ready();ready+=int(ok)
            card.set_online(ok, image is not None)
            if ok:live_fps.append(fps)
            points=None;persons=[];prediction=None;new_prediction=False
            selection=self.track_selections.setdefault(cid,TrackSelection(single_person=True))
            if self.preview_worker:
                token=camera_token+':'+self.tracking_epoch
                previous_token=self.tracking_tokens.get(cid)
                if previous_token!=token:
                    if previous_token is not None:
                        selection=TrackSelection(single_person=True);self.track_selections[cid]=selection
                        self.journal_tracking(cid,None,event='stream_reset')
                        self.log(cid.upper()+' tracking reset; single-person auto selection restarted')
                    self.tracking_tokens[cid]=token
                    self.last_prediction.pop(cid,None)
                    self.preview_worker.reset_camera(cid,token)
                if image is not None and ok:
                    last=self.pose_last_sent.get(cid,0)
                    if time.monotonic()-last>.10:
                        self.preview_worker.submit(cid,image,recv_ns,token,index,utc_stamp)
                        self.pose_last_sent[cid]=time.monotonic()
                prediction=self.preview_worker.get(cid)
                if prediction and prediction.get('stream_token')==token:
                    key=(token,prediction['timestamp_ns'])
                    if self.last_prediction.get(cid)!=key:
                        self.last_prediction[cid]=key;new_prediction=True
                        if prediction.get('error'):self.log(cid.upper()+' tracking error: '+prediction['error'])
                        else:selection.observe(prediction['persons'])
                        self.journal_tracking(cid,prediction)
                    if not prediction.get('error') and ok and abs(recv_ns-prediction['timestamp_ns'])<950_000_000:
                        persons=prediction['persons'];points=prediction['points']
                    elif selection.track_id is not None:selection.state='missing'
            if image is not None and (self.last_seen_idx.get(cid)!=index or new_prediction):
                self.last_seen_idx[cid]=index
                card.show_frame(image,points,persons,selection.track_id)
            if image is not None:
                card.raw_stats=f'Raw {fps:.1f} FPS' if self.preview_worker and self.preview_worker.backend=='yolo' else f'{fps:.1f} FPS  ·  {image.shape[1]}×{image.shape[0]}  ·  Raw video'
                card.stats.setText(card.raw_stats)
                card.stats.setToolTip(err)
            else:card.raw_stats=err or 'Waiting for frames…';card.stats.setText(card.raw_stats)
            card.update_tracking(prediction,selection,self.preview_worker is not None and self.preview_worker.backend=='yolo' and ok)
            state_key=(self.tracking_tokens.get(cid),selection.track_id,selection.state,ok)
            if self.selection_states.get(cid)!=state_key:
                self.selection_states[cid]=state_key
                self.journal_tracking(cid,prediction,event='selection_state')
        self.cam_summary.setText(f'{ready} / {len(self.cards)} sources live · Independent raw video per camera')
        self.camera_badge.setText(f'{ready} / {len(self.cards)} cameras')
        self.fps_badge.setText(f'FPS  {sum(live_fps)/len(live_fps):.1f}' if live_fps else 'FPS  —')
        self.pose_badge.setText(('Pose ' + self.pose_preset.currentText()) if self.preview_worker else 'Pose Off')
        self.layout_camera_grid();self.refresh_controls()
        if self.preview_worker and self.preview_worker.error:
            self.log('Pose preview error: '+self.preview_worker.error)
            self.disable_pose_preview()
        while True:
            try:self.log('Tracking journal: '+self.journal_errors.get_nowait())
            except queue.Empty:break
        while True:
            try:self.log(self.async_events.get_nowait())
            except queue.Empty:break

    def refresh_library(self):
        if not hasattr(self, 'library'): return
        takes = list(self.store.iter_takes())
        try:
            curation = read_edits(self.store.root)['takes']
        except (ValueError, OSError):
            curation = {}
        self.take_cache = takes
        self.refresh_progress()
        expanded = set()
        selected_key = None
        def visit(item, path):
            nonlocal selected_key
            key = path + (item.text(0),)
            if item.isExpanded(): expanded.add(key)
            if item.isSelected(): selected_key = key
            for i in range(item.childCount()): visit(item.child(i), key)
        for i in range(self.library.topLevelItemCount()):
            visit(self.library.topLevelItem(i), ())
        self.library.clear()
        people = {}
        for folder, meta in takes:
            pid = str(meta.get('person_id') or folder.parents[2].name)
            session = str(meta.get('session_id') or folder.parents[1].name)
            label_id = str(meta.get('label', ''))
            person_item = people.get(pid)
            if person_item is None:
                person_item = QTreeWidgetItem([pid, ''])
                person_item.setData(0, Qt.ItemDataRole.UserRole, ('person', pid))
                person_item.setForeground(0, QBrush(QColor('#A5B4FC')))
                bold = QFont(); bold.setBold(True)
                person_item.setFont(0, bold)
                self.library.addTopLevelItem(person_item)
                people[pid] = person_item
            session_item = next((person_item.child(j) for j in range(person_item.childCount())
                                 if person_item.child(j).text(0) == session), None)
            if session_item is None:
                session_item = QTreeWidgetItem([session, ''])
                session_item.setForeground(0, QBrush(QColor('#67E8F9')))
                person_item.addChild(session_item)
            label_text = 'Label ' + label_id
            label_item = next((session_item.child(j) for j in range(session_item.childCount())
                               if session_item.child(j).text(0) == label_text), None)
            if label_item is None:
                label_item = QTreeWidgetItem([label_text, ''])
                label_item.setForeground(0, QBrush(QColor('#CBD5E1')))
                session_item.addChild(label_item)
            key = folder.relative_to(self.store.root / 'recordings').as_posix()
            edit = curation.get(key, {})
            flag = ('EXCLUDED' if edit.get('exclude_take') else 'CURATED' if edit else
                    str(meta.get('status', 'unknown')))
            take_item = QTreeWidgetItem([folder.name, flag])
            take_item.setData(0, Qt.ItemDataRole.UserRole, ('take', str(folder)))
            status = str(meta.get('status', 'unknown')).lower()
            take_item.setForeground(1, QBrush(QColor(
                '#34D399' if status == 'ok' else '#FBBF24' if status == 'needs_review' else '#94A3B8')))
            label_item.addChild(take_item)
        for item in people.values():
            count = sum(1 for f, m in takes if str(m.get('person_id')) == item.text(0))
            item.setText(1, f'{count} takes')
        self.dataset_summary.setText(
            f'{len(people)} people · {len(takes)} takes · {len(curation)} curation decisions · raw unchanged')
        self.library.sortItems(0, Qt.SortOrder.AscendingOrder)
        def restore(item, path):
            key = path + (item.text(0),)
            if key in expanded: item.setExpanded(True)
            if key == selected_key: self.library.setCurrentItem(item)
            for i in range(item.childCount()): restore(item.child(i), key)
        for i in range(self.library.topLevelItemCount()):
            restore(self.library.topLevelItem(i), ())

    def toggle_library_branch(self, item, column):
        if item.childCount():
            item.setExpanded(not item.isExpanded())

    def open_dataset_take_in_replay(self, item, column):
        role = item.data(0, Qt.ItemDataRole.UserRole)
        if not role or role[0] != 'take': return
        folder = Path(role[1])
        parts = folder.relative_to(self.store.root / 'recordings').parts
        if len(parts) != 4:return
        person, session, lid, take = parts
        if not lid.startswith('label_'):return
        self.set_page(4)
        self.replay_inspector.choose(person, session, int(lid[6:]), take)

    def show_pose_quality(self):
        """Scan saved raw JSONL on a worker; Qt widgets stay on main thread."""
        if self.core.active or self.core.finalizing:
            self.alert('Recording active', 'Finish recording before inspecting quality')
            return
        def scan():
            rows=quality_rows(self.store.root, progress=self.async_events.put)
            folder=(Path(__file__).resolve().parents[2]/'artifacts'/'audits'/'pose_quality_runs'/
                    datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            save_reports(rows, folder)
            self.ui_events.put(('pose_quality_done', (rows,str(folder))))
            return f'Pose quality analyzed · {len(rows)} camera clips'
        self.background_action(scan,'Pose quality (read-only)')

    def display_pose_quality(self, rows, report_dir):
        """A compact drill-down report: Person → Label → Camera → Take."""
        view=QDialog(self)
        view.setWindowTitle('Pose quality · current dataset')
        view.resize(1050, 690)
        layout=QVBoxLayout(view)
        overall=aggregate(rows,tuple())
        if overall:
            t=overall[0]
            layout.addWidget(label(
                f"{t['clips']} clips · {t['selected']}/{t['frames']} frames with pose · "
                f"{t['recoverable']} potentially recoverable · "
                f"{t['duplicate_frames']} frames with multiple boxes",'h2'))
        tree=QTreeWidget(view)
        tree.setColumnCount(7)
        tree.setHeaderLabels(['Person / Label / Camera / Take','Clips','Frames','Pose %',
                              'Projected %','Duplicates','Recoverable'])
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(True)
        tree.setExpandsOnDoubleClick(False)
        tree.setSortingEnabled(False)
        tree.header().setStretchLastSection(False)
        tree.header().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        for k in range(1,7): tree.header().setSectionResizeMode(k,QHeaderView.ResizeMode.ResizeToContents)
        by_person={}
        for row in rows:
            by_person.setdefault(row['person'],[]).append(row)
        for person, items in sorted(by_person.items()):
            p=QTreeWidgetItem(tree)
            self.fill_quality_item(p,person,aggregate(items,tuple())[0])
            by_label={}
            for item in items: by_label.setdefault(item['label'],[]).append(item)
            for lid, label_rows in sorted(by_label.items()):
                l=QTreeWidgetItem(p)
                self.fill_quality_item(l,f'Label {lid}',aggregate(label_rows,tuple())[0])
                by_camera={}
                for item in label_rows:by_camera.setdefault(item['camera'],[]).append(item)
                for cid, camera_rows in sorted(by_camera.items()):
                    c=QTreeWidgetItem(l)
                    self.fill_quality_item(c,cid,aggregate(camera_rows,tuple())[0])
                    for item in sorted(camera_rows,key=lambda r:(r['session'],r['take'])):
                        q=QTreeWidgetItem(c)
                        self.fill_quality_item(q,f"{item['session']} / {item['take']}",item)
        tree.itemClicked.connect(lambda item,column: item.setExpanded(not item.isExpanded()) if item.childCount() else None)
        layout.addWidget(tree)
        layout.addWidget(label('Projected = dry-run selector on saved detections, not new YOLO inference.','muted'))
        layout.addWidget(label('Reports: '+str(report_dir),'muted'))
        view.exec()

    @staticmethod
    def fill_quality_item(item,name,entry):
        values=[str(name),str(entry.get('clips',1)),str(entry['frames']),
                f"{entry['coverage_pct']:.1f}%",f"{entry['projected_pct']:.1f}%",
                str(entry['duplicate_frames']),str(entry['recoverable'])]
        for col,value in enumerate(values): item.setText(col,value)
        rate=entry['coverage_pct']
        if not rate:item.setForeground(3,QBrush(QColor('#FB7185')))
        elif rate<80:item.setForeground(3,QBrush(QColor('#FBBF24')))
        else:item.setForeground(3,QBrush(QColor('#34D399')))

    def selected_dataset_person(self):
        items = self.library.selectedItems()
        if not items: return None
        item = items[0]
        while item.parent() is not None:
            item = item.parent()
        return item.text(0)

    def on_library_selection(self):
        item = self.library.currentItem()
        if item is None: return
        role = item.data(0, Qt.ItemDataRole.UserRole)
        if role and role[0] == 'take':
            folder = Path(role[1])
            cams = sorted(f.name.replace('_raw.mp4','') for f in folder.glob('*_raw.mp4'))
            files = sum((folder/'keypoints'/f'{c}_yolo17_raw.txt').is_file() for c in cams)
            self.dataset_summary.setText(
                f'{folder.relative_to(self.store.root)} · cameras {len(cams)} · YOLO17 {files}/{len(cams)}')
        else:
            person = self.selected_dataset_person()
            n = sum(m.get('person_id') == person for _, m in self.take_cache)
            self.dataset_summary.setText(f'{person} · {n} takes · choose Validate person to check')

    def selected_reextract_people(self):
        return sorted(item.text() for item in self.re_people.selectedItems())

    def set_reextract_people(self, select_all):
        self.re_people.blockSignals(True)
        for i in range(self.re_people.count()):
            self.re_people.item(i).setSelected(bool(select_all))
        self.re_people.blockSignals(False)
        self.refresh_reextract_scope()

    def refresh_reextract_options(self):
        if not hasattr(self, 're_people'): return
        previous = set(self.selected_reextract_people())
        first_fill = self.re_people.count() == 0
        self.re_people.blockSignals(True)
        self.re_people.clear()
        people = sorted({str(m.get('person_id')) for _, m in self.take_cache if m.get('person_id')})
        self.re_people.addItems(people)
        for i in range(self.re_people.count()):
            item = self.re_people.item(i)
            item.setSelected(item.text() in previous or
                             (first_fill and self.reextract_mode.currentData() == 'missing'))
        self.re_people.blockSignals(False)
        self.refresh_reextract_scope()
        old_model = self.re_model.currentData()
        self.re_model.clear()
        for i in range(self.pose_preset.count()):
            data = self.pose_preset.itemData(i)
            if data and data[0] == 'yolo' and Path(data[1]).is_file():
                self.re_model.addItem(self.pose_preset.itemText(i), data[1])
        index = self.re_model.findData(old_model)
        if index < 0:
            index = next((i for i in range(self.re_model.count())
                          if Path(self.re_model.itemData(i)).name == 'yolo11n-pose.pt'), 0)
        if self.re_model.count(): self.re_model.setCurrentIndex(index)
        self.update_reextract_mode()
        self.refresh_split_people()

    def refresh_reextract_scope(self):
        if not hasattr(self, 're_session'): return
        people = set(self.selected_reextract_people())
        self.re_people_title.setText(f'Persons  ·  {len(people)} selected')
        takes = [(p, m) for p, m in self.take_cache if m.get('person_id') in people]
        for combo, choices, label_text in (
            (self.re_session, sorted({m.get('session_id') for p,m in takes if m.get('session_id')}), 'All sessions'),
            (self.re_label, sorted({m.get('label') for p,m in takes if m.get('label') is not None}), 'All labels'),
            (self.re_take, sorted({p.name for p,m in takes}), 'All takes'),
        ):
            previous = combo.currentData()
            combo.blockSignals(True)
            combo.clear(); combo.addItem(label_text, None)
            for choice in choices: combo.addItem(str(choice), choice)
            index = combo.findData(previous)
            if index >= 0: combo.setCurrentIndex(index)
            combo.blockSignals(False)
        previous_camera = self.re_camera.currentData()
        self.re_camera.blockSignals(True)
        self.re_camera.clear(); self.re_camera.addItem('Both / all cameras', None)
        for cid in sorted({cid for p,m in takes for cid in m.get('cameras',{})}):
            self.re_camera.addItem(cid, cid)
        index = self.re_camera.findData(previous_camera)
        if index >= 0: self.re_camera.setCurrentIndex(index)
        self.re_camera.blockSignals(False)

    def update_reextract_mode(self):
        if not hasattr(self, 'reextract_mode'): return
        mode = self.reextract_mode.currentData()
        if self._reextract_last_mode != mode:
            if self._reextract_last_mode is not None:
                self.set_reextract_people(mode == 'missing')
            self._reextract_last_mode = mode
        selected = mode == 'selected'
        all_mode = mode in ('all','zero')
        for c in (self.re_session, self.re_label, self.re_take, self.re_camera):
            c.setEnabled(selected)
        self.re_people.setEnabled(not all_mode)
        self.run_reextract_btn.setText('Re-extract zero-pose clips' if mode == 'zero' else
                                      'Re-extract ALL clips' if all_mode else
                                      'Reprocess selected' if selected else 'Recover missing')
        self.reextract_summary.setText('')

    def reextract_selection(self):
        mode = self.reextract_mode.currentData()
        if mode in ('all', 'zero'):
            return list_targets(self.store.root, mode=mode)
        people = self.selected_reextract_people()
        if not people:
            raise ValueError('Choose at least one person')
        matches = []
        for person in people:
            matches.extend(list_targets(self.store.root, mode=mode,
                person=person,
                session=self.re_session.currentData() if mode == 'selected' else None,
                label=self.re_label.currentData() if mode == 'selected' else None,
                take=self.re_take.currentData() if mode == 'selected' else None,
                camera=self.re_camera.currentData() if mode == 'selected' else None))
        return matches

    @staticmethod
    def _is_unselected(combo):
        return combo.currentData() is None

    def preview_reextract(self):
        try:
            matches = self.reextract_selection()
            persons = sorted({t['person'] for t in matches})
            takes = {(t['person'], t['session'], t['label'], t['take']) for t in matches}
            existing = sum(t['raw'].is_file() for t in matches)
            selected_names = ', '.join(persons[:3]) + (f' +{len(persons)-3}' if len(persons) > 3 else '')
            summary = (f'{len(persons)} persons ({selected_names or "none"}) · '
                       f'{len(takes)} takes · {len(matches)} camera clips · '
                       f'{existing} existing / {len(matches)-existing} missing')
            if self.reextract_mode.currentData() == 'selected':
                broad = all(self._is_unselected(c) for c in
                            (self.re_session, self.re_label, self.re_take))
                summary += (' · ALL takes for selected persons; typed confirmation required'
                            if broad else ' · Overwrite requires confirmation and backup')
            elif self.reextract_mode.currentData() == 'all':
                summary += ' · FULL dataset; typed confirmation required'
            elif self.reextract_mode.currentData() == 'zero':
                summary += ' · Zero selected pose; typed confirmation required'
            else:
                summary += ' · Missing files only'
            self.reextract_summary.setText(summary)
        except Exception as exc:
            self.alert('Invalid selection', exc)

    def run_reextract(self):
        if self.core.active or self.core.finalizing or self.busy:
            self.alert('Busy', 'Finish recording/offline task first')
            return
        try:
            matches = self.reextract_selection()
            model = self.re_model.currentData()
            if not model or not Path(model).is_file():
                raise ValueError('No local YOLO-Pose weight available')
            if not matches:
                self.reextract_summary.setText('No matching clips. Dataset unchanged.')
                return
            active_mode = self.reextract_mode.currentData()
            all_mode = active_mode in ('all','zero')
            overwrite = active_mode in ('selected', 'all','zero')
            broad_person_scope = (self.reextract_mode.currentData() == 'selected' and
                                  all(self._is_unselected(c) for c in
                                      (self.re_session, self.re_label, self.re_take)))
            if all_mode or broad_person_scope:
                scope = ('ZERO-pose clips' if active_mode == 'zero' else 'FULL dataset' if all_mode else
                         'ALL takes for ' + ', '.join(self.selected_reextract_people()))
                answer, accepted = QInputDialog.getText(
                    self, 'Confirm broad re-extraction',
                    f'{scope}: {len(matches)} camera clips. Existing YOLO17 files and metadata '
                    f'will be backed up before replacement.\nType REEXTRACT {len(matches)} to proceed:')
                if not accepted or answer.strip() != f'REEXTRACT {len(matches)}':
                    self.reextract_summary.setText('Re-extraction cancelled; no changes.')
                    return
            if overwrite:
                answer = QMessageBox.question(self, 'Confirm overwrite',
                    f'Re-extract {len(matches)} camera clips using {Path(model).name}?\n'
                    'Old keypoints and metadata will be backed up. MP4 and labels stay unchanged.\n'
                    'If another checkpoint was previously used, your dataset may contain mixed extractors until the entire run succeeds.',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if answer != QMessageBox.StandardButton.Yes: return
            output_base = Path(__file__).resolve().parents[2] / 'artifacts' / 'reextract_backups'
            device = ['auto', 'cpu', '0'][self.preview_device.currentIndex()]
            def job():
                result = reextract_targets(self.store.root, matches, model,
                    backup_root=output_base, overwrite=overwrite,
                    device=device, progress=self.async_events.put)
                self.ui_events.put(('reextract_done', result))
                return f"{result['succeeded']}/{result['selected']} succeeded, {result['failed']} failed"
            self.background_action(job, 'Re-extract YOLO17')
        except Exception as exc:
            self.alert('Re-extract unavailable', exc)

    def refresh_split_people(self):
        if not hasattr(self, 'split_test_person'): return
        people = sorted({m['person_id'] for _, m in self.take_cache if m.get('person_id')})
        old_test = self.split_test_person.currentData()
        old_val = self.split_val_person.currentData()
        for combo, empty in ((self.split_test_person, 'Choose test person'),
                             (self.split_val_person, 'No validation person')):
            combo.clear()
            combo.addItem(empty, None)
            for pid in people: combo.addItem(pid, pid)
        if self.split_test_person.findData(old_test) >= 0:
            self.split_test_person.setCurrentIndex(self.split_test_person.findData(old_test))
        if self.split_val_person.findData(old_val) >= 0:
            self.split_val_person.setCurrentIndex(self.split_val_person.findData(old_val))

    def update_split_mode(self):
        mode = self.split_mode.currentIndex()
        self.split_people_row.setVisible(mode == 0)
        self.split_fractions.setVisible(mode == 1)
        self.split_summary.setText('No changes to raw dataset.')

    def current_split_plan(self):
        if self.split_mode.currentIndex() == 0:
            test_pid = self.split_test_person.currentData()
            val_pid = self.split_val_person.currentData()
            if not test_pid or test_pid == val_pid:
                raise ValueError('Choose a test person distinct from validation')
            people = sorted({m['person_id'] for _,m in self.take_cache if m.get('person_id')})
            assignments = {p:('test' if p==test_pid else 'val' if p==val_pid else 'train')
                           for p in people}
            return make_split_plan(self.store.root, mode='subject', assignments=assignments)
        return make_split_plan(self.store.root, mode='within',
            train_pct=self.split_train_pct.value(),val_pct=self.split_val_pct.value())

    def preview_split(self):
        try:
            plan = self.current_split_plan()
            self.split_summary.setText(
                f"Entries: {plan['counts']} · warnings: {len(plan['warnings'])} · "
                f"missing person/class coverage: {len(plan['person_class_missing'])}")
        except Exception as exc:
            self.alert('Split cannot be created', exc)

    def export_split(self):
        try:
            plan = self.current_split_plan()
            if not plan['entries']:
                raise ValueError('No frames in split plan')
            output = Path(__file__).resolve().parents[2] / 'artifacts' / 'dataset_outputs' / 'split_runs'
            target = export_plan(plan, output)
            self.split_summary.setText('Saved split manifest: ' + str(target))
            self.log('Split manifests exported to '+str(target))
        except Exception as exc:
            self.alert('Split export failed',exc)

    def export_clean_clicked(self):
        """One-click validation, materialization and clean ZIP on a worker."""
        def run():
            result = export_clean_dataset(self.store.root, progress=self.async_events.put)
            return ('Exported {clips} clips, {takes} takes, {frames} frames → {zip}'
                    .format(**result))
        self.background_action(run, 'Export Clean Dataset')

    def background_action(self,action,name):
        if self.closing:return
        if self.core.active or self.core.finalizing:
            self.alert('Đang ghi','Hãy hoàn tất ghi hình trước khi xử lý dataset');return
        if self.busy:self.alert('Đang xử lý','Chỉ chạy một công việc offline mỗi lần');return
        self.busy=True
        def runner():
            try:
                self.async_events.put('▶ '+name)
                result=action()
                self.async_events.put('✓ '+name+' | '+str(result)[:600])
            except Exception as e:self.async_events.put(f'✖ {name}: {type(e).__name__}: {e}')
            finally:self.busy=False
        self.offline_thread=threading.Thread(target=runner,name='dataset-processing',daemon=True)
        self.offline_thread.start()

    def validate_clicked(self, person_only=False):
        person = self.selected_dataset_person() if person_only else None
        if person_only and person is None:
            self.alert('Select person', 'Choose a person in Dataset before validating')
            return
        def check():
            result = audit_dataset(self.store.root, progress=self.async_events.put, person=person)
            if result['errors']:
                raise RuntimeError(f'{len(result["errors"])} file/reference errors: ' + '; '.join(result['errors'][:2]))
            return {'clips': result['clip_camera_records'],
                    'people': len(result['people']),
                    'errors': len(result['errors']),
                    'warnings': len(result['warnings']),
                    'status_mismatch': result['metadata_manifest_status_differ']}
        self.background_action(check, 'Read-only dataset validation')

    def open_folder(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.root)))

    def eventFilter(self,obj,event):
        if event.type()==QEvent.Type.KeyPress and self.isActiveWindow():
            focus=QApplication.focusWidget()
            # Include descendants: editors in tables and editable combos have child widgets.
            current=focus
            while current:
                if isinstance(current,(QLineEdit,QPlainTextEdit,QTextEdit,QTableWidget)):
                    return super().eventFilter(obj,event)
                current=current.parentWidget()
            if event.modifiers()==Qt.KeyboardModifier.NoModifier:
                key=event.key()
                if key in (Qt.Key.Key_S,Qt.Key.Key_E) or Qt.Key.Key_0<=key<=Qt.Key.Key_5:
                    if event.isAutoRepeat():return True
                    if key==Qt.Key.Key_S:self.start_record()
                    elif key==Qt.Key.Key_E:self.end_record()
                    elif self.label_combo.isEnabled():
                        index=self.label_combo.findData(int(key-Qt.Key.Key_0))
                        if index>=0:self.label_combo.setCurrentIndex(index)
                    return True
        return super().eventFilter(obj,event)

    def closeEvent(self,event):
        if getattr(self,'_shutdown_complete',False):
            self.refresh_timer.stop()
            self.replay_inspector.shutdown()
            QApplication.instance().removeEventFilter(self)
            event.accept();return
        event.ignore()
        if self.closing:return
        if self.core.active or self.core.finalizing:
            answer=QMessageBox.question(self,'Save and close?',
                'Finish the current take and save all videos before closing?')
            if answer!=QMessageBox.StandardButton.Yes:return
        self.closing=True;self.finish_tracking_journal();self.disable_pose_preview();self.refresh_controls()
        self.log('Closing · Waiting for video writers and camera workers…')
        def finish():
            # Serialize shutdown behind any camera reconfiguration already underway.
            while self.camera_busy:time.sleep(.05)
            try:
                self.core.close()
                for journal_thread in self.journal_threads:journal_thread.join(timeout=10)
                if any(t.is_alive() for t in self.journal_threads):raise RuntimeError('Tracking sidecar is still saving')
                if self.offline_thread:
                    self.offline_thread.join(timeout=30)
                    if self.offline_thread.is_alive():
                        raise RuntimeError('Dataset processing is still running. Close again when it finishes.')
                for worker in self.pose_workers:
                    worker.stop();worker.thread.join(timeout=5)
                    if worker.thread.is_alive():
                        raise RuntimeError('Pose inference is still finishing. Close again shortly.')
                self.ui_events.put(('close_done',None))
            except Exception as exc:self.ui_events.put(('close_done',str(exc)))
        threading.Thread(target=finish,name='safe-shutdown',daemon=True).start()


def launch(root):
    app=QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont('Segoe UI',10))
    app.setStyleSheet(THEME)
    w=Window(root)
    available=app.primaryScreen().availableGeometry()
    w.resize(min(1600,available.width()),min(900,available.height()-40))
    w.show()
    return app.exec()
