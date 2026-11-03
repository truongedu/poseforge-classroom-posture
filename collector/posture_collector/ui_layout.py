"""Dashboard composition; recording and dataset actions remain in ui.Window."""
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QBoxLayout, QGridLayout, QComboBox,
    QLineEdit, QCheckBox, QStackedWidget, QTextEdit, QStyle,
)
from .ui_components import label, panel, button, scrollable

POSTURE_NAMES = ['Sitting Straight', 'Hunched Forward', 'Head on Desk',
                 'Leaning Backward', 'Turning Sideways', 'Raising Hand']


class DashboardLayout:
    def build(self):
        root = QWidget(); root.setObjectName('workspace'); self.setCentralWidget(root)
        outer = QHBoxLayout(root); outer.setSpacing(0); outer.setContentsMargins(0, 0, 0, 0)
        self.sidebar = QFrame(); self.sidebar.setObjectName('sidebar'); self.sidebar.setFixedWidth(216)
        side = QVBoxLayout(self.sidebar); side.setContentsMargins(16, 26, 16, 18); side.setSpacing(8)
        brand = label('PoseForge', 'headline'); brand.setStyleSheet('color:#A5B4FC;font-size:23px;')
        side.addWidget(brand); side.addSpacing(20)
        entries = [('Collect', QStyle.StandardPixmap.SP_ComputerIcon),
                   ('Cameras', QStyle.StandardPixmap.SP_DriveNetIcon),
                   ('Dataset', QStyle.StandardPixmap.SP_DirIcon),
                   ('Tools', QStyle.StandardPixmap.SP_FileDialogDetailedView),
                   ('Replay', QStyle.StandardPixmap.SP_MediaPlay)]
        for index, (text, icon) in enumerate(entries):
            nav = button(text, lambda checked=False, i=index: self.set_page(i), 'nav')
            nav.setIcon(self.style().standardIcon(icon)); nav.setCheckable(True)
            side.addWidget(nav); self.nav.append(nav)
        side.addStretch()
        self.app_status = label('Ready', 'muted'); side.addWidget(self.app_status)
        outer.addWidget(self.sidebar)
        self.stack = QStackedWidget(); self.stack.setMinimumSize(0, 0); outer.addWidget(self.stack, 1)
        self.stack.addWidget(self.build_collector())
        self.stack.addWidget(self.build_cameras())
        self.stack.addWidget(self.build_tools())
        self.stack.addWidget(self.build_utilities())
        self.stack.addWidget(self.build_replay())
        self.set_page(0)
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().showMessage('Ready · Set up cameras before recording')

    def top_header(self, title, description):
        layout = QVBoxLayout(); layout.setSpacing(5)
        layout.addWidget(label(title, 'headline'))
        if description: layout.addWidget(label(description, 'muted'))
        return layout

    @staticmethod
    def field(layout, title, widget):
        group = QVBoxLayout(); group.setSpacing(5)
        group.addWidget(label(title, 'muted')); group.addWidget(widget)
        layout.addLayout(group)

    @staticmethod
    def wrapped_check(layout, text):
        row = QHBoxLayout(); row.setSpacing(8)
        check = QCheckBox(); check.setAccessibleName(text)
        row.addWidget(check, 0, Qt.AlignmentFlag.AlignTop)
        description = label(text, 'muted')
        row.addWidget(description, 1); layout.addLayout(row)
        return check

    def build_collector(self):
        page = QWidget(); whole = QVBoxLayout(page)
        whole.setContentsMargins(22, 20, 22, 16); whole.setSpacing(14)
        self.collector_header = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.collector_header.setSpacing(14)
        heading = QWidget(); heading.setLayout(self.top_header('Collect', 'Record videos and keypoints'))
        heading.layout().setContentsMargins(0, 0, 0, 0)
        self.collector_header.addWidget(heading, 1)
        badge_group = QWidget(); statuses = QHBoxLayout(badge_group)
        statuses.setContentsMargins(0, 0, 0, 0); statuses.setSpacing(8)
        self.camera_badge = label('0 / 0 cameras', 'tag')
        self.record_badge = label('Idle', 'tag')
        self.fps_badge = label('FPS  —', 'tag')
        self.pose_badge = label('Pose  Off', 'tag')
        for badge in (self.camera_badge, self.record_badge, self.fps_badge, self.pose_badge):
            badge.setWordWrap(False); statuses.addWidget(badge)
        statuses.addStretch(); self.collector_header.addWidget(badge_group)
        whole.addLayout(self.collector_header)
        mid = QHBoxLayout(); mid.setSpacing(16)
        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(10)
        preview_top = QHBoxLayout(); preview_top.addWidget(label('Camera workspace', 'h2')); preview_top.addStretch()
        self.sources_link = button('Camera settings', lambda: self.set_page(1))
        preview_top.addWidget(self.sources_link); lv.addLayout(preview_top)
        self.preview_content = QWidget(); grid_outer = QVBoxLayout(self.preview_content)
        grid_outer.setContentsMargins(0, 0, 0, 0); grid_outer.setSpacing(10)
        self.camera_grid = QGridLayout(); self.camera_grid.setSpacing(12)
        grid_outer.addLayout(self.camera_grid); grid_outer.addStretch(1)
        self.empty_camera = label('No camera sources. Add a source in Camera Sources.', 'muted')
        self.camera_grid.addWidget(self.empty_camera, 0, 0)
        self.preview_scroll = scrollable(self.preview_content); lv.addWidget(self.preview_scroll, 1)
        self.cam_summary = label('Configure sources to begin.', 'muted'); lv.addWidget(self.cam_summary)
        mid.addWidget(left, 1)
        controls = QWidget(); right = QVBoxLayout(controls)
        right.setContentsMargins(0, 0, 6, 0); right.setSpacing(12)
        capture, form = panel(); self.recording_card = capture
        form.addWidget(label('Recording', 'h2'))
        self.subjects_combo = QComboBox(); self.subjects_combo.setMinimumWidth(0)
        self.subjects_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.subjects_combo.setMinimumContentsLength(8)
        self.field(form, 'Person', self.subjects_combo)
        self.session_combo = QComboBox(); self.session_combo.setEditable(True)
        self.session_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.session_combo.addItem('session_'+datetime.now().strftime('%Y%m%d_%H%M'))
        self.session_edit = self.session_combo.lineEdit()
        self.session_edit.setPlaceholderText('Select or enter a session ID')
        self.field(form, 'Session', self.session_combo)
        self.label_combo = QComboBox(); self.label_combo.setMinimumWidth(0)
        self.label_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.label_combo.setMinimumContentsLength(10)
        self.field(form, 'Posture (0–5)', self.label_combo)
        stats = QHBoxLayout()
        self.duration = label('00:00', 'timer'); stats.addWidget(self.duration); stats.addStretch()
        self.take_number = label('Next take: 001', 'muted'); stats.addWidget(self.take_number)
        form.addLayout(stats)
        actions = QHBoxLayout(); actions.setSpacing(8)
        self.start_btn = button('Start · S', self.start_record, 'start')
        self.end_btn = button('Stop · E', self.end_record, 'stop')
        actions.addWidget(self.start_btn, 3); actions.addWidget(self.end_btn, 2); form.addLayout(actions)
        self.takes_hint = label('Connect all enabled cameras to start.', 'muted'); form.addWidget(self.takes_hint)
        self.auto_extract = self.wrapped_check(form, 'Auto-extract keypoints after saving')
        self.auto_extract.setChecked(True); right.addWidget(capture)
        right.addWidget(self.build_pose_controls())
        self.add_subject_toggle = button('+ New person')
        self.add_subject_toggle.setCheckable(True)
        right.addWidget(self.add_subject_toggle)
        subject, sv = panel(); self.subject_panel = subject
        sv.addWidget(label('New person', 'h2'))
        self.new_pid = QLineEdit(); self.new_pid.setPlaceholderText('Person ID · e.g. TQB, DTT, NVA')
        self.new_pid.setMaxLength(24); self.new_pid.setAccessibleName('New Person ID'); sv.addWidget(self.new_pid)
        self.consent_ref = QLineEdit(); self.consent_ref.setPlaceholderText('Optional note / consent reference')
        self.consent_ref.setAccessibleName('Consent note'); sv.addWidget(self.consent_ref)
        self.consent_check = self.wrapped_check(sv, 'Written participant consent has been obtained.')
        self.add_person_btn = button('Add Person', self.add_subject); sv.addWidget(self.add_person_btn)
        right.addWidget(subject)
        subject.setVisible(False)
        self.add_subject_toggle.toggled.connect(subject.setVisible)
        self.progress_toggle = button('Show take counts')
        self.progress_toggle.setCheckable(True)
        right.addWidget(self.progress_toggle)
        progress, pv = panel(); self.progress_panel = progress
        pv.addWidget(label('Takes by posture', 'h2'))
        self.progress_caption = label('Select a person to view recorded takes.', 'muted'); pv.addWidget(self.progress_caption)
        self.progress_rows = {}
        for lid, name in enumerate(POSTURE_NAMES):
            row = QHBoxLayout(); row.setSpacing(8)
            row.addWidget(label(f'{lid}  ·  {name}'), 1)
            count = label('0 takes', 'muted'); count.setWordWrap(False); row.addWidget(count)
            self.progress_rows[lid] = count; pv.addLayout(row)
        right.addWidget(progress)
        progress.setVisible(False)
        self.progress_toggle.toggled.connect(progress.setVisible)
        right.addStretch(1)
        self.control_scroll = scrollable(controls); self.control_scroll.setFixedWidth(380)
        mid.addWidget(self.control_scroll); whole.addLayout(mid, 1)
        self.log_card, log_layout = panel(); log_layout.setContentsMargins(12, 8, 12, 10); log_layout.setSpacing(6)
        log_top = QHBoxLayout(); log_top.addWidget(label('Event Log', 'h2')); log_top.addStretch()
        self.clear_log_btn = button('Clear', self.clear_logs); log_top.addWidget(self.clear_log_btn)
        self.collapse_log_btn = button('Show log', self.toggle_log); log_top.addWidget(self.collapse_log_btn)
        log_layout.addLayout(log_top)
        self.logs = QTextEdit(); self.logs.setReadOnly(True); self.logs.setAcceptRichText(False)
        self.logs.document().setMaximumBlockCount(500)
        log_layout.addWidget(self.logs); self.logs.setVisible(False)
        self.log_card.setFixedHeight(54); whole.addWidget(self.log_card)
        self.log('Collector ready.')
        return page

    def clear_logs(self):
        self.logs.clear()

    def toggle_log(self):
        self._log_manual = True
        expanded = self.logs.isVisible()
        self.logs.setVisible(not expanded)
        self.log_card.setFixedHeight(54 if expanded else 150)
        self.collapse_log_btn.setText('Expand' if expanded else 'Collapse')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'control_scroll'):
            self.collector_header.setDirection(QBoxLayout.Direction.TopToBottom if self.width() < 1400 else QBoxLayout.Direction.LeftToRight)
            self.sidebar.setFixedWidth(184 if self.width() < 1280 else 216)
            self.control_scroll.setFixedWidth(348 if self.width() < 1500 else 380)
            # Event log stays collapsed unless explicitly opened.
            self.layout_camera_grid()

    def layout_camera_grid(self):
        if not hasattr(self, 'preview_scroll'):
            return
        columns = 2 if self.preview_scroll.viewport().width() >= 620 else 1
        if columns == getattr(self, '_camera_columns', None):
            return
        self._camera_columns = columns
        for card in self.cards.values():
            self.camera_grid.removeWidget(card)
        for index, card in enumerate(self.cards.values()):
            self.camera_grid.addWidget(card, index//columns, index % columns)
        self.camera_grid.setColumnStretch(0, 1)
        self.camera_grid.setColumnStretch(1, 1 if columns == 2 else 0)
