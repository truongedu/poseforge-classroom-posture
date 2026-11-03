"""Reusable dashboard widgets. Sizes are Qt device-independent pixels."""
from __future__ import annotations

import cv2
from pathlib import Path
from PySide6.QtCore import Qt, QPoint, QRectF, QSize, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QSizePolicy, QScrollArea, QComboBox,
)
from .pose import COCO_BONES

THEME = '''
QWidget { color:#F8FAFC; font-family:"Segoe UI"; font-size:13px; }
QMainWindow, QWidget#workspace { background:#0B1020; }
QDialog { background:#111827; }
QLabel { background:transparent; border:0; }
QFrame#sidebar { background:#111827; border-right:1px solid #273449; }
QFrame#panel, QFrame#cameraFrame { background:#111827; border:1px solid #273449; border-radius:10px; }
QLabel#headline { font-size:25px; font-weight:600; }
QLabel#h2 { font-size:15px; font-weight:600; }
QLabel#muted { color:#94A3B8; }
QLabel#timer { font-size:27px; font-weight:600; color:#22D3EE; }
QLabel#tag { background:#172033; border:1px solid #273449; border-radius:8px; padding:6px 10px; color:#CBD5E1; }
QLineEdit, QComboBox, QPlainTextEdit, QTextEdit, QTableWidget {
 background:#0B1020; color:#F8FAFC; border:1px solid #273449;
 border-radius:7px; padding:6px 9px; selection-background-color:#6366F1;
}
QLineEdit, QComboBox { min-height:20px; }
QLineEdit:focus, QComboBox:focus { border-color:#818CF8; }
QLineEdit:disabled, QComboBox:disabled { color:#94A3B8; background:#172033; }
QComboBox::drop-down { width:24px; border:0; }
QComboBox::down-arrow { image:url("__ASSETS__/chevron_down.svg"); width:14px; height:14px; }
QComboBox QAbstractItemView { background:#172033; color:#F8FAFC; selection-background-color:#6366F1; }
QPushButton { background:#172033; border:1px solid #334155; border-radius:7px; padding:8px 12px; min-height:20px; font-weight:600; }
QPushButton:hover { background:#273449; border-color:#64748B; }
QPushButton:focus { border-color:#A5B4FC; }
QPushButton:pressed { background:#334155; }
QPushButton:disabled { background:#151D2D; color:#64748B; border-color:#273449; }
QPushButton#primary, QPushButton#start { background:#6366F1; border-color:#818CF8; }
QPushButton#primary:hover, QPushButton#start:hover { background:#7578FA; }
QPushButton#start, QPushButton#stop { min-height:26px; }
QPushButton#stop:enabled { background:#B91C1C; border-color:#EF4444; }
QPushButton#start:disabled, QPushButton#primary:disabled { background:#252A50; border-color:#353A64; color:#8B90BA; }
QPushButton#nav { background:transparent; border:1px solid transparent; text-align:left; padding:11px 12px; font-weight:400; }
QPushButton#nav:checked { background:#262B50; border-color:#41477B; color:#C7D2FE; font-weight:600; }
QCheckBox { background:transparent; spacing:8px; }
QCheckBox::indicator { width:17px; height:17px; border:1px solid #64748B; border-radius:4px; background:#0B1020; }
QCheckBox::indicator:checked { background:#6366F1; border-color:#818CF8; image:url("__ASSETS__/check.svg"); }
QScrollArea { background:transparent; border:0; }
QScrollArea > QWidget > QWidget { background:transparent; }
QScrollBar:vertical { background:#111827; width:10px; margin:0; }
QScrollBar::handle:vertical { background:#3B4962; border-radius:5px; min-height:28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:none; }
QHeaderView::section { background:#172033; color:#94A3B8; padding:9px; border:0; border-bottom:1px solid #273449; }
QHeaderView { background:#111827; }
QTableCornerButton::section { background:#172033; border:0; }
QTableWidget { gridline-color:#273449; }
QTableWidget::item { padding:4px; }
QStatusBar { background:#111827; color:#94A3B8; font-size:12px; }
QStatusBar::item { border:0; }
QToolTip { background:#273449; color:#F8FAFC; border:1px solid #64748B; padding:6px; }
/* Two distinct, large entry points; only their active task expands. */
QPushButton#toolExtract, QPushButton#toolSplit {
  text-align:left; font-size:17px; font-weight:700; min-height:78px;
  padding:12px 22px; border-radius:13px;
}
QPushButton#toolExtract { background:#172F40; color:#A5F3FC; border:1px solid #256579; }
QPushButton#toolExtract:hover { background:#20465B; border-color:#22D3EE; }
QPushButton#toolExtract:checked { background:#164E63; border:2px solid #67E8F9; }
QPushButton#toolSplit { background:#2B244D; color:#DDD6FE; border:1px solid #6152A4; }
QPushButton#toolSplit:hover { background:#393163; border-color:#A78BFA; }
QPushButton#toolSplit:checked { background:#433477; border:2px solid #C4B5FD; }
QFrame#personPicker { background:#101B2C; border:1px solid #30405A; border-radius:10px; }
QListWidget#personMulti, QTreeWidget#datasetTree {
  background:#0F192B; color:#E2E8F0; border:1px solid #34435D; border-radius:8px;
  outline:0; alternate-background-color:#142036;
  selection-background-color:#3730A3; selection-color:#FFFFFF;
}
QListWidget#personMulti::item { padding:5px 10px; border-radius:5px; }
QListWidget#personMulti::item:hover:!selected { background:#1E3652; }
QListWidget#personMulti::item:selected { background:#344CA0; color:#FFFFFF; }
QTreeWidget#datasetTree::item { min-height:24px; padding:3px 6px; border-radius:3px; }
QTreeWidget#datasetTree::item:hover:!selected { background:#233653; }
QTreeWidget#datasetTree::item:selected { background:#353C82; color:#FFFFFF; }
QSplitter::handle { background:#0B1020; }
'''
THEME = THEME.replace('__ASSETS__', (Path(__file__).parent/'ui_assets').as_posix())


def label(text, kind=None):
    widget = QLabel(text)
    if kind:
        widget.setObjectName(kind)
    widget.setWordWrap(True)
    widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
    return widget


def panel(layout=None):
    frame = QFrame()
    frame.setObjectName('panel')
    layout = layout or QVBoxLayout()
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)
    frame.setLayout(layout)
    frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
    return frame, layout


def button(text, method=None, role=None):
    widget = QPushButton(text)
    if role:
        widget.setObjectName(role)
    if method:
        widget.clicked.connect(method)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    return widget


def scrollable(content):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(content)
    scroll.setMinimumSize(0, 0)
    return scroll


class VideoSurface(QWidget):
    """Paint into a 16:9 viewport without a pixmap growing QLabel's size hint."""
    def __init__(self):
        super().__init__()
        self.frame = None
        self.offline = True
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def sizeHint(self):
        return QSize(320, 180)

    def minimumSizeHint(self):
        return QSize(0, 0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#0B1020'))
        if self.frame is not None and not self.offline:
            size = self.frame.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF((self.width()-size.width())/2, (self.height()-size.height())/2,
                            size.width(), size.height())
            painter.drawImage(target, self.frame)
            return
        painter.setPen(QPen(QColor('#141E31'), 1))
        for x in range(0, self.width(), 32):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 32):
            painter.drawLine(0, y, self.width(), y)
        cx, cy = self.width()/2, self.height()/2 - 29
        painter.setPen(QPen(QColor('#818CF8'), 2))
        painter.setBrush(QColor('#1B2440'))
        painter.drawRoundedRect(QRectF(cx-20, cy-14, 32, 28), 5, 5)
        painter.drawLine(QPoint(int(cx+12), int(cy-6)), QPoint(int(cx+24), int(cy-13)))
        painter.drawLine(QPoint(int(cx+24), int(cy-13)), QPoint(int(cx+24), int(cy+13)))
        painter.drawLine(QPoint(int(cx+24), int(cy+13)), QPoint(int(cx+12), int(cy+6)))
        painter.setPen(QColor('#E2E8F0'))
        font = painter.font(); font.setPixelSize(14); font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.drawText(QRectF(4, cy+25, self.width()-8, 22), Qt.AlignmentFlag.AlignCenter,
                         'Camera not connected')
        font.setPixelSize(12); font.setWeight(QFont.Weight.Normal); painter.setFont(font)
        painter.setPen(QColor('#94A3B8'))
        painter.drawText(QRectF(4, cy+50, self.width()-8, 22), Qt.AlignmentFlag.AlignCenter,
                         'Configure source in Camera Sources')


class CameraCard(QFrame):
    track_selected=Signal(str,object)
    def __init__(self, cid, name, view):
        super().__init__()
        self.cid = cid
        self.setObjectName('cameraFrame')
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12); layout.setSpacing(10)
        top = QHBoxLayout()
        title = label(cid.upper(), 'h2'); title.setToolTip(name)
        top.addWidget(title)
        top.addWidget(label(view.title(), 'muted')); top.addStretch()
        self.state = label('Offline', 'muted'); self.state.setWordWrap(False)
        top.addWidget(self.state)
        self.track_combo=QComboBox();self.track_combo.addItem('Target Track ID',None)
        self.track_combo.setEnabled(False)
        self.track_combo.setMinimumWidth(0)
        self.track_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.track_combo.setMinimumContentsLength(9)
        self.track_combo.setToolTip('Choose which track belongs to the active dataset participant in this camera')
        self.track_combo.activated.connect(lambda index:self.track_selected.emit(self.cid,self.track_combo.itemData(index)))
        # Track selection sits below the status row. A single row containing
        # title, view, status and the combo forced the camera card wider than
        # the preview on narrow windows (horizontal overflow at 960 px).
        layout.addLayout(top)
        self.track_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.track_combo)
        self.image = VideoSurface(); layout.addWidget(self.image)
        self.stats = label('No frames received · Raw video', 'muted')
        self.stats.setWordWrap(False)
        self.stats.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.stats)
        self.tracking_status=self.stats

    def update_tracking(self,result,selection,enabled=True):
        self.track_combo.setEnabled(enabled)
        selected=selection.track_id
        people=result.get('persons',[]) if result else []
        ids=sorted({p['track_id'] for p in people if p['track_id'] is not None})
        entries=[('Choose ID',None)]+[(f'ID {tid}',tid) for tid in ids]
        if selected is not None and selected not in ids:entries.append((f'ID {selected} · missing',selected))
        signature=(tuple(entries),selected)
        if signature!=getattr(self,'_track_signature',None):
            self._track_signature=signature;self.track_combo.blockSignals(True);self.track_combo.clear()
            for text,tid in entries:self.track_combo.addItem(text,tid)
            self.track_combo.setCurrentIndex(max(0,self.track_combo.findData(selected)))
            self.track_combo.blockSignals(False)
        raw=getattr(self,'raw_stats',self.stats.text())
        if not enabled:text=raw
        elif result and result.get('error'):text='Tracking error · see Event Log'
        elif result:
            state=selection.state.replace('_',' ')
            text=f'{raw} · Track {result["tracking_fps"]:.1f} FPS · {state}'
        else:text=f'{raw} · Loading ByteTrack'
        self.tracking_status.setText(text)
        self.tracking_status.setToolTip(text+(' · '+result['error'] if result and result.get('error') else ''))
        self.tracking_status.setStyleSheet('color:#F59E0B' if selection.state in ('missing','reselect_required','selection_required') else 'color:#94A3B8')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.image.setFixedHeight(max(156, round((self.width()-24)*9/16)))
        self.setFixedHeight(self.layout().sizeHint().height())

    def set_online(self, online, stalled=False):
        self.state.setText('Live' if online else 'Stalled' if stalled else 'Offline')
        self.state.setStyleSheet('color:#10B981' if online else 'color:#F59E0B' if stalled else 'color:#94A3B8')
        self.image.offline = not online; self.image.update()

    def show_frame(self, frame, points=None, persons=None, selected_id=None):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3*w, QImage.Format.Format_RGB888).copy()
        if persons:
            painter=QPainter(image);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            colors=['#22D3EE','#A78BFA','#FBBF24','#34D399','#FB7185','#60A5FA']
            font=painter.font();font.setPixelSize(max(20,round(w/28)));painter.setFont(font)
            for person in persons:
                tid=person['track_id'];selected=tid is not None and tid==selected_id
                color=QColor(colors[(tid-1)%len(colors)] if tid is not None else '#94A3B8')
                painter.setPen(QPen(color,max(2,round(w/320)) if selected else max(1,round(w/500))))
                bbox=person.get('bbox')
                if bbox:
                    x1,y1,x2,y2=bbox;painter.drawRect(QRectF(x1,y1,x2-x1,y2-y1))
                    text=(f'ID {tid}' if tid is not None else 'Unassigned')+(' · target' if selected else '')
                    text_width=painter.fontMetrics().horizontalAdvance(text)
                    text_x=max(4,min(int(x1),w-text_width-4))
                    text_y=max(painter.fontMetrics().height(),int(y1)-6)
                    painter.fillRect(QRectF(text_x-3,text_y-painter.fontMetrics().ascent()-2,text_width+6,painter.fontMetrics().height()+3),QColor(11,16,32,220))
                    painter.drawText(QPoint(text_x,text_y),text)
                kp=person.get('keypoints')
                if not kp:continue
                for a,b in COCO_BONES:
                    if a<len(kp) and b<len(kp) and len(kp[a])>=3 and len(kp[b])>=3 and kp[a][2]>.28 and kp[b][2]>.28:
                        painter.drawLine(QPoint(int(kp[a][0]),int(kp[a][1])),QPoint(int(kp[b][0]),int(kp[b][1])))
                painter.setBrush(color)
                for point in kp:
                    if len(point)>=3 and point[2]>.28:
                        painter.drawEllipse(QPoint(int(point[0]),int(point[1])),max(2,round(w/300)),max(2,round(w/300)))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.end()
        elif points:
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor('#22D3EE'), max(2, round(w/400))))
            for a, b in COCO_BONES:
                if a < len(points) and b < len(points) and points[a][2] > .28 and points[b][2] > .28:
                    painter.drawLine(QPoint(int(points[a][0]), int(points[a][1])),
                                     QPoint(int(points[b][0]), int(points[b][1])))
            painter.setBrush(QColor('#A5B4FC')); painter.setPen(Qt.PenStyle.NoPen)
            for x, y, confidence in points:
                if confidence > .28:
                    painter.drawEllipse(QPoint(int(x), int(y)), max(2, round(w/240)), max(2, round(w/240)))
            painter.end()
        self.image.frame = image; self.image.update()
