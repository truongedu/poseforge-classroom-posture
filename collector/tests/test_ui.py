"""Real Qt widgets and local OpenCV feeds; no phones or pose weights required."""
import csv
import json
import time

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from posture_collector.storage import DataStore, save_json
from posture_collector.ui import Window, THEME, PosePreview
from posture_collector.pose import person_record


@pytest.fixture(scope='session')
def app():
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion'); app.setStyleSheet(THEME)
    return app


def wait_until(predicate, timeout=10000):
    deadline = time.monotonic()+timeout/1000
    while not predicate():
        assert time.monotonic() < deadline, 'Timed out waiting for asynchronous UI/backend work'
        QTest.qWait(30)


@pytest.fixture
def window(app, tmp_path):
    w = Window(tmp_path)
    errors = []
    w.alert = lambda title, message: errors.append((title, str(message)))
    w.test_errors = errors
    w.show(); w.activateWindow(); QTest.qWait(120)
    yield w
    if not getattr(w, '_shutdown_complete', False):
        w.core.close()
        wait_until(lambda: not w.camera_busy)
        w.close()
        wait_until(lambda: not w.isVisible())
    assert not errors, errors


def create_feeds(root):
    configs = []
    for index, color in enumerate([(30, 55, 85), (100, 50, 20)], 1):
        file = root/f'feed{index}.mp4'
        writer = cv2.VideoWriter(str(file), cv2.VideoWriter_fourcc(*'mp4v'), 20, (320, 180))
        assert writer.isOpened()
        for _ in range(200):
            writer.write(np.full((180, 320, 3), color, np.uint8))
        writer.release()
        configs.append(dict(id=f'cam{index:02d}', name=f'Test camera {index}',
                            view='front' if index == 1 else 'oblique', url=str(file), enabled=True))
    return configs


@pytest.mark.parametrize('size', [(1920, 1080), (1600, 900), (1366, 768), (1093, 614), (960, 620)])
def test_layout_scroll_and_navigation(window, size):
    w = window; w.resize(*size); QTest.qWait(180)
    # Window managers can cap requested sizes at the available desktop area.
    # Verify the requested size is honored within that platform constraint.
    screen_width = w.screen().availableGeometry().width()
    expected_width = min(size[0], screen_width)
    assert expected_width - 100 <= w.width() <= size[0]
    assert w.height() <= size[1]
    assert w.width() >= w.minimumWidth()
    assert not w.control_scroll.horizontalScrollBar().maximum()
    assert not w.preview_scroll.horizontalScrollBar().maximum()
    # Recording actions are reachable without scrolling the initial control viewport.
    for btn in (w.start_btn, w.end_btn):
        assert btn.height() >= btn.sizeHint().height()
        point = btn.mapTo(w.control_scroll.viewport(), btn.rect().bottomRight())
        assert point.y() < w.control_scroll.viewport().height(), (size, point)
    for badge in (w.camera_badge, w.record_badge, w.fps_badge, w.pose_badge):
        assert badge.width() >= badge.sizeHint().width()
        assert badge.mapTo(w.centralWidget(), badge.rect().topLeft()).y() >= 0
    # V3 intentionally collapses the progress section. A scrollbar is not
    # required if the compact controls fit, and the optional section must
    # remain reachable after the user expands it.
    assert not w.progress_panel.isVisible()
    w.progress_toggle.click()
    QTest.qWait(80)
    assert w.progress_panel.isVisibleTo(w.control_scroll)
    w.control_scroll.ensureWidgetVisible(w.progress_rows[5])
    QTest.qWait(80)
    last_row = w.progress_rows[5]
    row_top = last_row.mapTo(w.control_scroll.viewport(), last_row.rect().topLeft()).y()
    row_bottom = last_row.mapTo(w.control_scroll.viewport(), last_row.rect().bottomLeft()).y()
    assert row_top >= 0 and row_bottom < w.control_scroll.viewport().height(), (size, row_top, row_bottom)
    w.progress_toggle.click()
    for index, nav in enumerate(w.nav):
        nav.click(); assert w.stack.currentIndex() == index; assert nav.isChecked()
    w.set_page(0); before = w.logs.isVisible()
    w.toggle_log(); assert w.logs.isVisible() != before
    w.toggle_log(); assert w.logs.isVisible() == before
    w.log('Test event'); assert 'Test event' in w.logs.toPlainText()
    w.clear_logs(); assert not w.logs.toPlainText()


def test_progress_uses_metadata_and_next_take(window):
    w = window; w.store.add_subject('TQB', True); w.store.add_subject('NVA', True)
    w.refresh_subjects(); w.subjects_combo.setCurrentText('TQB'); w.session_edit.setText('session_test')
    for pid, lid, status in [('TQB', 0, 'ok'), ('TQB', 0, 'needs_review'), ('NVA', 1, 'ok')]:
        folder = w.store.create_take(pid, lid, 'session_test')
        save_json(folder/'metadata.json', dict(person_id=pid, session_id='session_test', label=lid, status=status))
    w.refresh_library()
    assert w.progress_rows[0].text() == '2 takes'
    assert w.progress_rows[1].text() == '0 takes'
    assert w.take_number.text() == 'Next take 003'
    assert w.progress_rows[0].toolTip() == '1 need review'
    w.subjects_combo.setCurrentText('NVA')
    assert w.progress_rows[1].text() == '1 take'
    assert w.session_combo.findText('session_test') >= 0


def test_subject_form_and_camera_row_actions(window):
    w = window
    w.new_pid.setText('tqb'); w.consent_ref.setText('TEST ONLY consent note')
    w.consent_check.setChecked(True); w.add_person_btn.click()
    assert w.subjects_combo.currentText() == 'TQB'
    assert w.store.subjects()['TQB']['consent_reference'] == 'TEST ONLY consent note'
    assert not w.consent_check.isChecked() and not w.new_pid.text()
    w.cameras_table.selectRow(0); w.remove_camera_row(); w.add_camera_row()
    ids = [c['id'] for c in w.camera_settings()]
    assert len(ids) == len(set(ids)) == 2


def test_two_cameras_shortcuts_raw_video_and_safe_close(window, tmp_path, monkeypatch):
    w = window
    configs = create_feeds(tmp_path)
    w.store.set_cameras(configs); w.populate_cameras()
    w.store.add_subject('DTT', True); w.refresh_subjects()
    w.subjects_combo.setCurrentText('DTT'); w.session_edit.setText('session_ui_test')
    w.label_combo.setCurrentIndex(w.label_combo.findData(4)); w.auto_extract.setChecked(False)
    assert not w.start_btn.isEnabled(); assert not w.end_btn.isEnabled()
    w.connect_cameras()
    wait_until(lambda: not w.camera_busy and len(w.core.streams) == 2 and all(s.ready() for s in w.core.streams.values()))
    w.refresh(); assert w.start_btn.isEnabled()
    w.session_edit.setText('bad session!'); assert not w.start_btn.isEnabled()
    w.session_edit.setText('session_ui_test'); w.refresh()
    w.new_pid.setFocus(); QTest.keyClick(w.new_pid, Qt.Key.Key_S)
    assert w.core.active is None; assert w.new_pid.text().lower() == 's'
    w.new_pid.clear(); w.start_btn.setFocus(); w.activateWindow(); QTest.qWait(60)
    QTest.keyClick(w, Qt.Key.Key_S)
    assert w.core.active is not None
    folder = w.core.active['folder']
    # Repeat and direct duplicate commands must neither create takes nor show errors.
    QApplication.sendEvent(w, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_S, Qt.KeyboardModifier.NoModifier, 's', True, 5))
    w.start_record(); assert w.core.active['folder'] == folder
    assert not w.subjects_combo.isEnabled() and not w.session_combo.isEnabled()
    assert not w.label_combo.isEnabled() and not w.connect_btn.isEnabled()
    assert not w.disconnect_btn.isEnabled() and w.end_btn.isEnabled()
    # Paint a bright preview skeleton repeatedly while the independent writers record.
    points = [(50+i*8, 40+i*4, 1.0) for i in range(13)]
    for _ in range(54):
        for cid, card in w.cards.items():
            frame = w.core.streams[cid].snapshot()[0]
            card.show_frame(frame, points,persons=[person_record(0,points+[(100,80,1.0)]*4,[20,10,180,170],.95,1)],selected_id=1)
        QTest.qWait(100)
    assert w.duration.text() != '00:00'
    assert 10 < float(w.fps_badge.text().split()[-1]) < 35
    w.end_btn.setFocus(); QTest.keyClick(w, Qt.Key.Key_E)
    assert w.core.active is None; w.end_record()
    wait_until(lambda: not w.core.finalizing and (folder/'metadata.json').exists())
    w.refresh()
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['person_id'] == 'DTT' and meta['label'] == 4
    assert meta['session_id'] == 'session_ui_test' and meta['status'] == 'ok', meta
    assert folder.relative_to(w.store.root).parts == ('recordings', 'DTT', 'session_ui_test', 'label_4', 'take_001')
    for index, expected in enumerate([(30, 55, 85), (100, 50, 20)], 1):
        cid = f'cam{index:02d}'; video = cv2.VideoCapture(str(folder/f'{cid}_raw.mp4'))
        decoded = 0
        while True:
            ok, frame = video.read()
            if not ok: break
            # Every frame remains a flat camera color: any skeleton would fail.
            assert np.max(frame.std(axis=(0, 1))) < 1
            assert np.abs(frame.mean(axis=(0, 1))-np.array(expected)).max() < 12
            decoded += 1
        video.release()
        rows = list(csv.DictReader((folder/f'{cid}_timestamps.csv').open()))
        assert decoded == len(rows) == meta['cameras'][cid]['frames']
        stamps = [int(row['host_recv_monotonic_ns']) for row in rows]
        assert all(a <= b for a, b in zip(stamps, stamps[1:]))
        assert stamps[0] >= meta['start_host_monotonic_ns']
        assert rows[0]['host_recv_utc'] == meta['cameras'][cid]['first_host_received_utc']
        assert all(row['host_recv_utc'] for row in rows)
    assert w.progress_rows[4].text() == '1 take'
    assert w.take_number.text() == 'Next take 002'
    # Close during a subsequent recording: Qt stays alive until writers finish.
    w.start_btn.setFocus(); QTest.keyClick(w, Qt.Key.Key_S); assert w.core.active
    QTest.qWait(200)
    monkeypatch.setattr(QMessageBox, 'question', lambda *a: QMessageBox.StandardButton.Yes)
    w.close(); wait_until(lambda: not w.isVisible())
    assert w.core.active is None and not w.core.finalizing and not w.core.streams


def test_slow_pose_worker_does_not_block_ui(window, monkeypatch):
    import posture_collector.ui as ui
    class SlowPose:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def detect(self, frame): time.sleep(.4); return []
    monkeypatch.setattr(ui, 'PoseRunner', SlowPose)
    worker = PosePreview('yolo', 'test-only-stub')
    try:
        for _ in range(8): worker.submit('cam01', np.zeros((180, 320, 3), np.uint8), time.monotonic_ns())
        started = time.monotonic(); window.refresh(); window.nav[1].click()
        assert time.monotonic()-started < .2
        assert worker.queue.qsize() <= 2
    finally:
        worker.stop(); worker.thread.join(2)


def test_camera_actions_return_immediately_and_report_failure(window):
    w = window
    def slow_connection(): time.sleep(.4)
    started = time.monotonic(); w.camera_action(slow_connection, 'Testing reconnect')
    assert time.monotonic()-started < .2
    assert not w.start_btn.isEnabled() and not w.connect_btn.isEnabled()
    w.nav[1].click(); assert w.stack.currentIndex() == 1
    wait_until(lambda: not w.camera_busy)
    def failure(): raise RuntimeError('Simulated connection error')
    w.camera_action(failure, 'Testing connection failure')
    wait_until(lambda: not w.camera_busy)
    assert w.test_errors.pop() == ('Camera connection error', 'Simulated connection error')


def test_tools_cards_and_multi_person_reextract(window):
    """A single selected task is visible and batch scope includes multiple people."""
    w = window
    for pid in ('CHANH', 'DGD', 'KIET'):
        w.store.add_subject(pid, True)
        for label_id in (0, 1):
            take = w.store.create_take(pid, label_id, 'session_trial')
            (take/'cam01_raw.mp4').write_bytes(b'video-test-only')
            save_json(take/'metadata.json', {
                'person_id': pid, 'session_id': 'session_trial', 'label': label_id,
                'status': 'ok', 'cameras': {'cam01': {'view': 'front'}}})
    w.refresh_library()
    w.set_page(3)
    assert not w.tool_task_stack.isVisible()

    w.reextract_task.click()
    assert w.tool_task_stack.isVisible()
    assert w.tool_task_stack.currentIndex() == 0
    assert w.reextract_task.isChecked() and not w.split_task.isChecked()
    assert w.selected_reextract_people() == ['CHANH', 'DGD', 'KIET']
    assert len(w.reextract_selection()) == 6

    w.reextract_mode.setCurrentIndex(1)
    assert w.selected_reextract_people() == []  # no implicit overwrite target
    w.re_people.item(0).setSelected(True)
    w.re_people.item(2).setSelected(True)
    assert w.selected_reextract_people() == ['CHANH', 'KIET']
    # Person(s) alone is a valid batch selection (all sessions/labels/takes).
    whole_people_targets = w.reextract_selection()
    assert len(whole_people_targets) == 4
    assert {t['person'] for t in whole_people_targets} == {'CHANH', 'KIET'}
    w.preview_reextract()
    assert '2 persons' in w.reextract_summary.text()
    assert '4 camera clips' in w.reextract_summary.text()
    assert 'typed confirmation' in w.reextract_summary.text()
    label_index = w.re_label.findData(1)
    assert label_index >= 0
    w.re_label.setCurrentIndex(label_index)
    targets = w.reextract_selection()
    assert len(targets) == 2
    assert {t['person'] for t in targets} == {'CHANH', 'KIET'}
    assert all(t['label'] == 1 for t in targets)
    assert all(t['camera'] == 'cam01' for t in targets)
    w.set_page(0); w.set_page(3)
    assert w.selected_reextract_people() == ['CHANH', 'KIET']
    w.split_task.click()
    assert w.tool_task_stack.isVisible()
    assert w.tool_task_stack.currentIndex() == 1
    assert not w.reextract_task.isChecked() and w.split_task.isChecked()
    w.split_task.click()
    assert not w.tool_task_stack.isVisible()


def test_dataset_tree_single_click_expands_and_preserves_state(window):
    """Click on text (not tiny disclosure triangle) opens a person in one click."""
    w = window
    w.store.add_subject('CHANH', True)
    take = w.store.create_take('CHANH', 1, 'session_trial')
    save_json(take/'metadata.json', {
        'person_id': 'CHANH', 'session_id': 'session_trial',
        'label': 1, 'status': 'needs_review', 'cameras': {}})
    w.set_page(2)
    tree = w.library
    person = tree.topLevelItem(0)
    assert person.text(0) == 'CHANH'
    assert not person.isExpanded()
    # Click in the row body, which is the normal user interaction.
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton,
                     pos=tree.visualItemRect(person).center())
    assert person.isExpanded()
    assert person.childCount() == 1
    w.refresh_library()
    person = tree.topLevelItem(0)
    assert person.isExpanded()
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton,
                     pos=tree.visualItemRect(person).center())
    assert not person.isExpanded()


def test_replay_page_selection_and_frame_overlay(window, tmp_path):
    root = window.store.root
    # Test a real MP4 with a one-frame YOLO17 row, no live pose weights required.
    folder = root/'recordings'/'TQB'/'session_replay'/'label_0'/'take_001'
    (folder/'keypoints').mkdir(parents=True)
    (folder/'metadata.json').write_text(json.dumps({'person_id':'TQB',
        'session_id':'session_replay','label':0,'status':'ok','cameras':{'cam01':{'view':'front'}}}))
    video = folder/'cam01_raw.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160,120))
    assert writer.isOpened()
    for _ in range(12): writer.write(np.zeros((120,160,3),dtype=np.uint8))
    writer.release()
    raw = folder/'keypoints'/'cam01_yolo17_raw.txt'
    raw.write_text(''.join(json.dumps({'backend':'yolo','frame_index':i,'width':160,'height':120,
        'relative_ms':i*100, 'status':'ok','keypoints':[[50,60,0.95] for _ in range(17)],
        'bbox':[20,20,100,100]})+'\n' for i in range(12)))
    window.set_page(4)
    assert window.replay_inspector.choose('TQB','session_replay',0,'take_001','cam01')
    viewer=window.replay_inspector
    assert viewer.frame_total==12
    viewer.seek(5)
    assert viewer.frame_index==5 and 'Frame 6/12' in viewer.frame_info.text()
    viewer.next_btn.click(); assert viewer.frame_index==6
    viewer.prev_btn.click(); assert viewer.frame_index==5
    window.set_page(0)
    assert not viewer.playing


def test_full_mode_does_not_allow_scoped_filter(window):
    # Full mode must use a distinct selection branch, not a silently enlarged
    # Person+Label scope in the overwrite controller.
    window.store.add_subject('TQB', True)
    folder=window.store.create_take('TQB', 0, 'session_test')
    save_json(folder/'metadata.json', {'person_id':'TQB','session_id':'session_test',
        'label':0,'status':'ok','cameras':{'cam01':{'view':'front'}}})
    (folder/'cam01_raw.mp4').write_bytes(b'video')
    window.refresh_library()
    window.set_page(3)
    window.set_tool_task('reextract')
    window.reextract_mode.setCurrentIndex(window.reextract_mode.findData('all'))
    assert window.reextract_selection()[0]['camera']=='cam01'
    assert not window.re_people.isEnabled()
    assert not window.re_label.isEnabled()
    assert window.run_reextract_btn.text()=='Re-extract ALL clips'
