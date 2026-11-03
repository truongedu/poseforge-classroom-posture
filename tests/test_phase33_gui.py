"""Offscreen Qt smoke for Phase 3.3B. ROS pytest plugins must be disabled."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication
from inference.gui import MainWindow, AdvancedDialog, available_classifiers, choose_model_paths
from inference.ui_runtime import Settings


def test_gui_layout_model_options_and_roi(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    root = tmp_path
    for name in ('mlp_wide/v002', 'mlp_base/v003'):
        p = root / 'artifacts/models' / name / 'model_final.pt'
        p.parent.mkdir(parents=True)
        p.touch()
    assert len(available_classifiers(root)) == 2
    win = MainWindow('fake_yolo.pt','fake_model')
    try:
        assert win.device == '0' and win.cls_device == 'cuda:0'
        assert not win.end.isEnabled()
        assert win.mode.currentText() == 'Testcase'
        assert len(win.panels) == 1
        for _ in range(3):
            win.add_source()
        assert len(win.panels) == 4
        assert not win.add.isEnabled()
        assert win.grid.itemAtPosition(1,1).widget() is win.panels[3]
        win.remove_source(win.panels[1])
        assert len(win.panels) == 3
        assert win.grid.itemAtPosition(1,0).widget() is win.panels[2]
        dialog = AdvancedDialog(Settings(stable_frames=8,min_conf=.72),'fake_pose.pt','fake_model')
        assert dialog.result_settings('guarded').stable_frames == 8
        dialog.populate(Settings())
        assert dialog.result_settings('fast') == Settings()
        win.panels[0].view.set_roi([(0.,0.),(.5,0.),(.5,1.)])
        assert len(win.panels[0].view.roi) == 3
        assert 'source' not in win.store.allKeys()
    finally:
        win.close()


def test_start_end_reuses_live_testcase_and_continuous_stops(tmp_path, monkeypatch):
    import threading
    from inference import gui
    app = QApplication.instance() or QApplication([])
    pose = tmp_path / 'pose.pt'
    pose.touch()
    model = tmp_path / 'mlp_wide/v002'
    model.mkdir(parents=True)
    (model/'model_final.pt').touch()
    image = tmp_path/'input.png'
    image.touch()

    class StubSession:
        def __init__(self, *args, **kwargs):
            self.stop = threading.Event()
            self.testcase = False
            self.finished = 0
            self.resumed = 0

        def start(self, testcase=False):
            self.testcase = testcase

        def finish_case(self):
            self.finished += 1

        def next_case(self):
            self.resumed += 1
            return True

        def close(self, timeout=0):
            self.stop.set()
            return True

    monkeypatch.setattr(gui, 'InferenceSession', StubSession)
    win = MainWindow(pose, model)
    try:
        win.panels[0].kind.setCurrentText('image')
        win.panels[0].value.setText(str(image))
        win.start_clicked()
        first = win.session
        assert first.testcase and win.end.isEnabled() and not win.start.isEnabled()
        win.end_clicked()
        assert first.finished == 1
        win.on_event('case',-1,('ready',1))
        assert win.start.isEnabled() and not win.end.isEnabled()
        win.start_clicked()
        assert win.session is first and first.resumed == 1
        win.on_event('state',-1,'READY')
        win.mode.setCurrentText('Continuous')
        win.start_clicked()
        assert not win.session.testcase
        second = win.session
        win.end_clicked()
        assert second.stop.is_set()
    finally:
        win.close()


def test_gui_persists_model_choices_without_camera_urls(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from inference import gui
    app = QApplication.instance() or QApplication([])
    cfg = tmp_path / 'settings.ini'
    def test_store(*_):
        return QSettings(str(cfg), QSettings.Format.IniFormat)
    monkeypatch.setattr(gui, 'QSettings', test_store)
    pose = tmp_path / 'pose.pt'
    pose.touch()
    classifier = tmp_path / 'mlp_wide' / 'v002'
    classifier.mkdir(parents=True)
    (classifier / 'model_final.pt').touch()
    win = MainWindow(pose, classifier)
    try:
        win.panels[0].value.setText('rtsp://user:secret@127.0.0.1/stream')
        win._save_settings()
        persisted = test_store()
        assert persisted.value('pose_model') == str(pose)
        assert persisted.value('classifier_model') == str(classifier)
        assert not any('source' in key or 'url' in key for key in persisted.allKeys())
        assert 'secret' not in cfg.read_text()
        assert choose_model_paths(None, None, persisted) == (pose, classifier)
        alternate = tmp_path / 'explicit.pt'
        assert choose_model_paths(alternate, None, persisted) == (alternate, classifier)
    finally:
        win.close()


def test_advanced_settings_reject_invalid_checkpoints(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    app = QApplication.instance() or QApplication([])
    dialog = AdvancedDialog(Settings(), tmp_path / 'missing.pt', tmp_path / 'missing_model')
    dialogs = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: dialogs.append(args))
    dialog.accept()
    assert dialogs
    assert dialog.result() != dialog.DialogCode.Accepted
