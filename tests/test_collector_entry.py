"""Explicit dataset root must bypass the legacy-dir warning without deleting files."""
import importlib
import sys
import types
from pathlib import Path

import pytest


def test_collector_explicit_data_overrides_legacy_guard(monkeypatch, tmp_path):
    entry = importlib.import_module('collector.main')
    monkeypatch.setattr(entry, '__file__', str(tmp_path/'main.py'))
    legacy = tmp_path / 'Real_Posture_Dataset'
    legacy.mkdir()
    target = tmp_path / 'runtime'
    monkeypatch.setattr(entry, 'DEFAULT_COLLECTOR_DATA_ROOT', target)

    gui = types.ModuleType('posture_collector.ui')
    gui.launch = lambda path: Path(path)
    monkeypatch.setitem(sys.modules, 'posture_collector.ui', gui)

    monkeypatch.setattr(sys, 'argv', ['main.py'])
    with pytest.raises(SystemExit, match='Old collector'):
        entry.main()

    monkeypatch.setattr(sys, 'argv', ['main.py', '--data', str(target)])
    assert entry.main() == target
    assert legacy.is_dir()
