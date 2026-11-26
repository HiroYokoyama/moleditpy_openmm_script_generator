import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MOLEDITPY_HEADLESS", "1")


@pytest.fixture(scope="session")
def qapp():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app
    # Destroy every widget while Qt is still alive.  A dialog kept by the
    # plugin's module-level context would otherwise be destroyed during
    # interpreter shutdown, after Qt, and abort the process.
    plugin = sys.modules.get("openmm_script_generator")
    if plugin is not None:
        plugin._context = None
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QtCore = pytest.importorskip("PySide6.QtCore")
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    app.processEvents()
