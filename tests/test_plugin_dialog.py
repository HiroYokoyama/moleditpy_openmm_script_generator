"""Plugin entry point and dialog (real PySide6 offscreen, RDKit)."""

import json
import os

import pytest

pytest.importorskip("PySide6.QtWidgets")
Chem = pytest.importorskip("rdkit.Chem")
from helpers import EXAMPLES
from rdkit.Chem import AllChem

import openmm_script_generator as plugin


def _ethanol(cell=True):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    if cell:
        mol.SetProp(
            "moleditpy_cell",
            json.dumps(
                {
                    "lattice": [[30, 0, 0], [0, 30, 0], [0, 0, 30]],
                    "origin": [-15, -15, -15],
                    "pbc": [True] * 3,
                }
            ),
        )
    return mol


class FakeContext:
    def __init__(self, mol=None):
        self.current_molecule = mol
        self.actions = []
        self.handlers = {}
        self.windows = {}
        self.modified = 0

    def add_export_action(self, label, callback):
        self.actions.append((label, callback))

    def register_save_handler(self, cb):
        self.handlers["save"] = cb

    def register_load_handler(self, cb):
        self.handlers["load"] = cb

    def register_document_reset_handler(self, cb):
        self.handlers["reset"] = cb

    def get_main_window(self):
        return None

    def register_window(self, key, window):
        self.windows[key] = window

    def get_window(self, key):
        return self.windows.get(key)

    def mark_project_modified(self):
        self.modified += 1


def test_metadata():
    assert plugin.PLUGIN_VERSION == "0.1.2"
    assert plugin.PLUGIN_AUTHOR == "HiroYokoyama"
    assert plugin.PLUGIN_LICENSE == "LGPL-3.0-or-later"
    assert plugin.PLUGIN_SUPPORTED_MOLEDITPY_VERSION == ">=5.0.0, <6.0.0"
    assert plugin.PLUGIN_DEPENDENCIES == []
    assert not hasattr(plugin, "run")


def test_initialize_save_load_and_cell_memory(qapp):
    mol = _ethanol()
    ctx = FakeContext(mol)
    plugin.initialize(ctx)
    assert ctx.actions[0][0] == "OpenMM Script..."
    state = ctx.handlers["save"]()
    assert state["cell"]["n_atoms"] == 9 and "settings" not in state  # dialog never opened
    # A reload (the host drops the property) restores it once the event loop turns.
    ctx.current_molecule = _ethanol(cell=False)
    ctx.handlers["load"]({"cell": state["cell"], "settings": {"steps": 1234}})
    assert plugin.current_settings["steps"] == 1234
    plugin._restore_cell_on_molecule()
    assert ctx.current_molecule.HasProp("moleditpy_cell")
    ctx.handlers["reset"]()
    assert plugin._remembered_cell == {} and plugin.current_settings == {}


def test_dialog_preview_presets_and_save(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    from openmm_script_generator.main_dialog import OpenMMScriptDialog

    mol = _ethanol()
    settings = {}
    store = str(tmp_path / "settings.json")
    marks = []
    dlg = OpenMMScriptDialog(
        persistent_settings=settings,
        get_molecule=lambda: mol,
        store_path=store,
        version="0.1.0",
        mark_modified=lambda: marks.append(1),
    )
    assert "1 unparameterised" in dlg.messages.toPlainText()
    dlg.mdff_edit.setText(os.path.join(EXAMPLES, "ethanol.mdff.json"))
    dlg.update_preview()
    assert "1 from mdff" in dlg.messages.toPlainText()
    assert "MDFF_ETOH" in dlg.xml_view.toPlainText()
    assert "CRYST1   30.000" in dlg.pdb_view.toPlainText()
    assert dlg.save_button.isEnabled() and marks
    assert settings["mdff_path"].endswith("ethanol.mdff.json")

    dlg.preset_combo.setCurrentText("Solvated NPT, 300 K, 1 bar")
    dlg.apply_selected_preset()
    assert dlg.read_settings()["barostat"] is True
    monkeypatch.setattr(
        "openmm_script_generator.main_dialog.QInputDialog.getText", lambda *a, **k: ("mine", True)
    )
    dlg.save_preset()
    with open(store, encoding="utf-8") as handle:
        assert "mine" in json.load(handle)["presets"]
    dlg.preset_combo.setCurrentText("mine (user)")
    dlg.delete_preset()
    with open(store, encoding="utf-8") as handle:
        assert "mine" not in json.load(handle)["presets"]

    dlg.widgets["protein_ff"].setCurrentIndex(dlg.widgets["protein_ff"].findData("charmm36"))
    dlg.update_preview()
    assert "cannot combine" in dlg.messages.toPlainText() and not dlg.save_button.isEnabled()
    dlg.widgets["protein_ff"].setCurrentIndex(dlg.widgets["protein_ff"].findData("amber14"))
    dlg.update_preview()

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    dlg.save_files()
    assert (tmp_path / "run_system.py").exists() and (tmp_path / "system_mdff.xml").exists()
    dlg.close()


def test_dialog_without_molecule(qapp, tmp_path):
    from openmm_script_generator.main_dialog import OpenMMScriptDialog

    dlg = OpenMMScriptDialog(get_molecule=lambda: None, store_path=str(tmp_path / "s.json"))
    assert "No molecule" in dlg.messages.toPlainText()
    assert not dlg.save_button.isEnabled()
    dlg.mdff_edit.setText(str(tmp_path / "missing.mdff.json"))
    dlg.close()


def test_open_dialog_is_a_singleton(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(plugin, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    ctx = FakeContext(_ethanol())
    plugin.initialize(ctx)
    plugin._open_dialog(None)
    first = ctx.windows[plugin.WINDOW_ID]
    plugin._open_dialog(None)
    assert ctx.windows[plugin.WINDOW_ID] is first
    assert "settings" in ctx.handlers["save"]()
    first.close()


def test_molecule_data_extracts_residue_info_and_sdf():
    from openmm_script_generator.molecule import molecule_data

    mol = Chem.MolFromPDBBlock(
        "HETATM    1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "HETATM    2  C1  LIG A   2       3.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    3  O1  LIG A   2       4.400   0.000   0.000  1.00  0.00           O\n"
        "CONECT    2    3\nEND\n",
        removeHs=False,
    )
    data = molecule_data(mol)
    assert data.residue_info[0].resname == "HOH" and data.residue_info[1].name == "C1"
    assert data.fragment_labels[1] == data.fragment_labels[2]
    block = data.sdf_for_atoms([1, 2])
    assert block.rstrip().endswith("$$$$") and "V2000" in block
    with pytest.raises(ValueError, match="3D"):
        molecule_data(Chem.MolFromSmiles("C"))


def _assert_tabs_scroll_and_ignore_the_wheel(dlg):
    """Long settings tabs scroll (the dialog fits a laptop screen) and an unfocused
    field does not change when the wheel scrolls over it."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QDoubleSpinBox,
        QScrollArea,
        QSpinBox,
    )

    assert dlg.minimumSizeHint().height() < 600
    pages = [a.widget() for a in dlg.findChildren(QScrollArea) if a.widget() is not None]
    assert len(pages) >= 3
    spins = [
        s
        for page in pages
        for s in page.findChildren(QAbstractSpinBox)
        if isinstance(s, (QSpinBox, QDoubleSpinBox))
        and s.isEnabled()
        and s.value() < s.maximum()
    ]
    spin = spins[0]
    assert spin.focusPolicy() == Qt.FocusPolicy.StrongFocus
    before = spin.value()
    event = QWheelEvent(
        QPointF(5, 5),
        QPointF(5, 5),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(spin, event)
    assert spin.value() == before


def test_settings_tabs_scroll_and_ignore_the_wheel(qapp, tmp_path):
    from openmm_script_generator.main_dialog import OpenMMScriptDialog

    dlg = OpenMMScriptDialog(get_molecule=lambda: None, store_path=str(tmp_path / "s.json"))
    _assert_tabs_scroll_and_ignore_the_wheel(dlg)
    dlg.close()
