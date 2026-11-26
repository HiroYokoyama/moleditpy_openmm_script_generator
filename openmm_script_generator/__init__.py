"""OpenMM Script Generator: a self-contained OpenMM run script and PDB from the current molecule.

Force fields come from OpenMM's built-in Amber14 / CHARMM36 files for
proteins, water and ions, from a ``moleditpy-mdff`` file (Force Field Builder)
converted to an OpenMM ForceField XML, or from openmmforcefields when the user
has it.  OpenMM itself is never imported by the plugin.
"""

import logging
import os

PLUGIN_NAME = "OpenMM Script Generator"
PLUGIN_VERSION = "0.1.3"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Writes a self-contained OpenMM run script and a PDB (with CRYST1) from the current "
    "molecule: Amber14/CHARMM36 for proteins, water and ions, small molecules from a "
    "moleditpy-mdff file or openmmforcefields; PME, constraints, Langevin, barostat, "
    "reporters, presets and an optional OpenMM check."
)
PLUGIN_CATEGORY = "Input Generator"
PLUGIN_TAGS = ["Generator"]
PLUGIN_DEPENDENCIES = []
PLUGIN_OPTIONAL_DEPENDENCIES = ["openmm", "openmmforcefields"]
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=5.0.0, <6.0.0"
PLUGIN_LICENSE = "LGPL-3.0-or-later"

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")
WINDOW_ID = "openmm_script_generator_dialog"
CELL_PROPERTY = "moleditpy_cell"

_log = logging.getLogger(__name__)

_context = None
_dialog_opened = False
#: last moleditpy_cell seen on the molecule: {"cell": json text, "n_atoms": int}
_remembered_cell = {}


def get_default_settings():
    from .options import get_default_settings as defaults

    return defaults()


current_settings = {}


def _remember_cell(mol):
    try:
        if mol is not None and mol.HasProp(CELL_PROPERTY):
            _remembered_cell.clear()
            _remembered_cell.update({"cell": mol.GetProp(CELL_PROPERTY), "n_atoms": int(mol.GetNumAtoms())})
    except Exception as exc:  # noqa: BLE001 - pragma: no cover - RDKit guard
        _log.debug("%s: could not remember the cell: %s", PLUGIN_NAME, exc)


def _remembered_cell_for(n_atoms):
    if _remembered_cell.get("n_atoms") == n_atoms:
        return _remembered_cell.get("cell")
    return None


def _restore_cell_on_molecule():
    """Put the saved cell back on the molecule after a project load (the host drops it)."""
    if _context is None or not _remembered_cell:
        return
    try:
        mol = _context.current_molecule
        if mol is None or mol.HasProp(CELL_PROPERTY):
            return
        if mol.GetNumAtoms() == _remembered_cell.get("n_atoms"):
            mol.SetProp(CELL_PROPERTY, _remembered_cell["cell"])
    except Exception as exc:  # noqa: BLE001 - pragma: no cover - host API guard
        _log.debug("%s: could not restore the cell: %s", PLUGIN_NAME, exc)


def _open_dialog(mw):
    """Open the dialog (not named ``run``: the host would add a second menu entry)."""
    global _dialog_opened

    if _context is not None:
        mw = _context.get_main_window()
        existing = _context.get_window(WINDOW_ID)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

    from .main_dialog import OpenMMScriptDialog

    def _get_molecule():
        try:
            if _context is not None:
                return _context.current_molecule
        except Exception as exc:  # noqa: BLE001 - pragma: no cover - host API guard
            _log.warning("%s: could not read the molecule: %s", PLUGIN_NAME, exc)
        return getattr(mw, "current_mol", None)

    def _get_cif_viewer():
        try:
            from .structure_panel import find_cif_viewer_widget

            return find_cif_viewer_widget(mw)
        except Exception as exc:  # noqa: BLE001 - pragma: no cover - other plugin's internals
            _log.debug("%s: CIF Viewer lookup failed: %s", PLUGIN_NAME, exc)
            return None

    def _mark_modified():
        if _context is not None:
            try:
                _context.mark_project_modified()
            except Exception:  # noqa: BLE001, S110 - pragma: no cover - host API guard
                pass

    _dialog_opened = True
    dlg = OpenMMScriptDialog(
        parent=mw,
        persistent_settings=current_settings,
        get_molecule=_get_molecule,
        get_cif_viewer=_get_cif_viewer,
        get_remembered_cell=_remembered_cell_for,
        remember_cell=_remember_cell,
        store_path=SETTINGS_FILE,
        version=PLUGIN_VERSION,
        mark_modified=_mark_modified,
    )
    if _context is not None:
        _context.register_window(WINDOW_ID, dlg)
    dlg.show()


def initialize(context):
    """Entry point for the V4 plugin API."""
    global _context
    _context = context

    def show_dialog():
        _open_dialog(context.get_main_window())

    context.add_export_action("OpenMM Script...", show_dialog)

    def save_state():
        try:
            _remember_cell(context.current_molecule)
        except Exception:  # noqa: BLE001, S110 - pragma: no cover - host API guard
            pass
        state = {}
        if _dialog_opened:
            from .options import normalize_settings

            state["settings"] = normalize_settings(current_settings)
        if _remembered_cell:
            state["cell"] = dict(_remembered_cell)
        return state

    def load_state(data):
        if not isinstance(data, dict):
            return
        saved = data.get("settings")
        if isinstance(saved, dict):
            current_settings.clear()
            current_settings.update(saved)
            dlg = context.get_window(WINDOW_ID)
            if dlg is not None:
                try:
                    dlg.apply_settings(current_settings)
                except Exception as exc:  # noqa: BLE001 - pragma: no cover - host API guard
                    _log.warning("%s: could not apply loaded state: %s", PLUGIN_NAME, exc)
        cell = data.get("cell")
        if isinstance(cell, dict) and isinstance(cell.get("cell"), str):
            _remembered_cell.clear()
            _remembered_cell.update({"cell": cell["cell"], "n_atoms": int(cell.get("n_atoms", -1))})
            # Load handlers run before the host restores the molecule.
            try:
                from PySide6.QtCore import QTimer

                QTimer.singleShot(0, _restore_cell_on_molecule)
            except Exception:  # noqa: BLE001, S110 - pragma: no cover - Qt guard
                pass

    def handle_reset():
        global _dialog_opened
        dlg = context.get_window(WINDOW_ID)
        if dlg is not None and dlg.isVisible():
            return
        current_settings.clear()
        _remembered_cell.clear()
        _dialog_opened = False

    context.register_save_handler(save_state)
    context.register_load_handler(load_state)
    context.register_document_reset_handler(handle_reset)
