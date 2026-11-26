"""OpenMM Script Generator dialog: options on the left, live preview on the right."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

from PySide6.QtCore import QProcess, Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import cell as cell_mod
from . import options
from .mdff import MdffError, load_mdff

PREVIEW_LINES = 400
CURRENT_PRESET = "(current settings)"


def _combo(items):
    box = QComboBox()
    for key, label in items:
        box.addItem(label, key)
    return box


def _dspin(lo, hi, decimals, step, suffix=""):
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _ispin(lo, hi, step=1, suffix=""):
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    if suffix:
        spin.setSuffix(suffix)
    return spin


class OpenMMScriptDialog(QDialog):
    def __init__(
        self,
        parent=None,
        persistent_settings=None,
        get_molecule=None,
        get_cif_viewer=None,
        get_remembered_cell=None,
        remember_cell=None,
        store_path=None,
        version="",
        mark_modified=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("OpenMM Script Generator")
        self.resize(1180, 760)
        self.persistent_settings = persistent_settings if persistent_settings is not None else {}
        self.get_molecule = get_molecule
        self.get_cif_viewer = get_cif_viewer
        self.get_remembered_cell = get_remembered_cell
        self.remember_cell = remember_cell
        self.store_path = store_path
        self.version = version
        self.mark_modified = mark_modified
        self._updating = False
        self._mdff_cache = (None, None, None)  # path, mtime, doc
        self._result = None
        self._process = None
        self._store = options.load_store(store_path) if store_path else {"last": {}, "presets": {}}
        self.widgets = {}

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.update_preview)

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_preset_row())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_option_tabs())
        splitter.addWidget(self._build_preview())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 750])
        layout.addWidget(splitter, 1)
        layout.addLayout(self._build_buttons())

        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save_files)

        initial = options.normalize_settings({**self._store.get("last", {}), **self.persistent_settings})
        self.apply_settings(initial)

    # -- construction ------------------------------------------------------

    def _register(self, key, widget):
        self.widgets[key] = widget
        if isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._changed)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(self._changed)
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(self._changed)
        else:
            widget.valueChanged.connect(self._changed)
        return widget

    def _build_preset_row(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(240)
        row.addWidget(self.preset_combo)
        self.apply_preset_button = QPushButton("Apply")
        self.apply_preset_button.clicked.connect(self.apply_selected_preset)
        row.addWidget(self.apply_preset_button)
        self.save_preset_button = QPushButton("Save as preset...")
        self.save_preset_button.clicked.connect(self.save_preset)
        row.addWidget(self.save_preset_button)
        self.delete_preset_button = QPushButton("Delete preset")
        self.delete_preset_button.clicked.connect(self.delete_preset)
        row.addWidget(self.delete_preset_button)
        row.addStretch(1)
        self._refresh_presets()
        return row

    def _build_option_tabs(self):
        tabs = QTabWidget()

        # System tab
        system = QWidget()
        sys_layout = QVBoxLayout(system)
        self.molecule_label = QLabel("-")
        self.molecule_label.setWordWrap(True)
        sys_layout.addWidget(self.molecule_label)

        box_group = QGroupBox("Periodic box")
        form = QFormLayout(box_group)
        form.addRow("Source:", self._register("box_mode", _combo(options.BOX_MODES)))
        form.addRow("Padding:", self._register("padding", _dspin(0.0, 200.0, 2, 1.0, " A")))
        lengths = QHBoxLayout()
        for key in ("box_a", "box_b", "box_c"):
            lengths.addWidget(self._register(key, _dspin(1.0, 10000.0, 2, 1.0, " A")))
        form.addRow("a / b / c:", lengths)
        sys_layout.addWidget(box_group)

        ff_group = QGroupBox("Force field")
        form = QFormLayout(ff_group)
        self.family_combo = self._register("protein_ff", _combo(options.PROTEIN_FORCE_FIELDS))
        form.addRow("Built-in:", self.family_combo)
        self.water_combo = self._register("water_model", QComboBox())
        form.addRow("Water / ions:", self.water_combo)
        mdff_row = QHBoxLayout()
        self.mdff_edit = self._register("mdff_path", QLineEdit())
        self.mdff_edit.setPlaceholderText("*.mdff.json from the Force Field Builder (optional)")
        mdff_row.addWidget(self.mdff_edit, 1)
        browse = QPushButton("...")
        browse.setFixedWidth(32)
        browse.clicked.connect(self.browse_mdff)
        mdff_row.addWidget(browse)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self.mdff_edit.setText(""))
        mdff_row.addWidget(clear)
        form.addRow("moleditpy-mdff:", mdff_row)
        form.addRow(
            "Other molecules:",
            self._register("small_molecule_generator", _combo(options.SMALL_MOLECULE_GENERATORS)),
        )
        form.addRow("OpenFF version:", self._register("smirnoff_version", QLineEdit()))
        form.addRow("GAFF version:", self._register("gaff_version", QLineEdit()))
        hydrogens = QHBoxLayout()
        hydrogens.addWidget(
            self._register("add_hydrogens", QCheckBox("Add missing hydrogens (standard residues), pH"))
        )
        hydrogens.addWidget(self._register("ph", _dspin(0.0, 14.0, 1, 0.5)))
        form.addRow(hydrogens)
        sys_layout.addWidget(ff_group)
        sys_layout.addStretch(1)
        tabs.addTab(system, "System")

        # Simulation tab
        sim = QWidget()
        sim_layout = QVBoxLayout(sim)
        group = QGroupBox("Interactions")
        form = QFormLayout(group)
        form.addRow(
            "Nonbonded method:", self._register("nonbonded_method", _combo(options.NONBONDED_METHODS))
        )
        form.addRow("Cutoff:", self._register("cutoff", _dspin(0.3, 5.0, 2, 0.1, " nm")))
        form.addRow("Ewald tolerance:", self._register("ewald_tolerance", _dspin(1e-6, 1e-2, 6, 1e-4)))
        form.addRow(self._register("dispersion_correction", QCheckBox("Long-range dispersion correction")))
        form.addRow("Constraints:", self._register("constraints", _combo(options.CONSTRAINTS)))
        form.addRow(self._register("rigid_water", QCheckBox("Rigid water")))
        form.addRow(
            "Hydrogen mass (0 = off):", self._register("hydrogen_mass", _dspin(0.0, 5.0, 2, 0.5, " amu"))
        )
        sim_layout.addWidget(group)

        group = QGroupBox("Dynamics")
        form = QFormLayout(group)
        form.addRow("Integrator:", self._register("integrator", _combo(options.INTEGRATORS)))
        form.addRow("Temperature:", self._register("temperature", _dspin(1.0, 5000.0, 2, 10.0, " K")))
        form.addRow("Friction:", self._register("friction", _dspin(0.0, 100.0, 3, 0.1, " /ps")))
        form.addRow("Time step:", self._register("timestep", _dspin(0.1, 10.0, 2, 0.5, " fs")))
        form.addRow(self._register("barostat", QCheckBox("Monte Carlo barostat (NPT)")))
        form.addRow("Pressure:", self._register("pressure", _dspin(0.0, 100000.0, 3, 1.0, " bar")))
        form.addRow(
            "Barostat interval:", self._register("barostat_frequency", _ispin(1, 100000, 5, " steps"))
        )
        form.addRow("MD steps:", self._register("steps", _ispin(0, 2_000_000_000, 1000)))
        sim_layout.addWidget(group)

        group = QGroupBox("Minimisation and platform")
        form = QFormLayout(group)
        form.addRow(self._register("minimize", QCheckBox("Minimise before MD")))
        form.addRow(
            "Tolerance:", self._register("minimize_tolerance", _dspin(0.001, 10000.0, 3, 1.0, " kJ/mol/nm"))
        )
        form.addRow(
            "Max iterations (0 = until converged):",
            self._register("minimize_max_iterations", _ispin(0, 10_000_000, 100)),
        )
        form.addRow("Platform:", self._register("platform", _combo([(p, p) for p in options.PLATFORMS])))
        form.addRow(
            "Precision (CUDA/OpenCL):",
            self._register("precision", _combo([(p, p) for p in options.PRECISIONS])),
        )
        sim_layout.addWidget(group)
        sim_layout.addStretch(1)
        tabs.addTab(sim, "Simulation")

        # Output tab
        out = QWidget()
        out_layout = QVBoxLayout(out)
        group = QGroupBox("Files")
        form = QFormLayout(group)
        form.addRow("Base name:", self._register("base_name", QLineEdit()))
        form.addRow(self._register("dcd", QCheckBox("DCD trajectory")))
        form.addRow(self._register("xyz", QCheckBox("Extended XYZ trajectory (Large System Viewer)")))
        form.addRow(self._register("pdb_trajectory", QCheckBox("Multi-model PDB trajectory")))
        form.addRow(
            "Trajectory interval:",
            self._register("trajectory_interval", _ispin(1, 100_000_000, 100, " steps")),
        )
        form.addRow(self._register("state_csv", QCheckBox("State data CSV (energies, temperature, density)")))
        form.addRow(
            "Report interval:", self._register("report_interval", _ispin(1, 100_000_000, 100, " steps"))
        )
        form.addRow(self._register("checkpoint", QCheckBox("Checkpoints")))
        form.addRow(
            "Checkpoint interval:",
            self._register("checkpoint_interval", _ispin(1, 100_000_000, 1000, " steps")),
        )
        out_layout.addWidget(group)
        group = QGroupBox("Check with OpenMM")
        form = QFormLayout(group)
        self.python_edit = self._register("check_python", QLineEdit())
        self.python_edit.setPlaceholderText("Python with OpenMM (blank: this Python)")
        python_row = QHBoxLayout()
        python_row.addWidget(self.python_edit, 1)
        pick = QPushButton("...")
        pick.setFixedWidth(32)
        pick.clicked.connect(self.browse_python)
        python_row.addWidget(pick)
        form.addRow("Python:", python_row)
        self.openmm_label = QLabel("")
        self.openmm_label.setWordWrap(True)
        form.addRow(self.openmm_label)
        out_layout.addWidget(group)
        out_layout.addStretch(1)
        tabs.addTab(out, "Output")
        return tabs

    def _build_preview(self):
        self.preview_tabs = QTabWidget()
        font = QFont("Courier New", 9)
        self.messages = QPlainTextEdit()
        self.messages.setReadOnly(True)
        self.preview_tabs.addTab(self.messages, "Summary")
        self.script_view = QPlainTextEdit()
        self.pdb_view = QPlainTextEdit()
        self.xml_view = QPlainTextEdit()
        for view, title in (
            (self.script_view, "Run script"),
            (self.pdb_view, "PDB"),
            (self.xml_view, "Force field XML"),
        ):
            view.setReadOnly(True)
            view.setFont(font)
            view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            self.preview_tabs.addTab(view, title)
        self.check_view = QPlainTextEdit()
        self.check_view.setReadOnly(True)
        self.check_view.setFont(font)
        self.preview_tabs.addTab(self.check_view, "OpenMM check")
        return self.preview_tabs

    def _build_buttons(self):
        row = QHBoxLayout()
        self.status_label = QLabel("")
        row.addWidget(self.status_label, 1)
        self.check_button = QPushButton("Check with OpenMM")
        self.check_button.clicked.connect(self.run_check)
        row.addWidget(self.check_button)
        self.save_button = QPushButton("Save files...")
        self.save_button.clicked.connect(self.save_files)
        row.addWidget(self.save_button)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        return row

    # -- settings ------------------------------------------------------------

    def _fill_water(self, family, selected):
        self.water_combo.blockSignals(True)
        self.water_combo.clear()
        for key, label in options.WATER_MODELS.get(family, options.WATER_MODELS["amber14"]):
            self.water_combo.addItem(label, key)
        index = self.water_combo.findData(selected)
        self.water_combo.setCurrentIndex(max(index, 0))
        self.water_combo.blockSignals(False)

    def apply_settings(self, settings) -> None:
        settings = options.normalize_settings(settings or {})
        self._updating = True
        try:
            self._fill_water(settings["protein_ff"], settings["water_model"])
            for key, widget in self.widgets.items():
                if key == "water_model":
                    continue
                value = settings[key]
                if isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                else:
                    widget.setValue(value)
        finally:
            self._updating = False
        self.update_preview()

    def read_settings(self) -> dict:
        result = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QComboBox):
                result[key] = widget.currentData()
            elif isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                result[key] = widget.text().strip()
            else:
                result[key] = widget.value()
        return options.normalize_settings(result)

    def _changed(self, *_args):
        if self._updating:
            return
        if self.sender() is self.family_combo:
            self._fill_water(self.family_combo.currentData(), self.water_combo.currentData())
        self._timer.start()

    # -- presets ---------------------------------------------------------------

    def _refresh_presets(self, select=None):
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(CURRENT_PRESET)
        for name in options.BUILTIN_PRESETS:
            self.preset_combo.addItem(name)
        for name in sorted(self._store.get("presets", {})):
            self.preset_combo.addItem(f"{name} (user)", name)
        if select:
            index = self.preset_combo.findText(select)
            if index >= 0:
                self.preset_combo.setCurrentIndex(index)
        self.preset_combo.blockSignals(False)

    def _selected_preset(self):
        text = self.preset_combo.currentText()
        if text == CURRENT_PRESET:
            return None, False
        data = self.preset_combo.currentData()
        if data:
            return data, True
        return text, False

    def apply_selected_preset(self):
        name, _user = self._selected_preset()
        if name is None:
            return
        self.apply_settings(options.apply_preset(self.read_settings(), name, self._store.get("presets")))
        self.status_label.setText(f"Applied preset '{name}'.")

    def save_preset(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if name in options.BUILTIN_PRESETS:
            QMessageBox.warning(self, "Save preset", "That name belongs to a built-in preset.")
            return
        settings = self.read_settings()
        for key in ("mdff_path", "check_python", "base_name"):
            settings.pop(key, None)
        self._store.setdefault("presets", {})[name] = settings
        self._save_store()
        self._refresh_presets(select=f"{name} (user)")

    def delete_preset(self):
        name, user = self._selected_preset()
        if not user:
            QMessageBox.information(self, "Delete preset", "Only user presets can be deleted.")
            return
        self._store.get("presets", {}).pop(name, None)
        self._save_store()
        self._refresh_presets()

    def _save_store(self):
        if not self.store_path:
            return
        try:
            options.save_store(self.store_path, self._store)
        except OSError as exc:
            self.status_label.setText(f"Could not save settings.json: {exc}")

    # -- inputs ----------------------------------------------------------------

    def browse_mdff(self):
        start = os.path.dirname(self.mdff_edit.text()) if self.mdff_edit.text() else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "moleditpy-mdff file", start, "moleditpy-mdff (*.mdff.json *.json);;All files (*)"
        )
        if path:
            self.mdff_edit.setText(path)

    def browse_python(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Python with OpenMM", "", "Python (python*);;All files (*)"
        )
        if path:
            self.python_edit.setText(path)

    def _mdff_document(self, path):
        if not path:
            return None
        mtime = os.path.getmtime(path) if os.path.exists(path) else None
        cached_path, cached_mtime, doc = self._mdff_cache
        if cached_path == path and cached_mtime == mtime and doc is not None:
            return doc
        doc = load_mdff(path)
        self._mdff_cache = (path, mtime, doc)
        return doc

    def _cells(self, mol):
        molecule_cell = None
        notes = []
        try:
            molecule_cell = cell_mod.cell_from_molecule(mol)
        except ValueError as exc:
            notes.append(str(exc))
        if molecule_cell is not None and self.remember_cell is not None:
            self.remember_cell(mol)
        if molecule_cell is None and self.get_remembered_cell is not None and mol is not None:
            text = self.get_remembered_cell(mol.GetNumAtoms())
            if text:
                try:
                    molecule_cell = cell_mod.parse_cell_json(text, "the cell saved with the project")
                except ValueError:
                    molecule_cell = None
        viewer_cell = None
        if self.get_cif_viewer is not None:
            try:
                viewer_cell = cell_mod.cell_from_viewer(self.get_cif_viewer())
            except Exception:  # noqa: BLE001 - pragma: no cover - other plugin's internals
                viewer_cell = None
        return molecule_cell, viewer_cell, notes

    # -- preview ---------------------------------------------------------------

    def _persist(self, settings):
        self.persistent_settings.clear()
        self.persistent_settings.update(settings)
        self._store["last"] = settings
        self._save_store()
        if self.mark_modified is not None:
            try:
                self.mark_modified()
            except Exception:  # noqa: BLE001, S110 - pragma: no cover - host API guard
                pass

    def update_preview(self, *_args) -> None:
        if self._updating:
            return
        from .generator import generate
        from .molecule import molecule_data

        settings = self.read_settings()
        if settings != options.normalize_settings(self.persistent_settings):
            self._persist(settings)
        self._update_widget_states(settings)

        mol = self.get_molecule() if self.get_molecule is not None else None
        self._result = None
        messages = []
        if mol is None or mol.GetNumAtoms() == 0:
            self.molecule_label.setText("No molecule is loaded.")
            self._show_messages(["No molecule is loaded. Build or open a 3D structure first."], [], [])
            self._clear_views()
            return
        self.molecule_label.setText(f"Current molecule: {mol.GetNumAtoms()} atoms.")
        try:
            doc = self._mdff_document(settings["mdff_path"])
        except MdffError as exc:
            self._show_messages([], [], [f"moleditpy-mdff: {exc}"])
            self._clear_views()
            return
        molecule_cell, viewer_cell, notes = self._cells(mol)
        try:
            data = molecule_data(mol, "MoleditPy system")
        except ValueError as exc:
            self._show_messages([], [], [str(exc)])
            self._clear_views()
            return
        result = generate(data, settings, self.version, doc, molecule_cell, viewer_cell)
        result.warnings[:0] = notes
        self._result = result
        self._show_messages(result.summary + messages, result.warnings, result.errors)
        if not result.ok:
            self._clear_views()
            self.save_button.setEnabled(False)
            self.check_button.setEnabled(False)
            return
        self.save_button.setEnabled(True)
        self.check_button.setEnabled(True)
        base = settings["base_name"] or "system"
        self.script_view.setPlainText(result.files.get(result.script_name, ""))
        pdb = result.files.get(f"{base}.pdb", "")
        lines = pdb.splitlines()
        if len(lines) > PREVIEW_LINES:
            pdb = "\n".join(lines[:PREVIEW_LINES]) + f"\n... ({len(lines) - PREVIEW_LINES} more lines)\n"
        self.pdb_view.setPlainText(pdb)
        self.xml_view.setPlainText(result.files.get(f"{base}_mdff.xml", "(no moleditpy-mdff file loaded)"))

    def _clear_views(self):
        for view in (self.script_view, self.pdb_view, self.xml_view):
            view.setPlainText("")
        self.save_button.setEnabled(False)
        self.check_button.setEnabled(False)

    def _show_messages(self, summary, warnings, errors):
        parts = []
        if errors:
            parts.append("ERRORS (nothing can be written until these are fixed):")
            parts += [f"  - {e}" for e in errors]
            parts.append("")
        parts += summary
        if warnings:
            parts.append("")
            parts.append("Warnings:")
            parts += [f"  - {w}" for w in warnings]
        self.messages.setPlainText("\n".join(parts))
        if errors:
            self.status_label.setText(f"{len(errors)} error(s); see Summary.")
        elif warnings:
            self.status_label.setText(f"Ready, {len(warnings)} warning(s).")
        else:
            self.status_label.setText("Ready.")

    def _update_widget_states(self, settings):
        mode = settings["box_mode"]
        self.widgets["padding"].setEnabled(mode in ("auto", "padding"))
        for key in ("box_a", "box_b", "box_c"):
            self.widgets[key].setEnabled(mode == "lengths")
        self.widgets["pressure"].setEnabled(settings["barostat"])
        self.widgets["barostat_frequency"].setEnabled(settings["barostat"])
        self.widgets["friction"].setEnabled(settings["integrator"] == "LangevinMiddle")
        self.widgets["ph"].setEnabled(settings["add_hydrogens"])
        generator = settings["small_molecule_generator"]
        self.widgets["smirnoff_version"].setEnabled(generator == "smirnoff")
        self.widgets["gaff_version"].setEnabled(generator == "gaff")
        method = settings["nonbonded_method"]
        self.widgets["cutoff"].setEnabled(method != "NoCutoff")
        self.widgets["ewald_tolerance"].setEnabled(method in ("PME", "LJPME", "Ewald"))
        self.widgets["precision"].setEnabled(settings["platform"] in ("auto", "CUDA", "OpenCL"))
        python = settings["check_python"]
        if python:
            self.openmm_label.setText("The check runs the script with --check in that Python.")
        elif getattr(sys, "frozen", False):
            self.openmm_label.setText("Enter a Python that has OpenMM to run the check.")
        elif importlib.util.find_spec("openmm") is not None:
            self.openmm_label.setText("OpenMM is installed for this Python.")
        else:
            self.openmm_label.setText(
                "OpenMM is not installed for this Python; enter a Python that has it "
                "(for example a conda environment)."
            )

    # -- output ----------------------------------------------------------------

    def _write(self, folder, ask=True) -> bool:
        result = self._result
        existing = [name for name in result.files if os.path.exists(os.path.join(folder, name))]
        if existing and ask:
            answer = QMessageBox.question(
                self,
                "Overwrite files",
                "These files exist and will be overwritten:\n" + "\n".join(existing),
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        for name, text in result.files.items():
            with open(os.path.join(folder, name), "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        return True

    def save_files(self):
        if self._result is None or not self._result.ok:
            QMessageBox.warning(self, "OpenMM Script Generator", "Fix the errors shown in Summary first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the OpenMM files")
        if not folder:
            return
        try:
            if not self._write(folder):
                return
        except OSError as exc:
            QMessageBox.critical(self, "OpenMM Script Generator", f"Could not write the files:\n{exc}")
            return
        self.status_label.setText(f"Wrote {len(self._result.files)} files to {folder}")
        QMessageBox.information(
            self,
            "OpenMM Script Generator",
            "Wrote:\n"
            + "\n".join(self._result.files)
            + f"\n\nto {folder}\n\nRun:  python {self._result.script_name}",
        )

    def _python(self):
        python = self.python_edit.text().strip()
        if python:
            return python
        if getattr(sys, "frozen", False):
            return ""
        return sys.executable

    def run_check(self):
        if self._result is None or not self._result.ok:
            return
        python = self._python()
        if not python:
            QMessageBox.information(self, "Check with OpenMM", "Enter a Python that has OpenMM (Output tab).")
            return
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            return
        self._check_dir = tempfile.mkdtemp(prefix="openmm_check_")
        self._write(self._check_dir, ask=False)
        self.check_view.setPlainText(f"$ {python} {self._result.script_name} --check\n")
        self.preview_tabs.setCurrentWidget(self.check_view)
        self._process = QProcess(self)
        self._process.setWorkingDirectory(self._check_dir)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_check_output)
        self._process.finished.connect(self._check_finished)
        self._process.errorOccurred.connect(self._check_error)
        self.check_button.setEnabled(False)
        self._process.start(python, [os.path.join(self._check_dir, self._result.script_name), "--check"])

    def _read_check_output(self):
        data = bytes(self._process.readAllStandardOutput()).decode("utf-8", "replace")
        self.check_view.moveCursor(self.check_view.textCursor().MoveOperation.End)
        self.check_view.insertPlainText(data)

    def _check_finished(self, code, _status=None):
        self._read_check_output()
        self.check_button.setEnabled(True)
        verdict = "passed" if code == 0 else f"failed (exit code {code})"
        self.check_view.appendPlainText(f"\nCheck {verdict}. Files: {self._check_dir}")
        self.status_label.setText(f"OpenMM check {verdict}.")

    def _check_error(self, _error):
        self.check_button.setEnabled(True)
        self.check_view.appendPlainText(
            f"\nCould not start the Python interpreter: {self._process.errorString()}"
        )

    def closeEvent(self, event):
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.kill()
            self._process.waitForFinished(2000)
        super().closeEvent(event)
