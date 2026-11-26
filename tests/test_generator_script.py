"""End-to-end generation; runs the scripts under OpenMM when one is available.

Set ``OPENMM_PYTHON`` to an interpreter with OpenMM to run the OpenMM tier
when the test interpreter has none.
"""

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest
from helpers import example

from openmm_script_generator import mdff, options
from openmm_script_generator.generator import MoleculeData, generate

ELEMENTS = ["C", "C", "O"] + ["H"] * 6 + ["O", "H", "H"] * 4 + ["Na", "Cl"]
BONDS = (
    [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7), (2, 8)]
    + [(9 + 3 * w, 10 + 3 * w) for w in range(4)]
    + [(9 + 3 * w, 11 + 3 * w) for w in range(4)]
)


def _coords():
    base = np.array(
        [
            [0, 0, 0],
            [1.52, 0, 0],
            [2.0, 1.35, 0],
            [-0.4, 1.0, 0],
            [-0.4, -0.5, 0.9],
            [-0.4, -0.5, -0.9],
            [1.9, -0.5, 0.9],
            [1.9, -0.5, -0.9],
            [2.9, 1.3, 0],
        ],
        dtype=float,
    )
    waters = []
    for w in range(4):
        o = np.array([6.0 + 3.0 * w, 0, 0])
        waters += [o, o + [0.96, 0, 0], o + [-0.24, 0.93, 0]]
    return np.vstack([base, waters, [[0, 6, 0], [0, -6, 0]]])


def _data():
    return MoleculeData(list(ELEMENTS), _coords(), list(BONDS), title="test")


def _settings(**changes):
    settings = options.get_default_settings()
    settings.update(box_mode="lengths", box_a=30.0, box_b=30.0, box_c=30.0)
    settings.update(changes)
    return options.normalize_settings(settings)


def _doc():
    return mdff.parse_mdff(example("ethanol_water_nacl.mdff.json"))


def test_generated_files_and_config():
    result = generate(_data(), _settings(barostat=True, xyz=True), "0.1.0", _doc())
    assert result.ok, result.errors
    assert list(result.files) == ["run_system.py", "system_mdff.xml", "system.pdb"]
    script = result.files["run_system.py"]
    compile(script, "run_system.py", "exec")
    namespace = {}
    exec(  # noqa: S102 - the CONFIG block of the generated script
        compile(script.split("HERE = ")[0], "cfg", "exec"),
        namespace,
    )
    config = namespace["CONFIG"]
    assert config["templates"] == {"ETOH": "MDFF_ETOH", "HOH": "MDFF_SOL", "NA": "MDFF_NA", "CL": "MDFF_CL"}
    assert config["forcefield_files"] == ["amber14-all.xml", "amber14/tip3p.xml"]
    assert config["barostat"] is True and config["reporters"]["xyz"] == "system_traj.xyz"
    assert "CRYST1   30.000   30.000   30.000" in result.files["system.pdb"]
    assert "23 atoms in 7 residues (7 from mdff)" in result.summary[0]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"cutoff": 2.0}, "half the smallest box width"),
        ({"protein_ff": "charmm36"}, "cannot combine"),
    ],
)
def test_blocking_errors(changes, message):
    result = generate(_data(), _settings(**changes), "0.1.0", _doc())
    assert any(message in e for e in result.errors)


def test_mismatched_mdff_and_missing_water_model_are_errors():
    data = _data()
    data.elements[0] = "N"
    assert any("moleditpy-mdff" in e for e in generate(data, _settings(), "0.1.0", _doc()).errors)
    result = generate(_data(), _settings(water_model="none"), "0.1.0")
    assert any("water model" in e for e in result.errors)


def test_without_mdff_small_molecules_are_reported_and_sdf_written_for_openmmforcefields():
    result = generate(_data(), _settings(), "0.1.0")
    assert result.ok and any("No parameters for residue(s) L01" in w for w in result.warnings)
    data = _data()
    data.sdf_for_atoms = lambda atoms: f"{len(atoms)} atoms\n$$$$\n"
    result = generate(data, _settings(small_molecule_generator="smirnoff"), "0.1.0")
    assert result.files["system_L01.sdf"] == "9 atoms\n$$$$\n"
    assert "openmmforcefields (smirnoff) for: L01" in result.summary[-1]


def test_non_periodic_falls_back_to_nocutoff_and_drops_the_barostat():
    result = generate(_data(), _settings(box_mode="none", barostat=True), "0.1.0", _doc())
    assert result.ok
    namespace = {}
    exec(  # noqa: S102 - the CONFIG block of the generated script
        compile(result.files["run_system.py"].split("HERE = ")[0], "cfg", "exec"),
        namespace,
    )
    assert namespace["CONFIG"]["nonbonded_method"] == "NoCutoff"
    assert namespace["CONFIG"]["barostat"] is False
    assert any("barostat" in w for w in result.warnings)
    assert "CRYST1" not in result.files["system.pdb"]


def test_empty_molecule():
    result = generate(MoleculeData([], np.zeros((0, 3))), _settings(), "0.1.0")
    assert result.errors == ["No molecule is loaded."]


def test_presets_and_settings_store(tmp_path):
    settings = options.apply_preset(options.get_default_settings(), "NPT with HMR, 4 fs")
    assert settings["timestep"] == 4.0 and settings["hydrogen_mass"] == 1.5
    with pytest.raises(KeyError):
        options.apply_preset(settings, "nope")
    path = str(tmp_path / "settings.json")
    options.save_store(path, {"last": {"steps": 5}, "presets": {"mine": {"steps": 7}}})
    store = options.load_store(path)
    assert options.apply_preset(settings, "mine", store["presets"])["steps"] == 7
    assert options.normalize_settings({"steps": "12", "water_model": "nope"})["steps"] == 12
    assert options.normalize_settings({"water_model": "nope"})["water_model"] == "tip3p"
    assert options.forcefield_files({"protein_ff": "charmm36", "water_model": "tip3p"}) == [
        "charmm36.xml",
        "charmm36/water.xml",
    ]
    assert options.forcefield_files({"protein_ff": "none", "water_model": "opc"}) == ["opc.xml"]


# -- OpenMM tier -------------------------------------------------------------------


def _openmm_python():
    explicit = os.environ.get("OPENMM_PYTHON")
    if explicit:
        return explicit
    if importlib.util.find_spec("openmm") is not None:
        return sys.executable
    return None


OPENMM_PYTHON = _openmm_python()
needs_openmm = pytest.mark.skipif(OPENMM_PYTHON is None, reason="OpenMM is not available")


def _write(result, folder):
    for name, text in result.files.items():
        (folder / name).write_text(text, encoding="utf-8", newline="\n")


@needs_openmm
@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"protein_ff": "none", "water_model": "none", "barostat": True},
    ],
)
def test_check_runs_in_openmm(tmp_path, changes):
    result = generate(_data(), _settings(**changes), "0.1.0", _doc())
    _write(result, tmp_path)
    run = subprocess.run(
        [OPENMM_PYTHON, str(tmp_path / result.script_name), "--check", "--platform", "Reference"],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert "CHECK OK" in run.stdout


@needs_openmm
def test_short_run_writes_reporters(tmp_path):
    settings = _settings(
        steps=20, report_interval=10, trajectory_interval=10, xyz=True, minimize_max_iterations=50
    )
    result = generate(_data(), settings, "0.1.0", _doc())
    _write(result, tmp_path)
    run = subprocess.run(
        [OPENMM_PYTHON, str(tmp_path / result.script_name), "--platform", "Reference"],
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    for name in ("system_traj.dcd", "system_traj.xyz", "system_state.csv", "system_final.pdb", "system.chk"):
        assert (tmp_path / name).exists(), name
    xyz = (tmp_path / "system_traj.xyz").read_text().splitlines()
    assert xyz[0] == "23" and xyz[1].startswith('Lattice="')


@needs_openmm
def test_unparameterised_molecule_is_explained(tmp_path):
    result = generate(_data(), _settings(), "0.1.0")
    _write(result, tmp_path)
    run = subprocess.run(
        [OPENMM_PYTHON, str(tmp_path / result.script_name), "--check"],
        capture_output=True,
        check=False,
        text=True,
        timeout=300,
    )
    assert run.returncode == 2 and "No force field parameters for: L01" in run.stderr
