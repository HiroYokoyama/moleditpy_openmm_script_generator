"""Settings, choices and built-in presets."""

from __future__ import annotations

import copy
import json
import os

PROTEIN_FORCE_FIELDS: list[tuple[str, str]] = [
    ("amber14", "Amber14 (ff14SB, nucleic acids, lipids, GLYCAM)"),
    ("charmm36", "CHARMM36"),
    ("none", "None (moleditpy-mdff / openmmforcefields only)"),
]

#: water model id -> label, per protein force field family
WATER_MODELS: dict[str, list[tuple[str, str]]] = {
    "amber14": [
        ("tip3p", "TIP3P"),
        ("tip3pfb", "TIP3P-FB"),
        ("spce", "SPC/E"),
        ("tip4pew", "TIP4P-Ew"),
        ("tip4pfb", "TIP4P-FB"),
        ("opc", "OPC"),
        ("opc3", "OPC3"),
        ("none", "None"),
    ],
    "charmm36": [
        ("tip3p", "CHARMM TIP3P"),
        ("tip3p-pme-b", "TIP3P-PME-b"),
        ("tip3p-pme-f", "TIP3P-PME-f"),
        ("spce", "SPC/E"),
        ("tip4pew", "TIP4P-Ew"),
        ("tip4p2005", "TIP4P/2005"),
        ("tip5p", "TIP5P"),
        ("none", "None"),
    ],
    "none": [
        ("tip3p", "TIP3P"),
        ("tip3pfb", "TIP3P-FB"),
        ("spce", "SPC/E"),
        ("tip4pew", "TIP4P-Ew"),
        ("tip4pfb", "TIP4P-FB"),
        ("opc", "OPC"),
        ("opc3", "OPC3"),
        ("tip5p", "TIP5P"),
        ("none", "None"),
    ],
}

#: models with virtual sites need Modeller.addExtraParticles()
VIRTUAL_SITE_WATERS = frozenset({"tip4pew", "tip4pfb", "opc", "tip4p2005", "tip5p"})

SMALL_MOLECULE_GENERATORS = [
    ("none", "None"),
    ("smirnoff", "OpenFF (SMIRNOFF) via openmmforcefields"),
    ("gaff", "GAFF via openmmforcefields (needs AmberTools)"),
]

NONBONDED_METHODS = [
    ("PME", "PME"),
    ("LJPME", "LJ-PME"),
    ("Ewald", "Ewald"),
    ("CutoffPeriodic", "Cutoff (periodic, reaction field)"),
    ("CutoffNonPeriodic", "Cutoff (non-periodic, reaction field)"),
    ("NoCutoff", "No cutoff"),
]
PERIODIC_METHODS = frozenset({"PME", "LJPME", "Ewald", "CutoffPeriodic"})

CONSTRAINTS = [
    ("HBonds", "Bonds to hydrogen"),
    ("None", "None"),
    ("AllBonds", "All bonds"),
    ("HAngles", "All bonds and angles with hydrogen"),
]

INTEGRATORS = [
    ("LangevinMiddle", "Langevin middle (NVT)"),
    ("Verlet", "Verlet (NVE)"),
]

PLATFORMS = ["auto", "CUDA", "OpenCL", "CPU", "Reference"]
PRECISIONS = ["mixed", "single", "double"]

BOX_MODES = [
    ("auto", "Automatic (molecule cell, CIF Viewer, else padding)"),
    ("molecule", "Cell of the current molecule"),
    ("cif", "Cell of the CIF Viewer panel"),
    ("padding", "Box around the molecule with padding"),
    ("lengths", "Box lengths"),
    ("none", "No box (non-periodic)"),
]


def get_default_settings() -> dict:
    return {
        "base_name": "system",
        "protein_ff": "amber14",
        "water_model": "tip3p",
        "mdff_path": "",
        "small_molecule_generator": "none",
        "smirnoff_version": "openff-2.2.0",
        "gaff_version": "gaff-2.11",
        "add_hydrogens": False,
        "ph": 7.0,
        "box_mode": "auto",
        "padding": 10.0,
        "box_a": 30.0,
        "box_b": 30.0,
        "box_c": 30.0,
        "nonbonded_method": "PME",
        "cutoff": 1.0,
        "ewald_tolerance": 0.0005,
        "dispersion_correction": True,
        "constraints": "HBonds",
        "rigid_water": True,
        "hydrogen_mass": 0.0,
        "integrator": "LangevinMiddle",
        "temperature": 300.0,
        "friction": 1.0,
        "timestep": 2.0,
        "barostat": False,
        "pressure": 1.0,
        "barostat_frequency": 25,
        "minimize": True,
        "minimize_tolerance": 10.0,
        "minimize_max_iterations": 0,
        "steps": 50000,
        "report_interval": 1000,
        "trajectory_interval": 1000,
        "dcd": True,
        "xyz": False,
        "pdb_trajectory": False,
        "state_csv": True,
        "checkpoint": True,
        "checkpoint_interval": 10000,
        "platform": "auto",
        "precision": "mixed",
        "check_python": "",
    }


#: Settings each built-in preset changes; everything else keeps its value.
BUILTIN_PRESETS: dict[str, dict] = {
    "Vacuum quick test": {
        "box_mode": "none",
        "nonbonded_method": "NoCutoff",
        "barostat": False,
        "integrator": "LangevinMiddle",
        "timestep": 1.0,
        "constraints": "HBonds",
        "hydrogen_mass": 0.0,
        "steps": 10000,
        "report_interval": 500,
        "trajectory_interval": 500,
        "dcd": False,
        "xyz": True,
    },
    "Solvated NVT, 300 K": {
        "box_mode": "auto",
        "nonbonded_method": "PME",
        "cutoff": 1.0,
        "barostat": False,
        "integrator": "LangevinMiddle",
        "temperature": 300.0,
        "timestep": 2.0,
        "constraints": "HBonds",
        "hydrogen_mass": 0.0,
        "steps": 50000,
    },
    "Solvated NPT, 300 K, 1 bar": {
        "box_mode": "auto",
        "nonbonded_method": "PME",
        "cutoff": 1.0,
        "barostat": True,
        "pressure": 1.0,
        "integrator": "LangevinMiddle",
        "temperature": 300.0,
        "timestep": 2.0,
        "constraints": "HBonds",
        "hydrogen_mass": 0.0,
        "steps": 500000,
    },
    "NPT with HMR, 4 fs": {
        "box_mode": "auto",
        "nonbonded_method": "PME",
        "barostat": True,
        "integrator": "LangevinMiddle",
        "timestep": 4.0,
        "constraints": "HBonds",
        "hydrogen_mass": 1.5,
        "steps": 250000,
    },
}


def normalize_settings(settings: dict) -> dict:
    """Defaults for missing keys, values coerced to the default's type."""
    result = get_default_settings()
    for key, default in list(result.items()):
        if key not in settings:
            continue
        value = settings[key]
        try:
            if isinstance(default, bool):
                result[key] = bool(value)
            elif isinstance(default, int):
                result[key] = int(value)
            elif isinstance(default, float):
                result[key] = float(value)
            else:
                result[key] = str(value)
        except (TypeError, ValueError):
            pass
    families = dict(PROTEIN_FORCE_FIELDS)
    if result["protein_ff"] not in families:
        result["protein_ff"] = "amber14"
    if result["water_model"] not in dict(WATER_MODELS[result["protein_ff"]]):
        result["water_model"] = "tip3p"
    return result


def apply_preset(settings: dict, name: str, user_presets=None) -> dict:
    preset = BUILTIN_PRESETS.get(name)
    if preset is None and user_presets:
        preset = user_presets.get(name)
    if preset is None:
        raise KeyError(name)
    merged = copy.deepcopy(settings)
    merged.update(preset)
    return normalize_settings(merged)


def forcefield_files(settings: dict) -> list[str]:
    """Built-in OpenMM force field files for the selected families."""
    family = settings.get("protein_ff", "amber14")
    water = settings.get("water_model", "tip3p")
    files = []
    if family == "amber14":
        files.append("amber14-all.xml")
        if water != "none":
            files.append(f"amber14/{water}.xml")
    elif family == "charmm36":
        files.append("charmm36.xml")
        if water != "none":
            files.append("charmm36/water.xml" if water == "tip3p" else f"charmm36/{water}.xml")
    elif water != "none":
        files.append(f"{water}.xml")
    return files


def load_store(path: str) -> dict:
    """``{"last": settings, "presets": {name: settings}}`` from settings.json."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {"last": {}, "presets": {}}
    if not isinstance(data, dict):
        return {"last": {}, "presets": {}}
    presets = data.get("presets") if isinstance(data.get("presets"), dict) else {}
    last = data.get("last") if isinstance(data.get("last"), dict) else {}
    return {"last": last, "presets": presets}


def save_store(path: str, store: dict) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(store, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)
