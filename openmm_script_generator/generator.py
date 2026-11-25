"""Turn molecule data and settings into the set of output files.

No Qt and no RDKit here: the dialog extracts :class:`MoleculeData` from the
host molecule (see ``molecule.py``) and calls :func:`generate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import cell as cell_mod
from .ffxml import build_ffxml, scale_conflict
from .mdff import MdffDocument, match_system
from .options import PERIODIC_METHODS, forcefield_files
from .pdb_writer import write_pdb
from .script_writer import build_config, build_script
from .system_model import ResidueInfo, build_system


@dataclass
class MoleculeData:
    elements: List[str]
    coords: np.ndarray  # (n, 3) Angstrom
    bonds: List[Tuple[int, int]] = field(default_factory=list)
    residue_info: Optional[List[Optional[ResidueInfo]]] = None
    fragment_labels: Optional[List[Optional[str]]] = None
    formal_charges: Optional[List[int]] = None
    title: str = ""
    #: returns an SDF block for a list of host atom indices (one fragment)
    sdf_for_atoms: Optional[Callable[[Sequence[int]], str]] = None


@dataclass
class GeneratedFiles:
    files: Dict[str, str] = field(default_factory=dict)  # file name -> text
    script_name: str = ""
    summary: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _min_width(lengths, angles) -> float:
    """Smallest perpendicular width of the cell (Angstrom)."""
    from .cell_model import cell_vectors

    vectors = cell_vectors(lengths, angles)
    volume = abs(float(np.linalg.det(vectors)))
    widths = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        widths.append(volume / float(np.linalg.norm(np.cross(vectors[j], vectors[k]))))
    return min(widths)


def generate(
    data: MoleculeData,
    settings: dict,
    version: str,
    doc: Optional[MdffDocument] = None,
    molecule_cell=None,
    viewer_cell=None,
) -> GeneratedFiles:
    out = GeneratedFiles()
    base = (settings.get("base_name") or "system").strip() or "system"
    n = len(data.elements)
    if n == 0:
        out.errors.append("No molecule is loaded.")
        return out

    # 1. moleditpy-mdff coverage
    ffres = None
    assignment = None
    if doc is not None:
        assignment, problems = match_system(doc, data.elements)
        if problems:
            out.errors.extend(f"moleditpy-mdff: {p}" for p in problems)
            assignment = None
        ffres = build_ffxml(doc)
        out.warnings.extend(f"moleditpy-mdff: {w}" for w in ffres.warnings)
        conflict = scale_conflict(doc, settings.get("protein_ff", "amber14"), forcefield_files(settings))
        if conflict:
            out.errors.append(conflict)

    # 2. residues
    system = build_system(
        data.elements,
        data.bonds,
        data.residue_info,
        assignment,
        ffres.templates if ffres is not None else None,
        data.fragment_labels,
    )
    out.warnings.extend(system.warnings)
    counts = system.counts()

    # 3. box
    try:
        box = cell_mod.resolve_box(
            data.coords,
            settings.get("box_mode", "auto"),
            molecule_cell,
            viewer_cell,
            float(settings.get("padding", 10.0)),
            (float(settings.get("box_a", 30.0)), float(settings.get("box_b", 30.0)), float(settings.get("box_c", 30.0))),
        )
    except ValueError as exc:
        out.errors.append(str(exc))
        return out
    out.warnings.extend(box.notes)

    method = settings.get("nonbonded_method", "PME")
    if box.periodic:
        width = _min_width(box.lengths, box.angles)
        cutoff_a = float(settings.get("cutoff", 1.0)) * 10.0
        if method != "NoCutoff" and cutoff_a * 2.0 > width:
            out.errors.append(
                f"The cutoff ({cutoff_a / 10:.2f} nm) must be at most half the smallest box width "
                f"({width / 10:.3f} nm). Use a larger box or a shorter cutoff."
            )
        if method in ("NoCutoff", "CutoffNonPeriodic"):
            out.warnings.append(f"{method} ignores the periodic box written to the PDB.")
    elif method in PERIODIC_METHODS:
        out.warnings.append(f"{method} needs a periodic box; the script uses NoCutoff instead.")

    # 4. force field coverage
    family = settings.get("protein_ff", "amber14")
    water = settings.get("water_model", "tip3p")
    if counts.get("protein") and family == "none":
        out.errors.append(
            f"{counts['protein']} standard protein / nucleic-acid residue(s) need the Amber14 or "
            "CHARMM36 force field."
        )
    if (counts.get("water") or counts.get("ion")) and water == "none":
        out.errors.append(
            "Water or ions outside the moleditpy-mdff file need a water model "
            "(the water files also hold the ion parameters)."
        )
    others = system.other_residue_names()
    generator = settings.get("small_molecule_generator", "none")
    sdf_files: Dict[str, str] = {}
    if others:
        if generator == "none":
            out.warnings.append(
                f"No parameters for residue(s) {', '.join(others)}: load a moleditpy-mdff file "
                "from the Force Field Builder, or select an openmmforcefields generator. The "
                "script stops with an explanation until then."
            )
        elif data.sdf_for_atoms is None:
            out.errors.append("Cannot write SDF files for the small molecules.")
        else:
            for resname in others:
                residue = system.first_residue(resname)
                try:
                    block = data.sdf_for_atoms(residue.atoms)
                except Exception as exc:  # RDKit sanitisation problems
                    out.errors.append(f"Could not write an SDF for residue {resname}: {exc}")
                    continue
                name = f"{base}_{resname}.sdf"
                sdf_files[resname] = name
                out.files[name] = block
    if settings.get("add_hydrogens") and (counts.get("other") or counts.get("mdff")):
        out.warnings.append(
            "Add missing hydrogens works for standard residues only; small molecules must "
            "already carry their hydrogens."
        )

    # 5. dynamics sanity
    timestep = float(settings.get("timestep", 2.0))
    if timestep > 2.0 and float(settings.get("hydrogen_mass", 0.0)) <= 1.008:
        out.warnings.append(
            f"A {timestep:g} fs time step usually needs hydrogen mass repartitioning "
            "(hydrogen mass about 1.5 amu or more)."
        )
    if timestep > 1.0 and settings.get("constraints") == "None":
        out.warnings.append("Time steps above 1 fs need constraints on bonds to hydrogen.")
    if settings.get("barostat") and not box.periodic:
        out.warnings.append("The barostat needs a periodic box and is left out.")
    if settings.get("barostat") and settings.get("integrator") == "Verlet":
        out.warnings.append("A barostat with the Verlet integrator gives NPH, not NPT.")

    # 6. files
    pdb_name = f"{base}.pdb"
    ffxml_name = f"{base}_mdff.xml" if ffres is not None and assignment is not None else None
    templates = {}
    extras = {}
    if ffxml_name:
        out.files[ffxml_name] = ffres.xml
        used = {r.resname for r in system.residues if r.kind == "mdff"}
        templates = {i.resname: i.template for i in ffres.templates.values() if i.resname in used}
        extras = {k: v for k, v in ffres.extras.items() if k in used}
    out.files[pdb_name] = write_pdb(
        system,
        data.elements,
        box.coords,
        box.lengths,
        box.angles,
        data.bonds,
        data.title,
        data.formal_charges,
    )
    config = build_config(
        settings,
        pdb_name,
        ffxml_name,
        templates,
        extras,
        system.bond_definitions,
        sdf_files,
        box.periodic,
    )
    pieces = [f"{system.n_atoms} atoms in {len(system.residues)} residues"]
    kinds = [("mdff", "from mdff"), ("protein", "protein/nucleic"), ("water", "water"),
             ("ion", "ions"), ("other", "unparameterised")]
    detail = ", ".join(f"{counts[k]} {label}" for k, label in kinds if counts.get(k))
    if detail:
        pieces.append(f"({detail})")
    summary_line = " ".join(pieces)
    out.summary.append(summary_line)
    out.summary.append("Box: " + box.describe())
    files = config["forcefield_files"] + ([ffxml_name] if ffxml_name else [])
    out.summary.append("Force field files: " + (", ".join(files) if files else "none"))
    if sdf_files:
        out.summary.append(f"openmmforcefields ({generator}) for: {', '.join(sdf_files)}")
    if ffres is not None and ffres.n_extra_terms and ffxml_name:
        out.summary.append(f"{ffres.n_extra_terms} extra term(s) added by the script")
    script_name = f"run_{base}.py"
    out.script_name = script_name
    out.files = {script_name: build_script(config, version, "; ".join(out.summary), script_name), **out.files}
    return out
