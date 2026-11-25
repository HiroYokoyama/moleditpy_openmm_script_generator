"""Periodic box for the OpenMM system.

Sources, in the order ``auto`` tries them: the suite's ``moleditpy_cell``
molecule property, the CIF Viewer panel's lattice, then a box around the
molecule with padding.  ``lengths`` and ``none`` are explicit user choices.

PDB CRYST1 records describe the cell by lengths and angles, and OpenMM builds
the box vectors in the standard orientation (a along x, b in the xy plane).
Coordinates are therefore re-expressed in that orientation through fractional
coordinates, which is a rigid rotation for a right-handed lattice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .cell_model import cell_vectors, lattice_parameters

CELL_PROPERTY = "moleditpy_cell"


@dataclass
class CellSpec:
    lattice: np.ndarray  # rows are cell vectors, Angstrom
    origin: np.ndarray
    pbc: Tuple[bool, bool, bool] = (True, True, True)
    source: str = ""


@dataclass
class Box:
    coords: np.ndarray  # Angstrom, in the box frame
    lengths: Optional[Tuple[float, float, float]] = None
    angles: Optional[Tuple[float, float, float]] = None
    source: str = "none"
    notes: List[str] = field(default_factory=list)

    @property
    def periodic(self) -> bool:
        return self.lengths is not None

    def describe(self) -> str:
        if not self.periodic:
            return "No periodic box (non-periodic system)."
        a, b, c = self.lengths
        al, be, ga = self.angles
        text = f"{a:.3f} x {b:.3f} x {c:.3f} A"
        if max(abs(al - 90), abs(be - 90), abs(ga - 90)) > 1e-3:
            text += f", angles {al:.2f} / {be:.2f} / {ga:.2f} deg"
        return f"{text} (from {self.source})"


def parse_cell_json(text: str, source: str = CELL_PROPERTY) -> CellSpec:
    """Parse the suite's cell JSON; raises ValueError when unusable."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{source} must be a JSON object")
    lattice = np.asarray(data.get("lattice"), dtype=float)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError(f"{source} has no 3x3 lattice")
    origin = np.asarray(data.get("origin", [0.0, 0.0, 0.0]), dtype=float)
    if origin.shape != (3,):
        origin = np.zeros(3)
    pbc = data.get("pbc", [True, True, True])
    if not isinstance(pbc, (list, tuple)) or len(pbc) != 3:
        pbc = [True, True, True]
    return CellSpec(lattice, origin, tuple(bool(p) for p in pbc), source)


def cell_from_molecule(mol) -> Optional[CellSpec]:
    """The ``moleditpy_cell`` property of an RDKit molecule, or None."""
    if mol is None:
        return None
    try:
        if not mol.HasProp(CELL_PROPERTY):
            return None
        return parse_cell_json(mol.GetProp(CELL_PROPERTY), "the molecule's moleditpy_cell")
    except ValueError:
        raise
    except Exception:  # pragma: no cover - non-RDKit objects
        return None


def cell_from_viewer(widget) -> Optional[CellSpec]:
    """Lattice of the structure shown in the CIF Viewer panel, or None."""
    structure = getattr(widget, "structure", None) if widget is not None else None
    lattice = getattr(structure, "lattice", None)
    if lattice is None:
        return None
    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3):
        return None
    return CellSpec(lattice, np.zeros(3), (True, True, True), "the CIF Viewer panel")


def cell_to_json(lattice, origin=(0.0, 0.0, 0.0), pbc=(True, True, True)) -> str:
    return json.dumps(
        {
            "lattice": [[float(v) for v in row] for row in np.asarray(lattice, dtype=float)],
            "origin": [float(v) for v in origin],
            "pbc": [bool(p) for p in pbc],
        }
    )


def box_from_cell(coords: np.ndarray, spec: CellSpec) -> Box:
    lattice = np.asarray(spec.lattice, dtype=float)
    det = float(np.linalg.det(lattice))
    if abs(det) < 1e-9:
        raise ValueError(f"The cell from {spec.source} has zero volume.")
    if det < 0:
        raise ValueError(
            f"The cell from {spec.source} is left-handed; OpenMM needs a right-handed "
            "cell. Swap two cell vectors in the builder."
        )
    lengths, angles = lattice_parameters(lattice)
    standard = cell_vectors(lengths, angles)
    fractional = (np.asarray(coords, dtype=float) - spec.origin) @ np.linalg.inv(lattice)
    box = Box(fractional @ standard, tuple(lengths), tuple(angles), spec.source)
    if not all(spec.pbc):
        box.notes.append(
            f"The cell from {spec.source} is not periodic along every axis; OpenMM is "
            "periodic in all three directions or none, so the cell is used as fully periodic."
        )
    if not np.allclose(standard, lattice, atol=1e-6):
        box.notes.append(
            "The cell was rotated into OpenMM's orientation (a along x, b in the xy plane); "
            "coordinates were rotated with it."
        )
    return box


def box_from_padding(coords: np.ndarray, padding: float) -> Box:
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        raise ValueError("The molecule has no atoms.")
    low, high = coords.min(axis=0), coords.max(axis=0)
    lengths = np.maximum(high - low + 2.0 * float(padding), 1.0)
    shifted = coords - low + float(padding)
    return Box(shifted, tuple(float(v) for v in lengths), (90.0, 90.0, 90.0),
               f"the molecule's extent plus {float(padding):g} A padding")


def box_from_lengths(coords: np.ndarray, lengths: Sequence[float]) -> Box:
    coords = np.asarray(coords, dtype=float)
    lengths = tuple(float(v) for v in lengths)
    if min(lengths) <= 0:
        raise ValueError("Box lengths must be positive.")
    centre = (coords.min(axis=0) + coords.max(axis=0)) / 2.0 if len(coords) else np.zeros(3)
    shifted = coords - centre + np.asarray(lengths) / 2.0
    box = Box(shifted, lengths, (90.0, 90.0, 90.0), "the box lengths entered in the dialog")
    extent = coords.max(axis=0) - coords.min(axis=0) if len(coords) else np.zeros(3)
    if np.any(extent > np.asarray(lengths)):
        box.notes.append("The molecule is larger than the box along at least one axis.")
    return box


def resolve_box(coords, mode: str, molecule_cell=None, viewer_cell=None,
                padding: float = 10.0, lengths=(30.0, 30.0, 30.0)) -> Box:
    """Pick the box for ``mode`` in ``auto``/``molecule``/``cif``/``padding``/``lengths``/``none``."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if mode == "none":
        return Box(coords.copy(), source="none")
    if mode == "padding":
        return box_from_padding(coords, padding)
    if mode == "lengths":
        return box_from_lengths(coords, lengths)
    if mode in ("auto", "molecule") and molecule_cell is not None:
        return box_from_cell(coords, molecule_cell)
    if mode == "molecule":
        raise ValueError("The current molecule has no moleditpy_cell property.")
    if mode in ("auto", "cif") and viewer_cell is not None:
        return box_from_cell(coords, viewer_cell)
    if mode == "cif":
        raise ValueError("The CIF Viewer panel is not open or holds no structure.")
    box = box_from_padding(coords, padding)
    box.notes.append("No cell was found on the molecule or in the CIF Viewer; a padded box is used.")
    return box
