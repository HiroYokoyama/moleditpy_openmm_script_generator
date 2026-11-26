"""Reader and validator for the suite's ``moleditpy-mdff`` force field hand-off JSON.

The format is written by the Force Field Builder plugin and read by the
LAMMPS / GROMACS / OpenMM generators.  Units are GROMACS units: nm, kJ/mol,
degrees, elementary charge and amu.  Everything here is plain Python so it can
be tested without Qt, RDKit or numpy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

FORMAT_NAME = "moleditpy-mdff"
SUPPORTED_VERSIONS = (1,)


class MdffError(ValueError):
    """The file is not a usable moleditpy-mdff document."""


@dataclass
class MdffAtom:
    name: str
    element: str
    type: str
    charge: float
    mass: float
    sigma: float
    epsilon: float


@dataclass
class MdffMoleculeType:
    name: str
    atoms: list[MdffAtom]
    bonds: list[tuple[int, int, float, float]]  # i, j, r0 (nm), k (kJ/mol/nm^2)
    angles: list[tuple[int, int, int, float, float]]  # i, j, k, theta0 (deg), k
    # i, j, k, l, periodicity, phase (deg), k (kJ/mol), improper
    torsions: list[tuple[int, int, int, int, int, float, float, bool]]
    exclusions_nrexcl: int = 3
    constraints: list[tuple[int, int, float]] = field(default_factory=list)
    settles: dict[str, float] | None = None
    force_field: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def total_charge(self) -> float:
        return sum(atom.charge for atom in self.atoms)

    @property
    def elements(self) -> list[str]:
        return [atom.element for atom in self.atoms]

    def is_water(self) -> bool:
        return sorted(self.elements) == ["H", "H", "O"]


@dataclass
class MdffSystemEntry:
    type: str
    count: int
    atom_offset: int


@dataclass
class MdffDocument:
    source: str
    combination_rule: int
    fudge_lj: float
    fudge_qq: float
    molecule_types: list[MdffMoleculeType]
    system: list[MdffSystemEntry]
    path: str = ""
    warnings: list[str] = field(default_factory=list)

    def type_by_name(self, name: str) -> MdffMoleculeType | None:
        for mol_type in self.molecule_types:
            if mol_type.name == name:
                return mol_type
        return None

    @property
    def n_system_atoms(self) -> int:
        total = 0
        for entry in self.system:
            mol_type = self.type_by_name(entry.type)
            if mol_type is not None:
                total = max(total, entry.atom_offset + entry.count * mol_type.n_atoms)
        return total


def _number(value: Any, what: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise MdffError(f"{what} is not a number: {value!r}") from None
    if not math.isfinite(result):
        raise MdffError(f"{what} is not finite: {value!r}")
    return result


def _integer(value: Any, what: str) -> int:
    number = _number(value, what)
    if number != int(number):
        raise MdffError(f"{what} must be an integer: {value!r}")
    return int(number)


def _index(value: Any, n_atoms: int, what: str) -> int:
    index = _integer(value, what)
    if not 0 <= index < n_atoms:
        raise MdffError(f"{what} = {index} is outside 0..{n_atoms - 1}")
    return index


def _parse_atom(raw: Any, where: str) -> MdffAtom:
    if not isinstance(raw, dict):
        raise MdffError(f"{where} is not an object")
    element = str(raw.get("element", "")).strip()
    if not element:
        raise MdffError(f"{where} has no element")
    element = element[0].upper() + element[1:].lower()
    return MdffAtom(
        name=str(raw.get("name", "") or element).strip(),
        element=element,
        type=str(raw.get("type", "") or element).strip(),
        charge=_number(raw.get("charge", 0.0), f"{where}.charge"),
        mass=_number(raw.get("mass", 0.0), f"{where}.mass"),
        sigma=_number(raw.get("sigma", 0.0), f"{where}.sigma"),
        epsilon=_number(raw.get("epsilon", 0.0), f"{where}.epsilon"),
    )


def _parse_type(raw: Any, position: int) -> MdffMoleculeType:
    if not isinstance(raw, dict):
        raise MdffError(f"molecule_types[{position}] is not an object")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise MdffError(f"molecule_types[{position}] has no name")
    where = f"molecule type {name!r}"
    raw_atoms = raw.get("atoms")
    if not isinstance(raw_atoms, list) or not raw_atoms:
        raise MdffError(f"{where} has no atoms")
    atoms = [_parse_atom(a, f"{where} atom {i}") for i, a in enumerate(raw_atoms)]
    n = len(atoms)

    bonds = []
    for i, bond in enumerate(raw.get("bonds") or []):
        w = f"{where} bond {i}"
        bonds.append(
            (
                _index(bond.get("i"), n, f"{w}.i"),
                _index(bond.get("j"), n, f"{w}.j"),
                _number(bond.get("r0"), f"{w}.r0"),
                _number(bond.get("k"), f"{w}.k"),
            )
        )
    angles = []
    for i, angle in enumerate(raw.get("angles") or []):
        w = f"{where} angle {i}"
        angles.append(
            (
                _index(angle.get("i"), n, f"{w}.i"),
                _index(angle.get("j"), n, f"{w}.j"),
                _index(angle.get("k_atom"), n, f"{w}.k_atom"),
                _number(angle.get("theta0"), f"{w}.theta0"),
                _number(angle.get("k"), f"{w}.k"),
            )
        )
    torsions = []
    for i, tor in enumerate(raw.get("torsions") or []):
        w = f"{where} torsion {i}"
        torsions.append(
            (
                _index(tor.get("i"), n, f"{w}.i"),
                _index(tor.get("j"), n, f"{w}.j"),
                _index(tor.get("k_atom"), n, f"{w}.k_atom"),
                _index(tor.get("l"), n, f"{w}.l"),
                _integer(tor.get("periodicity"), f"{w}.periodicity"),
                _number(tor.get("phase", 0.0), f"{w}.phase"),
                _number(tor.get("k"), f"{w}.k"),
                bool(tor.get("improper", False)),
            )
        )
    constraints = []
    for i, con in enumerate(raw.get("constraints") or []):
        w = f"{where} constraint {i}"
        constraints.append(
            (
                _index(con.get("i"), n, f"{w}.i"),
                _index(con.get("j"), n, f"{w}.j"),
                _number(con.get("r0", con.get("length")), f"{w}.r0"),
            )
        )
    settles = raw.get("settles")
    if settles is not None and not isinstance(settles, dict):
        settles = {"present": 1.0}
    return MdffMoleculeType(
        name=name,
        atoms=atoms,
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        exclusions_nrexcl=_integer(raw.get("exclusions_nrexcl", 3), f"{where}.exclusions_nrexcl"),
        constraints=constraints,
        settles=settles,
        force_field=str(raw.get("force_field", "") or ""),
    )


def parse_mdff(data: Any, path: str = "") -> MdffDocument:
    """Validate a decoded JSON object and return an :class:`MdffDocument`."""
    if not isinstance(data, dict):
        raise MdffError("The top level of the file must be a JSON object.")
    if data.get("format") != FORMAT_NAME:
        raise MdffError(f'"format" must be "{FORMAT_NAME}", found {data.get("format")!r}.')
    version = data.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise MdffError(
            f"Unsupported moleditpy-mdff version {version!r}; this generator reads "
            f"version {', '.join(str(v) for v in SUPPORTED_VERSIONS)}."
        )
    raw_types = data.get("molecule_types")
    if not isinstance(raw_types, list) or not raw_types:
        raise MdffError('"molecule_types" must be a non-empty list.')
    types = [_parse_type(raw, i) for i, raw in enumerate(raw_types)]
    names = [t.name for t in types]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise MdffError(f"Duplicate molecule type names: {', '.join(duplicates)}")

    system = []
    for i, raw in enumerate(data.get("system") or []):
        if not isinstance(raw, dict):
            raise MdffError(f"system[{i}] is not an object")
        type_name = str(raw.get("type", ""))
        if type_name not in names:
            raise MdffError(f"system[{i}] refers to an unknown molecule type {type_name!r}")
        count = _integer(raw.get("count", 1), f"system[{i}].count")
        offset = _integer(raw.get("atom_offset", 0), f"system[{i}].atom_offset")
        if count < 0 or offset < 0:
            raise MdffError(f"system[{i}] has a negative count or offset")
        system.append(MdffSystemEntry(type_name, count, offset))

    doc = MdffDocument(
        source=str(data.get("source", "")),
        combination_rule=_integer(data.get("combination_rule", 2), "combination_rule"),
        fudge_lj=_number(data.get("fudge_lj", 0.5), "fudge_lj"),
        fudge_qq=_number(data.get("fudge_qq", 0.8333), "fudge_qq"),
        molecule_types=types,
        system=system,
        path=path,
    )
    doc.warnings.extend(document_warnings(doc))
    return doc


def load_mdff(path: str) -> MdffDocument:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise MdffError(f"Could not read {path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise MdffError(f"{path} is not valid JSON: {exc}") from None
    return parse_mdff(data, path=path)


def document_warnings(doc: MdffDocument) -> list[str]:
    """Things OpenMM cannot reproduce exactly, or that look wrong."""
    warnings = []
    if doc.combination_rule != 2:
        warnings.append(
            f"combination_rule {doc.combination_rule} (geometric mixing) cannot be expressed "
            "by OpenMM's NonbondedForce, which always uses Lorentz-Berthelot (rule 2)."
        )
    for mol_type in doc.molecule_types:
        nrexcl = mol_type.exclusions_nrexcl
        # Exclusions beyond the molecule's longest path change nothing.
        if (nrexcl < 3 and mol_type.n_atoms >= 4) or (nrexcl > 3 and mol_type.n_atoms >= 5):
            warnings.append(
                f"{mol_type.name}: nrexcl = {mol_type.exclusions_nrexcl}; OpenMM always "
                "excludes 1-2 and 1-3 pairs and scales 1-4 pairs (nrexcl = 3)."
            )
        charge = mol_type.total_charge
        if abs(charge - round(charge)) > 1e-3:
            warnings.append(f"{mol_type.name}: net charge {charge:+.4f} is not an integer.")
        for atom in mol_type.atoms:
            if atom.mass <= 0.0:
                warnings.append(f"{mol_type.name}: atom {atom.name} has no mass.")
                break
        bonded = {frozenset((i, j)) for i, j, _r, _k in mol_type.bonds}
        for i, j, _r in mol_type.constraints:
            if frozenset((i, j)) not in bonded:
                warnings.append(
                    f"{mol_type.name}: constraint {i}-{j} is not a bond; the run script adds it explicitly."
                )
    return warnings


#: The ``force_field`` label Force Field Builder gives proteins built with Amber
#: ff14SB.  OpenMM's amber14-all.xml is the same force field, so these types are
#: left to it rather than converted.
FF14SB_LABEL = "Amber ff14SB (proteins)"


def without_ff14sb(doc: MdffDocument) -> tuple[MdffDocument, list[str]]:
    """Return ``doc`` without its ff14SB protein types, and their names."""
    names = [t.name for t in doc.molecule_types if t.force_field == FF14SB_LABEL]
    if not names:
        return doc, []
    rest = replace(
        doc,
        molecule_types=[t for t in doc.molecule_types if t.name not in names],
        system=[e for e in doc.system if e.type not in names],
        warnings=list(doc.warnings),
    )
    return rest, names


def match_system(
    doc: MdffDocument, elements: Sequence[str]
) -> tuple[list[tuple[str, int, int] | None], list[str]]:
    """Map every host atom to ``(type name, copy number, index within type)``.

    Returns the per-atom assignment (None for atoms the file does not cover) and
    a list of problems.  Element mismatches make the whole mapping unusable, so
    the assignment is cleared when any are found.
    """
    n = len(elements)
    assignment: list[tuple[str, int, int] | None] = [None] * n
    problems: list[str] = []
    if not doc.system:
        problems.append(
            'The file has no "system" section, so its molecule types cannot be placed '
            "onto the atoms of the current molecule."
        )
        return assignment, problems
    for entry in doc.system:
        mol_type = doc.type_by_name(entry.type)
        if mol_type is None:  # pragma: no cover - parse_mdff rejects this
            continue
        for copy in range(entry.count):
            start = entry.atom_offset + copy * mol_type.n_atoms
            for local, atom in enumerate(mol_type.atoms):
                index = start + local
                if index >= n:
                    problems.append(
                        f"{entry.type} copy {copy + 1} needs atom {index + 1}, but the molecule "
                        f"has only {n} atoms."
                    )
                    return [None] * n, problems
                if assignment[index] is not None:
                    problems.append(f"Atom {index + 1} is claimed by two system entries.")
                    return [None] * n, problems
                if elements[index] != atom.element:
                    problems.append(
                        f"Atom {index + 1} is {elements[index]} in the molecule but "
                        f"{atom.element} ({atom.name}) in {entry.type}."
                    )
                    return [None] * n, problems
                assignment[index] = (entry.type, copy, local)
    return assignment, problems
