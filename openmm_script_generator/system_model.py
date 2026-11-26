"""Split the host molecule into OpenMM residues.

Plain-Python model built from per-atom arrays (elements, bonds, optional PDB
residue information, optional moleditpy-mdff coverage).  Each atom ends up in
exactly one residue of one of these kinds:

``mdff``     a molecule type of the loaded moleditpy-mdff file (pinned template)
``protein``  a standard amino-acid / nucleotide residue from PDB residue info
``water``    water, renamed HOH / O H1 H2 so OpenMM's water models match it
``ion``      a monatomic ion, named as in OpenMM's amber14 ion templates
``other``    anything else; needs parameters from mdff or openmmforcefields
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .ffxml import RESERVED_RESIDUE_NAMES

STANDARD_POLYMER_RESIDUES = frozenset(
    [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "ASH",
        "CYS",
        "CYX",
        "CYM",
        "GLN",
        "GLU",
        "GLH",
        "GLY",
        "HIS",
        "HID",
        "HIE",
        "HIP",
        "HSD",
        "HSE",
        "HSP",
        "ILE",
        "LEU",
        "LYS",
        "LYN",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "ACE",
        "NME",
        "NMA",
        "SEC",
        "DA",
        "DC",
        "DG",
        "DT",
        "A",
        "C",
        "G",
        "U",
        "RA",
        "RC",
        "RG",
        "RU",
        "DA5",
        "DA3",
        "DC5",
        "DC3",
        "DG5",
        "DG3",
        "DT5",
        "DT3",
        "RA5",
        "RA3",
        "RC5",
        "RC3",
        "RG5",
        "RG3",
        "RU5",
        "RU3",
    ]
)

WATER_ALIASES = frozenset({"HOH", "WAT", "SOL", "H2O", "TIP", "TIP3", "TIP3P", "T3P", "SPC", "OPC", "TP3"})

#: element -> (residue name, atom name) as in OpenMM's amber14 ion templates
ION_NAMES: dict[str, tuple[str, str]] = {
    "Li": ("LI", "LI"),
    "Na": ("NA", "NA"),
    "K": ("K", "K"),
    "Rb": ("RB", "RB"),
    "Cs": ("CS", "CS"),
    "Mg": ("MG", "MG"),
    "Ca": ("CA", "CA"),
    "Zn": ("ZN", "ZN"),
    "F": ("F", "F"),
    "Cl": ("CL", "CL"),
    "Br": ("BR", "BR"),
    "I": ("IOD", "I"),
}


@dataclass
class ResidueInfo:
    """PDB residue information of one host atom."""

    resname: str
    name: str
    resnum: int = 1
    chain: str = ""
    icode: str = ""
    hetero: bool = False


@dataclass
class Residue:
    resname: str
    kind: str
    atoms: list[int]  # host atom indices
    names: list[str]
    chain: str = "A"
    resseq: int = 1
    template: str | None = None  # pinned ForceField template name (mdff only)
    label: str = ""  # fragment identity (canonical SMILES when known)

    @property
    def hetero(self) -> bool:
        return self.kind != "protein"


@dataclass
class SystemModel:
    residues: list[Residue]
    bond_definitions: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def order(self) -> list[int]:
        return [index for residue in self.residues for index in residue.atoms]

    @property
    def n_atoms(self) -> int:
        return sum(len(r.atoms) for r in self.residues)

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for residue in self.residues:
            result[residue.kind] = result.get(residue.kind, 0) + 1
        return result

    def other_residue_names(self) -> list[str]:
        names = []
        for residue in self.residues:
            if residue.kind == "other" and residue.resname not in names:
                names.append(residue.resname)
        return names

    def first_residue(self, resname: str) -> Residue | None:
        for residue in self.residues:
            if residue.resname == resname:
                return residue
        return None


def _fragments(n_atoms: int, bonds: Sequence[tuple[int, int]]) -> list[int]:
    parent = list(range(n_atoms))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in bonds:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return [find(i) for i in range(n_atoms)]


def _is_water(atoms: Sequence[int], elements, bonded) -> bool:
    if sorted(elements[a] for a in atoms) != ["H", "H", "O"]:
        return False
    oxygen = next(a for a in atoms if elements[a] == "O")
    return all(frozenset((oxygen, a)) in bonded for a in atoms if a != oxygen)


def _water_names(atoms, elements) -> list[str]:
    names, h = [], 0
    for a in atoms:
        if elements[a] == "O":
            names.append("O")
        else:
            h += 1
            names.append(f"H{h}")
    return names


def _element_names(atoms, elements) -> list[str]:
    counters: dict[str, int] = {}
    names = []
    for a in atoms:
        element = elements[a].upper()
        counters[element] = counters.get(element, 0) + 1
        names.append(f"{element}{counters[element]}"[:4])
    return names


def _unique_names(names: list[str], atoms, elements) -> list[str]:
    if len(set(names)) == len(names) and all(names) and all(len(n) <= 4 for n in names):
        return names
    return _element_names(atoms, elements)


def build_system(
    elements: Sequence[str],
    bonds: Sequence[tuple[int, int]],
    residue_info: Sequence[ResidueInfo | None] | None = None,
    mdff_assignment: Sequence[tuple[str, int, int] | None] | None = None,
    templates=None,
    fragment_labels: Sequence[str | None] | None = None,
) -> SystemModel:
    """Group the atoms into residues.

    ``mdff_assignment`` comes from :func:`mdff.match_system` and ``templates``
    from :func:`ffxml.build_ffxml` (``TemplateInfo`` keyed by type name).
    """
    n = len(elements)
    residue_info = list(residue_info) if residue_info is not None else [None] * n
    mdff_assignment = list(mdff_assignment) if mdff_assignment is not None else [None] * n
    labels = list(fragment_labels) if fragment_labels is not None else [None] * n
    templates = templates or {}
    bonded = {frozenset(b) for b in bonds}
    fragment = _fragments(n, bonds)
    has_bond = {a for b in bonds for a in b}
    warnings: list[str] = []

    # 1. Owner key per atom, then groups in order of first appearance.
    groups: dict[tuple, list[int]] = {}
    for atom in range(n):
        assigned = mdff_assignment[atom]
        info = residue_info[atom]
        if assigned is not None:
            key = ("mdff", assigned[0], assigned[1])
        elif info is not None and info.resname.strip():
            key = ("pdb", info.chain, info.resnum, info.icode, info.resname.strip().upper())
        else:
            key = ("frag", fragment[atom])
        groups.setdefault(key, []).append(atom)

    # Atoms of one fragment without residue info but split across owners stay
    # together through the fragment key; nothing else to merge.
    residues: list[Residue] = []
    other_signatures: dict[tuple, str] = {}
    used_other_names = set()
    label_names: dict[str, int] = {}

    def other_name(base: str, signature: tuple) -> str:
        if signature in other_signatures:
            return other_signatures[signature]
        candidate = base[:4] or "UNL"
        counter = 1
        while (
            candidate in used_other_names
            or candidate in RESERVED_RESIDUE_NAMES
            or candidate in STANDARD_POLYMER_RESIDUES
            or candidate in {v[0] for v in ION_NAMES.values()}
        ):
            suffix = str(counter)
            candidate = (base[: 4 - len(suffix)] or "L") + suffix
            counter += 1
        used_other_names.add(candidate)
        other_signatures[signature] = candidate
        return candidate

    for key, atoms in groups.items():
        kind = key[0]
        if kind == "mdff":
            info = templates.get(key[1])
            if info is None:
                raise ValueError(f"No template for molecule type {key[1]!r}")
            local = [mdff_assignment[a][2] for a in atoms]
            names = [info.atom_names[i] for i in local]
            residues.append(Residue(info.resname, "mdff", atoms, names, template=info.template, label=key[1]))
            continue

        if kind == "pdb":
            resname = key[4]
            chain = (key[1] or "A")[:1]
            infos = [residue_info[a] for a in atoms]
            hho = sorted(elements[a] for a in atoms) == ["H", "H", "O"]
            if hho and (resname in WATER_ALIASES or _is_water(atoms, elements, bonded)):
                residues.append(Residue("HOH", "water", atoms, _water_names(atoms, elements), chain))
                continue
            if len(atoms) == 1 and elements[atoms[0]] in ION_NAMES and atoms[0] not in has_bond:
                resname_ion, atom_name = ION_NAMES[elements[atoms[0]]]
                residues.append(Residue(resname_ion, "ion", atoms, [atom_name], chain))
                continue
            names = _unique_names([i.name.strip().upper() for i in infos], atoms, elements)
            if resname in STANDARD_POLYMER_RESIDUES:
                residues.append(Residue(resname, "protein", atoms, names, chain, key[2]))
                continue
            signature = _signature(atoms, names, elements, bonded, resname)
            residues.append(
                Residue(
                    other_name(resname, signature),
                    "other",
                    atoms,
                    names,
                    chain,
                    label=labels[atoms[0]] or resname,
                )
            )
            continue

        # fragment without residue info
        if _is_water(atoms, elements, bonded):
            residues.append(Residue("HOH", "water", atoms, _water_names(atoms, elements)))
            continue
        if len(atoms) == 1 and elements[atoms[0]] in ION_NAMES:
            resname_ion, atom_name = ION_NAMES[elements[atoms[0]]]
            residues.append(Residue(resname_ion, "ion", atoms, [atom_name]))
            continue
        names = _element_names(atoms, elements)
        label = labels[atoms[0]] or ""
        if label not in label_names:
            label_names[label] = len(label_names) + 1
        base = f"L{label_names[label]:02d}"
        signature = _signature(atoms, names, elements, bonded, label or base)
        residues.append(Residue(other_name(base, signature), "other", atoms, names, label=label))

    # Number residues per chain in order (PDB allows 4 digits; wrap after 9999).
    for number, residue in enumerate(residues, start=1):
        if residue.kind != "protein":
            residue.resseq = number
        residue.chain = residue.chain or "A"

    # Bonds for residues OpenMM does not already know.
    bond_definitions: dict[str, list[tuple[str, str]]] = {}
    for residue in residues:
        if residue.kind not in ("mdff", "other") or residue.resname in bond_definitions:
            continue
        position = {atom: k for k, atom in enumerate(residue.atoms)}
        pairs = []
        for i, j in bonds:
            if i in position and j in position:
                pairs.append((residue.names[position[i]], residue.names[position[j]]))
        if pairs:
            bond_definitions[residue.resname] = pairs

    # Covalent links between hetero residues cannot be represented.
    owner = {}
    for r_index, residue in enumerate(residues):
        for atom in residue.atoms:
            owner[atom] = r_index
    crossing = 0
    for i, j in bonds:
        a, b = residues[owner[i]], residues[owner[j]]
        if owner[i] != owner[j] and (a.kind != "protein" or b.kind != "protein"):
            crossing += 1
    if crossing:
        warnings.append(
            f"{crossing} bond(s) join different residues where one is not a standard "
            "polymer residue; OpenMM templates cannot describe such links."
        )
    return SystemModel(residues, bond_definitions, warnings)


def _signature(atoms, names, elements, bonded, label) -> tuple:
    position = {atom: k for k, atom in enumerate(atoms)}
    pairs = []
    for a in atoms:
        for b in atoms:
            if a < b and frozenset((a, b)) in bonded:
                pairs.append(tuple(sorted((names[position[a]], names[position[b]]))))
    return (
        label,
        tuple(sorted(zip(names, (elements[a] for a in atoms)))),
        tuple(sorted(pairs)),
    )
