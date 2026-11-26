"""Convert a ``moleditpy-mdff`` document into an OpenMM ForceField XML file.

Every molecule type becomes one residue template whose atoms each get their own
atom type, so every bonded term is written for exactly the atoms it belongs to.
OpenMM matches templates on elements and bonds, and the run script pins each
residue to its template by name, so built-in force fields cannot capture them.

Two kinds of terms cannot be expressed exactly in a ForceField file and are
returned as *extra terms* that the generated run script adds by atom name:

* several impropers on the same centre and neighbour set (OpenMM applies only
  one improper definition per centre and neighbour triple), impropers whose
  atoms do not form a star around one centre, and propers or angles whose atoms
  are not a bonded chain;
* constraints between atoms that are not bonded.

Improper atom order, checked against OpenMM 8.6: with the default ordering an
``<Improper type1=c type2=a type3=b type4=d>`` becomes the torsion (a, b, c, d);
with ``ordering="charmm"`` it becomes (c, a, b, d).  A dihedral equals its
reverse, so a centre in any of the four positions maps onto one of these.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from xml.sax.saxutils import quoteattr

from .mdff import MdffDocument, MdffMoleculeType
from .symmetry import automorphisms

#: Residue names OpenMM already knows bonds for (Topology._standardBonds) or
#: rewrites through pdbNames.xml.  Templates never take these names.
RESERVED_RESIDUE_NAMES = frozenset(
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
        "NH2",
        "NMA",
        "SEC",
        "PYL",
        "DA",
        "DC",
        "DG",
        "DT",
        "DU",
        "DI",
        "A",
        "C",
        "G",
        "U",
        "I",
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
        "HOH",
        "WAT",
        "SOL",
        "H2O",
        "TIP",
        "TIP3",
        "TIP4",
        "TIP5",
        "T3P",
        "T4P",
        "T5P",
        "SPC",
        "OPC",
    ]
)


@dataclass
class TemplateInfo:
    """How one molecule type appears in the PDB and in the XML."""

    type_name: str
    template: str
    resname: str
    atom_names: list[str]
    is_water: bool


@dataclass
class FfxmlResult:
    xml: str
    templates: dict[str, TemplateInfo]  # keyed by molecule type name
    extras: dict[str, list[dict]] = field(default_factory=dict)  # keyed by resname
    warnings: list[str] = field(default_factory=list)

    @property
    def n_extra_terms(self) -> int:
        return sum(len(terms) for terms in self.extras.values())

    def bond_definitions(self, doc: MdffDocument) -> dict[str, list[tuple[str, str]]]:
        """Bonds by atom name for ``Topology.loadBondDefinitions`` (water excluded)."""
        result = {}
        for mol_type in doc.molecule_types:
            info = self.templates.get(mol_type.name)
            if info is None or info.is_water or not mol_type.bonds:
                continue
            names = info.atom_names
            result[info.resname] = [(names[i], names[j]) for i, j, _r, _k in mol_type.bonds]
        return result


def _sanitize(text: str, length: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return cleaned[:length]


def assign_residue_names(doc: MdffDocument, taken=()) -> dict[str, str]:
    """Unique PDB residue names (at most 4 characters) for every molecule type.

    The first water type becomes ``HOH`` so OpenMM's rigid-water handling
    recognises it; other names avoid residues OpenMM treats as standard.
    """
    used = {name.upper() for name in taken}
    result: dict[str, str] = {}
    water_done = False
    for mol_type in doc.molecule_types:
        if mol_type.is_water() and not water_done:
            result[mol_type.name] = "HOH"
            water_done = True
            used.add("HOH")
            continue
        base = _sanitize(mol_type.name, 4) or "MOL"
        candidate = base
        counter = 1
        while candidate in used or (candidate in RESERVED_RESIDUE_NAMES and mol_type.n_atoms > 1):
            suffix = str(counter)
            candidate = (base[: 4 - len(suffix)] or "M") + suffix
            counter += 1
        used.add(candidate)
        result[mol_type.name] = candidate
    return result


def assign_atom_names(mol_type: MdffMoleculeType, water: bool = False) -> list[str]:
    """Unique atom names of at most four characters, stable for one type."""
    if water:
        names = []
        h_count = 0
        for atom in mol_type.atoms:
            if atom.element == "O":
                names.append("O")
            else:
                h_count += 1
                names.append(f"H{h_count}")
        return names
    names: list[str] = []
    used = set()
    counters: dict[str, int] = {}
    for atom in mol_type.atoms:
        name = _sanitize(atom.name, 4) if atom.name else ""
        if not name or name in used:
            element = atom.element.upper()
            while True:
                counters[element] = counters.get(element, 0) + 1
                name = f"{element}{counters[element]}"[:4]
                if name not in used:
                    break
        used.add(name)
        names.append(name)
    return names


def _fmt(value: float) -> str:
    return repr(float(value))


def _chain_key(atoms: tuple[int, ...]) -> tuple[int, ...]:
    return min(atoms, atoms[::-1])


def _star_centre(atoms: tuple[int, int, int, int], bonded) -> int | None:
    """Position (0-3) of the atom bonded to the other three, else None."""
    for position, centre in enumerate(atoms):
        others = [a for a in atoms if a != centre]
        if len(set(atoms)) == 4 and all(frozenset((centre, o)) in bonded for o in others):
            return position
    return None


def build_ffxml(doc: MdffDocument, residue_names: dict[str, str] | None = None) -> FfxmlResult:
    """Return the ForceField XML, template metadata and extra terms."""
    residue_names = residue_names or assign_residue_names(doc)
    templates: dict[str, TemplateInfo] = {}
    extras: dict[str, list[dict]] = {}
    warnings: list[str] = list(doc.warnings)

    atom_types: list[str] = []
    residues: list[str] = []
    bond_lines: list[str] = []
    angle_lines: list[str] = []
    proper_lines: list[str] = []
    improper_default: list[str] = []
    improper_charmm: list[str] = []
    nonbonded_lines: list[str] = []

    used_templates = set()
    for mol_type in doc.molecule_types:
        template = "MDFF_" + (_sanitize(mol_type.name, 24) or "MOL")
        while template in used_templates:
            template += "X"
        used_templates.add(template)
        water = residue_names.get(mol_type.name) == "HOH"
        names = assign_atom_names(mol_type, water=water)
        resname = residue_names[mol_type.name]
        templates[mol_type.name] = TemplateInfo(mol_type.name, template, resname, names, water)
        types = [f"{template}-{i}" for i in range(mol_type.n_atoms)]
        extra = extras.setdefault(resname, [])
        symmetry = _Symmetry(mol_type)
        coulomb14, lj14 = scale_factors(doc)
        for index in symmetry.asymmetric_atoms():
            atom = mol_type.atoms[index]
            extra.append(
                {
                    "kind": "nonbonded",
                    "atoms": [names[index]],
                    "charge": atom.charge,
                    "sigma": atom.sigma,
                    "epsilon": atom.epsilon,
                    "coulomb14": coulomb14,
                    "lj14": lj14,
                }
            )
        if symmetry.asymmetric_masses():
            warnings.append(
                f"{mol_type.name}: equivalent atoms have different masses; OpenMM may assign "
                "them to either atom."
            )

        for t, atom in zip(types, mol_type.atoms):
            atom_types.append(
                f"  <Type name={quoteattr(t)} class={quoteattr(t)} "
                f'element={quoteattr(atom.element)} mass="{_fmt(atom.mass)}"/>'
            )
            nonbonded_lines.append(
                f'  <Atom type={quoteattr(t)} sigma="{_fmt(atom.sigma)}" epsilon="{_fmt(atom.epsilon)}"/>'
            )

        residue = [f"  <Residue name={quoteattr(template)}>"]
        for name, t, atom in zip(names, types, mol_type.atoms):
            residue.append(
                f'   <Atom name={quoteattr(name)} type={quoteattr(t)} charge="{_fmt(atom.charge)}"/>'
            )
        bonded = set()
        for i, j, r0, k in mol_type.bonds:
            key = frozenset((i, j))
            if i == j:
                warnings.append(f"{mol_type.name}: bond {i}-{j} joins an atom to itself; skipped.")
                continue
            if key in bonded:
                extra.append({"kind": "bond", "atoms": [names[i], names[j]], "length": r0, "k": k})
                continue
            bonded.add(key)
            residue.append(f"   <Bond atomName1={quoteattr(names[i])} atomName2={quoteattr(names[j])}/>")
            bond_lines.append(
                f"  <Bond type1={quoteattr(types[i])} type2={quoteattr(types[j])} "
                f'length="{_fmt(r0)}" k="{_fmt(k)}"/>'
            )
        residue.append("  </Residue>")
        residues.extend(residue)

        seen_angles = set()
        for i, j, k_atom, theta0, k in mol_type.angles:
            key = _chain_key((i, j, k_atom))
            chain = frozenset((i, j)) in bonded and frozenset((j, k_atom)) in bonded
            if not chain or key in seen_angles or not symmetry.angle_invariant((i, j, k_atom)):
                extra.append(
                    {
                        "kind": "angle",
                        "atoms": [names[i], names[j], names[k_atom]],
                        "theta0": math.radians(theta0),
                        "k": k,
                    }
                )
                continue
            seen_angles.add(key)
            angle_lines.append(
                f"  <Angle type1={quoteattr(types[i])} type2={quoteattr(types[j])} "
                f'type3={quoteattr(types[k_atom])} angle="{_fmt(math.radians(theta0))}" '
                f'k="{_fmt(k)}"/>'
            )

        propers: dict[tuple[int, ...], list[tuple[int, float, float]]] = {}
        impropers: dict[tuple[int, frozenset], dict[tuple[int, ...], list]] = {}
        for i, j, k_atom, l_atom, n, phase, k, improper in mol_type.torsions:
            atoms = (i, j, k_atom, l_atom)
            term = (n, math.radians(phase), k)
            if k == 0.0:
                continue
            if not symmetry.torsion_invariant(atoms, improper):
                extra.append(_torsion_extra(names, atoms, term))
                continue
            if not improper:
                chain = (
                    len(set(atoms)) == 4
                    and frozenset((i, j)) in bonded
                    and frozenset((j, k_atom)) in bonded
                    and frozenset((k_atom, l_atom)) in bonded
                )
                if chain:
                    propers.setdefault(_chain_key(atoms), []).append(term)
                    continue
            else:
                position = _star_centre(atoms, bonded)
                if position is not None:
                    centre = atoms[position]
                    group = impropers.setdefault((centre, frozenset(a for a in atoms if a != centre)), {})
                    group.setdefault(_chain_key(atoms), []).append(term)
                    continue
            extra.append(_torsion_extra(names, atoms, term))

        for atoms, terms in propers.items():
            proper_lines.append(_torsion_line("Proper", [types[a] for a in atoms], terms))

        for (centre, _neighbours), variants in impropers.items():
            if len(variants) > 1:
                for atoms, terms in variants.items():
                    for term in terms:
                        extra.append(_torsion_extra(names, atoms, term))
                continue
            ((atoms, terms),) = variants.items()
            position = atoms.index(centre)
            if position in (1, 3):  # use the reversed dihedral, which is equal
                atoms = atoms[::-1]
                position = 3 - position
            if position == 2:  # (a, b, c, d) -> default ordering type1=c
                ordered = [atoms[2], atoms[0], atoms[1], atoms[3]]
                improper_default.append(_torsion_line("Improper", [types[a] for a in ordered], terms))
            else:  # centre first -> charmm ordering keeps (c, a, b, d)
                improper_charmm.append(_torsion_line("Improper", [types[a] for a in atoms], terms))

        for i, j, r0 in mol_type.constraints:
            if frozenset((i, j)) not in bonded:
                extra.append({"kind": "constraint", "atoms": [names[i], names[j]], "length": r0})

        if not extra:
            extras.pop(resname, None)

    if any(extras.values()):
        count = sum(len(v) for v in extras.values())
        warnings.append(
            f"{count} term(s) cannot be written as ForceField XML templates; the run script "
            "adds them by atom name after creating the system."
        )

    lines = [
        "<ForceField>",
        " <Info>",
        "  <Source>" + _escape(doc.source or doc.path or "moleditpy-mdff") + "</Source>",
        (
            "  <Reference>Converted from moleditpy-mdff version 1 by the MoleditPy OpenMM "
            "Script Generator.</Reference>"
        ),
        " </Info>",
        " <AtomTypes>",
        *atom_types,
        " </AtomTypes>",
        " <Residues>",
        *residues,
        " </Residues>",
    ]
    if bond_lines:
        lines += [" <HarmonicBondForce>", *bond_lines, " </HarmonicBondForce>"]
    if angle_lines:
        lines += [" <HarmonicAngleForce>", *angle_lines, " </HarmonicAngleForce>"]
    if proper_lines or improper_default:
        lines += [" <PeriodicTorsionForce>", *proper_lines, *improper_default, " </PeriodicTorsionForce>"]
    if improper_charmm:
        lines += [
            ' <PeriodicTorsionForce ordering="charmm">',
            *improper_charmm,
            " </PeriodicTorsionForce>",
        ]
    coulomb14, lj14 = scale_factors(doc)
    lines += [
        f' <NonbondedForce coulomb14scale="{_fmt(coulomb14)}" lj14scale="{_fmt(lj14)}">',
        '  <UseAttributeFromResidue name="charge"/>',
        *nonbonded_lines,
        " </NonbondedForce>",
        "</ForceField>",
        "",
    ]
    return FfxmlResult("\n".join(lines), templates, extras, warnings)


#: OpenMM refuses to combine NonbondedForce tags whose 1-4 scales differ by
#: more than 1e-5, so the usual rounded Amber value 0.8333 is written as 5/6.
SCALE_SNAP = 1e-3


def scale_factors(doc: MdffDocument) -> tuple[float, float]:
    """(coulomb14scale, lj14scale) as written to the XML."""
    coulomb = doc.fudge_qq
    if abs(coulomb - 5.0 / 6.0) < SCALE_SNAP:
        coulomb = 5.0 / 6.0
    return coulomb, doc.fudge_lj


#: 1-4 scales of OpenMM's built-in families (coulomb, lj)
BUILTIN_SCALES = {"amber14": (5.0 / 6.0, 0.5), "charmm36": (1.0, 1.0), "none": (5.0 / 6.0, 0.5)}


def scale_conflict(doc: MdffDocument, family: str, files) -> str:
    """Explanation when the mdff 1-4 scales cannot be combined with ``files``."""
    if not files:
        return ""
    coulomb, lj = scale_factors(doc)
    ref_coulomb, ref_lj = BUILTIN_SCALES.get(family, (5.0 / 6.0, 0.5))
    if abs(coulomb - ref_coulomb) <= 1e-5 and abs(lj - ref_lj) <= 1e-5:
        return ""
    return (
        f"The moleditpy-mdff file scales 1-4 interactions by {doc.fudge_qq:g} (Coulomb) and "
        f"{doc.fudge_lj:g} (LJ), but the built-in {family} files use {ref_coulomb:.4g} and "
        f"{ref_lj:g}. OpenMM cannot combine force fields with different 1-4 scales: choose "
        "a force field family with the same scales, or 'None' with water model 'None' so "
        "that every molecule comes from the mdff file."
    )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class _Symmetry:
    """Which parameters of a molecule type survive OpenMM's symmetric atom mapping."""

    def __init__(self, mol_type: MdffMoleculeType):
        self.mol_type = mol_type
        n = mol_type.n_atoms
        bonds = [(i, j) for i, j, _r, _k in mol_type.bonds]
        perms, complete = automorphisms(mol_type.elements, bonds)
        self.perms = [p for p in perms if any(a != b for a, b in enumerate(p))]
        self.complete = complete
        if not complete:  # assume any atoms of one element and degree may be swapped
            degree = [0] * n
            for i, j in bonds:
                degree[i] += 1
                degree[j] += 1
            self.group = {a: (mol_type.atoms[a].element, degree[a]) for a in range(n)}
            groups: dict[tuple, list[int]] = {}
            for a in range(n):
                groups.setdefault(self.group[a], []).append(a)
            self.free = {a for g in groups.values() if len(g) > 1 for a in g}
        self.angles: dict[tuple, list] = {}
        for i, j, k, theta0, kk in mol_type.angles:
            self.angles.setdefault(_chain_key((i, j, k)), []).append((round(theta0, 9), round(kk, 6)))
        self.torsions: dict[tuple, list] = {}
        for i, j, k, l, n_, phase, kk, improper in mol_type.torsions:
            if kk == 0.0:
                continue
            key = (_chain_key((i, j, k, l)), improper)
            self.torsions.setdefault(key, []).append((n_, round(phase, 9), round(kk, 6)))
        for table in (self.angles, self.torsions):
            for value in table.values():
                value.sort()

    def _atoms_fixed(self, atoms) -> bool:
        return all(a not in self.free for a in atoms)

    def asymmetric_atoms(self) -> list[int]:
        atoms = self.mol_type.atoms

        def params(a):
            return (round(atoms[a].charge, 9), round(atoms[a].sigma, 9), round(atoms[a].epsilon, 9))

        if not self.complete:
            by_group: dict[tuple, set] = {}
            for a in self.free:
                by_group.setdefault(self.group[a], set()).add(params(a))
            return sorted(a for a in self.free if len(by_group[self.group[a]]) > 1)
        bad = set()
        for perm in self.perms:
            for a, b in enumerate(perm):
                if params(a) != params(b):
                    bad.add(a)
        return sorted(bad)

    def asymmetric_masses(self) -> bool:
        atoms = self.mol_type.atoms
        return any(
            abs(atoms[a].mass - atoms[b].mass) > 1e-9 for perm in self.perms for a, b in enumerate(perm)
        )

    def angle_invariant(self, atoms) -> bool:
        if not self.complete:
            return self._atoms_fixed(atoms)
        reference = self.angles.get(_chain_key(atoms))
        return all(self.angles.get(_chain_key(tuple(p[a] for a in atoms))) == reference for p in self.perms)

    def torsion_invariant(self, atoms, improper) -> bool:
        if not self.complete:
            return self._atoms_fixed(atoms)
        reference = self.torsions.get((_chain_key(atoms), improper))
        return all(
            self.torsions.get((_chain_key(tuple(p[a] for a in atoms)), improper)) == reference
            for p in self.perms
        )


def _torsion_line(tag: str, types: list[str], terms) -> str:
    parts = [f"  <{tag}"]
    for position, t in enumerate(types, start=1):
        parts.append(f"type{position}={quoteattr(t)}")
    for index, (n, phase, k) in enumerate(terms, start=1):
        parts.append(f'periodicity{index}="{int(n)}" phase{index}="{_fmt(phase)}" k{index}="{_fmt(k)}"')
    return " ".join(parts) + "/>"


def _torsion_extra(names, atoms, term) -> dict:
    n, phase, k = term
    return {
        "kind": "torsion",
        "atoms": [names[a] for a in atoms],
        "periodicity": int(n),
        "phase": phase,
        "k": k,
    }
