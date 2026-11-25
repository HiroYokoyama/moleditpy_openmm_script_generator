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
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import quoteattr

from .mdff import MdffDocument, MdffMoleculeType

#: Residue names OpenMM already knows bonds for (Topology._standardBonds) or
#: rewrites through pdbNames.xml.  Templates never take these names.
RESERVED_RESIDUE_NAMES = frozenset(
    """
    ALA ARG ASN ASP ASH CYS CYX CYM GLN GLU GLH GLY HIS HID HIE HIP HSD HSE HSP ILE LEU LYS LYN
    MET PHE PRO SER THR TRP TYR VAL ACE NME NH2 NMA SEC PYL DA DC DG DT DU DI A C G U I
    RA RC RG RU DA5 DA3 DC5 DC3 DG5 DG3 DT5 DT3 RA5 RA3 RC5 RC3 RG5 RG3 RU5 RU3
    HOH WAT SOL H2O TIP TIP3 TIP4 TIP5 T3P T4P T5P SPC OPC
    """.split()
)


@dataclass
class TemplateInfo:
    """How one molecule type appears in the PDB and in the XML."""

    type_name: str
    template: str
    resname: str
    atom_names: List[str]
    is_water: bool


@dataclass
class FfxmlResult:
    xml: str
    templates: Dict[str, TemplateInfo]  # keyed by molecule type name
    extras: Dict[str, List[dict]] = field(default_factory=dict)  # keyed by resname
    warnings: List[str] = field(default_factory=list)

    @property
    def n_extra_terms(self) -> int:
        return sum(len(terms) for terms in self.extras.values())

    def bond_definitions(self, doc: MdffDocument) -> Dict[str, List[Tuple[str, str]]]:
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


def assign_residue_names(doc: MdffDocument, taken=()) -> Dict[str, str]:
    """Unique PDB residue names (at most 4 characters) for every molecule type.

    The first water type becomes ``HOH`` so OpenMM's rigid-water handling
    recognises it; other names avoid residues OpenMM treats as standard.
    """
    used = {name.upper() for name in taken}
    result: Dict[str, str] = {}
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
        while candidate in used or (
            candidate in RESERVED_RESIDUE_NAMES and mol_type.n_atoms > 1
        ):
            suffix = str(counter)
            candidate = (base[: 4 - len(suffix)] or "M") + suffix
            counter += 1
        used.add(candidate)
        result[mol_type.name] = candidate
    return result


def assign_atom_names(mol_type: MdffMoleculeType, water: bool = False) -> List[str]:
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
    names: List[str] = []
    used = set()
    counters: Dict[str, int] = {}
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


def _chain_key(atoms: Tuple[int, ...]) -> Tuple[int, ...]:
    return min(atoms, atoms[::-1])


def _star_centre(atoms: Tuple[int, int, int, int], bonded) -> Optional[int]:
    """Position (0-3) of the atom bonded to the other three, else None."""
    for position, centre in enumerate(atoms):
        others = [a for a in atoms if a != centre]
        if len(set(atoms)) == 4 and all(frozenset((centre, o)) in bonded for o in others):
            return position
    return None


def build_ffxml(doc: MdffDocument, residue_names: Optional[Dict[str, str]] = None) -> FfxmlResult:
    """Return the ForceField XML, template metadata and extra terms."""
    residue_names = residue_names or assign_residue_names(doc)
    templates: Dict[str, TemplateInfo] = {}
    extras: Dict[str, List[dict]] = {}
    warnings: List[str] = list(doc.warnings)

    atom_types: List[str] = []
    residues: List[str] = []
    bond_lines: List[str] = []
    angle_lines: List[str] = []
    proper_lines: List[str] = []
    improper_default: List[str] = []
    improper_charmm: List[str] = []
    nonbonded_lines: List[str] = []

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

        for t, atom in zip(types, mol_type.atoms):
            atom_types.append(
                f'  <Type name={quoteattr(t)} class={quoteattr(t)} '
                f'element={quoteattr(atom.element)} mass="{_fmt(atom.mass)}"/>'
            )
            nonbonded_lines.append(
                f'  <Atom type={quoteattr(t)} sigma="{_fmt(atom.sigma)}" '
                f'epsilon="{_fmt(atom.epsilon)}"/>'
            )

        residue = [f"  <Residue name={quoteattr(template)}>"]
        for name, t, atom in zip(names, types, mol_type.atoms):
            residue.append(
                f"   <Atom name={quoteattr(name)} type={quoteattr(t)} "
                f'charge="{_fmt(atom.charge)}"/>'
            )
        bonded = set()
        for i, j, r0, k in mol_type.bonds:
            key = frozenset((i, j))
            if i == j:
                warnings.append(f"{mol_type.name}: bond {i}-{j} joins an atom to itself; skipped.")
                continue
            if key in bonded:
                extra.append(
                    {"kind": "bond", "atoms": [names[i], names[j]], "length": r0, "k": k}
                )
                continue
            bonded.add(key)
            residue.append(
                f"   <Bond atomName1={quoteattr(names[i])} atomName2={quoteattr(names[j])}/>"
            )
            bond_lines.append(
                f'  <Bond type1={quoteattr(types[i])} type2={quoteattr(types[j])} '
                f'length="{_fmt(r0)}" k="{_fmt(k)}"/>'
            )
        residue.append("  </Residue>")
        residues.extend(residue)

        seen_angles = set()
        for i, j, k_atom, theta0, k in mol_type.angles:
            key = _chain_key((i, j, k_atom))
            chain = frozenset((i, j)) in bonded and frozenset((j, k_atom)) in bonded
            if not chain or key in seen_angles:
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
                f'  <Angle type1={quoteattr(types[i])} type2={quoteattr(types[j])} '
                f'type3={quoteattr(types[k_atom])} angle="{_fmt(math.radians(theta0))}" '
                f'k="{_fmt(k)}"/>'
            )

        propers: Dict[Tuple[int, ...], List[Tuple[int, float, float]]] = {}
        impropers: Dict[Tuple[int, frozenset], Dict[Tuple[int, ...], List]] = {}
        for i, j, k_atom, l_atom, n, phase, k, improper in mol_type.torsions:
            atoms = (i, j, k_atom, l_atom)
            term = (n, math.radians(phase), k)
            if k == 0.0:
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
                    group = impropers.setdefault(
                        (centre, frozenset(a for a in atoms if a != centre)), {}
                    )
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
            (atoms, terms), = variants.items()
            position = atoms.index(centre)
            if position in (1, 3):  # use the reversed dihedral, which is equal
                atoms = atoms[::-1]
                position = 3 - position
            if position == 2:  # (a, b, c, d) -> default ordering type1=c
                ordered = [atoms[2], atoms[0], atoms[1], atoms[3]]
                improper_default.append(
                    _torsion_line("Improper", [types[a] for a in ordered], terms)
                )
            else:  # centre first -> charmm ordering keeps (c, a, b, d)
                improper_charmm.append(
                    _torsion_line("Improper", [types[a] for a in atoms], terms)
                )

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
        "  <Reference>Converted from moleditpy-mdff version 1 by the MoleditPy OpenMM "
        "Script Generator.</Reference>",
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
    lines += [
        f' <NonbondedForce coulomb14scale="{_fmt(doc.fudge_qq)}" lj14scale="{_fmt(doc.fudge_lj)}">',
        '  <UseAttributeFromResidue name="charge"/>',
        *nonbonded_lines,
        " </NonbondedForce>",
        "</ForceField>",
        "",
    ]
    return FfxmlResult("\n".join(lines), templates, extras, warnings)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _torsion_line(tag: str, types: List[str], terms) -> str:
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
