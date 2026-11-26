"""PDB writer for the residue model (CRYST1, ATOM/HETATM, TER, CONECT).

Atom serial numbers above 99999 switch to hexadecimal as OpenMM's own writer
does.  CONECT records are written for hetero residues only while every serial
fits in five decimal digits; the run script also loads bond definitions for
every non-standard residue, so OpenMM never depends on them.
"""

from __future__ import annotations

from collections.abc import Sequence


def _format_index(index: int, places: int) -> str:
    if index < 10**places:
        return f"{index:{places}d}"
    shifted = (index - 10**places + 10 * 16 ** (places - 1)) % (16**places)
    return f"{shifted:{places}X}"


def _atom_name_field(name: str, element: str) -> str:
    # One-letter elements start in column 14 when the name is shorter than 4.
    if len(name) < 4 and len(element) == 1:
        return f" {name:<3s}"
    return f"{name:<4s}"[:4]


def write_pdb(
    system,
    elements: Sequence[str],
    coords,
    lengths: tuple[float, float, float] | None = None,
    angles: tuple[float, float, float] | None = None,
    bonds: Sequence[tuple[int, int]] = (),
    title: str = "",
    formal_charges: Sequence[int] | None = None,
) -> str:
    """Return PDB text.  ``coords`` are indexed by host atom index (Angstrom)."""
    lines = []
    if title:
        lines.append(f"TITLE     {title[:70]}")
    lines.append("REMARK    Written by the MoleditPy OpenMM Script Generator")
    if lengths is not None:
        a, b, c = lengths
        al, be, ga = angles or (90.0, 90.0, 90.0)
        lines.append(f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} P 1           1")
    serial_of: dict[int, int] = {}
    serial = 0
    previous_chain = None
    previous_kind = None
    for residue in system.residues:
        record = "HETATM" if residue.hetero else "ATOM  "
        if previous_chain is not None and (
            residue.chain != previous_chain or (previous_kind == "protein") != (residue.kind == "protein")
        ):
            lines.append("TER")
        resseq = f"{residue.resseq % 10000:4d}"
        for atom, name in zip(residue.atoms, residue.names):
            serial += 1
            serial_of[atom] = serial
            element = elements[atom]
            x, y, z = (float(v) for v in coords[atom])
            charge = ""
            if formal_charges is not None and formal_charges[atom]:
                q = int(formal_charges[atom])
                charge = f"{abs(q)}{'+' if q > 0 else '-'}"
            lines.append(
                f"{record}{_format_index(serial, 5)} {_atom_name_field(name, element)} "
                f"{residue.resname:<4s}{residue.chain[:1]}{resseq}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element.upper():>2s}{charge:2s}"
            )
        previous_chain = residue.chain
        previous_kind = residue.kind
    lines.append("TER")

    if serial <= 99999 and bonds:
        hetero_atoms = {a for r in system.residues if r.hetero for a in r.atoms}
        partners: dict[int, list] = {}
        for i, j in bonds:
            if i in hetero_atoms or j in hetero_atoms:
                partners.setdefault(serial_of[i], []).append(serial_of[j])
                partners.setdefault(serial_of[j], []).append(serial_of[i])
        for s in sorted(partners):
            others = sorted(partners[s])
            for start in range(0, len(others), 4):
                chunk = "".join(f"{o:5d}" for o in others[start : start + 4])
                lines.append(f"CONECT{s:5d}{chunk}")
    lines.append("END")
    return "\n".join(lines) + "\n"
