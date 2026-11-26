"""Extract :class:`MoleculeData` from an RDKit molecule (the host's 3D molecule)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .generator import MoleculeData
from .system_model import ResidueInfo


def _symbol(atom) -> str:
    symbol = atom.GetSymbol()
    if symbol == "*" and atom.HasProp("custom_symbol"):
        return atom.GetProp("custom_symbol")
    return symbol


def molecule_data(mol, title: str = "") -> MoleculeData:
    from rdkit import Chem

    n = mol.GetNumAtoms()
    if mol.GetNumConformers() == 0:
        raise ValueError("The molecule has no 3D coordinates; generate 3D coordinates first.")
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n)], dtype=float).reshape(-1, 3)
    elements = [_symbol(mol.GetAtomWithIdx(i)) for i in range(n)]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    charges = [mol.GetAtomWithIdx(i).GetFormalCharge() for i in range(n)]

    infos: list[ResidueInfo | None] = []
    any_info = False
    for i in range(n):
        info = mol.GetAtomWithIdx(i).GetPDBResidueInfo()
        if info is None or not info.GetResidueName().strip():
            infos.append(None)
            continue
        any_info = True
        infos.append(
            ResidueInfo(
                resname=info.GetResidueName().strip(),
                name=info.GetName().strip(),
                resnum=int(info.GetResidueNumber()),
                chain=info.GetChainId().strip(),
                icode=info.GetInsertionCode().strip(),
                hetero=bool(info.GetIsHeteroAtom()),
            )
        )

    labels: list[str | None] = [None] * n
    try:
        for frag in Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False):
            if len(frag) > 200:  # polymers: canonical SMILES is slow and not needed
                continue
            try:
                smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(frag), canonical=True)
            except Exception:  # noqa: BLE001
                smiles = None
            for a in frag:
                labels[a] = smiles
    except Exception:  # noqa: BLE001, S110 - pragma: no cover - RDKit guard
        pass

    def sdf_for_atoms(atoms: Sequence[int]) -> str:
        atoms = list(atoms)
        keep = set(atoms)
        editable = Chem.RWMol(mol)
        for idx in sorted(range(n), reverse=True):
            if idx not in keep:
                editable.RemoveAtom(idx)
        fragment = editable.GetMol()
        for atom in fragment.GetAtoms():
            atom.SetNoImplicit(True)
        Chem.SanitizeMol(fragment)
        Chem.AssignStereochemistryFrom3D(fragment)
        return Chem.MolToMolBlock(fragment) + "$$$$\n"

    return MoleculeData(
        elements=elements,
        coords=coords,
        bonds=bonds,
        residue_info=infos if any_info else None,
        fragment_labels=labels,
        formal_charges=charges,
        title=title,
        sdf_for_atoms=sdf_for_atoms,
    )
