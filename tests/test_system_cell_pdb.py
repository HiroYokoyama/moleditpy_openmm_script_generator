import json

import numpy as np
import pytest
from helpers import example

from openmm_script_generator import cell, ffxml, mdff
from openmm_script_generator.pdb_writer import write_pdb
from openmm_script_generator.system_model import ResidueInfo, build_system

# ethanol (9) + water (3) + Na + Cl, ethanol atoms C C O H H H H H H
ELEMENTS = ["C", "C", "O", "H", "H", "H", "H", "H", "H", "O", "H", "H", "Na", "Cl"]
BONDS = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7), (2, 8), (9, 10), (9, 11)]


def test_fragments_become_water_ions_and_named_small_molecules():
    labels = ["CCO"] * 9 + ["O"] * 3 + ["[Na+]", "[Cl-]"]
    system = build_system(ELEMENTS, BONDS, fragment_labels=labels)
    kinds = [(r.resname, r.kind) for r in system.residues]
    assert kinds == [("L01", "other"), ("HOH", "water"), ("NA", "ion"), ("CL", "ion")]
    assert system.residues[1].names == ["O", "H1", "H2"]
    assert system.residues[0].names[:3] == ["C1", "C2", "O1"]
    assert len(system.bond_definitions["L01"]) == 8 and "HOH" not in system.bond_definitions
    assert system.other_residue_names() == ["L01"]
    assert system.n_atoms == 14 and system.order == list(range(14))


def test_identical_fragments_share_a_residue_name_and_different_ones_do_not():
    elements = ["C", "O", "C", "O", "C", "N"]
    bonds = [(0, 1), (2, 3), (4, 5)]
    labels = ["CO", "CO", "CO", "CO", "CN", "CN"]
    system = build_system(elements, bonds, fragment_labels=labels)
    assert [r.resname for r in system.residues] == ["L01", "L01", "L02"]


def test_pdb_residue_info_keeps_protein_and_normalises_water_and_ions():
    elements = ["N", "C", "O", "H", "H", "Na"]
    bonds = [(0, 1), (2, 3), (2, 4)]
    infos = [
        ResidueInfo("ALA", "N", 1, "A"),
        ResidueInfo("ALA", "CA", 1, "A"),
        ResidueInfo("WAT", "OW", 2, "B"),
        ResidueInfo("WAT", "HW1", 2, "B"),
        ResidueInfo("WAT", "HW2", 2, "B"),
        ResidueInfo("SOD", "SOD", 3, "B"),
    ]
    system = build_system(elements, bonds, infos)
    assert [(r.resname, r.kind, r.chain) for r in system.residues] == [
        ("ALA", "protein", "A"),
        ("HOH", "water", "B"),
        ("NA", "ion", "B"),
    ]
    assert system.residues[0].names == ["N", "CA"]
    assert system.residues[1].names == ["O", "H1", "H2"]


def test_mdff_assignment_uses_template_names():
    doc = mdff.parse_mdff(example("ethanol_water_nacl.mdff.json"))
    elements = ["C", "C", "O"] + ["H"] * 6 + ["O", "H", "H"] * 4 + ["Na", "Cl"]
    assignment, problems = mdff.match_system(doc, elements)
    result = ffxml.build_ffxml(doc)
    system = build_system(elements, [], None, assignment, result.templates)
    assert problems == []
    assert [r.template for r in system.residues][:2] == ["MDFF_ETOH", "MDFF_SOL"]
    assert system.counts() == {"mdff": 7}
    assert system.residues[1].resname == "HOH"


def test_links_between_hetero_residues_are_reported():
    system = build_system(["C", "C"], [(0, 1)], [ResidueInfo("LIG", "C1", 1), ResidueInfo("LIG", "C1", 2)])
    assert "join different residues" in system.warnings[0]


# -- cells -----------------------------------------------------------------


def _distances(coords):
    coords = np.asarray(coords)
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)


def test_cell_json_round_trip_and_errors():
    text = cell.cell_to_json(np.eye(3) * 20.0, (1, 2, 3))
    spec = cell.parse_cell_json(text)
    assert spec.lattice[0, 0] == 20.0 and list(spec.origin) == [1, 2, 3] and all(spec.pbc)
    for bad in ("not json", "[]", json.dumps({"lattice": [[1, 2]]})):
        with pytest.raises(ValueError):
            cell.parse_cell_json(bad)


def test_rotated_triclinic_cell_maps_to_standard_orientation():
    rng = np.random.default_rng(1)
    lattice = np.array([[20.0, 0, 0], [4.0, 18.0, 0], [2.0, 3.0, 17.0]])
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rotated = lattice @ rotation.T
    origin = np.array([-5.0, 1.0, 2.0])
    coords = origin + rng.random((6, 3)) @ rotated
    box = cell.box_from_cell(coords, cell.CellSpec(rotated, origin))
    assert box.lengths == pytest.approx(tuple(np.linalg.norm(lattice, axis=1)))
    assert np.allclose(_distances(box.coords), _distances(coords))
    fractional = box.coords @ np.linalg.inv(cell.cell_vectors(box.lengths, box.angles))
    assert np.all(fractional > -1e-9) and np.all(fractional < 1 + 1e-9)
    assert any("rotated" in note for note in box.notes)


def test_left_handed_and_partially_periodic_cells():
    with pytest.raises(ValueError, match="left-handed"):
        cell.box_from_cell(np.zeros((1, 3)), cell.CellSpec(np.diag([10.0, 10.0, -10.0]), np.zeros(3)))
    box = cell.box_from_cell(
        np.zeros((1, 3)), cell.CellSpec(np.eye(3) * 10, np.zeros(3), (True, True, False))
    )
    assert any("fully periodic" in note for note in box.notes)


def test_resolve_box_modes():
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 2.0, 0.0]])
    padded = cell.resolve_box(coords, "padding", padding=5.0)
    assert padded.lengths == pytest.approx((14.0, 12.0, 10.0))
    assert padded.coords.min() == pytest.approx(5.0)
    fixed = cell.resolve_box(coords, "lengths", lengths=(20, 20, 20))
    assert np.allclose(fixed.coords.mean(axis=0), [10, 10, 10])
    assert not cell.resolve_box(coords, "none").periodic
    viewer = cell.CellSpec(np.eye(3) * 15, np.zeros(3), source="the CIF Viewer panel")
    assert cell.resolve_box(coords, "auto", None, viewer).source == "the CIF Viewer panel"
    auto = cell.resolve_box(coords, "auto")
    assert "padded box" in auto.notes[-1]
    with pytest.raises(ValueError):
        cell.resolve_box(coords, "molecule")
    with pytest.raises(ValueError):
        cell.resolve_box(coords, "cif")


def test_cell_from_viewer_reads_the_structure_lattice():
    class Structure:
        lattice = ((10, 0, 0), (0, 11, 0), (0, 0, 12))

    class Widget:
        structure = Structure()

    assert cell.cell_from_viewer(Widget()).lattice[2, 2] == 12
    assert cell.cell_from_viewer(None) is None


# -- PDB writer -----------------------------------------------------------------


def test_pdb_columns_cryst1_ter_and_conect():
    labels = ["CCO"] * 9 + ["O"] * 3 + ["[Na+]", "[Cl-]"]
    system = build_system(ELEMENTS, BONDS, fragment_labels=labels)
    coords = np.arange(42, dtype=float).reshape(14, 3)
    charges = [0] * 12 + [1, -1]
    text = write_pdb(system, ELEMENTS, coords, (30, 31, 32), (90, 90, 120), BONDS, "test", charges)
    lines = text.splitlines()
    cryst = next(line for line in lines if line.startswith("CRYST1"))
    assert cryst[6:15].strip() == "30.000" and cryst[47:54].strip() == "120.00"
    atoms = [line for line in lines if line.startswith("HETATM")]
    assert len(atoms) == 14
    first = atoms[0]
    assert first[12:16] == " C1 " and first[17:21].strip() == "L01" and first[21] == "A"
    assert float(first[30:38]) == 0.0 and float(first[46:54]) == 2.0
    assert first[76:78].strip() == "C"
    assert atoms[12][76:80] == "NA1+" and atoms[13][76:80] == "CL1-"
    conect = [line for line in lines if line.startswith("CONECT")]
    assert conect[0].startswith("CONECT    1    2    4    5")
    assert lines[-1] == "END"


def test_large_serials_switch_to_hex_and_skip_conect():
    n = 100002
    elements = ["Ar"] * n
    system = build_system(elements, [])
    text = write_pdb(system, elements, np.zeros((n, 3)), bonds=[(0, 1)])
    atoms = [line for line in text.splitlines() if line.startswith("HETATM")]
    assert atoms[99998][6:11] == "99999" and atoms[99999][6:11] == "A0000"
    assert "CONECT" not in text
