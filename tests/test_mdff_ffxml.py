import xml.etree.ElementTree as ET

import pytest
from helpers import atom, document, example, formaldehyde

from openmm_script_generator import ffxml, mdff
from openmm_script_generator.symmetry import automorphisms, equivalence_classes

# -- mdff reader ---------------------------------------------------------------


def test_examples_parse_and_cover_their_system():
    doc = mdff.parse_mdff(example("ethanol_water_nacl.mdff.json"))
    assert [t.name for t in doc.molecule_types] == ["ETOH", "SOL", "NA", "CL"]
    assert doc.n_system_atoms == 23
    assert doc.warnings == []
    assert doc.type_by_name("SOL").is_water()
    assert abs(doc.type_by_name("ETOH").total_charge) < 1e-9


@pytest.mark.parametrize(
    "change, message",
    [
        ({"format": "other"}, "format"),
        ({"version": 2}, "Unsupported"),
        ({"molecule_types": []}, "non-empty"),
    ],
)
def test_top_level_problems_are_rejected(change, message):
    data = example("ethanol.mdff.json")
    data.update(change)
    with pytest.raises(mdff.MdffError, match=message):
        mdff.parse_mdff(data)


def test_bad_indices_and_unknown_types_are_rejected():
    data = example("ethanol.mdff.json")
    data["molecule_types"][0]["bonds"][0]["j"] = 99
    with pytest.raises(mdff.MdffError, match="outside"):
        mdff.parse_mdff(data)
    data = example("ethanol.mdff.json")
    data["system"][0]["type"] = "NOPE"
    with pytest.raises(mdff.MdffError, match="unknown molecule type"):
        mdff.parse_mdff(data)


def test_warnings_for_things_openmm_cannot_reproduce():
    data = example("ethanol.mdff.json")
    data["combination_rule"] = 3
    data["molecule_types"][0]["exclusions_nrexcl"] = 2
    data["molecule_types"][0]["atoms"][0]["charge"] += 0.1
    text = " ".join(mdff.parse_mdff(data).warnings)
    assert "Lorentz-Berthelot" in text and "nrexcl = 2" in text and "not an integer" in text


def test_match_system_maps_atoms_and_reports_mismatches():
    doc = mdff.parse_mdff(example("ethanol_water_nacl.mdff.json"))
    elements = ["C", "C", "O"] + ["H"] * 6 + ["O", "H", "H"] * 4 + ["Na", "Cl"]
    assignment, problems = mdff.match_system(doc, elements)
    assert problems == []
    assert assignment[9] == ("SOL", 0, 0) and assignment[20] == ("SOL", 3, 2)
    assert assignment[22] == ("CL", 0, 0)
    wrong = list(elements)
    wrong[21] = "K"
    assignment, problems = mdff.match_system(doc, wrong)
    assert all(a is None for a in assignment) and "Atom 22 is K" in problems[0]
    _assignment, problems = mdff.match_system(doc, elements[:10])
    assert "only 10 atoms" in problems[0]


# -- XML conversion ------------------------------------------------------------


def _xml(doc_dict):
    doc = mdff.parse_mdff(doc_dict)
    return doc, ffxml.build_ffxml(doc)


def test_xml_is_well_formed_with_one_type_per_atom():
    doc, result = _xml(example("ethanol_water_nacl.mdff.json"))
    root = ET.fromstring(result.xml)
    types = root.find("AtomTypes").findall("Type")
    assert len(types) == 9 + 3 + 1 + 1
    residues = {r.get("name"): r for r in root.find("Residues")}
    assert set(residues) == {"MDFF_ETOH", "MDFF_SOL", "MDFF_NA", "MDFF_CL"}
    assert len(residues["MDFF_ETOH"].findall("Bond")) == 8
    nb = root.find("NonbondedForce")
    assert float(nb.get("coulomb14scale")) == pytest.approx(5 / 6, abs=1e-12)  # 0.8333 snapped
    assert nb.get("lj14scale") == "0.5"
    angle = root.find("HarmonicAngleForce").find("Angle")
    assert float(angle.get("angle")) < 3.2  # radians
    # propers: one line per bonded chain, both periodic terms merged where present
    assert len(root.find("PeriodicTorsionForce").findall("Proper")) == 12
    info = result.templates["SOL"]
    assert info.resname == "HOH" and info.atom_names == ["O", "H1", "H2"] and info.is_water
    assert result.templates["ETOH"].atom_names[:3] == ["C1", "C2", "O1"]
    defs = result.bond_definitions(doc)
    assert "HOH" not in defs and len(defs["ETOH"]) == 8


def test_residue_names_avoid_standard_residues_and_are_unique():
    types = [formaldehyde()]
    for name in ("ALA", "ALANINE_LONG_NAME", "ALAN"):
        t = formaldehyde()
        t["name"] = name
        types.append(t)
    doc = mdff.parse_mdff(document(types))
    names = ffxml.assign_residue_names(doc)
    assert names["ALA"] != "ALA"
    assert len(set(names.values())) == len(names)
    assert all(len(n) <= 4 for n in names.values())


@pytest.mark.parametrize(
    "improper, ordering, expected",
    [
        ((0, 1, 2, 3), "charmm", ["0", "1", "2", "3"]),  # centre first
        ((1, 0, 2, 3), "default", ["0", "3", "2", "1"]),  # centre second -> reversed
        ((1, 2, 0, 3), "default", ["0", "1", "2", "3"]),  # centre third
        ((1, 2, 3, 0), "charmm", ["0", "3", "2", "1"]),  # centre fourth -> reversed
    ],
)
def test_improper_centre_position_maps_to_an_exact_ordering(improper, ordering, expected):
    t = formaldehyde([improper], charges=(0.4, -0.5, 0.05, 0.05))
    # Formyl fluoride has no graph symmetry, so the improper stays in the XML.
    t["atoms"][3].update(name="F", element="F", type="F", mass=18.998)
    _doc, result = _xml(document([t]))
    root = ET.fromstring(result.xml)
    forces = root.findall("PeriodicTorsionForce")
    chosen = [f for f in forces if (f.get("ordering") or "default") == ordering]
    impropers = [i for f in chosen for i in f.findall("Improper")]
    assert len(impropers) == 1
    got = [impropers[0].get(f"type{n}").split("-")[-1] for n in range(1, 5)]
    assert got == expected


def test_several_impropers_on_one_centre_become_extra_terms():
    t = formaldehyde([(1, 2, 0, 3), (2, 3, 0, 1)])
    _doc, result = _xml(document([t]))
    assert "<Improper" not in result.xml
    torsions = [e for e in result.extras["FORM"] if e["kind"] == "torsion"]
    assert len(torsions) == 2 and torsions[0]["atoms"] == ["O", "H1", "C", "H2"]


def test_terms_not_invariant_under_symmetry_become_extras():
    # One improper with the H atoms in a fixed order is not invariant under the
    # H1/H2 swap, so OpenMM might apply it to the wrong permutation.
    t = formaldehyde([(1, 2, 0, 3)])
    t["torsions"][0]["phase"] = 170.0
    _doc, result = _xml(document([t]))
    kinds = [e["kind"] for e in result.extras["FORM"]]
    assert kinds == ["torsion"]
    # Different angle force constants on equivalent H angles are moved too.
    t = formaldehyde()
    t["angles"][1]["k"] = 999.0
    _doc, result = _xml(document([t]))
    assert sorted(e["kind"] for e in result.extras["FORM"]) == ["angle", "angle"]


def test_asymmetric_charges_are_reapplied_by_name():
    t = formaldehyde(charges=(0.4, -0.5, 0.02, 0.08))
    _doc, result = _xml(document([t]))
    fixes = [e for e in result.extras["FORM"] if e["kind"] == "nonbonded"]
    assert [f["atoms"][0] for f in fixes] == ["H1", "H2"]
    assert fixes[0]["charge"] == pytest.approx(0.02)
    assert fixes[0]["coulomb14"] == pytest.approx(5 / 6)


def test_non_chain_terms_and_unbonded_constraints_are_extras():
    t = formaldehyde()
    t["angles"].append({"i": 1, "j": 2, "k_atom": 3, "theta0": 90.0, "k": 10.0})
    t["torsions"].append(
        {"i": 1, "j": 2, "k_atom": 3, "l": 0, "periodicity": 1, "phase": 0.0, "k": 1.0, "improper": False}
    )
    t["constraints"] = [{"i": 2, "j": 3, "r0": 0.18}]
    _doc, result = _xml(document([t]))
    kinds = sorted(e["kind"] for e in result.extras["FORM"])
    assert kinds == ["angle", "constraint", "torsion"]


def test_scale_conflict_only_against_different_families():
    doc = mdff.parse_mdff(example("ethanol.mdff.json"))
    assert ffxml.scale_conflict(doc, "amber14", ["amber14-all.xml"]) == ""
    assert "cannot combine" in ffxml.scale_conflict(doc, "charmm36", ["charmm36.xml"])
    assert ffxml.scale_conflict(doc, "charmm36", []) == ""


# -- symmetry -----------------------------------------------------------------


def test_automorphisms_of_ethanol_and_benzene():
    t = mdff.parse_mdff(example("ethanol.mdff.json")).molecule_types[0]
    bonds = [(i, j) for i, j, _r, _k in t.bonds]
    perms, complete = automorphisms(t.elements, bonds)
    assert complete and len(perms) == 3 * 2 * 1 * 2  # CH3 (3!) x CH2 (2!)
    classes = equivalence_classes(t.n_atoms, perms)
    assert len(set(classes[3:6])) == 1 and classes[0] != classes[1]
    ring = [(i, (i + 1) % 6) for i in range(6)] + [(i, i + 6) for i in range(6)]
    perms, complete = automorphisms(["C"] * 6 + ["H"] * 6, ring)
    assert complete and len(perms) == 12
    perms, complete = automorphisms(["C"] * 6 + ["H"] * 6, ring, limit=5)
    assert not complete and len(perms) == 5


def test_incomplete_symmetry_falls_back_to_element_and_degree_groups(monkeypatch):
    monkeypatch.setattr(ffxml, "automorphisms", lambda e, b: ([tuple(range(len(e)))], False))
    t = formaldehyde([(1, 2, 0, 3)], charges=(0.4, -0.5, 0.02, 0.08))
    ion = {"name": "NAI", "atoms": [atom("NA", "Na", 1.0, 22.99)], "bonds": [], "angles": [], "torsions": []}
    _doc, result = _xml(document([t, ion]))
    kinds = sorted(e["kind"] for e in result.extras["FORM"])
    assert kinds == ["angle", "angle", "angle", "nonbonded", "nonbonded", "torsion"]
