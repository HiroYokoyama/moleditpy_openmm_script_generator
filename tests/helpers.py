"""Shared test data."""

import copy
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")


def example(name):
    with open(os.path.join(EXAMPLES, name), encoding="utf-8") as handle:
        return json.load(handle)


def atom(name, element, charge=0.0, mass=1.0, sigma=0.3, epsilon=0.1, type_=None):
    return {
        "name": name,
        "element": element,
        "type": type_ or element,
        "charge": charge,
        "mass": mass,
        "sigma": sigma,
        "epsilon": epsilon,
    }


def document(types, system=None, **top):
    doc = {
        "format": "moleditpy-mdff",
        "version": 1,
        "source": "test",
        "combination_rule": 2,
        "fudge_lj": 0.5,
        "fudge_qq": 0.8333,
        "molecule_types": copy.deepcopy(types),
        "system": system or [],
    }
    doc.update(top)
    return doc


def formaldehyde(impropers=(), charges=(0.4, -0.5, 0.05, 0.05)):
    """C0 bonded to O1, H2, H3."""
    atoms = [
        atom("C", "C", charges[0], 12.011),
        atom("O", "O", charges[1], 15.999),
        atom("H1", "H", charges[2], 1.008),
        atom("H2", "H", charges[3], 1.008),
    ]
    torsions = [
        {"i": i, "j": j, "k_atom": k, "l": l, "periodicity": 2, "phase": 180.0, "k": 4.0, "improper": True}
        for i, j, k, l in impropers
    ]
    return {
        "name": "FORM",
        "atoms": atoms,
        "bonds": [
            {"i": 0, "j": 1, "r0": 0.12, "k": 1000.0},
            {"i": 0, "j": 2, "r0": 0.11, "k": 1000.0},
            {"i": 0, "j": 3, "r0": 0.11, "k": 1000.0},
        ],
        "angles": [
            {"i": 1, "j": 0, "k_atom": 2, "theta0": 120.0, "k": 400.0},
            {"i": 1, "j": 0, "k_atom": 3, "theta0": 120.0, "k": 400.0},
            {"i": 2, "j": 0, "k_atom": 3, "theta0": 120.0, "k": 300.0},
        ],
        "torsions": torsions,
    }
