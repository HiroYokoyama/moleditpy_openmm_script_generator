# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

`moleditpy_openmm_script_generator` is a MoleditPy plugin. It writes three things:
a self-contained OpenMM run script, a PDB with `CRYST1`, and an OpenMM ForceField
XML converted from a moleditpy-mdff JSON file.

| Module | Role |
|---|---|
| `__init__.py` | entry point; persistence of settings and `moleditpy_cell` |
| `main_dialog.py` | dialog and previews |
| `generator.py` | pipeline and checks |
| `script_writer.py` | the run script: `CONFIG` dict plus fixed body |
| `ffxml.py` / `symmetry.py` | mdff to XML conversion |
| `system_model.py` | residues |
| `cell.py` | boxes |
| `pdb_writer.py` | PDB output |
| `options.py` | settings and presets |

## Shared modules: do NOT edit them here

`cell_model.py`, `elements.py`, `cell_preview.py` and `structure_panel.py` are
materialized from the `_periodic_shared` submodule
([moleditpy-periodic-shared](https://github.com/HiroYokoyama/moleditpy-periodic-shared)).
They are gitignored and excluded from Ruff:

```bash
git submodule update --init
python scripts/materialize_shared.py
```

## Testing

```bash
python -m pytest tests/ -v
```

- Headless.
- The OpenMM tier in `tests/test_generator_script.py` runs the generated scripts.
  It needs `openmm` to be importable, or `OPENMM_PYTHON` set to an interpreter
  that has it. It skips otherwise.
- CI installs OpenMM on the Ubuntu jobs so the tier runs there.

## Things that are easy to get wrong

- OpenMM matches residue templates on the bond graph only, so symmetry-equivalent
  atoms can be permuted. Terms that are not invariant under the graph's
  automorphisms must go into `extras`, which the script adds by atom name.
  Do not put them in the XML.
- Impropers: OpenMM's default `<Improper>` ordering puts the centre (type1) third.
  `ordering="charmm"` puts it first. OpenMM applies only one improper per centre
  and neighbour set.
- All NonbondedForce generators must agree on the 1-4 scales:
  - amber14: 5/6 and 0.5
  - charmm36: 1.0 and 1.0
- MoleditPy drops the `moleditpy_cell` property on undo and on project save. The
  plugin remembers it and stores it with its project state. Load handlers run
  before the molecule is restored, so it is put back with
  `QTimer.singleShot(0, ...)`.
- No module-level `run()`: the host would add a duplicate Plugins-menu entry.
- OpenMM is never a dependency. It is listed in `PLUGIN_OPTIONAL_DEPENDENCIES`
  and only used in a subprocess for the check.
- Generated files are written with `newline="\n"`.

## Copied maintenance scripts

See [scripts/COMMON_TOOLS.md](scripts/COMMON_TOOLS.md).
