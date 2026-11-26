"""Generate the self-contained OpenMM run script.

The script body is fixed text; everything that depends on the system and the
dialog goes into one ``CONFIG`` dictionary at the top, which users may edit.
It imports only OpenMM (plus openmmforcefields / the OpenFF toolkit when a
small-molecule generator is selected) and explains anything that is missing.
"""

from __future__ import annotations

import datetime
import pprint

from .options import PERIODIC_METHODS, VIRTUAL_SITE_WATERS, forcefield_files


def build_config(
    settings: dict,
    pdb_name: str,
    ffxml_name: str | None = None,
    templates: dict[str, str] | None = None,
    extra_terms: dict[str, list[dict]] | None = None,
    bond_definitions: dict[str, list] | None = None,
    sdf_files: dict[str, str] | None = None,
    periodic: bool = False,
) -> dict:
    """The ``CONFIG`` dictionary written into the script."""
    base = settings.get("base_name") or "system"
    method = settings["nonbonded_method"]
    if not periodic and method in PERIODIC_METHODS:
        method = "NoCutoff"
    generator = settings.get("small_molecule_generator", "none")
    return {
        "pdb": pdb_name,
        "forcefield_files": forcefield_files(settings),
        "mdff_xml": ffxml_name or "",
        "templates": dict(templates or {}),
        "bond_definitions": {k: [list(p) for p in v] for k, v in (bond_definitions or {}).items()},
        "extra_terms": dict(extra_terms or {}),
        "small_molecules": {
            "generator": generator if sdf_files else "none",
            "sdf_files": dict(sdf_files or {}),
            "smirnoff_version": settings.get("smirnoff_version", "openff-2.2.0"),
            "gaff_version": settings.get("gaff_version", "gaff-2.11"),
            "cache": f"{base}_templates.json",
        },
        "add_hydrogens": bool(settings.get("add_hydrogens")),
        "ph": float(settings.get("ph", 7.0)),
        "add_extra_particles": settings.get("water_model") in VIRTUAL_SITE_WATERS,
        "nonbonded_method": method,
        "cutoff_nm": float(settings["cutoff"]),
        "ewald_error_tolerance": float(settings["ewald_tolerance"]),
        "dispersion_correction": bool(settings["dispersion_correction"]),
        "constraints": settings["constraints"],
        "rigid_water": bool(settings["rigid_water"]),
        "hydrogen_mass_amu": float(settings["hydrogen_mass"]),
        "integrator": settings["integrator"],
        "temperature_K": float(settings["temperature"]),
        "friction_per_ps": float(settings["friction"]),
        "timestep_fs": float(settings["timestep"]),
        "barostat": bool(settings["barostat"]) and periodic,
        "pressure_bar": float(settings["pressure"]),
        "barostat_frequency": int(settings["barostat_frequency"]),
        "minimize": bool(settings["minimize"]),
        "minimize_tolerance": float(settings["minimize_tolerance"]),
        "minimize_max_iterations": int(settings["minimize_max_iterations"]),
        "steps": int(settings["steps"]),
        "report_interval": int(settings["report_interval"]),
        "trajectory_interval": int(settings["trajectory_interval"]),
        "reporters": {
            "dcd": f"{base}_traj.dcd" if settings["dcd"] else "",
            "xyz": f"{base}_traj.xyz" if settings["xyz"] else "",
            "pdb": f"{base}_traj.pdb" if settings["pdb_trajectory"] else "",
            "state_csv": f"{base}_state.csv" if settings["state_csv"] else "",
            "checkpoint": f"{base}.chk" if settings["checkpoint"] else "",
        },
        "checkpoint_interval": int(settings["checkpoint_interval"]),
        "minimized_pdb": f"{base}_minimized.pdb",
        "final_pdb": f"{base}_final.pdb",
        "final_state": f"{base}_final_state.xml",
        "platform": settings["platform"],
        "precision": settings["precision"],
    }


def build_script(config: dict, version: str, summary: str = "", script_name: str = "run_openmm.py") -> str:
    stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    header = HEADER.format(
        version=version,
        stamp=stamp,
        summary=summary.replace('"""', "'''") or "-",
        script=script_name,
    )
    config_text = "CONFIG = " + pprint.pformat(config, indent=1, width=96, sort_dicts=False) + "\n"
    return header + config_text + BODY


HEADER = '''#!/usr/bin/env python3
"""OpenMM simulation script.

Written by the MoleditPy OpenMM Script Generator {version} on {stamp}.

System: {summary}

Run:    python {script}            (minimise and run the simulation)
        python {script} --check    (build the system, print the energy, stop)
        python {script} --help     (other options)

Needs OpenMM:  conda install -c conda-forge openmm   (or: pip install openmm)
Edit CONFIG below to change the run; paths are relative to this file.
"""

import argparse
import io
import math
import os
import sys
import time
from xml.sax.saxutils import quoteattr

'''

BODY = r'''
HERE = os.path.dirname(os.path.abspath(__file__))

ADVICE = (
    "  Small molecules: parameterise them with the MoleditPy Force Field Builder plugin and\n"
    "  load its .mdff.json in the OpenMM Script Generator, or select an openmmforcefields\n"
    "  generator there (needs: conda install -c conda-forge openmmforcefields openff-toolkit).\n"
    "  Proteins: choose Amber14 or CHARMM36 and make sure residue and atom names follow the\n"
    "  PDB conventions; enable 'Add missing hydrogens' if hydrogens are absent."
)


def here(name):
    return name if os.path.isabs(name) else os.path.join(HERE, name)


def fail(message):
    print("ERROR: " + message, file=sys.stderr)
    sys.exit(2)


def import_openmm():
    try:
        import openmm
        from openmm import app, unit
    except ImportError:
        fail(
            "OpenMM is not installed for this Python (%s).\n"
            "  Install it with:  conda install -c conda-forge openmm\n"
            "  or:               pip install openmm" % sys.executable
        )
    return openmm, app, unit


def load_bond_definitions(app):
    """Teach OpenMM the bonds of non-standard residues (independent of CONECT)."""
    definitions = CONFIG["bond_definitions"]
    if not definitions:
        return
    parts = ["<Residues>"]
    for residue, bonds in definitions.items():
        parts.append("<Residue name=%s>" % quoteattr(residue))
        for first, second in bonds:
            parts.append("<Bond from=%s to=%s/>" % (quoteattr(first), quoteattr(second)))
        parts.append("</Residue>")
    parts.append("</Residues>")
    app.Topology.loadBondDefinitions(io.StringIO("".join(parts)))


def add_small_molecule_generator(forcefield, needed):
    settings = CONFIG["small_molecules"]
    names = ", ".join(sorted(needed))
    try:
        from openff.toolkit import Molecule
    except ImportError:
        try:
            from openff.toolkit.topology import Molecule
        except ImportError:
            fail(
                "Residue(s) %s are to be parameterised by openmmforcefields, but the OpenFF "
                "toolkit is not installed.\n%s" % (names, ADVICE)
            )
    try:
        from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
    except ImportError:
        fail(
            "Residue(s) %s are to be parameterised by openmmforcefields, which is not "
            "installed.\n%s" % (names, ADVICE)
        )
    molecules = []
    for residue, sdf in settings["sdf_files"].items():
        if residue in needed:
            molecules.append(Molecule.from_file(here(sdf), allow_undefined_stereo=True))
    cache = here(settings["cache"]) if settings.get("cache") else None
    if settings["generator"] == "gaff":
        generator = GAFFTemplateGenerator(
            molecules=molecules, forcefield=settings["gaff_version"], cache=cache
        )
    else:
        generator = SMIRNOFFTemplateGenerator(
            molecules=molecules, forcefield=settings["smirnoff_version"], cache=cache
        )
    forcefield.registerTemplateGenerator(generator.generator)
    print("Small molecules %s: %s via openmmforcefields" % (names, settings["generator"]))


def build_forcefield(app):
    files = list(CONFIG["forcefield_files"])
    if CONFIG["mdff_xml"]:
        files.append(here(CONFIG["mdff_xml"]))
    if not files:
        fail("No force field files are configured.")
    try:
        forcefield = app.ForceField(*files)
    except Exception as exc:
        fail(
            "Could not load the force field files %s:\n  %s\n"
            "  The built-in Amber14 and CHARMM36 files need OpenMM 7.5 or newer." % (files, exc)
        )
    print("Force field files: " + ", ".join(CONFIG["forcefield_files"] + ([CONFIG["mdff_xml"]] if CONFIG["mdff_xml"] else [])))
    return forcefield


def add_extra_terms(openmm, unit, system, topology):
    """Terms the ForceField XML cannot express, added by residue and atom name."""
    extra = CONFIG["extra_terms"]
    if not extra:
        return 0
    bonds = angles = torsions = None
    fixes = {}
    count = 0
    for residue in topology.residues():
        terms = extra.get(residue.name)
        if not terms:
            continue
        index = {atom.name: atom.index for atom in residue.atoms()}
        for term in terms:
            try:
                atoms = [index[name] for name in term["atoms"]]
            except KeyError as exc:
                fail("Residue %s %s has no atom %s for an extra term." % (residue.name, residue.id, exc))
            kind = term["kind"]
            if kind == "bond":
                if bonds is None:
                    bonds = openmm.HarmonicBondForce()
                    system.addForce(bonds)
                bonds.addBond(atoms[0], atoms[1], term["length"], term["k"])
            elif kind == "angle":
                if angles is None:
                    angles = openmm.HarmonicAngleForce()
                    system.addForce(angles)
                angles.addAngle(atoms[0], atoms[1], atoms[2], term["theta0"], term["k"])
            elif kind == "torsion":
                if torsions is None:
                    torsions = openmm.PeriodicTorsionForce()
                    system.addForce(torsions)
                torsions.addTorsion(*atoms, term["periodicity"], term["phase"], term["k"])
            elif kind == "constraint":
                system.addConstraint(atoms[0], atoms[1], term["length"])
            elif kind == "nonbonded":
                fixes[atoms[0]] = term
            count += 1
    if fixes:
        fix_nonbonded(openmm, system, topology, fixes)
    return count


def fix_nonbonded(openmm, system, topology, fixes):
    """Put charges / LJ back on the named atoms and redo their 1-4 pairs.

    OpenMM maps chemically equivalent atoms to a template in any order, so
    parameters that differ between equivalent atoms are set again by name.
    """
    force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    for index, term in fixes.items():
        force.setParticleParameters(index, term["charge"], term["sigma"], term["epsilon"])
    neighbours = {}
    for bond in topology.bonds():
        a, b = bond[0].index, bond[1].index
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    for k in range(force.getNumExceptions()):
        p1, p2, _qq, _sigma, _eps = force.getExceptionParameters(k)
        if p1 not in fixes and p2 not in fixes:
            continue
        near = set(neighbours.get(p1, ()))
        for n in list(near):
            near |= neighbours.get(n, set())
        if p2 in near:  # 1-2 or 1-3 pair: stays excluded
            continue
        term = fixes.get(p1) or fixes.get(p2)
        q1, s1, e1 = force.getParticleParameters(p1)
        q2, s2, e2 = force.getParticleParameters(p2)
        force.setExceptionParameters(
            k, p1, p2,
            q1 * q2 * term["coulomb14"],
            (s1 + s2) / 2.0,
            (e1 * e2).sqrt() * term["lj14"],
        )


class XYZReporter(object):
    """Extended XYZ trajectory (Angstrom) that the Large System Viewer can open."""

    def __init__(self, filename, interval, topology):
        self._out = open(filename, "w")
        self._interval = interval
        self._symbols = [
            atom.element.symbol if atom.element is not None else "X" for atom in topology.atoms()
        ]

    def describeNextReport(self, simulation):
        steps = self._interval - simulation.currentStep % self._interval
        return (steps, True, False, False, False)

    def report(self, simulation, state):
        from openmm import unit

        positions = state.getPositions(asNumpy=False).value_in_unit(unit.angstrom)
        comment = 'Properties=species:S:1:pos:R:3 step=%d time_ps=%.4f' % (
            simulation.currentStep,
            state.getTime().value_in_unit(unit.picosecond),
        )
        vectors = state.getPeriodicBoxVectors()
        if vectors is not None and simulation.topology.getPeriodicBoxVectors() is not None:
            flat = [v for vec in vectors.value_in_unit(unit.angstrom) for v in vec]
            comment = 'Lattice="%s" ' % " ".join("%.5f" % v for v in flat) + comment
        lines = [str(len(self._symbols)), comment]
        for symbol, (x, y, z) in zip(self._symbols, positions):
            lines.append("%-2s %12.5f %12.5f %12.5f" % (symbol, x, y, z))
        self._out.write("\n".join(lines) + "\n")
        self._out.flush()

    def __del__(self):
        try:
            self._out.close()
        except Exception:
            pass


def make_integrator(openmm, unit):
    timestep = CONFIG["timestep_fs"] * unit.femtoseconds
    if CONFIG["integrator"] == "Verlet":
        return openmm.VerletIntegrator(timestep)
    return openmm.LangevinMiddleIntegrator(
        CONFIG["temperature_K"] * unit.kelvin, CONFIG["friction_per_ps"] / unit.picosecond, timestep
    )


def platform_candidates(openmm, requested):
    names = [requested] if requested and requested != "auto" else ["CUDA", "OpenCL", "CPU", "Reference"]
    result = []
    for name in names:
        try:
            platform = openmm.Platform.getPlatformByName(name)
        except Exception:
            if requested and requested != "auto":
                available = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
                fail("Platform %s is not available. Available: %s" % (name, ", ".join(available)))
            continue
        properties = {}
        if name in ("CUDA", "OpenCL"):
            properties["Precision"] = CONFIG["precision"]
        result.append((platform, properties))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the OpenMM simulation set up in CONFIG.")
    parser.add_argument("--check", action="store_true", help="build the system, report the energy and stop")
    parser.add_argument("--platform", help="override the platform (CUDA, OpenCL, CPU, Reference)")
    parser.add_argument("--steps", type=int, help="override the number of MD steps")
    parser.add_argument("--no-minimize", action="store_true", help="skip energy minimisation")
    args = parser.parse_args(argv)

    openmm, app, unit = import_openmm()
    print("OpenMM %s, Python %s" % (openmm.__version__, sys.version.split()[0]))

    load_bond_definitions(app)
    try:
        pdb = app.PDBFile(here(CONFIG["pdb"]))
    except Exception as exc:
        fail("Could not read %s: %s" % (CONFIG["pdb"], exc))
    forcefield = build_forcefield(app)

    modeller = app.Modeller(pdb.topology, pdb.positions)
    if CONFIG["add_hydrogens"]:
        before = modeller.topology.getNumAtoms()
        modeller.addHydrogens(forcefield, pH=CONFIG["ph"])
        print("Added %d hydrogens at pH %.1f" % (modeller.topology.getNumAtoms() - before, CONFIG["ph"]))

    templates = CONFIG["templates"]
    generated = set(CONFIG["small_molecules"]["sdf_files"])
    if CONFIG["small_molecules"]["generator"] != "none":
        needed = {r.name for r in modeller.topology.residues() if r.name in generated}
        if needed:
            add_small_molecule_generator(forcefield, needed)
    if CONFIG["add_extra_particles"]:
        modeller.addExtraParticles(forcefield)
    topology, positions = modeller.topology, modeller.positions

    # Pinned residues may match built-in templates too (e.g. mdff water and the
    # TIP3P file); OpenMM's own check raises on that ambiguity, so leave them out.
    unpinned = app.Modeller(topology, positions)
    unpinned.delete([r for r in topology.residues() if r.name in templates])
    unmatched = {}
    for residue in forcefield.getUnmatchedResidues(unpinned.topology):
        if residue.name in generated and CONFIG["small_molecules"]["generator"] != "none":
            continue
        unmatched.setdefault(residue.name, []).append(residue.id)
    if unmatched:
        listing = "; ".join(
            "%s (%d residue%s, first %s)" % (name, len(ids), "" if len(ids) == 1 else "s", ids[0])
            for name, ids in sorted(unmatched.items())
        )
        fail("No force field parameters for: %s\n%s" % (listing, ADVICE))

    residue_templates = {r: templates[r.name] for r in topology.residues() if r.name in templates}
    periodic = topology.getPeriodicBoxVectors() is not None
    method = getattr(app, CONFIG["nonbonded_method"])
    if not periodic and CONFIG["nonbonded_method"] in ("PME", "LJPME", "Ewald", "CutoffPeriodic"):
        fail("%s needs a periodic box, but %s has no CRYST1 record." % (CONFIG["nonbonded_method"], CONFIG["pdb"]))
    constraints = {"None": None, "HBonds": app.HBonds, "AllBonds": app.AllBonds, "HAngles": app.HAngles}[
        CONFIG["constraints"]
    ]
    kwargs = dict(
        nonbondedMethod=method,
        constraints=constraints,
        rigidWater=CONFIG["rigid_water"],
        residueTemplates=residue_templates,
    )
    if CONFIG["nonbonded_method"] != "NoCutoff":
        kwargs["nonbondedCutoff"] = CONFIG["cutoff_nm"] * unit.nanometer
    if CONFIG["nonbonded_method"] in ("PME", "LJPME", "Ewald"):
        kwargs["ewaldErrorTolerance"] = CONFIG["ewald_error_tolerance"]
    if CONFIG["hydrogen_mass_amu"] > 0:
        kwargs["hydrogenMass"] = CONFIG["hydrogen_mass_amu"] * unit.amu
    try:
        system = forcefield.createSystem(topology, **kwargs)
    except Exception as exc:
        fail("OpenMM could not build the system:\n  %s\n%s" % (exc, ADVICE))

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce) and periodic:
            force.setUseDispersionCorrection(CONFIG["dispersion_correction"])
    n_extra = add_extra_terms(openmm, unit, system, topology)
    if CONFIG["barostat"]:
        system.addForce(
            openmm.MonteCarloBarostat(
                CONFIG["pressure_bar"] * unit.bar, CONFIG["temperature_K"] * unit.kelvin, CONFIG["barostat_frequency"]
            )
        )

    simulation = None
    errors = []
    for platform, properties in platform_candidates(openmm, args.platform or CONFIG["platform"]):
        try:
            simulation = app.Simulation(topology, system, make_integrator(openmm, unit), platform, properties)
            simulation.context.setPositions(positions)
            break
        except Exception as exc:
            errors.append("%s: %s" % (platform.getName(), exc))
            simulation = None
    if simulation is None:
        fail("No platform could run the system:\n  " + "\n  ".join(errors))

    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print("Atoms: %d   residues: %d   constraints: %d   extra terms: %d"
          % (system.getNumParticles(), topology.getNumResidues(), system.getNumConstraints(), n_extra))
    print("Periodic box: %s" % (topology.getPeriodicBoxVectors() if periodic else "none"))
    print("Forces: " + ", ".join(type(f).__name__ for f in system.getForces()))
    print("Platform: %s" % simulation.context.getPlatform().getName())
    print("Initial potential energy: %.3f kJ/mol" % energy)
    if not math.isfinite(energy):
        print("WARNING: the energy is not finite; check for overlapping atoms.")
    if args.check:
        print("CHECK OK: the system was created successfully.")
        return 0

    if CONFIG["minimize"] and not args.no_minimize:
        print("Minimising ...")
        simulation.minimizeEnergy(
            tolerance=CONFIG["minimize_tolerance"] * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=CONFIG["minimize_max_iterations"],
        )
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        print("Minimised potential energy: %.3f kJ/mol"
              % state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        with open(here(CONFIG["minimized_pdb"]), "w") as handle:
            app.PDBFile.writeFile(topology, state.getPositions(), handle)

    steps = args.steps if args.steps is not None else CONFIG["steps"]
    simulation.context.setVelocitiesToTemperature(CONFIG["temperature_K"] * unit.kelvin)
    reporters = CONFIG["reporters"]
    interval = max(1, CONFIG["trajectory_interval"])
    if reporters["dcd"]:
        simulation.reporters.append(app.DCDReporter(here(reporters["dcd"]), interval))
    if reporters["xyz"]:
        simulation.reporters.append(XYZReporter(here(reporters["xyz"]), interval, topology))
    if reporters["pdb"]:
        simulation.reporters.append(app.PDBReporter(here(reporters["pdb"]), interval))
    report = max(1, CONFIG["report_interval"])
    fields = dict(step=True, time=True, potentialEnergy=True, kineticEnergy=True,
                  totalEnergy=True, temperature=True, speed=True)
    if periodic:
        fields.update(volume=True, density=True)
    if reporters["state_csv"]:
        simulation.reporters.append(app.StateDataReporter(here(reporters["state_csv"]), report, separator=",", **fields))
    simulation.reporters.append(
        app.StateDataReporter(sys.stdout, report, step=True, temperature=True, speed=True,
                              progress=True, remainingTime=True, totalSteps=steps, separator="  ")
    )
    if reporters["checkpoint"]:
        simulation.reporters.append(app.CheckpointReporter(here(reporters["checkpoint"]), max(1, CONFIG["checkpoint_interval"])))

    print("Running %d steps of %.2f fs (%.3f ns) ..." % (steps, CONFIG["timestep_fs"], steps * CONFIG["timestep_fs"] * 1e-6))
    started = time.time()
    simulation.step(steps)
    print("Finished in %.1f s" % (time.time() - started))

    state = simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=periodic)
    with open(here(CONFIG["final_pdb"]), "w") as handle:
        app.PDBFile.writeFile(topology, state.getPositions(), handle)
    simulation.saveState(here(CONFIG["final_state"]))
    if reporters["checkpoint"]:
        simulation.saveCheckpoint(here(reporters["checkpoint"]))
    print("Wrote %s and %s" % (CONFIG["final_pdb"], CONFIG["final_state"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
