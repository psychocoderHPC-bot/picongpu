#!/usr/bin/env python3
"""End-to-end tests for the openPMD restart metadata handling.

The test needs a PIConGPU binary built with a deck whose geometry matches the
checkpoint, a checkpoint written by that deck, and h5py.  It is not part of the
regular unit-test build because it requires a GPU and a prepared checkpoint.

Example (rosi_v100, after building a periodic tiny deck):

    env -u LD_PRELOAD python3 openpmd_restart_metadata_test.py \
      --picongpu /path/to/build/picongpu \
      --checkpoint /path/to/checkpoint_000000.h5 \
      --series checkpoint \
      --step 0 \
      --setup /path/to/deck \
      --workdir /tmp/restart-metadata-test \
      --grid 16 16 8 --periodic 1 1 1 \
      --mpirun "mpirun --allow-run-as-root -n 1"

Each scenario copies the checkpoint, mutates only the openPMD metadata and
runs the restart.  The unitSI scenario additionally checks that the loaded
fields are exactly the expected multiple of the unmodified run.
"""

import argparse
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np


FIELDS = ("E", "B")


def checkpoint_step_path(checkpoint, step):
    return f"data/{step}/fields"


def copy_checkpoint(source, destination):
    os.makedirs(destination, exist_ok=True)
    target = os.path.join(destination, "checkpoint_000000.h5")
    shutil.copy2(source, target)
    with open(os.path.join(destination, "checkpoints.txt"), "w") as handle:
        handle.write("0\n")
    return target


def scale_unit_si(path, step, factor):
    with h5py.File(path, "r+") as handle:
        fields = handle[checkpoint_step_path(path, step)]
        for name, mesh in fields.items():
            if name.startswith(("E", "B", "Convolutional PML")):
                for component in mesh.values():
                    component.attrs["unitSI"] = float(component.attrs["unitSI"]) * factor


def break_grid_spacing(path, step):
    with h5py.File(path, "r+") as handle:
        mesh = handle[f"{checkpoint_step_path(path, step)}/E"]
        spacing = list(mesh.attrs["gridSpacing"])
        spacing[len(spacing) // 2] *= 2.0
        mesh.attrs["gridSpacing"] = spacing


def remove_pml_records(path, step):
    with h5py.File(path, "r+") as handle:
        fields = handle[checkpoint_step_path(path, step)]
        for name in list(fields.keys()):
            if name.startswith("Convolutional PML"):
                del fields[name]


def break_id_provider_metadata(path, step):
    with h5py.File(path, "r+") as handle:
        mesh = handle[f"{checkpoint_step_path(path, step)}/picongpu_idProvider"]
        for name in list(mesh.attrs.keys()):
            del mesh.attrs[name]


def break_field_datatype(path, step):
    with h5py.File(path, "r+") as handle:
        mesh = handle[f"{checkpoint_step_path(path, step)}/E"]
        rewrite_dataset(mesh, "x", np.float64)


def rewrite_dataset(group, name, dtype):
    dataset = group[name]
    data = dataset[:]
    attributes = dict(dataset.attrs)
    del group[name]
    replacement = group.create_dataset(name, data=data.astype(dtype))
    for attribute, value in attributes.items():
        replacement.attrs[attribute] = value
    return replacement


def convert_floating_records_to_double(path, step):
    with h5py.File(path, "r+") as handle:
        fields = handle[checkpoint_step_path(path, step)]
        for mesh_name in list(fields.keys()):
            mesh = fields[mesh_name]
            if not isinstance(mesh, h5py.Group):
                continue
            for component_name in list(mesh.keys()):
                component = mesh[component_name]
                if isinstance(component, h5py.Dataset) and component.dtype.kind == "f":
                    rewrite_dataset(mesh, component_name, np.float64)

        particles = handle[f"data/{step}/particles"]
        for species_name in list(particles.keys()):
            species = particles[species_name]
            for record_name in list(species.keys()):
                record = species[record_name]
                if isinstance(record, h5py.Dataset):
                    if record.dtype.kind == "f":
                        rewrite_dataset(species, record_name, np.float64)
                else:
                    for component_name in list(record.keys()):
                        component = record[component_name]
                        if isinstance(component, h5py.Dataset) and component.dtype.kind == "f":
                            rewrite_dataset(record, component_name, np.float64)


def break_particle_datatype(path, step, species, record):
    with h5py.File(path, "r+") as handle:
        group = handle[f"data/{step}/particles/{species}"]
        rewrite_dataset(group, record, np.int32)


def run_restart(args, workdir, checkpoint_dir, periodic, extra_arguments=None):
    run_dir = os.path.join(workdir, "run")
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir)
    command = args.mpirun.split() + [
        args.picongpu,
        "-d",
        "1",
        "1",
        "1",
        "-g",
        *[str(value) for value in args.grid],
        "--periodic",
        *[str(value) for value in periodic],
        "--currentInterpolation",
        "none",
        "--checkpoint.restart",
        "--checkpoint.restart.step",
        str(args.step),
        "--checkpoint.restart.directory",
        os.path.abspath(checkpoint_dir),
        "--checkpoint.restart.file",
        args.series,
        "--openPMD.pluginConfig",
        os.path.join(args.setup, "etc", "openPMD_config_E.toml"),
        "--openPMD.pluginConfig",
        os.path.join(args.setup, "etc", "openPMD_config_B.toml"),
        "-s",
        str(args.steps),
    ]
    if extra_arguments:
        command.extend(extra_arguments)
    environment = dict(os.environ)
    environment.pop("LD_PRELOAD", None)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    result = subprocess.run(
        command,
        cwd=run_dir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result, run_dir


def read_field(path, step, name, component):
    with h5py.File(path, "r") as handle:
        return handle[f"data/{step}/fields/{name}/{component}"][:]


def scenario_unit_si(args, source, workdir, baseline_run):
    checkpoint_dir = os.path.join(workdir, "unit2", "checkpoint")
    path = copy_checkpoint(source, checkpoint_dir)
    scale_unit_si(path, args.step, 2.0)
    result, run_dir = run_restart(args, os.path.join(workdir, "unit2"), checkpoint_dir, args.periodic)
    check(result.returncode == 0, "factor-2 unitSI restart failed:\n" + result.stdout[-2000:])

    output_step = args.step + args.steps
    for name in FIELDS:
        for component in "xyz":
            reference = read_field(
                os.path.join(baseline_run, "openPMD", f"{name}_{output_step:06d}.h5"),
                output_step,
                name,
                component,
            )
            converted = read_field(
                os.path.join(run_dir, "openPMD", f"{name}_{output_step:06d}.h5"),
                output_step,
                name,
                component,
            )
            expected = (reference.astype(np.float64) * 2.0).astype(np.float32)
            check(
                np.array_equal(expected.view(np.uint32), converted.view(np.uint32)),
                f"{name}/{component} was not converted by the factor-2 unitSI",
            )
    print("PASS factor-2 unitSI is converted to internal units")


def scenario_grid_spacing(args, source, workdir):
    checkpoint_dir = os.path.join(workdir, "gridspacing", "checkpoint")
    path = copy_checkpoint(source, checkpoint_dir)
    break_grid_spacing(path, args.step)
    result, _ = run_restart(args, os.path.join(workdir, "gridspacing"), checkpoint_dir, args.periodic)
    check(result.returncode != 0, "mismatched gridSpacing was accepted")
    check(
        "gridSpacing" in result.stdout and "differs from the deck value" in result.stdout,
        "mismatched gridSpacing did not produce the expected error:\n" + result.stdout[-2000:],
    )
    print("PASS mismatched gridSpacing is rejected with a clear error")


def scenario_missing_pml(args, source, workdir):
    checkpoint_dir = os.path.join(workdir, "no-pml", "checkpoint")
    path = copy_checkpoint(source, checkpoint_dir)
    remove_pml_records(path, args.step)
    result, _ = run_restart(args, os.path.join(workdir, "no-pml"), checkpoint_dir, args.periodic)
    check(result.returncode == 0, "checkpoint without PML records failed:\n" + result.stdout[-2000:])
    print("PASS checkpoint without PML records restarts with periodic boundaries")


def scenario_id_provider(args, source, workdir):
    checkpoint_dir = os.path.join(workdir, "idprovider", "checkpoint")
    path = copy_checkpoint(source, checkpoint_dir)
    break_id_provider_metadata(path, args.step)
    result, _ = run_restart(args, os.path.join(workdir, "idprovider"), checkpoint_dir, args.periodic)
    check(result.returncode != 0, "malformed idProvider mesh was accepted")
    check(
        "picongpu_idProvider" in result.stdout,
        "malformed idProvider did not name the record:\n" + result.stdout[-2000:],
    )
    print("PASS malformed idProvider metadata is rejected with a clear error")


def scenario_double_conversion(args, source, workdir):
    checkpoint_dir = os.path.join(workdir, "double", "checkpoint")
    path = copy_checkpoint(source, checkpoint_dir)
    convert_floating_records_to_double(path, args.step)

    baseline_checkpoints = os.path.join(workdir, "double", "baseline-checkpoints")
    baseline_dir = os.path.join(workdir, "double-baseline", "checkpoint")
    copy_checkpoint(source, baseline_dir)
    baseline, baseline_run = run_restart(
        args,
        os.path.join(workdir, "double-baseline"),
        baseline_dir,
        args.periodic,
        ["--checkpoint.period", "1", "--checkpoint.directory", baseline_checkpoints],
    )
    check(baseline.returncode == 0, "float32 baseline restart failed:\n" + baseline.stdout[-2000:])

    converted_checkpoints = os.path.join(workdir, "double", "converted-checkpoints")
    result, run_dir = run_restart(
        args,
        os.path.join(workdir, "double"),
        checkpoint_dir,
        args.periodic,
        ["--checkpoint.period", "1", "--checkpoint.directory", converted_checkpoints],
    )
    check(result.returncode == 0, "double-record restart failed:\n" + result.stdout[-2000:])
    check(
        "is converted to" in result.stdout,
        "double-record restart did not warn about the conversion:\n" + result.stdout[-2000:],
    )

    output_step = args.step + args.steps
    for name in FIELDS:
        for component in "xyz":
            reference = read_field(
                os.path.join(baseline_run, "openPMD", f"{name}_{output_step:06d}.h5"),
                output_step,
                name,
                component,
            )
            converted = read_field(
                os.path.join(run_dir, "openPMD", f"{name}_{output_step:06d}.h5"),
                output_step,
                name,
                component,
            )
            check(
                np.allclose(reference, converted, rtol=1e-6, atol=1e-9),
                f"{name}/{component} differs after double-to-float conversion: "
                f"max abs diff {np.max(np.abs(reference - converted))}",
            )

    baseline_particles = read_checkpoint_particles(
        os.path.join(baseline_checkpoints, f"checkpoint_{output_step:06d}.h5"),
        output_step,
    )
    converted_particles = read_checkpoint_particles(
        os.path.join(converted_checkpoints, f"checkpoint_{output_step:06d}.h5"),
        output_step,
    )
    check(
        baseline_particles.keys() == converted_particles.keys(),
        "particle record sets differ after double-to-float conversion",
    )
    for key, reference in baseline_particles.items():
        converted = converted_particles[key]
        check(
            np.allclose(reference, converted, rtol=1e-6, atol=1e-9),
            f"particle record {key} differs after double-to-float conversion: "
            f"max abs diff {np.max(np.abs(reference.astype(np.float64) - converted.astype(np.float64)))}",
        )
    print("PASS double field and particle records are converted to float")


def scenario_integer_field(args, source, workdir):
    checkpoint_dir = os.path.join(workdir, "integer-field", "checkpoint")
    path = copy_checkpoint(source, checkpoint_dir)
    with h5py.File(path, "r+") as handle:
        mesh = handle[f"{checkpoint_step_path(path, args.step)}/E"]
        rewrite_dataset(mesh, "x", np.int32)
    result, _ = run_restart(args, os.path.join(workdir, "integer-field"), checkpoint_dir, args.periodic)
    check(result.returncode != 0, "integer field datatype was accepted")
    check(
        "only FLOAT and DOUBLE checkpoint data can be converted" in result.stdout
        and "E/x" in result.stdout,
        "integer field datatype did not produce the expected error:\n" + result.stdout[-2000:],
    )
    print("PASS integer field datatype is rejected with a clear error")


def scenario_integer_particle(args, source, workdir):
    checkpoint_dir = os.path.join(workdir, "integer-particle", "checkpoint")
    path = copy_checkpoint(source, checkpoint_dir)
    break_particle_datatype(path, args.step, "electrons", "weighting")
    result, _ = run_restart(
        args, os.path.join(workdir, "integer-particle"), checkpoint_dir, args.periodic
    )
    check(result.returncode != 0, "integer particle record was accepted")
    check(
        "only FLOAT and DOUBLE checkpoint data can be converted" in result.stdout
        and "weighting" in result.stdout,
        "integer particle record did not produce the expected error:\n" + result.stdout[-2000:],
    )
    print("PASS integer particle record is rejected with a clear error")


def read_checkpoint_particles(path, step):
    result = {}
    with h5py.File(path, "r") as handle:
        particles = handle[f"data/{step}/particles"]
        for species_name in particles:
            species = particles[species_name]
            for record_name in species:
                if record_name == "particlePatches":
                    continue
                record = species[record_name]
                if isinstance(record, h5py.Dataset):
                    result[(species_name, record_name)] = record[:]
                else:
                    for component_name in record:
                        result[(species_name, record_name, component_name)] = record[component_name][:]
    return result


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picongpu", required=True)
    parser.add_argument("--checkpoint", required=True, help="checkpoint file to use as scenario source")
    parser.add_argument("--series", default="checkpoint", help="checkpoint.restart.file value")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1, help="number of steps to run")
    parser.add_argument("--setup", required=True, help="deck setup directory with etc/openPMD_config_E/B.toml")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--grid", nargs=3, type=int, default=(16, 16, 8))
    parser.add_argument("--periodic", nargs=3, type=int, default=(1, 1, 1))
    parser.add_argument("--device", default=0)
    parser.add_argument("--mpirun", default="mpirun --allow-run-as-root -n 1")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    baseline_dir = os.path.join(args.workdir, "baseline", "checkpoint")
    copy_checkpoint(args.checkpoint, baseline_dir)
    baseline, baseline_run = run_restart(args, os.path.join(args.workdir, "baseline"), baseline_dir, args.periodic)
    check(baseline.returncode == 0, "unmodified restart failed:\n" + baseline.stdout[-2000:])
    print("PASS unmodified checkpoint restarts")

    scenario_unit_si(args, args.checkpoint, args.workdir, baseline_run)
    scenario_grid_spacing(args, args.checkpoint, args.workdir)
    scenario_missing_pml(args, args.checkpoint, args.workdir)
    scenario_id_provider(args, args.checkpoint, args.workdir)
    scenario_double_conversion(args, args.checkpoint, args.workdir)
    scenario_integer_field(args, args.checkpoint, args.workdir)
    scenario_integer_particle(args, args.checkpoint, args.workdir)
    print("all restart metadata tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print("FAIL:", error, file=sys.stderr)
        sys.exit(1)
