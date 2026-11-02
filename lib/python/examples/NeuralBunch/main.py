#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Literal

from picongpu import picmi

"""
@file Minimal PICMI example for the external neural electron-bunch initializer

The example creates a small openPMD electron bunch with the deterministic
generator in ``tools/neural-bunch/generator``, loads it through
``picmi.NeuralBunchInitializer`` and runs two PIConGPU steps. Both HDF5 and
ADIOS2/BP5 series are supported.

The destination species is deliberately *not* given an ``initial_distribution``:
the external file supplies every particle, so a density initializer would
conflict with it (the PICMI validation rejects that combination).
"""

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
GENERATOR = REPO_ROOT / "tools" / "neural-bunch" / "generator" / "generate_bunch.py"
SMOKE_MODEL = REPO_ROOT / "tools" / "neural-bunch" / "training" / "smoke" / "model"

NUMBER_OF_CELLS = (16, 16, 8)
CELL_SIZE_SI = (1.0e-7, 1.0e-7, 1.0e-7)
GRID_UPPER_BOUND = tuple(n * dx for n, dx in zip(NUMBER_OF_CELLS, CELL_SIZE_SI))
N_PARTICLES = 16
BACKEND_SUFFIX = {"hdf5": ".h5", "adios2": ".bp5"}
MODE: Literal["run", "write"] = "run"


def data_directory(backend: Literal["hdf5", "adios2"]) -> Path:
    return HERE / "neural_bunch_data" / backend


def series_path(backend: Literal["hdf5", "adios2"]) -> Path:
    return data_directory(backend) / ("bunch" + BACKEND_SUFFIX[backend])


def generate_series(backend: Literal["hdf5", "adios2"]) -> Path:
    """Create the deterministic openPMD source series if it is missing."""
    series = series_path(backend)
    if series.exists():
        return series

    data_directory(backend).mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(GENERATOR),
        "--output",
        str(series.with_suffix("")),
        "--backend",
        backend,
        "--iteration",
        "0",
        "--species",
        "e",
        "--model",
        "stationary",
        "--seed",
        "1001",
        "--n",
        str(N_PARTICLES),
        "--grid",
        *map(str, NUMBER_OF_CELLS),
        "--cell-size",
        *map(str, CELL_SIZE_SI),
        "--centroid",
        "8",
        "8",
        "4",
        "--sigma",
        "2",
        "2",
        "2",
    ]
    print("generating bunch:", " ".join(command))
    subprocess.run(command, check=True)
    return series


def build_simulation(
    backend: Literal["hdf5", "adios2"] = "hdf5",
    field_initializer: Literal["neural", "validatedNeural", "referenceOnly"] = "referenceOnly",
    max_steps: int = 2,
) -> picmi.Simulation:
    """Build the PICMI simulation for the requested backend and policy."""
    grid = picmi.Cartesian3DGrid(
        number_of_cells=list(NUMBER_OF_CELLS),
        lower_bound=[0.0, 0.0, 0.0],
        upper_bound=list(GRID_UPPER_BOUND),
        lower_boundary_conditions=["periodic", "periodic", "periodic"],
        upper_boundary_conditions=["periodic", "periodic", "periodic"],
    )
    solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=1.0)

    electrons = picmi.Species(particle_type="electron", name="electrons")

    initializer = picmi.NeuralBunchInitializer(
        filename=str(series_path(backend)),
        iteration=0,
        source_species="e",
        model_path=str(SMOKE_MODEL),
        chunk_size=1 << 20,
        tile_batch=1,
        field_initializer=field_initializer,
        enable_reference_fallback=True,
        time_policy=0,
        species=electrons,
    )

    return picmi.Simulation(
        max_steps=max_steps,
        solver=solver,
        species=[electrons],
        layouts=[None],
        picongpu_neural_bunch=initializer,
        picongpu_typical_ppc=1,
    )


# the compiling example test imports ``sim`` from this module
sim = build_simulation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=tuple(BACKEND_SUFFIX), default="hdf5")
    parser.add_argument(
        "--field-initializer",
        choices=("neural", "validatedNeural", "referenceOnly"),
        default="referenceOnly",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    generate_series(args.backend)
    if args.generate_only:
        raise SystemExit(0)

    simulation = build_simulation(args.backend, args.field_initializer, args.steps)
    match MODE:
        case "run":
            simulation.run(
                setup_dir=HERE / "neural_bunch_data" / args.backend / "setup",
                run_dir=HERE / "neural_bunch_data" / args.backend / "run",
            )
        case "write":
            simulation.write_input_file(HERE / "neural_bunch_data" / args.backend / "setup")
        case _:
            raise ValueError(f"Unknown {MODE=}.")
