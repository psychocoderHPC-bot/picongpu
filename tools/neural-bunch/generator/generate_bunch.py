#!/usr/bin/env python3
"""Deterministic synthetic electron-bunch generator writing openPMD 2.0 series.

Writes one particle species with position, positionOffset, momentum, weighting,
id, charge and mass records, plus a single global particlePatches entry (unless
``--no-patches``).  Both HDF5 and ADIOS2/BP5 backends write exactly the same
arrays; the arrays are a pure function of the model parameters and ``--seed``.

No PIConGPU code is used.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import math
import os
import platform
import sys

import numpy as np

import nb_common as common

try:
    import openpmd_api as io
except ImportError as error:  # pragma: no cover - environment guard
    sys.exit(
        "openpmd_api is not importable. Use the host venv, e.g.\n"
        "  python generate_bunch.py ... (use an environment with openpmd_api)\n"
        f"original error: {error}"
    )

Q_E = common.ELECTRON_CHARGE
M_E = common.ELECTRON_MASS
C_LIGHT = common.SPEED_OF_LIGHT

UNIT_DIMENSION = {
    "position": {"L": 1.0, "M": 0.0, "T": 0.0, "I": 0.0, "Theta": 0.0, "N": 0.0, "J": 0.0},
    "positionOffset": {"L": 1.0, "M": 0.0, "T": 0.0, "I": 0.0, "Theta": 0.0, "N": 0.0, "J": 0.0},
    "momentum": {"L": 1.0, "M": 1.0, "T": -1.0, "I": 0.0, "Theta": 0.0, "N": 0.0, "J": 0.0},
    "weighting": dict(common.ZERO_DIM),
    "id": dict(common.ZERO_DIM),
    "charge": {"L": 0.0, "M": 0.0, "T": 1.0, "I": 1.0, "Theta": 0.0, "N": 0.0, "J": 0.0},
    "mass": {"L": 0.0, "M": 1.0, "T": 0.0, "I": 0.0, "Theta": 0.0, "N": 0.0, "J": 0.0},
}

MACRO_WEIGHTED = {
    "position": 0,
    "positionOffset": 0,
    "momentum": 1,
    "weighting": 1,
    "id": 0,
    "charge": 0,
    "mass": 0,
}

WEIGHTING_POWER = {
    "position": 0.0,
    "positionOffset": 0.0,
    "momentum": 1.0,
    "weighting": 1.0,
    "id": 0.0,
    "charge": 1.0,
    "mass": 1.0,
}

DIMENSION_ENUM = {
    "L": "L",
    "M": "M",
    "T": "T",
    "I": "I",
    "Theta": "theta",
    "N": "N",
    "J": "J",
}


# ---------------------------------------------------------------------------
# Deterministic sampling helpers
# ---------------------------------------------------------------------------


def norm_ppf(probability):
    """Inverse standard-normal CDF (Acklam rational approximation, |err|<1.2e-9)."""
    p = np.asarray(probability, dtype=np.float64)
    out = np.empty_like(p)

    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    low = p < 0.02425
    high = p > 1.0 - 0.02425
    middle = ~(low | high)

    if np.any(low):
        q = np.sqrt(-2.0 * np.log(p[low]))
        out[low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if np.any(high):
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        out[high] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q
        out[middle] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    return out


def stratified_normals(count: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-hypercube (stratified) standard normals, one array per dimension."""
    strata = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
    return norm_ppf(strata)


def sample_standard_normals(count: int, rng: np.random.Generator, quiet: bool) -> np.ndarray:
    if quiet:
        return stratified_normals(count, rng)
    return rng.standard_normal(count)


def parse_vector(text: str) -> np.ndarray:
    parts = [part for part in text.replace(";", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected three comma-separated numbers, got {text!r}")
    vector = np.asarray([float(part) for part in parts], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise argparse.ArgumentTypeError(f"non-finite vector {text!r}")
    return vector


def parse_axis(text: str) -> np.ndarray:
    mapping = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    if text in mapping:
        return mapping[text]
    return parse_vector(text)


def betas_to_u(beta: np.ndarray) -> np.ndarray:
    beta2 = float(np.dot(beta, beta))
    if beta2 >= 1.0:
        raise ValueError(f"|beta| >= 1 is not physical: {beta}")
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    return gamma * beta


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def make_weights(args) -> tuple[np.ndarray, float]:
    count = args.n
    if args.weights == "uniform":
        return np.ones(count, dtype=np.float64), float(count)
    if args.weights == "linear":
        if count == 1:
            return np.ones(1, dtype=np.float64), 1.0
        weights = 0.5 + np.arange(count, dtype=np.float64) / (count - 1)
        return weights, float(weights.sum())
    if args.weights == "two-tier":
        split = int(math.floor(args.weight_fraction * count))
        weights = np.ones(count, dtype=np.float64)
        weights[:split] = args.weight_ratio
        return weights, float(weights.sum())
    raise ValueError(f"unknown weight mode {args.weights}")


def sample_positions(args, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(args.grid, dtype=np.float64)
    centroid = np.asarray(args.centroid, dtype=np.float64)
    sigma = np.asarray(args.sigma, dtype=np.float64)
    position_cells = np.empty((args.n, 3), dtype=np.float64)

    for axis in range(3):
        normals = sample_standard_normals(args.n, rng, args.quiet)
        position_cells[:, axis] = centroid[axis] + sigma[axis] * normals

    # Truncate at the domain boundary by deterministically resampling.  The
    # default parameters leave >4 sigma of margin, so this normally runs once.
    for _attempt in range(1000):
        outside = np.any((position_cells < 0.0) | (position_cells >= grid), axis=1)
        bad = int(np.count_nonzero(outside))
        if bad == 0:
            break
        for axis in range(3):
            if args.quiet:
                uniforms = (np.arange(bad) + rng.random(bad)) / max(bad, 1)
                normals = norm_ppf(uniforms)
            else:
                normals = rng.standard_normal(bad)
            position_cells[outside, axis] = centroid[axis] + sigma[axis] * normals
    else:  # pragma: no cover - only with absurd parameters
        raise SystemExit("could not truncate the Gaussian bunch inside the domain; reduce sigma or move the centroid")

    return position_cells, grid


def component_u(entry: dict) -> np.ndarray:
    if "beta" in entry:
        beta = np.asarray(entry["beta"], dtype=np.float64)
        if beta.shape != (3,):
            raise ValueError(f"mixed component 'beta' must have three entries: {entry}")
        return betas_to_u(beta)
    beta_scalar = float(entry.get("beta_scalar", entry.get("beta_magnitude", 0.0)))
    direction = np.asarray(entry.get("direction", [1.0, 0.0, 0.0]), dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError(f"mixed component direction is zero: {entry}")
    return betas_to_u(beta_scalar * direction / norm)


def sample_momenta(args, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    u = np.zeros((args.n, 3), dtype=np.float64)
    details = {"model": args.model}
    thermal = args.u_sigma
    direction = parse_axis(args.direction if args.direction else "x")

    if args.model == "stationary":
        details["gamma_target"] = 1.0
        details["beta_target"] = 0.0
    elif args.model == "drift":
        if not (0.0 <= args.beta < 1.0):
            raise SystemExit("--beta must satisfy 0 <= beta < 1")
        u[:] = betas_to_u(args.beta * direction / np.linalg.norm(direction))
        details["beta_target"] = args.beta
        details["gamma_target"] = float(1.0 / math.sqrt(1.0 - args.beta**2))
        details["direction"] = direction.tolist()
    elif args.model == "gamma":
        if args.gamma < 1.0:
            raise SystemExit("--gamma must be >= 1")
        magnitude = math.sqrt(args.gamma**2 - 1.0)
        u[:] = magnitude * direction / np.linalg.norm(direction)
        details["gamma_target"] = args.gamma
        details["beta_target"] = float(magnitude / args.gamma)
        details["direction"] = direction.tolist()
    elif args.model == "counterstreaming":
        if not (0.0 < args.beta < 1.0):
            raise SystemExit("--beta must satisfy 0 < beta < 1 for counterstreaming")
        axis = parse_axis(args.axis)
        axis = axis / np.linalg.norm(axis)
        magnitude = betas_to_u(args.beta * axis)
        n_plus = (args.n + 1) // 2
        u[:n_plus] = magnitude
        u[n_plus:] = -magnitude
        details["beta_target"] = args.beta
        details["axis"] = axis.tolist()
        details["lobes"] = [n_plus, args.n - n_plus]
    elif args.model == "mixed":
        spec = json.loads(args.mix_spec)
        if not isinstance(spec, list) or not spec:
            raise SystemExit("--mix-spec must be a non-empty JSON list")
        fractions = np.asarray([float(entry["fraction"]) for entry in spec], dtype=np.float64)
        if np.any(fractions <= 0.0):
            raise SystemExit("mixed component fractions must be positive")
        fractions /= fractions.sum()
        assignment = rng.choice(len(spec), size=args.n, p=fractions)
        for index, entry in enumerate(spec):
            mask = assignment == index
            base = component_u(entry)
            u[mask] = base
            sigma_u = float(entry.get("u_sigma", 0.0))
            if sigma_u > 0.0:
                u[mask] += rng.standard_normal((int(mask.sum()), 3)) * sigma_u
        details["components"] = spec
    else:
        raise SystemExit(f"unknown model {args.model}")

    if thermal > 0.0:
        u += rng.standard_normal((args.n, 3)) * thermal
        details["u_sigma"] = thermal

    # Return single-electron u = p/(m_e*c); build() folds the weight into the
    # raw macro momentum so that u = momentum_raw / w.
    return u, details


def build(args) -> tuple[dict, dict]:
    seed_sequence = np.random.SeedSequence(args.seed)
    position_stream, velocity_stream = seed_sequence.spawn(2)
    rng_position = np.random.Generator(np.random.PCG64(position_stream))
    rng_velocity = np.random.Generator(np.random.PCG64(velocity_stream))

    position_cells, grid = sample_positions(args, rng_position)
    # PIConGPU-compatible layout: position is the in-cell fraction in [0,1),
    # positionOffset the integer cell index; physical = (both) * cellSize.
    position_offset = np.floor(position_cells).astype(np.float64)
    position = position_cells - position_offset

    u, momentum_details = sample_momenta(args, rng_velocity)
    weighting, expected_weight = make_weights(args)
    ids = np.arange(1, args.n + 1, dtype=np.uint64)
    # Macro momentum in units of m_e*c with weightingPower=1: raw = u * w.
    momentum = u * weighting[:, None]

    data = {
        "position": np.ascontiguousarray(position, dtype=np.float64),
        "positionOffset": np.ascontiguousarray(position_offset, dtype=np.float64),
        "momentum": np.ascontiguousarray(momentum, dtype=np.float64),
        "weighting": np.ascontiguousarray(weighting, dtype=np.float64),
        "id": np.ascontiguousarray(ids, dtype=np.uint64),
    }
    params = {
        "model": args.model,
        "seed": args.seed,
        "n": args.n,
        "grid": [int(value) for value in args.grid],
        "cell_size_SI": [float(value) for value in args.cell_size],
        "centroid_cells": [float(value) for value in args.centroid],
        "sigma_cells": [float(value) for value in args.sigma],
        "quiet": bool(args.quiet),
        "weights": args.weights,
        "weight_ratio": float(args.weight_ratio),
        "weight_fraction": float(args.weight_fraction),
        "u_sigma": float(args.u_sigma),
        "momentum": momentum_details,
        "momentum_unit_SI": common.MOMENTUM_UNIT_SI,
        "expected_total_weight": expected_weight,
        "position_convention": "(position + positionOffset) * cellSize, position raw in [0,1)",
    }
    return data, params


# ---------------------------------------------------------------------------
# Validation before writing
# ---------------------------------------------------------------------------


def validate_before_write(data: dict, params: dict, args) -> dict:
    problems = []
    n = data["position"].shape[0]
    if n != args.n:
        problems.append(f"particle count {n} != requested {args.n}")
    for name, array in data.items():
        if not np.all(np.isfinite(array)):
            problems.append(f"{name}: non-finite values")
    if np.any(data["weighting"] <= 0.0):
        problems.append("weights must be strictly positive")
    if np.any(np.diff(data["weighting"]) < 0.0) and args.weights == "linear":
        problems.append("linear weights are not monotonic")

    total_weight = float(data["weighting"].sum())
    if not math.isclose(total_weight, params["expected_total_weight"], rel_tol=1e-12):
        problems.append(f"total weight {total_weight!r} != expected {params['expected_total_weight']!r}")
    total_charge = Q_E * total_weight

    grid = np.asarray(params["grid"], dtype=np.float64)
    cell = np.asarray(params["cell_size_SI"], dtype=np.float64)
    extent = grid * cell
    fraction = data["position"]
    if np.any(fraction < 0.0) or np.any(fraction >= 1.0):
        problems.append("position raw values are not in-cell fractions [0,1)")
    if not np.allclose(data["positionOffset"], np.round(data["positionOffset"]), rtol=0.0, atol=1e-9):
        problems.append("positionOffset is not an integer cell index")
    if np.any(data["positionOffset"] < 0.0) or np.any(data["positionOffset"] >= grid):
        problems.append("positionOffset has a cell index outside the grid")
    position = (fraction + data["positionOffset"]) * cell
    if np.any(position < 0.0) or np.any(position >= extent):
        problems.append("decoded physical positions fall outside [0, grid*cellSize)")

    if not np.array_equal(data["id"], np.arange(1, n + 1, dtype=np.uint64)):
        problems.append("ids are not 1..N")

    # Decode the single-electron u exactly as a reader would.
    u = data["momentum"] * (M_E * C_LIGHT) / (M_E * C_LIGHT) / data["weighting"][:, None]
    umag = np.linalg.norm(u, axis=1)
    gamma = np.sqrt(1.0 + umag * umag)
    if not np.all(np.isfinite(gamma)) or np.any(gamma < 1.0 - 1e-12):
        problems.append("invalid gamma from momentum")

    model = params["model"]
    if model == "gamma" and args.u_sigma == 0.0:
        target = params["momentum"]["gamma_target"]
        if not np.allclose(gamma, target, rtol=1e-12, atol=0.0):
            problems.append(f"gamma model: realized gamma [{gamma.min()}, {gamma.max()}] != {target}")
    if model == "drift" and args.u_sigma == 0.0:
        target = params["momentum"]["gamma_target"]
        if not np.allclose(gamma, target, rtol=1e-12, atol=0.0):
            problems.append(f"drift model: realized gamma [{gamma.min()}, {gamma.max()}] != {target}")
    if model == "counterstreaming" and args.u_sigma == 0.0:
        n_plus, n_minus = params["momentum"]["lobes"]
        plus = umag[:n_plus]
        minus = umag[n_plus : n_plus + n_minus]
        if not np.allclose(plus, plus[0], rtol=1e-12) or not np.allclose(minus, minus[0], rtol=1e-12):
            problems.append("counterstreaming lobes are not monoenergetic")

    if problems:
        message = ["refusing to write invalid bunch:"]
        message.extend(f"  - {problem}" for problem in problems)
        raise SystemExit("\n".join(message))

    stats = common.momentum_stats(data, u)
    return {
        "count": n,
        "total_weight": total_weight,
        "total_charge_C": total_charge,
        "charge_per_macro_SI": Q_E,
        "mass_per_macro_kg": M_E,
        "momentum": stats,
        "position_min_SI": [float(value) for value in position.min(axis=0)],
        "position_max_SI": [float(value) for value in position.max(axis=0)],
        "all_positions_inside_domain": True,
    }


# ---------------------------------------------------------------------------
# openPMD writing
# ---------------------------------------------------------------------------


def resolve_output(args) -> tuple[str, str]:
    suffix = ".h5" if args.backend == "hdf5" else ".bp5"
    output = args.output
    if "%T" in output:
        output = output.replace("%T", str(args.iteration))
    if not output.endswith(suffix):
        if os.path.splitext(output)[1] in (".h5", ".hdf5", ".bp5", ".bp"):
            raise SystemExit(
                f"--output suffix {os.path.splitext(output)[1]!r} does not match "
                f"--backend {args.backend!r} (expected {suffix})"
            )
        output += suffix
    directory = os.path.dirname(os.path.abspath(output))
    if directory:
        os.makedirs(directory, exist_ok=True)
    return output, suffix


def unit_dimension_map(values: dict) -> dict:
    return {getattr(io.Unit_Dimension, DIMENSION_ENUM[name]): float(value) for name, value in values.items()}


def write_record(species, record_name: str, array: np.ndarray, args, keep: list) -> None:
    record = species[record_name]
    record.set_unit_dimension(unit_dimension_map(UNIT_DIMENSION[record_name]))
    record.set_attribute("macroWeighted", np.int32(MACRO_WEIGHTED[record_name]))
    record.set_attribute("weightingPower", np.float64(WEIGHTING_POWER[record_name]))
    record.set_attribute("timeOffset", np.float64(0.0))

    count = array.shape[0]
    if array.ndim == 2:
        for axis, component in enumerate(common.POSITION_OFFSET_NAMES):
            component_record = record[component]
            component_record.reset_dataset(io.Dataset(np.dtype("float64"), [count]))
            if record_name in ("position", "positionOffset"):
                component_record.set_unit_SI(float(args.cell_size[axis]))
            elif record_name == "momentum":
                component_record.set_unit_SI(common.MOMENTUM_UNIT_SI)
            else:
                component_record.set_unit_SI(1.0)
            write_chunks(component_record, array[:, axis], args.chunks, keep)
    else:
        dtype = np.dtype("uint64") if record_name == "id" else np.dtype("float64")
        component_record = record[record_name]
        component_record.reset_dataset(io.Dataset(dtype, [count]))
        component_record.set_unit_SI(1.0)
        if record_name == "weighting":
            component_record.set_attribute("macroWeighted", np.uint32(1))
            component_record.set_attribute("weightingPower", np.float64(1.0))
        write_chunks(component_record, array, args.chunks, keep)


def write_chunks(component_record, values: np.ndarray, chunks: int, keep: list) -> None:
    count = values.shape[0]
    bounds = np.linspace(0, count, chunks + 1).astype(int)
    for low, high in zip(bounds[:-1], bounds[1:]):
        if high <= low:
            continue
        block = np.ascontiguousarray(values[low:high])
        keep.append(block)
        component_record.store_chunk(block, [int(low)], [int(high - low)])


def write_constant_record(species, record_name: str, value: float, count: int, keep: list) -> None:
    record = species[record_name]
    record.set_unit_dimension(unit_dimension_map(UNIT_DIMENSION[record_name]))
    record.set_attribute("macroWeighted", np.int32(MACRO_WEIGHTED[record_name]))
    record.set_attribute("weightingPower", np.float64(WEIGHTING_POWER[record_name]))
    record.set_attribute("timeOffset", np.float64(0.0))
    component_record = record[record_name]
    # Extent must be the particle count (not 1): openPMD rejects mixed extents
    # when reading the species otherwise.
    component_record.reset_dataset(io.Dataset(np.dtype("float64"), [count]))
    component_record.set_unit_SI(1.0)
    component_record.make_constant(np.float64(value))


def write_particle_patches(species, count: int, grid, keep: list) -> None:
    patches = species.particle_patches
    for name, value in (("numParticles", count), ("numParticlesOffset", 0)):
        component = patches[name][io.Record.SCALAR]
        component.reset_dataset(io.Dataset(np.dtype("uint64"), [1]))
        component.set_unit_SI(1.0)
        block = np.asarray([value], dtype=np.uint64)
        keep.append(block)
        component.store_chunk(block, [0], [1])
    for name, values in (("offset", (0, 0, 0)), ("extent", tuple(int(v) for v in grid))):
        record = patches[name]
        for axis, value in zip(common.POSITION_OFFSET_NAMES, values):
            component = record[axis]
            component.reset_dataset(io.Dataset(np.dtype("uint64"), [1]))
            component.set_unit_SI(1.0)
            block = np.asarray([value], dtype=np.uint64)
            keep.append(block)
            component.store_chunk(block, [0], [1])


def write_series(output: str, data: dict, params: dict, args) -> None:
    series = io.Series(output, io.Access.create)
    series.set_openPMD("2.0.0")
    iteration = series.iterations[args.iteration]
    iteration.open()
    species = iteration.particles[args.species]

    keep = []
    for record_name in ("position", "positionOffset", "momentum"):
        write_record(species, record_name, data[record_name], args, keep)
    for record_name in ("weighting", "id"):
        write_record(species, record_name, data[record_name], args, keep)
    write_constant_record(species, "charge", Q_E, args.n, keep)
    write_constant_record(species, "mass", M_E, args.n, keep)

    if not args.no_patches:
        write_particle_patches(species, args.n, params["grid"], keep)

    species.set_attribute("grid", [int(value) for value in params["grid"]])
    species.set_attribute("cellSizeSI", [float(value) for value in params["cell_size_SI"]])
    species.set_attribute("numParticles", int(args.n))
    species.set_attribute("particleShape", np.float64(2.0))
    species.set_attribute("currentDeposition", "direct")
    species.set_attribute("particlePush", "vay")
    species.set_attribute("particleInterpolation", "linear")

    series.flush()
    # The Python bindings keep writable spans alive until the Series is
    # destroyed; drop only after the flush and keep the payload referenced.
    _ = keep
    del series
    gc.collect()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="series path (suffix added if missing)")
    parser.add_argument("--backend", choices=("hdf5", "adios2"), default="hdf5", help="storage backend")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--species", default=common.DEFAULT_SPECIES)
    parser.add_argument(
        "--model",
        choices=("stationary", "drift", "gamma", "counterstreaming", "mixed"),
        default="stationary",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n", type=int, default=4096, help="particle count")
    parser.add_argument("--grid", type=int, nargs=3, default=(64, 64, 64), metavar=("NX", "NY", "NZ"))
    parser.add_argument(
        "--cell-size",
        type=float,
        nargs=3,
        default=(1e-7, 1e-7, 1e-7),
        metavar=("DX", "DY", "DZ"),
        help="cell size in metres",
    )
    parser.add_argument(
        "--centroid",
        type=float,
        nargs=3,
        default=(32.0, 32.0, 32.0),
        metavar=("CX", "CY", "CZ"),
        help="centroid in cells",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        nargs=3,
        default=(6.0, 6.0, 6.0),
        metavar=("SX", "SY", "SZ"),
        help="Gaussian sigma in cells",
    )
    parser.add_argument("--quiet", action="store_true", help="stratified/Latin-hypercube sampling")
    parser.add_argument("--beta", type=float, default=0.1, help="drift/counterstreaming beta")
    parser.add_argument("--direction", default="x", help="drift direction: x|y|z or 'dx,dy,dz'")
    parser.add_argument("--gamma", type=float, default=2.0, help="target gamma for --model gamma")
    parser.add_argument("--axis", default="x", help="counterstreaming axis: x|y|z or vector")
    parser.add_argument(
        "--mix-spec",
        default='[{"fraction":0.5,"beta":[0.01,0,0]},{"fraction":0.3,"beta":[0,0.1,0]},'
        '{"fraction":0.2,"beta":[-0.05,0,0.02]}]',
        help="JSON list of mixed-velocity components",
    )
    parser.add_argument("--u-sigma", type=float, default=0.0, help="isotropic thermal spread in u = p/(m c)")
    parser.add_argument("--weights", choices=("uniform", "linear", "two-tier"), default="uniform")
    parser.add_argument("--weight-ratio", type=float, default=2.5)
    parser.add_argument("--weight-fraction", type=float, default=0.4)
    parser.add_argument("--no-patches", action="store_true", help="write a flat record without particlePatches")
    parser.add_argument("--chunks", type=int, default=1, help="number of store_chunk calls per component")
    parser.add_argument("--manifest", default=None, help="manifest path (default: <output>.manifest.json)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n must be >= 1")
    if any(value <= 0 for value in args.grid):
        parser.error("--grid entries must be positive")
    if any(value <= 0.0 for value in args.cell_size):
        parser.error("--cell-size entries must be positive")
    if any(value <= 0.0 for value in args.sigma):
        parser.error("--sigma entries must be positive")
    if args.chunks < 1 or args.chunks > args.n:
        parser.error("--chunks must satisfy 1 <= chunks <= n")
    if args.weights == "two-tier" and args.weight_ratio <= 0.0:
        parser.error("--weight-ratio must be positive")
    args.cell_size = tuple(float(value) for value in args.cell_size)
    args.grid = tuple(int(value) for value in args.grid)
    args.centroid = tuple(float(value) for value in args.centroid)
    args.sigma = tuple(float(value) for value in args.sigma)

    data, params = build(args)
    summary = validate_before_write(data, params, args)

    output, suffix = resolve_output(args)
    write_series(output, data, params, args)

    # Independent cross-check sidecar: decoded physical positions and u.
    cell = np.asarray(args.cell_size, dtype=np.float64)
    physical = (data["position"] + data["positionOffset"]) * cell
    u = data["momentum"] / data["weighting"][:, None]
    sidecar_file = common.sidecar_path(output)
    sidecar = common.write_sidecar(sidecar_file, physical, u, data["weighting"], data["id"])

    manifest_path = args.manifest or (output + ".manifest.json")
    files = common.file_manifest(output)
    manifest = {
        "format": "picongpu-external-bunch",
        "version": 1,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "generate_bunch.py",
        "openPMD_version": str(io.__version__),
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "backend": args.backend,
        "backend_reported": suffix,
        "series": os.path.abspath(output),
        "iteration": int(args.iteration),
        "species": args.species,
        "particles": int(args.n),
        "chunks": int(args.chunks),
        "patches": not args.no_patches,
        "dtypes": {
            "position": "float64",
            "positionOffset": "float64",
            "momentum": "float64",
            "weighting": "float64",
            "id": "uint64",
            "charge": "float64 (constant)",
            "mass": "float64 (constant)",
        },
        "units": {
            "position": "in-cell fraction [0,1), unitSI=dx,dy,dz, unitDimension L=1",
            "positionOffset": "integer cell index, float64, unitSI=dx,dy,dz, unitDimension L=1",
            "physical_position": "(position + positionOffset) * cellSize, metres",
            "momentum": f"macro momentum in m_e*c units, unitSI={common.MOMENTUM_UNIT_SI!r}, "
            "unitDimension L=1,M=1,T=-1, macroWeighted=1, weightingPower=1; raw=u*w",
            "u": "momentum * unitSI / (m_e*c) / w^weightingPower",
            "weighting": "unitSI=1, macroWeighted=1, weightingPower=1",
            "charge": f"{Q_E!r} C (constant), unitSI=1, I=1,T=1",
            "mass": f"{M_E!r} kg (constant), unitSI=1, M=1",
            "position_convention": "(position + positionOffset) * cellSize",
        },
        "parameters": params,
        "validation": summary,
        "array_sha256": common.array_checksums(data),
        "constant_values": {"charge": {"charge": Q_E}, "mass": {"mass": M_E}},
        "sidecar": sidecar,
        "files": files,
        "file_bytes_total": common.total_bytes(files),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {output} ({args.backend}, {args.n} particles, {args.chunks} chunk(s))")
    print(f"  total_weight={summary['total_weight']!r} total_charge={summary['total_charge_C']!r} C")
    print("  gamma: min={gamma_min:.6g} max={gamma_max:.6g} mean={gamma_mean:.6g}".format(**summary["momentum"]))
    print(f"  manifest: {manifest_path}")
    print(f"  sidecar: {sidecar_file} (sha256 {sidecar['sha256'][:16]}...)")
    print(f"  bytes: {manifest['file_bytes_total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
