"""Scaled real-reference dataset for the deliverable model.

Full aspect corners in [0.1, 10], gamma log-sampled in [1, 100] including both
endpoints, distances 4..512 cells, sub-cell source positions, all direction
classes, and multi-source bunches split into variable-size chunks.  Physical
configurations are assigned to train/val/test by CRC32 of the configuration
key, so no physical configuration is shared between splits.

Targets are produced by the compiled C++ FP64 reference (``_nbref_cpp``),
which is validated against ``reference/convkernel.py`` by
``scripts/validate_fast.py``.  The manifest records the extension SHA-256, the
C++ header SHA-256, the Python reference SHA-256, the per-sample preregistered
scale ``S0`` and ``d_min_cells`` before training.
"""

from __future__ import annotations

import json
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np

from . import reference_data, sampling
from .fast_reference import load_fast_reference
from .reference_loader import sha256_file

SPLITS = ("train", "val", "test")
SPLIT_SEED_OFFSET = {"train": 0, "val": 1000, "test": 2000}

GAMMAS = [float(x) for x in np.logspace(0.0, 2.0, 13)]
ASPECT_VALUES = [0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0]
ASPECT_PAIRS = [(ay, az) for ay in ASPECT_VALUES for az in ASPECT_VALUES]
DIRECTIONS = ["random", "axis_x", "axis_y", "axis_z", "diag_p", "diag_m"]
POSITION_MODES = ["random", "subcell", "near", "far", "translation", "scaled", "thin"]

BUNCH_KINDS = ["gaussian", "two_lobe", "counterstreaming", "crossing", "unequal_weights"]
#: counterstreaming/crossing are restricted to gamma <= 10: a gamma-100 beam
#: whose trajectory passes close to the tile has a Lorentz-contracted feature
#: far below the TSC cell size, which the FP64 quadrature resolves only at a
#: very high cost (tens of seconds per source).  The regime stays represented
#: by moving gaussian/two-lobe/unequal-weight bunches at gamma 100.
BUNCH_GAMMAS_BY_KIND = {
    "gaussian": [1.0, 10.0, 100.0],
    "two_lobe": [1.0, 10.0, 100.0],
    "unequal_weights": [1.0, 10.0, 100.0],
    "counterstreaming": [1.0, 10.0],
    "crossing": [1.0, 10.0],
}
BUNCH_ASPECTS = [(0.1, 0.1), (0.1, 10.0), (1.0, 1.0), (10.0, 0.1), (10.0, 10.0)]
BUNCH_SEEDS = [0, 1]
BUNCH_N_BY_GAMMA = {1.0: [16, 32, 64], 10.0: [16, 32], 100.0: [16]}
BUNCH_FLOOR_CELLS = {1.0: 12.0, 10.0: 16.0, 100.0: 24.0}
BUNCH_CHUNK_SIZES = (1, 17, 1024, 5)

TOTAL_TILES_ESTIMATE = len(GAMMAS) * len(DIRECTIONS) * len(ASPECT_PAIRS) * len(POSITION_MODES) * 1.15 + 400


def _gamma_label(g):
    return f"g{g:.4g}"


def _split_of(key: str) -> str:
    return SPLITS[zlib.crc32(key.encode("utf-8")) % len(SPLITS)]


# --------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------
def scenario_configs():
    for gamma in GAMMAS:
        for direction in DIRECTIONS:
            for ai, (ay, az) in enumerate(ASPECT_PAIRS):
                for pm in POSITION_MODES:
                    key = f"scenario|{gamma:.9g}|{direction}|{ai}|{pm}"
                    yield key, (gamma, direction, ai, pm)


def _bunch_n(gamma, kind_index, gi, ai, bs):
    choices = BUNCH_N_BY_GAMMA[float(gamma)]
    return choices[(kind_index * 7 + gi * 5 + ai * 3 + bs * 2) % len(choices)]


def bunch_configs():
    for ki, kind in enumerate(BUNCH_KINDS):
        for gi, gamma in enumerate(BUNCH_GAMMAS_BY_KIND[kind]):
            for ai, (ay, az) in enumerate(BUNCH_ASPECTS):
                for bs in BUNCH_SEEDS:
                    seed = 104729 * ki + 7919 * int(gamma) + 1009 * ai + 97 * bs + 13
                    n = _bunch_n(gamma, ki, gi, ai, bs)
                    key = f"bunch|{kind}|{gamma:.9g}|{ai}|{bs}"
                    yield key, (kind, gamma, ai, bs, seed, n)


def descriptors_for_split(split: str, extra_fraction: int = 5) -> List[dict]:
    descs = []
    for key, (gamma, direction, ai, pm) in scenario_configs():
        if _split_of(key) != split:
            continue
        ay, az = ASPECT_PAIRS[ai]
        glabel = _gamma_label(gamma)
        indices = [0]
        # Do not duplicate the expensive gamma>10 near/sub-cell regimes.
        expensive = gamma > 10.0 and pm in ("near", "subcell")
        if not expensive and zlib.crc32((key + "|extra").encode("utf-8")) % extra_fraction == 0:
            indices.append(1)
        for si in indices:
            descs.append(
                {
                    "kind": "scenario",
                    "config": [glabel, direction, f"aspect_{ai}", pm],
                    "gamma": float(gamma),
                    "gamma_name": glabel,
                    "direction": direction,
                    "aspect_index": int(ai),
                    "aspects": [float(ay), float(az)],
                    "position_mode": pm,
                    "sample_index": int(si),
                    "split_key": key,
                }
            )
    for key, (kind, gamma, ai, bs, seed, n) in bunch_configs():
        if _split_of(key) != split:
            continue
        ay, az = BUNCH_ASPECTS[ai]
        sizes = sampling.chunk_sizes(n, BUNCH_CHUNK_SIZES)
        for ci in range(len(sizes)):
            descs.append(
                {
                    "kind": "bunch",
                    "bunch_kind": kind,
                    "gamma": float(gamma),
                    "gamma_name": _gamma_label(gamma),
                    "direction": "bunch",
                    "position_mode": "bunch",
                    "aspect_index": int(ai),
                    "aspects": [float(ay), float(az)],
                    "bunch_split": "bunch",
                    "bunch_seed": int(seed),
                    "n_particles": int(n),
                    "chunk_index": int(ci),
                    "chunk_sizes": [int(s) for s in sizes],
                    "sample_index": int(ci),
                    "split_key": key,
                }
            )
    descs.sort(key=lambda d: (d["split_key"], d.get("sample_index", 0), d.get("chunk_index", 0)))
    return descs


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
def _scenario_sample(desc: dict, split: str, seed: int) -> sampling.Sample:
    rng = np.random.default_rng(
        sampling._stable_seed("scaledsample", split, seed, desc["split_key"], desc["sample_index"])
    )
    gamma = float(desc["gamma"])
    beta = float(np.sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma))))
    unit = sampling._direction(rng, desc["direction"])
    u = beta * gamma * unit
    ay, az = float(desc["aspects"][0]), float(desc["aspects"][1])
    dx = float(2.0 ** rng.integers(-2, 3))
    origin = rng.uniform(-30.0, 30.0, size=3) * dx
    pos, origin, dx, dmin = sampling._place_single(rng, origin, dx, ay, az, desc["position_mode"])
    particles = sampling._particle_dict(pos[None, :], u[None, :], np.array([1.0]))
    return sampling.Sample(
        origin=np.asarray(origin, dtype=np.float64),
        dx=dx,
        a_y=ay,
        a_z=az,
        particles=particles,
        meta={
            "kind": "scenario",
            "split": split,
            "seed": int(seed),
            "gamma": gamma,
            "direction": desc["direction"],
            "position_mode": desc["position_mode"],
            "d_min_cells": float(dmin),
        },
    )


def _min_distance_cells(sample: sampling.Sample) -> float:
    pts = sampling.output_points(sample.origin, sample.dx, sample.a_y, sample.a_z)
    p = sample.particles
    pos = np.stack([p["x"], p["y"], p["z"]], axis=1)
    return float(np.min(np.linalg.norm(pos[:, None, :] - pts[None, :, :], axis=2)) / sample.dx)


def _ensure_domain(sample: sampling.Sample, floor_cells: float = 4.5) -> sampling.Sample:
    """Shift a bunch radially until every source is at least ``floor_cells``
    away from the nearest staggered output point (the reference near branch is
    not part of the neural domain)."""
    pts = sampling.output_points(sample.origin, sample.dx, sample.a_y, sample.a_z)
    p = sample.particles
    pos = np.stack([p["x"], p["y"], p["z"]], axis=1)
    for _ in range(24):
        d = np.linalg.norm(pos[:, None, :] - pts[None, :, :], axis=2)
        k, q = np.unravel_index(np.argmin(d), d.shape)
        dmin = float(d[k, q]) / sample.dx
        if dmin >= floor_cells:
            break
        direction = pos[k] - pts[q]
        nrm = np.linalg.norm(direction)
        if nrm == 0.0:
            direction = np.array([1.0, 0.0, 0.0])
            nrm = 1.0
        shift = (floor_cells - dmin + 0.25) * sample.dx * direction / nrm
        pos = pos + shift
    p["x"], p["y"], p["z"] = pos[:, 0].copy(), pos[:, 1].copy(), pos[:, 2].copy()
    return sample


def _bunch_sample(desc: dict, split: str, seed: int) -> sampling.Sample:
    ay, az = float(desc["aspects"][0]), float(desc["aspects"][1])
    full = sampling.sample_bunch(
        desc["bunch_kind"],
        split=desc["bunch_split"],
        seed=int(desc["bunch_seed"]),
        n_particles=int(desc["n_particles"]),
        gamma=float(desc["gamma"]),
        a_y=ay,
        a_z=az,
        dx=1.0,
    )
    floor = BUNCH_FLOOR_CELLS.get(float(desc["gamma"]), 12.0)
    _ensure_domain(full, floor_cells=floor)
    chunks = full.chunk(tuple(int(s) for s in desc["chunk_sizes"]))
    return chunks[int(desc["chunk_index"])]


def make_sample(desc: dict, split: str, seed: int) -> sampling.Sample:
    if desc["kind"] == "scenario":
        return _scenario_sample(desc, split, seed)
    if desc["kind"] == "bunch":
        return _bunch_sample(desc, split, seed)
    raise ValueError(desc["kind"])


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def reference_info(module_path, hpp_path, python_reference_path) -> dict:
    return {
        "kind": "convkernel_cpp",
        "module": str(module_path),
        "sha256": sha256_file(module_path),
        "cpp_header": str(hpp_path),
        "cpp_header_sha256": sha256_file(hpp_path),
        "python_reference": str(python_reference_path),
        "python_reference_sha256": sha256_file(python_reference_path),
        "quadrature": {
            "policy": "adaptive error control at every point: order 4/tol 1e-7/max_level 5, order 6/tol 1e-6 above |u|=10; no tight near override",
            "low": {"order": 4, "tol": 1e-7, "max_level": 5, "max_order": 16},
            "high": {"order": 6, "tol": 1e-6, "max_level": 5, "max_order": 16},
            "convergence_evidence": "reference/benchmark.md: order 4/6 vs order 8 worst 4.6e-8 over aspects 0.1..10, distances 4..512, gamma 1/10/100",
        },
        "validation": "C++ extension vs reference/convkernel.py on the 10 golden tiles and the 225-case aspect grid; see cpp_validation.json",
    }


def scaled_quadrature(sample: sampling.Sample) -> dict:
    """Adaptive FP64 quadrature per source, without the very tight near rule.

    The C++ reference is error-controlled at every observation point.  The
    full-aspect convergence study (`reference/benchmark.md`) shows order 4/6
    against order 8 agrees to 4.6e-8 over aspects 0.1..10, distances 4..512 and
    gamma 1/10/100, so the order-4 (order-6 above |u|=10) adaptive rule at
    tol=1e-7 is used everywhere.  This keeps the target error far below the
    1e-3 accuracy gate (and below the golden-vector validation limit) while
    avoiding the minutes-per-tile cost of the tol=1e-12 near rule.
    """
    umax = 0.0
    p = sample.particles
    for i in range(sample.n):
        umax = max(umax, float(np.linalg.norm([p["ux"][i], p["uy"][i], p["uz"][i]])))
    if umax > 10.0:
        # measured: tol 1e-6 and tol 1e-7 agree to ~1e-5 on the worst
        # near-aligned gamma-100 tile (the max_order cap, not tol, sets the
        # floor), while tol 1e-6 roughly halves the cost.
        return {"order": 6, "tol": 1e-6, "max_level": 5, "max_order": 16}
    return {"order": 4, "tol": 1e-7, "max_level": 5, "max_order": 16}


def _pack_samples(samples, mod):
    quads = [scaled_quadrature(s) for s in samples]
    tiles = [reference_data.call_tile_field(mod, s, q) for s, q in zip(samples, quads)]
    targets = np.stack([reference_data.flatten_tile(t) for t in tiles], axis=0).astype(np.float64)
    s0 = np.stack([reference_data.preregistered_scales(t) for t in tiles], axis=0).astype(np.float64)
    dmin = np.array([reference_data._dmin_cells(s) for s in samples], dtype=np.float64)
    origins = np.stack([s.origin for s in samples]).astype(np.float64)
    dx = np.array([s.dx for s in samples], dtype=np.float64)
    ay = np.array([s.a_y for s in samples], dtype=np.float64)
    az = np.array([s.a_z for s in samples], dtype=np.float64)
    pos, u, w, idx = [], [], [], []
    for si, s in enumerate(samples):
        n = s.n
        p = s.particles
        pos.append(np.stack([p["x"], p["y"], p["z"]], axis=1))
        u.append(np.stack([p["ux"], p["uy"], p["uz"]], axis=1))
        w.append(np.asarray(p["w"], dtype=np.float64))
        idx.append(np.full(n, si, dtype=np.int64))
    return {
        "targets": targets,
        "quadrature_order": np.array([q["order"] for q in quads], dtype=np.int64),
        "s0": s0,
        "d_min_cells": dmin,
        "origin": origins,
        "dx": dx,
        "a_y": ay,
        "a_z": az,
        "pos": np.concatenate(pos, axis=0) if pos else np.zeros((0, 3)),
        "u": np.concatenate(u, axis=0) if u else np.zeros((0, 3)),
        "w": np.concatenate(w, axis=0) if w else np.zeros((0,)),
        "sample_index": np.concatenate(idx, axis=0) if idx else np.zeros((0,), dtype=np.int64),
    }


def _pack_task(args):
    split, seed, descs, module_path = args
    mod = load_fast_reference(module_path)
    samples = [make_sample(d, split, seed) for d in descs]
    return _pack_samples(samples, mod)


def _write_manifest(path: Path, manifest: dict):
    reference_data._write_manifest(path, manifest)


def generate_split(
    root,
    split,
    *,
    base_seed=20261101,
    module_path=None,
    hpp_path=None,
    python_reference_path=None,
    jobs=1,
    shard_size=128,
    resume=True,
) -> dict:
    from .fast_reference import find_extension

    module_path = Path(module_path or find_extension())
    split_dir = Path(root) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.json"
    seed = int(base_seed) + SPLIT_SEED_OFFSET[split]
    ref_info = reference_info(module_path, hpp_path, python_reference_path)

    if manifest_path.is_file() and resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["split"] != split or manifest["seed"] != seed:
            raise ValueError("existing manifest has a different split/seed")
        if manifest["reference"]["sha256"] != ref_info["sha256"]:
            raise ValueError("reference extension changed since the manifest was written")
    else:
        descs = descriptors_for_split(split)
        manifest = {
            "format": reference_data.DATASET_FORMAT,
            "version": 1,
            "generator": "scaled_v1",
            "split": split,
            "seed": seed,
            "base_seed": int(base_seed),
            "dtype": "float64",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reference": ref_info,
            "feature_kind": "fourier_v1",
            "aspect_range": [0.1, 10.0],
            "gamma_values": GAMMAS,
            "aspect_values": ASPECT_VALUES,
            "directions": DIRECTIONS,
            "position_modes": POSITION_MODES,
            "bunch_kinds": BUNCH_KINDS,
            "target_order": "c + 6*(i + nx*(j + ny*k))",
            "descriptors": descs,
            "shard_size": int(shard_size),
            "shards": [],
        }
        _write_manifest(manifest_path, manifest)

    descs = manifest["descriptors"]
    manifest["shards"] = [s for s in manifest.get("shards", []) if (split_dir / s["file"]).is_file()]
    done = {s["file"] for s in manifest["shards"]}
    tasks = []
    for start in range(0, len(descs), int(manifest["shard_size"])):
        fname = f"shard_{start // int(manifest['shard_size']):05d}.npz"
        if fname in done:
            continue
        chunk = descs[start : start + int(manifest["shard_size"])]
        tasks.append((start, fname, chunk))

    if jobs <= 1:
        for start, fname, chunk in tasks:
            packed = _pack_task((split, seed, chunk, str(module_path)))
            _store_shard(split_dir, manifest, manifest_path, start, fname, chunk, packed)
    elif tasks:
        with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            futures = {
                ex.submit(_pack_task, (split, seed, chunk, str(module_path))): (
                    start,
                    fname,
                    chunk,
                )
                for start, fname, chunk in tasks
            }
            for fut in as_completed(futures):
                start, fname, chunk = futures[fut]
                packed = fut.result()
                _store_shard(split_dir, manifest, manifest_path, start, fname, chunk, packed)

    manifest["shards"].sort(key=lambda s: s["descriptor_start"])
    manifest["n_tiles"] = len(descs)
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_manifest(manifest_path, manifest)
    return manifest


def _store_shard(split_dir, manifest, manifest_path, start, fname, chunk, packed):
    path = split_dir / fname
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **packed)
    os.replace(tmp, path)
    manifest["shards"] = [s for s in manifest["shards"] if s["file"] != fname]
    manifest["shards"].append(
        {
            "file": fname,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "n_samples": len(chunk),
            "descriptor_start": start,
        }
    )
    for i, d in enumerate(chunk):
        d["s0"] = [float(packed["s0"][i, 0]), float(packed["s0"][i, 1])]
        d["d_min_cells"] = float(packed["d_min_cells"][i])
        d["quadrature_order"] = int(packed["quadrature_order"][i])
        d["n_particles"] = int(np.sum(packed["sample_index"] == i))
    manifest["shards"].sort(key=lambda s: s["descriptor_start"])
    _write_manifest(manifest_path, manifest)


def generate_all(root, *, base_seed=20261101, jobs=1, shard_size=128, splits=SPLITS, **kw):
    out = {}
    for sp in splits:
        print(f"[scaled] {sp}", flush=True)
        out[sp] = generate_split(root, sp, base_seed=base_seed, jobs=jobs, shard_size=shard_size, **kw)
        print(
            f"[scaled] {sp}: {out[sp]['n_tiles']} tiles, {len(out[sp]['shards'])} shards",
            flush=True,
        )
    return out
