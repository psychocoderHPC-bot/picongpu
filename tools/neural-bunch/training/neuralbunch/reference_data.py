"""Reference-target generation: on-demand, sharded, FP64, resumable.

Layout of a dataset root::

    <root>/<split>/manifest.json
    <root>/<split>/shard_00000.npz

Each shard stores particle arrays, tile geometry, the float64 target tiles in
artifact order, and a sample index.  The manifest records the planned
descriptors, seeds, reference file hash and per-shard SHA-256, so an interrupted
run resumes without recomputing finished shards.  Only one shard is held in
memory at a time.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from . import sampling
from .features import compute_features
from .reference_loader import (
    DEFAULT_REFERENCE,
    ReferenceUnavailable,
    call_tile_field,
    flatten_tile,
    load_module,
    load_standin,
    sha256_file,
)

DATASET_FORMAT = "picongpu-neural-bunch-dataset"
SHARD_SIZE_DEFAULT = 8
#: validated fast path: cross-checked against order=6,tol=1e-9,max_level=5
DEFAULT_QUADRATURE = {"order": 4, "tol": 1e-7, "max_level": 4}
HIGH_GAMMA_QUADRATURE = {"order": 6, "tol": 1e-9, "max_level": 5}
NEAR_HIGH_QUADRATURE = {"order": 8, "tol": 1e-12, "max_level": 7}
HIGH_GAMMA_THRESHOLD = 10.0
HIGH_NEAR_THRESHOLD = 5.0
NEAR_DISTANCE_CELLS = 10.0


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_descriptors(split, n_configs, samples_per_config, include_bunches, bunch_kinds,
                     n_particles, bunch_gammas):
    descs = []
    configs = sampling.scenario_keys(split, limit=n_configs)
    for cfg in configs:
        for si in range(int(samples_per_config)):
            descs.append({"kind": "scenario", "config": list(cfg), "sample_index": si})
    if include_bunches:
        for bi, kind in enumerate(bunch_kinds):
            for gi, gamma in enumerate(bunch_gammas):
                descs.append(
                    {
                        "kind": "bunch",
                        "bunch_kind": kind,
                        "gamma": float(gamma),
                        "n_particles": int(n_particles),
                        "sample_index": bi * len(bunch_gammas) + gi,
                    }
                )
    return descs


def sample_from_descriptor(split, seed, desc, aspect_range=(0.1, 10.0)) -> sampling.Sample:
    if desc["kind"] == "scenario":
        return sampling.make_scenario_sample(split, seed, tuple(desc["config"]),
                                            desc["sample_index"], aspect_range=aspect_range)
    if desc["kind"] == "bunch":
        return sampling.sample_bunch(
            desc["bunch_kind"],
            split=split,
            seed=seed + 100000 + int(desc["sample_index"]),
            n_particles=desc["n_particles"],
            gamma=desc["gamma"],
        )
    raise ValueError(desc["kind"])


def generate_target(sample, mod, quadrature=None):
    """On-demand FP64 target for one sample, shape ``(1536,)``."""
    return flatten_tile(call_tile_field(mod, sample, quadrature))


def preregistered_scales(tile):
    """Per-group dimensional norm scale S0 from the reference tile itself.

    ``S0_G = sqrt(N_G/3) * max_cell ||G_cell||`` where ``G_cell`` is the
    3-component vector at one staggered output point and ``N_G = 768``.  For a
    spatially uniform field this equals ``||A_ref||``; for a field that peaks
    near the source it is larger, so weak/cancelling tiles are normalized by a
    preregistered configuration scale instead of by their own tiny norm.
    """
    e = np.asarray(tile)[..., :3]
    b = np.asarray(tile)[..., 3:]
    n_e = np.linalg.norm(e, axis=-1).max()
    n_b = np.linalg.norm(b, axis=-1).max()
    return np.array([np.sqrt(e.size / 3.0) * n_e, np.sqrt(b.size / 3.0) * n_b])


def _resolve_reference(reference_path, allow_standin, quadrature):
    if allow_standin:
        mod = load_standin()
        return mod, {"kind": "analytic_standin", "file": None, "sha256": None,
                     "warning": mod.STANDIN_WARNING, "quadrature": dict(quadrature)}
    mod = load_module(reference_path)
    path = Path(reference_path) if reference_path else DEFAULT_REFERENCE
    return mod, {"kind": "convkernel", "file": str(path), "sha256": sha256_file(path),
                 "quadrature": dict(quadrature)}


def _dmin_cells(sample):
    pts = sampling.output_points(sample.origin, sample.dx, sample.a_y, sample.a_z)
    m = np.inf
    for k in range(sample.n):
        p = np.array([sample.particles["x"][k], sample.particles["y"][k], sample.particles["z"][k]])
        m = min(m, float(np.min(np.linalg.norm(pts - p, axis=1)) / sample.dx))
    return m


def _sample_quadrature(sample, quadrature):
    if quadrature is not None:
        return dict(quadrature)
    umax = 0.0
    for i in range(sample.n):
        umax = max(umax, float(np.linalg.norm([
            sample.particles["ux"][i], sample.particles["uy"][i],
            sample.particles["uz"][i]])))
    dmin = _dmin_cells(sample)
    if umax > HIGH_NEAR_THRESHOLD and dmin < NEAR_DISTANCE_CELLS:
        return dict(NEAR_HIGH_QUADRATURE)
    if umax > HIGH_GAMMA_THRESHOLD:
        return dict(HIGH_GAMMA_QUADRATURE)
    return dict(DEFAULT_QUADRATURE)


def _pack_shard(samples, mod, quadrature):
    quads = [_sample_quadrature(s, quadrature) for s in samples]
    tiles = [call_tile_field(mod, s, q) for s, q in zip(samples, quads)]
    targets = np.stack([flatten_tile(t) for t in tiles], axis=0).astype(np.float64)
    s0 = np.stack([preregistered_scales(t) for t in tiles], axis=0).astype(np.float64)
    dmin = np.array([_dmin_cells(s) for s in samples], dtype=np.float64)
    origins = np.stack([s.origin for s in samples]).astype(np.float64)
    dx = np.array([s.dx for s in samples], dtype=np.float64)
    ay = np.array([s.a_y for s in samples], dtype=np.float64)
    az = np.array([s.a_z for s in samples], dtype=np.float64)
    pos, u, w, idx = [], [], [], []
    off = 0
    for si, s in enumerate(samples):
        n = s.n
        p = s.particles
        pos.append(np.stack([p["x"], p["y"], p["z"]], axis=1))
        u.append(np.stack([p["ux"], p["uy"], p["uz"]], axis=1))
        w.append(np.asarray(p["w"], dtype=np.float64))
        idx.append(np.full(n, si, dtype=np.int64))
        off += n
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


def _write_manifest(path: Path, manifest: dict):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_dataset(root, split, seed=0, n_configs=None, samples_per_config=1,
                  include_bunches=False, bunch_kinds=None, n_particles=64,
                  bunch_gammas=(2.0, 10.0, 100.0), shard_size=SHARD_SIZE_DEFAULT,
                  reference_path=None, quadrature=None, aspect_range=(0.1, 10.0),
                  _allow_standin=False) -> dict:
    """Generate (or resume) one split.  Returns the manifest."""
    split_dir = Path(root) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.json"

    quad = None if quadrature is None else dict(quadrature)
    pol = {"policy": "order 8 (tight) for |u|>5 and d_min<10; order 6 for |u|>10; else order 4",
           "low": DEFAULT_QUADRATURE, "high": HIGH_GAMMA_QUADRATURE,
           "near_high": NEAR_HIGH_QUADRATURE}
    mod, ref_info = _resolve_reference(reference_path, _allow_standin, quad or pol)
    if bunch_kinds is None:
        bunch_kinds = list(sampling.BUNCH_KINDS)

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split") != split or manifest.get("seed") != int(seed):
            raise ValueError(f"{manifest_path} exists with a different split/seed; refusing to mix")
        if ref_info["kind"] == "convkernel" and manifest["reference"].get("sha256") not in (None, ref_info["sha256"]):
            raise ValueError("reference implementation changed since the manifest was written")
    else:
        descs = make_descriptors(
            split, n_configs, samples_per_config, include_bunches, bunch_kinds,
            n_particles, bunch_gammas,
        )
        manifest = {
            "format": DATASET_FORMAT,
            "version": 1,
            "split": split,
            "seed": int(seed),
            "dtype": "float64",
            "created_utc": _now(),
            "reference": ref_info,
            "feature_kind": "fourier_v1",
            "aspect_range": [float(aspect_range[0]), float(aspect_range[1])],
            "target_order": "c + 6*(i + nx*(j + ny*k))",
            "descriptors": descs,
            "shard_size": int(shard_size),
            "shards": [],
        }
        _write_manifest(manifest_path, manifest)

    descs = manifest["descriptors"]
    manifest["shards"] = [s for s in manifest.get("shards", []) if (split_dir / s["file"]).is_file()]
    done = {s["file"] for s in manifest["shards"]}
    for start in range(0, len(descs), int(manifest["shard_size"])):
        fname = f"shard_{start // int(manifest['shard_size']):05d}.npz"
        if fname in done:
            continue
        chunk = descs[start:start + int(manifest["shard_size"])]
        samples = [sample_from_descriptor(split, seed, d, aspect_range) for d in chunk]
        packed = _pack_shard(samples, mod, quad)
        path = split_dir / fname
        tmp = path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **packed)
        os.replace(tmp, path)
        manifest["shards"].append(
            {
                "file": fname,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "n_samples": len(samples),
                "descriptor_start": start,
            }
        )
        for i, d in enumerate(chunk):
            d["s0"] = [float(packed["s0"][i, 0]), float(packed["s0"][i, 1])]
            d["d_min_cells"] = float(packed["d_min_cells"][i])
            d["quadrature_order"] = int(packed["quadrature_order"][i])
        manifest["shards"].sort(key=lambda s: s["descriptor_start"])
        _write_manifest(manifest_path, manifest)
    manifest["created_utc"] = manifest["created_utc"] if manifest["shards"] else _now()
    _write_manifest(manifest_path, manifest)
    return manifest


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
class SplitDataset:
    """In-memory view of one generated split with precomputed model inputs."""

    def __init__(self, split_dir, feature_kind=None):
        self.split_dir = Path(split_dir)
        self.manifest = json.loads((self.split_dir / "manifest.json").read_text(encoding="utf-8"))
        self.feature_kind = feature_kind or self.manifest.get("feature_kind", "fourier_v1")
        self._load()

    def _load(self):
        feats, src_w, tile_index, origins, dxs, ays, azs, targets = [], [], [], [], [], [], [], []
        s0s, dmins, poss, us, ws, sidxs = [], [], [], [], [], []
        n_prev = 0
        for sh in self.manifest["shards"]:
            path = self.split_dir / sh["file"]
            if sha256_file(path) != sh["sha256"]:
                raise ValueError(f"shard hash mismatch: {path}")
            z = np.load(path)
            n_samples = int(z["targets"].shape[0])
            ay = z["a_y"]
            az = z["a_z"]
            pos = z["pos"]
            u = z["u"]
            w = z["w"]
            si = z["sample_index"].astype(np.int64)
            # zero-weight padding/masking happens before any transform
            keep = w != 0
            origin_per = z["origin"][si[keep]]
            r = (pos[keep] - origin_per) / z["dx"][si[keep]][:, None]
            f = compute_features(self.feature_kind, r, u[keep], ay[si[keep]], az[si[keep]])
            feats.append(f)
            src_w.append(-w[keep])  # q/e = -w
            tile_index.append(si[keep] + n_prev)
            origins.append(z["origin"])
            dxs.append(z["dx"])
            ays.append(ay)
            azs.append(az)
            targets.append(z["targets"])
            s0s.append(z["s0"] if "s0" in z else np.ones((n_samples, 2)))
            dmins.append(z["d_min_cells"] if "d_min_cells" in z else np.zeros(n_samples))
            poss.append(pos)
            us.append(u)
            ws.append(w)
            sidxs.append(si + n_prev)
            n_prev += n_samples
        self.feats = np.concatenate(feats) if feats else np.zeros((0, 69))
        self.src_w = np.concatenate(src_w) if src_w else np.zeros((0,))
        self.tile_index = np.concatenate(tile_index) if tile_index else np.zeros((0,), dtype=np.int64)
        self.origin = np.concatenate(origins) if origins else np.zeros((0, 3))
        self.dx = np.concatenate(dxs) if dxs else np.zeros((0,))
        self.a_y = np.concatenate(ays) if ays else np.zeros((0,))
        self.a_z = np.concatenate(azs) if azs else np.zeros((0,))
        self.targets = np.concatenate(targets) if targets else np.zeros((0, 1536))
        self.s0 = np.concatenate(s0s) if s0s else np.ones((0, 2))
        self.d_min_cells = np.concatenate(dmins) if dmins else np.zeros((0,))
        self.pos = np.concatenate(poss) if poss else np.zeros((0, 3))
        self.u = np.concatenate(us) if us else np.zeros((0, 3))
        self.w = np.concatenate(ws) if ws else np.zeros((0,))
        self.sample_index = np.concatenate(sidxs) if sidxs else np.zeros((0,), dtype=np.int64)
        self.n_tiles = len(self.targets)
        self.descriptors = self.manifest.get("descriptors", [])

    @property
    def reference_kind(self):
        return self.manifest["reference"]["kind"]


def load_split(root, split, feature_kind=None) -> SplitDataset:
    return SplitDataset(Path(root) / split, feature_kind=feature_kind)


def available_splits(root):
    root = Path(root)
    return [p.name for p in root.iterdir() if (p / "manifest.json").is_file()] if root.is_dir() else []
