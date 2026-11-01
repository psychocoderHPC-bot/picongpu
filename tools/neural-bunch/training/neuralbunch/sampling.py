"""Deterministic physical scenario and single-source sampling.

A ``Sample`` is one tile: a cell origin, ``dx`` (the normalization length and
the x cell size), the aspects ``a_y = dy/dx`` and ``a_z = dz/dx``, and a particle
bunch.  Particle arrays use

* ``x, y, z``  physical positions (same length unit as ``dx``),
* ``ux, uy, uz``  momentum over ``m_e c``,
* ``w``  positive weight magnitude; the model and reference apply ``q/e = -w``.

Physical configurations (gamma regime, direction class, aspect bucket, position
regime) are assigned to train/val/test splits deterministically, so the splits
never share a configuration.  Row seeds are distinct per split.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from .model import TILE, DEFAULT_AXES

OUT_DIM = TILE[0] * TILE[1] * TILE[2] * 6
NX, NY, NZ = TILE

# gamma "regimes": two beta endpoints and three gamma endpoints
GAMMA_SPECS = [
    ("beta0.01", "beta", 0.01),
    ("beta0.1", "beta", 0.1),
    ("gamma2", "gamma", 2.0),
    ("gamma10", "gamma", 10.0),
    ("gamma100", "gamma", 100.0),
    ("gamma100thin", "gamma", 100.0),
]
DIRECTIONS = ["random", "axis_x", "axis_y", "axis_z", "diag_p", "diag_m"]
POSITION_MODES = ["random", "subcell", "near", "far", "translation", "scaled", "thin"]
BUNCH_KINDS = [
    "gaussian",
    "ellipsoidal",
    "flat_top",
    "two_lobe",
    "counterstreaming",
    "crossing",
    "mixed_energy",
    "unequal_weights",
]
ASPECT_ANCHORS = np.asarray(DEFAULT_AXES)
SPLITS = ("train", "val", "test")


@dataclass
class Sample:
    origin: np.ndarray
    dx: float
    a_y: float
    a_z: float
    particles: Dict[str, np.ndarray]
    meta: Dict[str, object] = field(default_factory=dict)

    def copy_with(self, particles=None) -> "Sample":
        return Sample(
            self.origin.copy(),
            float(self.dx),
            float(self.a_y),
            float(self.a_z),
            particles if particles is not None else {k: v.copy() for k, v in self.particles.items()},
            dict(self.meta),
        )

    @property
    def n(self) -> int:
        return len(self.particles["w"])

    def chunk(self, sizes: Sequence[int]) -> List["Sample"]:
        """Split the bunch into chunks, preserving geometry and total sum."""
        out = []
        start = 0
        i = 0
        n = self.n
        while start < n:
            size = max(1, int(sizes[i % len(sizes)]))
            stop = min(n, start + size)
            part = {k: v[start:stop].copy() for k, v in self.particles.items()}
            out.append(self.copy_with(part))
            start = stop
            i += 1
        return out


def _stable_seed(*parts) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest(), "little")


def assign_split(key: str) -> str:
    """Deterministic physical-configuration -> split mapping (not row based)."""
    return SPLITS[zlib.crc32(key.encode("utf-8")) % 3]


def _direction(rng, kind: str) -> np.ndarray:
    if kind == "random":
        v = rng.normal(size=3)
        return v / np.linalg.norm(v)
    if kind == "axis_x":
        return np.array([1.0, 0.0, 0.0])
    if kind == "axis_y":
        return np.array([0.0, 1.0, 0.0])
    if kind == "axis_z":
        return np.array([0.0, 0.0, 1.0])
    if kind == "diag_p":
        return np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    if kind == "diag_m":
        return np.array([1.0, -1.0, 1.0]) / np.sqrt(3.0)
    raise ValueError(kind)


def _momenta(rng, gamma_name: str, kind: str, gamma_value: float):
    if gamma_name.startswith("beta"):
        beta = gamma_value
        gamma = 1.0 / np.sqrt(1.0 - beta * beta)
    else:
        gamma = gamma_value
        beta = np.sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)))
    unit = _direction(rng, kind)
    u = beta * gamma * unit
    return u, gamma


def output_points(origin, dx, a_y, a_z) -> np.ndarray:
    """The 1536 staggered output points, physical coordinates, component-fastest."""
    i = np.arange(NX)
    j = np.arange(NY)
    k = np.arange(NZ)
    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")
    # Yee-style stagger in cell units, ordered (Ex,Ey,Ez,Bx,By,Bz)
    stag = {
        "Ex": (0.5, 1.0, 1.0),
        "Ey": (1.0, 0.5, 1.0),
        "Ez": (1.0, 1.0, 0.5),
        "Bx": (1.0, 0.5, 0.5),
        "By": (0.5, 1.0, 0.5),
        "Bz": (0.5, 0.5, 1.0),
    }
    pts = []
    for c in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
        ox, oy, oz = stag[c]
        x = origin[0] + (ii + ox) * dx
        y = origin[1] + (jj + oy) * a_y * dx
        z = origin[2] + (kk + oz) * a_z * dx
        pts.append(np.stack([x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")], axis=-1))
    return np.concatenate(pts, axis=0)


def _particle_dict(pos, u, w):
    return {
        "x": np.asarray(pos[:, 0], dtype=np.float64),
        "y": np.asarray(pos[:, 1], dtype=np.float64),
        "z": np.asarray(pos[:, 2], dtype=np.float64),
        "ux": np.asarray(u[:, 0], dtype=np.float64),
        "uy": np.asarray(u[:, 1], dtype=np.float64),
        "uz": np.asarray(u[:, 2], dtype=np.float64),
        "w": np.asarray(w, dtype=np.float64),
    }


def _place_single(rng, origin, dx, ay, az, mode):
    """Place one source in the neural domain, ``d_min >= 4`` cells.

    The source is offset from a random staggered output point by a distance in
    ``[4, 512]`` cells (near ``[4,8]``, far ``[256,512]``, otherwise
    log-uniform).  Placements that fall inside the TSC support box of any
    staggered output point are rejected by increasing the offset, so the target
    stays in the reference's smooth far branch (its near branch is much slower
    and less accurate for extreme aspects).
    """
    if mode == "scaled":
        s = int(rng.integers(-3, 4))
        dx = float(dx * (2.0 ** s))
    if mode == "translation":
        origin = origin + rng.uniform(-1000.0, 1000.0, size=3) * dx
    pts = output_points(origin, dx, ay, az)
    cells = np.array([dx, ay * dx, az * dx])
    p = pts[rng.integers(0, len(pts))]
    if mode == "near":
        d_cells = rng.uniform(4.0, 8.0)
    elif mode == "far":
        d_cells = rng.uniform(256.0, 512.0)
    else:
        d_cells = float(np.exp(rng.uniform(np.log(4.0), np.log(512.0))))
    pos, dmin = p, 0.0
    d_try = d_cells
    for _ in range(96):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if mode == "thin":
            v[1:] *= 0.05
            v /= np.linalg.norm(v)
        candidate = p + d_try * dx * v
        if mode == "subcell":
            t = rng.normal(size=3)
            t -= v * np.dot(t, v)
            t /= max(np.linalg.norm(t), 1e-300)
            candidate = candidate + rng.uniform(0.0, 1.2) * dx * t
        cc = (pts - candidate) / cells
        # the reference's own near criterion, so the fast fixed-rule branch is used
        excess = np.maximum(np.abs(cc) - 1.5, 0.0)
        in_support = np.any(np.linalg.norm(excess, axis=1) < 1.0)
        dd = float(np.min(np.linalg.norm(pts - candidate, axis=1)) / dx)
        if dd >= 4.0 and not in_support:
            pos, dmin = candidate, dd
            break
        d_try = min(512.0, d_try * 1.25)
    else:
        pos = p + 256.0 * dx * np.array([1.0, 0.0, 0.0])
        dmin = float(np.min(np.linalg.norm(pts - pos, axis=1)) / dx)
    return pos, origin, dx, dmin


def make_sample(split: str, seed: int, gamma_name: str, direction: str,
                position_mode: str, sample_index: int = 0,
                aspect_range=(0.1, 10.0), aspects=None) -> Sample:
    """Build one deterministic single-source tile sample."""
    rng = np.random.default_rng(
        _stable_seed("sample", split, seed, gamma_name, direction, position_mode, sample_index)
    )
    gname = gamma_name
    gval = dict((n, v) for n, _, v in GAMMA_SPECS)[gname]
    u, gamma = _momenta(rng, gname, direction, gval)
    lo, hi = np.log10(aspect_range[0]), np.log10(aspect_range[1])
    if aspects is None:
        a_y = float(10.0 ** rng.uniform(lo, hi))
        a_z = float(10.0 ** rng.uniform(lo, hi))
    else:
        a_y, a_z = float(aspects[0]), float(aspects[1])
    dx = float(2.0 ** rng.integers(-2, 3))
    origin = rng.uniform(-30.0, 30.0, size=3) * dx
    pos, origin, dx, dmin = _place_single(rng, origin, dx, a_y, a_z, position_mode)
    particles = _particle_dict(pos[None, :], u[None, :], np.array([1.0]))
    return Sample(
        origin=np.asarray(origin, dtype=np.float64),
        dx=dx,
        a_y=a_y,
        a_z=a_z,
        particles=particles,
        meta={
            "split": split,
            "seed": int(seed),
            "gamma_name": gname,
            "gamma": float(gamma),
            "direction": direction,
            "position_mode": position_mode,
            "d_min_cells": float(dmin),
            "sample_index": int(sample_index),
        },
    )


def config_key(gamma_name, direction, aspect_bucket, position_mode):
    return f"{gamma_name}|{direction}|{aspect_bucket}|{position_mode}"


def _aspect_bucket(a_y: float) -> int:
    la = np.log10(a_y)
    return int(np.clip(np.searchsorted(np.log10(ASPECT_ANCHORS), la), 0, len(ASPECT_ANCHORS) - 1))


def iter_configs(split: str):
    """Yield the physical configurations assigned to ``split``."""
    for gname, _, _ in GAMMA_SPECS:
        for direction in DIRECTIONS:
            for ai in range(len(ASPECT_ANCHORS)):
                for pm in POSITION_MODES:
                    key = config_key(gname, direction, ai, pm)
                    if assign_split(key) == split:
                        yield (gname, direction, ai, pm)


def scenario_keys(split: str, limit: Optional[int] = None) -> List[tuple]:
    keys = list(iter_configs(split))
    if limit is not None and int(limit) < len(keys):
        rng = np.random.default_rng(1234567 + zlib.crc32(split.encode("utf-8")))
        idx = rng.permutation(len(keys))[: int(limit)]
        keys = [keys[i] for i in sorted(idx)]
    return keys


def make_scenario_sample(split: str, seed: int, config, sample_index: int = 0,
                         aspect_range=(0.1, 10.0)) -> Sample:
    gname, direction, ai, pm = config
    rng = np.random.default_rng(_stable_seed("aspect", split, seed, config))
    # Continuous aspect, not only the anchor, inside the bucket neighbourhood.
    lo = np.log10(ASPECT_ANCHORS[max(0, ai - 1)])
    hi = np.log10(ASPECT_ANCHORS[min(len(ASPECT_ANCHORS) - 1, ai + 1)])
    a_y = float(np.clip(10.0 ** rng.uniform(lo, hi), aspect_range[0], aspect_range[1]))
    lo_r, hi_r = np.log10(aspect_range[0]), np.log10(aspect_range[1])
    a_z = float(10.0 ** rng.uniform(lo_r, hi_r))
    # aspects are fixed *before* placement, otherwise the source can end up
    # inside the support box after the aspect is changed
    sample = make_sample(split, seed, gname, direction, pm, sample_index,
                         aspect_range=aspect_range, aspects=(a_y, a_z))
    sample.meta["aspect_bucket"] = ai
    return sample


# --------------------------------------------------------------------------
# multi-particle bunches (fine-tuning / sum-validation)
# --------------------------------------------------------------------------
def sample_bunch(kind: str, split: str = "train", seed: int = 0,
                 n_particles: int = 64, gamma: float = 10.0,
                 a_y: float = 1.0, a_z: float = 1.0, dx: float = 1.0) -> Sample:
    if kind not in BUNCH_KINDS:
        raise ValueError(f"unknown bunch kind {kind!r}")
    rng = np.random.default_rng(_stable_seed("bunch", split, seed, kind, n_particles))
    origin = rng.uniform(-20.0, 20.0, size=3) * dx
    # bunch centroid outside the tile, at ~12 cells from a staggered point, so
    # every source stays in the neural domain (d_min >= 4 cells)
    pts = output_points(origin, dx, a_y, a_z)
    p = pts[rng.integers(0, len(pts))]
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    centre = p + 12.0 * dx * v
    sigma = np.array([0.5, 0.5 * a_y, 0.5 * a_z]) * dx
    n = int(n_particles)
    u = np.zeros((n, 3))
    w = np.ones(n)
    beta = np.sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)))
    ux0 = beta * gamma

    if kind in ("gaussian", "unequal_weights", "mixed_energy"):
        pos = centre + rng.normal(size=(n, 3)) * sigma
    elif kind == "ellipsoidal":
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        r = rng.uniform(0, 1, size=(n, 1)) ** (1.0 / 3.0)
        pos = centre + v * r * sigma
    elif kind == "flat_top":
        pos = centre + rng.uniform(-1, 1, size=(n, 3)) * sigma
    elif kind == "two_lobe":
        off = np.array([2.0 * a_y * dx, 0.0, 0.0])
        side = rng.integers(0, 2, size=n)[:, None]
        pos = centre + np.where(side == 0, -off, off) + rng.normal(size=(n, 3)) * sigma * 0.4
    elif kind == "counterstreaming":
        pos = centre + rng.normal(size=(n, 3)) * sigma
        side = np.where(rng.integers(0, 2, size=n) == 0, -1.0, 1.0)
        u[:, 0] = side * ux0
    elif kind == "crossing":
        pos = centre + rng.normal(size=(n, 3)) * sigma
        side = rng.integers(0, 2, size=n)
        u[:, 0] = np.where(side == 0, ux0, 0.0)
        u[:, 1] = np.where(side == 1, ux0, 0.0)
    elif kind == "unequal_weights":
        w = rng.uniform(0.1, 2.0, size=n)
    else:
        pos = centre + rng.normal(size=(n, 3)) * sigma

    if kind == "mixed_energy":
        g = 10.0 ** rng.uniform(0.0, 2.0, size=n)
        b = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (g * g)))
        u[:, 0] = b * g * rng.choice([-1.0, 1.0], size=n)
    elif kind in ("gaussian", "ellipsoidal", "flat_top", "two_lobe", "unequal_weights"):
        u[:, 0] = ux0
    elif kind == "counterstreaming":
        pass
    elif kind == "crossing":
        pass
    # zero out tiny numerical u_y/u_z
    return Sample(
        origin=np.asarray(origin, dtype=np.float64),
        dx=float(dx),
        a_y=float(a_y),
        a_z=float(a_z),
        particles=_particle_dict(pos, u, w),
        meta={"split": split, "seed": int(seed), "bunch_kind": kind, "gamma": float(gamma)},
    )


def chunk_sizes(n: int, sizes=(1, 17, 1024, 5)) -> List[int]:
    """Variable chunk sizes cycling through ``sizes`` until ``n`` is covered."""
    out = []
    total = 0
    i = 0
    while total < n:
        s = int(sizes[i % len(sizes)])
        s = min(s, n - total)
        out.append(s)
        total += s
        i += 1
    return out
