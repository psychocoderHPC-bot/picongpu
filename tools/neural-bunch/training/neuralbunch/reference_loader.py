"""Adapter around the FP64 reference implementation.

Expected reference API (``reference/convkernel.py``)::

    tile_field(x_tile, particles, dx, dy, dz) -> (8, 8, 4, 6) float64
    point_field(...)
    shape_field(...)

``x_tile`` is the tile cell origin (physical), ``particles`` is a structured
NumPy array with fields ``x, y, z, ux, uy, uz, w`` (physical coordinates,
``u = p/(m_e c)``, positive ``w``; the reference applies ``q/e = -w``), and
``dx, dy, dz`` are the cell sizes.  ``tile_field`` returns normalized fields
with ``q/e`` already applied.  A dict-of-arrays and an ``(N, 7)`` array are also
accepted by this adapter for convenience.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np

DEFAULT_REFERENCE = Path(
    os.environ.get(
        "NEURALBUNCH_REFERENCE",
        str(Path(__file__).resolve().parents[2] / "reference" / "convkernel.py"),
    )
)
PARTICLE_FIELDS = ["x", "y", "z", "ux", "uy", "uz", "w"]


class ReferenceUnavailable(RuntimeError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path=None):
    path = Path(path or DEFAULT_REFERENCE)
    if not path.is_file():
        raise ReferenceUnavailable(
            f"reference implementation not found at {path}; it is being implemented "
            f"separately. Set NEURALBUNCH_REFERENCE or pass --reference."
        )
    spec = importlib.util.spec_from_file_location("neuralbunch_convkernel", path)
    if spec is None or spec.loader is None:
        raise ReferenceUnavailable(f"cannot import reference module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "tile_field"):
        raise ReferenceUnavailable(f"{path} does not define tile_field")
    return mod


def load_standin():
    """Clearly marked analytic stand-in, for unit tests and smoke plumbing only."""
    from . import testing_standin

    return testing_standin


def structured_particles(particles):
    if isinstance(particles, np.ndarray) and particles.dtype.names:
        missing = [f for f in PARTICLE_FIELDS if f not in particles.dtype.names]
        if missing:
            raise ValueError(f"particle array missing fields {missing}")
        return particles
    n = len(particles["w"]) if isinstance(particles, dict) else len(particles)
    arr = np.empty(n, dtype=[(f, "f8") for f in PARTICLE_FIELDS])
    if isinstance(particles, dict):
        for f in PARTICLE_FIELDS:
            arr[f] = np.asarray(particles[f], dtype=np.float64)
    else:
        a = np.asarray(particles, dtype=np.float64)
        if a.shape != (n, 7):
            raise ValueError("particle array must be (N, 7)")
        for i, f in enumerate(PARTICLE_FIELDS):
            arr[f] = a[:, i]
    return arr


def _particle_list(particles):
    """The reference's native particle format: a list of
    ``{"x": (3,), "u": (3,), "w": scalar}`` mappings."""
    p = structured_particles(particles)
    return [
        {
            "x": np.array([p["x"][i], p["y"][i], p["z"][i]], dtype=np.float64),
            "u": np.array([p["ux"][i], p["uy"][i], p["uz"][i]], dtype=np.float64),
            "w": float(p["w"][i]),
        }
        for i in range(len(p))
    ]


def call_tile_field(mod, sample, quadrature=None):
    dx = float(sample.dx)
    dy = float(sample.a_y * sample.dx)
    dz = float(sample.a_z * sample.dx)
    origin = np.asarray(sample.origin, dtype=np.float64)
    kwargs = dict(quadrature or {})
    out = mod.tile_field(origin, _particle_list(sample.particles), dx, dy, dz, **kwargs)
    out = np.asarray(out, dtype=np.float64)
    if out.size != 8 * 8 * 4 * 6:
        raise ValueError(f"reference tile_field returned shape {out.shape}, expected (8,8,4,6)")
    return out.reshape(8, 8, 4, 6)


def flatten_tile(arr) -> np.ndarray:
    """(8,8,4,6)[i,j,k,c] -> artifact order ``c + 6*(i + nx*(j + ny*k))``."""
    arr = np.asarray(arr)
    return arr.transpose(2, 1, 0, 3).reshape(-1).astype(np.float64)


def target_from_sample(mod, sample, quadrature=None) -> np.ndarray:
    return flatten_tile(call_tile_field(mod, sample, quadrature))
