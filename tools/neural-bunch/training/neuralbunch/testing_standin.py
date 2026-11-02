"""CLEARLY MARKED ANALYTIC STAND-IN -- NOT A PHYSICS REFERENCE.

This module exists only so that unit tests and the plumbing smoke run can
execute before (or without) ``reference/convkernel.py``.  It returns a smooth,
finite, deterministic surrogate field.  It is not Maxwell, it is not the TSC
shape integral, and **any error measured against it is not validation of the
neural model**.  Datasets generated with it carry
``reference.kind = "analytic_standin"`` in their manifest.
"""

from __future__ import annotations

import numpy as np

STANDIN_WARNING = "ANALYTIC STAND-IN: not a physics reference; not validation"
_COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
_STAG = {
    "Ex": (0.5, 1.0, 1.0),
    "Ey": (1.0, 0.5, 1.0),
    "Ez": (1.0, 1.0, 0.5),
    "Bx": (1.0, 0.5, 0.5),
    "By": (0.5, 1.0, 0.5),
    "Bz": (0.5, 0.5, 1.0),
}


def _points(x_tile, dx, dy, dz):
    i, j, k = np.meshgrid(np.arange(8), np.arange(8), np.arange(4), indexing="ij")
    pts = []
    for c in _COMPONENTS:
        ox, oy, oz = _STAG[c]
        x = x_tile[0] + (i + ox) * dx
        y = x_tile[1] + (j + oy) * dy
        z = x_tile[2] + (k + oz) * dz
        pts.append(np.stack([x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")], -1))
    return np.concatenate(pts, 0)


def _iter_particles(particles):
    if isinstance(particles, dict):
        for p in range(len(particles["w"])):
            yield (
                np.array([particles["x"][p], particles["y"][p], particles["z"][p]]),
                np.array([particles["ux"][p], particles["uy"][p], particles["uz"][p]]),
                float(particles["w"][p]),
            )
    elif isinstance(particles, np.ndarray) and particles.dtype.names:
        for p in particles:
            yield (
                np.array([p["x"], p["y"], p["z"]]),
                np.array([p["ux"], p["uy"], p["uz"]]),
                float(p["w"]),
            )
    else:
        for p in particles:
            yield np.asarray(p["x"], dtype=np.float64), np.asarray(p["u"], dtype=np.float64), float(p["w"])


def tile_field(x_tile, particles, dx, dy, dz, **kwargs):
    x_tile = np.asarray(x_tile, dtype=np.float64)
    pts = _points(x_tile, dx, dy, dz)
    out = np.zeros((6, 256), dtype=np.float64)
    for pos, u, w in _iter_particles(particles):
        if w == 0.0:
            continue
        d = (np.linalg.norm(pts - pos, axis=1) / dx).reshape(6, 256)
        kern = np.exp(-0.5 * (d / 3.0) ** 2)
        gamma = np.sqrt(1.0 + np.dot(u, u))
        # crude anisotropic, charge-signed response (purely a stand-in)
        e = -w * kern / (1.0 + 0.1 * d)
        out[0] += e[0] * (1.0 + 0.1 * u[0])
        out[1] += e[1] * (0.2 * u[1] + 0.01 * gamma)
        out[2] += e[2] * (0.2 * u[2] - 0.01 * gamma)
        b = -w * kern * 0.05 * (u[0] + u[1] + u[2])
        out[3] += b[3] * (1.0 + 0.1 * u[0])
        out[4] += b[4] * (0.5 + 0.1 * u[1])
        out[5] += b[5] * (0.5 + 0.1 * u[2])
    return out.reshape(6, 4, 8, 8).transpose(3, 2, 1, 0)
