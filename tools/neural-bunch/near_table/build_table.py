#!/usr/bin/env python3
"""GPU generator for the TSC-convolved moving-charge near-field table (T145).

For fixed velocity u and cell aspect cells, stores the six per-unit-charge
field components  K(r) = int W(s) Kpoint((r-s) cells, u) ds^3  on a graded 3D
displacement grid r (cell units).  Panel/transform quadrature structure is taken
from the validated reference (indfield.py); the tensor-product reduction runs on
the GPU (cuPy).  Writes float64 binary + JSON metadata (no pickle).

K(r) is an ODD function of r (E,B odd in R, W even) => K(-r) = -K(r), K(0) = 0.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indfield as ref  # validated reference (TSC split/transform + point field)


def _components(Rx, Ry, Rz, beta, inv_g2, uu2):
    import cupy as cp

    r2 = Rx * Rx + Ry * Ry + Rz * Rz
    if uu2 == 0.0:
        r3 = r2 * cp.sqrt(r2)
        Ex, Ey, Ez = Rx / r3, Ry / r3, Rz / r3
        z = cp.zeros_like(Ex)
        return [Ex, Ey, Ez, z, z, z]
    rbeta = beta[0] * Rx + beta[1] * Ry + beta[2] * Rz
    D = r2 * inv_g2 + rbeta * rbeta
    D15 = D * cp.sqrt(D)
    Ex = inv_g2 * Rx / D15
    Ey = inv_g2 * Ry / D15
    Ez = inv_g2 * Rz / D15
    Bx = beta[1] * Ez - beta[2] * Ey
    By = beta[2] * Ex - beta[0] * Ez
    Bz = beta[0] * Ey - beta[1] * Ex
    return [Ex, Ey, Ez, Bx, By, Bz]


def _velocity(u):
    uu2 = float(np.dot(u, u))
    if uu2 == 0.0:
        return (0.0, 0.0, 0.0), 1.0, 0.0
    inv_g2 = 1.0 / (1.0 + uu2)
    beta = (np.asarray(u) / np.sqrt(1.0 + uu2)).tolist()
    return tuple(beta), inv_g2, uu2


def axis_rule(r, singular, order, refine):
    panels = ref._axis_panels(float(r), bool(singular))
    s, w = ref._axis_rule(panels, int(order), int(refine))
    w = w * ref.tsc_shape(s)
    return np.ascontiguousarray(s, dtype=np.float64), np.ascontiguousarray(w, dtype=np.float64)


def _reduce3(F, Wx, Wy, Wz):
    F = F * Wz[None, None, :]
    F = F.sum(axis=2)
    F = F * Wy[None, :]
    F = F.sum(axis=1)
    F = F * Wx
    return F.sum()


def _reduce_batch(F, Wx, Wy, Wz):
    F = F * Wz[None, None, None, :]
    F = F.sum(axis=3)
    F = F * Wy[None, None, :]
    F = F.sum(axis=2)
    F = F * Wx[None, :]
    return F.sum(axis=1)


def build_table(u, cells, grid, order_near=12, refine_near=3, order_far=12, refine_far=0,
                batch=256, verbose=True):
    import cupy as cp

    u = np.asarray(u, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    n = g.size
    K = np.zeros((n, n, n, 6), dtype=np.float64)
    beta, inv_g2, uu2 = _velocity(u)
    cg = cp.asarray(cells)
    S = cp.cuda.Stream()
    S.use()

    fx, fw = axis_rule(0.0, False, order_far, refine_far)
    fy, fwy = axis_rule(0.0, False, order_far, refine_far)
    fz, fwz = axis_rule(0.0, False, order_far, refine_far)
    Fx, Fy, Fz = cp.asarray(fx), cp.asarray(fy), cp.asarray(fz)
    Fwx, Fwy, Fwz = cp.asarray(fw), cp.asarray(fwy), cp.asarray(fwz)

    def far_kernel(rs):
        rc = cp.asarray(rs, dtype=np.float64)
        Rx = (rc[:, 0][:, None, None, None] - Fx[None, :, None, None]) * cg[0]   # (B,a,1,1)
        Ry = (rc[:, 1][:, None, None, None] - Fy[None, None, :, None]) * cg[1]   # (B,1,b,1)
        Rz = (rc[:, 2][:, None, None, None] - Fz[None, None, None, :]) * cg[2]   # (B,1,1,c)
        comps = _components(Rx, Ry, Rz, beta, inv_g2, uu2)                        # 6 x (B,a,b,c)
        out = np.empty((rs.shape[0], 6), dtype=np.float64)
        for j, c in enumerate(comps):
            out[:, j] = cp.asnumpy(_reduce_batch(c, Fwx, Fwy, Fwz))
        return out

    rlist = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    sing = np.all(np.abs(rlist) < 1.5, axis=1)
    far_mask = ~sing
    if verbose:
        print(f"  grid {n}^3 = {n**3} pts; in-support={int(sing.sum())} far={int(far_mask.sum())}", flush=True)
    flat = np.arange(rlist.shape[0])
    iz = (flat % n).astype(np.int64)
    iy = ((flat // n) % n).astype(np.int64)
    ix = (flat // (n * n)).astype(np.int64)

    far_pts = rlist[far_mask]
    out_far = np.empty((far_pts.shape[0], 6), dtype=np.float64)
    for i in range(0, far_pts.shape[0], batch):
        out_far[i:i + batch] = far_kernel(far_pts[i:i + batch])
    far_idx = np.nonzero(far_mask)[0]
    K[ix[far_idx], iy[far_idx], iz[far_idx], :] = out_far

    near_idx = np.nonzero(sing)[0]
    for c in range(0, near_idx.size):
        k = near_idx[c]
        r = rlist[k]
        sx, wx = axis_rule(r[0], True, order_near, refine_near)
        sy, wy = axis_rule(r[1], True, order_near, refine_near)
        sz, wz = axis_rule(r[2], True, order_near, refine_near)
        Sx, Sy, Sz = cp.asarray(sx), cp.asarray(sy), cp.asarray(sz)
        Wx, Wy, Wz = cp.asarray(wx), cp.asarray(wy), cp.asarray(wz)
        rc = cp.asarray(r, dtype=np.float64)
        Rx = (rc[0] - Sx) * cg[0]
        Ry = (rc[1] - Sy) * cg[1]
        Rz = (rc[2] - Sz) * cg[2]
        comps = _components(Rx[:, None, None], Ry[None, :, None], Rz[None, None, :], beta, inv_g2, uu2)
        vals = np.empty(6, dtype=np.float64)
        for j, cc in enumerate(comps):
            vals[j] = cp.asnumpy(_reduce3(cc, Wx, Wy, Wz))
        K[ix[k], iy[k], iz[k], :] = vals
        if verbose and (c + 1) % 1000 == 0:
            print(f"    in-support {c + 1}/{near_idx.size}", flush=True)
    S.synchronize()
    return K, {"n": n, "shape": [n, n, n, 6], "dtype": "float64", "components": list(ref.COMPONENTS)}


def _meta(u, cells, grid, cfg):
    return {
        "task": "T145 near-kernel table", "schema": "nktable/v1", "components": list(ref.COMPONENTS),
        "units": {"displacement": "cell units per axis (r_i = (x_obs-x_src)_i / cell_i)",
                  "field": "per-unit-charge normalised (E_n=4pi*eps0 E/q, B_n=4pi*eps0 c B/q), dx-independent"},
        "gamma": float(np.sqrt(1.0 + np.sum(np.asarray(u) ** 2))),
        "u_cell": [float(x) for x in np.asarray(u, dtype=float)],
        "cells": [float(x) for x in np.asarray(cells, dtype=float)],
        "grid": [float(x) for x in np.asarray(grid, dtype=float)],
        "config": cfg,
        "reference": "indfield.py (validated: stationary 2.4e-10, in-support floor ~1.4e-4 @ order>=12/refine>=3)",
        "symmetry": "K(r) odd in r: K(-r) = -K(r), K(0)=0",
    }


def save_table(K, meta, path_bin, path_json):
    K = np.ascontiguousarray(K, dtype=np.float64)
    K.tofile(path_bin)
    h = hashlib.sha256()
    with open(path_bin, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    meta = dict(meta)
    meta["binary"] = os.path.basename(path_bin)
    meta["binary_bytes"] = int(K.nbytes)
    meta["binary_sha256"] = h.hexdigest()
    with open(path_json, "w") as f:
        json.dump(meta, f, indent=2)
    return h.hexdigest()


def graded_grid(rmax, h_near, h_far, r_near):
    n = int(round(r_near / h_near))
    near = list(np.arange(-n, n + 1) * h_near)
    far = []
    v = r_near + h_far
    while v <= rmax + 1e-9:
        far.append(v)
        v += h_far
    pos = near + far + [-x for x in reversed(far)]
    return np.array(sorted(set(np.round(pos, 12))), dtype=np.float64)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--dir", type=float, nargs=3, default=[0.6, 0.5, 0.62])
    ap.add_argument("--cells", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--rmax", type=float, default=13.0)
    ap.add_argument("--h-near", type=float, default=0.2)
    ap.add_argument("--h-far", type=float, default=0.5)
    ap.add_argument("--r-near", type=float, default=4.0)
    ap.add_argument("--order-near", type=int, default=12)
    ap.add_argument("--refine-near", type=int, default=3)
    ap.add_argument("--order-far", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="table/nktable")
    a = ap.parse_args()
    d = np.array(a.dir, dtype=float); d /= np.linalg.norm(d)
    u = np.sqrt(max(a.gamma * a.gamma - 1.0, 0.0)) * d
    grid = graded_grid(a.rmax, a.h_near, a.h_far, a.r_near)
    cfg = dict(order_near=a.order_near, refine_near=a.refine_near, order_far=a.order_far,
               refine_far=0, h_near=a.h_near, h_far=a.h_far, r_near=a.r_near, rmax=a.rmax)
    print(f"gamma={a.gamma:.3f} u={u} cells={a.cells}", flush=True)
    print(f"grid n={grid.size} h_near={a.h_near} h_far={a.h_far} r_near={a.r_near} rmax={a.rmax}", flush=True)
    K, meta = build_table(u, a.cells, grid, a.order_near, a.refine_near, a.order_far, 0,
                          batch=a.batch, verbose=True)
    full = _meta(u, a.cells, grid, cfg); full.update(meta)
    tag = f"gamma{a.gamma:g}_ay{a.cells[1]:g}_az{a.cells[2]:g}"
    sha = save_table(K, full, f"{a.out}_{tag}.bin", f"{a.out}_{tag}.json")
    print(f"wrote {a.out}_{tag}.bin sha256={sha} bytes={K.nbytes} ({K.nbytes / 1e6:.1f} MB)", flush=True)
