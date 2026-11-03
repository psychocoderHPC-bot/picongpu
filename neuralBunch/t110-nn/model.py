"""Pure-numpy inference for the T110 residual surrogate (no torch at runtime).

Reads <tag>.npz (plain weights) + <tag>.json (feature/normalisation metadata).
Exposes predict(points) -> (E, cB) for the near-field judge.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.environ.get("T110_MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TAG = os.environ.get("T110_MODEL_TAG", "c5_med_weighted")


def _point_field(R, u):
    R = np.asarray(R, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    uu2 = np.sum(u * u, axis=-1)
    inv_g2 = 1.0 / (1.0 + uu2)
    gamma = np.sqrt(1.0 + uu2)
    beta = u / gamma[..., None]
    r2 = np.sum(R * R, axis=-1)
    rbeta = np.sum(R * beta, axis=-1)
    D = r2 * inv_g2 + rbeta * rbeta
    E = (inv_g2[..., None] * R) / (D ** 1.5)[..., None]
    B = np.cross(beta, E)
    static = uu2 == 0.0
    if np.any(static):
        Es = R / (r2 ** 1.5)[..., None]
        E = np.where(static[..., None], Es, E)
        B = np.where(static[..., None], 0.0, B)
    return E, B


def _features(R_cells, u, cells, gamma, feature_set):
    R = R_cells * cells
    r = np.linalg.norm(R, axis=1)
    un = np.linalg.norm(u, axis=1)
    b = np.where(un[:, None] > 1e-12, u / np.maximum(un[:, None], 1e-30), 0.0)
    rpar = np.sum(R * b, axis=1)
    rper = np.sqrt(np.maximum(r * r - rpar * rpar, 0.0))
    lg = np.log(gamma)
    lay = np.log(cells[:, 1])
    laz = np.log(cells[:, 2])
    inv = 1.0 / (1.0 + r)
    gpar = gamma * rpar * inv
    gperp = gamma * rper * inv
    if feature_set == "raw":
        cols = [R[:, 0], R[:, 1], R[:, 2], b[:, 0], b[:, 1], b[:, 2],
                lg, lay, laz, r, rpar, rper]
    elif feature_set == "phys":
        cols = [R[:, 0] * inv, R[:, 1] * inv, R[:, 2] * inv,
                b[:, 0], b[:, 1], b[:, 2], lg, lay, laz,
                r * inv, rpar * inv, rper * inv,
                gpar, gperp, inv, gpar * gpar, gperp * gperp]
    elif feature_set == "fourier":
        cols = [R[:, 0] * inv, R[:, 1] * inv, R[:, 2] * inv,
                b[:, 0], b[:, 1], b[:, 2], lg, lay, laz,
                r * inv, rpar * inv, rper * inv, inv]
        for k in (1, 2, 3):
            cols += [np.sin(k * gpar), np.cos(k * gpar),
                     np.sin(k * gperp), np.cos(k * gperp)]
    elif feature_set == "rff":
        cols = [R[:, 0] * inv, R[:, 1] * inv, R[:, 2] * inv,
                b[:, 0], b[:, 1], b[:, 2], lg, lay, laz,
                r * inv, rpar * inv, rper * inv, inv, gpar, gperp]
        for f in (0.5, 1.0, 2.0, 4.0, 8.0):
            cols += [np.sin(f * R[:, 0] * inv), np.cos(f * R[:, 0] * inv),
                     np.sin(f * R[:, 1] * inv), np.cos(f * R[:, 1] * inv),
                     np.sin(f * R[:, 2] * inv), np.cos(f * R[:, 2] * inv)]
            cols += [np.sin(f * gpar), np.cos(f * gpar),
                     np.sin(f * gperp), np.cos(f * gperp)]
    else:
        raise ValueError(feature_set)
    return np.stack(cols, axis=1)


def load(tag=None):
    tag = tag or DEFAULT_TAG
    meta = json.load(open(os.path.join(HERE, f"{tag}.json")))
    w = np.load(os.path.join(HERE, meta["weights"]))
    return meta, w


def _forward(w, meta, X):
    x = (X - np.asarray(meta["mu"])) / np.asarray(meta["sd"])
    weight_names = sorted([nm for nm in w.files if nm.endswith("weight")],
                          key=lambda s: int(s.split(".")[0]))
    for j, nm in enumerate(weight_names):
        W = w[nm].astype(np.float64)
        bias_name = nm.replace("weight", "bias")
        b = w[bias_name].astype(np.float64) if bias_name in w.files else 0.0
        x = x @ W.T + b
        if j + 1 < len(weight_names):  # SiLU on every hidden layer
            x = x / (1.0 + np.exp(-x))
    ob = float(meta.get("out_bound", 0.0))
    if ob > 0:
        x = ob * np.tanh(x)
    return x


def predict(points, tag=None):
    meta, w = load(tag)
    R_cells = np.asarray([p["R_cells"] for p in points], dtype=np.float64)
    u = np.asarray([p["u"] for p in points], dtype=np.float64)
    cells = np.asarray([p["cells"] for p in points], dtype=np.float64)
    gamma = np.sqrt(1.0 + np.sum(u * u, axis=1))
    X = _features(R_cells, u, cells, gamma, meta["feature_set"])
    y = _forward(w, meta, X)
    y = y * np.asarray(meta["yscale"])
    if meta.get("target_scale", "gamma") == "gamma":
        y = y * gamma[:, None]
    elif meta.get("target_scale", "gamma") == "gamma07":
        y = y * (gamma ** 0.7)[:, None]
    R_phys = R_cells * cells
    norm = np.maximum(np.sum(R_phys * R_phys, axis=1), 1e-12)
    E_point, _ = _point_field(R_phys, u)
    E = E_point + y / norm[:, None]
    cB = np.cross(u / gamma[:, None], E)
    return E, cB
