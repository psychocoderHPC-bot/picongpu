"""T116 experiment 1: verify exact identities of the TSC-convolved field.

Reference: ~/pr24/pr24w53-ref/indfield.py via nearfield_judge.reference.
All quantities are per-unit-charge in the reference normalisation.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/pr24/t103"))
from nearfield_judge import reference  # noqa: E402

REFDIR = os.path.expanduser("~/pr24/pr24w53-ref")


def field(Rc, u, cells):
    p = {"R_cells": np.asarray(Rc, float), "u": np.asarray(u, float),
         "cells": np.asarray(cells, float)}
    E, B = reference.integrate_one(p, reference_dir=REFDIR, refine_inside=2)
    return np.asarray(E, float), np.asarray(B, float)


def rel(a, b):
    return float(np.max(np.abs(a - b) / (1.0 + np.abs(b))))


def uvec(gamma, d):
    if gamma <= 1.0 + 1e-12:
        return np.zeros(3)
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    return np.sqrt(max(gamma * gamma - 1.0, 0.0)) * d


def main():
    rng = np.random.default_rng(7)
    cases = []
    base = [
        ("g2_oblique_iso", 2.0, [0.6, 0.5, 0.62], [0.7, -0.4, 0.9], [1.0, 1.0, 1.0]),
        ("g10_oblique_aniso", 10.0, [0.6, 0.5, 0.62], [0.7, -0.4, 0.9], [1.0, 0.3, 2.5]),
        ("g100_diag_inside", 100.0, [1, 1, 1], [0.5, -0.6, 0.3], [1.0, 1.0, 1.0]),
        ("g100_oblique_far", 100.0, [0.6, 0.5, 0.62], [2.5, 1.8, -2.0], [1.0, 0.25, 4.0]),
        ("stationary_aniso", 1.0, [0.0, 0.0, 0.0], [0.7, -0.4, 0.9], [1.0, 0.4, 3.0]),
    ]
    out = {}
    for name, g, d, Rc, cells in base:
        u = uvec(g, d)
        Rc = np.asarray(Rc, float)
        cells = np.asarray(cells, float)
        E, B = field(Rc, u, cells)
        Em, Bm = field(-Rc, u, cells)          # inversion, u fixed
        Eu, Bu = field(Rc, -u, cells)          # velocity sign
        beta = u / np.sqrt(1.0 + u @ u)
        rec = {
            "E_over_R": rel(Em, -E),
            "B_over_R": rel(Bm, -B),
            "E_u_flip": rel(Eu, E),
            "B_u_flip_odd": rel(Bu, -B),
            "B_eq_beta_x_E_ref": rel(B, np.cross(beta, E)),
            "absE": float(np.linalg.norm(E)), "absB": float(np.linalg.norm(B)),
        }
        # axis sign flips applied to BOTH R and u
        flips = {}
        for i in range(3):
            S = np.ones(3)
            S[i] = -1.0
            Es, Bs = field(S * Rc, S * u, cells)
            flips[f"E_flip{i}"] = rel(Es, S * E)
            # axial: B' = det(S) S B = -S B
            flips[f"B_flip{i}"] = rel(Bs, -S * B)
        rec["axis_flips"] = flips
        # axis permutations applied to R, u and cells (grid symmetry)
        perms = {}
        for perm in [(1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]:
            p = np.asarray(perm)
            Ep, Bp = field(Rc[p], u[p], cells[p])
            perms["".join(map(str, perm))] = {
                "E_perm": rel(Ep, E[p]), "B_perm_axial": rel(Bp, B[p]),
            }
        rec["axis_perms"] = perms
        out[name] = rec
        print(name, json.dumps(rec), flush=True)

    # pilot internal consistency: cB_conv == beta x E_conv
    d = np.load(os.path.expanduser("~/pr24/t112/pilot_all.npz"), allow_pickle=False)
    u = d["u"]
    beta = u / np.sqrt(1.0 + np.einsum("ij,ij->i", u, u))[:, None]
    cB = np.cross(beta, d["E_conv"])
    out["pilot_cB_eq_beta_x_E_conv"] = rel(cB, d["cB_conv"])
    out["pilot_cB_point_eq_beta_x_E_point"] = rel(
        np.cross(beta, d["E_point"]), d["cB_point"])
    # pilot: config direction vs -direction pairing (does the set contain both?)
    keys = set(d["config_key"])
    paired = 0
    for k in keys:
        parts = k.split("|")
        dname = parts[1]
        flip = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y",
                "+z": "-z", "-z": "+z"}
        if dname in flip:
            if "|".join([parts[0], flip[dname]] + parts[2:]) in keys:
                paired += 1
    out["pilot_direction_pairs"] = paired
    out["pilot_keys"] = len(keys)
    print("PILOT", json.dumps({k: v for k, v in out.items() if k.startswith("pilot")}), flush=True)
    with open(os.path.expanduser("~/pr24/t116/identities.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
