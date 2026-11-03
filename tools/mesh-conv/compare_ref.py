#!/usr/bin/env python3
"""Validate the C++/CUDA mesh field against the validated indfield.py reference.

Input: the CSV written by `meshconv --mode validate` (gamma,aspect,axis,d,comp,mesh).
Reference: indfield.integrate_point (the independent reference used by T139/T142/
T149), evaluated at the STAGGERED position (target + offset_c)*cells, with the
source sitting on the centre node.  Serially in this process.

Error convention (identical to T139/T142/T149):
    rel = |ref - mesh| / (1 + max(|ref|, |mesh|))
We also report the plain relative error |ref-mesh|/|ref| where |ref| > 0.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import indfield  # noqa: E402

N = 48
CENTER = N // 2
CACHE_FILE = os.path.join(HERE, "refcache.npz")


def vel(gamma):
    if gamma <= 1.0:
        return np.zeros(3)
    g = float(gamma)
    beta = np.sqrt(1.0 - 1.0 / (g * g))
    d = np.array([1.0, 2.0, 1.0])
    d /= np.linalg.norm(d)
    return g * beta * d


def load_cache():
    if os.path.exists(CACHE_FILE):
        z = np.load(CACHE_FILE)
        return {k: z[k] for k in z.files}
    return {}


def reference(gamma, aspect, axis, d, cache):
    key = f"{gamma}|{aspect}|{axis}|{d}"
    if key in cache:
        return cache[key]
    cells = np.array([1.0, aspect, 1.0])
    src = np.array([CENTER, CENTER, CENTER], dtype=float) * cells
    tgt = np.array([CENTER, CENTER, CENTER], dtype=float)
    if axis == "x":
        tgt[0] += d
    else:
        tgt[1] += d
    u = vel(gamma)
    ref = np.empty(6)
    for c in range(6):
        xo = (tgt + indfield.OFFSETS[c]) * cells
        e, b = indfield.integrate_point(
            xo, src, u, cells[0], cells[1], cells[2],
            order=16, refine=(1 if d <= 2 else 0), chunk=8)
        ref[c] = e[c] if c < 3 else b[c - 3]
    cache[key] = ref
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="validation.json")
    a = ap.parse_args()

    rows = []
    with open(a.csv) as f:
        header = f.readline().strip().split(",")
        assert header == ["gamma", "aspect", "axis", "d", "comp", "mesh"], header
        for line in f:
            p = line.strip().split(",")
            if len(p) != 6:
                continue
            rows.append((int(float(p[0])), float(p[1]), p[2], int(float(p[3])),
                         p[4], float(p[5])))

    cache = load_cache()
    comps = list(indfield.COMPONENTS)
    worst = {}        # (gamma,aspect,comp) -> max rel
    worst_plain = {}  # (gamma,aspect,comp) -> max plain rel
    worst_point = {}
    bands = {"all": (2, 16), "d8": (8, 16), "d12": (12, 16)}
    worst_band = {}
    per_d = {}
    for gamma, aspect, axis, d, comp, mesh in rows:
        refv = reference(gamma, aspect, axis, d, cache)
        c = comps.index(comp)
        ref = float(refv[c])
        rel = abs(ref - mesh) / (1.0 + max(abs(ref), abs(mesh)))
        key = (gamma, aspect, comp)
        if rel > worst.get(key, -1.0):
            worst[key] = rel
            worst_point[key] = dict(axis=axis, d=d, ref=ref, mesh=mesh)
        for bname, (lo, hi) in bands.items():
            if lo <= d <= hi:
                bk = (gamma, aspect, comp, bname)
                worst_band[bk] = max(worst_band.get(bk, 0.0), rel)
        if abs(ref) > 0:
            plain = abs(ref - mesh) / abs(ref)
            worst_plain[key] = max(worst_plain.get(key, 0.0), plain)
        per_d.setdefault((gamma, aspect, axis), {})[d] = max(
            per_d.get((gamma, aspect, axis), {}).get(d, 0.0), rel)

    np.savez(CACHE_FILE, **cache)

    out = {"N": N, "rows": len(rows), "worst_rel": {}, "worst_plain_rel": {},
           "worst_point": {}, "per_d": {}, "worst_rel_band": {}}
    for k, v in worst.items():
        out["worst_rel"][f"{k[0]}|{k[1]}|{k[2]}"] = v
    for k, v in worst_band.items():
        out["worst_rel_band"][f"{k[0]}|{k[1]}|{k[2]}|{k[3]}"] = v
    for k, v in worst_plain.items():
        out["worst_plain_rel"][f"{k[0]}|{k[1]}|{k[2]}"] = v
    for k, v in worst_point.items():
        out["worst_point"][f"{k[0]}|{k[1]}|{k[2]}"] = v
    for k, v in per_d.items():
        out["per_d"][f"{k[0]}|{k[1]}|{k[2]}"] = {str(d): e for d, e in sorted(v.items())}

    print("worst |ref-mesh|/(1+max) per component, d=2..16, both axes")
    gammas = sorted({r[0] for r in rows})
    aspects = sorted({r[1] for r in rows})
    for g in gammas:
        for A in aspects:
            line = "  gamma=%3d aspect=%-5g " % (g, A)
            for comp in comps:
                line += "%s=%.3e " % (comp, worst.get((g, A, comp), 0.0))
            print(line)
    print("\nworst |ref-mesh|/(1+max), far band d=8..16 only")
    for g in gammas:
        for A in aspects:
            line = "  gamma=%3d aspect=%-5g " % (g, A)
            for comp in comps:
                line += "%s=%.3e " % (comp, worst_band.get((g, A, comp, "d8"), 0.0))
            print(line)
    print("\nplain relative |ref-mesh|/|ref| (far points can be tiny)")
    for g in gammas:
        for A in aspects:
            line = "  gamma=%3d aspect=%-5g " % (g, A)
            for comp in comps:
                line += "%s=%.2e " % (comp, worst_plain.get((g, A, comp), 0.0))
            print(line)
    overall = max(worst.values()) if worst else 0.0
    print(f"\nOVERALL worst rel = {overall:.3e}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
