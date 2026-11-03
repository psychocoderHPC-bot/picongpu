#!/usr/bin/env python3
"""Validate a near-kernel table (T145): worst-case error over regions/azimuths.

Measures at off-grid query points:
  - interpolation error : interp(table) vs reference at the exact point, at the
    SAME order the table used (o12/r3 near, o8 far) -> pure table-interpolation
    error.  Interpolants: trilinear (8 corners) and cubic (27 corners).
  - reference floor scan: dense in-support |ref(o12/r3)-ref(o12/r2)| (Richardson).
Reports worst-case relative error per component, per region.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indfield as ref


def load_table(bin_path):
    with open(bin_path.replace(".bin", ".json")) as f:
        meta = json.load(f)
    K = np.fromfile(bin_path, dtype=np.float64)
    g = np.array(meta["grid"], dtype=np.float64)
    n = len(g)
    return K.reshape(n, n, n, 6), meta, g


def _bracket(g, r):
    n = g.size
    if r <= g[0]:
        return 0
    if r >= g[-1]:
        return n - 2
    i = int(np.searchsorted(g, r)) - 1
    return max(0, min(i, n - 2))


def trilinear(r, g, K):
    n = g.size
    out = np.empty(6, dtype=np.float64)
    i = [_bracket(g, r[a]) for a in range(3)]
    t = [(r[a] - g[i[a]]) / (g[i[a] + 1] - g[i[a]]) for a in range(3)]
    for c in range(6):
        v = K[:, :, :, c]
        a00 = v[i[0], i[1], i[2]] * (1 - t[0]) + v[i[0] + 1, i[1], i[2]] * t[0]
        a10 = v[i[0], i[1] + 1, i[2]] * (1 - t[0]) + v[i[0] + 1, i[1] + 1, i[2]] * t[0]
        a01 = v[i[0], i[1], i[2] + 1] * (1 - t[0]) + v[i[0] + 1, i[1], i[2] + 1] * t[0]
        a11 = v[i[0], i[1] + 1, i[2] + 1] * (1 - t[0]) + v[i[0] + 1, i[1] + 1, i[2] + 1] * t[0]
        a0 = a00 * (1 - t[1]) + a10 * t[1]
        a1 = a01 * (1 - t[1]) + a11 * t[1]
        out[c] = a0 * (1 - t[2]) + a1 * t[2]
    return out


def _lagrange4(g, v, i):
    """4 Lagrange weights for points i-1,i,i+1,i+2 at x (clamped), x in [g[i],g[i+1]]."""
    idx = [i - 1, i, i + 1, i + 2]
    x = None
    return idx  # placeholder


def cubic_interp(r, g, K):
    """Tensor-product cubic via 4-point Lagrange per axis (27 corners), non-uniform grid."""
    n = g.size
    out = np.empty(6, dtype=np.float64)
    idx = []
    for a in range(3):
        i = _bracket(g, r[a])
        idx.append([max(0, i - 1), i, i + 1, min(n - 1, i + 2)])
    xw = []
    for a in range(3):
        i = _bracket(g, r[a])
        xs = np.array([g[max(0, i - 1)], g[i], g[i + 1], g[min(n - 1, i + 2)]])
        x = r[a]
        # weights for the 4 abscissae (Lagrange); handle repeated nodes at clamped edges
        w = np.zeros(4)
        for k in range(4):
            num = 1.0
            for m in range(4):
                if m == k:
                    continue
                if abs(xs[k] - xs[m]) < 1e-30:
                    num = 0.0
                    break
                num *= (x - xs[m]) / (xs[k] - xs[m])
            w[k] = num
        xw.append(w)
    for c in range(6):
        v = K[:, :, :, c]
        acc = 0.0
        for a in range(4):
            wa = xw[0][a]
            for b in range(4):
                wb = wa * xw[1][b]
                for cc in range(4):
                    acc += wb * xw[2][cc] * v[idx[0][a], idx[1][b], idx[2][cc]]
        out[c] = acc
    return out


def _rel(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    return float(np.max(np.abs(a - b)) / (1.0 + np.max(np.abs(b))))


def ref_at(r, u, cells, order_near, refine_near, order_far):
    sng = bool(np.all(np.abs(r) < 1.5))
    o, rf = (order_near, refine_near) if sng else (order_far, 0)
    E, B = ref.integrate_point(r, np.zeros(3), u, *cells, order=o, refine=rf)
    return np.concatenate([E, B]), sng


def region_of(r):
    m = float(np.max(np.abs(r)))
    if m < 1.5:
        return "in-support"
    if m <= 1.6:
        return "face"
    if m < 2.5:
        return "just-out"
    return "shell"


def test_points(udir, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    dirs = [d / np.linalg.norm(d) for d in (rng.normal(size=3) for _ in range(60))
            if np.linalg.norm(d) > 0.1]
    dirs.append(udir)
    dirs.append(-udir)
    for ax in range(3):
        e = np.zeros(3); e[ax] = 1.0; dirs.append(e)
    shells = [0.5, 1.0, 1.3, 1.5, 1.8, 2.2, 3.0, 5.0, 8.0, 12.0]
    for R in shells:
        for d in dirs:
            r = R * d / np.max(np.abs(d))
            if R <= 1.3:
                r = R * d
            pts.append(np.array(r, dtype=float))
    return pts


def validate(bin_path, interp=trilinear, verbose=True):
    K, meta, g = load_table(bin_path)
    u = np.array(meta["u_cell"]); cells = np.array(meta["cells"])
    cfg = meta["config"]
    on, rn, of = cfg["order_near"], cfg["refine_near"], cfg["order_far"]
    udir = u / np.linalg.norm(u) if np.linalg.norm(u) > 0 else np.array([0.0, 1.0, 0.0])
    pts = test_points(udir)
    regs = {"in-support": [], "face": [], "just-out": [], "shell": []}
    worst = {rk: np.zeros(6) for rk in regs}
    for r in pts:
        ti = interp(r, g, K)
        rf, _ = ref_at(r, u, cells, on, rn, of)
        rk = region_of(r)
        regs[rk].append(_rel(ti, rf))
        per = np.abs(ti - rf) / (1.0 + np.abs(rf))
        worst[rk] = np.maximum(worst[rk], per)
    if verbose:
        name = "trilinear" if interp is trilinear else "cubic"
        print(f"== {name}: gamma={meta['gamma']:.3g} cells={cells} n={len(g)} ({len(pts)} pts) ==")
        comp = list(meta["components"])
        print("  %-11s" % "region", " ".join("%8s" % c for c in comp), "   max")
        for rk in regs:
            print("  %-11s" % rk, " ".join("%8.2e" % v for v in worst[rk]), "  %.2e" % worst[rk].max())
        print("  %-11s %9.2e\n" % ("OVERALL", max(worst[rk].max() for rk in regs)))
    return worst


def floor_scan(u, cells, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0; wr = None
    for _ in range(300):
        r = rng.uniform(-1.4, 1.4, 3)
        v3 = ref_at(r, u, cells, 12, 3, 12)[0]
        v2 = ref_at(r, u, cells, 12, 2, 12)[0]
        e = _rel(v3, v2)
        if e > worst:
            worst = e; wr = r.copy()
    return worst, wr


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--floor", action="store_true")
    a = ap.parse_args()
    K, meta, g = load_table(a.table)
    validate(a.table, trilinear)
    validate(a.table, cubic_interp)
    if a.floor:
        u = np.array(meta["u_cell"]); cells = np.array(meta["cells"])
        w, wr = floor_scan(u, cells)
        print("== reference floor scan (o12/r3 vs o12/r2, in-support, 300 pts) ==")
        print("   worst Richardson diff = %.3e at r=%s" % (w, wr))
