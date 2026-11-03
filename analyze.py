"""Deeper analysis once refs exist.
 A) gamma=1 (u=0): does the 1/r remainder NEED the octant split?  F(n) octant vs single.
 B) gamma=100 (u!=0): R(n) converges (R(n)-R(64)->0) but S_part+R(n) plateaus at a floor
    != 0 (isotropic-vs-anisotropic box mismatch) => no convergence to F_ref.
"""
import os
import numpy as np
import t123lib as L

POS = L.positions(n_dir=12, radii=(0.5, 1.0), face=1.49)


def norm6(E, B, Er, Br):
    a = np.concatenate([E, B]); b = np.concatenate([Er, Br])
    return float(np.max(np.abs(a - b)) / (1.0 + np.max(np.maximum(np.abs(a), np.abs(b)))))


print("=== A) gamma=1: octant vs single-box remainder ===", flush=True)
for a in [0.25, 4.0]:
    cells = np.array([1.0, a, 1.0])
    d = np.load(f"ref_g1_a{a}.npz"); Eref, Bref, u = d["E"], d["B"], d["u"]
    for n in [2, 4, 8, 16, 32]:
        wo = ws = 0.0
        for i in range(len(POS)):
            xo = POS[i] * cells
            Eo, Bo, _ = L.full_field(xo, np.zeros(3), u, cells, n, n_box=16, octant=True)
            E1, B1, _ = L.full_field(xo, np.zeros(3), u, cells, n, n_box=16, octant=False)
            wo = max(wo, norm6(Eo, Bo, Eref[i], Bref[i]))
            ws = max(ws, norm6(E1, B1, Eref[i], Bref[i]))
        print(f"  a={a:5.2f} n={n:3d}  octant={wo:.3e}  single={ws:.3e}", flush=True)

print("\n=== B) gamma=100: remainder convergence vs total floor ===", flush=True)
for a in [0.25, 4.0]:
    cells = np.array([1.0, a, 1.0])
    d = np.load(f"ref_g100_a{a}.npz"); Eref, Bref, u = d["E"], d["B"], d["u"]
    for i in [0, 1, len(POS) // 2]:
        xo = POS[i] * cells
        t = POS[i]
        Wt = float(L.tsc(t[0]) * L.tsc(t[1]) * L.tsc(t[2]))
        R64, _, _ = L.remainder(xo, u, cells, 64, Wt)
        for n in [8, 16, 32, 64]:
            Rn, _, _ = L.remainder(xo, u, cells, n, Wt)
            En, Bn, _ = L.full_field(xo, np.zeros(3), u, cells, n, n_box=16)
            rconv = float(np.max(np.abs(Rn - R64)) / (1.0 + np.max(np.abs(R64))))
            total = norm6(En, Bn, Eref[i], Bref[i])
            print(f"  a={a:5.2f} i={i} t={np.round(t,2)} n={n:3d}  R(n)-R64={rconv:.3e}  F(n)-ref={total:.3e}", flush=True)
print("ANALYZE DONE", flush=True)
