"""Measure F(n) = S_part(exact box) + R(n) vs the per-position independent reference.
Reads ref1_g{g}_a{a}_i{i}.npz.  Reports worst-case normalized error over positions."""
import os, sys
import numpy as np
import t123lib as L

GAMMAS = [1, 10, 100]
ASPECTS = ["0.25", "4.0"]
NS = [2, 3, 4, 6, 8, 12, 16, 32, 64]
N_BOX = 16
N_BOX_NODES = 6 * N_BOX * N_BOX
POS = L.positions(n_dir=12, radii=(0.5, 1.0), face=1.49)
NPOS = len(POS)

results = []
print(f"{'g':>4} {'a':>5} {'n':>3} {'worst_err':>10} {'@pos':>4} {'nodes':>7} {'speedup':>8} {'cost_us':>9}")
for g in GAMMAS:
    for a in ASPECTS:
        cells = np.array([1.0, float(a), 1.0])
        u = L.u_for(g, [0.6, 0.5, 0.62])
        # load all refs for this config
        have = all(os.path.exists(f"ref1_g{g}_a{a}_i{i}.npz") for i in range(NPOS))
        if not have:
            print(f"g={g} a={a}: refs not ready, skip"); continue
        Eref = np.array([np.load(f"ref1_g{g}_a{a}_i{i}.npz")["E"] for i in range(NPOS)])
        Bref = np.array([np.load(f"ref1_g{g}_a{a}_i{i}.npz")["B"] for i in range(NPOS)])
        for n in NS:
            worst = 0.0; wi = -1; max_oct = 0
            for i in range(NPOS):
                xo = POS[i] * cells
                En, Bn, octs = L.full_field(xo, np.zeros(3), u, cells, n, n_box=N_BOX)
                A = np.concatenate([En, Bn]); B = np.concatenate([Eref[i], Bref[i]])
                e = float(np.max(np.abs(A - B)) / (1.0 + np.max(np.maximum(np.abs(A), np.abs(B)))))
                max_oct = max(max_oct, octs)
                if e > worst:
                    worst, wi = e, i
            nodes = 8 * n**3 + N_BOX_NODES
            speedup = L.BASE_NODES / nodes
            cost = L.BASE_MS * 1000.0 * nodes / L.BASE_NODES
            results.append([g, a, n, worst, wi, nodes, speedup])
            print(f"{g:>4} {a:>5} {n:>3} {worst:>10.3e} {wi:>4} {nodes:>7} {speedup:>8.1f} {cost:>9.1f}")

import pickle
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)
print("MEASURE DONE", flush=True)
