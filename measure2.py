"""T123: smooth-remainder node count.
Pure remainder  R(n) = int_S (W-Wt) K ds  on 8 octants (n/axis), R(64) converged (~3e-6).
Pure box       Sp(128) = box_field(xo,0,cells,128)*(Wt/vol)  [3-vec E only; analytic/asinh in production, 0 nodes].
Total field    F: E = Sp + Re, B = Br   (box is isotropic Coulomb -> E only; B entirely from remainder,
              matching full_field which returns (Sp+Re, Br)).   Reference = indfield o12/r3.
Metrics (worst over 32 positions):
  rem_worst = max|R(n)-R(64)| / (1+max|ref|)   <- the remainder's own quadrature error
  F_worst   = max|F(n)-ref|  / (1+max|ref|)     <- what the simulation sees (box exact)
For u=0 the isotropic box is exact so F_worst ~ rem_worst; for u!=0 a floor appears."""
import os
import numpy as np
import t123lib as L

GAMMAS = [1, 10, 100]
ASPECTS = ["0.25", "4.0"]
NS = [2, 3, 4, 6, 8, 12, 16, 24, 32]
NBOX = 128
POS = L.positions(n_dir=12, radii=(0.5, 1.0), face=1.49)
NPOS = len(POS)

print(f"{'g':>4} {'a':>5} {'n':>3} {'rem_worst':>11} {'@pos':>4} {'F_worst':>11} {'@pos':>4} {'oct':>3} {'nodes=8n^3':>10} {'spdup':>7}")
for g in GAMMAS:
    for a in ASPECTS:
        cells = np.array([1.0, float(a), 1.0])
        vol = float(cells[0] * cells[1] * cells[2])
        u = L.u_for(g, [0.6, 0.5, 0.62])
        if not all(os.path.exists(f"ref1_g{g}_a{a}_i{i}.npz") for i in range(NPOS)):
            print(f"{g:>4} {a:>5}: refs not ready"); continue
        Eref = np.array([np.load(f"ref1_g{g}_a{a}_i{i}.npz")["E"] for i in range(NPOS)])
        Bref = np.array([np.load(f"ref1_g{g}_a{a}_i{i}.npz")["B"] for i in range(NPOS)])
        Ref = np.concatenate([Eref, Bref], axis=1)
        scale = 1.0 + np.max(np.abs(Ref), axis=1)          # per-position scale
        # hoist per-position converged remainder R(64) + pure box Sp(128) E-field
        R64 = []; Sp = []
        for i in range(NPOS):
            xo = POS[i] * cells
            Wt = float(L.tsc(POS[i][0]) * L.tsc(POS[i][1]) * L.tsc(POS[i][2]))
            Re, Br, _ = L.remainder(xo, u, cells, 64, Wt)
            R64.append(np.concatenate([Re, Br]))
            Sp.append(L.box_field(xo, np.zeros(3), cells, NBOX) * (Wt / vol))
        for n in NS:
            remw = 0.0; remi = -1; Fw = 0.0; Fi = -1; maxoct = 0
            for i in range(NPOS):
                xo = POS[i] * cells
                Wt = float(L.tsc(POS[i][0]) * L.tsc(POS[i][1]) * L.tsc(POS[i][2]))
                Re, Br, octs = L.remainder(xo, u, cells, n, Wt)
                Rn = np.concatenate([Re, Br])
                maxoct = max(maxoct, octs)
                e_rem = float(np.max(np.abs(Rn - R64[i])) / scale[i])
                F = np.concatenate([Sp[i] + Re, Br])      # E = box+rem, B = rem
                e_F = float(np.max(np.abs(F - Ref[i])) / scale[i])
                if e_rem > remw: remw, remi = e_rem, i
                if e_F > Fw: Fw, Fi = e_F, i
            nodes = 8 * n**3
            print(f"{g:>4} {a:>5} {n:>3} {remw:>11.3e} {remi:>4} {Fw:>11.3e} {Fi:>4} {maxoct:>3} {nodes:>10} {L.BASE_NODES/nodes:>7.1f}")
print("MEASURE2 DONE", flush=True)
