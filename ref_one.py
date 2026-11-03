"""Compute ONE reference (g,a,i) and save to ref1_g{g}_a{a}_i{i}.npz.
Run as a subprocess with single-threaded BLAS env (set by the launcher)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t123lib as L

g = int(sys.argv[1]); a = sys.argv[2]; i = int(sys.argv[3])
POS = L.positions(n_dir=12, radii=(0.5, 1.0), face=1.49)
u = L.u_for(g, [0.6, 0.5, 0.62])
cells = np.array([1.0, float(a), 1.0])
xo = POS[i] * cells
e, b = L.ref_field(xo, np.zeros(3), u, cells, order=12, refine=3)
np.savez(f"ref1_g{g}_a{a}_i{i}.npz", E=np.asarray(e, float), B=np.asarray(b, float))
