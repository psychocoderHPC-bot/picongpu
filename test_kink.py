"""Test: does splitting the remainder at the TSC kinks (|x|=0.5,1.5) in addition to t
restore spectral convergence and cut the node count?  R64 (t-only split, n=64) is the
converged remainder (3e-6) used as the target.  Reports worst-case rel err + node count."""
import numpy as np, os
import t123lib as L

def segs_kink(td):
    """td = scalar observer coord in one dimension (in [-1.5,1.5])."""
    pts = sorted(set([-1.5, -0.5, 0.5, 1.5] + ([float(td)] if -1.5 < float(td) < 1.5 else [])))
    return [(a, b) for a, b in zip(pts[:-1], pts[1:]) if b > a]

def remainder_kink(R0, u, cells, n, w_offset):
    R0 = np.asarray(R0, float); c = np.asarray(cells, float); u = np.asarray(u, float)
    t = R0 / c
    tn, wn = L._gauss(n)
    S0, S1, S2 = segs_kink(t[0]), segs_kink(t[1]), segs_kink(t[2])
    acc = np.zeros(6); nodes = 0
    for (ax, bx) in S0:
        hx = 0.5*(bx-ax); mx = 0.5*(bx+ax); sx = mx + hx*tn; wx = hx*wn
        for (ay, by) in S1:
            hy = 0.5*(by-ay); my = 0.5*(by+ay); sy = my + hy*tn; wy = hy*wn
            for (az, bz) in S2:
                hz = 0.5*(bz-az); mz = 0.5*(bz+az); sz = mz + hz*tn; wz = hz*wn
                SX, SY, SZ = np.meshgrid(sx, sy, sz, indexing="ij")
                nodes += SX.size
                Ww = L.tsc(SX)*L.tsc(SY)*L.tsc(SZ) - w_offset
                WW = wx[:,None,None]*wy[None,:,None]*wz[None,None,:]
                R = np.stack([R0[0]-SX*c[0], R0[1]-SY*c[1], R0[2]-SZ*c[2]], axis=-1)
                E, B = L.point_field(R, u)
                w = WW*Ww
                acc[0]+=np.sum(w*E[...,0]); acc[1]+=np.sum(w*E[...,1]); acc[2]+=np.sum(w*E[...,2])
                acc[3]+=np.sum(w*B[...,0]); acc[4]+=np.sum(w*B[...,1]); acc[5]+=np.sum(w*B[...,2])
    return acc[:3], acc[3:], nodes

POS = L.positions(n_dir=12, radii=(0.5, 1.0), face=1.49)
for g in [1, 10]:
    for a in ["0.25", "4.0"]:
        cells = np.array([1.0, float(a), 1.0]); u = L.u_for(g, [0.6, 0.5, 0.62])
        Eref = np.array([np.load(f"ref1_g{g}_a{a}_i{i}.npz")["E"] for i in range(len(POS))])
        Bref = np.array([np.load(f"ref1_g{g}_a{a}_i{i}.npz")["B"] for i in range(len(POS))])
        Ref = np.concatenate([Eref, Bref], axis=1); scale = 1.0 + np.max(np.abs(Ref), axis=1)
        R64 = []
        for i in range(len(POS)):
            xo = POS[i]*cells
            Wt = float(L.tsc(POS[i][0])*L.tsc(POS[i][1])*L.tsc(POS[i][2]))
            Re, Br, _ = L.remainder(xo, u, cells, 64, Wt)
            R64.append(np.concatenate([Re, Br]))
        line = f"g={g:>3} a={a:>4} | "
        for n in [2, 4, 6, 8, 12]:
            remw = 0.0; nd = 0
            for i in range(len(POS)):
                xo = POS[i]*cells
                Wt = float(L.tsc(POS[i][0])*L.tsc(POS[i][1])*L.tsc(POS[i][2]))
                Re, Br, nd2 = remainder_kink(xo, u, cells, n, Wt)
                e = float(np.max(np.abs(np.concatenate([Re,Br]) - R64[i]))/scale[i])
                remw = max(remw, e); nd = max(nd, nd2)
            line += f"n={n:>2}:{remw:>9.2e}({nd:>6}) "
        print(line, flush=True)
print("KINK TEST DONE", flush=True)
