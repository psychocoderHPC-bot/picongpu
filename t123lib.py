"""T123 shared library.

In-support decomposition (ReferenceKernel.hpp, u=0):
    F = W(t)*box_field/vol   +   R,   R = int_S (W(s)-W(t)) K(R0 - s*c) ds
S_part = box_field: divergence-theorem surface integral over 6 faces (isotropic
Coulomb kernel), spectrally convergent (n=16 -> 2e-11).  R: 1/r integrable
remainder, plain Gauss on octants split at t.

F(n) = S_part + R(n).  Reference F_true = indfield.py (independent, power-transform).
For u!=0 the isotropic box does NOT cancel the anisotropic 1/r^2 peak.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "GOTO_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

# Pure-Newton Gauss-Legendre (port of ExactShapeField.hpp gaussLegendre).  No
# LAPACK/OpenBLAS -> no sysv semaphores, so it survives the fakeroot faked-sysv
# contention on the loaded node.  Monkeypatched so indfield avoids LAPACK too.
_PI = 3.141592653589793238462643383279502884
_GA = {}


def gauss_legendre_pure(n):
    n = int(n)
    if n in _GA:
        return _GA[n]
    node = np.zeros(n); weight = np.zeros(n)
    m = (n + 1) // 2
    for i in range(m):
        z = np.cos(_PI * (i + 0.75) / (n + 0.5))
        pp = 0.0
        for _ in range(100):
            p1, p2 = 1.0, 0.0
            for j in range(1, n + 1):
                p3, p2 = p2, p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) <= 1e-15:
                break
        node[i], node[n - 1 - i] = -z, z
        w = 2.0 / ((1.0 - z * z) * pp * pp)
        weight[i], weight[n - 1 - i] = w, w
    _GA[n] = (node, weight)
    return node, weight


np.polynomial.legendre.leggauss = gauss_legendre_pure

import sys
sys.path.insert(0, os.path.expanduser("~/pr24/pr24w53-ref"))
import indfield  # independent reference

TOL = 3e-4  # target worst-case (3x margin under 1e-3)
BASE_NODES = 372736.0
BASE_MS = 21.8  # per-slot cost at 372736 nodes (background)


def point_field(R, u):
    R = np.asarray(R, float)
    u = np.asarray(u, float)
    uu2 = float(np.dot(u, u))
    inv_g2 = 1.0 / (1.0 + uu2)
    r2 = np.sum(R * R, axis=-1)
    if uu2 == 0.0:
        E = R / (r2 ** 1.5)[..., None]
        B = np.zeros_like(R)
        return E, B
    beta = u / np.sqrt(1.0 + uu2)
    rb = np.sum(R * beta, axis=-1)
    D = r2 * inv_g2 + rb * rb
    E = inv_g2 * R / (D ** 1.5)[..., None]
    B = np.cross(beta, E)
    return E, B


def tsc(x):
    x = np.asarray(x, float)
    ax = np.abs(x)
    w = np.where(ax < 0.5, 0.75 - ax * ax, 0.0)
    outer = (ax >= 0.5) & (ax < 1.5)
    return np.where(outer, 0.5 * (1.5 - ax) ** 2, w)


def _gauss(n):
    return gauss_legendre_pure(n)


def box_field(x_obs, x_src, cells, n):
    """Isotropic analytic box field (divergence theorem, 6 faces, Gauss n/face)."""
    xn, wn = _gauss(n)
    cells = np.asarray(cells, float)
    ext = np.array([1.5 * cells[0], 1.5 * cells[1], 1.5 * cells[2]])
    lo = np.asarray(x_src, float) - ext
    hi = np.asarray(x_src, float) + ext
    x_obs = np.asarray(x_obs, float)
    out = np.zeros(3)
    for axis in range(3):
        for sgn in (-1, 1):
            fixed = lo[axis] if sgn < 0 else hi[axis]
            df = x_obs[axis] - fixed
            o0 = (axis + 1) % 3
            o1 = (axis + 2) % 3
            h0 = 0.5 * (hi[o0] - lo[o0]); m0 = 0.5 * (hi[o0] + lo[o0])
            h1 = 0.5 * (hi[o1] - lo[o1]); m1 = 0.5 * (hi[o1] + lo[o1])
            v0 = m0 + h0 * xn[None, :]
            v1 = m1 + h1 * xn[:, None]
            d0 = v0 - x_obs[o0]
            d1 = v1 - x_obs[o1]
            P = np.sum((h0 * h1) * wn[None, :] * wn[:, None] / np.sqrt(d0 * d0 + d1 * d1 + df * df))
            out[axis] += sgn * P
    return out


def _segs(t):
    out = []
    for a in range(3):
        s = []
        if t[a] > -1.5:
            s.append((-1.5, t[a]))
        if t[a] < 1.5:
            s.append((t[a], 1.5))
        out.append(s)
    return out


def remainder(R0, u, cells, n, w_offset, octant=True):
    """R = int_S (W(s)-w_offset) K(R0 - s*c) ds, plain Gauss n/axis on octants at t.
    Returns (E, B, n_octants)."""
    R0 = np.asarray(R0, float); c = np.asarray(cells, float); u = np.asarray(u, float)
    t = R0 / c
    tn, wn = _gauss(n)
    if octant:
        S0, S1, S2 = _segs(t)
    else:
        S0 = S1 = S2 = [(-1.5, 1.5)]
    acc = np.zeros(6)
    for (ax, bx) in S0:
        hx = 0.5 * (bx - ax); mx = 0.5 * (bx + ax)
        sx = mx + hx * tn; wx = hx * wn
        for (ay, by) in S1:
            hy = 0.5 * (by - ay); my = 0.5 * (by + ay)
            sy = my + hy * tn; wy = hy * wn
            for (az, bz) in S2:
                hz = 0.5 * (bz - az); mz = 0.5 * (bz + az)
                sz = mz + hz * tn; wz = hz * wn
                SX, SY, SZ = np.meshgrid(sx, sy, sz, indexing="ij")
                Ww = tsc(SX) * tsc(SY) * tsc(SZ) - w_offset
                WW = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
                R = np.stack([R0[0] - SX * c[0], R0[1] - SY * c[1], R0[2] - SZ * c[2]], axis=-1)
                E, B = point_field(R, u)
                w = WW * Ww
                acc[0] += np.sum(w * E[..., 0]); acc[1] += np.sum(w * E[..., 1]); acc[2] += np.sum(w * E[..., 2])
                acc[3] += np.sum(w * B[..., 0]); acc[4] += np.sum(w * B[..., 1]); acc[5] += np.sum(w * B[..., 2])
    return acc[:3], acc[3:], len(S0) * len(S1) * len(S2)


def full_field(x_obs, x_src, u, cells, n_rem, n_box=16, octant=True):
    """F(n) = S_part(exact box) + R(n).  Returns (E, B, n_octants)."""
    cells = np.asarray(cells, float)
    R0 = np.asarray(x_obs, float) - np.asarray(x_src, float)
    t = R0 / cells
    Wt = float(tsc(t[0]) * tsc(t[1]) * tsc(t[2]))
    Sp = box_field(x_obs, x_src, cells, n_box) * (Wt / (cells[0] * cells[1] * cells[2]))
    Re, Br, n_oct = remainder(R0, u, cells, n_rem, Wt, octant)
    return Sp + Re, Br, n_oct


def ref_field(x_obs, x_src, u, cells, order=12, refine=3):
    """Independent reference (indfield.py). Returns (E, B)."""
    cells = np.asarray(cells, float)
    e, b = indfield.integrate_point(x_obs, x_src, u, cells[0], cells[1], cells[2],
                                    order=order, refine=refine, chunk=8)
    return np.asarray(e, float), np.asarray(b, float)


def u_for(gamma, dirv):
    d = np.asarray(dirv, float); d = d / np.linalg.norm(d)
    return np.sqrt(max(gamma * gamma - 1.0, 0.0)) * d


def positions(n_dir=12, radii=(0.5, 1.0), face=1.49):
    """In-support positions t = R0/cells in [-1.5,1.5]^3.  Azimuths (fib sphere)
    scaled to interior (0.5) and to the face (1.0 -> dominant axis at +/-face)."""
    pos = [np.zeros(3)]  # center
    N = n_dir
    ga = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        y = 1 - 2 * (i + 0.5) / N
        r = np.sqrt(max(1 - y * y, 0.0))
        th = ga * i
        d = np.array([r * np.cos(th), y, r * np.sin(th)])
        d = d / np.linalg.norm(d)
        for rad in radii:
            scale = rad * face / np.max(np.abs(d))
            pos.append(d * scale)
    for a in range(3):
        for s in (-1, 1):
            p = np.zeros(3); p[a] = s * face
            pos.append(p)
    pos.append(np.array([1.4, 1.4, 1.4]))
    return [np.asarray(p, float) for p in pos]
