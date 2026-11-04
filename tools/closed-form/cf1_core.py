"""NS3b core: closed-form in-support field.

Stationary in-support field (cell units, cells = (1,1,1)):
    F(R0) = W(s*) * E_box8T(R0) + R_rem
    s* = R0/cells (observation point in cell units)
    W(s*) = prod_d W_1d(s*_d)                      (TSC shape)
    E_box8T = field of a uniform unit-density support box [-1.5,1.5]^3 at R0
              (8 x T closed form, O(1))
    R_rem = int (S(s) - W(s*)) K(s* - s) d^3 s     (smooth-ish 1/r remainder)

Moving (beamed) kernel:  E_beam(R) = (1-beta^2) R / D^{3/2}, D = r^2 (1-beta^2) + (R.beta)^2
    singular part:  W(s*) * E_box_beam,  E_box_beam = int_support E_beam(R0 - s) d^3 s
    axial (beta || axis):  E_box_beam = (1/gamma) (E'_x, gamma E'_y, gamma E'_z)
                           with E' the Coulomb 8xT field of the gamma-stretched box
    oblique:  E_box_beam = int_{box'} (-M^-1 d')/|d'|^3 d^3 d',  box' = M * box  (no closed form)
"""
from __future__ import annotations
import math
import numpy as np
from indfield import tsc_shape, point_field, integrate_point, integrate_point_converged

# ---------------------------------------------------------------- primitives
def T(A, B, u):
    """int_0^A int_0^B 1/sqrt(u^2 + X^2 + Y^2) dX dY  (verified NS3).

    CF1 fix: the u->0 branch must fire for |u| < eps, not u < eps; the closed
    form is even in u, so negative u (field point beyond a box face) must use
    the full formula.  Old branch broke the eight-term form for all points
    outside the box in any axis (interior/face points unaffected: u >= 0).
    """
    if abs(u) < 1e-300:
        return A * math.asinh(B / A) + B * math.asinh(A / B)
    return (A * math.asinh(B / math.sqrt(u * u + A * A))
            + B * math.asinh(A / math.sqrt(u * u + B * B))
            - u * math.atan(A * B / (u * math.sqrt(A * A + u * u + B * B))))


def box_field_8T(p, lo, hi):
    """Coulomb field at p of unit-density box [lo,hi]^3, in cell units. 8 x T."""
    E = np.zeros(3)
    ax0, ay0, az0 = lo
    ax1, ay1, az1 = hi
    # E_i(p) = int (p - s)/|p - s|^3 dV = div-thm face integral:
    # E_x = sum_{y,z corners} [T(Ay,Az, u_xp) - T(Ay,Az, u_xm)]  (NS3 _conv.py form)
    def sx_edge(): return (ax1 - p[0], p[0] - ax0)
    def sy_edge(): return (ay1 - p[1], p[1] - ay0)
    def sz_edge(): return (az1 - p[2], p[2] - az0)
    for sy in sy_edge():
        for sz in sz_edge():
            E[0] += T(sy, sz, ax1 - p[0]) - T(sy, sz, p[0] - ax0)
    for sx in sx_edge():
        for sz in sz_edge():
            E[1] += T(sx, sz, ay1 - p[1]) - T(sx, sz, p[1] - ay0)
    for sx in sx_edge():
        for sy in sy_edge():
            E[2] += T(sx, sy, az1 - p[2]) - T(sx, sy, p[2] - az0)
    return E


def support_box_field(p, stretch_axis=None, gamma=1.0, sstar=None):
    """8xT Coulomb field at p of the support box (optionally stretched along
    stretch_axis by gamma, with the stretch applied about the field point, i.e.
    to the box expressed in d = s - p coordinates)."""
    if sstar is None:
        sstar = p
    lo = -1.5 - sstar
    hi = 1.5 - sstar
    if stretch_axis is not None:
        hi[stretch_axis] = gamma * hi[stretch_axis]
        lo[stretch_axis] = gamma * lo[stretch_axis]
    # field at the origin of the d-coordinates == field at p of the original box
    return box_field_8T(np.zeros(3), lo, hi)


def S_shape(s):
    """TSC shape S(s) = W(sx) W(sy) W(sz), s in cell units (vector or (...,3))."""
    return tsc_shape(s[..., 0]) * tsc_shape(s[..., 1]) * tsc_shape(s[..., 2])


def W_at(sstar):
    """3D TSC shape S(s*) = W(sx) W(sy) W(sz) at the projection point."""
    sstar = np.asarray(sstar, dtype=float)
    return float(tsc_shape(sstar[0]) * tsc_shape(sstar[1]) * tsc_shape(sstar[2]))


# ------------------------------------------------------------- remainder GL
def _panels_axis(sstar_i, scheme):
    """Intervals of [-1.5,1.5] for one axis, split at knots and (optionally) s*_i."""
    knots = [-1.5, -0.5, 0.5, 1.5]
    if scheme == "P0":
        return [[-1.5, 1.5]]
    if scheme == "P4":
        if -1.5 < sstar_i < 1.5:
            return [[-1.5, sstar_i], [sstar_i, 1.5]]
        return [[-1.5, 1.5]]
    pts = list(knots)
    if scheme in ("P2", "P3") and -1.5 < sstar_i < 1.5:
        pts = sorted(set(pts) | {sstar_i})
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def _axis_rule(a, b, sstar_axis, scheme, order, xi, wi, power):
    """(s nodes, |ds/dt| weights) for one axis interval [a,b]."""
    if scheme == "P3":
        if abs(a - sstar_axis) <= 1e-12:
            s = a + (b - a) * (0.5 * xi + 0.5) ** power
            jac = 0.5 * (b - a) * power * (0.5 * xi + 0.5) ** (power - 1)
        elif abs(b - sstar_axis) <= 1e-12:
            s = b - (b - a) * (0.5 * xi + 0.5) ** power
            jac = 0.5 * (b - a) * power * (0.5 * xi + 0.5) ** (power - 1)
        else:
            s = 0.5 * (a + b) + 0.5 * (b - a) * xi
            jac = np.full(order, 0.5 * (b - a))
        return s * 1.0, jac * wi
    s = 0.5 * (a + b) + 0.5 * (b - a) * xi
    return s, np.full(order, 0.5 * (b - a)) * wi


def remainder_panels(sstar, scheme, order, power=5):
    """Return (nodes (N,3) in s-units, weights (N,)) for the remainder integral.

    P0: whole box, plain GL.
    P1: knot split (27 panels), plain GL.
    P2: knot + s* split, plain GL.
    P3: knot + s* split, power transform t^power from the s* edge on axes where
        the panel edge touches s* (same scheme as indfield/GPU1).
    """
    xi, wi = np.polynomial.legendre.leggauss(order)
    pa = _panels_axis(sstar[0], scheme)
    pb = _panels_axis(sstar[1], scheme)
    pc = _panels_axis(sstar[2], scheme)
    nodes = []
    weights = []
    for (ax_, bx_) in pa:
        for (ay_, by_) in pb:
            for (az_, bz_) in pc:
                sx, wx = _axis_rule(ax_, bx_, sstar[0], scheme, order, xi, wi, power)
                sy, wy = _axis_rule(ay_, by_, sstar[1], scheme, order, xi, wi, power)
                sz, wz = _axis_rule(az_, bz_, sstar[2], scheme, order, xi, wi, power)
                SX = np.broadcast_to(sx[:, None, None], (order, order, order))
                SY = np.broadcast_to(sy[None, :, None], (order, order, order))
                SZ = np.broadcast_to(sz[None, None, :], (order, order, order))
                WF = (np.broadcast_to(wx[:, None, None], (order, order, order))
                      * np.broadcast_to(wy[None, :, None], (order, order, order))
                      * np.broadcast_to(wz[None, None, :], (order, order, order))).ravel()
                nodes.append(np.stack([SX, SY, SZ], axis=-1).reshape(-1, 3))
                weights.append(WF)
    return (np.concatenate(nodes), np.concatenate(weights))


def evaluate_remainder(nodes, weights, sstar, u, scheme_weight="sub"):
    """R_rem = int (S - W*) K d^3s  (stationary K) or beamed K.

    scheme_weight: 'sub' -> (S-W)*K ; 'full' -> S*K (no subtraction, GPU1 style).
    Returns (E (3,), B (3,)).
    """
    s = nodes
    S = S_shape(s)
    R = sstar - s                       # field point minus source (cell units)
    # floor: a node that rounds exactly onto the singular point is a
    # measure-zero pathology (float rounding of the power transform); treat
    # it as 1e-14 away (only fires for high order/refine combinations).
    if u is None or (isinstance(u, np.ndarray) and not np.any(u)):
        r2 = np.maximum(np.einsum("ij,ij->i", R, R), 1e-28)
        r = np.sqrt(r2)
        K = R / r2[:, None] / r[:, None]
        B = np.zeros_like(K)
    else:
        uu2 = float(np.sum(u * u))
        inv_g2 = 1.0 / (1.0 + uu2)
        gam = math.sqrt(1.0 + uu2)
        beta = u / gam
        r2 = np.einsum("ij,ij->i", R, R)
        rb = R @ beta
        D = np.maximum(r2 * inv_g2 + rb * rb, 1e-28)
        K = (inv_g2 * R) / (D ** 1.5)[:, None]
        B = np.cross(beta, K)
    if scheme_weight == "sub":
        w = weights * (S - W_at(sstar))
    elif scheme_weight == "full":
        w = weights * S
    else:  # uniform: unit density (the singular box)
        w = weights
    E = np.einsum("i,ij->j", w, K)
    Bv = np.einsum("i,ij->j", w, B)
    return E, Bv


def insupport_field(sstar, u, scheme, order, power=5, subtract=True):
    """Full in-support field by extraction + remainder quadrature.

    stationary: W*E_box8T + rem
    moving axial: W*(E'_x/g, E'_y, E'_z) [stretched box] + rem  (passed via boxfield)
    Returns (E, B, n_nodes).
    """
    if u is None or (isinstance(u, np.ndarray) and not np.any(u)):
        box = support_box_field(sstar)
        Bbox = np.zeros(3)
    else:
        uu2 = float(np.sum(u * u))
        gam = math.sqrt(1.0 + uu2)
        beta = u / gam
        # axial if beta parallel to an axis
        n = beta / np.linalg.norm(beta)
        axial = None
        for ax in range(3):
            if abs(abs(n[ax]) - 1.0) < 1e-12:
                axial = ax
                break
        if axial is not None:
            Eprim = support_box_field(sstar, stretch_axis=axial, gamma=gam)
            box = Eprim.copy()
            box[axial] /= gam
            Bbox = np.cross(beta, box)
        else:
            raise ValueError("oblique singular part: use insupport_field_oblique")
    nodes, weights = remainder_panels(sstar, scheme, order, power)
    if subtract:
        rem_e, rem_b = evaluate_remainder(nodes, weights, sstar, u, "sub")
        E = W_at(sstar) * box + rem_e
        B = W_at(sstar) * Bbox + rem_b
    else:
        rem_e, rem_b = evaluate_remainder(nodes, weights, sstar, u, "full")
        E, B = rem_e, rem_b
    return E, B, len(nodes)


# ------------------------------------------------------------- oblique part
def M_boost(u):
    uu2 = float(np.sum(u * u))
    gam = math.sqrt(1.0 + uu2)
    beta = u / gam
    n = beta / np.linalg.norm(beta)
    return gam, np.eye(3) + (gam - 1.0) * np.outer(n, n), beta


def oblique_box_beam(sstar, u, order=8, power=5):
    """E_box_beam for oblique beta, no closed form: direct quadrature.

    E_box_beam = int_{C0} gamma (-d) / |M d|^3 d^3 d   (lab frame, d = s - s*)
    Quadrature: 8 octants at d=0, power transform toward d=0, GL `order`.
    (Equivalently the rest-frame Coulomb integral over the sheared box M*C0.)
    """
    gam, Mm, beta = M_boost(u)
    # C0 in d = s - s* coords: faces at d_i = -1.5 - s*_i (lo) and 1.5 - s*_i (hi).
    # octant sigma_i: d_i in [0, hi] (sigma=+1) or [lo, 0] (sigma=-1).
    Lp = 1.5 - sstar          # extent on the + side (d_i in [0, Lp])
    Ln = 1.5 + sstar          # extent on the - side (d_i in [-Ln, 0])
    signs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    xi, wi = np.polynomial.legendre.leggauss(order)
    t = 0.5 * xi + 0.5
    acc = np.zeros(3)
    for (sx, sy, sz) in signs:
        L = np.array([Lp[0] if sx > 0 else Ln[0],
                      Lp[1] if sy > 0 else Ln[1],
                      Lp[2] if sz > 0 else Ln[2]])
        if min(L) <= 1e-14:
            continue

        tp = t ** power
        jx = (0.5 * L[0] * power * t ** (power - 1) * wi).reshape(order, 1, 1)
        jy = (0.5 * L[1] * power * t ** (power - 1) * wi).reshape(1, order, 1)
        jz = (0.5 * L[2] * power * t ** (power - 1) * wi).reshape(1, 1, order)
        J = jx * jy * jz
        T1 = np.broadcast_to(sx * L[0] * tp[:, None, None], (order, order, order))
        T2 = np.broadcast_to(sy * L[1] * tp[None, :, None], (order, order, order))
        T3 = np.broadcast_to(sz * L[2] * tp[None, None, :], (order, order, order))
        dvec = np.stack([T1, T2, T3], axis=-1)               # (n,n,n,3) d (lab frame)
        dprim = dvec @ Mm.T                                  # M d; rho = |M d|
        rp = np.linalg.norm(dprim, axis=-1)
        integrand = (-gam * dvec) / (rp ** 3)[..., None]     # gamma * (-d) / rho^3
        acc += np.sum(integrand * J[..., None], axis=(0, 1, 2))
    E = acc
    B = np.cross(beta, E)
    return E, B, 8 * order ** 3


def insupport_field_oblique(sstar, u, order_box=8, power=5, rem_scheme="P2", rem_order=8,
                            rem_power=5):
    """Full moving in-support field for oblique beta.

    singular part by rest-frame quadrature (order_box), remainder by lab-frame panels.
    """
    Ebox, Bbox, nbox = oblique_box_beam(sstar, u, order=order_box, power=power)
    nodes, weights = remainder_panels(sstar, rem_scheme, rem_order, rem_power)
    rem_e, rem_b = evaluate_remainder(nodes, weights, sstar, u, "sub")
    E = W_at(sstar) * Ebox + rem_e
    B = W_at(sstar) * Bbox + rem_b
    return E, B, nbox + len(nodes)


# ------------------------------------------------------------------ testing
def ref_converged(sstar, u, tol=1e-10):
    x_obs = sstar
    return integrate_point_converged(x_obs, np.zeros(3), np.asarray(u, float),
                                     1, 1, 1, tol=tol, order=12,
                                     max_order=32, refine=0, max_refine=4, chunk=8)


def ref_fixed(sstar, u, order=12, refine=3):
    """Fixed high-order reference (order 12, refine 3): established converged
    setting of this run (GPU1: 'converged ref o10r3==o10r4'; K3 refs o12r3)."""
    x_obs = sstar
    return integrate_point(x_obs, np.zeros(3), np.asarray(u, float),
                           1, 1, 1, order=order, refine=refine, chunk=8)


def interior_points(n=48, seed=7):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        p = rng.uniform(-1.4, 1.4, 3)
        pts.append(p)
    return np.array(pts)


def face_points(n=24, seed=11):
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        ax = rng.integers(3)
        s = rng.choice([-1.5, 1.5])
        p = rng.uniform(-1.2, 1.2, 3)
        p[ax] = s
        pts.append(p)
    return np.array(pts)


def rel_err(model, ref):
    """worst per-component |m-r| / (1 + max|r|)"""
    m = np.concatenate([model[0], model[1]])
    r = np.concatenate([ref[0], ref[1]])
    return float(np.max(np.abs(m - r)) / (1.0 + np.max(np.abs(r))))
