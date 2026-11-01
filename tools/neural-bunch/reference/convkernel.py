"""Independent FP64 physics reference for the neural bunch-field initializer.

Conventions (see conventions.md for the source citations)
-----------------------------------------------------------
* Lengths are physical (SI) unless a name ends in ``_cell``.
* ``u = p / (m_e c)`` is dimensionless and ``gamma = sqrt(1 + |u|^2)``.
* ``point_field`` and ``shape_field`` return the *per-unit-charge* fields

      E_n = 4*pi*eps0 * E / q ,      B_n = 4*pi*eps0 * c * B / q ,

  so that for a point charge ``E_n = (1-beta^2) R / D^(3/2)`` and
  ``B_n = beta x E_n``.  These are exactly the quantities that the network
  predicts before the signed charge factor ``q/e`` is applied.

* ``tile_field`` returns the normalized tile

      E_tilde = (q/e) * L^2 * E_n ,   B_tilde = (q/e) * L^2 * B_n ,
      L = dx , q/e = -w ,

  with ``w`` the particle weight.  The tile is indexed ``[i, j, k, c]`` with
  ``i < 8, j < 8, k < 4`` and ``c`` in
  ``(Ex, Ey, Ez, Bx, By, Bz)``.  Serialization order is
  ``c + 6*(i + 8*(j + 8*k))``; use :func:`flatten_tile` /
  :func:`unflatten_tile`.

There is no epsilon regularization anywhere: near-singular shape integrands are
resolved by adaptive per-axis panel subdivision and order increase.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "COMPONENTS",
    "OFFSETS",
    "TILE_SHAPE",
    "point_field",
    "shape_field",
    "tile_field",
    "tile_field_flat",
    "flatten_tile",
    "unflatten_tile",
    "output_positions",
    "tsc_shape",
]

COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")

#: Yee offsets in cell units, in COMPONENTS order.  Verified against
#: pico-pr17/include/picongpu/fields/YeeCell.hpp:80-82 and :124-126.
OFFSETS = np.array(
    [
        (0.5, 0.0, 0.0),  # Ex
        (0.0, 0.5, 0.0),  # Ey
        (0.0, 0.0, 0.5),  # Ez
        (0.0, 0.5, 0.5),  # Bx
        (0.5, 0.0, 0.5),  # By
        (0.5, 0.5, 0.0),  # Bz
    ],
    dtype=np.float64,
)

TILE_SHAPE = (8, 8, 4, 6)

#: Gauss-Legendre panels per shape axis (support of TSC is [-1.5, 1.5]).
BASE_PANELS = ((-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5))

_EPS_FLOAT = np.finfo(np.float64).eps


def _as_f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def gamma_of(u):
    """Lorentz factor from dimensionless momentum ``u``."""
    u = _as_f64(u)
    return np.sqrt(1.0 + np.sum(u * u, axis=-1))


def tsc_shape(x):
    """PIConGPU TSC assignment function in cell units.

    ``W(x) = 3/4 - x^2`` for ``|x| < 1/2``, ``1/2*(3/2-|x|)^2`` for
    ``1/2 <= |x| < 3/2``, zero otherwise (TSC.hpp:108-131, and the explicit
    piecewise form in TSC.hpp:110-114).
    """
    ax = np.abs(_as_f64(x))
    w = np.where(ax < 0.5, 0.75 - ax * ax, 0.0)
    outer = (ax >= 0.5) & (ax < 1.5)
    w = np.where(outer, 0.5 * (1.5 - ax) ** 2, w)
    return w


def point_field(R, u):
    """Per-unit-charge field of a uniformly moving point charge.

    Parameters
    ----------
    R : array_like, shape (..., 3)
        Present-time displacement ``x_obs - x_source(t)`` in physical units.
    u : array_like, shape (3,) or (..., 3)
        Dimensionless momentum ``p / (m_e c)``.

    Returns
    -------
    (E_n, B_n) : two arrays with the same shape as ``R``
        ``E_n = 4*pi*eps0 E/q`` and ``B_n = 4*pi*eps0 c B/q``.

    Notes
    -----
    Stable decomposition (no cancellation at large gamma)::

        invGamma2 = 1 / (1 + |u|^2)
        R_par     = R . bhat,       bhat = u / |u|
        R_perp    = R - R_par * bhat
        D         = R_par^2 + R_perp^2 * invGamma2
        E_n       = invGamma2 * R / D^(3/2)

    ``u == 0`` takes an explicit branch so that ``B_n`` is exactly zero.
    """
    R = _as_f64(R)
    u = _as_f64(u)
    R, u = np.broadcast_arrays(R, u) if u.shape != R.shape else (R, u)

    uu2 = np.sum(u * u, axis=-1)
    inv_g2 = 1.0 / (1.0 + uu2)
    speed = np.sqrt(uu2)
    is_static = speed == 0.0
    # Only used to keep the general branch finite; the static result is
    # selected below, so this is a branch guard and not a physical epsilon.
    safe_speed = np.where(is_static, 1.0, speed)
    bhat = u / safe_speed[..., None]
    beta = u / np.sqrt(1.0 + uu2)[..., None]

    r_par = np.sum(R * bhat, axis=-1)
    r_perp = R - r_par[..., None] * bhat
    denom = (r_par * r_par + np.sum(r_perp * r_perp, axis=-1) * inv_g2) ** 1.5

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        E_moving = (inv_g2[..., None] * R) / denom[..., None]
        B_moving = np.cross(beta, E_moving)

        r2 = np.sum(R * R, axis=-1)
        E_static = R / (r2**1.5)[..., None]

    E = np.where(is_static[..., None], E_static, E_moving)
    B = np.where(is_static[..., None], np.zeros_like(E_static), B_moving)
    return E, B


# ---------------------------------------------------------------------------
# Quadrature machinery
# ---------------------------------------------------------------------------


def _gauss_legendre(panels, order: int):
    """Gauss-Legendre nodes/weights on a list of intervals."""
    x, w = np.polynomial.legendre.leggauss(int(order))
    nodes = []
    weights = []
    for a, b in panels:
        half = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        nodes.append(half * x + mid)
        weights.append(half * w)
    return np.concatenate(nodes), np.concatenate(weights)


def _graded_panels(target, level, base=BASE_PANELS):
    """Geometrically graded per-axis panelling around ``target``.

    A panel that contains the projection of the singular point is split into
    intervals whose widths halve towards the target.  Panels more than one
    width away are left untouched, so the panel count grows linearly with the
    level while the 1/r^2 peak is resolved from both sides.  At ``level == 0``
    the three base panels are returned unchanged.
    """
    level = int(level)
    if level <= 0:
        return list(base)

    out = []
    for a, b in base:
        h = b - a
        if target < a - h or target > b + h:
            out.append((a, b))
            continue

        points = {a, b}
        if a <= target <= b:
            for k in range(level + 1):
                off = h * 0.5 * (0.5**k)
                points.add(min(b, max(a, target - off)))
                points.add(min(b, max(a, target + off)))
        elif target < a:
            for k in range(level + 1):
                points.add(min(b, a + h * (0.5**k)))
        else:
            for k in range(level + 1):
                points.add(max(a, b - h * (0.5**k)))

        ordered = sorted(points)
        out.extend((ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1) if ordered[i + 1] > ordered[i])
    return out


def _tensor_integral(nodes, weights_w, R0, cells, u, chunk):
    """Tensor-product integral of ``W * K`` on a fixed node grid.

    ``weights_w`` already contains the Gauss weight times the TSC factor for
    each axis.  Chunking over the x nodes bounds peak memory.
    """
    nx = nodes[0].size
    wx_all = weights_w[0]
    wy = weights_w[1]
    wz = weights_w[2]
    nx_nodes = nodes[0]
    ny_nodes = nodes[1]
    nz_nodes = nodes[2]

    acc_e = np.zeros(3, dtype=np.float64)
    acc_b = np.zeros(3, dtype=np.float64)

    for c0 in range(0, nx, int(chunk)):
        c1 = min(nx, c0 + int(chunk))
        sx = nx_nodes[c0:c1, None, None]
        sy = ny_nodes[None, :, None]
        sz = nz_nodes[None, None, :]

        rx = R0[0] - sx * cells[0]
        ry = R0[1] - sy * cells[1]
        rz = R0[2] - sz * cells[2]
        R = np.stack(np.broadcast_arrays(rx, ry, rz), axis=-1)

        e, b = point_field(R, u)
        w = wx_all[c0:c1, None, None] * wy[None, :, None] * wz[None, None, :]

        acc_e += np.sum(e * w[..., None], axis=(0, 1, 2))
        acc_b += np.sum(b * w[..., None], axis=(0, 1, 2))

    return acc_e, acc_b


def _integral_on_panels(panels, order, R0, cells, u, chunk):
    nodes = []
    weights_w = []
    for ax_panels in panels:
        n, w = _gauss_legendre(ax_panels, order)
        nodes.append(n)
        weights_w.append(w * tsc_shape(n))
    return _tensor_integral(nodes, weights_w, R0, cells, u, chunk)


def _relative_error(a, b):
    scale = 1.0 + max(np.max(np.abs(a)), np.max(np.abs(b)))
    return float(np.max(np.abs(a - b)) / scale)


def _adaptive_integral(R0, cells, u, order, tol, max_level, max_order, chunk):
    """Error-controlled per-axis grading plus order increase.

    Every observation point is checked: two Gauss orders are evaluated on the
    same panels and their difference is the Richardson-style estimate.  If it
    exceeds ``tol`` the per-axis panels are graded around the target projection
    and the order pair is raised by two.  At ``max_level`` the order keeps
    increasing on the same panels (reusing the previous high-order result)
    until ``max_order``.  No fixed epsilon is added to any denominator.
    """
    targets = (R0[0] / cells[0], R0[1] / cells[1], R0[2] / cells[2])
    level = 0
    order_lo = int(order)
    panels = [_graded_panels(targets[a], level) for a in range(3)]
    e_lo, b_lo = _integral_on_panels(panels, order_lo, R0, cells, u, chunk)
    e_hi, b_hi = _integral_on_panels(panels, order_lo + 2, R0, cells, u, chunk)

    while True:
        err = max(_relative_error(e_lo, e_hi), _relative_error(b_lo, b_hi))
        if err <= tol:
            return e_hi, b_hi, err, level, True
        if level < int(max_level):
            level += 1
            order_lo += 2
            panels = [_graded_panels(targets[a], level) for a in range(3)]
            e_lo, b_lo = _integral_on_panels(panels, order_lo, R0, cells, u, chunk)
        else:
            if order_lo + 2 >= int(max_order):
                return e_hi, b_hi, err, level, False
            order_lo += 2
            e_lo, b_lo = e_hi, b_hi
        e_hi, b_hi = _integral_on_panels(panels, order_lo + 2, R0, cells, u, chunk)


def _flat_panels(target, level):
    return _graded_panels(target, level)


def _near_support(R0, cells):
    """Physical distance to the TSC support box.

    ``excess_i = max(|R_i| - 1.5*cell_i, 0)`` and the reported distance is
    ``|excess|``.  The point is flagged near when this physical distance is
    below ``2 * max(dx,dy,dz)``.  The flag is diagnostic only: the quadrature
    is error-controlled for every point regardless of it.
    """
    excess = np.maximum(np.abs(R0) - 1.5 * cells, 0.0)
    return float(np.linalg.norm(excess)) < 2.0 * float(np.max(cells))


def shape_field(
    x_obs,
    x_src,
    u,
    dx,
    dy,
    dz,
    *,
    adaptive=True,
    near=None,
    order=4,
    order_hi=None,
    tol=1e-7,
    max_level=5,
    max_order=16,
    level=None,
    chunk=8,
    return_info=False,
):
    """Shape-integrated per-unit-charge field of one particle.

    Computes ``integral S(s) K(x_obs - x_src - s, u) ds^3`` with
    ``S(s) = W(sx) W(sy) W(sz)`` and physical offset
    ``s = (sx*dx, sy*dy, sz*dz)``.  The measure ``ds^3`` is dimensionless and
    ``integral S ds^3 = 1``.

    With ``adaptive=True`` every point is error-controlled: orders
    ``order`` and ``order+2`` are compared on the same per-axis panels.  If the
    difference exceeds ``tol``, panels are geometrically graded around the
    target projection and the order is raised, up to ``max_level`` levels and
    ``max_order``.  No fixed "far" branch bypasses the estimate.

    Parameters
    ----------
    x_obs, x_src : array_like, shape (3,)
        Physical observation and present-time source positions.
    u : array_like, shape (3,)
        Dimensionless momentum.
    dx, dy, dz : float
        Cell sizes.
    adaptive : bool
        Error-control the integral.  If False a fixed rule is used (``level``
        panel refinements and ``order`` points per panel), for tests only.
    near : bool or None
        Diagnostics override for the returned info; it does not affect the
        numerical path.
    order : int
        Starting low Gauss order.  The estimate uses ``order`` and ``order+2``.
    order_hi : int or None
        Unused compatibility argument.
    tol : float
        Tolerance for the order-difference estimate.
    max_level, max_order : int
        Grading-depth and order caps.
    level : int or None
        Fixed refinement level for ``adaptive=False``.
    chunk : int
        Number of x-nodes evaluated at once (memory bound).
    return_info : bool
        Also return a dict with error, level, near flag and convergence flag.

    Returns
    -------
    E_n, B_n : ndarray, shape (3,)
        Per-unit-charge fields in the normalization documented at module level.
    """
    x_obs = _as_f64(x_obs)
    x_src = _as_f64(x_src)
    u = _as_f64(u)
    cells = np.array([dx, dy, dz], dtype=np.float64)
    if np.any(cells <= 0.0):
        raise ValueError("cell sizes must be positive")

    R0 = x_obs - x_src
    is_near = _near_support(R0, cells) if near is None else bool(near)

    if not adaptive:
        lvl = 0 if level is None else int(level)
        panels = [
            _flat_panels(R0[0] / cells[0], lvl),
            _flat_panels(R0[1] / cells[1], lvl),
            _flat_panels(R0[2] / cells[2], lvl),
        ]
        e, b = _integral_on_panels(panels, order, R0, cells, u, chunk)
        info = {
            "error": None,
            "level": lvl,
            "near": is_near,
            "converged": True,
        }
        return (e, b, info) if return_info else (e, b)

    e, b, err, lvl, converged = _adaptive_integral(R0, cells, u, order, tol, max_level, max_order, chunk)
    info = {
        "error": err,
        "level": lvl,
        "near": is_near,
        "converged": converged,
    }
    return (e, b, info) if return_info else (e, b)


# ---------------------------------------------------------------------------
# Tile assembly
# ---------------------------------------------------------------------------


def output_positions(x_tile, dx, dy, dz):
    """Physical positions of all staggered outputs, shape (8, 8, 4, 6, 3)."""
    x_tile = _as_f64(x_tile)
    cells = np.array([dx, dy, dz], dtype=np.float64)
    pos = np.empty((8, 8, 4, 6, 3), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            for k in range(4):
                idx = np.array([i, j, k], dtype=np.float64)
                pos[i, j, k] = x_tile + (idx[None, :] + OFFSETS) * cells[None, :]
    return pos


def _base_integrals_many(R0, u, cells, order, chunk):
    """Fixed base-panel tensor integral at many observation points."""
    nodes = []
    ww = []
    for _ in range(3):
        n, w = _gauss_legendre(BASE_PANELS, order)
        nodes.append(n)
        ww.append(w * tsc_shape(n))
    grid = np.stack(np.meshgrid(nodes[0], nodes[1], nodes[2], indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3) * cells[None, :]
    weight = (ww[0][:, None, None] * ww[1][None, :, None] * ww[2][None, None, :]).reshape(-1)

    npts = R0.shape[0]
    E = np.empty((npts, 3), dtype=np.float64)
    B = np.empty((npts, 3), dtype=np.float64)
    max_rows = max(200_000, int(chunk) * grid.shape[0])
    step = max(1, max_rows // grid.shape[0])
    for p0 in range(0, npts, step):
        p1 = min(npts, p0 + step)
        r = R0[p0:p1, None, :] - grid[None, :, :]
        u_b = np.broadcast_to(_as_f64(u), r.shape)
        e, b = point_field(r, u_b)
        E[p0:p1] = np.sum(e * weight[None, :, None], axis=1)
        B[p0:p1] = np.sum(b * weight[None, :, None], axis=1)
    return E, B


def _shape_many(
    pos,
    x_src,
    u,
    cells,
    *,
    adaptive,
    order,
    tol,
    max_level,
    max_order,
    chunk,
):
    """Shape field at many observation points for a single source.

    The error estimate is vectorized on the base panels for all points; only
    points whose order pair has not converged fall back to the per-point
    graded/order-increased integral.  This keeps far tiles fast while giving
    every point the same convergence check.
    """
    pos = _as_f64(pos)
    R0 = pos - _as_f64(x_src)

    if not adaptive:
        return _base_integrals_many(R0, u, cells, order, chunk)

    lo = int(order)
    E_lo, B_lo = _base_integrals_many(R0, u, cells, lo, chunk)
    E_hi, B_hi = _base_integrals_many(R0, u, cells, lo + 2, chunk)

    diff = np.maximum(np.max(np.abs(E_lo - E_hi), axis=1), np.max(np.abs(B_lo - B_hi), axis=1))
    scale = 1.0 + np.maximum(
        np.maximum(np.max(np.abs(E_lo), axis=1), np.max(np.abs(E_hi), axis=1)),
        np.maximum(np.max(np.abs(B_lo), axis=1), np.max(np.abs(B_hi), axis=1)),
    )
    err = diff / scale
    converged = err <= tol

    E = E_hi.copy()
    B = B_hi.copy()
    for p in np.nonzero(~converged)[0]:
        E[p], B[p] = shape_field(
            pos[p],
            x_src,
            u,
            cells[0],
            cells[1],
            cells[2],
            adaptive=True,
            order=order,
            tol=tol,
            max_level=max_level,
            max_order=max_order,
            chunk=chunk,
        )
    return E, B


def _unpack_particle(p):
    if isinstance(p, dict):
        x = p["x"]
        u = p["u"]
        w = p.get("w", 1.0)
        qoe = p.get("q_over_e", -float(w))
    else:
        x, u, w = p[0], p[1], p[2]
        qoe = -float(w)
    return _as_f64(x), _as_f64(u), float(qoe)


def tile_field(
    x_tile,
    particles,
    dx,
    dy,
    dz,
    *,
    apply_charge=True,
    adaptive=True,
    order=4,
    tol=1e-7,
    max_level=5,
    max_order=16,
    chunk=8,
):
    """Normalized (8, 8, 4, 6) tile for a set of particles.

    ``particles`` is an iterable of mappings ``{'x': ..., 'u': ..., 'w': ...}``
    or of tuples ``(x, u, w)``.  A ``q_over_e`` key overrides ``-w``.

    The returned array is indexed ``[i, j, k, c]`` with
    ``c = (Ex, Ey, Ez, Bx, By, Bz)``.  The normalization is
    ``E_tilde = (q/e) * dx^2 * E_n`` and ``B_tilde = (q/e) * dx^2 * B_n``;
    with ``apply_charge=False`` the signed charge factor is not applied and the
    returned values are per-unit-charge normalized fields (the full sum over
    particles).  Use :func:`flatten_tile` for the serialized order
    ``c + 6*(i + 8*(j + 8*k))``.
    """
    x_tile = _as_f64(x_tile)
    cells = np.array([dx, dy, dz], dtype=np.float64)
    if np.any(cells <= 0.0):
        raise ValueError("cell sizes must be positive")

    tile = np.zeros(TILE_SHAPE, dtype=np.float64)
    pos = output_positions(x_tile, dx, dy, dz).reshape(-1, 3)

    # Empty input must produce an exact zero tile.
    for p in particles:
        x_src, u, qoe = _unpack_particle(p)
        if qoe == 0.0:
            continue
        e, b = _shape_many(
            pos,
            x_src,
            u,
            cells,
            adaptive=adaptive,
            order=order,
            tol=tol,
            max_level=max_level,
            max_order=max_order,
            chunk=chunk,
        )
        # Each of the 1536 staggered points asks for exactly one component:
        # E channels first, then B channels (COMPONENTS order).
        c = np.arange(e.shape[0]) % 6
        use_e = c < 3
        e_col = np.minimum(c, 2)
        b_col = np.maximum(c - 3, 0)
        rows = np.arange(e.shape[0])
        values = np.where(use_e, e[rows, e_col], b[rows, b_col])
        factor = (qoe if apply_charge else 1.0) * (dx * dx)
        tile += factor * values.reshape(TILE_SHAPE)

    return tile


def flatten_tile(tile):
    """Serialize ``[i, j, k, c]`` to ``c + 6*(i + 8*(j + 8*k))``."""
    tile = _as_f64(tile)
    if tile.shape != TILE_SHAPE:
        raise ValueError(f"expected tile shape {TILE_SHAPE}, got {tile.shape}")
    return tile.transpose(2, 1, 0, 3).reshape(-1).copy()


def unflatten_tile(flat):
    """Inverse of :func:`flatten_tile`."""
    flat = _as_f64(flat)
    if flat.size != 8 * 8 * 4 * 6:
        raise ValueError("expected 1536 values")
    return flat.reshape(4, 8, 8, 6).transpose(2, 1, 0, 3).copy()


def tile_field_flat(x_tile, particles, dx, dy, dz, **kwargs):
    """Convenience wrapper around :func:`tile_field` and :func:`flatten_tile`."""
    return flatten_tile(tile_field(x_tile, particles, dx, dy, dz, **kwargs))
