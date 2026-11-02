"""Independent high-precision reference for the TSC-convolved Yee field.

This module deliberately uses a *different numerical formulation* from
``tools/neural-bunch/reference/convkernel.py`` (the in-repo reference) and from
the C++ production quadrature:

* the TSC support is split **at the knot points** ``-1.5, -0.5, 0.5, 1.5`` so
  the shape is a polynomial on every panel;
* when the target projects inside the support the panel is split again at the
  projection ``s* = R0 / cell`` and each resulting panel is mapped with the
  power transform ``s = c + (b - c) t**p`` (``p = TRANSFORM_POWER``).  The
  Jacobian ``(b-c) p t**(p-1)`` cancels the ``1/r**2`` point-kernel singularity,
  so the integrand is bounded at the formerly singular corner and **no epsilon
  is added to any denominator**;
* the 3-D integral is a tensor product of Gauss-Legendre rules computed at run
  time with :func:`numpy.polynomial.legendre.leggauss`.  No node/weight table is
  imported from the repository.

The point kernel is evaluated through the algebraically different but
equivalent form

    D = |R|**2 (1 - beta**2) + (R . beta)**2
    E_n = (1 - beta**2) R / D**1.5,   B_n = beta x E_n

instead of the ``R_par / R_perp`` decomposition used by the in-repo reference.
``u == 0`` takes an explicit branch so ``B_n`` is exactly zero.

Conventions match ``tools/neural-bunch/reference/conventions.md``:
``E_n = 4*pi*eps0 E/q``, ``B_n = 4*pi*eps0 c B/q``, lengths are physical,
``s`` is in cell units and ``ds**3`` is dimensionless with ``integral S ds**3 = 1``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "COMPONENTS", "OFFSETS", "TILE_SHAPE", "TSC_KNOTS", "TRANSFORM_POWER",
    "point_field", "tsc_shape", "integrate_point", "integrate_point_converged",
    "output_positions", "tile_field", "flatten_tile", "unflatten_tile",
    "tile_field_flat",
]

COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")

#: Yee offsets in cell units, in COMPONENTS order (same convention as the repo).
OFFSETS = np.array(
    [
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ],
    dtype=np.float64,
)

TILE_SHAPE = (8, 8, 4, 6)

#: TSC knot points; the shape ``W`` is a different quadratic on each interval.
TSC_KNOTS = (-1.5, -0.5, 0.5, 1.5)

#: Exponent of the singularity-cancelling coordinate transform.  ``p = 5``
#: makes the transformed integrand smooth enough at the singular corner that
#: partition refinement shows a clean order close to ``p``.
TRANSFORM_POWER = 5


def tsc_shape(x):
    """PIConGPU TSC assignment function, argument in cell units."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    w = np.where(ax < 0.5, 0.75 - ax * ax, 0.0)
    outer = (ax >= 0.5) & (ax < 1.5)
    return np.where(outer, 0.5 * (1.5 - ax) ** 2, w)


def point_field(R, u):
    """Per-unit-charge field of a uniformly moving point charge.

    ``R`` has shape ``(..., 3)``, ``u`` is ``(3,)`` or broadcastable to `R`.
    Returns ``(E_n, B_n)`` in the module normalization.
    """
    R = np.asarray(R, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    uu2 = np.sum(u * u, axis=-1)
    inv_g2 = 1.0 / (1.0 + uu2)          # 1 - beta**2
    gamma = np.sqrt(1.0 + uu2)
    beta = u / gamma[..., None] if np.ndim(gamma) else u / gamma

    r2 = np.sum(R * R, axis=-1)
    rbeta = np.sum(R * beta, axis=-1)
    # |R|^2 - |beta x R|^2 = |R|^2 (1-beta^2) + (R.beta)^2
    D = r2 * inv_g2 + rbeta * rbeta

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        E = (inv_g2[..., None] * R) / (D**1.5)[..., None]
        B = np.cross(beta, E)

    static = uu2 == 0.0
    if np.any(static):
        Es = R / (r2**1.5)[..., None]
        E = np.where(static[..., None], Es, E)
        B = np.where(static[..., None], 0.0, B)
    return E, B


def _gl(order):
    """Gauss-Legendre nodes/weights on ``[-1, 1]`` (computed at run time)."""
    return np.polynomial.legendre.leggauss(int(order))


def _base_intervals():
    return [(TSC_KNOTS[i], TSC_KNOTS[i + 1]) for i in range(len(TSC_KNOTS) - 1)]


def _axis_panels(s_star, singular):
    """Panels of one axis as ``(c, b, power)`` transformed to ``t in [0,1]``.

    ``s(t) = c + (b - c) t**power``.  When the target projects strictly inside
    the support box the knot intervals are split at ``s_star`` so the singular
    point becomes the endpoint ``c`` of every panel.
    """
    intervals = _base_intervals()
    if singular:
        out = []
        for a, b in intervals:
            if a < s_star < b:
                out.extend([(a, s_star), (s_star, b)])
            else:
                out.append((a, b))
        intervals = out

    panels = []
    for a, b in intervals:
        if singular:
            if abs(a - s_star) <= abs(b - s_star):
                c, other = a, b
            else:
                c, other = b, a
            panels.append((c, other, TRANSFORM_POWER))
        else:
            panels.append((a, b, 1))
    return panels


def _axis_rule(panels, order, refine):
    """Concatenated ``s`` nodes and Jacobian weights for one axis."""
    nseg = 2 ** int(refine)
    xi, wi = _gl(order)
    s_nodes = []
    weights = []
    for c, b, power in panels:
        db = b - c
        for seg in range(nseg):
            t0 = seg / nseg
            t1 = (seg + 1) / nseg
            mid = 0.5 * (t0 + t1)
            half = 0.5 * (t1 - t0)
            t = mid + half * xi
            s = c + db * t**power
            # |ds/dt|; the panel is always integrated in increasing s
            jac = abs(db) * power * t ** (power - 1)
            s_nodes.append(s)
            weights.append(wi * half * jac)
    return np.concatenate(s_nodes), np.concatenate(weights)


def _tensor_integral(R0, cells, u, panels, order, refine, chunk):
    sx, wx = _axis_rule(panels[0], order, refine)
    sy, wy = _axis_rule(panels[1], order, refine)
    sz, wz = _axis_rule(panels[2], order, refine)

    # Fold the separable TSC factor into the axis weights.
    wx = wx * tsc_shape(sx)
    wy = wy * tsc_shape(sy)
    wz = wz * tsc_shape(sz)

    acc_e = np.zeros(3)
    acc_b = np.zeros(3)
    nx = sx.size
    for c0 in range(0, nx, int(chunk)):
        c1 = min(nx, c0 + int(chunk))
        sxx = sx[c0:c1, None, None]
        syy = sy[None, :, None]
        szz = sz[None, None, :]
        R = np.stack(
            np.broadcast_arrays(
                R0[0] - sxx * cells[0],
                R0[1] - syy * cells[1],
                R0[2] - szz * cells[2],
            ),
            axis=-1,
        )
        e, b = point_field(R, u)
        w = wx[c0:c1, None, None] * wy[None, :, None] * wz[None, None, :]
        acc_e += np.sum(e * w[..., None], axis=(0, 1, 2))
        acc_b += np.sum(b * w[..., None], axis=(0, 1, 2))
    return acc_e, acc_b


def _check_cells(cells):
    cells = np.asarray(cells, dtype=np.float64)
    if cells.shape != (3,) or np.any(cells <= 0.0):
        raise ValueError("cells must be three positive numbers")
    return cells


def integrate_point(x_obs, x_src, u, dx, dy, dz, *, order=12, refine=0, chunk=8):
    """``integral S(s) K(x_obs - x_src - s*cells, u) ds**3`` at one point."""
    cells = _check_cells([dx, dy, dz])
    R0 = np.asarray(x_obs, dtype=np.float64) - np.asarray(x_src, dtype=np.float64)
    s_star = R0 / cells
    singular = bool(np.all(np.abs(s_star) < 1.5))
    panels = [_axis_panels(s_star[a], singular) for a in range(3)]
    return _tensor_integral(R0, cells, np.asarray(u, dtype=np.float64), panels, order, refine, chunk)


def _rel(a, b):
    a = np.concatenate([np.asarray(a[0]).ravel(), np.asarray(a[1]).ravel()])
    b = np.concatenate([np.asarray(b[0]).ravel(), np.asarray(b[1]).ravel()])
    scale = 1.0 + max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / scale)


def integrate_point_converged(x_obs, x_src, u, dx, dy, dz, *, tol=1e-10,
                              order=8, max_order=48, refine=0, max_refine=8, chunk=8):
    """Reference integral with an explicit order/partition estimator.

    The estimate is ``max|I_hi - I_lo| / (1 + max|I_hi|)`` over the six
    components.  Order is raised first, then the transformed partition is
    bisected.  The estimator is diagnostic only: it is never used inside the
    integral and no epsilon is added to the kernel.
    """
    def run(o, r):
        return integrate_point(x_obs, x_src, u, dx, dy, dz, order=o, refine=r, chunk=chunk)

    lo, hi = run(order, refine), run(order + 4, refine)
    info = {"order": order + 4, "refine": refine, "converged": False}
    while True:
        err = _rel(lo, hi)
        info["error"] = err
        if err <= tol:
            info["converged"] = True
            return hi[0], hi[1], info
        if order + 4 < max_order:
            order += 4
            lo, hi = hi, run(order + 4, refine)
            info["order"] = order + 4
        elif refine < max_refine:
            refine += 1
            order = max(8, order)
            lo, hi = run(order, refine), run(order + 4, refine)
            info["refine"] = refine
            info["order"] = order + 4
        else:
            return hi[0], hi[1], info


def output_positions(x_tile, dx, dy, dz):
    """Physical positions of all staggered outputs, shape ``(8, 8, 4, 6, 3)``."""
    x_tile = np.asarray(x_tile, dtype=np.float64)
    cells = np.array([dx, dy, dz], dtype=np.float64)
    pos = np.empty((8, 8, 4, 6, 3), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            for k in range(4):
                idx = np.array([i, j, k], dtype=np.float64)
                pos[i, j, k] = x_tile + (idx[None, :] + OFFSETS) * cells[None, :]
    return pos


def tile_field(x_tile, particles, dx, dy, dz, *, apply_charge=True,
               order=8, refine=0, refine_near=2, chunk=8):
    """Normalized ``(8, 8, 4, 6)`` tile ``[i, j, k, component]``.

    ``E_tilde = (q/e) dx**2 E_n``, ``B_tilde = (q/e) dx**2 B_n``, matching the
    project layout.  ``particles`` is an iterable of ``{'x', 'u', 'w'}``.
    """
    cells = _check_cells([dx, dy, dz])
    tile = np.zeros(TILE_SHAPE, dtype=np.float64)
    pos = output_positions(x_tile, dx, dy, dz).reshape(-1, 3)
    for p in particles:
        x_src = np.asarray(p["x"], dtype=np.float64)
        u = np.asarray(p["u"], dtype=np.float64)
        w = float(p.get("w", 1.0))
        qoe = float(p.get("q_over_e", -w))
        if qoe == 0.0:
            continue
        # Each of the 1536 staggered positions asks for exactly one component:
        # E channels first, then B channels (COMPONENTS order).
        vals = np.empty(pos.shape[0], dtype=np.float64)
        cells = np.array([dx, dy, dz])
        for n, x_obs in enumerate(pos):
            # Only outputs whose projected source lies inside the TSC box see
            # the near-singular integrand; they get the refined partition.
            s_star = (x_obs - x_src) / cells
            near = bool(np.all(np.abs(s_star) < 1.5))
            e, b = integrate_point(
                x_obs, x_src, u, dx, dy, dz,
                order=order, refine=(refine_near if near else refine), chunk=chunk,
            )
            c = n % 6
            vals[n] = e[c] if c < 3 else b[c - 3]
        factor = (qoe if apply_charge else 1.0) * (dx * dx)
        tile += factor * vals.reshape(TILE_SHAPE)
    return tile


def flatten_tile(tile):
    """Serialize ``[i, j, k, c]`` to ``c + 6*(i + 8*(j + 8*k))``."""
    tile = np.asarray(tile, dtype=np.float64)
    if tile.shape != TILE_SHAPE:
        raise ValueError(f"expected tile shape {TILE_SHAPE}, got {tile.shape}")
    return tile.transpose(2, 1, 0, 3).reshape(-1).copy()


def unflatten_tile(flat):
    flat = np.asarray(flat, dtype=np.float64)
    if flat.size != 8 * 8 * 4 * 6:
        raise ValueError("expected 1536 values")
    return flat.reshape(4, 8, 8, 6).transpose(2, 1, 0, 3).copy()


def tile_field_flat(x_tile, particles, dx, dy, dz, **kwargs):
    return flatten_tile(tile_field(x_tile, particles, dx, dy, dz, **kwargs))
