"""pytest suite for the independent FP64 reference (convkernel.py).

Run from this directory with ``python -m pytest -q``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

import convkernel as ck

mpmath = pytest.importorskip("mpmath")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mp_point_field(R, u):
    """Independent high-precision direct evaluation (80 decimal digits)."""
    mpmath.mp.dps = 80
    Rm = [mpmath.mpf(float(v)) for v in R]
    um = [mpmath.mpf(float(v)) for v in u]
    g2 = 1 + sum(x * x for x in um)
    gamma = mpmath.sqrt(g2)
    beta = [x / gamma for x in um]
    b2 = sum(x * x for x in beta)
    R2 = sum(x * x for x in Rm)
    cross = [
        beta[1] * Rm[2] - beta[2] * Rm[1],
        beta[2] * Rm[0] - beta[0] * Rm[2],
        beta[0] * Rm[1] - beta[1] * Rm[0],
    ]
    D = R2 - sum(x * x for x in cross)
    pref = (1 - b2) / (D ** mpmath.mpf(1.5))
    E = [pref * x for x in Rm]
    B = [
        beta[1] * E[2] - beta[2] * E[1],
        beta[2] * E[0] - beta[0] * E[2],
        beta[0] * E[1] - beta[1] * E[0],
    ]
    return np.array([float(x) for x in E]), np.array([float(x) for x in B])


def _scipy_cubature_shape(x_obs, x_src, u, cells, *, rtol=1e-4):
    """Independent adaptive cubature cross-check (scipy, singular point given)."""
    x_obs = np.asarray(x_obs, float)
    x_src = np.asarray(x_src, float)
    cells = np.asarray(cells, float)
    R0 = x_obs - x_src

    def integrand(s):
        R = R0[None, :] - s * cells[None, :]
        E, _ = ck.point_field(R, u)
        w = (
            ck.tsc_shape(s[:, 0])
            * ck.tsc_shape(s[:, 1])
            * ck.tsc_shape(s[:, 2])
        )
        return E * w[:, None]

    res = integrate.cubature(
        integrand,
        np.array([-1.5, -1.5, -1.5]),
        np.array([1.5, 1.5, 1.5]),
        rule="gk21",
        rtol=rtol,
        atol=0.0,
        max_subdivisions=500,
        points=[R0 / cells],
        workers=1,
    )
    return res.estimate


# ---------------------------------------------------------------------------
# point kernel
# ---------------------------------------------------------------------------


def test_stationary_field_is_radial_and_electron_sign():
    R = np.array([1.0, -2.0, 3.0])
    E, B = ck.point_field(R, np.zeros(3))
    assert np.all(B == 0.0)
    assert np.allclose(E, R / np.linalg.norm(R) ** 3, rtol=0, atol=1e-14)
    assert np.linalg.norm(np.cross(E, R)) < 1e-14
    assert abs(np.linalg.norm(E) - 1.0 / np.dot(R, R)) < 1e-14

    # Full tile: q/e = -w.  A stationary electron left of the tile pulls E
    # towards itself, i.e. E_x at the tile is negative.
    tile = ck.tile_field(
        np.zeros(3),
        [{"x": np.array([-4.0, 0.0, 0.0]), "u": np.zeros(3), "w": 1.0}],
        1.0,
        1.0,
        1.0,
    )
    Ex_ref, _ = ck.shape_field(
        np.array([0.5, 0.0, 0.0]),
        np.array([-4.0, 0.0, 0.0]),
        np.zeros(3),
        1.0,
        1.0,
        1.0,
    )
    assert tile[0, 0, 0, 0] == pytest.approx(-Ex_ref[0], rel=1e-13)
    assert tile[0, 0, 0, 0] < 0.0
    assert np.all(tile[..., 3:] == 0.0)


@pytest.mark.parametrize(
    "u",
    [
        [0.7, 0.0, 0.0],
        [-0.7, 0.0, 0.0],
        [0.0, 0.7, 0.0],
        [0.0, -0.7, 0.0],
        [0.0, 0.0, 0.7],
        [0.0, 0.0, -0.7],
        [0.5, -0.2, 0.3],
    ],
)
def test_motion_orientation_and_magnitude(u):
    u = np.array(u)
    R = np.array([0.3, 1.2, -0.7])
    E, B = ck.point_field(R, u)
    gamma = math.sqrt(1.0 + np.dot(u, u))
    beta = u / gamma
    assert np.allclose(B, np.cross(beta, E), rtol=1e-14, atol=1e-16)
    # |B| = |v x E| / c^2 in normalized units.
    assert abs(np.linalg.norm(B) - np.linalg.norm(np.cross(beta, E))) < 1e-15
    assert np.all(np.isfinite(E)) and np.all(np.isfinite(B))


@pytest.mark.parametrize("gamma", [1.0, 2.0, 10.0, 100.0])
def test_gamma_stability_against_mpmath(gamma):
    u = np.array([math.sqrt(gamma * gamma - 1.0), 0.0, 0.0])
    for R in ([0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.2, -1.3, 2.1]):
        E, B = ck.point_field(np.array(R), u)
        Em, Bm = _mp_point_field(R, u)
        assert np.all(np.isfinite(E)) and np.all(np.isfinite(B))
        assert np.allclose(E, Em, rtol=2e-13, atol=0.0)
        assert np.allclose(B, Bm, rtol=2e-13, atol=1e-15)


# ---------------------------------------------------------------------------
# quadrature
# ---------------------------------------------------------------------------


def test_shape_normalization_is_one():
    nodes = []
    weights = []
    x, w = np.polynomial.legendre.leggauss(80)
    for a, b in ck.BASE_PANELS:
        nodes.append(0.5 * (b - a) * x + 0.5 * (a + b))
        weights.append(0.5 * (b - a) * w)
    nodes = np.concatenate(nodes)
    weights = np.concatenate(weights)
    total = np.sum(weights * ck.tsc_shape(nodes))
    assert total == pytest.approx(1.0, abs=1e-14)


def test_far_gauss_convergence_rate():
    u = np.array([0.3, 0.1, 0.0])
    R0 = np.array([4.0, 1.0, -2.0])
    ref, _ = ck.shape_field(
        R0, np.zeros(3), u, 1.0, 1.0, 1.0, adaptive=False, order=16
    )
    orders = [2, 3, 4, 5, 6]
    errors = []
    for o in orders:
        E, _ = ck.shape_field(
            R0, np.zeros(3), u, 1.0, 1.0, 1.0, adaptive=False, order=o
        )
        errors.append(np.max(np.abs(E - ref)))

    assert errors[-1] < 1e-10
    # Monotone decrease and at least an order of magnitude per two orders.
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))
    assert errors[2] < errors[0] * 1e-2


def test_near_point_cross_check_with_scipy_cubature():
    x_obs = np.array([0.0, 0.0, 1.0])
    x_src = np.zeros(3)
    u = np.zeros(3)
    E, B, info = ck.shape_field(
        x_obs, x_src, u, 1.0, 1.0, 1.0, return_info=True
    )
    assert info["near"] is True
    assert info["error"] is not None
    ref = _scipy_cubature_shape(x_obs, x_src, u, [1.0, 1.0, 1.0], rtol=1e-4)
    # The scipy run reports not_converged for the singular integrand; its
    # own error estimate is ~1e-6.  Demand 1e-4 agreement.
    assert np.max(np.abs(E - ref)) < 1e-4
    assert np.all(B == 0.0)


def test_point_charge_limit():
    u = np.array([0.4, -0.1, 0.2])
    R0 = np.array([1.0, 0.3, -0.2])
    eps = 1e-5
    E, B = ck.shape_field(
        R0 + np.array([2.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        u,
        eps,
        eps,
        eps,
        adaptive=False,
        order=8,
    )
    Ep, Bp = ck.point_field(R0, u)
    assert np.allclose(E, Ep, rtol=1e-9, atol=1e-14)
    assert np.allclose(B, Bp, rtol=1e-9, atol=1e-14)


# ---------------------------------------------------------------------------
# symmetries, scaling, aspects
# ---------------------------------------------------------------------------


def test_translation_invariance():
    u = np.array([0.2, 0.3, -0.1])
    shift = np.array([7.0, -3.0, 2.0])
    E1, B1 = ck.shape_field(
        np.array([3.0, 1.0, 2.0]),
        np.array([0.5, -0.5, 0.8]),
        u,
        1.0,
        1.0,
        1.0,
        adaptive=False,
        order=6,
    )
    E2, B2 = ck.shape_field(
        np.array([3.0, 1.0, 2.0]) + shift,
        np.array([0.5, -0.5, 0.8]) + shift,
        u,
        1.0,
        1.0,
        1.0,
        adaptive=False,
        order=6,
    )
    assert np.allclose(E1, E2, rtol=1e-13, atol=0.0)
    assert np.allclose(B1, B2, rtol=1e-13, atol=0.0)


def test_uniform_spatial_scaling_point_and_shape():
    u = np.array([0.1, 0.2, 0.3])
    R = np.array([1.0, -2.0, 0.5])
    E1, B1 = ck.point_field(R, u)
    lam = 3.0
    E2, B2 = ck.point_field(lam * R, u)
    assert np.allclose(E2, E1 / lam**2, rtol=1e-14, atol=0.0)
    assert np.allclose(B2, B1 / lam**2, rtol=1e-14, atol=0.0)

    xobs = np.array([2.0, 0.0, 0.0])
    Es, Bs = ck.shape_field(
        xobs, np.zeros(3), u, 1.0, 1.0, 1.0, adaptive=False, level=3, order=8
    )
    Es2, Bs2 = ck.shape_field(
        lam * xobs,
        np.zeros(3),
        u,
        lam,
        lam,
        lam,
        adaptive=False,
        level=3,
        order=8,
    )
    assert np.allclose(Es2, Es / lam**2, rtol=1e-9, atol=1e-13)
    assert np.allclose(Bs2, Bs / lam**2, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("aspect", [0.1, 1.0, 10.0])
def test_aspects(aspect):
    u = np.array([0.3, 0.0, 0.0])
    R0 = np.array([100.0, 20.0, -40.0])
    E, B, info = ck.shape_field(
        R0,
        np.zeros(3),
        u,
        1.0,
        aspect,
        aspect,
        adaptive=False,
        order=8,
        return_info=True,
    )
    assert np.all(np.isfinite(E)) and np.all(np.isfinite(B))
    assert np.linalg.norm(E) > 0.0
    # A far shape integral must stay close to the point kernel at its centre.
    Ep, _ = ck.point_field(R0, u)
    assert np.linalg.norm(E - Ep) / np.linalg.norm(Ep) < 0.02


ASPECTS = [0.1, 0.5, 1.0, 2.0, 10.0]
ASPECT_PAIRS = (
    [(a, a) for a in ASPECTS]
    + [(a, 1.0) for a in ASPECTS]
    + [(1.0, a) for a in ASPECTS]
)
ASPECT_DISTANCES = [4.0, 8.0, 32.0, 128.0, 512.0]
ASPECT_GAMMAS = [1.0, 10.0, 100.0]


def aspect_cases():
    """Aspect corners x distances x gamma, with dy and dz varied independently."""
    for dy, dz in ASPECT_PAIRS:
        for d in ASPECT_DISTANCES:
            for gamma in ASPECT_GAMMAS:
                cells = np.array([1.0, dy, dz])
                R0 = np.array([d, 0.3, -0.2]) * cells
                u = np.array(
                    [0.0 if gamma == 1.0 else math.sqrt(gamma * gamma - 1.0), 0.0, 0.0]
                )
                yield cells, R0, u, (dy, dz, d, gamma)


def test_aspect_corner_convergence():
    """Order 4/6/8 must agree to <=1e-6 over the full aspect grid."""
    worst = 0.0
    worst_case = None
    for cells, R0, u, case in aspect_cases():
        values = []
        for order in (4, 6, 8):
            E, _ = ck.shape_field(
                R0, np.zeros(3), u, *cells, order=order
            )
            values.append(E)
        ref = values[-1]
        scale = max(1e-300, float(np.max(np.abs(ref))))
        err = max(
            float(np.max(np.abs(values[0] - ref))),
            float(np.max(np.abs(values[1] - ref))),
        ) / scale
        if err > worst:
            worst = err
            worst_case = case
    print(f"aspect grid worst order4/6/8 rel error: {worst:.3e} at {worst_case}")
    assert worst <= 1e-6


# ---------------------------------------------------------------------------
# tile structure
# ---------------------------------------------------------------------------


def test_empty_and_zero_weight_are_exact_zero():
    zero = np.zeros(ck.TILE_SHAPE)
    assert np.array_equal(ck.tile_field(np.zeros(3), [], 1.0, 1.0, 1.0), zero)
    particles = [
        {"x": np.array([0.0, 0.0, 0.0]), "u": np.zeros(3), "w": 0.0},
    ]
    assert np.array_equal(
        ck.tile_field(np.zeros(3), particles, 1.0, 1.0, 1.0), zero
    )
    particles = [(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.0, 0.0]), 0.0)]
    assert np.array_equal(
        ck.tile_field(np.zeros(3), particles, 1.0, 1.0, 1.0), zero
    )


def test_offsets_and_flatten_order_are_pinned():
    expected = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    assert np.array_equal(ck.OFFSETS, expected)
    assert ck.COMPONENTS == ("Ex", "Ey", "Ez", "Bx", "By", "Bz")

    x_tile = np.array([1.0, 2.0, 3.0])
    cells = np.array([0.5, 2.0, 4.0])
    pos = ck.output_positions(x_tile, *cells)
    for i in range(8):
        for j in range(8):
            for k in range(4):
                for c in range(6):
                    want = x_tile + (np.array([i, j, k]) + expected[c]) * cells
                    assert np.array_equal(pos[i, j, k, c], want)

    tile = np.empty(ck.TILE_SHAPE)
    for i in range(8):
        for j in range(8):
            for k in range(4):
                for c in range(6):
                    tile[i, j, k, c] = c + 6 * (i + 8 * (j + 8 * k))
    flat = ck.flatten_tile(tile)
    assert np.array_equal(flat, np.arange(8 * 8 * 4 * 6))
    assert np.array_equal(ck.unflatten_tile(flat), tile)


def test_tile_matches_direct_shape_integral_and_charge_factor():
    u = np.array([0.2, -0.1, 0.05])
    x_src = np.array([-4.0, 0.0, 0.0])
    w = 2.5
    tile = ck.tile_field(
        np.zeros(3),
        [{"x": x_src, "u": u, "w": w}],
        1.0,
        1.0,
        1.0,
        apply_charge=True,
    )
    for i, j, k, c in [(0, 0, 0, 0), (3, 5, 2, 4), (7, 7, 3, 2)]:
        x = ck.output_positions(np.zeros(3), 1.0, 1.0, 1.0)[i, j, k, c]
        E, B = ck.shape_field(x, x_src, u, 1.0, 1.0, 1.0)
        want = -w * (E[c] if c < 3 else B[c - 3])
        assert tile[i, j, k, c] == pytest.approx(want, rel=1e-13)

    # Per-unit-charge variant must not apply q/e.
    tile_unit = ck.tile_field(
        np.zeros(3),
        [{"x": x_src, "u": u, "w": w}],
        1.0,
        1.0,
        1.0,
        apply_charge=False,
    )
    assert np.allclose(tile_unit, tile / (-w), rtol=1e-15, atol=0.0)


def test_single_particle_all_1536_slots_offsets_and_components():
    """Every slot selects its own component at its own staggered position."""
    x_tile = np.array([1.0, -2.0, 0.5])
    dx, dy, dz = 0.8, 1.1, 0.6
    cells = np.array([dx, dy, dz])
    x_src = x_tile + np.array([-32.0, 0.25, -0.35]) * cells
    u = np.array([0.4, -0.2, 0.1])
    w = 1.7
    beta = u / math.sqrt(1.0 + np.dot(u, u))

    tile = ck.tile_field(
        x_tile,
        [{"x": x_src, "u": u, "w": w}],
        dx,
        dy,
        dz,
        order=6,
    )
    pos = ck.output_positions(x_tile, dx, dy, dz)
    expected = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )

    for i in range(8):
        for j in range(8):
            for k in range(4):
                for c in range(6):
                    x = pos[i, j, k, c]
                    # The six offsets used by this 256 x 6 slot matrix.
                    want_x = x_tile + (np.array([i, j, k]) + expected[c]) * cells
                    assert np.array_equal(x, want_x)

                    E, B = ck.shape_field(
                        x, x_src, u, dx, dy, dz, adaptive=False, order=6
                    )
                    component = E[c] if c < 3 else B[c - 3]
                    want = -w * dx * dx * component
                    assert tile[i, j, k, c] == pytest.approx(want, rel=1e-12)

                    if c < 3:
                        # E orientation must follow the point-charge result.
                        Ep, _ = ck.point_field(x - x_src, u)
                        cosang = np.dot(E, Ep) / (
                            np.linalg.norm(E) * np.linalg.norm(Ep)
                        )
                        assert cosang > 1.0 - 1e-6
                    else:
                        # B component must equal (beta x E) component.
                        B_cross = np.cross(beta, E)
                        assert tile[i, j, k, c] == pytest.approx(
                            -w * dx * dx * B_cross[c - 3], rel=1e-11
                        )

    flat = ck.flatten_tile(tile)
    assert flat.shape == (1536,)
    assert np.array_equal(ck.unflatten_tile(flat), tile)


def test_stationary_particle_gives_exactly_zero_b():
    particles = [
        {"x": np.array([-4.0, 0.0, 0.0]), "u": np.zeros(3), "w": 1.0},
        {"x": np.array([0.0, -5.0, 0.0]), "u": np.zeros(3), "w": -2.0},
    ]
    tile = ck.tile_field(np.zeros(3), particles, 1.0, 1.0, 1.0)
    assert np.all(np.isfinite(tile))
    assert np.all(tile[..., 3:] == 0.0)
    E1, B1 = ck.shape_field(
        np.array([0.5, 0.0, 0.0]),
        np.array([-4.0, 0.0, 0.0]),
        np.zeros(3),
        1.0,
        1.0,
        1.0,
    )
    assert np.all(B1 == 0.0)
    assert np.all(np.isfinite(E1))
