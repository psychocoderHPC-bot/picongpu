"""The frozen ``fourier_v1`` feature map (MODEL-FORMAT.md).

Feature vector, in this exact order:

1. the nine base inputs ``r_x, r_y, r_z, u_x, u_y, u_z, gamma, a_y, a_z``
2. for each axis ``x, y, z`` and each octave ``j`` in ``octaves`` (ascending),
   ``sin(pi * 2**j * r_axis)`` then ``cos(pi * 2**j * r_axis)``.

Axis-major, then octave ascending, then sin before cos.  The default octave
range ``[-3, 6]`` yields ``9 + 3 * 10 * 2 == 69`` features.
"""

from __future__ import annotations

import numpy as np

BASE_INPUTS = ["r_x", "r_y", "r_z", "u_x", "u_y", "u_z", "gamma", "a_y", "a_z"]
DEFAULT_OCTAVES = (-3, 6)
FEATURE_DIM = 69


def octave_list(octaves=DEFAULT_OCTAVES):
    lo, hi = octaves
    return list(range(int(lo), int(hi) + 1))


def feature_dim(kind="fourier_v1", octaves=None) -> int:
    # backward compatible: feature_dim((lo,hi)) still means fourier_v1
    if isinstance(kind, (tuple, list)):
        octaves = kind
        kind = "fourier_v1"
    if kind == "fourier_v1":
        oct = DEFAULT_OCTAVES if octaves is None else octaves
        return len(BASE_INPUTS) + 3 * len(octave_list(oct)) * 2
    if kind == "fourier_probe_v1":
        oct = PROBE_OCTAVES if octaves is None else octaves
        return len(BASE_INPUTS) + len(PROBE_CELL_OFFSETS) * 6 + 3 * len(octave_list(oct)) * 2
    if kind == "probe_dense_v1":
        return DENSE_DIM
    raise ValueError(f"unknown feature kind {kind!r}")


def _base(r, u, a_y, a_z) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if r.shape[-1] != 3 or u.shape[-1] != 3:
        raise ValueError("r and u must have a trailing axis of length 3")
    gamma = np.sqrt(1.0 + np.sum(u * u, axis=-1, keepdims=True))
    ay = np.asarray(a_y, dtype=np.float64)
    az = np.asarray(a_z, dtype=np.float64)
    if ay.ndim == 0:
        ay = ay[None]
    if az.ndim == 0:
        az = az[None]
    ay = np.broadcast_to(ay.reshape(-1, 1), (*r.shape[:-1], 1))
    az = np.broadcast_to(az.reshape(-1, 1), (*r.shape[:-1], 1))
    return np.concatenate([r, u, gamma, ay, az], axis=-1)


def fourier_v1(r, u, a_y, a_z, octaves=DEFAULT_OCTAVES) -> np.ndarray:
    """Return ``[..., 69]`` float64 features.

    ``r`` is in ``dx`` units relative to the tile cell origin, ``u`` is
    ``p/(m_e c)``, and ``a_y, a_z`` are ``dy/dx`` and ``dz/dx``.
    """
    base = _base(r, u, a_y, a_z)
    pieces = [base]
    r = np.asarray(r, dtype=np.float64)
    for axis in range(3):
        ra = r[..., axis:axis + 1]
        col_sin = []
        col_cos = []
        for j in octave_list(octaves):
            phase = np.pi * (2.0 ** j) * ra
            col_sin.append(np.sin(phase))
            col_cos.append(np.cos(phase))
        # octave-major interleave: sin_j0, cos_j0, sin_j1, cos_j1, ...
        inter = []
        for c in range(len(col_sin)):
            inter.append(col_sin[c])
            inter.append(col_cos[c])
        pieces.extend(inter)
    out = np.concatenate(pieces, axis=-1)
    return out.astype(np.float64, copy=False)


def fourier_v1_single(r, u, a_y, a_z, octaves=DEFAULT_OCTAVES) -> np.ndarray:
    return fourier_v1(np.asarray(r)[None, :], np.asarray(u)[None, :], a_y, a_z, octaves)[0]


# ---------------------------------------------------------------------------
# probe_v1: analytic point-charge probes inside the tile (additive encoder)
# ---------------------------------------------------------------------------
#: 4x4x2 probe grid at tile cell centres (cell units along x, y, z).
PROBE_OCTAVES = (-3, 1)
PROBE_CELL_OFFSETS = tuple(
    (2 * i + 0.5, 2 * j + 0.5, 2 * k + 0.5)
    for k in range(2)
    for j in range(4)
    for i in range(4)
)
PROBE_COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
PROBE_KIND = "fourier_probe_v1"
PROBE_DIM = len(BASE_INPUTS) + len(PROBE_CELL_OFFSETS) * 6 + 3 * len(octave_list(PROBE_OCTAVES)) * 2


def _probe_grid(a_y, a_z):
    """Probe positions in dx units from the tile origin, shape [..., P, 3]."""
    off = np.asarray(PROBE_CELL_OFFSETS, dtype=np.float64)  # [P,3]
    ay = np.asarray(a_y, dtype=np.float64)
    az = np.asarray(a_z, dtype=np.float64)
    scale = np.stack([np.ones_like(ay), ay, az], axis=-1)  # [...,3]
    return off * scale[..., None, :]


def point_field_dx(R, u):
    """Per-unit-charge field for ``R`` in dx units, ``u = p/(m_e c)``.

    Mirrors the stable uniform-motion decomposition used by the reference
    (``1/gamma^2`` direct, explicit static branch).  With ``R`` in dx units the
    returned values equal ``4*pi*eps0*dx^2*E/q`` and
    ``4*pi*eps0*c*dx^2*B/q``.
    """
    R = np.asarray(R, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    u = np.broadcast_to(u, R.shape)
    uu2 = np.sum(u * u, axis=-1)
    inv_g2 = 1.0 / (1.0 + uu2)
    speed = np.sqrt(uu2)
    static = speed == 0.0
    safe = np.where(static, 1.0, speed)
    bhat = u / safe[..., None]
    beta = u / np.sqrt(1.0 + uu2)[..., None]
    r_par = np.sum(R * bhat, axis=-1)
    r_perp = R - r_par[..., None] * bhat
    denom = (r_par * r_par + np.sum(r_perp * r_perp, axis=-1) * inv_g2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        E_mov = (inv_g2[..., None] * R) / denom[..., None]
        B_mov = np.cross(beta, E_mov)
        r2 = np.sum(R * R, axis=-1)
        E_st = R / (r2 ** 1.5)[..., None]
    E = np.where(static[..., None], E_st, E_mov)
    B = np.where(static[..., None], np.zeros_like(E_st), B_mov)
    return E, B


def probe_v1(r, u, a_y, a_z, octaves=PROBE_OCTAVES) -> np.ndarray:
    """Probe features, ordering: 9 base, 32x(E,B) probes, low-freq Fourier."""
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if r.ndim == 1:
        r = r[None, :]
    if u.ndim == 1:
        u = u[None, :]
    ay = np.asarray(a_y, dtype=np.float64)
    az = np.asarray(a_z, dtype=np.float64)
    if ay.ndim == 0:
        ay = np.full(r.shape[0], float(ay))
    if az.ndim == 0:
        az = np.full(r.shape[0], float(az))
    ay = np.broadcast_to(ay, (r.shape[0],))
    az = np.broadcast_to(az, (r.shape[0],))
    probes = _probe_grid(ay, az)                    # [N,P,3]
    R = probes - r[:, None, :]                      # x_obs - x_source
    u_rep = np.broadcast_to(u[:, None, :], R.shape)
    E, B = point_field_dx(R, u_rep)                 # [N,P,3] each
    eb = np.concatenate([E, B], axis=-1)            # [N,P,6]
    base = _base(r, u, ay, az)                      # [N,9]
    fourier = _fourier_terms(r, octaves)            # [N,30]
    return np.concatenate([base, eb.reshape(r.shape[0], -1), fourier], axis=-1)


def _fourier_terms(r, octaves):
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1:
        r = r[None, :]
    pieces = []
    for axis in range(3):
        ra = r[:, axis:axis + 1]
        for j in octave_list(octaves):
            ph = np.pi * (2.0 ** j) * ra
            pieces.append(np.sin(ph))
            pieces.append(np.cos(ph))
    return np.concatenate(pieces, axis=-1)


def compute_features(kind, r, u, a_y, a_z):
    if kind == "fourier_v1":
        return fourier_v1(r, u, a_y, a_z)
    if kind == PROBE_KIND:
        return probe_v1(r, u, a_y, a_z)
    if kind == DENSE_KIND:
        return probe_dense_v1(r, u, a_y, a_z)
    raise ValueError(f"unknown feature kind {kind!r}")


def feature_scale(kind):
    """Fixed per-feature divisor used only to condition the first MLP layer."""
    if kind == "fourier_v1":
        return np.array(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 2.0, 2.0]
            + [1.0] * 60, dtype=np.float64)
    if kind == PROBE_KIND:
        base = [64.0, 64.0, 64.0, 10.0, 10.0, 10.0, 100.0, 1.0, 1.0]
        probe = [0.05] * (len(PROBE_CELL_OFFSETS) * 6)
        four = [1.0] * (3 * len(octave_list(PROBE_OCTAVES)) * 2)
        return np.array(base + probe + four, dtype=np.float64)
    if kind == DENSE_KIND:
        base = [64.0, 64.0, 64.0, 10.0, 10.0, 10.0, 100.0, 1.0, 1.0]
        probe = [0.05] * (len(DENSE_CELL_OFFSETS) * 6)
        return np.array(base + probe, dtype=np.float64)
    raise ValueError(f"unknown feature kind {kind!r}")


# ---------------------------------------------------------------------------
# probe_dense_v1: one point-charge probe at every tile cell centre
# ---------------------------------------------------------------------------
DENSE_KIND = "probe_dense_v1"
DENSE_CELL_OFFSETS = tuple(
    (i + 0.5, j + 0.5, k + 0.5)
    for k in range(4)
    for j in range(8)
    for i in range(8)   # i fastest, then j, then k
)
DENSE_DIM = len(BASE_INPUTS) + len(DENSE_CELL_OFFSETS) * 6


def probe_dense_v1(r, u, a_y, a_z) -> np.ndarray:
    """9 base + 256 probes x [Ex,Ey,Ez,Bx,By,Bz]; no Fourier terms."""
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if r.ndim == 1:
        r = r[None, :]
    if u.ndim == 1:
        u = u[None, :]
    ay = np.asarray(a_y, dtype=np.float64)
    az = np.asarray(a_z, dtype=np.float64)
    if ay.ndim == 0:
        ay = np.full(r.shape[0], float(ay))
    if az.ndim == 0:
        az = np.full(r.shape[0], float(az))
    ay = np.broadcast_to(ay, (r.shape[0],))
    az = np.broadcast_to(az, (r.shape[0],))
    off = np.asarray(DENSE_CELL_OFFSETS, dtype=np.float64)
    scale = np.stack([np.ones_like(ay), ay, az], axis=-1)
    probes = off * scale[..., None, :]
    R = probes - r[:, None, :]
    u_rep = np.broadcast_to(u[:, None, :], R.shape)
    E, B = point_field_dx(R, u_rep)
    eb = np.concatenate([E, B], axis=-1).reshape(r.shape[0], -1)
    base = _base(r, u, ay, az)
    return np.concatenate([base, eb], axis=-1)
