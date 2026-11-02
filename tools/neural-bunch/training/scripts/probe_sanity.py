#!/usr/bin/env python3
"""Sanity check probe_v1 against the reference point-charge formula."""

import importlib.util
import sys
from pathlib import Path
import numpy as np

TRAIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN))
from neuralbunch.features import (  # noqa: E402
    DENSE_CELL_OFFSETS,
    DENSE_DIM,
    PROBE_CELL_OFFSETS,
    PROBE_DIM,
    compute_features,
    feature_dim,
    probe_dense_v1,
    probe_v1,
)

ref_path = TRAIN.parent / "reference" / "convkernel.py"
spec = importlib.util.spec_from_file_location("ck", ref_path)
ck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ck)

rng = np.random.default_rng(0)


def check(n, u, ay, az, label):
    r = rng.uniform(-8, 8, size=(n, 3))
    uu = np.broadcast_to(np.asarray(u, float), (n, 3)).copy()
    f = probe_v1(r, uu, ay, az)
    # independent reference evaluation at the same probes
    grid = np.asarray(PROBE_CELL_OFFSETS)
    probe = grid * np.array([1.0, ay, az])[None, :]
    R = probe[None, :, :] - r[:, None, :]
    Er, Br = ck.point_field(R, uu[:, None, :])
    eb = np.concatenate([Er, Br], axis=-1).reshape(n, -1)
    got = f[:, 9 : 9 + 192]
    d = np.abs(got - eb)
    scale = np.maximum(np.abs(eb), 1e-12)
    print(label, "n=%d finite=%s max_abs=%.3g max_rel=%.3g" % (n, np.isfinite(f).all(), d.max(), (d / scale).max()))
    assert np.isfinite(f).all()
    assert d.max() < 1e-10


check(8, 0.0, 1.0, 1.0, "static")
check(8, [30.0, 0, 0], 1.0, 1.0, "gamma100")
check(8, [0, 0, 99.995], 0.1, 10.0, "gamma100 aniso")
check(8, [1.5, -2.0, 0.3], 3.16227766, 0.316227766, "moving")
print(
    "dims",
    PROBE_DIM,
    feature_dim("fourier_probe_v1"),
    DENSE_DIM,
    feature_dim("probe_dense_v1"),
    feature_dim("fourier_v1"),
)
f = compute_features("fourier_v1", np.zeros((2, 3)), np.zeros((2, 3)), 1.0, 1.0)
print("fourier_v1 shape", f.shape)
f = compute_features("probe_dense_v1", np.zeros((2, 3)), np.zeros((2, 3)), 1.0, 1.0)
print("probe_dense_v1 shape", f.shape)
# dense probe exactness
n = 8
for u, ay, az, lab in [
    (0.0, 1, 1, "dense static"),
    ([30.0, 0, 0], 0.1, 10, "dense g100 aniso"),
    ([0, 0, 99.995], 3.162, 0.316, "dense g100"),
]:
    r = rng.uniform(-8, 8, (n, 3))
    uu = np.broadcast_to(np.asarray(u, float), (n, 3)).copy()
    f = probe_dense_v1(r, uu, ay, az)
    grid = np.asarray(DENSE_CELL_OFFSETS)
    probe = grid * np.array([1, ay, az])
    R = probe[None] - r[:, None]
    Er, Br = ck.point_field(R, uu[:, None])
    eb = np.concatenate([Er, Br], -1).reshape(n, -1)
    print(lab, "finite", bool(np.isfinite(f).all()), "maxdiff", float(np.abs(f[:, 9:] - eb).max()))
