#!/usr/bin/env python3
"""Analytic and invariance self-checks required by guide E2.

Checks
------
1. Stationary Coulomb limit: the independent integrator is compared with
   SciPy's adaptive Gauss-Kronrod cubature on the same TSC convolution.  This
   uses a second, unrelated cubature implementation.
2. ``B`` is exactly zero for a stationary source.
3. Translation invariance: the field depends only on ``x_obs - x_src``.
4. Uniform length scaling by ``s``: ``E_n -> E_n / s**2`` and
   ``B_n -> B_n / s**2`` (``E_n`` has units 1/length**2).
5. Boundedness across the guide scope: gamma 1..100 and aspect 0.1..10.

Run::

    python selfcheck.py [--output report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import indfield  # noqa: E402

U10 = np.sqrt(99.0) * np.array([0.6, 0.5, 0.62]) / np.linalg.norm([0.6, 0.5, 0.62])


def _scipy_shape(x_obs, x_src, u, cells, rtol=1e-6):
    from scipy import integrate

    R0 = np.asarray(x_obs, float) - np.asarray(x_src, float)
    cells = np.asarray(cells, float)

    def integrand(s):
        R = R0[None, :] - s * cells[None, :]
        E, _ = indfield.point_field(R, u)
        w = (indfield.tsc_shape(s[:, 0]) * indfield.tsc_shape(s[:, 1])
             * indfield.tsc_shape(s[:, 2]))
        return E * w[:, None]

    res = integrate.cubature(
        integrand, np.full(3, -1.5), np.full(3, 1.5), rule="gk21",
        rtol=rtol, atol=0.0, max_subdivisions=1000,
        points=[R0 / cells], workers=1,
    )
    return res.estimate, float(np.max(res.error)), res.status


def _rel(a, b):
    scale = 1.0 + max(np.max(np.abs(a)), np.max(np.abs(b)))
    return float(np.max(np.abs(a - b)) / scale)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = {}

    # 1. Coulomb limit: stationary TSC convolution vs SciPy cubature.
    coulomb = []
    for label, xo in [("inside", [0.2, 0.3, 0.4]), ("outside", [2.6, 1.1, -0.7])]:
        e, b = indfield.integrate_point(xo, [0.0, 0.0, 0.0], np.zeros(3),
                                        1.0, 1.0, 1.0, order=12, refine=3)
        es, err, status = _scipy_shape(xo, [0.0, 0.0, 0.0], np.zeros(3), [1, 1, 1])
        coulomb.append({
            "point": label, "indfield": e.tolist(), "scipy": es.tolist(),
            "max_abs_diff": float(np.max(np.abs(e - es))),
            "scipy_error_estimate": err, "scipy_status": status,
        })
        print(f"coulomb {label}: max|ind-scipy| = {np.max(np.abs(e - es)):.3e}", flush=True)
    report["coulomb_limit"] = coulomb

    # 2. B exactly zero for a stationary source.
    _, b = indfield.integrate_point([0.2, 0.3, 0.4], [0, 0, 0], np.zeros(3),
                                    1.0, 1.0, 1.0, order=10, refine=2)
    report["stationary_B_exact_zero"] = bool(np.all(b == 0.0))
    print(f"stationary B exactly zero: {report['stationary_B_exact_zero']}")

    # 3. Translation invariance.
    base_args = dict(order=12, refine=2)
    e0, b0 = indfield.integrate_point([0.2, 0.3, 0.4], [-0.7, 0.9, 0.1], U10,
                                     1.0, 0.7, 1.3, **base_args)
    d = np.array([13.2, -7.4, 3.9])
    e1, b1 = indfield.integrate_point(np.add([0.2, 0.3, 0.4], d),
                                     np.add([-0.7, 0.9, 0.1], d), U10,
                                     1.0, 0.7, 1.3, **base_args)
    report["translation"] = {"rel_E": _rel(e0, e1), "rel_B": _rel(b0, b1)}
    print(f"translation: relE={_rel(e0, e1):.3e} relB={_rel(b0, b1):.3e}")

    # 4. Uniform scaling by s: E,B -> /s**2.
    scale = 3.7
    e0, b0 = indfield.integrate_point([0.2, 0.3, 0.4], [0.0, 0.0, 0.0], U10,
                                     1.0, 1.0, 1.0, **base_args)
    e1, b1 = indfield.integrate_point(np.array([0.2, 0.3, 0.4]) * scale,
                                     np.zeros(3), U10,
                                     scale, scale, scale, **base_args)
    report["length_scaling"] = {
        "scale": scale,
        "rel_E_after_1_over_s2": _rel(e0, e1 * scale * scale),
        "rel_B_after_1_over_s2": _rel(b0, b1 * scale * scale),
    }
    print(f"scaling s={scale}: relE={_rel(e0, e1*scale*scale):.3e} "
          f"relB={_rel(b0, b1*scale*scale):.3e}")

    # 5. Scope sweep: gamma 1..100 and aspect 0.1..10 stay finite and converge.
    sweep = []
    for gamma in (1.0, 10.0, 100.0):
        u = np.zeros(3) if gamma == 1.0 else np.sqrt(gamma**2 - 1) * np.array([0.6, 0.5, 0.62]) / np.linalg.norm([0.6, 0.5, 0.62])
        for aspect in (0.1, 1.0, 10.0):
            cells = (1.0, aspect, 1.0 / aspect)
            xo = np.array([0.37, 0.61, 0.29]) * np.array(cells)
            e8, _ = indfield.integrate_point(xo, np.zeros(3), u, *cells, order=8, refine=2)
            e12, _ = indfield.integrate_point(xo, np.zeros(3), u, *cells, order=12, refine=2)
            sweep.append({
                "gamma": gamma, "aspect": aspect,
                "finite": bool(np.all(np.isfinite(e12))),
                "order_change_rel": _rel(e8, e12),
            })
    report["scope_sweep"] = sweep
    print("scope sweep finite:", all(s["finite"] for s in sweep))

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
