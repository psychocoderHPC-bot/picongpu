#!/usr/bin/env python3
"""Point-level cross-check of the independent field against the in-repo one.

The full-tile comparison in ``compare_repo.py`` must use the in-repo fixed-rule
mode, which is not error-controlled inside the TSC support.  This script checks
selected points with the in-repo *adaptive* reference (its intended near path)
and SciPy Gauss-Kronrod cubature as a third opinion.

The in-repo adaptive path takes ~20 s per near point; the exact-singular
coincidence is excluded from that column because it does not return in a
practical time.

Run::

    python point_compare.py --output point_compare.json [--with-scipy]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "reference"))
import indfield  # noqa: E402
import convkernel as ck  # noqa: E402
from cases import default_cases  # noqa: E402

#: (case name, component index, observation point, use repo adaptive)
CHECKS = [
    ("stationary_center", 0, [0.2, 0.3, 0.4], True),
    ("gamma10_oblique", 1, [4.0, 3.5, 1.0], True),
    ("aspect10", 1, [4.0, 5.0, 0.2], True),
    ("source_at_output", 0, [1.5, 1.0, 1.0], False),
]


def _scipy_shape(x_obs, x_src, u, cells, rtol=1e-6):
    from scipy import integrate
    R0 = np.asarray(x_obs) - np.asarray(x_src)
    cells = np.asarray(cells)

    def f(s):
        R = R0[None, :] - s * cells[None, :]
        E, _ = indfield.point_field(R, u)
        return E * (indfield.tsc_shape(s[:, 0]) * indfield.tsc_shape(s[:, 1])
                    * indfield.tsc_shape(s[:, 2]))[:, None]

    res = integrate.cubature(f, np.full(3, -1.5), np.full(3, 1.5), rule="gk21",
                             rtol=rtol, atol=0.0, max_subdivisions=1000,
                             points=[R0 / cells], workers=1)
    return res.estimate, float(np.max(res.error)), res.status


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--with-scipy", action="store_true")
    args = parser.parse_args(argv)
    by_name = {c["name"]: c for c in default_cases()}

    rows = []
    for name, c, xo, use_repo in CHECKS:
        case = by_name[name]
        dx, dy, dz = case["cells"]
        cells = np.array(case["cells"])
        xs = np.array(case["source"])
        u = np.array(case["u"])
        xo = np.array(xo, dtype=np.float64)
        t = time.time()
        e, b = indfield.integrate_point(xo, xs, u, dx, dy, dz, order=8, refine=3)
        mine = e[c] if c < 3 else b[c - 3]
        t_ind = time.time() - t
        row = {
            "case": name, "component": indfield.COMPONENTS[c],
            "x_obs": xo.tolist(), "independent": float(mine),
            "independent_seconds": round(t_ind, 1),
        }
        if use_repo:
            t = time.time()
            er, br, info = ck.shape_field(xo, xs, u, dx, dy, dz, return_info=True)
            theirs = er[c] if c < 3 else br[c - 3]
            row.update({
                "repo_adaptive": float(theirs),
                "repo_info": info,
                "rel_diff": float(abs(mine - theirs) / (1.0 + max(abs(mine), abs(theirs)))),
                "repo_seconds": round(time.time() - t, 1),
            })
        if args.with_scipy:
            sv, se, st = _scipy_shape(xo, xs, u, cells)
            row.update({
                "scipy": float(sv[c] if c < 3 else 0.0),
                "scipy_error_estimate": se, "scipy_status": st,
                "independent_vs_scipy_rel": float(
                    abs(mine - sv[c]) / (1.0 + max(abs(mine), abs(sv[c])))),
            })
        rows.append(row)
        msg = f"{name:18s} {row['component']} ind={mine: .8e}"
        if "repo_adaptive" in row:
            msg += f" repo={row['repo_adaptive']: .8e} rel={row['rel_diff']:.3e} converged={row['repo_info']['converged']}"
        if "scipy" in row:
            msg += f" scipy={row['scipy']: .8e} rel={row['independent_vs_scipy_rel']:.2e}"
        print(msg, flush=True)

    text = json.dumps({"rows": rows}, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
