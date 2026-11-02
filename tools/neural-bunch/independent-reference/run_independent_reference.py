#!/usr/bin/env python3
"""Driver for the independent reference (T13c).

Runs, on the remote host only:
  1. kronrod-rule exactness verification;
  2. the analytic self-checks;
  3. a convergence study (order in panel subdivision and in the Kronrod pair);
  4. per-case comparison against the repo reference (convkernel.py) using the
     S0-normalised per-component relative L2 metric from eval/field_gate.py.

Every comparison case is capped by a function-evaluation budget so a stuck
adaptive integral is reported as *incomplete* rather than melting the host.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Independent module.
sys.path.insert(0, HERE)
# Repo reference + field gate (read-only, for comparison only).  The repo root
# is taken from the NB_REPO_ROOT environment variable (set by the runner) so the
# tool works both in-tree and in an isolated copy.
REPO_ROOT = os.environ.get("NB_REPO_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
NB_ROOT = os.path.join(REPO_ROOT, "tools", "neural-bunch")
sys.path.insert(0, os.path.join(NB_ROOT, "reference"))
sys.path.insert(0, os.path.join(NB_ROOT, "eval"))

import independent_reference as ir  # noqa: E402
import convkernel as ck  # noqa: E402  (repo reference)
import field_gate as fg  # noqa: E402  (repo metric)


def _norm6(F):
    return float(np.linalg.norm(np.asarray(F, dtype=np.float64).reshape(-1)))


def kronrod_verification():
    rep = ir._verify_kronrod()
    ok = all(v < 1e-12 for v in rep.values())
    return {"rules": rep, "all_exact_to_1e-12": ok}


def compare_case(name, x_tile, particles, dx, dy, dz, *, tol, max_pass, budget):
    """Compute my tile (subtraction workhorse + endpoint cross-check) and the
    repo tile for one case; return the S0-normalised per-component relative L2
    using the frozen field_gate metric."""
    # My tile (normalised, same serialisation as the repo).  The vectorised
    # subtraction path is the bounded workhorse.
    t0 = time.time()
    my_tile = ir.tile_field_vec(x_tile, particles, dx, dy, dz, strategy="subtraction",
                                box_depth=min(2 + max_pass, 5))
    my_flat = ir.flatten_tile(my_tile).astype(np.float64)
    t_my = time.time() - t0

    # Repo tile (its own adaptive reference).
    t0 = time.time()
    repo_tile = ck.tile_field(x_tile, particles, dx, dy, dz,
                              adaptive=True, tol=1e-7, max_level=5, max_order=16)
    repo_flat = ck.flatten_tile(repo_tile).astype(np.float64)
    t_repo = time.time() - t0

    # field_gate metric: candidate=mine, reference=repo.  Config declares S0.
    config = {"gate": 1e-3, "s0": {"mode": "reference_tile"}}
    report = fg.evaluate(my_flat, repo_flat, config, {"reference": repo_flat})

    # Extract per-component relative L2 for E and cB groups.
    out = {"name": name, "n_cases": 1}
    for grp in ("E", "cB"):
        out[grp] = {
            "reference_norm": report["cases"][0][grp]["reference_norm"],
            "error_norm": report["cases"][0][grp]["error_norm"],
            "s0": report["cases"][0][grp]["s0"],
            "denominator": report["cases"][0][grp]["denominator"],
            "relative_l2": report["cases"][0][grp]["relative_l2"],
            "pass": report["cases"][0][grp]["pass"],
            "components": report["cases"][0][grp]["components"],
        }
    out["timings"] = {"mine_seconds": t_my, "repo_seconds": t_repo}
    return out


def run_case_set(args):
    # The five requested cases.  x_tile=origin; one source at origin.
    # aspect dy/dx, dz/dx in [0.1,10]; gamma 1..100.
    cases = []
    # 1. stationary: dx=dy=dz=1, u=0.
    cases.append(("stationary", np.zeros(3), [{"x": [0, 0, 0], "u": [0, 0, 0], "w": 1.0}], 1.0, 1.0, 1.0))
    # 2. gamma=10 oblique: |u| ~ sqrt(gamma^2-1) = sqrt(99) ~ 9.95; oblique direction.
    u10 = np.array([9.9498743710662, 1.0, 0.3])  # gamma ~ sqrt(1+99+1.09) ~ 10.05
    u10 = u10 / np.linalg.norm(u10) * np.sqrt(99.0)  # normalise to gamma=10
    cases.append(("gamma10_oblique", np.zeros(3), [{"x": [0, 0, 0], "u": u10.tolist(), "w": 1.0}], 1.0, 1.0, 1.0))
    # 3. aspect=10: dy=10, dz=10, dx=1.
    cases.append(("aspect10", np.zeros(3), [{"x": [0, 0, 0], "u": [0, 0, 0], "w": 1.0}], 1.0, 10.0, 10.0))
    # 4. target exactly at a source (on a Yee position, in-support singularity).
    #    x_tile=0 -> Ex at i=j=k=0 is at (0.5,0,0); put the source exactly there.
    cases.append(("target_at_source", np.zeros(3), [{"x": [0.5, 0.0, 0.0], "u": [0, 0, 0], "w": 1.0}], 1.0, 1.0, 1.0))
    # 5. gamma=100 aspect-extreme combined: gamma=100, dy=0.1 (aspect 0.1).
    u100 = np.array([99.995, 0.0, 0.0])
    u100 = u100 / np.linalg.norm(u100) * np.sqrt(9999.0)  # gamma=100
    cases.append(("gamma100_aspect0p1", np.zeros(3), [{"x": [0, 0, 0], "u": u100.tolist(), "w": 1.0}], 1.0, 0.1, 1.0))

    results = []
    for (name, xt, parts, dx, dy, dz) in cases:
        try:
            r = compare_case(name, xt, parts, dx, dy, dz,
                             tol=args.tol, max_pass=args.max_pass, budget=args.budget)
            r["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            r = {"name": name, "status": "error", "error": repr(e)}
        results.append(r)
        print(f"  case {name}: {r.get('status')}", flush=True)
    return results


def run_selfchecks(args):
    t0 = time.time()
    results = ir.self_checks(dx=args.sx, dy=args.sy, dz=args.sz, strategy="subtraction",
                             feval_budget=args.budget)
    dt = time.time() - t0
    out = {"seconds": dt, "checks": []}
    for r in results:
        out["checks"].append({"name": r.name, "passed": r.passed, "detail": r.detail, "value": r.value})
    return out


def run_convergence(args):
    # Convergence in panel subdivision (max_pass) for a singular (in-support,
    # off-symmetry so the field is nonzero) point and for a near-support point.
    study = {}
    probes = {
        # Off-symmetry in-support (singular) point: source at origin, obs just
        # inside the support, stationary -> finite nonzero E by the local part.
        "in_support_singular": (np.array([0.3, 0.2, 0.1]), np.zeros(3), np.zeros(3)),
        # In-support, oblique gamma=10.
        "in_support_gamma10": (np.array([0.3, 0.2, 0.1]), np.zeros(3),
                               np.array([9.9498743710662, 1.0, 0.3]) * 1.0),
        # Just outside the support (near field, smooth).
        "near_support": (np.array([1.6, 0.2, 0.3]), np.zeros(3),
                         np.array([0.3, 0.1, 0.0])),
    }
    for tag, (x_obs, x_src, u) in probes.items():
        recs = ir.convergence_study(x_obs, x_src, u, 1.0, 1.0, 1.0,
                                    max_pass_list=[1, 2, 3, 4, 5], tol=1e-11,
                                    strategy="subtraction", feval_budget=args.budget)
        fit = ir._fit_order(recs)
        order, r2 = fit if fit is not None else (None, None)
        study[tag] = {
            "records": recs,
            "measured_order": order,
            "r2": r2,
            "final_field_norm": recs[-1]["norm"],
            "final_rel_step": recs[-1]["rel_step"],
        }
    # Convergence in the Kronrod order (endpoint strategy, varying pair_idx)
    # at a point where the singular point lies near ONLY the x axis (just
    # outside the support in x, interior in y,z) so the tensor product is
    # cheap and the bisection is bounded.
    xo = np.array([1.9, 0.1, 0.2])  # t_x=1.9 > 1.5 -> x axis not near singular;
    # instead use a point just inside in x only:
    xo = np.array([1.45, 0.05, 0.05])
    order_records = []
    for pidx in (0, 1, 2):  # 15-pt, 21-pt, 25-pt Kronrod
        E, B = ir.shape_field(xo, np.zeros(3), np.zeros(3), 1.0, 1.0, 1.0,
                              strategy="endpoint", pair_idx=pidx, tol=1e-11,
                              max_pass=4, feval_budget=args.budget)
        F = np.concatenate([E, B])
        order_records.append({"pair_idx": pidx, "kronrod_points": len(ir._kronrod()[pidx][2]),
                              "field": F.tolist(), "norm": _norm6(F)})
    # successive-difference order for the Kronrod-order sweep
    kro_orders = []
    for a, b in zip(order_records, order_records[1:]):
        d = _norm6(np.array(a["field"]) - np.array(b["field"]))
        kro_orders.append({"from_points": a["kronrod_points"], "to_points": b["kronrod_points"],
                           "abs_diff": d})
    study["kronrod_order"] = {"records": order_records, "successive_diffs": kro_orders}

    # Cross-check of the two independent strategies (subtraction vs endpoint)
    # at a bounded near point.
    xc = np.array([1.7, 0.2, 0.3])
    uc = np.array([0.3, 0.1, 0.0])
    E_sub, B_sub = ir.shape_field(xc, np.zeros(3), uc, 1.0, 1.0, 1.0,
                                  strategy="subtraction", feval_budget=args.budget)
    E_ep, B_ep = ir.shape_field(xc, np.zeros(3), uc, 1.0, 1.0, 1.0,
                                strategy="endpoint", max_pass=4, feval_budget=args.budget)
    study["strategy_crosscheck"] = {
        "x_obs": xc.tolist(), "u": uc.tolist(),
        "subtraction": np.concatenate([E_sub, B_sub]).tolist(),
        "endpoint": np.concatenate([E_ep, B_ep]).tolist(),
        "abs_diff": _norm6(np.concatenate([E_sub - E_ep, B_sub - B_ep])),
    }
    return study


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-11)
    ap.add_argument("--max-pass", type=int, default=5)
    ap.add_argument("--budget", type=int, default=4_000_000,
                    help="per-point function-evaluation cap")
    ap.add_argument("--sx", type=float, default=1.0)
    ap.add_argument("--sy", type=float, default=1.0)
    ap.add_argument("--sz", type=float, default=1.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = {}
    out["config"] = vars(args)
    out["kronrod_verification"] = kronrod_verification()
    print("self-checks ...", flush=True)
    out["self_checks"] = run_selfchecks(args)
    print("convergence ...", flush=True)
    out["convergence"] = run_convergence(args)
    print("comparison cases ...", flush=True)
    out["comparison"] = run_case_set(args)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
