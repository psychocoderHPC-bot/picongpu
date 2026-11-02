#!/usr/bin/env python3
"""Compare this independent reference with the in-repo FP64 reference.

Candidate = :mod:`indfield` (knot split + singularity-cancelling power
transform + tensor Gauss-Legendre).  Reference = ``convkernel.tile_field`` from
``tools/neural-bunch/reference``.  The comparison metric is imported from the
frozen ``field_gate.py``; it is never reimplemented here.

The in-repo default tile path (``adaptive=True``) re-grades every near point and
takes many minutes per tile, so the *full-tile* comparison uses its fixed-rule
mode (``adaptive=False, order=12``).  Point-level checks use its default
adaptive path.  Both modes are the same in-repo reference code.

Run::

    python compare_repo.py --output-dir OUT [--quick]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_REF = HERE.parent / "reference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_REF))

import indfield  # noqa: E402
from cases import default_cases, particles_of  # noqa: E402
import convkernel as ck  # noqa: E402


def _load_gate():
    candidates = [
        HERE.parent / "eval" / "field_gate.py",
        Path("/tmp/alpaka-orchestration/pr24-20261102/t10-farfield/field_gate.py"),
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("field_gate", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("field_gate.py not found")


def _max_point_rel(candidate, reference):
    """Pointwise relative L2 for reporting only (not the frozen metric)."""
    d = candidate - reference
    scale = np.maximum(np.abs(reference), 1e-30)
    return float(np.max(np.abs(d) / (1.0 + np.max(np.abs(reference)))))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    gate = _load_gate()
    cases = default_cases()
    cand = []
    ref = []
    point_checks = []

    order = 8
    refine_near = 1 if args.quick else 2
    repo_order = 8 if args.quick else 12

    for case in cases:
        dx, dy, dz = case["cells"]
        name = case["name"]
        t = time.time()
        tile = indfield.tile_field(case["x_tile"], particles_of(case), dx, dy, dz,
                                   order=order, refine=0, refine_near=refine_near)
        t_ind = time.time() - t
        t = time.time()
        rtile = ck.tile_field(case["x_tile"], particles_of(case), dx, dy, dz,
                              adaptive=False, order=repo_order)
        t_repo = time.time() - t
        cflat = indfield.flatten_tile(tile)
        rflat = indfield.flatten_tile(rtile)
        cand.append(cflat)
        ref.append(rflat)
        point_checks.append({
            "case": name,
            "max_abs_diff": float(np.max(np.abs(tile - rtile))),
            "max_rel_scaled": _max_point_rel(cflat, rflat),
            "ind_seconds": round(t_ind, 1),
            "repo_seconds": round(t_repo, 1),
        })
        print(f"{name}: max|ind-repo|={np.max(np.abs(tile - rtile)):.3e} "
              f"ind {t_ind:.1f}s repo {t_repo:.1f}s", flush=True)

    np.savez(outdir / "independent.npz", reference=np.asarray(cand),
             case_ids=np.asarray([c["name"] for c in cases]))
    np.savez(outdir / "repo_reference.npz", reference=np.asarray(ref),
             case_ids=np.asarray([c["name"] for c in cases]))
    config = {
        "gate": 1e-3,
        "s0": {"mode": "reference_tile"},
        "case_ids": [c["name"] for c in cases],
    }
    (outdir / "gate.json").write_text(json.dumps(config, indent=2) + "\n")
    report = gate.evaluate(np.asarray(cand), np.asarray(ref), config,
                           {"reference": np.asarray(ref)})
    report["case_details"] = point_checks
    report["note"] = (
        "candidate=indfield, reference=convkernel.tile_field(adaptive=False,"
        f" order={repo_order}); near-partition refine={refine_near}"
    )
    (outdir / "field_gate_report.json").write_text(json.dumps(report, indent=2) + "\n")

    for entry, detail in zip(report["cases"], point_checks):
        e = entry["E"]["relative_l2"]
        cb = entry["cB"]["relative_l2"]
        print(f"GATE {entry['case']}: E={e} cB={cb} "
              f"maxabs={detail['max_abs_diff']:.3e}", flush=True)
    print(f"all_pass={report['summary']['all_pass']} "
          f"max_relative_l2={report['summary']['max_relative_l2']}")
    print(f"report: {outdir/'field_gate_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
