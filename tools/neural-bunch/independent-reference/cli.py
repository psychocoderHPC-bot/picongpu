#!/usr/bin/env python3
"""Write an independent-reference field dump in the project's tile layout.

The output ``.npz`` contains

* ``reference``: shape ``(n_cases, 1536)``, canonical
  ``c + 6*(i + 8*(j + 8*k))`` serialization;
* ``s0``: shape ``(n_cases, 2)`` with the frozen per-case scale
  ``sqrt(256)*max_point||G||`` for ``G in {E, cB}``;
* ``case_ids``: the case names.

The layout and normalization match ``tools/neural-bunch/model-format.md`` and
the metric in ``tools/neural-bunch/eval/field_gate.py``:
``E_tilde = (q/e) dx**2 E_n``, ``B_tilde = (q/e) dx**2 B_n``.

Usage::

    python cli.py --output reference.npz [--order 8 --refine 0 --refine-near 2]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import indfield  # noqa: E402
from cases import default_cases, particles_of  # noqa: E402


def _load_field_gate():
    candidates = [
        HERE.parent / "eval" / "field_gate.py",
        Path("/tmp/alpaka-orchestration/pr24-20261102/t10-farfield/field_gate.py"),
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("field_gate", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit("field_gate.py not found (pass it via FIELD_GATE or place it at tools/neural-bunch/eval/field_gate.py)")


def _s0(tile_flat):
    gate = _load_field_gate()
    # Reuse the frozen definition, never a local reimplementation.
    case = gate.canonical_cases(tile_flat.reshape(1, -1), "reference")[0]
    s0 = gate.reference_tile_s0(case)
    return [s0["E"], s0["cB"]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--refine", type=int, default=0)
    parser.add_argument("--refine-near", type=int, default=2)
    args = parser.parse_args(argv)

    cases = default_cases()
    tiles = []
    s0 = []
    for case in cases:
        dx, dy, dz = case["cells"]
        tile = indfield.tile_field(
            case["x_tile"], particles_of(case), dx, dy, dz,
            order=args.order, refine=args.refine, refine_near=args.refine_near,
        )
        flat = indfield.flatten_tile(tile)
        tiles.append(flat)
        s0.append(_s0(flat))
        print(f"{case['name']}: max|field| = {np.max(np.abs(flat)):.6e}", flush=True)

    out = Path(args.output)
    np.savez(
        out,
        reference=np.asarray(tiles),
        s0=np.asarray(s0),
        case_ids=np.asarray([c["name"] for c in cases]),
    )
    meta = {
        "order": args.order,
        "refine": args.refine,
        "refine_near": args.refine_near,
        "cases": cases,
        "layout": "reference[n, 1536] with index c + 6*(i + 8*(j + 8*k))",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out} ({len(tiles)} cases)")


if __name__ == "__main__":
    main()
