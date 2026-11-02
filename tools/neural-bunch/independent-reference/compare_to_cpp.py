#!/usr/bin/env python3
"""Compare my independent reference tiles against the repo C++ reference tiles
using the frozen field_gate S0-normalised per-component relative L2 metric.

Reads the C++ tiles (``<case>.tile.txt``, 1536 doubles) and computes my tiles
with the vectorised subtraction path, then evaluates the metric with
``field_gate`` (S0 from the C++ reference tile, the "reference").
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO_ROOT = os.environ["NB_REPO_ROOT"]
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "neural-bunch", "reference"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools", "neural-bunch", "eval"))

import independent_reference as ir  # noqa: E402
import field_gate as fg  # noqa: E402


def load_cpp_tile(path):
    with open(path) as fh:
        vals = [float(line) for line in fh if line.strip()]
    return np.array(vals, dtype=np.float64)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp-dir", default=os.path.join(HERE, "cpp_tiles"))
    ap.add_argument("--max-pass", type=int, default=4)
    ap.add_argument("--budget", type=int, default=2_000_000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # The 5 cases, mirroring the C++ driver exactly.
    def normu(u, u_norm):
        u = np.asarray(u, dtype=float)
        n = float(np.linalg.norm(u))
        return u / n * u_norm if n > 0 else u

    u10 = normu([9.9498743710662, 1.0, 0.3], np.sqrt(100.0 - 1.0))
    u100 = normu([99.995, 0.0, 0.0], np.sqrt(10000.0 - 1.0))
    cases = [
        ("stationary", np.zeros(3), [{"x": [0, 0, 0], "u": [0, 0, 0], "w": 1.0}], 1.0, 1.0, 1.0),
        ("gamma10_oblique", np.zeros(3), [{"x": [0, 0, 0], "u": u10.tolist(), "w": 1.0}], 1.0, 1.0, 1.0),
        ("aspect10", np.zeros(3), [{"x": [0, 0, 0], "u": [0, 0, 0], "w": 1.0}], 1.0, 10.0, 10.0),
        ("target_at_source", np.zeros(3), [{"x": [0.5, 0.0, 0.0], "u": [0, 0, 0], "w": 1.0}], 1.0, 1.0, 1.0),
        ("gamma100_aspect0p1", np.zeros(3), [{"x": [0, 0, 0], "u": u100.tolist(), "w": 1.0}], 1.0, 0.1, 1.0),
    ]

    results = []
    for name, xt, parts, dx, dy, dz in cases:
        cpp_path = os.path.join(args.cpp_dir, f"{name}.tile.txt")
        cpp_flat = load_cpp_tile(cpp_path)
        t0 = time.time()
        my_tile = ir.tile_field_vec(xt, parts, dx, dy, dz, strategy="subtraction",
                                    box_depth=min(2 + args.max_pass, 5))
        my_flat = ir.flatten_tile(my_tile).astype(np.float64)
        t_my = time.time() - t0
        config = {"gate": 1e-3, "s0": {"mode": "reference_tile"}}
        report = fg.evaluate(my_flat, cpp_flat, config, {"reference": cpp_flat})
        entry = {"name": name, "my_seconds": t_my}
        for grp in ("E", "cB"):
            g = report["cases"][0][grp]
            entry[grp] = {
                "reference_norm": g["reference_norm"],
                "error_norm": g["error_norm"],
                "s0": g["s0"],
                "denominator": g["denominator"],
                "denominator_source": g["denominator_source"],
                "relative_l2": g["relative_l2"],
                "pass": g["pass"],
                "components": g["components"],
            }
        results.append(entry)
        print(f"case {name}: E_rel2={entry['E']['relative_l2']:.3e} "
              f"cB_rel2={entry['cB']['relative_l2']:.3e} pass={entry['E']['pass'] and entry['cB']['pass']} "
              f"(my {t_my:.1f}s)", flush=True)

    with open(args.output, "w") as fh:
        json.dump({"cases": results}, fh, indent=2)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
