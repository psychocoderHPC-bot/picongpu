"""Cross-check the training quadrature against a tight FP64 quadrature.

Run from the training directory:

    ./venv/bin/python scripts/reference_crosscheck.py --samples 12 \
        --out crosscheck.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuralbunch import sampling  # noqa: E402
from neuralbunch.reference_data import (_sample_quadrature, preregistered_scales)  # noqa: E402
from neuralbunch.reference_loader import (  # noqa: E402
    call_tile_field,
    load_module,
    sha256_file,
    DEFAULT_REFERENCE,
)

TIGHT = {"order": 8, "tol": 1e-12, "max_level": 7}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--aspect-range", default="0.5,2.0")
    args = ap.parse_args(argv)

    ar = tuple(float(x) for x in args.aspect_range.split(","))
    mod = load_module(args.reference)
    ref_path = Path(args.reference) if args.reference else DEFAULT_REFERENCE
    rng = np.random.default_rng(12345)
    rows = []
    for i in range(args.samples):
        gname = sampling.GAMMA_SPECS[i % len(sampling.GAMMA_SPECS)][0]
        mode = sampling.POSITION_MODES[i % len(sampling.POSITION_MODES)]
        direction = sampling.DIRECTIONS[(i * 3) % len(sampling.DIRECTIONS)]
        sample = sampling.make_sample("train", i, gname, direction, mode, i,
                                      aspect_range=ar)
        q = _sample_quadrature(sample, None)
        t0 = time.time()
        fast = call_tile_field(mod, sample, q)
        t_fast = time.time() - t0
        t0 = time.time()
        tight = call_tile_field(mod, sample, TIGHT)
        t_tight = time.time() - t0
        row = {
            "i": i,
            "gamma_name": gname,
            "position_mode": mode,
            "direction": direction,
            "d_min_cells": float(sample.meta["d_min_cells"]),
            "a_y": float(sample.a_y),
            "a_z": float(sample.a_z),
            "quadrature": q,
            "time_fast_s": t_fast,
            "time_tight_s": t_tight,
        }
        for name, sl in (("E", slice(0, 3)), ("cB", slice(3, 6))):
            a = fast[..., sl].reshape(-1)
            b = tight[..., sl].reshape(-1)
            err = np.linalg.norm(a - b)
            ref = np.linalg.norm(b)
            s0 = preregistered_scales(fast)[0 if name == "E" else 1]
            row[f"{name}_rel"] = float(err / max(ref, s0))
            row[f"{name}_ref_norm"] = float(ref)
        rows.append(row)
        print(json.dumps(row), flush=True)

    out = {
        "reference_file": str(ref_path),
        "reference_sha256": sha256_file(ref_path),
        "training_quadrature_policy": "order 6 for |u|>10, else order 4",
        "aspect_range": list(ar),
        "tight_quadrature": TIGHT,
        "samples": rows,
        "max_E_rel": max(r["E_rel"] for r in rows),
        "max_cB_rel": max(r["cB_rel"] for r in rows),
    }
    print(json.dumps({k: out[k] for k in ("max_E_rel", "max_cB_rel")}, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
