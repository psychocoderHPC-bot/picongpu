#!/usr/bin/env python3
"""Per-distance / per-gamma breakdowns and coverage for a validation report."""
import argparse, json, math
from pathlib import Path
import numpy as np

BINS = [4, 8, 16, 32, 64, 128, 256, 513]


def stats(vals, absn=None):
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0}
    out = {
        "n": int(v.size),
        "mean": float(v.mean()),
        "p99": float(np.percentile(v, 99)),
        "max": float(v.max()),
    }
    if absn is not None:
        a = np.asarray(absn, dtype=bool)
        out["absolute_normalized_share"] = float(a.mean())
    return out


def group(meta, keyfn, label):
    groups = {}
    for m in meta:
        groups.setdefault(keyfn(m), []).append(m)
    out = {}
    for k in sorted(groups, key=str):
        rows = groups[k]
        out[str(k)] = {
            "E": stats([r["rel_E"] for r in rows],
                       [r["absolute_normalized_E"] for r in rows]),
            "cB": stats([r["rel_B"] for r in rows],
                        [r["absolute_normalized_B"] for r in rows]),
            "force": stats([r.get("rel_force", float("nan")) for r in rows],
                           [r.get("absolute_normalized_force", False) for r in rows]),
        }
    return out


def dist_bin(d):
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        if lo <= d < hi:
            return f"[{lo},{hi})"
    return "outside"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    rep = json.loads(Path(args.report).read_text())
    block = rep["splits"][args.split]
    meta = block["per_case_meta"]
    d = [float(m["d_min_cells"]) for m in meta]
    coverage = {
        "n_cases": len(meta),
        "n_within_domain_4_512": int(sum(4.0 <= x <= 512.0 for x in d)),
        "neural_fraction": float(np.mean([4.0 <= x <= 512.0 for x in d])),
        "d_min_min": float(min(d)), "d_min_median": float(np.median(d)),
        "d_min_max": float(max(d)),
        "distance_histogram": {},
        "gamma_values": sorted({m.get("gamma_name") for m in meta}),
        "bunch_cases": int(sum(m.get("kind") == "bunch" for m in meta)),
        "scenario_cases": int(sum(m.get("kind") != "bunch" for m in meta)),
    }
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        label = f"[{lo},{hi})"
        rows = [m for m in meta if lo <= float(m["d_min_cells"]) < hi]
        coverage["distance_histogram"][label] = len(rows)
    result = {
        "split": args.split,
        "overall": {
            "E": block["E"], "cB": block["cB"], "force": block["force"],
            "overall_rel_rms": block["overall_rel_rms"],
            "n_cases": block["n_cases"],
        },
        "coverage": coverage,
        "by_distance": group(meta, lambda m: dist_bin(float(m["d_min_cells"])), "distance"),
        "by_gamma": group(meta, lambda m: m.get("gamma_name") or "unknown", "gamma"),
        "by_position_mode": group(meta, lambda m: m.get("position_mode") or "unknown", "mode"),
        "by_gamma_distance": group(
            meta,
            lambda m: f"{m.get('gamma_name') or '?'}|{dist_bin(float(m['d_min_cells']))}",
            "gamma_distance",
        ),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
