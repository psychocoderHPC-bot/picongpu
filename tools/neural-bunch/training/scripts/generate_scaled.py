#!/usr/bin/env python3
"""Generate the scaled real-reference dataset with the compiled C++ reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from neuralbunch import scaled  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-seed", type=int, default=20261101)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--shard-size", type=int, default=128)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--module-path", default=None)
    ap.add_argument("--hpp", default=str(TRAIN_DIR / "cpp" / "neural_reference.hpp"))
    ap.add_argument(
        "--python-reference",
        default=str(TRAIN_DIR.parent / "reference" / "convkernel.py"),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if args.dry_run:
        total = 0
        summary = {}
        for sp in splits:
            d = scaled.descriptors_for_split(sp)
            n_src = sum(1 for x in d if x["kind"] == "scenario") + sum(
                x.get("n_particles", 0) for x in d if x["kind"] == "bunch"
            )
            # chunks do not store n_particles before packing; approximate with 32
            summary[sp] = {
                "tiles": len(d),
                "scenario_tiles": sum(1 for x in d if x["kind"] == "scenario"),
                "bunch_chunks": sum(1 for x in d if x["kind"] == "bunch"),
                "estimated_MiB": len(d) * 1536 * 8 / 2**20,
            }
            total += len(d)
        print(json.dumps({"splits": summary, "total_tiles": total,
                          "estimated_total_MiB": total * 1536 * 8 / 2**20}, indent=2))
        return 0

    out = scaled.generate_all(
        args.out,
        base_seed=args.base_seed,
        jobs=args.jobs,
        shard_size=args.shard_size,
        splits=splits,
        module_path=args.module_path,
        hpp_path=args.hpp,
        python_reference_path=args.python_reference,
    )
    summary = {}
    total_bytes = 0
    total_tiles = 0
    for sp, m in out.items():
        b = sum(s["bytes"] for s in m["shards"])
        total_bytes += b
        total_tiles += m.get("n_tiles", 0)
        summary[sp] = {
            "tiles": m.get("n_tiles", 0),
            "shards": len(m["shards"]),
            "bytes": b,
            "descriptors": len(m["descriptors"]),
        }
    summary["_total"] = {
        "tiles": total_tiles,
        "bytes": total_bytes,
        "MiB": total_bytes / 2**20,
        "reference_sha256": out[splits[0]]["reference"]["sha256"],
    }
    print(json.dumps(summary, indent=2))
    (Path(args.out) / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
