#!/usr/bin/env python3
"""Evaluate an exported artifact on val/test with the preregistered metric."""

import argparse
import json
import sys
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))
from neuralbunch.train import validate_artifact_on_data  # noqa: E402
from neuralbunch.parity import compare_artifact  # noqa: E402
from neuralbunch.export import load_artifact  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    prefix = str(Path(a.run) / "artifact")
    meta, _ = load_artifact(prefix)
    par = compare_artifact(prefix, n_particles=257, n_trials=3)
    rep = validate_artifact_on_data(
        prefix,
        a.data,
        tuple(s.strip() for s in a.splits.split(",")),
        str(Path(a.run) / "final_eval.json"),
        device=a.device,
    )
    print(
        json.dumps(
            {
                "parity": par,
                "splits": {
                    k: {
                        "E": v["E"]["mean_rel_rms"],
                        "cB": v["cB"]["mean_rel_rms"],
                        "force": v["force"]["mean_rel_rms"],
                        "n": v["n_cases"],
                    }
                    for k, v in rep["splits"].items()
                },
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
