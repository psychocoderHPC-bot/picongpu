"""Compact tables from the T116 run JSONs (component-wise relative error)."""
import json
import os
import sys

R = os.path.expanduser("~/pr24/t116/runs")


def load(tag):
    with open(os.path.join(R, tag + ".json")) as fh:
        return json.load(fh)


def main():
    tags = sys.argv[1:]
    for tag in tags:
        r = load(tag)
        print("===", tag, "params", r["n_params"], "best_epoch", r["best_epoch"])
        for split in ("train", "val", "test"):
            m = r[split]
            print(f"  {split:5s} p99 {m['p99']:.4e} worst {m['worst']:.4e} "
                  f"pooledE {m['pooled_E']:.4e}")
        print("  held-out by gamma (best-val checkpoint, component-wise):")
        for k, v in r["test_strat"].items():
            if k.startswith("gamma") and isinstance(v, dict):
                print(f"    {k:10s} n={v['n']:4d} p99 {v['p99']:.4e} worst {v['worst']:.4e}")
        print("  held-out by azimuth:")
        for k, v in r["test_strat"].items():
            if k.startswith("az"):
                print(f"    {k:14s} n={v['n']:4d} p99 {v['p99']:.4e} worst {v['worst']:.4e}")
        print("  final-epoch test:", f"p99 {r['final']['test']['p99']:.4e} "
              f"worst {r['final']['test']['worst']:.4e}")


if __name__ == "__main__":
    main()
