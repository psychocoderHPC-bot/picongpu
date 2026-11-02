#!/usr/bin/env python3
"""Post-process the stable-run final_eval.json into metric tables."""

import json
import sys
import numpy as np


def stats(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {}
    return dict(
        n=int(a.size),
        mean=float(a.mean()),
        p50=float(np.percentile(a, 50)),
        p90=float(np.percentile(a, 90)),
        p99=float(np.percentile(a, 99)),
        max=float(a.max()),
    )


def pass_share(a, gate):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.mean(a <= gate)) if a.size else float("nan")


def group_table(meta, key, gkey=None):
    gkey = gkey or ("rel_" + key)
    rel = np.array([m[gkey] for m in meta], float)
    ref = np.array([m["ref_norm_" + key] for m in meta], float)
    s0 = np.array([m["s0_" + key] for m in meta], float)
    den = np.maximum(ref, s0)
    den = np.where(den > 0, den, np.nan)
    zero = ref / den
    absn = ref < s0
    out = {
        "all": dict(stats(rel), zero=stats(zero), absnorm_share=float(absn.mean())),
        "pass_1e-3": pass_share(rel, 1e-3),
    }
    deg = (ref < 1e-6) | ~np.isfinite(den)
    out["nondegenerate"] = dict(
        stats(rel[~deg]), n_nondeg=int((~deg).sum()), zero_mean=float(np.nanmean(zero[~deg])) if (~deg).any() else None
    )
    return out


def breakdown(meta, key):
    rel = np.array([m["rel_" + key] for m in meta], float)
    d = np.array([m["d_min_cells"] for m in meta], float)
    g = np.array([m["gamma"] for m in meta], float)
    out = {}
    edges = [4, 8, 16, 32, 64, 128, 256, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        out["d_%g_%g" % (lo, hi)] = dict(stats(rel[m]), n_bin=int(m.sum()))
    gedges = [1, 3.1623, 10, 31.623, 100.1]
    for lo, hi in zip(gedges[:-1], gedges[1:]):
        m = (g >= lo) & (g < hi)
        out["g_%g_%g" % (lo, hi)] = dict(stats(rel[m]), n_bin=int(m.sum()))
    return out


def main(path):
    rep = json.load(open(path))
    for split, sl in rep["splits"].items():
        meta = sl["per_case_meta"]
        print("=== split", split, "n", len(meta))
        for key in ("E", "B"):
            t = group_table(meta, key)
            print(
                key,
                "all n=%d mean=%.3g p50=%.3g p90=%.3g p99=%.3g max=%.3g zero_mean=%.3g zero_p50=%.3g absnorm=%.3f pass1e-3=%.3f"
                % (
                    t["all"]["n"],
                    t["all"]["mean"],
                    t["all"]["p50"],
                    t["all"]["p90"],
                    t["all"]["p99"],
                    t["all"]["max"],
                    t["all"]["zero"]["mean"],
                    t["all"]["zero"]["p50"],
                    t["all"]["absnorm_share"],
                    t["pass_1e-3"],
                ),
            )
            nd = t["nondegenerate"]
            print(
                "   nondegenerate n=%d mean=%.3g p50=%.3g p90=%.3g p99=%.3g zero_mean=%.3g"
                % (nd["n_nondeg"], nd["mean"], nd["p50"], nd["p90"], nd["p99"], nd["zero_mean"])
            )
            print(
                "   by distance:",
                " ".join(
                    "%s:n%d mean%.2g p50%.2g" % (k, v["n_bin"], v["mean"], v["p50"])
                    for k, v in breakdown(meta, key).items()
                    if k.startswith("d_")
                    if "mean" in v
                ),
            )
            print(
                "   by gamma:   ",
                " ".join(
                    "%s:n%d mean%.2g p50%.2g" % (k, v["n_bin"], v["mean"], v["p50"])
                    for k, v in breakdown(meta, key).items()
                    if k.startswith("g_")
                    if "mean" in v
                ),
            )
        fr = np.array([m["rel_force"] for m in meta], float)
        print(
            "force mean=%.4g p50=%.4g p99=%.4g max=%.4g pass1e-2=%.3f zero? n=%d"
            % (
                np.nanmean(fr),
                np.nanpercentile(fr, 50),
                np.nanpercentile(fr, 99),
                np.nanmax(fr),
                pass_share(fr, 1e-2),
                len(fr),
            )
        )
        print("constraint", sl.get("constraint_diagnostics"))


if __name__ == "__main__":
    main(sys.argv[1])
