#!/usr/bin/env python3
"""Compare the float and double alpakaVendor FFT runs of the same mesh.

Both CSVs come from `meshconv --mode validate [--fft float]`: the deposit,
kernel build and readout are double in both, only the FFT precision differs.
Prints the peak-normalised roundoff |f-d|/(1+max) and the local-field fraction
|f-d|/|d| (only where |d| > 1e-8), worst over all rows and per distance band.
"""
import argparse
import csv
import math


def load(path):
    out = {}
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            key = (int(float(r["gamma"])), float(r["aspect"]), r["axis"],
                   int(float(r["d"])), r["comp"])
            out[key] = float(r["mesh"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("double_csv")
    ap.add_argument("float_csv")
    a = ap.parse_args()
    d = load(a.double_csv)
    f = load(a.float_csv)
    assert d.keys() == f.keys(), "row sets differ"

    bands = {"all": (2, 16), "d2-4": (2, 4), "d5-8": (5, 8),
             "d9-12": (9, 12), "d13-16": (13, 16)}
    peak = {b: 0.0 for b in bands}
    local = {b: 0.0 for b in bands}
    worst_peak = None
    worst_local = None
    for key, vd in d.items():
        vf = f[key]
        diff = abs(vf - vd)
        p = diff / (1.0 + max(abs(vf), abs(vd)))
        dist = key[3]
        for b, (lo, hi) in bands.items():
            if lo <= dist <= hi:
                if p > peak[b]:
                    peak[b] = p
                    if b == "all":
                        worst_peak = (key, vf, vd, p)
                if abs(vd) > 1e-8:
                    loc = diff / abs(vd)
                    if loc > local[b]:
                        local[b] = loc
                        if b == "all":
                            worst_local = (key, vf, vd, loc)
    print("float32 vs double alpakaVendor FFT, same double deposit/kernel")
    for b in bands:
        print(f"  band {b:<6s} peak-normalised max = {peak[b]:.3e}   "
              f"local-field max = {local[b]:.3e}")
    print(f"  worst peak row: {worst_peak}")
    print(f"  worst local row: {worst_local}")


if __name__ == "__main__":
    main()
