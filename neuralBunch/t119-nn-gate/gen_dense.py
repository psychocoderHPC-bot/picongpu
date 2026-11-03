"""T119 dense-row generator: many targets per physical configuration.

Selects a stratified subset of the existing pilot configurations (same keys,
same split assignment) and draws `rows` targets inside each configuration's
distance shell.  This is a data-SHAPE change: same config space, many more
rows per config.

Output: one npz per split under --out, plus meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.expanduser("~/pr24/t103"))
from nearfield_judge import dataset as ds  # noqa: E402


def pick(cfg_by_split, n_per_split, seed):
    """Stratify each split over gamma, pick evenly spaced configs per gamma."""
    rng = np.random.default_rng(seed)
    out = {}
    for split, groups in cfg_by_split.items():
        wanted = n_per_split[split]
        by_g = {}
        for g in groups:
            by_g.setdefault(g["gamma"], []).append(g)
        gammas = sorted(by_g)
        # allocate as evenly as possible over gammas
        base, rem = divmod(wanted, len(gammas))
        chosen = []
        for i, g in enumerate(gammas):
            k = base + (1 if i < rem else 0)
            gs = sorted(by_g[g], key=lambda x: x["key"])
            # evenly spaced, deterministic
            if k >= len(gs):
                idx = list(range(len(gs)))
            else:
                idx = np.linspace(0, len(gs) - 1, k).round().astype(int).tolist()
            chosen.extend(gs[j] for j in idx)
        out[split] = chosen
    return out


def gen_split(groups, rows, seed, refine, jobs, chunk):
    samples = []
    for g in groups:
        for row in range(rows):
            samples.append(ds._sample_row(g, seed, row, g["split"]))
    samples.sort(key=lambda s: (s["config_key"], s["row"]))
    chunks = [(samples[i:i + chunk], refine) for i in range(0, len(samples), chunk)]
    res = []
    t0 = time.time()
    with get_context("fork").Pool(processes=jobs) as pool:
        for i, r in enumerate(pool.imap(ds._worker, chunks)):
            res.append(r)
            if (i + 1) % max(1, len(chunks) // 10) == 0:
                print(f"  {i+1}/{len(chunks)} chunks {time.time()-t0:.0f}s", flush=True)
    cols = {k: [] for k in ("dE", "dcB", "norm", "Econv", "cBconv", "Epoint", "cBpoint")}
    for r in res:
        for k in cols:
            cols[k].append(r[k])
    arr = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    out = {
        "R_cells": np.asarray([s["R_cells"] for s in samples]),
        "u": np.asarray([s["u"] for s in samples]),
        "cells": np.asarray([s["cells"] for s in samples]),
        "gamma": np.asarray([s["gamma"] for s in samples]),
        "aspect_y": np.asarray([s["aspect_y"] for s in samples]),
        "aspect_z": np.asarray([s["aspect_z"] for s in samples]),
        "m0": np.asarray([s["m0"] for s in samples]),
        "m2": np.asarray([s["m2"] for s in samples]),
        "m4": np.asarray([s["m4"] for s in samples]),
        "r_cells": np.asarray([s["r_cells"] for s in samples]),
        "config_key": np.asarray([s["config_key"] for s in samples]),
        "split": np.asarray([s["split"] for s in samples]),
        "row": np.asarray([s["row"] for s in samples], dtype=np.int64),
        "delta_E_hat": arr["dE"],
        "delta_cB_hat": arr["dcB"],
        "norm": arr["norm"],
        "delta_E": arr["dE"] / arr["norm"][:, None],
        "delta_cB": arr["dcB"] / arr["norm"][:, None],
        "E_conv": arr["Econv"],
        "cB_conv": arr["cBconv"],
        "E_point": arr["Epoint"],
        "cB_point": arr["cBpoint"],
    }
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=150)
    ap.add_argument("--train-configs", type=int, default=210)
    ap.add_argument("--val-configs", type=int, default=45)
    ap.add_argument("--test-configs", type=int, default=45)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--jobs", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--refine-train", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--refine-test", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    allg = ds.build_groups(5400, a.seed, 2)
    by_split = {}
    for g in allg:
        by_split.setdefault(g["split"], []).append(g)
    picks = pick(by_split, {"train": a.train_configs, "val": a.val_configs,
                            "test": a.test_configs}, a.seed)
    meta = {"rows": a.rows, "seed": a.seed, "counts": {},
            "refine": {"train": a.refine_train, "val": a.refine_train,
                       "test": a.refine_test},
            "configs": {k: [g["key"] for g in v] for k, v in picks.items()},
            "gammas": {k: sorted({g["gamma"] for g in v}) for k, v in picks.items()}}
    for split in ("train", "val", "test"):
        fpath = os.path.join(a.out, f"{split}.npz")
        if a.skip_existing and os.path.exists(fpath):
            print(f"[{split}] exists, skip", flush=True)
            rr = dict(np.load(fpath, allow_pickle=True))
            meta["counts"][split] = {"configs": len(picks[split]), "rows": int(len(rr["gamma"])),
                                     "seconds": 0.0, "per_config": a.rows,
                                     "refine": a.refine_test if split == "test" else a.refine_train}
            continue
        groups = picks[split]
        ref = a.refine_test if split == "test" else a.refine_train
        print(f"[{split}] {len(groups)} configs x {a.rows} rows, refine {ref}", flush=True)
        out, sec = gen_split(groups, a.rows, a.seed, ref, a.jobs, a.chunk)
        np.savez_compressed(os.path.join(a.out, f"{split}.npz"), **out)
        meta["counts"][split] = {"configs": len(groups), "rows": int(len(out["gamma"])),
                                 "seconds": sec, "per_config": a.rows, "refine": ref}
        print(f"[{split}] wrote {len(out['gamma'])} rows in {sec:.0f}s", flush=True)
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    print("DONE", json.dumps({k: v["rows"] for k, v in meta["counts"].items()}), flush=True)


if __name__ == "__main__":
    main()
