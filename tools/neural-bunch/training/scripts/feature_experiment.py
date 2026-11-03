#!/usr/bin/env python3
"""Diagnostic: is the frozen fourier_v1 feature map the accuracy limiter?

Trains the same additive architecture on alternative *smooth* feature maps.
These models are diagnostics only; MODEL-FORMAT v1 fixes fourier_v1, so they
are never exported as deliverable artifacts.

Modes:
  fourier_v1 : control, the frozen 69-dim map
  lowfreq    : octaves -3..1 only (39 dims); removes the high-frequency aliasing
  smooth     : bounded radial features + raw velocity/aspect (10 dims)

Example:
  python scripts/feature_experiment.py --data /path/data --mode smooth \
      --out /path/exp_smooth --steps 20000 --device cuda:1
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import neuralbunch.model as M  # noqa: E402
from neuralbunch.features import fourier_v1  # noqa: E402
from neuralbunch.reference_data import load_split  # noqa: E402


def base_arrays(data):
    si = data.sample_index
    pos = data.pos
    u = data.u
    r = (pos - data.origin[si]) / data.dx[si][:, None]
    return r, u, data.a_y[si], data.a_z[si]


def make_features(data, mode):
    r, u, ay, az = base_arrays(data)
    if mode == "fourier_v1":
        return fourier_v1(r, u, ay, az), 69
    if mode == "lowfreq":
        f = fourier_v1(r, u, ay, az, octaves=(-3, 1))
        return f, f.shape[1]
    if mode == "smooth":
        gamma = np.sqrt(1.0 + np.sum(u * u, axis=1, keepdims=True))
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        rs = r / (1.0 + rn / 8.0)
        logr = np.log1p(rn)
        f = np.concatenate([u, gamma, ay[:, None], az[:, None], rs, logr], axis=1)
        return f.astype(np.float64), f.shape[1]
    raise ValueError(mode)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", default="smooth", choices=["fourier_v1", "lowfreq", "smooth"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--hidden", default="512,512")
    ap.add_argument("--latent", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args(argv)

    train = load_split(args.data, "train")
    val = load_split(args.data, "val")
    feats, dim = make_features(train, args.mode)
    train.feats = np.ascontiguousarray(feats)
    vfeats, _ = make_features(val, args.mode)
    val.feats = np.ascontiguousarray(vfeats)
    print("mode", args.mode, "feature_dim", dim, "train", train.n_tiles, "val", val.n_tiles, flush=True)

    M.feature_dim = lambda octaves=None: dim
    M.ModelConfig.validate = lambda self: self
    M._FEATURE_SCALE = np.ones(dim, dtype=np.float64)
    from neuralbunch.train import TrainConfig, train_one, evaluate

    cfg = TrainConfig(
        hidden=[int(x) for x in args.hidden.split(",")],
        latent=args.latent,
        lr=args.lr,
        weight_decay=1e-4,
        batch_tiles=args.batch,
        max_steps=args.steps,
        eval_every=args.eval_every,
        seed=0,
        force_weight=0.1,
        time_budget_s=1e9,
    )
    import torch

    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, history, best = train_one(cfg, train, val, dev, out_dir=out)
    ev = evaluate(model, val, cfg.model_config(), dev, cfg.force_weight)
    result = {
        "mode": args.mode,
        "feature_dim": dim,
        "config": vars(cfg),
        "val": {k: ev[k] for k in ("E", "cB", "force", "overall_rel_rms", "loss")},
        "per_case_meta": ev["per_case_meta"],
        "history": history,
    }
    (out / "feature_experiment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"mode": args.mode, "E": ev["E"], "cB": ev["cB"], "force": ev["force"], "overall": ev["overall_rel_rms"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
