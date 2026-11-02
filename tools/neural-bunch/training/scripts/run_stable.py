#!/usr/bin/env python3
"""Run the stable training recipe, export, inspect, parity and final evaluation."""

from __future__ import annotations
import argparse
import json
import sys
import time
import hashlib
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))

import torch  # noqa: E402
from neuralbunch.stable_train import train_stable  # noqa: E402
from neuralbunch.train import TrainConfig, load_checkpoint, build_model_from_config, evaluate, export_from_checkpoint  # noqa: E402
from neuralbunch.export import load_artifact  # noqa: E402
from neuralbunch.parity import compare_artifact  # noqa: E402
from neuralbunch.reference_data import load_split  # noqa: E402
from neuralbunch.model import DEFAULT_AXES  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latent", type=int, default=512)
    ap.add_argument("--feature-kind", default="fourier_v1")
    ap.add_argument("--hidden", default="512,512")
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--budget-s", type=float, default=7500.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--batch-tiles", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--floor", type=float, default=5e-2)
    ap.add_argument("--w-pat", type=float, default=1.0)
    ap.add_argument("--w-log", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-only-best", action="store_true")
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args(argv)

    dev = torch.device(
        args.device
        or (
            "cuda:1"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        hidden=[int(x) for x in args.hidden.split(",") if x.strip()],
        latent=args.latent,
        optimizer="adamw",
        lr=args.lr,
        weight_decay=1e-5,
        batch_tiles=args.batch_tiles,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        seed=args.seed,
        force_weight=0.0,
        trials=1,
        time_budget_s=args.budget_s,
        a_y_anchors=list(DEFAULT_AXES),
        a_z_anchors=list(DEFAULT_AXES),
        feature_gate_init="ones",
        feature_kind=args.feature_kind,
        lr_schedule="cosine",
        warmup_steps=500,
        notes="stable recipe: frozen gate, calibrated bank, bounded+log+pattern loss",
    )
    if not args.eval_only_best:
        train_stable(
            cfg,
            args.data,
            out,
            dev,
            time_budget_s=args.budget_s,
            max_steps=args.max_steps,
            eval_every=args.eval_every,
            floor=args.floor,
            w_pat=args.w_pat,
            w_log=args.w_log,
            lr=args.lr,
            seed=args.seed,
            feature_kind=args.feature_kind,
        )

    ck = load_checkpoint(out / "best.pt")
    model = build_model_from_config(TrainConfig(**ck["config"]))
    model.load_state_dict(ck["model_state"])
    model = model.double()
    prefix = out / "artifact"
    meta = export_from_checkpoint(str(out / "best.pt"), str(prefix))
    meta2, tens = load_artifact(str(prefix))
    parity = compare_artifact(str(prefix), n_particles=257, n_trials=3)
    art = {
        "path": str(prefix),
        "json": str(prefix.with_suffix(".json")),
        "bin": str(prefix.with_suffix(".bin")),
        "sha256": sha256_file(prefix.with_suffix(".bin")),
        "bytes": prefix.with_suffix(".bin").stat().st_size,
        "code_sha": meta.get("code_sha"),
        "selected_step": int(ck["step"]),
        "best_val_loss": float(ck.get("best_val", float("nan"))),
        "config": ck["config"],
        "parity": parity,
        "inspect": {
            "format": meta2["format"],
            "version": meta2["version"],
            "latent": meta2["encoder"]["output_dim"],
            "layers": [(layer["in"], layer["out"]) for layer in meta2["encoder"]["layers"]],
            "bank_entries": len(meta2["decoder"]["bank"]),
        },
    }
    (out / "artifact_identity.json").write_text(json.dumps(art, indent=2) + "\n")
    print(json.dumps(art, indent=1), flush=True)

    rep = {"artifact": art, "splits": {}}
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        d = load_split(args.data, sp)
        t0 = time.time()
        rep["splits"][sp] = evaluate(model, d, model.cfg, dev)
        rep["splits"][sp]["eval_seconds"] = time.time() - t0
        print(
            json.dumps(
                {
                    "event": "evaluated",
                    "split": sp,
                    "n": d.n_tiles,
                    "E": rep["splits"][sp]["E"]["mean_rel_rms"],
                    "cB": rep["splits"][sp]["cB"]["mean_rel_rms"],
                    "force": rep["splits"][sp]["force"]["mean_rel_rms"],
                }
            ),
            flush=True,
        )
    (out / "final_eval.json").write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
