"""``python -m neuralbunch {reference,train,validate,export,inspect,parity}``."""

from __future__ import annotations

import argparse
import json
import sys


def cmd_reference(args):
    from .reference_data import build_dataset

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    out = {}
    for sp in splits:
        print(f"[reference] {sp}", flush=True)
        m = build_dataset(
            args.out,
            sp,
            seed=args.seed + {"train": 0, "val": 1000, "test": 2000}[sp],
            n_configs=args.n_configs,
            samples_per_config=args.samples_per_config,
            include_bunches=args.include_bunches,
            bunch_kinds=args.bunch_kinds,
            n_particles=args.n_particles,
            bunch_gammas=args.bunch_gammas,
            shard_size=args.shard_size,
            reference_path=args.reference,
            quadrature=(None if not args.ref_fixed else
                        {"order": args.ref_order, "tol": args.ref_tol,
                         "max_level": args.ref_max_level}),
            aspect_range=(args.aspect_min, args.aspect_max),
        )
        out[sp] = {"shards": len(m["shards"]), "samples": len(m["descriptors"]),
                   "reference": m["reference"]["kind"]}
    print(json.dumps(out, indent=2))
    return 0


def cmd_train(args):
    from .train import TrainConfig, run_training

    cfg = TrainConfig(
        hidden=[int(x) for x in args.hidden.split(",") if x.strip()],
        latent=args.latent,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_tiles=args.batch_tiles,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        seed=args.seed,
        force_weight=args.force_weight,
        trials=args.trials,
        time_budget_s=args.time_budget_s,
        a_y_anchors=args._anchors,
        a_z_anchors=args._anchors,
        feature_gate_init=args.feature_gate,
        lr_schedule=args.lr_schedule,
    )
    eval_splits = tuple(x.strip() for x in args.eval_splits.split(",") if x.strip())
    model, best_cfg, report = run_training(
        args.data, args.out, cfg, resume=args.resume, device=args.device, splits=eval_splits
    )
    if args.export:
        from .export import write_artifact

        meta = write_artifact(model, args.export)
        print(f"exported {args.export}.json/.bin sha256={meta['artifact']['sha256']}")
    if args.validation_report:
        from .train import validate_artifact_on_data

        artifact = args.export or str(__import__("pathlib").Path(args.out) / "best.pt")
        if artifact.endswith(".pt"):
            from .train import export_from_checkpoint

            meta = export_from_checkpoint(artifact, args.out + "/exported")
            artifact = args.out + "/exported"
        validate_artifact_on_data(artifact, args.data, out_report=args.validation_report)
    print(json.dumps({"best_trial": report["best_trial"], "best_seed": report["best_seed"]}, indent=2))
    return 0


def cmd_validate(args):
    from .train import validate_artifact_on_data

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    report = validate_artifact_on_data(
        args.artifact, args.data, splits=splits, out_report=args.out, device=args.device
    )
    print(json.dumps({k: v.get("overall_rel_rms") for k, v in report["splits"].items()}, indent=2))
    return 0


def cmd_export(args):
    from .export import load_artifact
    from .train import export_from_checkpoint

    meta = export_from_checkpoint(args.checkpoint, args.out)
    print(f"wrote {args.out}.json and {args.out}.bin")
    print(json.dumps({k: meta[k] for k in ("format", "version", "code_sha")}, indent=2))
    if args.inspect:
        from .inspect import inspect_artifact, print_report

        print_report(inspect_artifact(args.out))
    return 0


def cmd_inspect(args):
    from .inspect import inspect_artifact, print_report
    from .export import ArtifactError

    rc = 0
    for a in args.artifacts:
        try:
            report = inspect_artifact(a)
            print(f"== {a}")
            print_report(report)
        except (ArtifactError, FileNotFoundError) as exc:
            print(f"== {a}\nartifact: INVALID: {exc}")
            rc = 1
    return rc


def cmd_parity(args):
    from .parity import compare_artifact

    res = compare_artifact(args.artifact, args.particles, args.trials)
    print(json.dumps(res, indent=2))
    return 0 if res["passed"] else 1


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m neuralbunch")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("reference", help="generate FP64 reference target shards")
    r.add_argument("--out", required=True)
    r.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--n-configs", type=int, default=None, dest="n_configs")
    r.add_argument("--samples-per-config", type=int, default=1)
    r.add_argument("--include-bunches", action="store_true")
    r.add_argument("--bunch-kinds", nargs="*", default=None)
    r.add_argument("--n-particles", type=int, default=64)
    r.add_argument("--bunch-gammas", nargs="*", type=float, default=[2.0, 10.0, 100.0])
    r.add_argument("--shard-size", type=int, default=8)
    r.add_argument("--reference", default=None)
    r.add_argument("--ref-order", type=int, default=5)
    r.add_argument("--ref-tol", type=float, default=1e-8)
    r.add_argument("--ref-max-level", type=int, default=5)
    r.add_argument("--ref-fixed", action="store_true",
                   help="use one fixed quadrature (--ref-order/--ref-tol/--ref-max-level); default is the per-sample policy")
    r.add_argument("--aspect-min", type=float, default=0.1)
    r.add_argument("--aspect-max", type=float, default=10.0)
    r.set_defaults(func=cmd_reference)

    t = sub.add_parser("train", help="train with checkpoints and reports")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--hidden", default="256,256")
    t.add_argument("--latent", type=int, default=128)
    t.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--batch-tiles", type=int, default=8)
    t.add_argument("--max-steps", type=int, default=2000)
    t.add_argument("--eval-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--force-weight", type=float, default=0.0)
    t.add_argument("--trials", type=int, default=1)
    t.add_argument("--time-budget-s", type=float, default=600.0)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--device", default=None)
    t.add_argument("--eval-splits", default="train,val,test",
                   help="splits evaluated during/after training; use train,val to keep test untouched")
    t.add_argument("--decoder-anchors", type=int, default=5,
                   help="number of log-spaced aspect anchors per axis on [0.1,10]")
    t.add_argument("--feature-gate", default="ones", choices=["ones", "lowfreq"],
                   help="initial learned gate on fourier_v1 features; lowfreq starts with octaves >=2 disabled")
    t.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"],
                   help="constant lr or 2k warmup + cosine decay to 2%% of peak")
    t.add_argument("--export", default=None, help="also export the final model to this prefix")
    t.add_argument("--validation-report", default=None)
    t.set_defaults(func=cmd_train)

    v = sub.add_parser("validate", help="evaluate an artifact against generated splits")
    v.add_argument("--artifact", required=True)
    v.add_argument("--data", required=True)
    v.add_argument("--splits", default="train,val,test")
    v.add_argument("--out", default=None)
    v.add_argument("--device", default=None)
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("export", help="export a checkpoint to MODEL-FORMAT v1")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--inspect", action="store_true")
    e.set_defaults(func=cmd_export)

    i = sub.add_parser("inspect", help="validate one or more artifacts")
    i.add_argument("artifacts", nargs="+")
    i.set_defaults(func=cmd_inspect)

    p = sub.add_parser("parity", help="torch vs numpy parity of an artifact")
    p.add_argument("--artifact", required=True)
    p.add_argument("--particles", type=int, default=257)
    p.add_argument("--trials", type=int, default=3)
    p.set_defaults(func=cmd_parity)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if getattr(args, "decoder_anchors", None):
        import numpy as _np

        n = int(args.decoder_anchors)
        if n < 2:
            ap.error("--decoder-anchors must be >= 2")
        args._anchors = [float(x) for x in _np.logspace(-1.0, 1.0, n)]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
