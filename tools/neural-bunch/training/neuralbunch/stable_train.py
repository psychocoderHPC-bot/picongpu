"""Stable training recipe for MODEL-FORMAT v1 (diagnosis-driven).

Fixes the collapse reported in the previous run:
* bounded per-case denominator ``max(||A||, S0, floor)`` in the *loss only*
  (evaluation always uses the preregistered -- no-floor -- metric);
* extra log-amplitude and unit-pattern terms so the 8-decade target amplitude
  range does not dominate the gradient;
* feature gate frozen at one (baked into the exported first layer exactly as
  before, so the artifact function is unchanged);
* decoder bank calibrated once so the initial output scale matches targets;
* AdamW (no decay on the bank), gradient clipping, warmup + cosine schedule;
* per-step monitoring of loss, z norm, bank/gate norm, output/target ratio.

No inference-time normalisation is introduced.  The artifact is still
``z = sum_i (q_i/e) h_theta(fourier_v1(...))``, ``F = D(a) z``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .reference_data import load_split
from .train import (
    Batcher,
    TrainConfig,
    _groups,
    build_model_from_config,
    save_checkpoint,
)

EPS = 1e-12


def make_model(cfg: TrainConfig, device):
    model = build_model_from_config(cfg).to(device)
    # freeze the multiplicative gate: it was the collapse handle last time
    model.encoder.feature_gate.requires_grad_(False)
    model.encoder.feature_gate.data.fill_(1.0)
    return model


@torch.no_grad()
def calibrate_bank(model, data, cfg, device, seed=0, n_tiles=256):
    """Scale D once so median(||F_pred||/||F_ref||) is one (bilinear-safe init)."""
    b = Batcher(data, device)
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(data.n_tiles, size=min(n_tiles, data.n_tiles), replace=False))
    # energy-weighted ratio: size the bank for the tiles that carry the energy,
    # not for the (numerous) near-zero-cancellation tiles.
    num = 0.0
    den = 0.0
    for s in range(0, len(ids), 64):
        tl = ids[s : s + 64]
        feats, w, ti, di, dw, tgt, s0, T = b.batch(tl, cfg.model_config())
        pred = model(feats, w, ti, di, dw, T)
        num += float(torch.sum(pred * pred))
        den += float(torch.sum(tgt * tgt))
    ratio = float(np.sqrt(num / max(den, 1e-30)))
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = 1.0
    with torch.no_grad():
        model.decoder.bank.mul_(1.0 / ratio)
    return ratio


def composite_loss(pred, target, s0, floor, w_pat=1.0, w_log=0.1, log_delta=2.0):
    """Bounded relative + scale-invariant pattern + log-amplitude."""
    pe, pb = _groups(pred)
    te, tb = _groups(target)
    s0 = s0.to(dtype=pred.dtype)
    total = pred.new_zeros(())
    floor = torch.as_tensor(floor, dtype=pred.dtype, device=pred.device)
    parts = {}
    for name, p, t, sc in (
        ("E", pe, te, s0[:, 0]),
        ("B", pb, tb, s0[:, 1]),
    ):
        tn = torch.sqrt(torch.sum(t * t, dim=(1, 2)) + EPS)
        pn = torch.sqrt(torch.sum(p * p, dim=(1, 2)) + EPS)
        den = torch.clamp(torch.maximum(torch.maximum(tn, sc), floor), min=EPS)
        rel_i = (((p - t) / den[:, None, None]) ** 2).sum(dim=(1, 2))
        strong = tn >= floor
        # relative term only on energy-carrying tiles; weak tiles are driven by
        # the bounded log-amplitude and the scale-free pattern term.
        rel = torch.where(strong, torch.clamp(rel_i, max=3072.0), torch.zeros_like(rel_i))
        pat = (((p / pn[:, None, None]) - (t / tn[:, None, None])) ** 2).sum(dim=(1, 2))
        pat = torch.where(strong, pat, torch.zeros_like(pat))
        dlog = torch.log(pn + EPS) - torch.log(tn + EPS)
        ad = torch.abs(dlog)
        logl = torch.where(ad < log_delta, dlog * dlog, 2.0 * log_delta * ad - log_delta * log_delta)
        # tiles with (numerically) zero field: relative amplitude/direction are
        # undefined (0/0).  Mask them from log/pattern and add a bounded
        # absolute shrink term so the exported field is small there.
        zero = tn < 1e-6
        logl = torch.where(zero, torch.zeros_like(logl), logl)
        pat = torch.where(zero, torch.zeros_like(pat), pat)
        pzero = torch.where(zero, (pn / 1e-3) ** 2, torch.zeros_like(pn))
        total = total + rel.mean() + w_pat * pat.mean() + w_log * logl.mean() + 0.02 * pzero.mean()
        parts[name] = float(rel.mean())
        parts[name + "_pat"] = float(pat.mean())
        parts[name + "_log"] = float(logl.mean())
    return total, parts


@torch.no_grad()
def z_stats(model, feats, src_w, tile_index, n_tiles):
    keep = src_w != 0
    h = model.encoder(feats[keep])
    hw = h * src_w[keep].to(h.dtype).unsqueeze(-1)
    z = torch.zeros((n_tiles, model.cfg.latent), dtype=hw.dtype, device=hw.device)
    if hw.shape[0]:
        z = z.index_add(0, tile_index[keep], hw)
    return float(torch.linalg.norm(z, dim=1).mean()), float(torch.linalg.norm(h, dim=1).mean())


@torch.no_grad()
def quick_metric(model, data, cfg, device, ids, bs=128):
    """Preregistered per-case metric on a tile subset (no force)."""
    model.eval()
    b = Batcher(data, device)
    re_all, rb_all, rn_all, bn_all, s0e, s0b = [], [], [], [], [], []
    for s in range(0, len(ids), bs):
        tl = np.sort(ids[s : s + bs])
        feats, w, ti, di, dw, tgt, s0, T = b.batch(tl, cfg.model_config())
        pred = model(feats, w, ti, di, dw, T).double()
        pe, pb = _groups(pred)
        te, tb = _groups(tgt.double())
        for p, t, out_re, out_rn, sc in ((pe, te, re_all, rn_all, s0[:, 0]), (pb, tb, rb_all, bn_all, s0[:, 1])):
            ref = torch.sqrt(torch.sum(t * t, dim=(1, 2)) + EPS)
            err = torch.sqrt(torch.sum((p - t) ** 2, dim=(1, 2)) + EPS)
            den = torch.clamp(torch.maximum(ref, sc.double()), min=EPS)
            out_re.extend((err / den).cpu().numpy().tolist())
            out_rn.extend(ref.cpu().numpy().tolist())
        s0e.extend(s0[:, 0].cpu().numpy().tolist())
        s0b.extend(s0[:, 1].cpu().numpy().tolist())
    out = {}
    for name, rel, ref, sc in (("E", re_all, rn_all, s0e), ("B", rb_all, bn_all, s0b)):
        rel = np.asarray(rel)
        ref = np.asarray(ref)
        sc = np.asarray(sc)
        den = np.maximum(ref, sc)
        zero = ref / np.maximum(den, EPS)
        out[name] = {
            "mean": float(rel.mean()),
            "p50": float(np.percentile(rel, 50)),
            "p90": float(np.percentile(rel, 90)),
            "p99": float(np.percentile(rel, 99)),
            "max": float(rel.max()),
            "weighted": float(np.sum(den * rel) / np.sum(den)),
            "absnorm_share": float(np.mean(ref < sc)),
            "zero_mean": float(zero.mean()),
            "zero_weighted": float(np.sum(den * zero) / np.sum(den)),
        }
    model.train()
    return out


def train_stable(
    cfg: TrainConfig,
    data_root,
    out_dir,
    device,
    val_subset=768,
    time_budget_s=7500.0,
    max_steps=60000,
    eval_every=2000,
    floor=5e-2,
    w_pat=0.5,
    w_log=0.1,
    lr=1e-3,
    warmup=500,
    seed=0,
    val_ids=None,
    feature_kind=None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    feature_kind = feature_kind or getattr(cfg, "feature_kind", "fourier_v1")
    train_data = load_split(data_root, "train", feature_kind=feature_kind)
    val_data = load_split(data_root, "val", feature_kind=feature_kind)
    if val_ids is None:
        stride = max(1, val_data.n_tiles // val_subset)
        val_ids = np.arange(0, val_data.n_tiles, stride)[:val_subset]
    val_ids = np.asarray(val_ids, dtype=np.int64)

    model = make_model(cfg, device)
    init_ratio = calibrate_bank(model, train_data, cfg, device, seed=seed)
    bank_norm0 = float(torch.linalg.norm(model.decoder.bank.detach()).cpu())
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )

    def factor(step):
        if step < warmup:
            return max(0.05, (step + 1) / warmup)
        t = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    batcher = Batcher(train_data, device)
    batcher.rng = np.random.default_rng(seed + 17)
    tn_all = np.linalg.norm(train_data.targets, axis=1)
    strong_ids = np.where(tn_all >= floor)[0]
    if len(strong_ids) < 4:
        strong_ids = np.arange(train_data.n_tiles)
    perm_s = batcher.rng.permutation(strong_ids)
    pos_s = 0
    curriculum_steps = int(0.15 * max_steps)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hist = []
    best = float("inf")
    best_step = 0
    step = 0
    t0 = time.time()
    mon = (out / "monitor.jsonl").open("w", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "start",
                "init_ratio": init_ratio,
                "n_train": train_data.n_tiles,
                "n_val_subset": int(len(val_ids)),
                "cfg": asdict(cfg),
                "floor": floor,
                "w_pat": w_pat,
                "w_log": w_log,
                "bank_norm0": bank_norm0,
            }
        ),
        flush=True,
    )
    while step < max_steps and (time.time() - t0) < time_budget_s:
        if step < curriculum_steps:
            if pos_s + cfg.batch_tiles > len(perm_s):
                perm_s = batcher.rng.permutation(strong_ids)
                pos_s = 0
            ids = np.sort(perm_s[pos_s : pos_s + cfg.batch_tiles])
            pos_s += cfg.batch_tiles
        else:
            ids = batcher.next_tiles(min(cfg.batch_tiles, train_data.n_tiles))
        feats, w, ti, di, dw, tgt, s0, T = batcher.batch(ids, cfg.model_config())
        model.train()
        pred = model(feats, w, ti, di, dw, T)
        loss, parts = composite_loss(pred, tgt, s0, floor, w_pat=w_pat, w_log=w_log)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0))
        opt.step()
        # fix the unidentifiable z<->D scale gauge: keep ||D|| constant so the
        # encoder (source dependent) carries the amplitude; the exported bank
        # is used as-is, so the inference function is unchanged.
        with torch.no_grad():
            bn = torch.linalg.norm(model.decoder.bank)
            if bn > 0 and bank_norm0 > 0:
                model.decoder.bank.mul_(bank_norm0 / bn)
        sched.step()
        step += 1
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                pn = float(torch.linalg.norm(pred, dim=1).mean())
                tn = float(torch.linalg.norm(tgt, dim=1).mean())
                zn, hn = z_stats(model, feats, w, ti, T)
            rec = {
                "step": step,
                "t": time.time() - t0,
                "loss": float(loss.detach()),
                "rel_E": parts["E"],
                "rel_B": parts["B"],
                "pat_E": parts["E_pat"],
                "pat_B": parts["B_pat"],
                "log_E": parts["E_log"],
                "log_B": parts["B_log"],
                "grad_norm": gnorm,
                "lr": float(opt.param_groups[0]["lr"]),
                "z_rms": zn,
                "h_rms": hn,
                "bank_norm": float(torch.linalg.norm(model.decoder.bank.detach())),
                "gate_norm": float(torch.linalg.norm(model.encoder.feature_gate.detach())),
                "pred_target": pn / max(tn, EPS),
            }
            mon.write(json.dumps(rec) + "\n")
            mon.flush()
            if step % 500 == 0:
                print(json.dumps(rec), flush=True)
        if step % eval_every == 0 or step == max_steps:
            qm = quick_metric(model, val_data, cfg, device, val_ids)
            vl, vparts = evaluate_loss(model, val_data, cfg, device, val_ids, floor, w_pat, w_log)
            rec = {
                "event": "eval",
                "step": step,
                "t": time.time() - t0,
                "val_loss": vl,
                "val_rel_E": vparts["E"],
                "val_rel_B": vparts["B"],
                "val_metric": qm,
                "lr": float(opt.param_groups[0]["lr"]),
            }
            hist.append(rec)
            mon.write(json.dumps(rec) + "\n")
            mon.flush()
            print(json.dumps(rec), flush=True)
            save_checkpoint(out / "last.pt", model, opt, cfg, step, hist, seed, best)
            if vl < best:
                best = vl
                best_step = step
                save_checkpoint(out / "best.pt", model, opt, cfg, step, hist, seed, best)
    mon.close()
    report = {
        "format": "neuralbunch-stable-training-report",
        "version": 1,
        "package_version": __version__,
        "device": str(device),
        "init_ratio": init_ratio,
        "steps": step,
        "elapsed_s": time.time() - t0,
        "best_val_loss": best,
        "best_step": best_step,
        "train_config": asdict(cfg),
        "floor": floor,
        "w_pat": w_pat,
        "w_log": w_log,
        "history": hist,
        "note": "loss floor/pattern/log terms are training-only; artifact metric uses preregistered S0",
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return model, report


@torch.no_grad()
def evaluate_loss(model, data, cfg, device, ids, floor, w_pat, w_log, bs=128):
    model.eval()
    b = Batcher(data, device)
    tot = 0.0
    n = 0
    pe_sum = 0.0
    pb_sum = 0.0
    for s in range(0, len(ids), bs):
        tl = np.sort(ids[s : s + bs])
        feats, w, ti, di, dw, tgt, s0, T = b.batch(tl, cfg.model_config())
        pred = model(feats, w, ti, di, dw, T)
        loss, parts = composite_loss(pred, tgt, s0, floor, w_pat=w_pat, w_log=w_log)
        tot += float(loss) * len(tl)
        n += len(tl)
        pe_sum += parts["E"] * len(tl)
        pb_sum += parts["B"] * len(tl)
    model.train()
    return tot / max(n, 1), {"E": pe_sum / max(n, 1), "B": pb_sum / max(n, 1)}


def export_best(run_dir, prefix):
    from .train import export_from_checkpoint

    meta = export_from_checkpoint(str(Path(run_dir) / "best.pt"), str(prefix))
    return meta
