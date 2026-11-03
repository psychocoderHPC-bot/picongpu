"""Training, checkpoints, evaluation and reporting.

The loss is scale-aware: each case (tile) is normalized by its own E and B RMS,
which makes the objective independent of source distance and charge.  An
optional force-sensitive term down-weights large-amplitude components so that
weak-field cases are not drowned out.  Constraint diagnostics (discrete
divergences on the tile) are reported but never replace runtime checks.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from . import __version__
from .export import compute_code_sha, write_artifact
from .features import DEFAULT_OCTAVES
from .model import (
    DEFAULT_AXES,
    TILE,
    ModelConfig,
    NeuralBunchModel,
    interp_indices_weights,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

EPS = 1e-12
_COMP = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")


@dataclass
class TrainConfig:
    hidden: Sequence[int] = (256, 256)
    latent: int = 128
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_tiles: int = 8
    max_steps: int = 2000
    eval_every: int = 100
    seed: int = 0
    force_weight: float = 0.0
    trials: int = 1
    time_budget_s: float = 600.0
    a_y_anchors: Sequence[float] = DEFAULT_AXES
    a_z_anchors: Sequence[float] = DEFAULT_AXES
    feature_kind: str = "fourier_v1"
    feature_gate_init: str = "ones"
    lr_schedule: str = "constant"
    warmup_steps: int = 2000
    notes: str = ""

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            hidden=tuple(int(h) for h in self.hidden),
            latent=int(self.latent),
            octaves=DEFAULT_OCTAVES,
            feature_kind=self.feature_kind,
            a_y_anchors=tuple(float(a) for a in self.a_y_anchors),
            a_z_anchors=tuple(float(a) for a in self.a_z_anchors),
            feature_gate_init=self.feature_gate_init,
        )


def _to_t(x, dtype=torch.float32, device=None):
    return torch.as_tensor(x, dtype=dtype, device=device)


class Batcher:
    def __init__(self, data, device):
        self.device = device
        self.feats = _to_t(data.feats, device=device)
        self.src_w = _to_t(data.src_w, device=device)
        self.tile_index = _to_t(data.tile_index, dtype=torch.long, device=device)
        self.targets = _to_t(data.targets, device=device)
        self.s0 = _to_t(getattr(data, "s0", np.ones((data.n_tiles, 2))), device=device)
        self.a_y = data.a_y
        self.a_z = data.a_z
        self.n_tiles = data.n_tiles
        self.cfg = None
        self.groups: List[np.ndarray] = []
        order = np.argsort(data.tile_index, kind="stable")
        ti = data.tile_index[order]
        bounds = np.searchsorted(ti, np.arange(self.n_tiles + 1))
        self.groups = [order[bounds[t] : bounds[t + 1]] for t in range(self.n_tiles)]
        self.rng = np.random.default_rng(0)
        self._perm = np.empty(0, dtype=np.int64)

    def next_tiles(self, batch):
        if self._perm.size < batch:
            self._perm = self.rng.permutation(self.n_tiles)
        sel = self._perm[:batch]
        self._perm = self._perm[batch:]
        return np.sort(sel)

    def batch(self, tile_ids, cfg):
        idx_parts = []
        ti_parts = []
        counts = []
        for local, t in enumerate(tile_ids):
            g = self.groups[t]
            idx_parts.append(g)
            ti_parts.append(np.full(len(g), local, dtype=np.int64))
            counts.append(len(g))
        idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int64)
        ti = np.concatenate(ti_parts) if ti_parts else np.zeros(0, dtype=np.int64)
        feats = self.feats[_to_t(idx, dtype=torch.long, device=self.device)]
        w = self.src_w[_to_t(idx, dtype=torch.long, device=self.device)]
        ti = _to_t(ti, dtype=torch.long, device=self.device)
        ay = self.a_y[tile_ids]
        az = self.a_z[tile_ids]
        di, dw = interp_indices_weights(ay, az, cfg.a_y_anchors, cfg.a_z_anchors)
        di = _to_t(di, dtype=torch.long, device=self.device)
        dw = _to_t(dw, dtype=torch.float32, device=self.device)
        tid = _to_t(tile_ids, dtype=torch.long, device=self.device)
        tgt = self.targets[tid]
        s0 = self.s0[tid]
        return feats, w, ti, di, dw, tgt, s0, len(tile_ids)


def _groups(F):
    # F: [T,1536] -> E [T,256,3], B [T,256,3] (component fastest)
    b = F.reshape(F.shape[0], -1, 6)
    return b[..., :3], b[..., 3:]


def loss_and_terms(pred, target, s0, force_weight=0.0):
    """Loss on the same preregistered scale as validation.

    For each case and group, ``denom = max(||A_ref||, S0)`` and the case term is
    ``||A_pred - A_ref||^2 / denom^2``.  ``S0`` is stored in the dataset
    manifest before training; no batch statistics enter the normalization.
    """
    pe, pb = _groups(pred)
    te, tb = _groups(target)
    if s0 is None:
        s0 = torch.zeros((pred.shape[0], 2), dtype=pred.dtype, device=pred.device)
    s0 = s0.to(device=pred.device, dtype=pred.dtype)
    ref_e = torch.sqrt(torch.sum(te * te, dim=(1, 2)) + EPS)
    ref_b = torch.sqrt(torch.sum(tb * tb, dim=(1, 2)) + EPS)
    den_e = torch.clamp(torch.maximum(ref_e, s0[:, 0]), min=EPS)
    den_b = torch.clamp(torch.maximum(ref_b, s0[:, 1]), min=EPS)
    n_e = float(te.shape[1] * te.shape[2])
    n_b = float(tb.shape[1] * tb.shape[2])
    de = ((pe - te) / den_e[:, None, None]) ** 2
    db = ((pb - tb) / den_b[:, None, None]) ** 2
    loss_e = de.sum(dim=(1, 2))
    loss_b = db.sum(dim=(1, 2))
    loss = loss_e + loss_b
    if force_weight > 0:
        rms_e = den_e / np.sqrt(n_e)
        rms_b = den_b / np.sqrt(n_b)
        we = 1.0 / (1.0 + torch.abs(te) / rms_e[:, None, None])
        wb = 1.0 / (1.0 + torch.abs(tb) / rms_b[:, None, None])
        fe = (de * we).sum(dim=(1, 2)) / torch.clamp(we.sum(dim=(1, 2)), min=EPS)
        fb = (db * wb).sum(dim=(1, 2)) / torch.clamp(wb.sum(dim=(1, 2)), min=EPS)
        loss = loss + force_weight * (fe + fb)
        loss_e = loss_e + force_weight * fe
        loss_b = loss_b + force_weight * fb
    return loss.mean(), lp2(loss_e), lp2(loss_b)


def lp2(t):
    return float(t.detach().double().mean().item())


def _metric_block(pred, target, s0, name):
    """Preregistered per-case metric, without excluding any case.

    ``rel_rms = ||A_model - A_ref|| / max(||A_ref||, S0)``; a case is
    ``absolute_normalized`` when ``||A_ref|| < S0``.
    """
    ref = target.double()
    err = pred.double() - ref
    if s0 is None:
        s0v = torch.zeros((ref.shape[0],), dtype=torch.float64, device=ref.device)
    else:
        s0v = s0.double().to(ref.device)
    ref_norm = torch.sqrt(torch.sum(ref * ref, dim=(1, 2)) + EPS)
    err_norm = torch.sqrt(torch.sum(err * err, dim=(1, 2)) + EPS)
    denom = torch.clamp(torch.maximum(ref_norm, s0v), min=EPS)
    rel = (err_norm / denom).cpu().numpy()
    den = denom.cpu().numpy()
    refn = ref_norm.cpu().numpy()
    absn = refn < s0v.cpu().numpy()
    finite = rel[np.isfinite(rel)]
    summary = {
        "mean_rel_rms": float(np.mean(finite)) if finite.size else float("nan"),
        "p99_rel_rms": float(np.percentile(finite, 99)) if finite.size else float("nan"),
        "max_rel_rms": float(np.max(finite)) if finite.size else float("nan"),
        "weighted_mean_rel_rms": float(np.sum(den * rel) / np.sum(den)) if np.sum(den) > 0 else float("nan"),
        "n": int(rel.size),
        "n_absolute_normalized": int(absn.sum()),
        "absolute_normalized_mean_rel_rms": float(np.mean(rel[absn])) if absn.any() else None,
        "absolute_normalized_max_rel_rms": float(np.max(rel[absn])) if absn.any() else None,
        "s0_mean": float(np.mean(s0v.cpu().numpy())) if s0v.numel() else 0.0,
        "s0_absolute_normalized_mean": float(np.mean(s0v.cpu().numpy()[absn])) if absn.any() else None,
    }
    return rel, den, absn, refn, summary


def constraint_diagnostics(pred, targets, a_y, a_z):
    """Discrete divergence of E and B in normalized tile coordinates."""
    T = pred.shape[0]
    arr = pred.detach().double().reshape(T, TILE[2], TILE[1], TILE[0], 6).cpu().numpy()
    # -> [T,c,i,j,k]
    arr = arr.transpose(0, 4, 3, 2, 1)
    d = []
    for t in range(T):
        spacing = (1.0, float(a_y[t]), float(a_z[t]))
        Ex, Ey, Ez = arr[t, 0], arr[t, 1], arr[t, 2]
        Bx, By, Bz = arr[t, 3], arr[t, 4], arr[t, 5]
        divE = (
            np.gradient(Ex, spacing[0], axis=0)
            + np.gradient(Ey, spacing[1], axis=1)
            + np.gradient(Ez, spacing[2], axis=2)
        )
        divB = (
            np.gradient(Bx, spacing[0], axis=0)
            + np.gradient(By, spacing[1], axis=1)
            + np.gradient(Bz, spacing[2], axis=2)
        )
        rms = np.sqrt(np.mean(Ex**2 + Ey**2 + Ez**2 + Bx**2 + By**2 + Bz**2)) + EPS
        d.append([float(np.sqrt(np.mean(divE**2)) / rms), float(np.sqrt(np.mean(divB**2)) / rms)])
    d = np.asarray(d)
    return {"mean_abs_divE_over_rms": float(d[:, 0].mean()), "mean_abs_divB_over_rms": float(d[:, 1].mean())}


#: Yee offsets in cell units, matching reference/convkernel.py
_YEE_OFFSETS = np.array(
    [
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    ],
    dtype=np.float64,
)


def _tile_components(F):
    """[T,1536] -> ndarray [T,c,i,j,k]."""
    T = F.shape[0]
    arr = F.detach().double().reshape(T, TILE[2], TILE[1], TILE[0], 6).cpu().numpy()
    return arr.transpose(0, 4, 3, 2, 1)


def _trilinear(tile, c, pos, origin, dx, ay, az):
    """Trilinear gather of component ``c`` at physical ``pos`` (clamped edge)."""
    cells = np.array([dx, ay * dx, az * dx])
    t = (pos - origin) / cells - _YEE_OFFSETS[c]
    hi = np.array([7.0, 7.0, 3.0])
    t = np.clip(t, 0.0, hi)
    i0 = np.floor(t).astype(np.int64)
    i0 = np.minimum(i0, hi.astype(np.int64) - 1)
    f = t - i0
    val = 0.0
    for a in (0, 1):
        wa = f[0] if a else 1.0 - f[0]
        for b in (0, 1):
            wb = f[1] if b else 1.0 - f[1]
            for d in (0, 1):
                wd = f[2] if d else 1.0 - f[2]
                idx = (i0[0] + a, i0[1] + b, i0[2] + d)
                val += wa * wb * wd * tile[c, idx[0], idx[1], idx[2]]
    return val


def trilinear_forces(F, data, tile_ids):
    """Normalized Lorentz force ``E + beta x cB`` at each source position.

    Explicitly trilinear from the six staggered components, not the PIConGPU
    gather.  Returns ``(g, beta_norm, E_norm, vxB_norm)`` arrays per case.
    """
    comps = _tile_components(F)
    g_all, beta_all, e_all, vxb_all = [], [], [], []
    for local, t in enumerate(tile_ids):
        m = data.sample_index == t
        pos = data.pos[m]
        u = data.u[m]
        if len(pos) == 0:
            g_all.append(np.zeros((0, 3)))
            beta_all.append(0.0)
            e_all.append(0.0)
            vxb_all.append(0.0)
            continue
        beta = u / np.sqrt(1.0 + np.sum(u * u, axis=1, keepdims=True))
        E = np.zeros_like(beta)
        B = np.zeros_like(beta)
        for c in range(6):
            vals = np.array(
                [_trilinear(comps[local], c, p, data.origin[t], data.dx[t], data.a_y[t], data.a_z[t]) for p in pos]
            )
            if c < 3:
                E[:, c] = vals
            else:
                B[:, c - 3] = vals
        vxb = np.cross(beta, B)
        g_all.append(E + vxb)
        beta_all.append(float(np.mean(np.linalg.norm(beta, axis=1))))
        e_all.append(float(np.linalg.norm(E)))
        vxb_all.append(float(np.linalg.norm(vxb)))
    return g_all, np.array(beta_all), np.array(e_all), np.array(vxb_all)


def force_metric(F_pred, F_ref, data, tile_ids, s0):
    g_m, beta, E_ref, vxb_ref = trilinear_forces(F_pred, data, tile_ids)
    g_r, _, _, _ = trilinear_forces(F_ref, data, tile_ids)
    rel, den, absn, refn, cancel = [], [], [], [], []
    for i in range(len(g_m)):
        gm, gr = g_m[i], g_r[i]
        e = float(np.linalg.norm(gm - gr))
        r = float(np.linalg.norm(gr))
        s0g = float(s0[i, 0] + beta[i] * s0[i, 1])
        denom = max(r, s0g, EPS)
        rel.append(e / denom)
        den.append(denom)
        refn.append(r)
        absn.append(r < s0g)
        cancel.append(E_ref[i] / max(vxb_ref[i], EPS))
    rel = np.asarray(rel)
    den = np.asarray(den)
    absn = np.asarray(absn)
    cancel = np.asarray(cancel)
    return {
        "definition": "trilinear interpolation of the six staggered components to each source, g = E_tilde + beta x cB_tilde; denom = max(||g_ref||, S0_E + beta*S0_B)",
        "mean_rel_rms": float(np.mean(rel)),
        "p99_rel_rms": float(np.percentile(rel, 99)),
        "max_rel_rms": float(np.max(rel)),
        "weighted_mean_rel_rms": float(np.sum(den * rel) / max(np.sum(den), EPS)),
        "n": int(rel.size),
        "n_absolute_normalized": int(absn.sum()),
        "absolute_normalized_mean_rel_rms": float(np.mean(rel[absn])) if absn.any() else None,
        "absolute_normalized_max_rel_rms": float(np.max(rel[absn])) if absn.any() else None,
        "cancellation_ratio_mean": float(np.mean(cancel)),
        "cancellation_ratio_median": float(np.median(cancel)),
        "per_case_rel_rms": [float(x) for x in rel],
        "per_case_absolute_normalized": [bool(x) for x in absn],
        "note": "cancellation_ratio = ||E|| / ||beta x cB|| of the reference at the sources",
    }


@torch.no_grad()
def evaluate(model, data, cfg, device, force_weight=0.0, batch=64):
    model.eval()
    batcher = Batcher(data, device)
    all_pred, all_tgt, all_s0, all_ids = [], [], [], []
    diag = []
    losses = []
    for start in range(0, data.n_tiles, batch):
        ids = np.arange(start, min(start + batch, data.n_tiles))
        feats, w, ti, di, dw, tgt, s0, T = batcher.batch(ids, cfg)
        pred = model(feats, w, ti, di, dw, T)
        losses.append(float(loss_and_terms(pred, tgt, s0, force_weight)[0].item()))
        all_pred.append(pred.double())
        all_tgt.append(tgt.double())
        all_s0.append(s0.double())
        all_ids.append(ids)
        diag.append(constraint_diagnostics(pred, tgt, data.a_y[ids], data.a_z[ids]))
    if not all_pred:
        return {"n_cases": 0}
    pred = torch.cat(all_pred)
    tgt = torch.cat(all_tgt)
    s0 = torch.cat(all_s0)
    ids = np.concatenate(all_ids)
    pe, pb = _groups(pred)
    te, tb = _groups(tgt)
    re_, den_e, abs_e, refn_e, sum_e = _metric_block(pe, te, s0[:, 0], "E")
    rb_, den_b, abs_b, refn_b, sum_b = _metric_block(pb, tb, s0[:, 1], "cB")
    force = force_metric(pred, tgt, data, ids, s0.cpu().numpy())
    diag_keys = diag[0].keys() if diag else []
    meta = []
    for i, t in enumerate(ids):
        d = data.descriptors[t] if t < len(data.descriptors) else {}
        cfg = d.get("config")
        if cfg:
            gname, direction, abucket, pmode = cfg
        else:
            gname, direction = d.get("gamma_name"), d.get("direction")
            abucket, pmode = d.get("aspect_bucket"), d.get("position_mode")
        meta.append(
            {
                "gamma_name": gname,
                "direction": direction,
                "position_mode": pmode,
                "aspect_bucket": abucket,
                "kind": d.get("kind"),
                "bunch_kind": d.get("bunch_kind"),
                "gamma": d.get("gamma"),
                "d_min_cells": float(data.d_min_cells[t]) if getattr(data, "d_min_cells", None) is not None else None,
                "a_y": float(data.a_y[t]),
                "a_z": float(data.a_z[t]),
                "s0_E": float(s0[i, 0]),
                "s0_B": float(s0[i, 1]),
                "ref_norm_E": float(refn_e[i]),
                "ref_norm_B": float(refn_b[i]),
                "rel_E": float(re_[i]),
                "rel_B": float(rb_[i]),
                "absolute_normalized_E": bool(abs_e[i]),
                "absolute_normalized_B": bool(abs_b[i]),
                "rel_force": float(force.get("per_case_rel_rms", [float("nan")] * len(ids))[i]),
                "absolute_normalized_force": bool(force.get("per_case_absolute_normalized", [False] * len(ids))[i]),
            }
        )
    return {
        "n_cases": int(data.n_tiles),
        "reference_kind": data.reference_kind,
        "loss": float(np.mean(losses)),
        "metric_definition": "per case rel_rms = ||A_model-A_ref|| / max(||A_ref||, S0); S0 preregistered in the dataset manifest from the reference configuration; absolute_normalized when ||A_ref|| < S0",
        "E": sum_e,
        "cB": sum_b,
        "force": force,
        "overall_rel_rms": {
            "E": float(np.linalg.norm(re_ * den_e) / max(np.linalg.norm(den_e), EPS)),
            "cB": float(np.linalg.norm(rb_ * den_b) / max(np.linalg.norm(den_b), EPS)),
        },
        "absolute_rms_error": {
            "E": float(np.sqrt(np.mean(((pe - te).detach().cpu().numpy()) ** 2))),
            "cB": float(np.sqrt(np.mean(((pb - tb).detach().cpu().numpy()) ** 2))),
        },
        "per_case_meta": meta,
        "constraint_diagnostics": {k: float(np.mean([d[k] for d in diag])) for k in diag_keys},
    }


def _env_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "package_version": __version__,
        "torch": getattr(torch, "__version__", None),
    }
    if torch is not None and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def save_checkpoint(path, model, opt, cfg: TrainConfig, step, history, seed, best_val):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "neuralbunch-checkpoint",
        "version": 1,
        "step": int(step),
        "config": asdict(cfg),
        "seed": int(seed),
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict() if opt is not None else None,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "history": history,
        "best_val": best_val,
        "code_sha": compute_code_sha(),
        "env": _env_info(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(payload, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model_from_config(cfg: TrainConfig):
    model = NeuralBunchModel(cfg.model_config())
    return model


def make_optimizer(model, cfg: TrainConfig):
    if cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


def train_one(cfg: TrainConfig, data, val_data, device, out_dir=None, resume=None, extra_meta=None):
    seed = int(cfg.seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    model = build_model_from_config(cfg).to(device)
    opt = make_optimizer(model, cfg)
    scheduler = None
    if cfg.lr_schedule == "cosine":
        import math

        warm = max(1, int(cfg.warmup_steps))
        total = max(warm + 1, int(cfg.max_steps))

        def _factor(s):
            if s < warm:
                return max(0.05, float(s + 1) / warm)
            t = min(1.0, float(s - warm) / float(total - warm))
            return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _factor)
    history = []
    step = 0
    best_val = float("inf")
    if resume:
        ck = load_checkpoint(resume)
        model.load_state_dict(ck["model_state"])
        if ck.get("optimizer_state"):
            opt.load_state_dict(ck["optimizer_state"])
        step = int(ck["step"])
        history = list(ck.get("history", []))
        best_val = float(ck.get("best_val", best_val))
        if "torch_rng" in ck:
            torch.set_rng_state(ck["torch_rng"])
        if "numpy_rng" in ck:
            np.random.set_state(ck["numpy_rng"])
        model.cfg = cfg.model_config()
    model_cfg = cfg.model_config()
    batcher = Batcher(data, device)
    batcher.rng = np.random.default_rng(seed + 17)
    t0 = time.time()
    steps_run = 0
    while step < cfg.max_steps and (time.time() - t0) < cfg.time_budget_s:
        if data.n_tiles == 0:
            break
        ids = batcher.next_tiles(min(cfg.batch_tiles, data.n_tiles))
        feats, w, ti, di, dw, tgt, s0, T = batcher.batch(ids, model_cfg)
        model.train()
        pred = model(feats, w, ti, di, dw, T)
        loss, le, lb = loss_and_terms(pred, tgt, s0, cfg.force_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        step += 1
        steps_run += 1
        if step % max(1, cfg.eval_every) == 0 or step == cfg.max_steps:
            val = evaluate(model, val_data, model_cfg, device, cfg.force_weight) if val_data else None
            rec = {
                "step": step,
                "train_loss": float(loss.detach().item()),
                "train_loss_E": le,
                "train_loss_B": lb,
                "elapsed_s": time.time() - t0,
                "lr": float(opt.param_groups[0]["lr"]),
            }
            if val is not None:
                rec["val_loss"] = val.get("loss")
                rec["val_overall_rel_rms"] = val.get("overall_rel_rms")
                v = val.get("loss", float("inf"))
                if v < best_val:
                    best_val = v
                    if out_dir is not None:
                        save_checkpoint(Path(out_dir) / "best.pt", model, opt, cfg, step, history, seed, best_val)
            history.append(rec)
            print(json.dumps(rec), flush=True)
    if out_dir is not None:
        save_checkpoint(Path(out_dir) / "last.pt", model, opt, cfg, step, history, seed, best_val)
    return model, history, best_val


def run_training(data_root, out_dir, cfg: TrainConfig, resume=False, splits=("train", "val", "test"), device=None):
    from .reference_data import available_splits, load_split

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    avail = set(available_splits(data_root))
    if "train" not in avail:
        raise FileNotFoundError(f"no train split under {data_root}")
    train_data = load_split(data_root, "train")
    val_data = load_split(data_root, "val") if "val" in avail else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = None
    if resume and (out_dir / "last.pt").is_file():
        resume_path = out_dir / "last.pt"
    trials = max(1, int(cfg.trials))
    trial_results = []
    best = None
    for t in range(trials):
        trial_cfg = TrainConfig(**asdict(cfg))
        trial_cfg.seed = int(cfg.seed) + t
        trial_dir = out_dir if trials == 1 else out_dir / f"trial_{t}"
        print(f"[trial {t}] seed={trial_cfg.seed} steps={trial_cfg.max_steps}", flush=True)
        model, history, best_val = train_one(
            trial_cfg,
            train_data,
            val_data,
            torch_device,
            trial_dir,
            resume=resume_path if t == 0 else None,
        )
        eval_splits = {}
        for sp in splits:
            if sp not in avail:
                continue
            d = train_data if sp == "train" else load_split(data_root, sp)
            eval_splits[sp] = evaluate(model, d, trial_cfg.model_config(), torch_device, trial_cfg.force_weight)
        result = {
            "trial": t,
            "seed": trial_cfg.seed,
            "config": asdict(trial_cfg),
            "trial_dir": str(trial_dir),
            "best_val_loss": best_val,
            "history": history,
            "evaluation": eval_splits,
        }
        trial_results.append(result)
        if best is None or best_val < best[0]:
            best = (best_val, model, trial_cfg, result)
    best_val, best_model, best_cfg, best_result = best
    ck_path = Path(best_result["trial_dir"]) / "best.pt"
    if ck_path.is_file():
        ck = load_checkpoint(ck_path)
        best_model.load_state_dict(ck["model_state"])
        best_result["selected_checkpoint"] = str(ck_path)
        best_result["selected_step"] = int(ck["step"])
        # re-evaluate the selected (best-val) weights, not the last step
        for sp in list(best_result["evaluation"].keys()):
            d = train_data if sp == "train" else load_split(data_root, sp)
            best_result["evaluation"][sp] = evaluate(
                best_model, d, best_cfg.model_config(), torch_device, best_cfg.force_weight
            )
    # always write the winning checkpoint explicitly
    save_checkpoint(
        out_dir / "best.pt",
        best_model,
        None,
        best_cfg,
        best_result.get("selected_step", best_cfg.max_steps),
        best_result["history"],
        best_cfg.seed,
        best_val,
    )
    report = {
        "format": "neuralbunch-training-report",
        "version": 1,
        "code_sha": compute_code_sha(),
        "env": _env_info(),
        "device": str(torch_device),
        "train_config": asdict(cfg),
        "trials": trial_results,
        "best_trial": best_result["trial"],
        "best_seed": best_result["seed"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return best_model, best_cfg, report


def export_from_checkpoint(checkpoint, out_prefix):
    ck = load_checkpoint(checkpoint)
    cfg = TrainConfig(**ck["config"])
    model = build_model_from_config(cfg)
    model.load_state_dict(ck["model_state"])
    meta = write_artifact(model, out_prefix, code_sha=ck.get("code_sha"))
    return meta


def validate_artifact_on_data(artifact, data_root, splits=("train", "val", "test"), out_report=None, device=None):
    from .parity import torch_model_from_artifact
    from .reference_data import available_splits, load_split

    from .export import load_artifact

    meta, tensors = load_artifact(artifact)
    model = torch_model_from_artifact(meta, tensors)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    cfg = model.cfg
    report = {
        "format": "neuralbunch-validation-report",
        "version": 1,
        "artifact": str(artifact),
        "code_sha": compute_code_sha(),
        "env": _env_info(),
        "splits": {},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for sp in splits:
        if sp not in available_splits(data_root):
            continue
        d = load_split(data_root, sp, feature_kind=meta["features"].get("kind", "fourier_v1"))
        report["splits"][sp] = evaluate(model, d, cfg, dev)
    report["reference_kind"] = {sp: report["splits"][sp].get("reference_kind") for sp in report["splits"]}
    if out_report:
        Path(out_report).parent.mkdir(parents=True, exist_ok=True)
        Path(out_report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Train the neural bunch-field initializer")
    ap.add_argument("--data", required=True, help="dataset root with train/val/test splits")
    ap.add_argument("--out", required=True, help="run output directory")
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-tiles", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force-weight", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--time-budget-s", type=float, default=600.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
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
    )
    _, _, report = run_training(args.data, args.out, cfg, resume=args.resume, device=args.device)
    print(json.dumps({"best_trial": report["best_trial"], "best_seed": report["best_seed"]}, indent=2))
    return 0
