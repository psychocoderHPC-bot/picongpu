"""T113: bounded shape-factor target for the near-field surrogate.

Target (formulation 1, per component):

    f = E_conv * D**1.5 * gamma**2          (D and gamma exactly as in
                                              Features.hpp::pointFieldDx)
    E_conv = f * gamma**-2 * D**-1.5

D = r_par**2 + perp**2/gamma**2 = |R|**2/gamma**2 + (R.beta)**2  (identical
algebra).  The point field is E_point = gamma**-2 R D**-1.5, so f -> R in the
far field and the reconstruction is exact there.  Because E_conv is finite and
D**1.5 -> 0 as R -> 0, f -> 0 at the source: the 1/r**2 singularity of
delta = E_conv - E_point is gone.

We train on arcsinh(f) (signed log) so the huge dynamic range of the analytic
factor gamma**2 D**1.5 is conditioned; arcsinh is a bijection, the
reconstruction stays exact.

Modes:
  conv       : Y = E_conv (baseline, T112 c2 style)
  shape      : Y = f (standardised)
  shape_log  : Y = arcsinh(f)
  ratio      : g = |E_conv|/|E_point| (arcsinh) + unit direction (3)

Runs on rosi_v100 with ~/pr24/t110-venv.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

DATA = os.path.expanduser("~/pr24/t112/pilot_all.npz")
OUT = os.path.expanduser("~/pr24/t113/runs")


def geom(d):
    Rc = np.asarray(d["R_cells"], dtype=np.float64)
    cells = np.asarray(d["cells"], dtype=np.float64)
    u = np.asarray(d["u"], dtype=np.float64)
    g = np.asarray(d["gamma"], dtype=np.float64)
    R = Rc * cells
    r = np.linalg.norm(R, axis=1)
    uu2 = np.einsum("ij,ij->i", u, u)
    invg2 = 1.0 / (1.0 + uu2)
    beta = u / np.sqrt(1.0 + uu2)[:, None]
    rbeta = np.einsum("ij,ij->i", R, beta)
    D = r * r * invg2 + rbeta * rbeta
    Ep = d["E_point"]
    return dict(Rc=Rc, R=R, r=r, u=u, g=g, beta=beta, rbeta=rbeta, D=D,
                invg2=invg2, Ep=Ep)


def feats(d, kind="phys"):
    G = geom(d)
    Rc, R, r, u, g, beta, rbeta, D, Ep = (G[k] for k in
        ("Rc", "R", "r", "u", "g", "beta", "rbeta", "D", "Ep"))
    rhat = R / np.maximum(r[:, None], 1e-30)
    bn = np.linalg.norm(beta, axis=1)
    bhat = beta / np.maximum(bn[:, None], 1e-30)
    rpar = np.einsum("ij,ij->i", R, bhat)
    rper = np.sqrt(np.maximum(r * r - rpar * rpar, 0.0))
    inv = 1.0 / (1.0 + r)
    xi = g * rper * inv
    eta = g * rpar * inv
    lg = np.log(g)
    ay = np.asarray(d["aspect_y"], dtype=np.float64)
    az = np.asarray(d["aspect_z"], dtype=np.float64)
    amax = np.max(np.abs(Rc), axis=1)
    dface = 1.5 - amax
    W = np.prod(np.clip(1.5 - np.abs(Rc), 0.0, None), axis=1)
    cols = [Rc[:, 0], Rc[:, 1], Rc[:, 2], R[:, 0], R[:, 1], R[:, 2],
            r, rhat[:, 0], rhat[:, 1], rhat[:, 2],
            beta[:, 0], beta[:, 1], beta[:, 2],
            bhat[:, 0], bhat[:, 1], bhat[:, 2],
            lg, np.log(ay), np.log(az), rpar, rper, xi, eta, inv, dface, W]
    if kind in ("phys", "phys2"):
        logD = np.log(np.maximum(D, 1e-300))
        lnS = np.log(g * g) + 1.5 * logD          # log(gamma^2 D^1.5)
        lep = np.log(np.maximum(np.linalg.norm(Ep, axis=1), 1e-300))
        cols += [logD, lnS, lep, Ep[:, 0], Ep[:, 1], Ep[:, 2]]
    return np.stack(cols, axis=1).astype(np.float32)


def mlp(nin, width, depth, out=3):
    layers, n = [], nin
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.SiLU()]
        n = width
    layers.append(nn.Linear(n, out))
    return nn.Sequential(*layers)


def targets(d, mode):
    G = geom(d)
    Ec = np.asarray(d["E_conv"], dtype=np.float64)
    Ep = G["Ep"]
    f = Ec * (G["D"] ** 1.5)[:, None] * (G["g"] ** 2)[:, None]
    if mode == "conv":
        return Ec, None
    if mode == "shape":
        return f, None
    if mode == "shape_log":
        return np.arcsinh(f), None
    if mode == "shape_signlog":
        return np.sign(f) * np.log1p(np.abs(f)), None
    if mode == "ratio":
        g = np.linalg.norm(Ec, axis=1) / np.maximum(np.linalg.norm(Ep, axis=1), 1e-300)
        dirn = Ec / np.maximum(np.linalg.norm(Ec, axis=1)[:, None], 1e-300)
        return np.concatenate([np.arcsinh(g)[:, None], dirn], axis=1), (g, dirn)
    raise ValueError(mode)


def reconstruct(y, mode, G, sd=None, mu=None):
    """model output y -> E_conv (n,3) float64."""
    if mode == "conv":
        return (y * sd + mu) if sd is not None else y
    invg2 = G["invg2"][:, None]
    g2 = (G["g"] ** 2)[:, None]
    Dm32 = (G["D"] ** -1.5)[:, None]
    if mode == "shape":
        f = (y * sd + mu) if sd is not None else y
        return f * invg2 * Dm32
    if mode == "shape_log":
        return np.sinh(np.clip(y, -25.0, 25.0)) * invg2 * Dm32
    if mode == "shape_signlog":
        return np.sign(y) * np.expm1(np.abs(y)) * invg2 * Dm32
    if mode == "ratio":
        gg = np.sinh(y[:, 0])
        dirn = y[:, 1:]
        dirn = dirn / np.maximum(np.linalg.norm(dirn, axis=1)[:, None], 1e-30)
        Epn = np.linalg.norm(G["Ep"], axis=1)
        return (gg * Epn)[:, None] * dirn
    raise ValueError(mode)


def metrics(pred_E, d):
    G = geom(d)
    beta = G["beta"]
    Ec = np.asarray(d["E_conv"], dtype=np.float64)
    Bc = np.asarray(d["cB_conv"], dtype=np.float64)
    pB = np.cross(beta, pred_E)
    err = np.concatenate([pred_E - Ec, pB - Bc], axis=1)
    ref = np.concatenate([Ec, Bc], axis=1)
    rel = np.abs(err) / (1.0 + np.abs(ref))
    names = ["Ex", "Ey", "Ez", "cBx", "cBy", "cBz"]
    return {
        "worst": float(np.max(rel)),
        "p99": float(np.quantile(rel, 0.99)),
        "median": float(np.median(rel)),
        "pooled_E": float(np.linalg.norm(pred_E - Ec) / max(np.linalg.norm(Ec), 1e-300)),
        "pooled_cB": float(np.linalg.norm(pB - Bc) / max(np.linalg.norm(Bc), 1e-300)),
        "per_comp_worst": {k: float(np.max(rel[:, i])) for i, k in enumerate(names)},
        "per_comp_p99": {k: float(np.quantile(rel[:, i], 0.99)) for i, k in enumerate(names)},
    }


def strat(pred_E, d):
    G = geom(d)
    Ec = np.asarray(d["E_conv"], dtype=np.float64)
    Bc = np.asarray(d["cB_conv"], dtype=np.float64)
    pB = np.cross(G["beta"], pred_E)
    rel = np.concatenate([pred_E - Ec, pB - Bc], axis=1) / (1.0 + np.abs(
        np.concatenate([Ec, Bc], axis=1)))
    g = np.asarray(d["gamma"])
    buck = np.array([k.split("|")[4] for k in d["config_key"]])
    out = {}
    for gg in np.unique(g):
        m = g == gg
        out[f"gamma{gg:g}"] = float(np.max(rel[m]))
        out[f"gamma{gg:g}|p99"] = float(np.quantile(rel[m], 0.99))
        for b in ["deep", "inside", "face", "near", "mid", "far"]:
            mm = m & (buck == b)
            if mm.any():
                out[f"gamma{gg:g}|{b}"] = float(np.max(rel[mm]))
    for b in ["deep", "inside", "face", "near", "mid", "far"]:
        m = buck == b
        if m.any():
            out[f"all|{b}"] = float(np.max(rel[m]))
    return out


def limit_check(d, sel):
    """f -> R and E_recon(f=R) == E_point exactly (far-field limit)."""
    G = geom(d)
    Ec = np.asarray(d["E_conv"], dtype=np.float64)
    Ep = G["Ep"]
    f = Ec * (G["D"] ** 1.5)[:, None] * (G["g"] ** 2)[:, None]
    R = G["R"]
    # representation limit: substituting f=R must reproduce E_point
    Erec_point = R * G["invg2"][:, None] * (G["D"] ** -1.5)[:, None]
    lim_exact = float(np.max(np.abs(Erec_point[sel] - Ep[sel]) /
                             (1.0 + np.abs(Ep[sel]))))
    # empirical approach rate of f to R
    rr = np.linalg.norm(R[sel], axis=1)
    dev = np.linalg.norm(f[sel] - R[sel], axis=1)
    order = np.argsort(rr)
    out = {"substitution_max_err": lim_exact}
    far = (np.linalg.norm(R[sel], axis=1) > 4.0)
    gs = np.asarray(d["gamma"])[sel]
    for gg in np.unique(gs[far]):
        m = far & (gs == gg)
        out[f"far_fdev_g{gg:g}"] = float(np.median(
            np.linalg.norm(f[sel][m] - R[sel][m], axis=1) / np.maximum(np.linalg.norm(R[sel][m], axis=1), 1e-30)))
    for lo, hi in [(0, 1.5), (1.5, 3), (3, 4.5), (4.5, 6.5), (6.5, 1e9)]:
        m = order[(rr >= lo) & (rr < hi)]
        if len(m):
            out[f"r[{lo:g},{hi:g})"] = {
                "n": int(len(m)),
                "f_dev_rel": float(np.median(dev[m] / np.maximum(rr[m], 1e-30))),
                "f_dev_abs": float(np.median(dev[m])),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="shape_log",
                    choices=["conv", "shape", "shape_log", "shape_signlog", "ratio"])
    ap.add_argument("--features", default="phys", choices=["base", "phys", "phys2"])
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--loss", default="rel2", choices=["rel2", "rel4"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    d = dict(np.load(DATA, allow_pickle=False))
    G = geom(d)
    X = feats(d, a.features)
    split = np.asarray(d["split"])
    trf, va, te = split == "train", split == "val", split == "test"
    if a.frac < 1.0:
        rng = np.random.default_rng(12345)
        tr = trf & (rng.random(len(split)) < a.frac)
    else:
        tr = trf
    muX = X[tr].mean(0); sdX = X[tr].std(0); sdX[sdX < 1e-12] = 1.0
    Xn = (X - muX) / sdX

    Y, _ = targets(d, a.mode)
    muY = Y[tr].mean(0) if a.mode in ("conv", "shape") else None
    sdY = Y[tr].std(0) if a.mode in ("conv", "shape") else None
    if sdY is not None:
        sdY[sdY < 1e-12] = 1.0
        Z = (Y - muY) / sdY
    else:
        Z = Y

    Xt = torch.from_numpy(Xn).to(dev)
    Zt = torch.from_numpy(Z.astype(np.float32)).to(dev)
    Ec_t = torch.from_numpy(np.asarray(d["E_conv"], dtype=np.float64).astype(np.float32)).to(dev)
    tr_idx = torch.from_numpy(np.where(tr)[0]).to(dev)

    Dm32_t = torch.from_numpy((G["D"] ** -1.5).astype(np.float32)).to(dev)[:, None]
    invg2_t = torch.from_numpy(G["invg2"].astype(np.float32)).to(dev)[:, None]
    Epn_t = torch.from_numpy(np.linalg.norm(G["Ep"], axis=1).astype(np.float32)).to(dev)[:, None]

    model = mlp(X.shape[1], a.width, a.depth, out=(4 if a.mode == "ratio" else 3)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def predict_all():
        model.eval()
        with torch.no_grad():
            y = model(Xt).cpu().numpy().astype(np.float64)
        return reconstruct(y, a.mode, G, sdY, muY)

    best = {"val": 1e30, "epoch": -1, "state": None}
    hist = []
    sdY_t = None if sdY is None else torch.from_numpy(sdY.astype(np.float32)).to(dev)
    muY_t = None if muY is None else torch.from_numpy(muY.astype(np.float32)).to(dev)

    for ep in range(a.epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        tot = 0.0
        for s in range(0, len(perm), a.batch):
            idx = perm[s:s + a.batch]
            out = model(Xt[idx])
            if a.mode == "conv":
                pE = out * sdY_t + muY_t
            elif a.mode == "shape":
                pE = (out * sdY_t + muY_t) * invg2_t[idx] * Dm32_t[idx]
            elif a.mode == "shape_log":
                pE = torch.sinh(torch.clamp(out, -25.0, 25.0)) * invg2_t[idx] * Dm32_t[idx]
            elif a.mode == "shape_signlog":
                pE = torch.sign(out) * torch.expm1(torch.abs(out)) * invg2_t[idx] * Dm32_t[idx]
            else:
                gg = torch.sinh(torch.clamp(out[:, 0:1], -25.0, 25.0))
                dirn = out[:, 1:]
                dirn = dirn / torch.clamp(torch.linalg.norm(dirn, dim=1, keepdim=True), min=1e-30)
                pE = gg * Epn_t[idx] * dirn
            ref = Ec_t[idx]
            e = (pE - ref) / (1.0 + torch.abs(ref))
            loss = torch.mean(e * e) if a.loss == "rel2" else torch.mean(e ** 4)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if ep % 25 == 0 or ep == a.epochs - 1:
            pv = predict_all()
            mv = metrics(pv[va], dictsel(d, va))
            hist.append({"epoch": ep, "val_worst": mv["worst"], "val_p99": mv["p99"],
                         "loss": tot / len(tr_idx)})
            score = mv["p99"]
            if score < best["val"]:
                best = {"val": score, "worst": mv["worst"], "epoch": ep,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            if ep % 200 == 0:
                print(f"[{a.tag}] ep {ep} loss {tot/len(tr_idx):.3e} val_worst {mv['worst']:.3e} "
                      f"p99 {mv['p99']:.3e}", flush=True)

    last_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best["state"])
    pall = predict_all()
    rep = {"tag": a.tag, "args": vars(a), "n_features": int(X.shape[1]),
           "n_params": int(sum(p.numel() for p in model.parameters())),
           "best_epoch": best["epoch"]}
    for name, sel in (("train", trf), ("val", va), ("test", te)):
        m = metrics(pall[sel], dictsel(d, sel))
        rep[name] = m
        rep[name + "_strat"] = strat(pall[sel], dictsel(d, sel))
    model.load_state_dict(last_state)
    plast = predict_all()
    rep["final"] = {nm: metrics(plast[sel], dictsel(d, sel)) for nm, sel in
                    (("train", tr), ("val", va), ("test", te))}
    rep["final_strat"] = strat(plast[te], dictsel(d, te))
    rep["final_train_strat"] = strat(plast[tr], dictsel(d, tr))
    rep["limit"] = limit_check(d, te)
    # far-field empirical: reconstruction with f:=R vs E_point (exactness) and f-R decay
    Gx = geom(d)
    for nm, sel in (("train", tr), ("test", te)):
        Ec = np.asarray(d["E_conv"], dtype=np.float64)[sel]
        pb = np.cross(Gx["beta"][sel], plast[sel])
        Bc = np.asarray(d["cB_conv"], dtype=np.float64)[sel]
        rel = np.abs(np.concatenate([plast[sel] - Ec, pb - Bc], axis=1)) / (
            1.0 + np.abs(np.concatenate([Ec, Bc], axis=1)))
        mx = rel.max(axis=1)
        top = np.argsort(mx)[-12:][::-1]
        di = np.where(sel)[0][top]
        rep[f"{nm}_worst_points"] = [{
            "row": int(di[j]), "rel": float(mx[top[j]]),
            "R_cells": [float(x) for x in Gx["Rc"][di[j]]],
            "gamma": float(Gx["g"][di[j]]),
            "ay": float(d["aspect_y"][di[j]]), "az": float(d["aspect_z"][di[j]]),
            "Ec": [float(x) for x in Ec[top[j]]],
            "pred": [float(x) for x in plast[sel][top[j]]],
            "Ep": [float(x) for x in Gx["Ep"][di[j]]],
            "config": str(d["config_key"][di[j]]),
        } for j in range(len(top))]
    with open(os.path.join(a.out, f"{a.tag}.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    w = {f"w{i}": v.detach().cpu().numpy() for i, v in enumerate(model.state_dict().values())}
    np.savez(os.path.join(a.out, f"{a.tag}.npz"), **w, feat_mu=muX, feat_sd=sdX,
             scaler_mu=np.array([] if muY is None else muY),
             scaler_sd=np.array([] if sdY is None else sdY),
             mode=np.array([a.mode]), features=np.array([a.features]))
    for nm in ("train", "val", "test"):
        mm = rep[nm]
        print(f"{nm.upper()} worst {mm['worst']:.3e} p99 {mm['p99']:.3e} "
              f"med {mm['median']:.3e} pooledE {mm['pooled_E']:.3e}", flush=True)
    print("STRAT", json.dumps(rep["test_strat"]), flush=True)
    print("LIMIT", json.dumps(rep["limit"]), flush=True)


def dictsel(d, sel):
    out = {}
    for k, v in d.items():
        out[k] = v[sel]
    return out


if __name__ == "__main__":
    main()
