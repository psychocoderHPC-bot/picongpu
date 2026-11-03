"""T116: symmetry-constrained near-field surrogate.

Modes
-----
hyper      : canonicalise the exact hyperoctahedral grid symmetry
             (axis permutations + axis sign flips applied to R_cells and u,
             cells permuted) and the exact velocity-sign symmetry u -> -u.
             The net predicts E in that canonical frame; the result is mapped
             back with the inverse signed permutation.
beam       : rotate to the beam frame (bhat -> +z), carrying the rotated cell
             basis so the representation stays exact; predict E in the beam
             frame, rotate back.
hyperbeam  : both (canonicalise, then beam-rotate the canonical configuration).

All modes predict E_conv only; cB is derived as beta x E_conv (exact identity).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

DATA = os.path.expanduser("~/pr24/t112/pilot_all.npz")
OUT = os.path.expanduser("~/pr24/t116/runs")

TINY = 1e-30


def canon(cells, u, Rc):
    """Exact canonical form under signed axis permutations + u -> -u.

    Returns canonical cells/u/Rc, the axis permutation p, the sign vector s and
    the velocity sign t such that x_canon = t * s * x[p] (u part) and
    Rc_canon = s * Rc[p].  E_lab[p] = s * E_canon (t does not act on E).
    """
    au = np.abs(u)
    aR = np.abs(Rc)
    p = np.lexsort((aR, au, cells), axis=1)          # primary key: cells
    Rcp = np.take_along_axis(Rc, p, axis=1)
    up = np.take_along_axis(u, p, axis=1)
    cellsn = np.take_along_axis(cells, p, axis=1)
    s = np.sign(Rcp)
    s[s == 0.0] = 1.0
    Rcn = s * Rcp
    un = s * up
    # global velocity sign: first non-zero component positive
    big = np.abs(un) > 1e-14
    first = np.argmax(big, axis=1)
    has = big.any(axis=1)
    t = np.ones(len(un))
    t[has] = np.sign(un[np.arange(len(un)), first])[has]
    t[t == 0] = 1.0
    un = un * t[:, None]
    return cellsn, un, Rcn, p, s, t


def inv_map(Yc, p, s):
    """canonical E (E_lab[p] = s*E_canon) -> lab frame."""
    Y = np.empty_like(Yc)
    np.put_along_axis(Y, p, s * Yc, axis=1)
    return Y


def beam_basis(u):
    """Rows Q with Q @ bhat = z.  u == 0 -> identity."""
    n = len(u)
    Q = np.tile(np.eye(3), (n, 1, 1))
    nrm = np.linalg.norm(u, axis=1)
    mov = nrm > 1e-14
    if not mov.any():
        return Q
    b = u[mov] / nrm[mov, None]
    ref = np.zeros(len(b), dtype=int)
    ref[np.abs(b[:, 0]) >= 0.9] = 1
    e = np.zeros((len(b), 3))
    e[np.arange(len(b)), ref] = 1.0
    e1 = np.cross(e, b)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1), TINY)[:, None]
    e2 = np.cross(b, e1)
    Qm = np.stack([e1, e2, b], axis=1)                   # rows
    Q[mov] = Qm
    return Q


def pts_beam(u):
    """pointing of the reference lab axis used for azimuth reporting."""
    n = len(u)
    nrm = np.linalg.norm(u, axis=1)
    b = np.zeros_like(u)
    mov = nrm > 1e-14
    b[mov] = u[mov] / nrm[mov, None]
    b[~mov] = np.array([0.0, 0.0, 1.0])
    ref = np.zeros_like(b)
    ref[:, 0] = 1.0
    bad = np.abs(np.einsum("ij,ij->i", ref, b)) > 0.9
    ref[bad] = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.einsum("ij,ij->i", ref, b)[:, None] * b
    e1 /= np.maximum(np.linalg.norm(e1, axis=1), TINY)[:, None]
    e2 = np.cross(b, e1)
    return e1, e2, b


def build(mode):
    d = dict(np.load(DATA, allow_pickle=False))
    Rc = np.asarray(d["R_cells"], float)
    u = np.asarray(d["u"], float)
    cells = np.asarray(d["cells"], float)
    g = np.asarray(d["gamma"], float)
    Ep = np.asarray(d["E_point"], float)
    Ec = np.asarray(d["E_conv"], float)
    n = len(Rc)
    lg = np.log(g)
    r = np.linalg.norm(Rc, axis=1)
    bn = np.linalg.norm(u, axis=1)
    bhat = u / np.maximum(bn, TINY)[:, None]
    rpar = np.einsum("ij,ij->i", Rc, bhat)
    rper = np.sqrt(np.maximum(r * r - rpar * rpar, 0.0))
    W = np.prod(np.clip(1.5 - np.abs(Rc), 0.0, None), axis=1)
    dface = 1.5 - np.max(np.abs(Rc), axis=1)

    if mode == "hyper":
        cellsn, un, Rcn, p, s, t = canon(cells, u, Rc)
        Y = s * np.take_along_axis(Ec, p, axis=1)
        Epn = s * np.take_along_axis(Ep, p, axis=1)
        bn2 = np.linalg.norm(un, axis=1)
        bh = un / np.maximum(bn2, TINY)[:, None]
        r2 = np.linalg.norm(Rcn, axis=1)
        rp = np.einsum("ij,ij->i", Rcn, bh)
        rr = np.sqrt(np.maximum(r2 * r2 - rp * rp, 0.0))
        Wn = np.prod(np.clip(1.5 - np.abs(Rcn), 0.0, None), axis=1)
        dfn = 1.5 - np.max(np.abs(Rcn), axis=1)
        cols = [Rcn[:, 0], Rcn[:, 1], Rcn[:, 2],
                np.log(cellsn[:, 0]), np.log(cellsn[:, 1]), np.log(cellsn[:, 2]),
                lg, un[:, 0], un[:, 1], un[:, 2], bh[:, 0], bh[:, 1], bh[:, 2],
                r2, rp, rr, Wn, dfn,
                Epn[:, 0], Epn[:, 1], Epn[:, 2],
                np.log(np.maximum(np.linalg.norm(Epn, axis=1), TINY))]
        X = np.stack(cols, axis=1).astype(np.float32)
        off = np.zeros_like(Y)
        meta = dict(Y=Y, Ep=Epn, p=p, s=s, Q=None, mode="hyper",
                    offset=off, Ec_model=Y.copy())

    elif mode == "beam":
        Q = beam_basis(u)
        Rp = Rc * cells
        rb = np.einsum("nij,nj->ni", Q, Rp)
        G = Q * cells[:, None, :]
        Eb = np.einsum("nij,nj->ni", Q, Ec)
        Epb = np.einsum("nij,nj->ni", Q, Ep)
        rp = np.linalg.norm(Rp, axis=1)
        cols = [rb[:, 0], rb[:, 1], rb[:, 2],
                G[:, 0, 0], G[:, 0, 1], G[:, 0, 2],
                G[:, 1, 0], G[:, 1, 1], G[:, 1, 2],
                G[:, 2, 0], G[:, 2, 1], G[:, 2, 2],
                lg, bn, rp,
                Epb[:, 0], Epb[:, 1], Epb[:, 2],
                np.log(np.maximum(np.linalg.norm(Epb, axis=1), TINY))]
        X = np.stack(cols, axis=1).astype(np.float32)
        off = np.zeros_like(Eb)
        meta = dict(Y=Eb, Ep=Epb, p=None, s=None, Q=Q, mode="beam",
                    offset=off, Ec_model=Eb.copy())

    elif mode == "hyperbeam":
        cellsn, un, Rcn, p, s, t = canon(cells, u, Rc)
        Ec_c = s * np.take_along_axis(Ec, p, axis=1)
        Ep_c = s * np.take_along_axis(Ep, p, axis=1)
        Q = beam_basis(un)
        Rp = Rcn * cellsn
        rb = np.einsum("nij,nj->ni", Q, Rp)
        G = Q * cellsn[:, None, :]
        Eb = np.einsum("nij,nj->ni", Q, Ec_c)
        Epb = np.einsum("nij,nj->ni", Q, Ep_c)
        bn2 = np.linalg.norm(un, axis=1)
        rp = np.linalg.norm(Rp, axis=1)
        cols = [rb[:, 0], rb[:, 1], rb[:, 2],
                G[:, 0, 0], G[:, 0, 1], G[:, 0, 2],
                G[:, 1, 0], G[:, 1, 1], G[:, 1, 2],
                G[:, 2, 0], G[:, 2, 1], G[:, 2, 2],
                lg, bn2, rp,
                Epb[:, 0], Epb[:, 1], Epb[:, 2],
                np.log(np.maximum(np.linalg.norm(Epb, axis=1), TINY))]
        X = np.stack(cols, axis=1).astype(np.float32)
        off = np.zeros_like(Eb)
        meta = dict(Y=Eb, Ep=Epb, p=p, s=s, Q=Q, mode="hyperbeam",
                    offset=off, Ec_model=Eb.copy())

    elif mode == "hyperres":
        cellsn, un, Rcn, p, s, t = canon(cells, u, Rc)
        Ec_c = s * np.take_along_axis(Ec, p, axis=1)
        Ep_c = s * np.take_along_axis(Ep, p, axis=1)
        bn2 = np.linalg.norm(un, axis=1)
        bh = un / np.maximum(bn2, TINY)[:, None]
        r2 = np.linalg.norm(Rcn, axis=1)
        rp = np.einsum("ij,ij->i", Rcn, bh)
        rr = np.sqrt(np.maximum(r2 * r2 - rp * rp, 0.0))
        Wn = np.prod(np.clip(1.5 - np.abs(Rcn), 0.0, None), axis=1)
        dfn = 1.5 - np.max(np.abs(Rcn), axis=1)
        cols = [Rcn[:, 0], Rcn[:, 1], Rcn[:, 2],
                np.log(cellsn[:, 0]), np.log(cellsn[:, 1]), np.log(cellsn[:, 2]),
                lg, un[:, 0], un[:, 1], un[:, 2], bh[:, 0], bh[:, 1], bh[:, 2],
                r2, rp, rr, Wn, dfn,
                Ep_c[:, 0], Ep_c[:, 1], Ep_c[:, 2],
                np.log(np.maximum(np.linalg.norm(Ep_c, axis=1), TINY))]
        X = np.stack(cols, axis=1).astype(np.float32)
        meta = dict(Y=Ec_c - Ep_c, Ep=Ep_c, p=p, s=s, Q=None, mode="hyperres",
                    offset=Ep_c.copy(), Ec_model=Ec_c.copy())
    else:
        raise ValueError(mode)
    return d, X, meta


def to_lab(meta, Yhat, Ep_target_beam=None):
    """model-space prediction -> lab E_conv (n,3) float64."""
    Yhat = Yhat + meta["offset"]
    if meta["mode"] in ("hyper", "hyperres"):
        return inv_map(Yhat, meta["p"], meta["s"])
    if meta["mode"] == "beam":
        return np.einsum("nji,nj->ni", meta["Q"], Yhat)       # Q^T @ y
    Ec = np.einsum("nji,nj->ni", meta["Q"], Yhat)             # beam -> canon
    return inv_map(Ec, meta["p"], meta["s"])


def mlp(nin, width, depth, out=3):
    layers, n = [], nin
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.SiLU()]
        n = width
    layers.append(nn.Linear(n, out))
    return nn.Sequential(*layers)


def metrics(pred_E, d, sel):
    G = beta_of(d, sel)
    Ec = np.asarray(d["E_conv"], float)[sel]
    Bc = np.asarray(d["cB_conv"], float)[sel]
    pB = np.cross(G, pred_E)
    err = np.concatenate([pred_E - Ec, pB - Bc], axis=1)
    ref = np.concatenate([Ec, Bc], axis=1)
    rel = np.abs(err) / (1.0 + np.abs(ref))
    names = ["Ex", "Ey", "Ez", "cBx", "cBy", "cBz"]
    return {
        "worst": float(rel.max()), "p99": float(np.quantile(rel, 0.99)),
        "median": float(np.median(rel)),
        "pooled_E": float(np.linalg.norm(pred_E - Ec) / max(np.linalg.norm(Ec), TINY)),
        "pooled_cB": float(np.linalg.norm(pB - Bc) / max(np.linalg.norm(Bc), TINY)),
        "per_comp_p99": {k: float(np.quantile(rel[:, i], 0.99)) for i, k in enumerate(names)},
        "per_comp_worst": {k: float(rel[:, i].max()) for i, k in enumerate(names)},
    }


def beta_of(d, sel):
    u = np.asarray(d["u"], float)[sel]
    return u / np.sqrt(1.0 + np.einsum("ij,ij->i", u, u))[:, None]


def rel_rows(pred_E, d, sel):
    b = beta_of(d, sel)
    Ec = np.asarray(d["E_conv"], float)[sel]
    Bc = np.asarray(d["cB_conv"], float)[sel]
    pB = np.cross(b, pred_E)
    rel = np.abs(np.concatenate([pred_E - Ec, pB - Bc], axis=1)) / (
        1.0 + np.abs(np.concatenate([Ec, Bc], axis=1)))
    return rel.max(axis=1)


def rel_matrix(pred_E, d, sel):
    """component-wise relative error, shape (n, 6) -- same definition as t113."""
    b = beta_of(d, sel)
    Ec = np.asarray(d["E_conv"], float)[sel]
    Bc = np.asarray(d["cB_conv"], float)[sel]
    pB = np.cross(b, pred_E)
    return np.abs(np.concatenate([pred_E - Ec, pB - Bc], axis=1)) / (
        1.0 + np.abs(np.concatenate([Ec, Bc], axis=1)))


def strat(pred_E, d, sel, Q=None, canon_p=None):
    rel = rel_matrix(pred_E, d, sel)
    rowmax = rel.max(axis=1)
    g = np.asarray(d["gamma"], float)[sel]
    Rc = np.asarray(d["R_cells"], float)[sel]
    u = np.asarray(d["u"], float)[sel]
    out = {}
    for gg in np.unique(g):
        m = g == gg
        out[f"gamma{gg:g}"] = {"n": int(m.sum()), "worst": float(rel[m].max()),
                               "p99": float(np.quantile(rel[m], 0.99)),
                               "rowmax_worst": float(rowmax[m].max()),
                               "rowmax_p99": float(np.quantile(rowmax[m], 0.99))}
    e1, e2, b = pts_beam(u)
    az = np.degrees(np.arctan2(np.einsum("ij,ij->i", Rc, e2),
                               np.einsum("ij,ij->i", Rc, e1)))
    for lo, hi in [(-180, -90), (-90, 0), (0, 90), (90, 180)]:
        m = (az >= lo) & (az < hi)
        if m.any():
            out[f"az[{lo},{hi})"] = {"n": int(m.sum()),
                                     "worst": float(rel[m].max()),
                                     "p99": float(np.quantile(rel[m], 0.99))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", required=True, choices=["hyper", "beam", "hyperbeam", "hyperres"])
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    d, X, meta = build(a.mode)
    Y = meta["Y"]
    split = np.asarray(d["split"])
    trf, va, te = split == "train", split == "val", split == "test"

    muX = X[trf].mean(0); sdX = X[trf].std(0); sdX[sdX < 1e-12] = 1.0
    Xn = (X - muX) / sdX
    muY = Y[trf].mean(0); sdY = Y[trf].std(0); sdY[sdY < 1e-12] = 1.0
    Z = (Y - muY) / sdY

    Xt = torch.from_numpy(Xn).to(dev)
    Zt = torch.from_numpy(Z.astype(np.float32)).to(dev)
    tr_idx = torch.from_numpy(np.where(trf)[0]).to(dev)
    muY_t = torch.from_numpy(muY.astype(np.float32)).to(dev)
    sdY_t = torch.from_numpy(sdY.astype(np.float32)).to(dev)
    off_t = torch.from_numpy(meta["offset"].astype(np.float32)).to(dev)
    Ecm_t = torch.from_numpy(meta["Ec_model"].astype(np.float32)).to(dev)

    model = mlp(X.shape[1], a.width, a.depth, 3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def predict_all():
        model.eval()
        with torch.no_grad():
            y = model(Xt).cpu().numpy().astype(np.float64)
        return to_lab(meta, y * sdY + muY)

    rng = np.random.default_rng(a.seed)
    best = {"val": 1e30, "epoch": -1, "state": None}
    hist = []
    Ec_t = torch.from_numpy(np.asarray(d["E_conv"], np.float32)).to(dev)
    for ep in range(a.epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        tot = 0.0
        for s in range(0, len(perm), a.batch):
            idx = perm[s:s + a.batch]
            out = model(Xt[idx])
            # loss in the model frame against the canonical/beam target
            ref = Ecm_t[idx]
            e = (out * sdY_t + muY_t + off_t[idx] - ref) / (1.0 + torch.abs(ref))
            loss = torch.mean(e * e)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if ep % 25 == 0 or ep == a.epochs - 1:
            pv = predict_all()
            mv = metrics(pv[va], d, va)
            hist.append({"epoch": ep, "val_p99": mv["p99"], "val_worst": mv["worst"],
                         "loss": tot / len(tr_idx)})
            if mv["p99"] < best["val"]:
                best = {"val": mv["p99"], "worst": mv["worst"], "epoch": ep,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            if ep % 200 == 0:
                print(f"[{a.tag}] ep {ep} loss {tot/len(tr_idx):.3e} "
                      f"val_p99 {mv['p99']:.3e} val_worst {mv['worst']:.3e}", flush=True)

    last_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rep = {"tag": a.tag, "args": vars(a), "n_features": int(X.shape[1]),
           "n_params": int(sum(p.numel() for p in model.parameters())),
           "best_epoch": best["epoch"], "hist": hist}

    model.load_state_dict(best["state"])
    pall = predict_all()
    for nm, sel in (("train", trf), ("val", va), ("test", te)):
        rep[nm] = metrics(pall[sel], d, sel)
        rep[nm + "_strat"] = strat(pall[sel], d, sel)
    model.load_state_dict(last_state)
    plast = predict_all()
    rep["final"] = {nm: metrics(plast[sel], d, sel)
                    for nm, sel in (("train", trf), ("val", va), ("test", te))}
    rep["final_train_strat"] = strat(plast[trf], d, trf)
    rep["final_test_strat"] = strat(plast[te], d, te)

    # worst held-out points for diagnosis
    rel = rel_rows(plast[te], d, te)
    di = np.where(te)[0]
    top = np.argsort(rel)[-10:][::-1]
    rep["test_worst_points"] = [{
        "row": int(di[int(t)]), "rel": float(rel[int(t)]),
        "R_cells": [float(x) for x in d["R_cells"][di[int(t)]]],
        "u": [float(x) for x in d["u"][di[int(t)]]],
        "gamma": float(d["gamma"][di[int(t)]]),
        "cells": [float(x) for x in d["cells"][di[int(t)]]],
        "config": str(d["config_key"][di[int(t)]])} for t in top]

    with open(os.path.join(a.out, f"{a.tag}.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    w = {f"w{i}": v.detach().cpu().numpy() for i, v in enumerate(model.state_dict().values())}
    np.savez(os.path.join(a.out, f"{a.tag}.npz"), **w, feat_mu=muX, feat_sd=sdX,
             scaler_mu=muY, scaler_sd=sdY, mode=np.array([a.mode]),
             pred_best_lab=pall, pred_final_lab=plast, split=split.astype("U5"))
    for nm in ("train", "val", "test"):
        mm = rep[nm]
        print(f"{nm.upper()} worst {mm['worst']:.3e} p99 {mm['p99']:.3e} "
              f"med {mm['median']:.3e} pooledE {mm['pooled_E']:.3e}", flush=True)
    print("TEST_STRAT", json.dumps(rep["test_strat"]), flush=True)
    print("FINAL_TEST_STRAT", json.dumps(rep["final_test_strat"]), flush=True)


if __name__ == "__main__":
    main()
