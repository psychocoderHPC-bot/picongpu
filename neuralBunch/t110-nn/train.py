"""T110 residual neural surrogate trainer (runs on rosi_v100).

Predicts delta_E_hat = (E_conv - E_point) * |R_phys|^2/dx^2 (3 outputs).
cB is derived exactly: cB = beta x E, beta = u/gamma.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

PILOT = os.path.expanduser("~/pr24/t103/data/pilot/shards/*.npz")

FKEYS = ["R_cells", "u", "cells", "gamma", "aspect_y", "aspect_z", "r_cells",
         "delta_E_hat", "delta_cB_hat", "norm", "E_conv", "E_point",
         "cB_conv", "cB_point", "split", "config_key", "row"]


def load_pilot(pattern=None):
    fs = sorted(glob.glob(pattern or PILOT))
    d = {}
    for k in FKEYS:
        d[k] = np.concatenate([np.load(f)[k] for f in fs], axis=0)
    return d


def derived(d, feature_set):
    Rc = np.asarray(d["R_cells"], dtype=np.float64)
    cells = np.asarray(d["cells"], dtype=np.float64)
    u = np.asarray(d["u"], dtype=np.float64)
    gamma = np.asarray(d["gamma"], dtype=np.float64)
    R = Rc * cells
    r = np.linalg.norm(R, axis=1)
    un = np.linalg.norm(u, axis=1)
    b = np.where(un[:, None] > 1e-12, u / np.maximum(un[:, None], 1e-30), 0.0)
    rpar = np.sum(R * b, axis=1)
    rper2 = np.maximum(r * r - rpar * rpar, 0.0)
    rper = np.sqrt(rper2)
    lg = np.log(gamma)
    lay = np.log(np.asarray(d["aspect_y"], dtype=np.float64))
    laz = np.log(np.asarray(d["aspect_z"], dtype=np.float64))
    inv = 1.0 / (1.0 + r)
    gpar = gamma * rpar * inv
    gperp = gamma * rper * inv
    nrm = np.maximum(np.asarray(d["norm"], dtype=np.float64), 1e-12)
    econ = np.linalg.norm(np.asarray(d["E_conv"], dtype=np.float64), axis=1)
    w = 1.0 / (nrm * (1.0 + econ))

    if feature_set == "raw":
        cols = [R[:, 0], R[:, 1], R[:, 2], b[:, 0], b[:, 1], b[:, 2],
                lg, lay, laz, r, rpar, rper]
    elif feature_set == "phys":
        cols = [R[:, 0] * inv, R[:, 1] * inv, R[:, 2] * inv,
                b[:, 0], b[:, 1], b[:, 2], lg, lay, laz,
                r * inv, rpar * inv, rper * inv,
                gpar, gperp, inv, gpar * gpar, gperp * gperp]
    elif feature_set == "fourier":
        cols = [R[:, 0] * inv, R[:, 1] * inv, R[:, 2] * inv,
                b[:, 0], b[:, 1], b[:, 2], lg, lay, laz,
                r * inv, rpar * inv, rper * inv, inv]
        for k in (1, 2, 3):
            cols += [np.sin(k * gpar), np.cos(k * gpar),
                     np.sin(k * gperp), np.cos(k * gperp)]
    elif feature_set == "rff":
        cols = [R[:, 0] * inv, R[:, 1] * inv, R[:, 2] * inv,
                b[:, 0], b[:, 1], b[:, 2], lg, lay, laz,
                r * inv, rpar * inv, rper * inv, inv, gpar, gperp]
        for f in (0.5, 1.0, 2.0, 4.0, 8.0):
            cols += [np.sin(f * R[:, 0] * inv), np.cos(f * R[:, 0] * inv),
                     np.sin(f * R[:, 1] * inv), np.cos(f * R[:, 1] * inv),
                     np.sin(f * R[:, 2] * inv), np.cos(f * R[:, 2] * inv)]
            cols += [np.sin(f * gpar), np.cos(f * gpar),
                     np.sin(f * gperp), np.cos(f * gperp)]
    else:
        raise ValueError(feature_set)
    X = np.stack(cols, axis=1)
    return X, R, b, r, gamma, w, nrm, econ


def mlp(nin, width, depth, drop, out=3, out_bound=0.0):
    layers = []
    n = nin
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.SiLU()]
        if drop > 0:
            layers.append(nn.Dropout(drop))
        n = width
    layers.append(nn.Linear(n, out))
    if out_bound > 0:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers), out_bound


def metrics(y_pred_hat, E_point, E_conv, cB_conv, u, gamma, norm, r, sel=None):
    """Judge-style max_rel on absolute fields. Returns dict of arrays/scalars."""
    if sel is None:
        sel = np.ones(len(r), dtype=bool)
    yp = y_pred_hat[sel]
    ep = E_point[sel]
    ec = E_conv[sel]
    bc = cB_conv[sel]
    gg = gamma[sel]
    rr = r[sel]
    nn_ = norm[sel]
    uu = u[sel]
    E = ep + yp / nn_[:, None]
    beta = uu / gg[:, None]
    cB = np.cross(beta, E)
    err = np.concatenate([E - ec, cB - bc], axis=1)
    ref = np.concatenate([ec, bc], axis=1)
    rel = np.abs(err) / (1.0 + np.abs(ref))
    pooled = np.linalg.norm(err, axis=0) / np.maximum(np.linalg.norm(ref, axis=0), 1e-300)
    return {
        "max_rel": float(np.max(rel)),
        "per_comp": [float(np.max(rel[:, c])) for c in range(6)],
        "pooled": [float(pooled[c]) for c in range(6)],
        "rel": rel, "r": rr, "gamma": gg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feature-set", default="phys")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--loss", default="weighted", choices=["plain", "weighted", "weighted_sq"])
    ap.add_argument("--target-scale", default="gamma", choices=["none", "gamma", "gamma07"])
    ap.add_argument("--deep-weight", type=float, default=1e-3)
    ap.add_argument("--weight-cap-pct", type=float, default=98.0)
    ap.add_argument("--out", default=os.path.expanduser("~/pr24/t110/models"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", type=int, default=0, help="1: add R->-R, delta->-delta copy")
    ap.add_argument("--out-bound", type=float, default=0.0)
    ap.add_argument("--shard-glob", default=None)
    ap.add_argument("--rows-per-config", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    d = load_pilot(args.shard_glob)
    if args.rows_per_config > 0:
        keep = np.asarray(d["row"], dtype=np.int64) < args.rows_per_config
        d = {k: v[keep] for k, v in d.items()}
        print(f"rows-per-config {args.rows_per_config}: kept {keep.sum()} of {len(keep)}", flush=True)
    X, R, b, r, gamma, w, nrm, econ = derived(d, args.feature_set)
    Y = np.asarray(d["delta_E_hat"], dtype=np.float64)
    Xneg = None
    if args.augment:
        dneg = dict(d)
        dneg["R_cells"] = -np.asarray(d["R_cells"], dtype=np.float64)
        Xneg, _, _, _, _, _, _, _ = derived(dneg, args.feature_set)
    E_point = np.asarray(d["E_point"], dtype=np.float64)
    E_conv = np.asarray(d["E_conv"], dtype=np.float64)
    cB_conv = np.asarray(d["cB_conv"], dtype=np.float64)
    u = np.asarray(d["u"], dtype=np.float64)
    sp = d["split"]
    tr = sp == "train"
    va = sp == "val"
    te = sp == "test"

    mu = X[tr].mean(0)
    sd = X[tr].std(0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    if args.target_scale == "gamma":
        tscale = gamma
    elif args.target_scale == "gamma07":
        tscale = (gamma ** 0.7)
    else:
        tscale = np.ones(len(gamma))
    Yscaled = Y / tscale[:, None]
    yscale = Yscaled[tr].std(0)
    yscale[yscale < 1e-12] = 1.0
    Ys = Yscaled / yscale

    # weight: judge-sensitivity, but judge radii start at 0.5 cells; the pilot
    # deep bucket (r<0.5) is not judged, so suppress it and cap the tail.
    w = w.copy()
    w[r < 0.45] *= args.deep_weight
    cap = np.quantile(w[tr], args.weight_cap_pct / 100.0)
    w = np.minimum(w, cap)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_idx = np.where(tr)[0]
    va_idx = np.where(va)[0]
    if Xneg is not None:
        X_aug = np.concatenate([X[tr], Xneg[tr]], axis=0)
        Y_aug = np.concatenate([Y[tr], -Y[tr]], axis=0)
        w_aug = np.concatenate([w[tr], w[tr]])
        tscale_aug = np.concatenate([tscale[tr], tscale[tr]])
    else:
        X_aug = X[tr]
        Y_aug = Y[tr]
        w_aug = w[tr]
        tscale_aug = tscale[tr]
    Xs_aug = (X_aug - mu) / sd
    Ys_aug = Y_aug / tscale_aug[:, None] / yscale
    Xt = torch.tensor(Xs_aug, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Ys_aug, dtype=torch.float32, device=dev)
    Wt = torch.tensor(w_aug, dtype=torch.float32, device=dev)

    net, _ob = mlp(X.shape[1], args.width, args.depth, args.drop, out_bound=args.out_bound)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = (1e9, None, -1)
    bad = 0
    hist = []
    for epoch in range(args.epochs):
        net.train()
        perm = np.random.permutation(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            xb = Xt[idx]
            yb = Yt[idx]
            wb = Wt[idx]
            pred = net(xb) * (_ob if _ob > 0 else 1.0)
            e2 = torch.sum((pred - yb) ** 2, dim=1)
            if args.loss == "plain":
                loss = e2.mean()
            elif args.loss == "weighted":
                loss = (wb * e2).sum() / wb.sum()
            else:
                loss = ((wb * wb) * e2).sum() / (wb * wb).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                yp = net(torch.tensor(Xs[va_idx], dtype=torch.float32, device=dev)).cpu().numpy() * (_ob if _ob > 0 else 1.0) * yscale * tscale[va_idx][:, None]
            m = metrics(yp, E_point[va_idx], E_conv[va_idx], cB_conv[va_idx], u[va_idx],
                        gamma[va_idx], nrm[va_idx], r[va_idx])
            val_all = m["max_rel"]
            rel = m["rel"]
            keep = r[va_idx] >= 0.45
            val = float(np.max(rel[keep]))
            hist.append({"epoch": epoch, "train": tot / len(Xt), "val_max_rel": val,
                         "val_per_comp": m["per_comp"], "val_all": val_all})
            if val < best[0]:
                best = (val, {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}, epoch)
                bad = 0
            else:
                bad += 1
            if epoch % 25 == 0:
                print(f"ep {epoch:4d} train {tot/len(Xt):.4e} val_max_rel {val:.4e} best {best[0]:.4e}", flush=True)
            if bad * 5 >= args.patience:
                print(f"early stop at {epoch}", flush=True)
                break

    # final evaluation with best weights on val + test
    sd_best = best[1]
    state = {k: torch.tensor(v, device=dev) for k, v in sd_best.items()}
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        ypred = net(torch.tensor(Xs, dtype=torch.float32, device=dev)).cpu().numpy() * (_ob if _ob > 0 else 1.0) * yscale * tscale[:, None]

    def full(sel):
        m = metrics(ypred[sel], E_point[sel], E_conv[sel], cB_conv[sel], u[sel],
                    gamma[sel], nrm[sel], r[sel])
        out = {"max_rel": m["max_rel"],
               "per_comp": dict(zip(["Ex", "Ey", "Ez", "cBx", "cBy", "cBz"], m["per_comp"])),
               "pooled": dict(zip(["Ex", "Ey", "Ez", "cBx", "cBy", "cBz"], m["pooled"]))}
        # stratified by gamma family
        strat = {}
        for gname, gval in [("gamma1", 1.0), ("gamma1.001", 1.001), ("gamma2", 2.0),
                            ("gamma10", 10.0), ("gamma100", 100.0)]:
            s = sel & np.isclose(gamma, gval)
            if s.sum():
                strat[gname] = float(np.max(m["rel"][s[sel]]))
        out["by_gamma"] = strat
        # by distance bucket
        rr_ = r[sel]
        rel_ = m["rel"]
        buckets = [("deep", 0.02, 0.5), ("inside", 0.5, 1.45), ("face", 1.45, 1.55),
                   ("near", 1.55, 2.5), ("mid", 2.5, 4.0), ("far", 4.0, 6.01)]
        bstrat = {}
        for nm, lo, hi in buckets:
            s = (rr_ >= lo) & (rr_ < hi)
            if s.sum():
                bstrat[nm] = float(np.max(rel_[s]))
        out["by_distance"] = bstrat
        rsel = r[sel]
        rel_all = m["rel"]
        keep = rsel >= 0.45
        out["max_rel_r045"] = float(np.max(rel_all[keep])) if keep.any() else None
        out["n"] = int(sel.sum())
        return out

    result = {"tag": args.tag, "args": vars(args), "best_epoch": best[2],
              "val": full(va), "test": full(te), "hist": hist}
    print(json.dumps({k: result[k] for k in ("tag", "best_epoch", "val", "test")}, indent=2), flush=True)

    # save weights (plain npz) + meta json (numpy-inference metadata)
    serial = {k: v.astype(np.float32) for k, v in sd_best.items()}
    np.savez(os.path.join(args.out, f"{args.tag}.npz"), **serial)
    meta = {"tag": args.tag, "feature_set": args.feature_set,
            "target_scale": args.target_scale, "out_bound": args.out_bound,
            "width": args.width, "depth": args.depth, "drop": args.drop,
            "nin": int(X.shape[1]), "mu": mu.tolist(), "sd": sd.tolist(),
            "yscale": yscale.tolist(),
            "weights": f"{args.tag}.npz",
            "cB_relation": "cB = beta x E, beta = u/gamma",
            "target": "delta_E_hat = (E_conv - E_point) * |R_phys|^2/dx^2"}
    with open(os.path.join(args.out, f"{args.tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.out, f"{args.tag}_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out, f"{args.tag}_pred_test.npz"), "wb") as f:
        np.savez(f, ypred=ypred[te], idx=np.where(te)[0])


if __name__ == "__main__":
    main()
