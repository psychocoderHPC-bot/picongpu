"""T140: learn the single-source near-field KERNEL with exact symmetry.

Hypothesis: learn K(R_cells, u, cells) = (E_conv, cB_conv) per unit charge for
ONE TSC source, enforce the verified order-96 signed-axis-permutation symmetry
by canonicalisation, and make superposition exact by construction
(E = sum_i q_i K_i).  Evaluate held-out worst case against the independent
reference and compare with the previous 0.69 baseline.

The pilot rows ARE per source: each row is one (R_cells, u, cells) and
E_conv/cB_conv is the independent reference for that single source.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

T116 = os.path.expanduser("~/pr24/t140/neuralBunch/t116-symmetry/train_t116.py")
DATA = os.path.expanduser("~/pr24/t112/pilot_all.npz")
T103 = os.path.expanduser("~/pr24/t103")
CORPUS = os.path.join(T103, "data", "corpus.json")
REFNPZ = os.path.join(T103, "data", "reference.npz")
TINY = 1e-30
NAMES = ["Ex", "Ey", "Ez", "cBx", "cBy", "cBz"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t116 = _load(T116, "t116_mod")


def feats_hyper(Rc, u, cells, gamma, Ep, ep_mode="raw"):
    """Canonical order-96 features, identical layout to t116 mode 'hyper'.

    ep_mode='norm' feeds E_point * |R_phys|^2/dx^2 instead of the raw point
    field.  That product is bounded O(1..1e2) whereas the raw point field
    reaches 1e4 inside an anisotropic support, so the network no longer has to
    cancel an out-of-distribution 1/r^2 feature.
    """
    cellsn, un, Rcn, p, s, t = t116.canon(cells, u, Rc)
    Epn = s * np.take_along_axis(Ep, p, axis=1)
    if ep_mode == "norm":
        R_phys = Rc * cells
        norm = np.maximum(np.einsum("ij,ij->i", R_phys, R_phys) /
                          (cells[:, 0] * cells[:, 0]), 1e-12)
        Epn = Epn * norm[:, None]
    lg = np.log(gamma)
    bn = np.linalg.norm(un, axis=1)
    bh = un / np.maximum(bn, TINY)[:, None]
    r = np.linalg.norm(Rcn, axis=1)
    rp = np.einsum("ij,ij->i", Rcn, bh)
    rr = np.sqrt(np.maximum(r * r - rp * rp, 0.0))
    W = np.prod(np.clip(1.5 - np.abs(Rcn), 0.0, None), axis=1)
    df = 1.5 - np.max(np.abs(Rcn), axis=1)
    cols = [Rcn[:, 0], Rcn[:, 1], Rcn[:, 2],
            np.log(cellsn[:, 0]), np.log(cellsn[:, 1]), np.log(cellsn[:, 2]),
            lg, un[:, 0], un[:, 1], un[:, 2], bh[:, 0], bh[:, 1], bh[:, 2],
            r, rp, rr, W, df,
            Epn[:, 0], Epn[:, 1], Epn[:, 2],
            np.log(np.maximum(np.linalg.norm(Epn, axis=1), TINY))]
    X = np.stack(cols, axis=1).astype(np.float32)
    return X, p, s, Epn


def mlp(nin, width, depth, out=3):
    layers, n = [], nin
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.SiLU()]
        n = width
    layers.append(nn.Linear(n, out))
    return nn.Sequential(*layers)


def beta_of(u):
    return u / np.sqrt(1.0 + np.einsum("ij,ij->i", u, u))[:, None]


def rel_matrix(E, cB, Ec, Bc):
    return np.abs(np.concatenate([E - Ec, cB - Bc], axis=1)) / (
        1.0 + np.abs(np.concatenate([Ec, Bc], axis=1)))


def dist_bucket(r):
    # same edges as nearfield_judge.dataset.DISTANCES
    out = np.full(len(r), "far", dtype=object)
    for name, lo, hi in (("deep", 0, 0.5), ("inside", 0.5, 1.45),
                         ("face", 1.45, 1.55), ("near", 1.55, 2.5),
                         ("mid", 2.5, 4.0), ("far", 4.0, 6.01)):
        out[(r >= lo) & (r < hi)] = name
    return out


def summarize(rel, g, r):
    row = rel.max(axis=1)
    out = {
        "n": int(len(rel)), "worst": float(rel.max()),
        "p99": float(np.quantile(rel, 0.99)),
        "median": float(np.median(rel)),
        "per_comp": {k: {"worst": float(rel[:, i].max()),
                         "p99": float(np.quantile(rel[:, i], 0.99)),
                         "median": float(np.median(rel[:, i]))}
                     for i, k in enumerate(NAMES)},
        "by_gamma": {}, "by_distance": {},
    }
    for gg in (1.0, 1.001, 2.0, 10.0, 100.0):
        m = g == gg
        if m.any():
            out["by_gamma"][f"{gg:g}"] = {
                "n": int(m.sum()), "worst": float(rel[m].max()),
                "p99": float(np.quantile(rel[m], 0.99)),
                "median": float(np.median(rel[m]))}
    db = dist_bucket(r)
    for name in ("deep", "inside", "face", "near", "mid", "far"):
        m = db == name
        if m.any():
            out["by_distance"][name] = {
                "n": int(m.sum()), "worst": float(rel[m].max()),
                "p99": float(np.quantile(rel[m], 0.99)),
                "median": float(np.median(rel[m]))}
    return out


def eval_rows(E, cB, d, sel):
    Ec = np.asarray(d["E_conv"], float)[sel]
    Bc = np.asarray(d["cB_conv"], float)[sel]
    rel = rel_matrix(E[sel], cB[sel], Ec, Bc)
    g = np.asarray(d["gamma"], float)[sel]
    r = np.asarray(d["r_cells"], float)[sel]
    return summarize(rel, g, r)


def corpus_predict(model, muX, sdX, muY, sdY, points, dev, ep_mode="raw"):
    Rc = np.asarray([p["R_cells"] for p in points], float)
    u = np.asarray([p["u"] for p in points], float)
    cells = np.asarray([p["cells"] for p in points], float)
    gamma = np.asarray([p["gamma"] for p in points], float)
    # analytic point field for the feature (independent reference helper)
    import sys
    os.environ.setdefault("NEARFIELD_INDEPENDENT_REF", os.path.expanduser("~/pr24/pr24w53-ref"))
    if T103 not in sys.path:
        sys.path.insert(0, T103)
    from nearfield_judge import reference  # noqa: E402
    Ep = np.asarray([reference.point_field(p) for p in points], float)[:, 0, :]
    X, p, s, _ = feats_hyper(Rc, u, cells, gamma, Ep, ep_mode)
    Xn = (X - muX) / sdX
    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(Xn).to(dev)).cpu().numpy().astype(np.float64)
    Y = y * sdY + muY
    E = t116.inv_map(Y, p, s)
    cB = np.cross(beta_of(u), E)
    return E, cB


def aggregate_test(E, cB, d, sel, rng, sizes=(2, 8, 32, 128), reps=4000):
    """Superposition over held-out single-source references.

    For a group of m held-out sources with charges q, compare sum q_i K_i with
    sum q_i ref_i.  This is exactly the aggregation the kernel would serve.
    """
    Ec = np.asarray(d["E_conv"], float)[sel]
    Bc = np.asarray(d["cB_conv"], float)[sel]
    n = len(Ec)
    out = {}
    for m in sizes:
        reps_m = max(200, int(reps * 8 / m))
        worst = 0.0
        p99s, meds, poole = [], [], []
        for _ in range(reps_m):
            idx = rng.choice(n, size=m, replace=False)
            q = rng.lognormal(0.0, 0.5, size=m)          # positive bunch charges
            pe = (q[:, None] * E[idx]).sum(0)
            pb = (q[:, None] * cB[idx]).sum(0)
            re = (q[:, None] * Ec[idx]).sum(0)
            rb = (q[:, None] * Bc[idx]).sum(0)
            rel = np.abs(np.concatenate([pe - re, pb - rb])) / (
                1.0 + np.abs(np.concatenate([re, rb])))
            worst = max(worst, float(rel.max()))
            p99s.append(float(np.quantile(rel, 0.99)))
            meds.append(float(np.median(rel)))
            poole.append(float(np.linalg.norm(np.concatenate([pe - re, pb - rb])) /
                               np.linalg.norm(np.concatenate([re, rb]))))
        out[f"m{m}"] = {"groups": reps_m, "worst": worst,
                        "p99": float(np.median(p99s)),
                        "median": float(np.median(meds)),
                        "pooled_median": float(np.median(poole))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/pr24/t140/runs"))
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--ep-mode", default="raw", choices=["raw", "norm"])
    ap.add_argument("--loss", default="mean", choices=["mean", "max"])
    ap.add_argument("--tail-weight", type=float, default=0.0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    d = dict(np.load(DATA, allow_pickle=False))
    Rc = np.asarray(d["R_cells"], float)
    u = np.asarray(d["u"], float)
    cells = np.asarray(d["cells"], float)
    g = np.asarray(d["gamma"], float)
    Ep = np.asarray(d["E_point"], float)
    Ec = np.asarray(d["E_conv"], float)
    Bc = np.asarray(d["cB_conv"], float)
    X, p, s, Epn = feats_hyper(Rc, u, cells, g, Ep, a.ep_mode)
    Y = s * np.take_along_axis(Ec, p, axis=1)     # canonical kernel target
    split = np.asarray(d["split"])
    trf, va, te = split == "train", split == "val", split == "test"
    tr_idx_all = np.where(trf)[0]
    if a.frac < 1.0:
        rngf = np.random.default_rng(1234 + a.seed)
        k = max(1, int(round(a.frac * len(tr_idx_all))))
        tr_idx_all = np.sort(rngf.choice(tr_idx_all, size=k, replace=False))
    trf_sub = np.zeros(len(X), bool)
    trf_sub[tr_idx_all] = True

    muX = X[trf_sub].mean(0); sdX = X[trf_sub].std(0); sdX[sdX < 1e-12] = 1.0
    Xn = ((X - muX) / sdX).astype(np.float32)
    muY = Y[trf_sub].mean(0); sdY = Y[trf_sub].std(0); sdY[sdY < 1e-12] = 1.0
    Z = ((Y - muY) / sdY).astype(np.float32)

    Xt = torch.from_numpy(Xn).to(dev)
    Zt = torch.from_numpy(Z).to(dev)
    tr_idx = torch.from_numpy(tr_idx_all.astype(np.int64)).to(dev)
    muY_t = torch.from_numpy(muY.astype(np.float32)).to(dev)
    sdY_t = torch.from_numpy(sdY.astype(np.float32)).to(dev)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(dev)
    Ec_t = torch.from_numpy(Ec.astype(np.float32)).to(dev)
    Bc_t = torch.from_numpy(Bc.astype(np.float32)).to(dev)
    # optional tail weight: high gamma AND at least one very thin cell axis
    w_np = np.ones(len(X), np.float32)
    if a.tail_weight > 0:
        thin = np.min(cells[:, 1:], axis=1) < 0.2
        tg = g >= 10.0
        w_np[tg & thin] += a.tail_weight
    Wt = torch.from_numpy(w_np).to(dev)

    model = mlp(X.shape[1], a.width, a.depth, 3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def predict_all():
        model.eval()
        with torch.no_grad():
            y = model(Xt).cpu().numpy().astype(np.float64)
        Ylab = t116.inv_map(y * sdY + muY, p, s)
        return Ylab, np.cross(beta_of(u), Ylab)

    def test_rel(E, cB):
        return rel_matrix(E[te], cB[te], Ec[te], Bc[te])

    rng = np.random.default_rng(a.seed)
    best = {"val": 1e30, "epoch": -1, "state": None}
    hist = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        tot = 0.0
        for st in range(0, len(perm), a.batch):
            idx = perm[st:st + a.batch]
            out = model(Xt[idx])
            ref = Yt[idx]
            e = (out * sdY_t + muY_t - ref) / (1.0 + torch.abs(ref))
            if a.loss == "max":
                loss = torch.mean((e * e).max(dim=1).values * Wt[idx])
            else:
                loss = torch.mean((e * e).mean(dim=1) * Wt[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if ep % 25 == 0 or ep == a.epochs - 1:
            E, cB = predict_all()
            rv = rel_matrix(E[va], cB[va], Ec[va], Bc[va])
            rte = test_rel(E, cB)
            rec = {"epoch": ep, "loss": tot / len(tr_idx),
                   "val_p99": float(np.quantile(rv, 0.99)),
                   "val_worst": float(rv.max()),
                   "test_p99": float(np.quantile(rte, 0.99)),
                   "test_worst": float(rte.max())}
            hist.append(rec)
            if rec["val_p99"] < best["val"]:
                best = {"val": rec["val_p99"], "epoch": ep,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            if ep % 200 == 0:
                print(f"[{a.tag}] ep {ep} loss {rec['loss']:.3e} "
                      f"val_p99 {rec['val_p99']:.3e} val_worst {rec['val_worst']:.3e} "
                      f"test_worst {rec['test_worst']:.3e}", flush=True)
    train_seconds = time.time() - t0

    model.load_state_dict(best["state"])
    E, cB = predict_all()
    rep = {"tag": a.tag, "args": vars(a), "n_features": int(X.shape[1]),
           "n_params": int(sum(x.numel() for x in model.parameters())),
           "n_train": int(len(tr_idx_all)), "best_epoch": best["epoch"],
           "train_seconds": train_seconds, "hist": hist,
           "target": "canonical E_conv (single-source kernel), cB=beta x E",
           "test": eval_rows(E, cB, d, te),
           "val": eval_rows(E, cB, d, va),
           "train": eval_rows(E, cB, d, trf_sub)}

    # Official judge on the fixed corpus (same metric implementation).
    with open(CORPUS) as fh:
        corpus = json.load(fh)
    points = corpus["points"]
    Ez = np.load(REFNPZ)
    E_ref, cB_ref = Ez["E"], Ez["cB"]
    Ecp, cBp = corpus_predict(model, muX, sdX, muY, sdY, points, dev, a.ep_mode)
    import sys
    if T103 not in sys.path:
        sys.path.insert(0, T103)
    from nearfield_judge import metrics
    jr = metrics.evaluate(points, E_ref, cB_ref, Ecp, cBp)
    rep["judge_corpus"] = {
        "gate_max_rel": jr["gate_max_rel"], "gate_s0": jr["gate_s0"],
        "gate_s0_pooled": jr["gate_s0_pooled"],
        "pooled_l2_global": jr["pooled_l2_global"],
        "components": {k: {"max_rel": v["max_rel"],
                           "max_rel_s0": v["max_rel_s0"],
                           "pooled_l2": v["pooled_l2"],
                           "median_abs_err": float(np.median(np.abs(
                               (Ecp if i < 3 else cBp)[:, i % 3] -
                               (E_ref if i < 3 else cB_ref)[:, i % 3])))}
                       for i, (k, v) in enumerate(jr["components"].items())},
    }

    # Superposition test over held-out rows.
    agrng = np.random.default_rng(a.seed + 77)
    rep["aggregation"] = aggregate_test(E, cB, d, te, agrng)

    # Inference benchmark (batched FP32; pure forward).
    bench = {}
    if not a.no_bench and torch.cuda.is_available():
        nf = X.shape[1]
        model.eval()
        with torch.no_grad():
            for bs in (1000, 10000, 100000, 1000000):
                xb = torch.randn(bs, nf, device=dev)
                for _ in range(3):
                    model(xb)
                torch.cuda.synchronize()
                t = time.time()
                reps = max(1, min(20, int(2e6 / bs)))
                for _ in range(reps):
                    model(xb)
                torch.cuda.synchronize()
                us = (time.time() - t) / (reps * bs) * 1e6
                bench[f"fp32_b{bs}"] = {"us_per_slot": us,
                                        "projected_2e6_s": us * 2e6 / 1e6}
        # numpy canonicalisation cost (single process)
        idx = np.arange(min(200000, len(X)))
        t = time.time()
        feats_hyper(Rc[idx], u[idx], cells[idx], g[idx], Ep[idx])
        bench["canon_numpy_us_per_slot"] = (time.time() - t) / len(idx) * 1e6
    rep["bench"] = bench

    with open(os.path.join(a.out, f"{a.tag}.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    w = {f"w{i}": v.detach().cpu().numpy() for i, v in enumerate(model.state_dict().values())}
    np.savez(os.path.join(a.out, f"{a.tag}.npz"), **w, feat_mu=muX, feat_sd=sdX,
             scaler_mu=muY, scaler_sd=sdY, pred_lab=E, pred_cB=cB,
             split=split.astype("U5"))
    print(f"[{a.tag}] n_train={len(tr_idx_all)} best_ep={best['epoch']} "
          f"train_s={train_seconds:.1f}", flush=True)
    print(f"[{a.tag}] TEST worst {rep['test']['worst']:.4e} p99 {rep['test']['p99']:.4e} "
          f"median {rep['test']['median']:.4e}", flush=True)
    print(f"[{a.tag}] JUDGE gate_max_rel {rep['judge_corpus']['gate_max_rel']:.4e} "
          f"pooled {max(rep['judge_corpus']['pooled_l2_global'].values()):.4e}", flush=True)
    print(f"[{a.tag}] AGG {json.dumps(rep['aggregation'])}", flush=True)
    print(f"[{a.tag}] BENCH {json.dumps(bench)}", flush=True)
    print(f"[{a.tag}] GAMMA {json.dumps(rep['test']['by_gamma'])}", flush=True)
    print(f"[{a.tag}] DIST {json.dumps(rep['test']['by_distance'])}", flush=True)


if __name__ == "__main__":
    main()
