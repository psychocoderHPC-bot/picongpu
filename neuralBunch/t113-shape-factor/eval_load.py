"""Evaluate a saved T113 run on subsets; also benchmark batched V100 forward."""
import argparse, json, os, time
import numpy as np
import torch
import train6 as T

d = dict(np.load(os.path.expanduser("~/pr24/t112/pilot_all.npz"), allow_pickle=False))
G = T.geom(d)
X = T.feats(d, "phys")
split = np.asarray(d["split"])

def load(tag):
    z = np.load(os.path.expanduser(f"~/pr24/t113/runs/{tag}.npz"), allow_pickle=False)
    mode = str(z["mode"][0]); feats = str(z["features"][0])
    mu, sd = z["feat_mu"], z["feat_sd"]
    sc_mu = z["scaler_mu"]; sc_sd = z["scaler_sd"]
    return z, mode, feats, mu, sd, sc_mu, sc_sd

def model_from(z, mode, nin):
    layers = []; i = 0
    while f"w{i}" in z.files:
        layers.append((z[f"w{i}"], z[f"w{i+1}"]))
        i += 2
    return layers, layers[0][0].shape[0]

def forward(layers, Xn, mode):
    h = Xn
    for j, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if j < len(layers) - 1:
            h = h * (1.0 / (1.0 + np.exp(-h)))
    return h

def predict(tag, sel):
    z, mode, feats, mu, sd, sc_mu, sc_sd = load(tag)
    X = T.feats(d, feats)
    nin = X.shape[1]
    layers, width = model_from(z, mode, nin)
    y = forward(layers, ((X[sel] - mu) / sd).astype(np.float32), mode).astype(np.float64)
    Gs = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in G.items()}
    sm = None if sc_mu.size == 0 else sc_mu
    ss = None if sc_sd.size == 0 else sc_sd
    return T.reconstruct(y, mode, Gs, ss, sm), mode, nin, sum(w.size for w, b in layers) + sum(b.size for w, b in layers)

for tag in ["e1_conv", "e3_shapelog", "e4_signlog", "f_full"]:
    for name, sel in (("test_all", split == "test"),
                      ("test_r45", (split == "test") & (np.asarray(d["r_cells"]) >= 0.45)),
                      ("train_all", split == "train"),
                      ("train_r45", (split == "train") & (np.asarray(d["r_cells"]) >= 0.45))):
        E, mode, nin, npar = predict(tag, sel)
        dd = T.dictsel(d, sel)
        m = T.metrics(E, dd)
        print(f"{tag} {name}: mode={mode} n={int(sel.sum())} worst={m['worst']:.3e} p99={m['p99']:.3e} "
              f"med={m['median']:.3e} pooledE={m['pooled_E']:.3e} pooledcB={m['pooled_cB']:.3e}", flush=True)

# ---- inference cost: batched forward of f_full MLP only, V100 ----
z, mode, feats, mu, sd, sc_mu, sc_sd = load("f_full")
nin = T.feats(d, feats).shape[1]
layers, width = model_from(z, mode, nin)
dev = torch.device("cuda:0")
torch_layers = []
for w, b in layers:
    torch_layers.append((torch.from_numpy(w).to(dev), torch.from_numpy(b).to(dev)))
for n in (10_000, 100_000, 1_000_000):
    x = torch.randn(n, nin, device=dev)
    for _ in range(3):
        h = x
        for j, (w, b) in enumerate(torch_layers):
            h = h @ w.T + b
            if j < len(torch_layers) - 1:
                h = torch.nn.functional.silu(h)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10):
        h = x
        for j, (w, b) in enumerate(torch_layers):
            h = h @ w.T + b
            if j < len(torch_layers) - 1:
                h = torch.nn.functional.silu(h)
    torch.cuda.synchronize(); dt = (time.time() - t0) / 10
    print(f"BENCH FWD n={n} {dt*1e6/n:.3f} us/slot (batch {dt*1e3:.2f} ms)", flush=True)
# fp16
torch_layers16 = [(w.half(), b.half()) for w, b in torch_layers]
for n in (100_000, 1_000_000):
    x = torch.randn(n, nin, device=dev, dtype=torch.half)
    for _ in range(3):
        h = x
        for j, (w, b) in enumerate(torch_layers16):
            h = h @ w.T + b
            if j < len(torch_layers16) - 1:
                h = torch.nn.functional.silu(h)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10):
        h = x
        for j, (w, b) in enumerate(torch_layers16):
            h = h @ w.T + b
            if j < len(torch_layers16) - 1:
                h = torch.nn.functional.silu(h)
    torch.cuda.synchronize(); dt = (time.time() - t0) / 10
    print(f"BENCH FWD fp16 n={n} {dt*1e6/n:.3f} us/slot", flush=True)
