"""T116: inference cost of the final hyper-canonical model (V100, batched)."""
import os

import numpy as np
import torch
import torch.nn as nn

NPZ = os.path.expanduser("~/pr24/t116/runs/t116_hyper.npz")


def mlp(nin, width, depth, out=3):
    layers, n = [], nin
    for _ in range(depth):
        layers += [nn.Linear(n, width), nn.SiLU()]
        n = width
    layers.append(nn.Linear(n, out))
    return nn.Sequential(*layers)


def canon_np(cells, u, Rc):
    au = np.abs(u); aR = np.abs(Rc)
    p = np.lexsort((aR, au, cells), axis=1)
    Rcp = np.take_along_axis(Rc, p, 1)
    up = np.take_along_axis(u, p, 1)
    cellsn = np.take_along_axis(cells, p, 1)
    s = np.sign(Rcp); s[s == 0] = 1
    Rcn = s * Rcp; un = s * up
    big = np.abs(un) > 1e-14
    first = np.argmax(big, 1); has = big.any(1)
    t = np.ones(len(un)); t[has] = np.sign(un[np.arange(len(un)), first])[has]
    t[t == 0] = 1
    return cellsn, un * t[:, None], Rcn, p, s


def bench(n, dtype):
    z = np.load(NPZ, allow_pickle=False)
    nin = z["feat_mu"].shape[0]
    model = mlp(nin, 512, 4)
    sd = [z[f"w{i}"] for i in range(len(model.state_dict()))]
    model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict(), sd)})
    model = model.cuda().to(dtype).eval()
    x = torch.randn(n, nin, device="cuda", dtype=dtype)
    for _ in range(3):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    import time
    ts = []
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    dt = float(np.median(ts))
    return dt / n * 1e6, n, int(sum(p.numel() for p in model.parameters()))


def canon_cost(n=1_000_000):
    import time
    rng = np.random.default_rng(0)
    Rc = rng.normal(size=(n, 3)); u = rng.normal(size=(n, 3)) * 10
    cells = np.exp(rng.uniform(-2, 2, size=(n, 3)))
    canon_np(cells[:1000], u[:1000], Rc[:1000])
    t0 = time.perf_counter(); canon_np(cells, u, Rc); dt = time.perf_counter() - t0
    return dt / n * 1e6


if __name__ == "__main__":
    for n in (10_000, 100_000, 1_000_000):
        f32, nn_, par = bench(n, torch.float32)
        f16, _, _ = bench(n, torch.float16)
        print(f"n={n} params={par} fp32={f32:.4f} us/slot fp16={f16:.4f} us/slot", flush=True)
    print(f"cpu canonicalisation (numpy, 3 axis sort + sign): {canon_cost():.4f} us/slot",
          flush=True)
