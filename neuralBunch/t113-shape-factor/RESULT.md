# T113 — bounded shape-factor target for the near-field surrogate

Run pr24-20261102. All training/evaluation on `rosi_v100` via compute MCP; the
Terok container only did file/git work. Nothing was pushed to the PR branch.

## 0. TL;DR

The prescribed shape factor (formulation 1) is bounded and its reconstruction is
exact, but it is the **worst** of the targets tested. Its inverse multiplies by
`D^-1.5`, so near the source it demands relative accuracy on a quantity that
scales as `D^1.5 ~ r^3`; the in-sample p99 stays at 0.16-0.24 instead of <1e-2.
Regressing `E_conv` itself — which is already bounded and smooth, because the TSC
convolution removes the `1/r^2` pole of the point field — is a factor ~60 better
in sample (p99 2.9e-3). The singular object was `delta`, not `E_conv`.

With the direct `E_conv` target the in-sample gate is met (worst 7.4e-3) but the
held-out gate is not (worst 0.69, p99 0.225). The learning curve (25/50/100% of
the rows) improves held-out p99 by only ~1.35x per doubling, so the remaining
limit is **data/generalisation**, not the target formulation — and closing it to
1e-2 would need ~1e3x more rows.

## 1. Point kernel read from the code (no invented normalisation)

`Features.hpp::pointFieldDx` (PR head, `pr24w107-prod`):

```
invG2 = 1/(1+|u|^2) = 1/gamma^2
bhat  = u/|u|
rPar  = R.bhat,  perp^2 = |R|^2 - rPar^2
D     = rPar^2 + perp^2*invG2          #  = |R|^2/gamma^2 + (R.beta)^2
denom = D^(3/2)
E     = invG2 * R / denom              # per unit charge; B = beta x E
```

This is algebraically identical to the independent reference
`indfield.py::point_field` (`D = |R|^2(1-beta^2) + (R.beta)^2`,
`E = (1-beta^2) R / D^1.5`), which is what the pilot stores as `E_point`.
The shape factor used here, therefore, is exactly

```
f_i = E_conv,i * D^(3/2) * gamma^2        (D, gamma from the code above)
E_recon = f * gamma^-2 * D^(-3/2)
```

## 2. Boundedness and the limit check

- `f` is bounded because `E_conv` is finite and `D^(3/2) -> 0` as `R -> 0`:
  `f -> 0` at the source, so the `1/r^2` pole of `delta = E_conv - E_point`
  is gone. Measured on the pilot: `|f|` in `[1.2e-8, 2.0e5]`.
- Exactness limit: substituting the far-field value `f -> R` into the
  reconstruction reproduces `E_point` to `6.3e-16` relative (analytic identity,
  checked on the held-out split). `E_conv -> E_point` as the support becomes
  negligible relative to `|R|`.
- Honest caveat: the limit is **not reached inside the sampled domain** for
  moving sources. Median `|f - R|/|R|` at `r>4` is 0.165 (gamma 1) but 0.48
  (gamma 10) and 2.09 (gamma 100): the TSC support is 1.5 cells while the
  pancake thickness is `~r/gamma`, so at gamma 100 the convolution still matters
  out to `r ~ 6` cells. The identity is exact; the "f=1 far away" intuition only
  holds for `r >> gamma * cell`.

## 3. Experiments (pilot 10k, whole-configuration train/val/test split)

Metric is the previous attempt's `|err|/(1+|ref|)` over E and cB (cB = beta x E,
verified 9e-13 in the pilot). "In-sample" = the 7097 train rows, final epoch;
"held-out" = the 1392 test rows.

| # | target | in-sample p99 | in-sample worst | test p99 | test worst |
|---|---|---|---|---|---|
| 1 | `delta` (T110/T112, reported before) | 7.9 | 7.6-23.5 | - | 7.6-23.5 |
| 2 | formulation 1 `f` (linear) | 3.4e3 | 3.8e5 | 2.7e3 | 3.6e5 |
| 3 | formulation 1 `arcsinh(f)` | 2.4e-1 | 8.8e-1 | 4.6e-1 | 5.2 |
| 4 | formulation 1 signed-log `sign(f)log1p|f|` | 1.6e-1 | 1.1 | 3.8e-1 | 1.4 |
| 5 | formulation 2 `|E_conv|/|E_point|` + direction | 3.3e-2 | 1.6e-1 | 3.2e-1 | 2.0 |
| 6 | **direct `E_conv`** | **2.9e-3** | **7.4e-3** | **2.3e-1** | **6.9e-1** |

All nets are 512x4 SiLU MLPs unless noted; the target loss is L2 on the
metric. Runs 3-4 show the failure mode clearly: the reconstruction factor
`gamma^-2 D^-1.5` blows error up where `D -> 0`, so even an in-sample fit cannot
drive the metric down. Restricting to the judge-relevant band `r_cells >= 0.45`
does not rescue it (shape-factor in-sample p99 2.4e-1, direct `E_conv` 2.9e-3).

### Direct `E_conv`, final model, per component

| component | in-sample worst | in-sample p99 | test worst | test p99 |
|---|---|---|---|---|
| Ex | 7.4e-3 | 3.2e-3 | 0.41 | 0.25 |
| Ey | 6.9e-3 | 3.4e-3 | 0.69 | 0.20 |
| Ez | 6.0e-3 | 3.4e-3 | 0.55 | 0.25 |
| cBx | 4.6e-3 | 1.9e-3 | 0.45 | 0.22 |
| cBy | 4.4e-3 | 1.9e-3 | 0.39 | 0.22 |
| cBz | 4.9e-3 | 1.8e-3 | 0.48 | 0.20 |

Held-out pooled L2: E 0.214, cB 0.218 (`r_cells>=0.45`: 0.209 / 0.211).

### Stratified by gamma (direct `E_conv`, final model)

| gamma | in-sample worst | in-sample p99 | test worst | test p99 |
|---|---|---|---|---|
| 1 | 6.7e-3 | 3.1e-3 | 0.135 | 2.4e-2 |
| 2 | 6.6e-3 | 2.2e-3 | 0.549 | 9.1e-2 |
| 10 | 4.5e-3 | 1.9e-3 | 0.446 | 1.7e-1 |
| 100 | 4.6e-3 | 1.5e-3 | 0.481 | 2.8e-1 |

The in-sample fit is flat across gamma, so the target itself is representable at
all gammas; the held-out degradation grows with gamma, i.e. it is an
interpolation problem in the harder high-gamma region.

## 4. Formulation limit or data limit?

**Data / generalisation, not the target formulation.** Evidence:

- In-sample (7097 rows) the direct `E_conv` target reaches p99 2.9e-3,
  worst 7.4e-3, i.e. below the 1e-2 bar. The shape-factor target does not
  (p99 0.16-0.24), which is a formulation/conditioning limit specific to the
  `D^-1.5` reconstruction.
- Held-out p99 falls monotonically with more rows (same net, same features):
  1774 rows 0.427, 3548 rows 0.305, 7097 rows 0.225. That is ~1.35x per
  doubling; extrapolating to p99 1e-2 needs ~10 doublings (~1e3x rows).
- A 37k-parameter 128x3 net gives test p99 0.290 (in-sample p99 2.2e-2); the
  512x4 (806k) net gives 0.225. Capacity is not the bottleneck.

Caveat: with 7097 rows and 806k parameters the in-sample number could partly be
memorisation, so "in-sample passes" is necessary but not sufficient evidence
that the function class is adequate. The learning-curve monotonicity is the
stronger of the two signals and it points at data.

## 5. Data-scaling decision

Do **not** commission the 1e5 run for the 1e-2 gate. At the measured 1.35x per
doubling, 1e5 rows (~52 min) buys ~0.08 p99, not 1e-2. The 1e-2 worst-case gate
would need ~1e3x the rows (~7e6, ~60 h of generator time) and even then the
worst-case tail may not follow the p99 slope. The useful next experiment is
denser sampling per physical configuration (more rows per config, same config
space), because the pilot has only ~2 rows per configuration and the angular
dependence is barely sampled; the current curve confounds "more rows" with
"more configurations".

## 6. Inference cost (measured on V100, batched)

512x4 MLP, 806,403 parameters, 1.6 MFLOP/slot. Pure forward, batch-averaged:

| batch | FP32 | FP16 |
|---|---|---|
| 1e4 | 0.193 us/slot | - |
| 1e5 | 0.168 us/slot | 0.049 us/slot |
| 1e6 | 0.159 us/slot | 0.049 us/slot |

Adding feature construction and the analytic point field puts a GPU-batched
implementation at roughly 0.3-0.6 us/slot FP32 (0.15-0.3 FP16), consistent with
T110. Inference is not the problem; the accuracy gate is.

## 7. Note on the previous attempt's numbers

T110/T112 selected the checkpoint on validation **worst-case**, and the worst
case is only ever beaten by the untrained initial model (epoch 0: val worst
0.93; after 1000 epochs: 1.3). Their reported "E_conv worst 0.87" was therefore
the random-initialisation model, not a trained one. This run selects on val p99
and reports the final epoch. The direct-`E_conv` numbers here are the first
honest measurement of that target.

## 8. Artifacts

Directory `neuralBunch/t113-shape-factor/` on the bot repo branch
`experimental/t113-shape-factor` (experimental only). SHA-256:

```
train6.py           75a44996a452976973b09cae04da9855cd403b7e2d81e74f92a3c08120bcdef7
eval_load.py        cee4c0c68241d13747acb648aa638ddec302e901ff64ebb473f7433954bb4530
f_full.json         a73d6c63ebbce8c79a7b6ef5add442dcd71a584df70f920be0a7872793c8ad50
f_full.npz          1cf88ba149e6064e7fa734110aa5970cb93758042c1c5f5431157155336c61da
f_half.json         09f04b5c6b56eef2f04a37d11898bcbdeb94671c850130a86047b627e21d341a
f_quarter.json      3252ccab257b9b23c477db8ddbb1e607601adcbce4e8bcfdee982d237ffedcb2
f_small.json        36bcb09826ddd0f2cef1142c15a2b6fffde2484cab8974fb3090607ae5989850
e3_shapelog.json    db8b25d96e9f748b92d23ebb7ab3795764c891179869097cb5c65161c911344f
e3_shapelog.npz     8a03d46dcc3adfd2776504f8e1ae8da230ebd82b7e4d58e505f2ede3d1adf9e5
e4_signlog.json     ac8fdf22fa29f1bdd010f192ea6da8ad162b92c2ab20273391fad9c52dbae445
e4_signlog.npz      e090133d1037845a1c19240e458d260eb8156f4eb5a419c892bd5a1062263910
e5_ratio.json       ad2cd978024b068e820aa2728a140271f7f535f49de08602c69a59e3948339fb
e5_ratio.npz        52ce5255e8cdf0411d05a5bb1932318dbf7aacc094f8d27e80702327328f2805
```

(NPZ weights are plain arrays plus `feat_mu/feat_sd/scaler_*/mode/features`; no
pickle. `eval_load.py` reconstructs the fields without torch.)
