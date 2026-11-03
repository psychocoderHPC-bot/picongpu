# T140 — single-source kernel with exact superposition

Run `pr24-20261102`. Training/evaluation on `rosi_v100` via compute MCP; the
Terok container only did file/git work. Nothing was pushed to the PR branch.

## 0. TL;DR

The pilot is **already per-source**, so no data were regenerated (10,000 rows:
7,097 train / 1,511 val / 1,392 test). Learning the single-source kernel `K`
with the verified order-96 symmetry canonicalisation reproduces the previous
best number exactly: held-out worst **0.68995**, p99 **0.12045**, median
**2.25e-3**. That is a **1.00x** change against the 0.69 baseline, not an
improvement. The reason is that the 0.69 model was already a per-source kernel
model: `nearfield_judge.dataset` evaluates one `indfield.integrate_one()` per
row and `E_conv` is that single source's field. The premise that the previous
surrogate learned an aggregated field does not hold for the pilot.

**Verdict: NO-GO.** Worst case is 69x above the 1e-2 gate. The measured
learning-curve slope is `worst ~ n^-0.50`; reaching 1e-2 from 0.69 would need
~5,000x more rows (~3.8e7 rows, ~14 days of the 48-core generator), far beyond
the ~10x budget.

## 1. Data format

`nearfield_judge/dataset.py` builds one sample from one `(R_cells, u, cells)`
and calls `reference.integrate_one(point)` once. Every shard row therefore
stores the reference field of **one TSC source**, not a sum over sources.
Arrays `E_conv`, `cB_conv`, `E_point`, `cB_point`, `delta_*` are all per-source.
No regeneration was needed.

Inputs reused from T116: `~/pr24/t112/pilot_all.npz` is the merged T103 pilot
(`sha256` dataset `e12225f7...a80f6`); split is by whole physical
configuration.

## 2. Model and training

* `K(R_cells, u, cells) -> E_conv` in the canonical frame of the exact
  order-96 group (signed axis permutations + `u -> -u`), reused from
  `experimental/t116-symmetry-nn` commit `0764517`.
* `cB = beta x E` is applied analytically after the kernel, so the cB
  components add exactly like E.
* Superposition is exact by construction: charges are applied outside the
  network, `E(x) = sum_i q_i K(x_i)`. The network never sees N, density, tile
  membership or chunk boundaries.
* 512x4 SiLU MLP, 801,283 parameters, AdamW, cosine LR, 1500 epochs, batch
  512, relative L2 loss, checkpoint on validation p99 (same protocol as T116).
  One run takes 38 s on one V100.

The T103 pilot rows are single-source, so "learn the kernel" and the T116
`hyper` model are the same experiment. That is the central finding.

## 3. Held-out results (test split, `|err|/(1+|ref|)` over E and cB)

Main run (`full_f1`, raw point-field feature, exact T116 feature layout):

| quantity | worst | p99 | median |
|---|---|---|---|
| all components | **6.8995e-1** | **1.2045e-1** | 2.2521e-3 |
| pooled E / cB | 1.2509e-1 / 1.3850e-1 | | |

Per component:

| comp | worst | p99 | median |
|---|---|---|---|
| Ex | 3.538e-1 | 1.388e-1 | 3.531e-3 |
| Ey | 6.900e-1 | 1.203e-1 | 3.558e-3 |
| Ez | 4.275e-1 | 1.199e-1 | 3.502e-3 |
| cBx | 4.048e-1 | 1.104e-1 | 5.206e-4 |
| cBy | 4.262e-1 | 1.378e-1 | 3.340e-4 |
| cBz | 2.778e-1 | 1.055e-1 | 5.603e-4 |

By gamma (worst / p99 / median):

| gamma | n | worst | p99 | median |
|---|---|---|---|---|
| 1 | 249 | 1.164e-1 | 2.422e-2 | 4.97e-6 |
| 1.001 | 264 | 2.021e-1 | 4.019e-2 | 6.47e-4 |
| 2 | 287 | 3.723e-1 | 7.533e-2 | 2.76e-3 |
| 10 | 290 | 4.275e-1 | 1.235e-1 | 3.88e-3 |
| 100 | 302 | 6.900e-1 | 2.015e-1 | 4.25e-3 |

By distance bucket (worst / p99 / median):

| bucket | n | worst | p99 | median |
|---|---|---|---|---|
| deep | 206 | 6.900e-1 | 1.863e-1 | 3.32e-3 |
| inside | 237 | 5.661e-1 | 1.455e-1 | 3.74e-3 |
| face | 283 | 2.660e-1 | 9.849e-2 | 2.28e-3 |
| near | 219 | 2.656e-1 | 8.732e-2 | 2.40e-3 |
| mid | 243 | 3.538e-1 | 1.007e-1 | 1.67e-3 |
| far | 204 | 2.015e-1 | 6.039e-2 | 1.56e-3 |

The worst test point is `gamma100|diag_ppp|0|0|deep`, `r=0.235`,
`cells=[1,0.119,0.122]`: `Ec=[3.479,0.298,-3.784]`, model
`[3.547,-0.825,-3.202]`. The dominant components are 1-2% off; the tail is a
small transverse component (Ey) predicted 3.8x wrong. The same point is the
worst for the T116 model, so this is not a new failure mode.

## 4. Fixed judge corpus (1,080 probes, independent reference)

This corpus probes exact axis alignments and extreme aspect ratios that the
pilot rarely samples. The raw-feature kernel fails much harder there:

* `gate_max_rel = 7.2262`, `gate_s0 = 43.92`, pooled E / cB =
  1.274 / 1.376.
* Worst: `gamma100_axial|ay0.25`, `inside_0.50`, `+y`, `cells=[1,0.25,1]`.
  Reference `Ey=1.855`; model `Ey=22.49`. The point field at that probe is
  `Ep_y=6400`, well above the pilot's maximum `|Ep|=978`. The network must
  cancel an out-of-distribution `1/r^2` input feature and does not.

A diagnostic variant feeds the bounded product `E_point*|R_phys|^2/dx^2`
(`ep_mode=norm`) instead of raw `E_point`. It fixes the corpus extrapolation
but not the pilot tail:

| variant | pilot worst | pilot p99 | judge gate_max_rel |
|---|---|---|---|
| raw `E_point` (T116 layout) | 6.900e-1 | 1.204e-1 | 7.226 |
| bounded `E_point*norm` | 8.650e-1 | 1.217e-1 | **0.798** |
| raw + max-component loss + 3x tail weight | 7.573e-1 | 1.432e-1 | 6.882 |

These variants confirm the tradeoff: the bounded feature removes the 1e4
dynamic range (judge 9.1x better) but does not move the pilot worst case, and
the tail-weighted loss makes p99 worse.

## 5. Superposition and aggregation

The model is exactly additive by construction: `q_i` enters after `K`, and
`K` never sees N or the other sources. Charge scaling and additivity are
therefore exact up to FP32 accumulation.

The claimed *accuracy* benefit does not follow. Over held-out rows, groups of
`m` sources with positive lognormal charges give (median / p99 / worst of the
aggregate component-relative error):

| m | median | p99 | worst |
|---|---|---|---|
| 2 | 5.53e-3 | 1.61e-2 | 1.251 |
| 8 | 2.24e-2 | 5.53e-2 | 1.546 |
| 32 | 5.39e-2 | 1.62e-1 | 1.772 |
| 128 | 8.40e-2 | 3.31e-1 | 2.643 |

Per-source errors add, while the reference vector sum partially cancels, so the
aggregate relative error grows roughly like `sqrt(m)`. "Errors cannot compound"
is false for the component-relative metric even though the model is exactly
linear. The aggregate test is synthetic (held-out per-source references,
heterogeneous configurations), so treat it as a linearity diagnostic, not a
production bunch simulation.

## 6. Learning curve and GO/NO-GO

Same protocol, training rows subsampled from the train split only:

| train rows | test worst | test p99 |
|---|---|---|
| 1,774 | 1.370 | 0.2653 |
| 3,548 | 0.900 | 0.1739 |
| 7,097 | 0.690 | 0.1204 |

Log-log slope: worst `~ n^-0.50`, p99 `~ n^-0.57`. At that slope, 10x more
rows (71k) gives worst `0.69 * 10^-0.50 = 0.221`, still 22x above the gate.
Reaching 1e-2 needs ~5,000x rows (3.8e7), i.e. ~14 days on the 48-core
generator (314 s per 1e4 rows). **NO-GO.**

## 7. Inference cost (V100, batched FP32, pure forward)

| batch | us/slot | 2e6 slots |
|---|---|---|
| 1e3 | 0.226 | 0.453 s |
| 1e4 | 0.191 | 0.381 s |
| 1e5 | 0.159 | 0.317 s |
| 1e6 | **0.151** | **0.302 s** |

NumPy canonicalisation + feature construction: 0.864-0.891 us/slot in a single
process, so end-to-end is ~1.02 us/slot and 2M slots ~2.04 s.

Per source: 1,536 slots per source-tile x 12-30 near tiles = 18,432-46,080
slot evaluations. Pure forward is 2.8-7.0 ms per source; end-to-end is
18.8-47.0 ms per source. Inference is irrelevant next to the accuracy gate.

## 8. Artefacts

Branch `experimental/t140-kernel-nn`, directory `neuralBunch/t140-kernel-nn/`:
`train_t140.py`, `RESULT.md`, `artifacts/full_f1.json`,
`artifacts/epnorm_f1.json`, `artifacts/maxloss_raw.json`,
`artifacts/frac025.json`, `artifacts/frac050.json`,
`artifacts/full_f1.npz` (weights + held-out predictions).

Inputs: pilot `pilot_all.npz` (T103 dataset hash `e12225f7...a80f6`),
reference `indfield.py` sha256 `a1355980...99519`.
