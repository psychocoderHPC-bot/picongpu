# T110 — neural surrogate for the near-field correction (Route B, pilot)

Run pr24-20261102. Execution on `rosi_v100` via compute MCP only; the Terok
container handled file/git operations. Nothing was pushed to the PR branch.

## 1. Pilot result (final model c10, held-out pilot, r_cells >= 0.45)

The judge-relevant radius is `|R_cells|` (the corpus labels 0.5-6 cells), so all
pilot metrics below use `r_cells`, not the physical `|R_phys|` (anisotropic cells
make those differ). Held-out test (1240 rows), val (1337 rows) in brackets.

| component | worst max_rel (test) | pooled L2 (test) | S0-floored (test) |
|---|---|---|---|
| Ex | 11.95 [8.83] | 1.11 | 391 [548] |
| Ey | 7.56 [9.23] | 0.93 | 96 [105] |
| Ez | 23.47 [17.99] | 1.54 | 346 [291] |
| cBx | 3.09 [9.00] | 0.82 | 315 [354] |
| cBy | 23.35 [13.78] | 1.46 | 808 [822] |
| cBz | 7.52 [3.14] | 1.09 | 493 [468] |

By gamma (test / val): gamma1 7.56/3.65; gamma2 4.27/5.10; gamma10 23.47/17.99;
gamma100 7.52/5.06. By distance (test): inside 7.84, face 7.52, near 2.69,
mid 1.08, far 0.39. The model does **not** collapse at high gamma on the pilot,
but every bucket is 3-4 orders above 1e-3.

Judge corpus (1080 held-out probes): `gate_max_rel = 2.04e3`,
`gate_s0 = 1.11e3`, pooled L2 (E, cB) = (362, 392). Worst configs:
gamma100 axial ay0.25 = 2.04e3, gamma100 axial iso = 146, gamma10 axial ay0.25 =
181; gamma100 oblique = 0.40-2.66; stationary iso = 0.33. The failure is one
corner: high-gamma, axial motion, low `a_y`, where `delta_hat` reaches ~-100
and the network predicts a few units.

## 2. Device-path inconsistency

Training used the independent reference only. At the T103 defect point (corpus
730, gamma100_axial, `a_y=0.25`, `R_cells=(0,1,0)`), reference `Ey=2.4499`,
current device `Ey=0.4335`. The surrogate returns `Ey=1359`: it does **not**
reproduce the device defect (it follows the reference's near-cancellation), but
it is equally wrong there. The defect did not contaminate training; this is the
expected high-gamma device/reference inconsistency.

## 3. Data scaling

8 architecture/config trials were run (c1-c10; c6 aborted, c9 is the half-data
control). Two findings:

- Capacity, not regularisation, moves the result: c8 (512x4) beat the 128-256
  wide nets; augmentation, target/gamma scaling, tanh bounds and Fourier
  features did not help.
- Data is **not** the bottleneck. 5400 rows (c9, 0.54x) vs 10000 rows (c8, 1x)
  gave worst hold-out 30.4 vs 46.1 (val) and 18.2 vs 24.2 (test): doubling the
  data did not improve. In-sample c5 is still p99 7.9 / median 0.055.

A 40000-row (4x) set was started but the generation process ended after ~6000
samples with no output shards; no 4x set exists. Given the flat 0.54x->1x
curve, the larger campaign is **not justified for this architecture**. Cost if
needed: the T103 pilot rate is 314 s/1e4 on 48 workers, i.e. ~52 min for 1e5 and
~1.75 h for 2e5, plus a few minutes of MLP training.

## 4. Inference cost

c10 = 798,723 parameters, 1.60 MFLOP/slot, 3.2 MB FP32 weights.
Measured: pure-numpy CPU (single thread) **32.8 us/slot**; batched V100 torch
forward (MLP only, batches 1e4-1e6) **0.15-0.19 us/slot**. Adding feature
construction and the analytic point field, a GPU-batched implementation is
projected at ~0.3-1 us/slot, so ~0.3-1 us/pair if a pair is one slot (6 Yee
slots would give ~2-6 us/pair). That clears the ~9 us/pair target by 1-2 orders
of magnitude on speed, but the accuracy gate fails, so the speed is not usable
yet. The plain-Python/numpy path at 32.8 us/slot does **not** beat the current
20 us/slot; the win requires the alpakaNN/CUDA batched path.

## 5. Artifacts (durable, experimental only)

`psychocoderHPC-bot/picongpu` branch `experimental/t110-nn-surrogate`,
directory `neuralBunch/t110-nn/`. Weights are plain `.npz` (no pickle) with a
JSON metadata sidecar; `model.py` is pure numpy inference (no torch/LibTorch).

SHA-256:
- `c10_rcells.npz` `d9d763a4418b8e01532d5a2cccc8518b7e8a36b76f4abe56cfd2ab0807131b45`
- `c10_rcells.json` `0dc0ee3e2cd8aae42c1a9e215c6e9cb6350b3e12756b5749efa7c56d82851421`
- `report_c10.json` `1674d5d07c4ec6ac16a5d268b8187996c2db8f6933ef2af3150eeed5adc66e01`
- `c10_pilot_metrics.json` `be65835d535108c117e8252d08db5b5baa1fbff73c88897ee2a04749a11f5622`

The c8 and c9 weights/results and the trainer are kept in the same branch for
reproduction. Nothing was pushed to `feature-neural-bunch-field-init`.

## Remaining work / escalation

The residual is representable, but the max-rel gate is controlled by rare
near-cancellation points (`delta_hat` up to ~100) that a smooth MLP on these 17
features cannot fit even in-sample. Next step is a physics-informed
representation (rest-frame/parallel-perpendicular quadrature features, or a
stencil/moment expansion of the point field), not more of the same data.
