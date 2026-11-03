# T119 — can a neural surrogate reach the near-field gate?

Run `pr24-20261102`. Generation, training and evaluation on `rosi_v100` via
compute MCP. The Terok container only did file and git work.

## 0. TL;DR

**NO-GO.** The best strategy (dense rows per configuration) reaches a held-out
worst case of **0.848**, still **~2.8e3x above the 3e-4 gate**. Dense rows are a
real lever for the tail: going from ~2 to 150 rows per config (same 210 configs)
improved the common-test worst case 17x and p99 1.7x, but the measured slope
(~3.75x worst per data doubling) still needs **~64x** more rows to reach 3e-4,
far outside the <10x affordability bound. Per-gamma specialists and the
hard-constrained log-g/direction output are both worse than the T116 single net.
Inference is not the obstacle (0.26 us/slot FP32 for the direct model).

## 1. Setup

- Reused T116 code (`experimental/t116-symmetry-nn` @ `0764517`) and the pilot
  `~/pr24/t112/pilot_all.npz` (5400 configs, ~2 rows/config, sha256
  `6a8cc50d...d89b`).
- Metric unchanged from T113/T116: component-wise `|err|/(1+|ref|)` over E and
  cB = beta x E. "Worst" is the gate.
- New dense corpus: 210 train / 45 val / 45 held-out configs (config split kept
  from the pilot), **150 targets per config** inside each config's distance
  shell. Train/val/test rows: 31500 / 6750 / 6750. Test configs are pilot test
  configs, unseen by every model.
- Dense test truth uses the same `indfield` policy at refine 2 (the pilot's
  setting). Its anisotropic inside floor (~3e-4) is irrelevant here: all
  reported errors are 3-5 orders above it.
- Experiments: 5 (specialists, log-g, sparse baseline, dense same-volume,
  dense full volume). Under the cap of 6.

## 2. Strategy 1 — per-gamma specialists (pilot, canonical hyper frame)

Separate 512x4 nets per gamma bin, same canonical frame and metric. Four
non-empty bins (the 3-10 bin has no pilot data).

| bin | train rows | in-sample p99 | held-out p99 | held-out worst |
|---|---|---|---|---|
| 1 - 1.5 | 2851 | 0.0070 | 0.0181 | 0.104 |
| 1.5 - 3 | 1423 | 0.0226 | 0.0966 | 0.441 |
| 3 - 10  | 0 | - | - | - |
| 10 - 30 | 1419 | 0.0529 | 0.2434 | 0.730 |
| 30 - 100| 1404 | 0.0253 | 0.3064 | 0.557 |
| **pooled** | 7097 | - | **0.1777** | **0.730** |

Versus the single T116 `hyper` net (p99 0.1204, worst 0.690), specialists help
only the easiest bin (g1-1.5 p99 0.018 vs 0.024) and are worse everywhere else.
Each specialist sees ~1/5 of the rows, so this is data starvation, not
representation. **Dead end.**

Inference: 801k params, ~0.26 us/slot FP32 (same architecture as the direct
model).

## 3. Strategy 3 — hard-constrained log-g + direction (pilot)

Output is `(log g, unit direction)` with `g = |E_conv|/|E_point|`, reconstructed
as `E_conv = exp(log g) |E_point| Ehat`, direction expressed in the point-field
frame. The far-field asymptote (`g -> 1`, `Ehat -> Ehat_point`) is encoded
structurally.

| split | p99 | worst |
|---|---|---|
| in-sample | 0.0676 | 0.799 |
| held-out | 0.3020 | 1.946 |

Held-out by gamma (p99 / worst): 1: 0.099/0.473, 1.001: 0.141/1.278,
2: 0.186/0.779, 10: 0.541/1.468, 100: 0.358/1.946.

Measured on the pilot, `|E_conv|/|E_point|` is not bounded near 1: it spans
3.5e-4 to 3.1e4, and the E_conv direction is nearly perpendicular to E_point
for many points (median perpendicular component 0.18-0.34, p99 ~0.94-1.0). So
the assumed asymptote is only a far-field statement, not a good global
conditioning. Worse than direct `E_conv` in every split. **Dead end.**

Inference: 1.23 us/slot FP32 (includes the reconstruction).

## 4. Strategy 2 — dense rows per config (the closest)

Common held-out set: the 45 dense test configs x 150 rows (6750 rows).

| model | train rows | rows/config | in-sample p99 | held-out p99 | held-out worst |
|---|---|---|---|---|---|
| sparse pilot (T116 reproduction) | 7097 | ~2 | 0.0169 | 0.1108 | 12.661 |
| dense same volume | 7140 | 34 | 0.0067 | 0.0975 | 14.426 |
| dense full | 31500 | 150 | 0.0294 | **0.0572** | **0.848** |

Held-out by gamma (p99 / worst):

| gamma | sparse | same volume | dense full |
|---|---|---|---|
| 1     | 0.026 / 1.47 | 0.024 / 1.34 | 0.021 / 0.179 |
| 1.001 | 0.032 / 0.265| 0.041 / 0.717| 0.019 / 0.193 |
| 2     | 0.085 / 0.467| 0.091 / 0.834| 0.043 / 0.154 |
| 10    | 0.182 / 2.39 | 0.154 / 3.00 | 0.094 / 0.848 |
| 100   | 0.158 / 12.7 | 0.123 / 14.4 | 0.075 / 0.737 |

Interpretation:

- **Data shape alone does almost nothing**: at matched volume (7097 vs 7140
  rows) 34 rows/config buys 1.14x on p99 and no worst-case gain (12.7 -> 14.4).
- **Rows per config at higher volume is the real lever for the tail**: same 210
  configs, 34 -> 150 rows (4.4x) gives p99 1.70x and worst 17x. The sparse
  model's 12.7 worst is a hard point in an unseen config that it never resolved;
  the dense model resolves it to 0.85.
- Reproduction check: the sparse baseline on its own pilot test gives p99
  0.1236 / worst 0.679, matching T116 (0.1204 / 0.690).

## 5. GO / NO-GO

**NO-GO.**

- Gate: held-out worst <= 3e-4 (ideally 3e-4 with margin). Best measured:
  **0.848**, i.e. **2827x** over the gate. Even comparing against the best prior
  pilot-test number (0.679) the factor is 2263x.
- Data scaling from the dense experiment: 4.4x rows -> 17x worst, i.e. ~3.75x
  per doubling. Reaching 3e-4 from 0.848 needs ~6 doublings, **~64x more rows**
  (~2e6 rows at 150/config), which is well outside the <10x affordability bound.
  Under a 10x budget the extrapolated worst is ~7e-3, still ~25x above the gate.
  p99 extrapolates to ~0.025, also far from 3e-4.
- Strategy 1 and strategy 3 both regress relative to the T116 single net and
  are not candidates.
- The slope is estimated from two volume points and the worst case is a tail
  statistic over 45 configs; treat ~64x as an order-of-magnitude statement. It
  is far enough above 10x that the verdict does not depend on the exact slope.

Cost if it had gone GO: 801k-param MLP, 0.26 us/slot FP32 at batch 1e5
(0.15 us pure forward in T116), below the analytic kernel's ~1 us/slot estimate.
Dense generation cost measured on 48 cores: 31500 train rows in 1036 s, 6750 val
rows in 245 s, 6750 test rows in 196 s (~25 min total); 1200-epoch training is
~4 min per model. Inference is affordable; accuracy is not.

## 6. Artefacts

All under `neuralBunch/t119-nn-gate/` on branch `experimental/t119-nn-gate`
(experimental only, never the PR branch).

```
t119_train.py          f063fdca203a4dfc8c7c9a1c94ff5b43bae20be64c43ad95aa59151044c57f7f
gen_dense.py           480a73525ee0957a0c42e4b48f36d3d3cfbeaeb0abb83ce32763949eee4db339
t119_specialists_pilot.json  16d98b2f0011542c6ce980dd18758515c7fdce379cd387a4a9be0f5ae9e79632
t119_logg_pilot.json         996d28050b3d14e97a809e3778458d4f1c50b59884dd4fdd7ada954cb462d20c
t119_sparse_pilot.json       2406a4d79e83c74d389b1cb20aa150b7640310896aed83b376e710c8034b3a50
t119_dense_samevol.json      c0a6006fe2bb114ba9744cbd36ae8fa627cf99c01779d2133fbc3cd817efd5a6
t119_dense_full.json         6dbd505e2703c597cf888fd5deb7a77efce26af6eba7822a749d95b11762deb7
dense_meta.json              7211fdb993c349211208be149e504e5784d04d48106ed2f32f543d4f82795be8
```

Dense corpus SHA-256: train `772e3176...749cfe`, val `085cca6c...6ec436a`,
test `b30c38f9...f3e07ec`. Pilot `6a8cc50d...d89b`. Nothing was pushed to the
PR branch. The `rosi` trees are not durable; this branch is the record.
