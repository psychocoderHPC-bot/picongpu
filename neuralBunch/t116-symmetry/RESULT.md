# T116 — exact symmetries for the near-field surrogate

Run `pr24-20261102`. Training and evaluation on `rosi_v100` via compute MCP.
The Terok container only did file and git work. Nothing was pushed to the PR
branch; the artefacts are on `experimental/t116-symmetry-nn`.

## 0. TL;DR

The exact discretely acting symmetries are real and can be enforced
structurally, but they buy only **1.9x** on held-out p99
(0.2298 -> 0.1204) and **nothing** on the held-out worst case
(0.678 -> 0.690). The literal beam-aligned representation is **worse**
(p99 0.55, worst 4.5). The reason is measured: rotating the source velocity to
+z while keeping the axis-aligned grid is *not* a symmetry when
`a_y, a_z != 1`; the exact content of the hypothesis is the finite
grid-symmetry group (signed axis permutations plus `u -> -u`), and that group
gives the 1.9x.

The remaining gap is still data/interpolation, not representation: in sample
the same net fits to p99 0.027, and the held-out error is flat in azimuth
(0.108-0.146) and rises with gamma. A data campaign for the 1e-2 gate is **not
justified** by this route.

## 1. Identity verification against the independent reference

Reference `indfield.py` (`sha256 a1355980...99519`) through
`nearfield_judge.reference.integrate_one` (order 12, refine 2 inside, the
validated moving-pancake panels outside). Metric:
`max |a-b| / (1+|b|)`. Cases: gamma 2 oblique isotropic, gamma 10 oblique
anisotropic `[1,0.3,2.5]`, gamma 100 diagonal inside, gamma 100 oblique far,
stationary anisotropic.

| identity | measured max error | verdict |
|---|---|---|
| `E_conv(-R; u, cells) = -E_conv(R; u, cells)` | 2.0e-15 | exact |
| `cB_conv(-R; u, cells) = -cB_conv(R; u, cells)` | 9.2e-16 | exact |
| `E_conv(R; -u, cells) = E_conv(R; u, cells)` | 0.0 (bitwise) | exact |
| `cB_conv(R; -u, cells) = -cB_conv(R; u, cells)` | 0.0 (bitwise) | exact |
| `cB = beta x E` (reference) | 1.8e-14 | exact |
| `cB_conv = beta x E_conv` (pilot, 10000 rows) | 1.8e-13 | exact |
| axis sign flip `S=diag(+-1)`: `E(SR; S u, cells) = S E(R;u,cells)` | 2.0e-15 | exact |
| same for `cB` (axial: `-S cB`) | 9.1e-15 | exact |
| axis permutation `P` with `cells,u`: `E(PR; Pu, P cells) = P E` | 2.0e-14 | exact |
| same for `cB` (`det(P) P cB`) | 1.0e-14 | exact |
| rotation of `(R,u)` with the diagonal `cells` fixed | iso 1.2e-14; `[1,0.3,2.5]` 0.036-0.211 | **not a symmetry** |

So the full order-96 group (48 signed axis permutations times `u -> -u`) acts
exactly on `E_conv` as a polar vector; `cB` is derived and follows as an axial
vector. The pilot already contains 3240 opposite-direction config pairs out of
5400 keys, so part of the `u -> -u` symmetry was learnable from data.

The anisotropic rotation row is the important correction to the hypothesis:
treating "rotate u to +z" as exact drops the orientation of the anisotropic
grid. The exact beam-frame representation must carry the rotated cell basis;
that variant was implemented and measured (S4 below), and it is slower to
generalise, not faster.

## 2. Representations

* `hyper` — canonicalise by the order-96 group: sort axes by
  `(cells, |u|, |R_cells|)`, sign-flip each axis so `R_cells >= 0`, flip the
  global sign of `u`. Predict `E_conv` in that canonical frame and map back with
  the inverse signed permutation.
* `beam` — deterministic rotation with `Q bhat = z`, carrying the rotated cell
  basis `Q diag(cells)` (9 numbers), predict `E_conv` in the beam frame and
  rotate back. Exact, but it does not reduce dimensionality.
* `hyperbeam` — canonicalise, then beam-rotate the canonical configuration.
* `hyperres` — `hyper` but regress `E_conv - E_point`, add the analytic point
  field back before the inverse map.
* baseline — the T113 direct-`E_conv` model (806k MLP, `phys` features),
  re-run on the same split to get a matched number.

All models: 512x4 SiLU MLP, 1500 epochs, AdamW, L2 relative loss on `E`,
`cB = beta x E`; checkpoint selected on validation p99 (same criterion as
T113). Split: 7097 train / 1511 val / 1392 test, by whole physical
configuration.

## 3. Results (component-wise `|err|/(1+|ref|)` over E and cB)

| model | train p99 | train worst | test p99 | test worst |
|---|---|---|---|---|
| baseline (direct `E_conv`) | 5.08e-2 | 2.55e-1 | 2.298e-1 | 6.776e-1 |
| **hyper** | **2.71e-2** | **1.77e-1** | **1.204e-1** | 6.903e-1 |
| beam | 5.40e-2 | 4.19e-1 | 5.475e-1 | 4.486 |
| hyperbeam | 3.98e-2 | 8.47e-1 | 2.874e-1 | 1.926 |
| hyperres | 7.68e-2 | 5.27e-1 | 2.251e-1 | 2.802 |

Held-out by gamma, p99 / worst:

| gamma | baseline | **hyper** | beam |
|---|---|---|---|
| 1 | 0.0307 / 0.156 | **0.0242 / 0.116** | 0.2297 / 1.199 |
| 1.001 | 0.0408 / 0.105 | **0.0403 / 0.202** | 0.3722 / 1.700 |
| 2 | 0.1053 / 0.542 | **0.0753 / 0.372** | 0.4308 / 1.404 |
| 10 | 0.1713 / 0.458 | **0.1235 / 0.428** | 0.5414 / 1.825 |
| 100 | 0.2915 / 0.467 | **0.2015 / 0.690** | 0.7113 / 4.486 |

Held-out by azimuth around the beam (hyper, p99 / worst):

| azimuth | p99 | worst |
|---|---|---|
| [-180,-90) | 0.108 | 0.309 |
| [-90,0) | 0.146 | 0.428 |
| [0,90) | 0.120 | 0.354 |
| [90,180) | 0.117 | 0.690 |

In sample by gamma (hyper, p99 / worst): 0.0160/0.177, 0.0199/0.064,
0.0224/0.056, 0.0248/0.150, 0.0417/0.122 for gamma 1, 1.001, 2, 10, 100.

In sample by gamma (beam, p99 / worst): 0.0487/0.185, 0.0415/0.419,
0.0592/0.380, 0.0539/0.166, 0.0588/0.374.

## 4. Comparison and verdict

* Overall held-out p99: **0.2298 -> 0.1204, a 1.91x improvement**
  (1.87x against the previously reported 0.225).
* Per gamma p99 improvement: 1.27x, 1.01x, 1.40x, 1.39x, 1.45x.
* Held-out worst case: 0.678 -> 0.690, **no improvement**. The tail is
  dominated by high-gamma inside/face points in unseen configurations.
* The exact-symmetry model also improves in sample (p99 2.7e-2 vs 5.1e-2), but
  the held-out gap is still ~4.4x, so the limit has not become representation
  conditioning.
* Azimuth is flat (0.108-0.146). There is no angular blind spot left to fix.

**Route verdict.** Enforcing the exact discrete symmetries is correct and
cheap, but it is a 1.9x improvement, not an order of magnitude. At the measured
T113 learning-curve slope of 1.35x per data doubling, 0.120 -> 0.010 needs
about 8 doublings (~250x rows, ~7e6 rows, ~60 h of generator time) without
extra structure. The 1.9x from symmetry is worth roughly two of those
doublings, so the campaign would still need ~6 doublings (~60x rows, ~6e5 rows,
several hours) for p99, and the worst-case number that the gate actually uses
did not move at all. **Do not commission a data campaign on this route.** The
beam-aligned representation as stated is additionally not exact for anisotropic
cells and measured worse in every split.

## 5. Inference cost (final `hyper` model, V100, batched, pure forward)

801,283 parameters, 22 features.

| batch | FP32 (us/slot) | FP16 (us/slot) |
|---|---|---|
| 1e4 | 0.193 | 0.047 |
| 1e5 | 0.157 | 0.037 |
| 1e6 | 0.151 | 0.036 |

CPU (numpy) canonicalisation: 0.273 us/slot. End-to-end FP32 is therefore
~0.42 us/slot plus feature construction; inference remains irrelevant next to
the accuracy gate.

## 6. Artefacts

Branch `experimental/t116-symmetry-nn`, directory `neuralBunch/t116-symmetry/`.
Inputs: pilot `sha256 6a8cc50d...6d89b`, reference `sha256 a1355980...99519`
(dataset SHA from the T103 manifest `e12225f7...a80f6`).

SHA-256 of the committed files:

```
train_t116.py            babdc68138c1a8867899a1c7f354999f4178a5080708f2d677bdef22edff7859
verify_identities.py     c64ddb7713699c380916e52942b5921d9b2bf8a62c3c037e9889ebf332106736
summarize.py             11c6b227053abf0a09b232e2aa968e9b0bd9eb25c7c2aa21fc88cc5f9670e1a8
bench_hyper.py           17f574e8d1515f1ff2d341403c14429a3fbfb86d3964be539660e283f4ba5f23
identities.json          46de96b0790543e204f35ce0e0198149bd76a7508cf6e391c7485382ecbc3351
t116_base.json           ef996c7b108c51005c287f80ccb02fcd9985ab827ed3c11bafd78ec0888d236d
t116_hyper.json          4987689069f506bde05a361901096706341ed068fb608ba5dc79a047fdc79340
t116_beam.json           5e4887072ae9bc5f2a3f956061179837fe6718c24a59cda2b7dc2395c47da081
t116_hyperbeam.json      3ba7e592eebb506ee7dcfc7761ce279286cd0c932c56606db966e01632a065e5
t116_hyperres.json       59c708667229ba2275fcfa7fa6da693d93998af18a898feced8bb5fd0435c3ef
t116_hyper.npz           ea52e9741f2c4015327a8818c511595d26e5903ae059950ee2feb894e67e0d08
```
