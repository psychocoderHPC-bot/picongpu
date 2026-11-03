# Neural bunch-field initializer: training and export

PyTorch training/export package for the additive per-source neural initializer.
The artifact contract is `../model-format.md` version 1.

Architecture (fixed): `z = sum_i (q_i/e) h_theta(fourier_v1(r_i,u_i,a_y,a_z))`,
`F_tilde = D(a_y,a_z) z`. `h_theta` is a SiLU MLP (default 256,256, latent 128).
`D` is a bank of linear maps on a 5x5 log-spaced aspect grid in `[0.1, 10]`,
bilinearly interpolated in `log10(a)` and trained against continuously sampled
aspects. No bias in the decoder, no batch statistics, no dropout, no positional
input ordering.

## Environment

```bash
cd tools/neural-bunch/training
python3 -m venv venv
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install numpy scipy pytest
```

CPU-only here. `torch` is used for training; `numpy` powers the artifact forward
and parity check.

## Package layout

```
neuralbunch/
  __main__.py        CLI: reference,train,validate,export,inspect,parity
  features.py        frozen fourier_v1 map (69 dims)
  model.py           torch encoder/bank + pure-numpy artifact forward
  sampling.py        deterministic scenarios, bunches, chunking, split assignment
  reference_loader.py adapter for reference/convkernel.py (read-only)
  reference_data.py  sharded FP64 targets, manifest, resume, dataset loader
  train.py           losses, optimizer, checkpoints, evaluation, reports
  export.py          MODEL-FORMAT v1 writer/loader + validation
  inspect.py         artifact inspection
  parity.py          torch vs numpy forward parity
  testing_standin.py ANALYTIC STAND-IN, tests/plumbing only
tests/               pytest suite
smoke/               random-weight plumbing fixture (not a trained model) and report
```

## Reference contract assumed by the adapter

`reference/convkernel.py` is implemented separately and must not be edited here.
`reference_loader.py` calls `tile_field(x_tile, particles, dx, dy, dz)` where
`x_tile` is the tile cell origin in physical units and `particles` is a
structured NumPy array with fields `x,y,z,ux,uy,uz,w` (physical positions,
`u=p/(m_e c)`, positive `w`, with `q/e=-w`). `tile_field` returns `(8,8,4,6)` in
[i,j,k,component] order. A dict-of-arrays and an `(N,7)` array are also accepted.
If the reference finalizes a different signature, adjust only the adapter.

## Commands

All commands run from `training/`.

Generate FP64 targets (real reference):

```bash
./venv/bin/python -m neuralbunch reference \
  --out data/dataset --split all --seed 0 \
  --n-configs 2000 --samples-per-config 4 --shard-size 32 \
  --include-bunches --n-particles 128 \
  --bunch-gammas 2 10 100
# optional: --aspect-min 0.5 --aspect-max 2.0 --ref-fixed --ref-order 6
```

`--reference PATH` overrides the default reference path; `--allow-standin`
generates a clearly labeled stand-in dataset (`reference.kind` is
`analytic_standin`) for plumbing only, never for accuracy claims.

Train (checkpoints `best.pt`/`last.pt`, `training_report.json`):

```bash
./venv/bin/python -m neuralbunch train \
  --data data/dataset --out runs/v1 \
  --hidden 256,256 --latent 128 --optimizer adamw \
  --lr 3e-4 --weight-decay 1e-4 --batch-tiles 32 \
  --max-steps 200000 --eval-every 2000 --seed 0 \
  --force-weight 0.1 --trials 3 --time-budget-s 43200 \
  --export artifacts/model_v1
```

Validate a checkpoint or artifact against every split:

```bash
./venv/bin/python -m neuralbunch validate \
  --artifact artifacts/model_v1 --data data/dataset \
  --splits train,val,test --out artifacts/validation_report.json
```

Export a checkpoint (writes `<prefix>.json` and `<prefix>.bin`):

```bash
./venv/bin/python -m neuralbunch export \
  --checkpoint runs/v1/best.pt --out artifacts/model_v1 --inspect
```

Inspect/reject artifacts (exit code 1 on invalid input):

```bash
./venv/bin/python -m neuralbunch inspect artifacts/model_v1
./venv/bin/python -m neuralbunch inspect smoke/model
```

Parity (torch forward vs pure-numpy artifact forward):

```bash
./venv/bin/python -m neuralbunch parity --artifact artifacts/model_v1 \
  --particles 257 --trials 3
```

Tests:

```bash
./venv/bin/python -m pytest -q
```

## Targets, scales and metrics

The real reference `../reference/convkernel.py` is now wired in; there is no
stand-in path in the dataset CLI. The dataset manifest stores, per sample and
before any training, the reference SHA-256, quadrature order, `d_min_cells`,
and the preregistered scale `S0_E, S0_B`. `S0_G = sqrt(N_G/3) * max_cell
||G_cell||` over the reference tile (`N_G = 768`), computed from the reference
configuration only.

For each case and group (`E`, `cB`):

```
rel_rms = ||A_model - A_ref|| / max(||A_ref||, S0)
```

No case is dropped. A case with `||A_ref|| < S0` is labelled
`absolute_normalized` and reported separately with its `S0`. Reports give
mean, p99, max, the scale-weighted mean, and per-case arrays for `E` and `cB`
separately. Training optimizes exactly `||A_pred-A_ref||^2 / max(||A_ref||,S0)^2`
per case, so weak/cancelling cases are neither reweighted by batch statistics
nor ignored.

The force metric trilinearly interpolates the six staggered components to each
source position (explicitly **trilinear**, not the PIConGPU gather), forms
`g = E_tilde + beta x cB_tilde`, and reports
`||g_model-g_ref|| / max(||g_ref||, S0_E + beta*S0_B)`, the absolute-normalized
count, and the cancellation ratio `||E|| / ||beta x cB||`. Divergence
diagnostics are reported only.

Quadrature policy for real targets: order 4 in the smooth far branch, order 6
for `|u| > 10`, and the tight order 8 rule for `|u| > 5` with `d_min < 10`
cells. A cross-check against the order-8 rule is in
`scripts/reference_crosscheck.py` output (max normalized difference 5e-10
over ten configurations).

## Artifact path policy

* dataset: `<root>/<split>/manifest.json` plus `shard_XXXXX.npz`; the manifest
  stores descriptors, seeds, reference SHA-256 and shard hashes, and is updated
  atomically after each shard so an interrupted run resumes.
* run: `<run>/best.pt`, `<run>/last.pt`, `<run>/training_report.json`; trials
  with `--trials > 1` live under `<run>/trial_<i>/`.
* artifact: export prefix `<name>` produces `<name>.json` and `<name>.bin`
  side by side; loaders verify byte count and SHA-256 before use.
* The PIConGPU checkout and `reference/` are never modified.

## Reproducibility

* Sampling is seeded per split (`--seed`, `+1000` val, `+2000` test); physical
  configurations (gamma regime, direction, aspect bucket, position regime) are
  assigned to train/val/test by CRC32 of the configuration key, never by row.
* The dataset manifest records reference file SHA-256, descriptors, seeds and
  per-shard hashes.
* Checkpoints store model/optimizer state, torch and NumPy RNG state, config,
  code SHA, environment and loss history; `--resume` continues from `last.pt`.
* The artifact `code_sha` is `NEURALBUNCH_CODE_SHA` when set, else the training
  directory Git HEAD, else a SHA-1 over the package sources (the training
  directory is not a repository here, so the source hash is used).

## Smoke run (plumbing only, NOT accuracy)

This is the historical first smoke run, made before the reference was fixed; the
dataset is labeled `analytic_standin`. A real-reference feasibility run is described in the feature documentation. The exported artifact is
real and format-valid; see `smoke/README.md` for the full error table. **The
smoke numbers are not physics validation.**

```bash
./venv/bin/python -m neuralbunch reference --out smoke/data --split all \
  --n-configs 12 --samples-per-config 2 --shard-size 4 --allow-standin --seed 0
./venv/bin/python -m neuralbunch train --data smoke/data --out smoke/run \
  --hidden 32,32 --latent 16 --max-steps 600 --eval-every 100 \
  --batch-tiles 8 --lr 1e-3 --seed 0 --export smoke/model \
  --validation-report smoke/validation_report.json
./venv/bin/python -m neuralbunch parity --artifact smoke/model --particles 257 --trials 3
```

Results: 24 train / 24 val / 24 test cases; `smoke/model.json` +
`smoke/model.bin` (2,472,896 bytes, SHA-256
`484b35b0...98dded1`); inspect accepts it; the test suite builds a corrupted
copy and checks that it is rejected with `payload sha256 mismatch`; torch/numpy parity max relative error
`4.67e-7`. Smoke errors against the stand-in (not physics validation): val E
mean/p99/max relative RMS 1.80/6.92/7.36, val B 2893/13439/14390; test E
2.61/20.49/24.20, test B 1523/9143/10002. These errors are large and expected
for 24 samples, 600 steps and a 16-wide latent; they prove the pipeline only.

## Recommended next GPU run

Generate the real dataset first, then train. On an exclusive GPU host:

```bash
./venv/bin/python -m neuralbunch reference --out data/dataset \
  --split all --seed 0 --n-configs 4000 --samples-per-config 4 --shard-size 64 \
  --include-bunches --n-particles 256 --bunch-gammas 2 10 100
./venv/bin/python -m neuralbunch train --data data/dataset \
  --out runs/v1 --hidden 256,256 --latent 128 --optimizer adamw \
  --lr 3e-4 --weight-decay 1e-4 --batch-tiles 32 --max-steps 200000 \
  --eval-every 2000 --seed 0 --force-weight 0.1 --trials 3 \
  --time-budget-s 43200 --export artifacts/model_v1
./venv/bin/python -m neuralbunch validate --artifact artifacts/model_v1 \
  --data data/dataset --splits train,val,test \
  --out artifacts/validation_report.json
```

Tune `--batch-tiles`, `--lr` and `--max-steps` against validation loss; keep the
test split untouched until the final report.
