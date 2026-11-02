# Neural bunch tooling

Python tooling and measured notes for the `PIC_USE_neuralBunch` PIConGPU
feature. The runtime contract and the field formulas are shared with
[`model-format.md`](model-format.md).

## Runtime status (measured)

The feature is experimental and `enabled = false` by default; a standard setup
is unchanged (KHI and LWFA were run with the feature compiled in and disabled).
When enabled, the default policy is `banded`: the exact FP64 reference in the
near band (`d < 4` cells, host or device quadrature) and the analytic point
field in the mid (`4 .. 48`) and far (`>= 48`) bands. `neural`,
`validatedNeural` and `referenceOnly` remain selectable. The learned
`mid_correction` came out worse than the point field alone and is opt-in; the
Taylor far head is opt-in too and off by default.

With the default `tileWideField = true` the whole point-field loop runs in one
tile-centric device kernel (one block per tile, 256 threads per 1536-output
group, deterministic ascending source order). The kernel skips near pairs
instead of integrating them exactly; it is disabled automatically for
`useExactDevice`, `useModelForNear`, `useModelForMid`, `useMidCorrection`, the
validation-only all-reference switch, and any non-`banded` policy. In the
default configuration all field math is on the device; the host only does MPI,
openPMD decoding, model load and hash verification, and particle frame
bookkeeping.

Measured cost: 5.2 ns per point-field pair; 5.71e8 pairs/s for the tile kernel
probe on 4 V100; 1M particles on a 32x32x16 grid initialize in 17.6 s; 1M
particles on 256x1024x256 reached >10.7 min before the allocation ended,
<=3.7e8 pairs/s aggregate, 12-15 min projected. A persistent cuBLAS handle cut
a batched decoder call from 310 us to 5 us, and the host per-pair classification
is gone from the default path.

Measured accuracy, preregistered `S0` metric (`S0 = 16 * max_cell ||G||`, per-case
rel RMS `||pred - ref|| / max(||ref||, S0)`, worst of E/B) plus per-point slices:
the point field gives gamma 2 mean 5.7e-4, gamma 10 mean 4.1e-3 / median 1.1e-3,
and gamma 100 mean 16 % (median 7.5e-2). The gamma dependence is physical: at
gamma 100 the point field is a pancake of thickness about `r/gamma`, comparable
to the 3-cell TSC support, so the shape integral differs from the point field.
The device quadrature matches the host reference to 4.9e-7 max relative error;
the host reference costs 5.7 us per smooth pair and 7.5 ms per in-support pair.
The learned model is about 15 % and is not the default. The trilinear force
stencil around a particle is never 4 cells clear of every source, so the point
field does not cover force at the particles; that needs the exact shape
quadrature. No learned artifact reaches the 1e-3 gate: architecture-compatible
encoders floor at 0.11-0.17 while the oracle basis reconstructs at 1.6e-7, which
locates the limit in the encoder rather than the feature basis.

Plots, cost tables and per-slice metrics are development artifacts under
`/tmp/alpaka-orchestration/neural-bunch/` (`plots-y/`, `plots-exp/`,
`mid-exp/`, `research/BANDS.md`, `validation/`) and are not committed.

## Pipeline stages

1. **Reference** (`reference/`) - independent FP64 implementation of the
   present-time field of uniformly moving point charges integrated over the
   PIConGPU TSC shape on an 8x8x4 Yee tile. The Python module
   (`convkernel.py`) is the numerical oracle; `reference/cpp/` is the
   header-only C++20 port used by the runtime fallback. Golden tiles and the
   aspect/near vector sets pin the C++ port to the Python reference. The runtime
   analytic point field (`pointFieldDx`), the banded policy and the FP64 device
   quadrature are maintained in `include/picongpu/particles/neuralBunch/`; the
   quadrature was ported from the NALA prototype (`nala/ExactShapeField.hpp`).
2. **Generator** (`generator/`) - seeded synthetic electron bunches written as
   openPMD 2.0 series (HDF5 or ADIOS2/BP5), for external-particle tests and
   independent I/O checks. `inspect_bunch.py` reads them back with the same
   code path used for real PIConGPU dumps.
3. **Training** (`training/`) - PyTorch training, loss/metrics, checkpointing,
   pure-NumPy artifact forward, parity check and export. The exported artifact
   is `model.json` + `model.bin` described by `model-format.md`.
4. **Export** - `python -m neuralbunch export` writes the two artifact files and
   records `artifact.bytes` and `artifact.sha256` in the JSON. Export also
   stamps `code_sha`.
5. **Inspection** - `python -m neuralbunch inspect` validates the JSON schema,
   tensor table, byte count and SHA-256 before use. The C++ loader applies the
   same checks collectively across MPI ranks.

## Dependencies

| Stage | Packages |
| --- | --- |
| reference (Python) | `numpy`, `scipy`, `pytest` |
| reference (C++ tests) | a C++20 compiler, OpenMP; no BLAS/FFT/Python |
| generator | `numpy`, `openpmd-api`, `h5py` (HDF5), ADIOS2 (BP5 backend) |
| training | `torch`, `numpy`, `scipy`, `pytest` |

Python 3.12 is the tested interpreter. Nothing here is installed by PIConGPU's
build system; these are standalone development tools.

## Reference tests

The Python and C++ reference implementations are independent. Run both after
any change to `convkernel.py` or `neural_reference.hpp`.

```bash
# Python reference tests (from tools/neural-bunch/reference)
python -m pytest -q

# C++ reference test, golden vectors from cpp/testdata
cd tools/neural-bunch/reference/cpp
g++ -std=c++20 -O3 -march=native -fopenmp -Wall -Wextra test_reference.cpp -o test_reference
OMP_NUM_THREADS=8 ./test_reference

# Optional: rebuild the golden vectors from the Python reference
cd tools/neural-bunch/reference
python cpp/generate_testdata.py
```

`test_reference` reads `./testdata` by default and accepts another directory as
its first argument. See `reference/cpp/README.md` for the exact numerical
settings and measured agreement.

## Generator

Run from `tools/neural-bunch/generator`. `--output` receives a `.h5` or `.bp5`
suffix; generated data must stay outside the checkout.

```bash
PY=${PYTHON:-python}
OUT=${OUT:-./neural-bunch-data}
mkdir -p "$OUT"

$PY generate_bunch.py --output "$OUT/stationary" --backend hdf5 \
    --seed 1001 --model stationary --n 4096
$PY generate_bunch.py --output "$OUT/stationary" --backend adios2 \
    --seed 1001 --model stationary --n 4096
$PY inspect_bunch.py "$OUT/stationary.h5" --compare "$OUT/stationary.bp5"
OUT="$OUT" bash examples.sh          # all 13 cases, both backends
$PY decode_piconpu.py --series simData_000000.h5   # real PIConGPU dump
```

`decode_piconpu.py` reads `--series` or the `PICONGPU_SERIES` environment
variable. See `generator/README.md` for the openPMD layout, sidecar and model
definitions.

## Training

Run from `tools/neural-bunch/training`.

```bash
python3 -m venv venv
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install numpy scipy pytest

# FP64 targets from the real reference
./venv/bin/python -m neuralbunch reference \
    --out data/dataset --split all --seed 0 \
    --n-configs 2000 --samples-per-config 4 --shard-size 32 \
    --include-bunches --n-particles 128 --bunch-gammas 2 10 100

# Train and export
./venv/bin/python -m neuralbunch train \
    --data data/dataset --out runs/v1 \
    --hidden 256,256 --latent 128 --optimizer adamw \
    --lr 3e-4 --weight-decay 1e-4 --batch-tiles 32 \
    --max-steps 200000 --eval-every 2000 --seed 0 \
    --force-weight 0.1 --trials 3 --time-budget-s 43200 \
    --export artifacts/model_v1

# Validate, inspect and check torch/numpy parity
./venv/bin/python -m neuralbunch validate \
    --artifact artifacts/model_v1 --data data/dataset \
    --splits train,val,test --out artifacts/validation_report.json
./venv/bin/python -m neuralbunch inspect artifacts/model_v1
./venv/bin/python -m neuralbunch parity --artifact artifacts/model_v1 \
    --particles 257 --trials 3
```

The reference adapter defaults to `tools/neural-bunch/reference/convkernel.py`
(relative to the package). Override it with `--reference PATH` or
`NEURALBUNCH_REFERENCE`. See `training/README.md` for all options and the
metric definition.

The checked-in `training/smoke/` directory is a **random-weight plumbing
fixture**, not a trained model. It exercises export, inspection, the validation
report and parity only. Its numbers are not physics validation.

## Reproducibility contract

- **Seeds.** The generator is a pure function of its arguments and `--seed`.
  Dataset sampling is seeded per split (`--seed`, `+1000` for val, `+2000` for
  test) and assigns physical configurations by CRC32 of the configuration key,
  not by row. Training stores the torch and NumPy RNG state in every
  checkpoint, so `--resume` continues deterministically.
- **Code SHA.** Exports record `code_sha`: `NEURALBUNCH_CODE_SHA` when set,
  otherwise the training directory Git HEAD, otherwise a SHA-1 over the package
  sources. For a release artifact set `NEURALBUNCH_CODE_SHA` to the full
  40-character commit you built PIConGPU from.
- **Data provenance.** Each dataset manifest records the reference file
  SHA-256, quadrature settings, descriptors, seeds and per-shard SHA-256.
- **Artifact identity.** `model.json` records `artifact.bytes` and
  `artifact.sha256` for `model.bin`. A loader must verify both before
  allocating tensors. Report the artifact SHA-256 next to every performance or
  accuracy claim.
- **Generated data.** Datasets, checkpoints (`*.pt`), run directories and the
  trained `fourier512_v1` artifact are not committed. Only the small smoke
  fixture is checked in, and it is clearly labelled.

## Layout

```
tools/neural-bunch/
  model-format.md        artifact contract v1 (shared with the C++ loader)
  reference/             Python FP64 reference, tests, golden data, C++ port
  generator/             seeded openPMD bunch generator and reader
  training/              PyTorch package, tests, scripts, smoke fixture
  CONTRIBUTING.md        regenerating golden vectors and validation commands
```
