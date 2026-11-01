# Contributing to the neural bunch tooling

Keep changes focused, run both reference implementations, and never commit
datasets, checkpoints (`*.pt`), run directories or the trained production
artifact. Generated data belongs outside the checkout.

## Regenerate the golden vectors

The golden tiles are produced by the Python reference and consumed by the C++
test. Regenerate them only when `reference/convkernel.py` changes on purpose,
and review the numerical settings in `reference/cpp/README.md` before doing so.

```bash
cd tools/neural-bunch/reference
python cpp/generate_testdata.py
git diff --stat cpp/testdata
```

`generate_testdata.py` needs `numpy` and `scipy`. It rewrites
`cpp/testdata/*.tile`, `cpp/testdata/aspect_cases.txt` and
`cpp/testdata/near_cases.txt`. A change to those files must be justified by a
change to the reference, not by a failing test.

## Run the C++ reference test

```bash
cd tools/neural-bunch/reference/cpp
g++ -std=c++20 -O3 -march=native -fopenmp -Wall -Wextra test_reference.cpp -o test_reference
OMP_NUM_THREADS=8 ./test_reference
```

The test reads `./testdata` by default and accepts another directory as its
first argument. It checks the 10 golden tiles, the 225 aspect-grid vectors, the
24 near-support vectors and the structural properties (zero-source, zero-weight,
stationary B, flat indexing, translation invariance and `E ~ 1/L^2` scaling).
Results are independent of `OMP_NUM_THREADS`.

The benchmark and single-tile tools build from the same header:

```bash
g++ -std=c++20 -O3 -march=native -fopenmp -Wall -Wextra benchmark.cpp -o benchmark
./benchmark
g++ -std=c++20 -O3 -march=native -fopenmp -Wall -Wextra tile_bench.cpp -o tile_bench
./tile_bench src_x src_y src_z tile_x tile_y tile_z dx dy dz [tol]
```

## Run the Python reference and training tests

```bash
cd tools/neural-bunch/reference
python -m pytest -q

cd ../training
python -m pytest -q
```

The training tests use the real reference adapter, whose default path is
relative to the package. Set `NEURALBUNCH_REFERENCE` only to point at a
different checkout.

## Paths and environment variables

Committed defaults must stay relative. Use these variables instead of editing
paths:

| variable | used by | meaning |
| --- | --- | --- |
| `NEURALBUNCH_REFERENCE` | training | path to `convkernel.py` |
| `NEURALBUNCH_CODE_SHA` | training export | full code SHA stamped into `model.json` |
| `PYTHON` | generator scripts | interpreter for `generate_bunch.py` / `inspect_bunch.py` |
| `OUT` | `generator/examples.sh` | generated-data directory |
| `PICONGPU_SERIES` | `generator/decode_piconpu.py` | real PIConGPU openPMD series |
| `NEURAL_BUNCH_FORCE_REFERENCE` | PIConGPU runtime | validation-only all-reference switch |

## Before committing

- `git status` shows only intended files.
- No committed file contains an absolute temporary or home-directory path.
- Both reference test suites and the C++ test pass.
- Report the artifact SHA-256 next to any accuracy or performance claim.
- Do not describe the feature as production ready; see
  `docs/neural-bunch/README.md` for the measured status.
