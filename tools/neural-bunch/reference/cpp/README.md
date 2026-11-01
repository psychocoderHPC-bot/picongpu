# C++ FP64 reference port

Header-only C++20 port of the validated Python reference in `../convkernel.py`
for the production fallback path.  No alpaka, PIConGPU, Python, BLAS or FFT
dependency.  All arithmetic is `double`.

## Files

- `neural_reference.hpp` — the whole implementation:
  `point_field`, `shape_field`, `tile_field`, `tsc_shape`, `offsets`,
  `Settings` (`order`, `tol`, `max_level`, `max_order`), `ShapeResult`,
  `TileReport`.
- `test_reference.cpp` — plain-main test runner.
- `benchmark.cpp` — tile benchmark mirroring `../benchmark_tile.py`.
- `tile_bench.cpp` — CLI: one source/tile, 1536 tile values plus timing and
  per-output convergence statistics.
- `generate_testdata.py` — regenerates the golden tiles from the Python
  reference (only needed when the reference changes; not needed to run tests).
- `testdata/*.tile` — checked-in golden tiles, 1536 values each in
  `c + 6*(i + 8*(j + 8*k))` order.
- `testdata/aspect_cases.txt` — 225 Python shape-field vectors over the aspect
  grid (header documents the 15 columns).
- `testdata/near_cases.txt` — 24 Python shape-field vectors just outside the
  support, for the <=1e-9 cross-check.

## Build and run

```bash
cd tools/neural-bunch/reference/cpp

g++ -std=c++20 -O3 -march=native -fopenmp -Wall -Wextra test_reference.cpp -o test_reference
OMP_NUM_THREADS=8 ./test_reference

g++ -std=c++20 -O3 -march=native -fopenmp -Wall -Wextra benchmark.cpp -o benchmark
./benchmark

g++ -std=c++20 -O3 -march=native -fopenmp -Wall -Wextra tile_bench.cpp -o tile_bench
./tile_bench src_x src_y src_z tile_x tile_y tile_z dx dy dz [tol]
```

`tile_bench` prints 1536 `tile <index> <value>` lines, then `time_seconds`,
`threads`, `outputs`, `near_outputs`, `not_converged`, `max_error` and
`converged`.  Results are bitwise identical for any `OMP_NUM_THREADS` (verified
1 vs 14).

`test_reference` reads `./testdata`.  It can be given another directory as its
first argument.

Regenerate the golden vectors (Python deps required only here):

```bash
cd tools/neural-bunch/reference
python cpp/generate_testdata.py
```

## Exact numerical settings for the golden vectors

Both the Python generator and the C++ tests use:

| setting | value |
|---|---|
| starting low order / error pair | `4` / `4` vs `6` |
| adaptive tolerance | `1e-7` |
| maximum per-axis grading level | `5` |
| maximum Gauss order | `16` |
| base panels per axis | `[-1.5,-0.5]`, `[-0.5,0.5]`, `[0.5,1.5]` |
| near diagnostic threshold | physical support-box distance `< 2*max(dx,dy,dz)` |

Every observation point is error-controlled: orders `order` and `order+2` are
compared on the same panels.  If the relative difference exceeds `tol`, panels
are geometrically graded around the projected target and the order pair rises
by two, up to `max_level`; after that the order keeps rising on the same panels
up to `max_order`.  The near flag is diagnostic only and never bypasses the
estimate.  No epsilon is added to any denominator.

## Measured accuracy (C++ vs Python golden)

`test_reference` result: **531 checks, 0 failures**, 10 golden tile cases, 225
aspect-grid vectors and 24 near-set vectors.  Worst near-set Python/C++
relative difference: **1.6e-14**.

| case | class | max relative (inf-norm) | limit |
|---|---|---:|---:|
| `far_iso_gamma1_stationary` | far | 6.3e-16 | 1e-10 |
| `far_iso_gamma2` | far | 8.7e-16 | 1e-10 |
| `far_iso_gamma10` | far | 9.7e-16 | 1e-10 |
| `far_iso_gamma100` | far | 9.9e-16 | 1e-10 |
| `far_anisotropic` | far | 7.1e-16 | 1e-10 |
| `mixed_sign_weights` | far | 5.5e-16 | 1e-10 |
| `far_aspect10_d128` | far aspect 10 | 2.3e-15 | 1e-10 |
| `far_aspect0p1_d128` | far aspect 0.1 | 2.1e-15 | 1e-10 |
| `far_aniso_independent_d32` | far dy!=dz | 2.0e-15 | 1e-10 |
| `near_just_outside_support` | near/adaptive | 1.09e-14 | 1e-7 |

Additional checks: empty sources and zero weight give bit-exact zero tiles;
stationary sources give bit-exact zero B channels; all 1536 slots match an
independent direct `shape_field` evaluation at the pinned six offsets and
flatten index; translation invariance 5.9e-16; `E ~ 1/L^2` scaling 5.3e-15.
The 225-case aspect grid (aspects 0.1..10, distances 4..512 cells, gamma
1/10/100, independent dy/dz) gives worst order 4/6/8 self-agreement 4.6e-8 and
worst Python-vs-C++ relative difference 1.3e-14.

Error-controlled tile with starting order 4 vs starting order 8 (N=1):
relative L2 6.4e-13 / 1.6e-14 / 1.5e-14 at 4 / 8 / 32 cells.

## Measured cost and comparison with Python

One tile = 1536 staggered outputs.  C++ numbers use 14 OpenMP threads and are
measured; Python numbers are the single-thread reference (N=256 projected from
measured N=16 because cost is linear in sources).

Isotropic cells (1,1,1), C++ with 14 OpenMP threads:

| distance | N=1 [s] | N=16 [s] | N=256 [s] | N=256 s/source |
|---:|---:|---:|---:|---:|
| 4 | 0.019 | 0.200 | 3.167 | 0.0124 |
| 8 | 0.014 | 0.202 | 4.236 | 0.0165 |
| 32 | 0.028 | 0.381 | 5.633 | 0.0220 |

Aspect 10 (cells 1,10,10), 256 sources:

| distance | C++ [s] | C++ s/source |
|---:|---:|---:|
| 4 | 10.340 | 0.0404 |
| 8 | 6.887 | 0.0269 |
| 32 | 5.372 | 0.0210 |
| 128 | 5.300 | 0.0207 |

Python reference (for comparison, no OpenMP batching): ~12-22 minutes for the
same 256-source aspect-10 tiles; C++ is the production fallback.

## Near-field cost (the production bottleneck)

In-support outputs no longer use per-output adaptive panel quadrature.  The
support box is split into the 27 TSC sub-boxes; the sub-box containing the
singular point is split at that point into corner boxes, each integrated with
three pyramids (apex at the singularity).  The `r^2` Jacobian cancels the
`1/r^2` kernel, so radial order 4 is exact for the polynomial W and the angular
order converges quickly.  Other sub-boxes use tensor Gauss.  Each output is
still error-controlled (angular 10 vs 14, box 8 vs 10, then increased).

One tile = 1536 outputs, source at the tile centre (`u=(0.3,0.1,0)`), tol 1e-7:

| threads | centre [s] | 2 cells outside [s] | 4 cells outside [s] | far (32 cells) [s] |
|---:|---:|---:|---:|---:|
| 1 | 0.433 | 0.087 | 0.068 | 0.105 |
| 2 | 0.253 | 0.043 | 0.043 | 0.064 |
| 4 | 0.206 | 0.038 | 0.034 | 0.033 |
| 8 | 0.105 | 0.023 | 0.024 | 0.020 |
| 14 | **0.079** | **0.022** | **0.017** | **0.021** |

The 2-4-cell-outside and far cases meet the 50 ms target at 14 threads.  The
all-near centre case reaches **79 ms** (floor; 33 ms single-thread upper bound
is not reached because 14 logical CPUs give ~5x scaling).  Old code: 94.3 s
for the centre tile (single thread), i.e. ~1200x faster now.  `max_error`
9.98e-8, `not_converged=0` for the centre tile.

## Limitations

- Host-only FP64 with OpenMP across (source, slot) pairs; no GPU/alpaka
  backend.  Load imbalance and SMT limit scaling to ~5x on 14 logical CPUs.
- Remaining near bottleneck: tensor Gauss in the 26 non-singular TSC sub-boxes
  per in-support output; this is what keeps the all-near centre tile at 79 ms
  instead of <=50 ms.
- The error field is an order-difference estimate, not a rigorous bound.
- Python values for points *inside* the support are themselves under-converged
  (up to ~5e-6 here); C++ box values match independent SciPy cubature to
  ~3.5e-8.  The 1e-9 Python/C++ check therefore uses the 24-point outside-
  support near set; in-support outputs are validated by the error estimate and
  cubature instead.
- Golden vectors are regenerated with Python order-4/6 error control; C++ and
  Python agree to ~1e-14 on the aspect grid.

- Only the physics normalized in `../conventions.md` is implemented; unit
  conversion to PIConGPU internal units is left to the caller.
- Golden vectors are text files with 17 significant digits; tests are
  deterministic and require no network.
