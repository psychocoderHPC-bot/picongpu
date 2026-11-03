# Reference benchmark (measured)

Environment: Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, g++ 13.3 `-O2`, Linux,
14 logical CPUs.  All values are FP64 wall-clock seconds, single-threaded code.
Python numbers with N=16 are measured; the N=256 column is the linear
projection (cost is linear in sources, measured at N=16).  C++ numbers at N=256
are measured directly.

A tile is 8 x 8 x 4 cells and 1536 staggered outputs.

## Quantities and rule

* Error control: orders `order` and `order+2` (default 4/6) are compared on the
  same panels; `err = max|I_hi-I_lo| / (1+max|I|)`.  If `err > tol` (1e-7) the
  per-axis panels are geometrically graded and the order pair is raised by two
  up to `max_level=5`; after that the order is raised on the same panels up to
  `max_order=16`.
* Near diagnostic (never bypasses the estimate): physical distance to the TSC
  support box, `excess_i=max(|R_i|-1.5*cell_i,0)`, `|excess| < 2*max(dx,dy,dz)`.
* The old fixed far order-4 path is gone.

## Convergence: order 4/6/8 self-agreement

Full grid: aspect corners `dy,dz in {0.1,0.5,1,2,10}` (including independent
`dy != dz`), distances 4/8/32/128/512 cells, gamma 1/10/100 (225 shape_field
cases).  Relative difference of order 4 and 6 against order 8:

| aspect | worst rel. difference |
|---:|---:|
| 0.1 | 9.7e-12 |
| 0.5 | 9.7e-12 |
| 1 | 4.6e-08 |
| 2 | 5.6e-10 |
| 10 | 4.6e-08 |

Worst by distance: 4 cells 4.6e-8, 8 cells 1.6e-8, 32 cells 4.1e-12,
128 cells 4.9e-15, 512 cells 5.6e-15.  All are below the 1e-6 requirement.

Golden tile comparison (C++ vs Python): 10 far/stationary/mixed cases
<= 2.3e-15, sub-cell near case 1.1e-14 (limit 1e-7).  The 225-case aspect grid
gives worst Python vs C++ 1.3e-14 relative (limit 1e-9).

## Cost, isotropic cells (dx=dy=dz=1)

| distance | N | C++ [s] | C++ s/src | Python N=16 [s] | Python s/src | Python 256 proj. [s] |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 0.139 | 0.139 | - | - | - |
| 4 | 16 | 2.171 | 0.136 | 44.87 | 2.804 | 718 |
| 4 | 256 | 36.157 | 0.141 | - | - | - |
| 8 | 16 | 2.484 | 0.155 | - | - | - |
| 8 | 256 | 36.850 | 0.144 | - | - | - |
| 32 | 1 | 0.134 | 0.134 | - | - | - |
| 32 | 16 | 2.239 | 0.140 | 45.67 | 2.854 | 731 |
| 32 | 256 | 36.277 | 0.142 | - | - | - |

## Cost, aspect 10 (dx=1, dy=dz=10)

| distance | N | C++ [s] | C++ s/src | Python N=16 [s] | Python s/src | Python 256 proj. [s] |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 16 | - | - | 83.93 | 5.245 | 1343 |
| 4 | 256 | 68.991 | 0.270 | - | - | - |
| 8 | 256 | 46.313 | 0.181 | - | - | - |
| 32 | 16 | - | - | 46.06 | 2.879 | 737 |
| 32 | 256 | 36.732 | 0.144 | - | - | - |
| 128 | 256 | 35.592 | 0.139 | - | - | - |

**Measured worst case:** C++ 256-source aspect-10 tile at 4 cells: 69.0 s
(0.270 s/source).  At the typical far distance 32 cells it is 36.7 s
(0.144 s/source).  The Python reference is ~12-22 minutes for the same 256
sources and is only used for golden-vector generation.

## Where adaptive quadrature dominates

Single observation, stationary source: 0.0006 s at 2 cells (flagged near but
converges at level 1), 0.484 s at 1 cell and 0.485 s at 0.5 cell (level 5,
error estimate ~1e-7 / 6e-7).  A tile whose outputs lie inside the support is
therefore hundreds of seconds even in C++; the aspect-10 benchmark above has
most outputs in the base-panel error-controlled branch, with only the outputs
near the support needing grading.
