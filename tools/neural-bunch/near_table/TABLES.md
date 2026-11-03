# Near-field tables (T150)

Precomputed single-source near-field tables for the parameter-gated table path of
the neuralBunch near pass (`ExactSettings::useNearTable`, default off; see
`include/picongpu/particles/neuralBunch/NearFieldTable.hpp`).

Quantity stored (per unit charge, dx-independent normalization
`E_n = 4 pi eps0 E / q`, `B_n = 4 pi eps0 c B / q`):

    K(r) = int_{[-1.5,1.5]^3} W(s) Kpoint((r - s) cells, u) ds^3

with `r` the observer-minus-source displacement in cell units
(`r_i = (x_obs - x_src)_i / cell_i`), `W` the TSC shape and `Kpoint` the
Lienard-Wiechert point field of one uniformly moving charge. The table is exact
for one velocity (no binning in `u`) and one cell aspect; the C++ path falls
back to the quadrature path wherever the table is out of domain (displacement
beyond the grid, or velocity/aspect mismatch).

Layout (schema `nktable/v1`): symmetric graded 3D grid, 77 points per axis
(h = 0.2 for |r| < 4, h = 0.5 out to rmax = 13 cell units per axis), 6
components [Ex Ey Ez Bx By Bz], AoS with x fastest
(`flat = ((ix*n + iy)*n + iz)*6 + c`), float64 binary + JSON metadata.
Each table is 21,913,584 bytes. `K(r)` is odd in `r` (verified at generation).

## Tables (run pr24-20261102, rosi_v100 GPU 1, built 2026-11-03)

Reference for validation: `indfield.py` (validated: stationary 2.4e-10,
in-support floor ~1.4e-4 at order 12 / refine 3; the vendored copy is the
independent reference, byte-identical to `~/pr24/pr24w53-ref/indfield.py`).

| table | gamma | cells (dx, dy, dz) | u (p/mc) | sha256 |
| --- | --- | --- | --- | --- |
| nk_gamma1_ay0.25_az1 | 1 | (1, 0.25, 1) | (0, 0, 0) | a5fa85deb32fb26a3c2c62115960930f355e73dd4b1e72234a44ef75a81f4834 |
| nk_gamma1_ay4_az1 | 1 | (1, 4, 1) | (0, 0, 0) | 5748149fe36706cf5fa11a60f6072abd435477ad37b700b4423792bc2824bad7 |
| nk_gamma10_ay0.25_az1 | 10 | (1, 0.25, 1) | (5.9867, 4.9889, 6.1863) | 2e26409fd8be0ed6f8dc3c72bfd2f5235475c62181891b0e677723f8fd8e77c3 |
| nk_gamma10_ay4_az1 | 10 | (1, 4, 1) | (5.9867, 4.9889, 6.1863) | 3d9f96d3f2b12cef1d8acbd8cfc10ab01e701fd8909637cfee5ea253c0cde614 |
| nk_gamma100_ay0.25_az1 | 100 | (1, 0.25, 1) | (60.1657, 50.1381, 62.1712) | 2bd0ce0d0d4e9243e17fb50c616d5a9b2186f96a77856bfe130e40a19b4d72e0 |
| nk_gamma100_ay4_az1 | 100 | (1, 4, 1) | (60.1657, 50.1381, 62.1712) | a9c8d3af1cd91a7370b1e1b26f22be339d0dbd36190e7b1e34a32fe5b9d26d44 |

Velocity direction (0.6, 0.5, 0.62) normalized; magnitude `sqrt(gamma^2 - 1)`.

Generator command (per table; `generate.sh` runs all six serially):

    CUDA_VISIBLE_DEVICES=1 env -u LD_PRELOAD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python build_table.py --gamma <1|10|100> --dir 0.6 0.5 0.62 --cells 1 <0.25|4> 1 \
      --rmax 13 --h-near 0.2 --h-far 0.5 --r-near 4 \
      --order-near 12 --refine-near 3 --order-far 8 --batch 512 --out <dir>/nk

Validation command (worst-case interpolation error vs `indfield.py` at the
table's own order, off-grid points, in-support / face / shell out to 12 cells):

    python validate_table.py <dir>/nk_<tag>.bin

The binaries themselves are campaign artifacts and are not committed to the
repository; they live in the run directory
`/tmp/alpaka-orchestration/pr24-20261102/t150-impl/table/` (rosi_v100).
