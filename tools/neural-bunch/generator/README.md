# Deterministic external openPMD electron-bunch generator

Synthetic electron bunches written as openPMD 2.0 particle series for PIConGPU
external-particle input and independent I/O tests.  No PIConGPU code is used.

| file | purpose |
| --- | --- |
| `generate_bunch.py` | CLI generator (HDF5 or ADIOS2/BP5), validation, manifest, sidecar |
| `inspect_bunch.py` | read-back, PIConGPU-layout decode, validation, sidecar/manifest check, HDF5-vs-BP5 comparison |
| `decode_piconpu.py` | same reader path applied to a real PIConGPU HDF5 dump |
| `nb_common.py` | constants, readers, decode/validation, checksums |
| `examples.sh` | all 13 cases, both backends, end-to-end checks |

Tested with Python 3.12, openPMD-api 0.17.1.post4, numpy 2.5.4, h5py 3.16.0
and ADIOS2 2.13.0.  Set `PYTHON` to select the interpreter and `OUT` for the
generated-data directory; generated data stays outside the checkout.

## Layout (verified against PIConGPU PR#17 output)

Physical position, component-wise:

```
x_phys = (position_raw + positionOffset_raw) * unitSI
```

* `position`: in-cell fraction in `[0,1)`, float64, `unitSI = dx,dy,dz`,
  `unitDimension L=1`, `macroWeighted=0`, `weightingPower=0`.
* `positionOffset`: integer cell index, float64 (PIConGPU writes int32),
  `unitSI = dx,dy,dz`, `unitDimension L=1`, `macroWeighted=0`,
  `weightingPower=0`.
* `momentum`: macro momentum in units of `m_e*c`, float64, `unitSI = m_e*c`
  (`2.7309245307378233e-22 kg m/s`), `unitDimension L=1,M=1,T=-1`,
  `macroWeighted=1`, `weightingPower=1`.  Raw value is `u * w`.
  Single-electron `u = momentum_raw * unitSI / (m_e*c) / w^weightingPower
  = momentum_raw / w`.
* `weighting`: raw `w`, float64, `unitSI=1`, `macroWeighted=1`,
  `weightingPower=1`.
* `id`: uint64, `1..N`, `unitSI=1`, `macroWeighted=0`, `weightingPower=0`.
* `charge` / `mass`: constant records, `unitSI=1`,
  `unitDimension (0,0,1,1,0,0,0)` / `(0,1,0,0,0,0,0)`, values
  `-1.602176634e-19 C` and `9.1093837015e-31 kg`.
  Total charge `= charge_value * sum(w^weightingPower)`.
* `particlePatches`: one global patch; `numParticles=[N]`,
  `numParticlesOffset=[0]`, `offset=[0,0,0]`, `extent=[NX,NY,NZ]`,
  uint64, `unitSI=1`.  `--no-patches` omits it; species attributes
  `grid`, `cellSizeSI`, `numParticles` are always written.

### Measured difference to the PR#17 baseline

The baseline uses one constant weight `w = 326.577392578125` per particle and
writes `momentum_raw = u` with `unitSI = m_e*c*w = 8.918582037699194e-20`.
This generator stores `momentum_raw = u*w` with `unitSI = m_e*c`, which is the
general form for per-particle weights.  Both decode to the same `u` with
`u = raw*unitSI/(m_e*c)/w`.  For `w=1` (most cases here) `raw = u` and the
decoded values match the baseline; the numeric `unitSI` differs by the folded
weight.  The baseline also stores charge/mass in PIC-internal units
(`value=-0.003062`, `unitSI=5.2323e-17`); this generator stores SI values with
`unitSI=1`, so physical values are identical.

## Sidecar

Every run writes `<series>.physical.npz` next to the data, e.g.
`stationary.h5.physical.npz` and `stationary.bp5.physical.npz`, containing
`position` (decoded absolute metres, Nx3), `u` (Nx3), `weighting` (N) and
`id` (N).  Its bytes and SHA-256 are recorded in the manifest.  It exists only
for independent cross-checks: the simulation consumes the openPMD file, never
the npz.

## Quick start

```bash
cd tools/neural-bunch/generator
PY=${PYTHON:-python}
OUT=${OUT:-./neural-bunch-data}

$PY generate_bunch.py --output "$OUT/stationary" \
    --backend hdf5 --seed 1001 --model stationary --n 4096
$PY generate_bunch.py --output "$OUT/stationary" \
    --backend adios2 --seed 1001 --model stationary --n 4096
$PY inspect_bunch.py "$OUT/stationary.h5" \
    --compare "$OUT/stationary.bp5"

OUT="$OUT" bash examples.sh

# real PIConGPU file through the same reader
$PY decode_piconpu.py
```

`--output` gets `.h5` or `.bp5` appended; `%T` is replaced by the iteration.

## Models

Spatial distribution is a seeded Gaussian ellipsoid in cells,
`x_cells = centroid + sigma*z`, deterministically truncated into the domain.

| model | momentum |
| --- | --- |
| `stationary` | `u=0`, gamma=1 |
| `drift` | `u = gamma*beta*direction/|direction|` (`--beta`, `--direction`) |
| `gamma` | `|u| = sqrt(gamma^2-1)` (`--gamma`, `--direction`) |
| `counterstreaming` | half `+beta*axis`, half `-beta*axis` |
| `mixed` | JSON `--mix-spec '[{"fraction":f,"beta":[bx,by,bz],"u_sigma":s}, ...]'` |

`--u-sigma` adds isotropic spread in `u`; `--quiet` enables stratified
Latin-hypercube positions.  Weights: `uniform|linear|two-tier`
(`--weight-ratio` 2.5, `--weight-fraction` 0.4).  `--chunks K` writes every
component in `K` chunks.

### Cases in `examples.sh`

| case | seed | parameters | n | grid | centroid | sigma |
| --- | --- | --- | --- | --- | --- | --- |
| `stationary` | 1001 | stationary | 4096 | 64^3 | 32,32,32 | 6,6,6 |
| `beta001` | 1002 | drift beta=0.01 | 4096 | 64^3 | 32,32,32 | 6,6,6 |
| `beta01` | 1003 | drift beta=0.1 | 4096 | 64^3 | 32,32,32 | 6,6,6 |
| `gamma1` | 1013 | gamma=1 | 4096 | 64^3 | 32,32,32 | 6,6,6 |
| `gamma2` | 1004 | gamma=2 | 4096 | 64^3 | 32,32,32 | 6,6,6 |
| `gamma10` | 1005 | gamma=10 | 4096 | 64^3 | 32,32,32 | 6,6,6 |
| `gamma100` | 1006 | gamma=100 | 4096 | 64^3 | 32,32,32 | 6,6,6 |
| `offaxis` | 1007 | drift beta=0.05 dir=(0.6,0.8,0) | 4096 | 64^3 | 20,44,32 | 4,5,4 |
| `mixed` | 1008 | 0.5*(0.01,0,0), 0.3*(0,0.1,0), 0.2*(-0.05,0,0.02) | 4096 | 64^3 | 32,32,32 | 6,6,6 |
| `counterstreaming` | 1009 | beta=0.3 along x | 4096 | 64^3 | 32,32,32 | 6,6,6 |
| `unequal` | 1010 | two-tier weights 2.5 (40%) | 4096 | 64^3 | 32,32,32 | 6,6,6 |
| `tiny257` | 1011 | stationary | 257 | 32^3 | 16,16,16 | 3,3,3 |
| `chunks1025` | 1012 | drift beta=0.02, 3 chunks | 1025 | 64^3 | 32,32,32 | 6,6,6 |

All cell sizes are isotropic `1e-7 m` (domain `6.4e-6 m`).

## Manifest and reproducibility

`<series>.manifest.json` records parameters, seed, backend, iteration, species,
chunks, dtypes, units, validation, per-component `array_sha256`, constant
values, sidecar bytes/SHA-256, and a file manifest.  Arrays are bit-for-bit
reproducible for the same seed and parameters in both backends; container file
bytes (and the zip container of the npz) can differ because of embedded
timestamps.  `inspect_bunch.py` verifies manifest hashes, sidecar SHA-256 and
decoded sidecar contents.

## Validation

Before writing, generation aborts unless: count matches, all values finite,
weights positive, total weight and charge consistent, ids `1..N`, momentum
decodes to a physical gamma matching the monoenergetic target,
`positionOffset` is an integer cell index, `position` raw is in `[0,1)`, and
decoded physical positions are inside `[0, grid*cellSize)`.  If the Gaussian
cannot be truncated into the domain, generation aborts and writes nothing.

`inspect_bunch.py` additionally validates patch sums, unique ids, unitSI /
unitDimension / macroWeighted / weightingPower, constants, and compares HDF5 vs
BP5 arrays and metadata.

## openPMD limitations observed

1. openPMD 0.17.x `load_chunk` returns an undefined buffer until
   `series_flush()`; both readers flush after every load.
2. Constant records need extent `[N]`, not `[1]`, or openPMD rejects the
   species on read (`Inconsistent extents`).
3. PIConGPU reads patch offset/extent as uint64; openPMD 2.0 describes them as
   floating-point cell coordinates.  uint64 is written here.
4. BP5 defaults to group-based iteration encoding (warning about metadata
   growth); harmless for a single iteration.
