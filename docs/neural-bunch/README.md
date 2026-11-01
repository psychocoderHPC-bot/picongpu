# Neural bunch field initializer (`PIC_USE_neuralBunch`)

Status: **experimental. The default field initializer is the neural model.**
This document describes the feature as implemented on this branch and reports
the measured validation result, including the parts that fail the accuracy gate.

The feature loads an externally generated electron bunch, adds the fields of
that bunch at the start of the simulation, and inserts the particles. Fields
come from a learned model where its domain of validity holds and from an
independent FP64 reference kernel everywhere else.

## Build switch

```bash
cmake ... -DPIC_USE_neuralBunch=ON ...
```

`PIC_USE_neuralBunch` is an `AUTO` dependency: enabling it requires openPMD and
the pinned alpakaNN/alpakaVendor forks. The build defines
`PIC_ENABLE_NEURAL_BUNCH=1`. When the dependency is unavailable or the option is
`OFF`, the build defines `PIC_ENABLE_NEURAL_BUNCH=0`, adds no alpakaNN/BLAS
dependency, and the initializer stage compiles to nothing.

## Parameters (`include/picongpu/param/neuralBunch.param`)

| field | type | meaning |
| --- | --- | --- |
| `Species` (type alias) | PIConGPU species | Species that receives the particles. Must be an electron species whose name matches `species`. |
| `filename` | `std::string` | openPMD series path, e.g. `bunch.h5` or `bunch.bp5`. |
| `iteration` | `std::uint64_t` | Iteration read for both passes. |
| `species` | `std::string` | Particle species name inside the openPMD file. |
| `modelPath` | `std::string` | Artifact path prefix; the loader reads `<modelPath>.json` and `<modelPath>.bin`. |
| `chunkSize` | `std::uint64_t` | Global source particles streamed per collective chunk. |
| `tileBatch` | `std::uint32_t` | Tiles per inference batch (accepted; the current implementation evaluates one tile at a time). |
| `fieldInitializer` | enum | `neural` (default), `validatedNeural`, or `referenceOnly`. |
| `enableReferenceFallback` | `bool` | Under `neural`, route out-of-domain pairs to the exact reference (`true`, default) or fail collectively (`false`). |
| `timePolicy` | `int` | Must stay `0`: the source time equals the injection time, so no linear drift is applied. |

The initializer rejects unsupported configurations with an actionable error
instead of clamping: non-3D simulations, a non-electron species, a species name
mismatch, `enableReferenceFallback=false`, `timePolicy!=0`, and non-zero source
iteration times are all errors.

## Two passes

The stage runs from `Simulation::fillSimulation()` in the non-restart branch,
after `ParticleInit` and before the final field guard exchange.

1. **Field pass.** Rank 0 reads the openPMD series and broadcasts bounded global
   source chunks. Each rank enumerates its owned 8x8x4 tiles, classifies every
   (source, tile) pair with the artifact domain rules, and evaluates each pair
   exactly once:
   - *neural*: build the `fourier_v1` features, run the SiLU encoder, accumulate
     `z = sum_i (q_i/e) h_i` in FP32, decode `F_tilde = W_interp(a_y,a_z) z`,
     apply the one physical scale conversion, and add all six components at
     their staggered Yee positions.
   - *reference*: FP64 shape-integrated TSC kernel on the host. This path is
     used when the distance is below `d_min_cells` or above `d_max_cells`, or
     when gamma or either aspect is outside the trained domain.
   Fields are zero (apart from independent backgrounds) at this point, so the
   store uses `Add` semantics. A pair is never evaluated by both paths.
2. **Particle pass.** Reopens the same series and iteration, streams the same
   chunks, and inserts only particles inside the rank's half-open local domain.
   `position` and `positionOffset` are added before the `unitSI` conversion;
   `u = p_SI/(m_e c)` is reconstructed from `momentum`, `unitSI`,
   `macroWeighted` and `weightingPower`. IDs are preserved or materialized so
   they stay globally unique. The stage verifies count, total weight, total
   charge and momentum against the file.

Per-rank and total statistics are logged: pair counts by route and reason,
I/O/broadcast/feature/encoder/reduce/decode/store/reference timings, decoded
bytes, max latent magnitude, field sums and non-finite counts.

## Validated band (MODEL-FORMAT v1 additive extension)

The model metadata may carry an optional object:

```json
"validation": {
  "validated_band": {
    "d_min_cells": 8.0,
    "d_max_cells": 64.0,
    "gamma_min": 2.0,
    "gamma_max": 50.0,
    "a_y_min": 0.5,
    "a_y_max": 2.0,
    "a_z_min": 0.5,
    "a_z_max": 2.0,
    "measured_max_rel_rms_e": 0.001,
    "measured_max_rel_rms_cb": 0.001,
    "measured_min_force_pass_share": 0.99,
    "source": "validation/report-id"
  }
}
```

The loader rejects non-finite values, empty `source`, non-increasing intervals,
negative measured errors, a force pass share outside `[0, 1]`, and any band that
is not contained in the artifact `domain` bounds. The band only changes
behaviour under `fieldInitializer = validatedNeural`. The default `neural`
policy uses the wider artifact `domain` and warns that the result is not
production-validated. To run a validated model, add this block and select
`validatedNeural`; nothing else changes.

### Measured architecture limit

Every architecture-compatible encoder tested (MLP with 16M parameters, SIREN,
RBF and output-position probes) plateaus at validation strong-field E relative
RMS 0.11-0.17 against the 1e-3 acceptance gate. The oracle basis reconstruction
reaches 1.6e-7, so the limit is the learned encoder, not the feature basis.
Consequently no validated band exists for any current artifact; the default
`neural` policy uses the model anyway and warns, while `referenceOnly` and
`validatedNeural` remain available for safe operation.

## Model artifact

The model is a `model.json` + `model.bin` pair defined by
[`tools/neural-bunch/model-format.md`](../../tools/neural-bunch/model-format.md).
The loader verifies `artifact.bytes` and `artifact.sha256` collectively across
ranks before allocating tensors, then validates the tensor table (64-byte
aligned, monotonic, non-overlapping), the tile geometry, the feature kind and
the domain metadata.

The validated artifact used for the measurements below:

- `fourier_v1`, 8x8x4 tile, 69 features, latent 512, hidden `[512,512]`
- 80,887,808 bytes
- sha256 `fcf0b28dbb62e597f864218746ee381a0e3ff0c78354dcf9f78c58f5f0949d8d`
- loader check:
  `LOAD_OK format=picongpu-neural-bunch version=1 latent=512 tile=8x8x4 featureDim=69 sha=fcf0...9d8d bytes=80887808`

This artifact is not committed to the repository.

## Supported domain

- 3D simulations only.
- gamma in `[1, 100]`.
- cell aspects `a_y = dy/dx` and `a_z = dz/dx` in `[0.1, 10]`, including
  anisotropic cells (`a_y != a_z`).
- distance metric: minimum over the 1536 staggered output points of
  `|x_out - x_source(t)| / dx`, with `d_min_cells = 4` and
  `d_max_cells = 512`.
- TSC particle shape, support 3 cells per axis.
- openPMD series in HDF5 or ADIOS2 (BP5). The reader implements the additive
  `position + positionOffset` rule and the `macroWeighted`/`weightingPower`
  rule; it rejects files whose `position` exceeds one cell while `unitSI` is a
  cell size.
- The simulation cell size is taken from the PIConGPU grid; there is no cubic
  cell assumption.

## Restart behaviour

The stage is called only from the non-restart branch of `fillSimulation()`, so
a restart never runs it and never adds fields or particles twice. Measured on a
checkpoint at step 2: no `neuralBunch` log lines after restart, initialization
1.305 s instead of 77.54 s for a full hybrid init, and restart step-3 fields
bitwise identical to the original run (E and cB relative RMS 0). Particle
count, weight and charge are unchanged. Particle ordering may change after
restart, so element-wise per-particle comparisons are not meaningful.

## Feature-off behaviour

With `PIC_USE_neuralBunch=OFF` or a missing dependency, the translation unit
compiles to nothing, `-DPIC_ENABLE_NEURAL_BUNCH=0` is defined, and no
alpakaNN/alpakaVendor targets are linked. The rest of PIConGPU is unaffected.
There is also a validation-only environment switch,
`NEURAL_BUNCH_FORCE_REFERENCE=1`, which routes every non-zero-weight pair to
the reference kernel. It exists to produce the all-reference comparison and is
not a production option.

## STATUS (measured truth)

Validation is described in full in `validation/REPORT.md` on the development
worktree; the numbers below are that report's measured values. Host: 2 GPUs,
CUDA 12.5.82, 32x32x16 grid, stationary n=16 bunch, 10 steps.

**Hybrid vs all-reference, step 0**

- E relative RMS **3.17e-1** (mean abs 2831 V/m, p99 1.330e4 V/m).
- Force relative RMS **1.23** (reference RMS 4.43e-15 N); force uses a
  trilinear gather, not the PIConGPU gather.
- cB relative RMS is **undefined at step 0** because the reference B is exactly
  zero. At steps 1/2/10 the cB relative RMS is 0.885 / 0.832 / 1.004.
- Hybrid init wall time 77.54 s, of which reference fallback 74.35 s and
  inference 0.117 s. All-reference init wall time 152.8 s. **The reference
  fallback dominates initialization cost**, because every source is in support
  for its own tile (860 neural / 164 reference pairs, 16.0% neural coverage).
- The run is stable over 10 steps, finite, and preserves particle count, total
  weight and total charge. That demonstrates the pipeline, not physics
  accuracy.

**Model-only accuracy (the gate is not met)**

- Per-case E mean relative RMS **0.31** on the strong tiles.
- The acceptance gate is **1e-3**, so the model-only path **does not meet the
  gate**.
- Rank, convention and data issues were ruled out as causes.
- The encoder in-sample floor is about **0.24**.

**Conclusion**

The pipeline is stable, restart-safe, rank-local and preserves
count/weight/charge. The default `neural` policy exercises the trained model
inside the artifact domain and the exact reference elsewhere, but no shipped
artifact carries a validated band, so the output is **not a physics pass**.
Use `validatedNeural` for banded operation or `referenceOnly` for the exact
path. The dominant runtime cost is the reference fallback. Not measured: peak
host/device memory. Not run: n=64 hybrid, BP5 end-to-end, and all-reference
gamma=10 over 10 steps.

## PICMI usage

The initializer is available to PICMI simulations through
`picmi.NeuralBunchInitializer`. Attach it to the simulation with
`picongpu_neural_bunch`; the destination species is passed to the initializer
object itself.

```python
import picmi

electrons = picmi.Species(particle_type="electron", name="electrons")

initializer = picmi.NeuralBunchInitializer(
    filename="bunch.h5",           # or bunch.bp5
    iteration=0,
    source_species="e",            # species name inside the series
    model_path="model",            # <model_path>.json and .bin
    chunk_size=1 << 20,
    tile_batch=1,
    field_initializer="referenceOnly",  # neural | validatedNeural | referenceOnly
    enable_reference_fallback=True,
    time_policy=0,
    species=electrons,
)

sim = picmi.Simulation(
    max_steps=2,
    solver=picmi.ElectromagneticSolver(grid=grid, method="Yee"),
    species=[electrons],
    layouts=[None],
    picongpu_neural_bunch=initializer,
)
```

PICMI derives the cell geometry `dx`/`dy`/`dz` from the grid and the Yee field
layout from the solver. The generated `neuralBunch.param` maps the destination
species to `PIC_<species>` (an alias of the generated C++ species) and carries
the requested values. `filename` and `model_path` may be relative to the
generated setup directory; they are resolved there and the original inputs plus
artifact digests are stored in the setup metadata (`pypicongpu_rendering_context.json`).

Options mirror the C++ parameters:

| PICMI field | C++ field | default | meaning |
| --- | --- | --- | --- |
| `filename` | `filename` | required | openPMD series path |
| `iteration` | `iteration` | `0` | iteration inside the series |
| `source_species` | `species` | `"e"` | particle species inside the series |
| `model_path` | `modelPath` | required unless `referenceOnly` | artifact path prefix |
| `chunk_size` | `chunkSize` | `1048576` | source particles per collective chunk |
| `tile_batch` | `tileBatch` | `1` | tiles per inference batch |
| `field_initializer` | `fieldInitializer` | `"neural"` | `neural`, `validatedNeural`, `referenceOnly` |
| `enable_reference_fallback` | `enableReferenceFallback` | `True` | reference outside the model domain |
| `time_policy` | `timePolicy` | `0` | must stay `0` |

Validation rejects non-3D grids, non-electron destination species, particle
shapes other than TSC (`"quadratic"`), field solvers without the Yee layout
(`other:None`), a second initializer for the same species (a density
`initial_distribution` or a density operation), missing series/artifact files at
render time, and `time_policy != 0`.

Build with `-DPIC_USE_neuralBunch=ON`. The generated parameter file exists for
every PICMI setup, but PIConGPU runs the stage only when the feature is
compiled in, so only setups that actually configure `picongpu_neural_bunch`
should use that build. See
[`lib/python/examples/NeuralBunch/`](../../lib/python/examples/NeuralBunch/) for a
runnable HDF5/ADIOS2 example.

## Related tooling

Reference implementation, generator, training code, golden vectors and the
reproducibility contract are in
[`tools/neural-bunch/`](../../tools/neural-bunch/README.md).
