# Neural bunch field initializer (`PIC_USE_neuralBunch`)

Status: **experimental and disabled by default.** When a setup enables it, the
default field policy is `banded`: the analytic point field for the mid and far
bands and the exact reference for the near band. A standard PIConGPU setup that
does not set `enabled = true` is unchanged; the KHI and LWFA setups were run
with the feature compiled in and the stage disabled, and produced their usual
output. The learned model remains selectable but is not the default and does not
meet the 1e-3 accuracy gate (see below).

The feature loads an externally generated electron bunch, adds the fields of
that bunch at the start of the simulation, and inserts the particles. In the
default policy the field comes from the analytic point field away from the
sources and from the exact FP64 reference next to them.

## Build switch

```bash
cmake ... -DPIC_USE_neuralBunch=ON ...
```

`PIC_USE_neuralBunch` is an `AUTO` dependency: enabling it requires openPMD and
the pinned alpakaNN/alpakaVendor forks. The build defines
`PIC_ENABLE_NEURAL_BUNCH=1`. When the dependency is unavailable or the option is
`OFF`, the build defines `PIC_ENABLE_NEURAL_BUNCH=0`, adds no alpakaNN/BLAS
dependency, and the initializer stage compiles to nothing.

## Default semantics

- `enabled = false` by default. The stage returns immediately: no file read, no
  model load, no collective calls, no field or particle changes. Any hand-written
  setup that wants the feature must set `enabled = true`; PICMI does so only when
  `picongpu_neural_bunch` is configured.
- When the feature is enabled and `fieldInitializer` is left unset, it resolves
  to `banded`.
- `neural`, `validatedNeural` and `referenceOnly` remain selectable. `neural` is
  kept for compatibility and is not the default. `validatedNeural` uses a model
  only inside an artifact `validation.validated_band`; no shipped artifact
  carries one. `referenceOnly` is the exact FP64 path and needs no model
  artifact.

## Parameters (`include/picongpu/param/neuralBunch.param`)

| field | type | meaning |
| --- | --- | --- |
| `enabled` | `bool` | Master switch, default `false`. |
| `Species` (type alias) | PIConGPU species | Species that receives the particles. Must be an electron species whose name matches `species`. |
| `filename` | `std::string` | openPMD series path, e.g. `bunch.h5` or `bunch.bp5`. |
| `iteration` | `std::uint64_t` | Iteration read for both passes. |
| `species` | `std::string` | Particle species name inside the openPMD file. |
| `modelPath` | `std::string` | Artifact path prefix; the loader reads `<modelPath>.json` and `<modelPath>.bin`. Not needed for `referenceOnly` or `banded` without a model. |
| `chunkSize` | `std::uint64_t` | Global source particles streamed per collective chunk; also bounds the tile-wide source scratch. |
| `tileBatch` | `std::uint32_t` | Tiles whose accumulated values one store kernel writes. |
| `fieldInitializer` | enum | Unset means `banded` when enabled. `neural`, `validatedNeural`, `referenceOnly`, `banded` are selectable. |
| `enableReferenceFallback` | `bool` | Under `neural`, route out-of-domain pairs to the exact reference (`true`, default) or fail collectively; must stay `true`. |
| `bandedNearMaxCells` | `double` | `banded` near-band limit in cells, default `4`. |
| `bandedFarMinCells` | `double` | `banded` far-band start in cells, default `48`. |
| `useModelForNear` | `bool` | `banded`: use the model for near pairs instead of the exact reference. Default `false`; enabling it warns because the near band was never validated. |
| `useModelForMid` | `std::optional<bool>` | `banded`: use the primary encoder/decoder for mid pairs. Default off (analytic point field only). |
| `useMidCorrection` | `bool` | `banded`: add the learned `mid_correction` head. Default off; measured worse than the point field alone. |
| `useExactDevice` | `bool` | Evaluate near and mid with the FP64 device quadrature instead of the host reference or the point field. Default `false`. |
| `bandedExactMaxCells` | `double` | Device-quadrature distance cutoff, default `128` cells. `<= 0` selects aspect-scaled limits: 48 cells up to aspect 2, 512 cells at aspect 10. |
| `tileWideField` | `bool` | Use the tile-centric device kernel for the analytic near/mid/far point field. Default `true`. |
| `tileWideBandStats` | `bool` | Count near/mid/far pairs on the device. Default `true`; set `false` on production runs. |
| `timePolicy` | `int` | Must stay `0`: the source time equals the injection time, so no linear drift is applied. |

The initializer rejects unsupported configurations with an actionable error
instead of clamping: non-3D simulations, a non-electron species, a species name
mismatch, `enableReferenceFallback=false`, `timePolicy!=0`, and non-zero source
iteration times are all errors.

## Band semantics (`banded`, the enabled default)

- **Near** (`d < bandedNearMaxCells`, default 4 cells): the exact FP64 reference
  on the host, or the FP64 device quadrature when `useExactDevice` is set. The
  model is used only when `useModelForNear` is explicitly enabled and a near or
  primary model is present.
- **Mid** (`4 .. 48` cells): the analytic point field. The primary model and the
  learned `mid_correction` head are opt-in and both were measured worse than the
  point field alone.
- **Far** (`d >= bandedFarMinCells`, default 48 cells): the analytic point field.
  An optional `far_multipole` Taylor head is opt-in and is not used by default.

Distances are the MODEL-FORMAT minimum over the 1536 staggered tile outputs,
measured in `dx` units. `a_y` and `a_z` come from the grid; there is no cubic
cell assumption.

### Tile-centric device kernel (`tileWideField`)

With the default `tileWideField = true` and no near/mid model, correction or
exact quadrature, the whole source-tile loop runs on the device. Each source
chunk is uploaded once (`pos/dx`, `u`, `q/e`); one block owns one tile and 256
staggered outputs, so a tile uses six blocks (1536 threads). Every thread walks
the chunk sources in ascending order, which fixes the accumulation order for a
given chunk size. The kernel evaluates the analytic point field, applies the
near threshold on the device, and records near/mid/far counters.

The tile kernel computes the point field in every band, so near pairs at or
inside `bandedNearMaxCells` are skipped rather than evaluated exactly. Use the
legacy banded path (disable `tileWideField`, or select `useExactDevice`) when
the near band must be exact.

The tile-wide path is disabled automatically when `useExactDevice`,
`useModelForNear`, `useModelForMid`, `useMidCorrection` or the validation-only
`NEURAL_BUNCH_FORCE_REFERENCE` environment switch is set. It is also disabled by
any policy other than `banded`. The legacy path then classifies every
(source, tile) pair on the host and keeps the exact near reference.

## Two passes

The stage runs from `Simulation::fillSimulation()` in the non-restart branch,
after `ParticleInit` and before the final field guard exchange, but only when
`enabled = true`.

1. **Field pass.** Rank 0 reads the openPMD series and broadcasts bounded global
   source chunks. Each rank enumerates its owned 8x8x4 tiles and adds the field
   of every source once:
   - *banded*: the composition above. Exact device quadrature uses fixed-order
     tensor Gauss on graded TSC sub-panels for smooth outputs and a sub-box
     decomposition for in-support outputs.
   - *neural/validatedNeural*: the `fourier_v1` features, the SiLU encoder and
     the latent reduction `z = sum_i (q_i/e) h_i` in FP32, then the decoder
     `F_tilde = W_interp(a_y,a_z) z`, the one physical scale conversion, and all
     six components at their staggered Yee positions.
   - *referenceOnly*: the FP64 shape-integrated TSC kernel on the host.
   Fields are zero (apart from independent backgrounds) at this point, so the
   store uses `Add` semantics. A pair is never evaluated by two paths.
2. **Particle pass.** Reopens the same series and iteration, streams the same
   chunks, and inserts only particles inside the rank's half-open local domain.
   `position` and `positionOffset` are added before the `unitSI` conversion;
   `u = p_SI/(m_e c)` is reconstructed from `momentum`, `unitSI`,
   `macroWeighted` and `weightingPower`. IDs are preserved or materialized so
   they stay globally unique. The stage verifies count, total weight, total
   charge and momentum against the file.

Per-rank and total statistics are logged: pair counts by route and reason,
I/O/broadcast/feature/encoder/reduce/decode/store/reference/quadrature timings,
decoded bytes, max latent magnitude, field sums and non-finite counts.

## All-GPU status

In the default banded configuration every field computation runs on the
accelerator: feature construction, encoder and decoder, band accumulation into
one `float[ntiles,1536]` device buffer, the point-field/quadrature kernels, and
the store kernel that applies the E/B normalisation directly into the PIConGPU
`FieldE`/`FieldB` device data boxes. The stage performs one `wait(queue)` after
all chunks and bands are enqueued, then one `synchronize()` per field to refresh
the host mirror for the final diagnostics. There is no per-tile or per-source
queue wait and no device-to-host copy in the field path.

The host performs only: MPI and chunk broadcast, openPMD decoding, model
load and hash verification, source-coordinate upload, and particle frame
bookkeeping (frame allocation, `IdProvider::getNewIdHost`,
`splitIntoListOfFrames`). The legacy near reference and the diagnostic tile
recompute still evaluate FP64 on the host when those options are selected.

`chunkSize` bounds the uploaded source scratch; 65536 is a safe value when the
particle heap has already claimed most of the device memory.
`tileWideBandStats = false` skips the device distance pass that only feeds the
statistics.

## Measured performance

| configuration | measurement |
| --- | --- |
| Analytic point field | 5.2 ns per pair |
| Tile-centric kernel, probe | 5.71e8 pairs/s on 4 V100 |
| 1M particles, 32x32x16 grid | 17.6 s initialization |
| 1M particles, 256x1024x256 grid | partial run stopped at >10.7 min (allocation ended); measured aggregate <=3.7e8 pairs/s; 12-15 min projected |
| Persistent cuBLAS handle | 310 us -> 5 us per batched decoder call |
| Host per-pair classification | removed from the default path |

The large-grid result is partial: the run ended after the device allocation
ended, not at completion, so the 12-15 min figure is a projection from the
measured aggregate rate.

## Measured accuracy

The preregistered metric is `S0 = 16 * max_cell ||G||` taken from the same
reference tile, with per-case relative RMS `||pred - ref|| / max(||ref||, S0)`,
worst of E and B (`research/BANDS.md`). It measures accuracy against the field
scale of the reference configuration, not against the local field. The
per-point slices below use `||pred - ref|| / max(||ref||, 1 % of the RMS
reference magnitude)` per staggered output, vector norm over the three E or cB
components, on two full-resolution planes and one stride-2 plane.

**Analytic point field, covered outputs (per-point relative error, E):**

| gamma | per-point mean | per-point median |
| --- | --- | --- |
| 2 | 5.7e-4 | 3.5e-4 |
| 10 | 4.1e-3 | 1.1e-3 |
| 100 | 1.6e-1 (16 %) | 7.5e-2 |

Under the S0 metric the point field clears 1e-3 for aspect 1 from `d = 4` at
gamma 1 and from `d = 32` at gamma 10/100 (axial and diagonal). For aspect <= 2
the anisotropic cases need 32-256 cells depending on direction, and aspect 10 on
the diagonal never clears 1e-3 up to 512 cells (1.8e-3 at `d = 512`). The
per-point values are harsher because they divide by the local field: gamma 2 and
gamma 10 stay at or near the gate for typical outputs, gamma 100 does not.

The gamma dependence is physical. The target is the TSC shape integrated over a
3-cell lab-frame support. At gamma = 100 the point field is a pancake whose
transverse thickness at distance `r` is about `r/gamma`, comparable to the
3-cell shape already at the box corners, so the shape integral differs from the
point field by percent-level amounts per source. Both the analytic field and the
FP64 reference are correct; the point field is simply a poor approximation of
the convolved shape there.

**Exact device quadrature:** agrees with the host FP64 reference to 4.9e-7 max
relative error. On the host the same reference costs 5.7 us per smooth pair and
7.5 ms per in-support pair, which is why the near band is expensive and why the
device port exists. `useExactDevice` runs the near and mid bands through it.

**Learned model:** about 15 % error, i.e. 0.11-0.17 strong-field E relative RMS
for every architecture-compatible encoder tried. The best probe artifact reaches
0.36-0.78 per-point on covered g10 outputs; the end-to-end hybrid measured 0.31.
It is not the default.

**Force at particles is not covered by the point field.** None of the trilinear
stencil points around a particle is 4 cells from every source (median nearest
distance 0.8 cells). A force-capable initializer needs the exact shape
quadrature; do not use the point-field bands for force validation.

## The learned 1e-3 gate is not met

No learned artifact reaches the 1e-3 acceptance gate and none is shipped with a
validated band. The measured architecture floor is 0.11-0.17 for MLP, SIREN, RBF
and output-position probes alike, against an oracle basis reconstruction of
1.6e-7. That gap locates the limit in the encoder, not in the feature basis: the
point-field feature family does not carry the TSC shape integral that the target
requires. The learned mid correction fails as well (held-out 0.044 mean vs
0.030 for the point field alone, and no run reaches 1e-3). The learned path is
therefore an experimental, explicitly selectable option; the default runtime
path is analytic plus exact quadrature.

## Plots and cost artifacts

The figures, cost tables and per-slice metrics quoted above live in the
development artifacts outside this repository:

- `/tmp/alpaka-orchestration/neural-bunch/plots-y/` (per-point slices, Figures)
- `/tmp/alpaka-orchestration/neural-bunch/plots-exp/`
- `/tmp/alpaka-orchestration/neural-bunch/mid-exp/`
- `/tmp/alpaka-orchestration/neural-bunch/research/` (`BANDS.md`, thresholds)
- `/tmp/alpaka-orchestration/neural-bunch/validation/`

They are development outputs and are not committed.

## Model artifact

The model is a `model.json` + `model.bin` pair defined by
[`tools/neural-bunch/model-format.md`](../../tools/neural-bunch/model-format.md).
The loader verifies `artifact.bytes` and `artifact.sha256` collectively across
ranks before allocating tensors, then validates the tensor table (64-byte
aligned, monotonic, non-overlapping), the tile geometry, the feature kind and
the domain metadata.

The artifact used for the measurements above:

- `fourier_v1`, 8x8x4 tile, 69 features, latent 512, hidden `[512,512]`
- 80,887,808 bytes
- sha256 `fcf0b28dbb62e597f864218746ee381a0e3ff0c78354dcf9f78c58f5f0949d8d`
- loader check:
  `LOAD_OK format=picongpu-neural-bunch version=1 latent=512 tile=8x8x4 featureDim=69 sha=fcf0...9d8d bytes=80887808`

This artifact is not committed to the repository.

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
behaviour under `fieldInitializer = validatedNeural`. No current artifact carries
a validated band; `banded` does not need one.

## Supported domain

- 3D simulations only.
- Artifact domain: gamma in `[1, 100]`.
- Cell aspects `a_y = dy/dx` and `a_z = dz/dx` in `[0.1, 10]`, including
  anisotropic cells (`a_y != a_z`).
- Distance metric: minimum over the 1536 staggered output points of
  `|x_out - x_source(t)| / dx`.
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
the host reference kernel and disables the tile-wide path. It exists to produce
the all-reference comparison and is not a production option.

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
    # model_path="model",          # only for neural/validatedNeural
    chunk_size=1 << 20,
    tile_batch=1,
    field_initializer="banded",    # banded (default) | neural | validatedNeural | referenceOnly
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
artifact digests are stored in the setup metadata
(`pypicongpu_rendering_context.json`).

Options mirror the C++ parameters:

| PICMI field | C++ field | default | meaning |
| --- | --- | --- | --- |
| `filename` | `filename` | required | openPMD series path |
| `iteration` | `iteration` | `0` | iteration inside the series |
| `source_species` | `species` | `"e"` | particle species inside the series |
| `model_path` | `modelPath` | required only for `neural`/`validatedNeural` | artifact path prefix |
| `chunk_size` | `chunkSize` | `1048576` | source particles per collective chunk |
| `tile_batch` | `tileBatch` | `1` | tiles per store batch |
| `field_initializer` | `fieldInitializer` | `"banded"` | `banded`, `neural`, `validatedNeural`, `referenceOnly` |
| `enable_reference_fallback` | `enableReferenceFallback` | `True` | reference outside the model domain |
| `time_policy` | `timePolicy` | `0` | must stay `0` |

Validation rejects non-3D grids, non-electron destination species, particle
shapes other than TSC (`"quadratic"`), field solvers without the Yee layout
(`other:None`), a second initializer for the same species (a density
`initial_distribution` or a density operation), missing series/artifact files at
render time, and `time_policy != 0`.

Build with `-DPIC_USE_neuralBunch=ON`. The generated parameter file emits
`enabled = true` when `picongpu_neural_bunch` is configured and
`enabled = false` otherwise. The file exists for
every PICMI setup, but PIConGPU runs the stage only when the feature is
compiled in, so only setups that actually configure `picongpu_neural_bunch`
should use that build. See
[`lib/python/examples/NeuralBunch/`](../../lib/python/examples/NeuralBunch/) for a
runnable HDF5/ADIOS2 example.

## Related tooling

Reference implementation, generator, training code, golden vectors and the
reproducibility contract are in
[`tools/neural-bunch/`](../../tools/neural-bunch/README.md).
