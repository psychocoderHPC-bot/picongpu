# Neural bunch field model artifact, format v1 (repository interface contract)

Status: interface contract for the `PIC_USE_neuralBunch` feature. The Python
exporter and the C++ loader MUST implement this document. Changes require a
version bump.

## Files

Two files, same directory, referenced by a single path prefix:

- `<name>.json` - UTF-8 metadata, no comments.
- `<name>.bin` - flat little-endian tensor payload, row-major.

`<name>.json` contains `artifact.sha256` and `artifact.bytes` for `<name>.bin`.
A loader MUST verify byte count and SHA-256 before allocating tensor storage.

## Metadata schema (required keys)

```json
{
  "format": "picongpu-neural-bunch",
  "version": 1,
  "dtype": "float32",
  "endianness": "little",
  "created_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "code_sha": "<40 hex>",
  "tile": {
    "nx": 8, "ny": 8, "nz": 4,
    "components": ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
    "flatten": "c + 6*(i + nx*(j + ny*k))"
  },
  "features": {
    "kind": "fourier_v1",
    "octaves": [-3, 6],
    "inputs": ["r_x", "r_y", "r_z", "u_x", "u_y", "u_z", "gamma", "a_y", "a_z"],
    "dim": 69
  },
  "encoder": {
    "kind": "mlp",
    "activation": "silu",
    "layers": [
      {"weight": "enc.0.w", "bias": "enc.0.b", "in": 69, "out": 256},
      {"weight": "enc.1.w", "bias": "enc.1.b", "in": 256, "out": 256},
      {"weight": "enc.2.w", "bias": "enc.2.b", "in": 256, "out": 128}
    ],
    "output_dim": 128,
    "last_layer_activation": "none"
  },
  "decoder": {
    "kind": "geometry_bank_bilinear",
    "axes": {"a_y": [0.1, "...", 10.0], "a_z": [0.1, "...", 10.0]},
    "interpolation": "bilinear in log10(a), clamped at the outer anchors",
    "matrix_shape": [128, 1536],
    "bias": false,
    "bank": ["dec.0.0.w", "..."], "bank_entry_shape": [128, 1536]
  },
  "normalization": {
    "length": "dx",
    "charge": "q_i/e = -w_i",
    "field_E": "E = e/(4*pi*eps0*L^2) * E_tilde",
    "field_B": "B = e/(4*pi*eps0*c*L^2) * B_tilde",
    "latent": "none"
  },
  "domain": {
    "gamma": [1.0, 100.0],
    "a_y": [0.1, 10.0],
    "a_z": [0.1, 10.0],
    "d_min_cells": 4.0,
    "d_max_cells": 512.0,
    "distance_metric": "minimum over the 1536 staggered output points of |x_out - x_source(t)| / dx, with x_out in physical coordinates"
  },
  "reference": {
    "shape": "TSC",
    "shape_support_cells": [2.0, 2.0, 2.0],
    "integrator": "adaptive FP64 quadrature of the constant-velocity point kernel over the lab-frame shape",
    "description": "<short text>"
  },
  "fallback": {
    "policy": "per (source, tile, component-group): reference if distance < d_min_cells or > d_max_cells or gamma/aspect outside domain",
    "component_groups": {"E": "Ex,Ey,Ez", "B": "Bx,By,Bz"}
  },
  "validation": {
    "tolerance_field_rel_rms": 0.001,
    "tolerance_force_rel_rms": 0.01,
    "report": "validation_report.json",
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
  },
  "tensors": [
    {"name": "enc.0.w", "offset_bytes": 0, "count": 17664, "shape": [69, 256], "order": "row_major_in_out"},
    {"name": "enc.0.b", "offset_bytes": 70656, "count": 256, "shape": [256]}
  ],
  "artifact": {"file": "model.bin", "bytes": 1234567, "sha256": "<64 hex>"},
  "license": "MPL-2.0"
}
```

`validation.validated_band` is optional and is a v1 additive extension. When it
is present the loader requires finite values, a non-empty `source`, strictly
increasing intervals, non-negative measured errors, a force pass share in
`[0, 1]`, and a band contained in the `domain` bounds. When it is absent the
runtime disables the neural path and routes every pair to the reference kernel
(the production default for current artifacts).

Additional required rules:

- `tensors[].offset_bytes` must be 64-byte aligned, monotonically increasing,
  non-overlapping, and contained in `artifact.bytes`.
- All tensors are `float32` row-major. Encoder weight tensors are stored as
  `[in, out]` so that the forward pass is one row-major GEMM
  `Y[N,out] = X[N,in] * W[in,out]` plus bias. Biases are `[out]`.
- The decoder bank is ordered `index = ia_z * len(axes.a_y) + ia_y` with `a_y`
  varying fastest; each bank entry is `[latent, 1536]` (`in,out`), so the
  decoder is `F[1,1536] = z[1,latent] * W[latent,1536]`.
- Decoder has no bias and no activation. `F_tilde = W_interp(a_y, a_z) z`.
- No batch statistics, dropout, or input-order dependence may exist.
- Encoder input `r` is in `dx` units relative to the tile cell origin,
  `u = p/(m_e c)`, `gamma = sqrt(1+|u|^2)`, `a_y = dy/dx`, `a_z = dz/dx`.
- Features are exactly: the 9 `inputs` in the given order, followed by, for each
  axis in `[x, y, z]` and each octave `j` from `octaves[0]` to `octaves[1]`
  inclusive, `sin(pi*2^j*r_axis)` then `cos(pi*2^j*r_axis)`.
  Ordering: axis-major, then octave ascending, sin before cos.
- Disabled/unsupported combinations (wrong tile, non-3D, unlisted shape,
  non-finite data, out-of-range gamma/aspect) MUST be rejected by the loader
  with an actionable error, not silently clamped.

## Runtime contract

PIConGPU's FDTD stores E and B at the SAME integer time between steps
(E^{n+1} = E^n + c^2 dt curl B^{n+1/2}; B is split into two half updates and is
stored at integer times; during `updateBeforeCurrent` it is temporarily at
n+1/2). Therefore the initializer injects a single co-temporal set of all six
components at the simulation start time.

For one source and one tile:

1. Source position at the injection time: `x(t0) = x_file + beta*c*(t0-t_file)`
   (uniform motion, beta = u/gamma). For the shipped example `t_file == t0 == 0`.
2. Classify the (source, tile) pair with `domain.distance_metric` as `neural`
   or `reference`.
3. Neural: `z += (q/e) * h(features(r, u, a))`, then `F_tilde = W z`, then one
   exact physical scale conversion into PIConGPU internal units, then store all
   six components at their staggered positions.
4. Reference: FP64 shape-integrated kernel for all six components.
5. Each pair is evaluated exactly once; never both paths for the same pair.

## openPMD decoding rules (verified on PIConGPU PR#17 output)

Measured on a real PIConGPU series (`simData_000000.h5`, baseline run):

- `position`: raw in-cell fraction in [0,1); `unitSI` = cellSize per axis.
- `positionOffset`: raw integer cell index; `unitSI` = cellSize per axis.
- physical position = (position + positionOffset) * unitSI, per axis.
- `momentum`: macro momentum with `unitSI` = m_e*c (8.918582037699194e-20 for
  electrons), `macroWeighted = 1`, `weightingPower = 1`.
  single-electron `u = momentum * unitSI / (m_e*c * w^weightingPower)`.
  Never interpret the raw momentum as `u`.
- `weighting`: raw w, unitSI = 1, `macroWeighted = 1`, `weightingPower = 1`.
- `charge` / `mass`: constant records in SI (charge unit C, mass unit kg).
- Any reader must implement the additive position rule and the
  macroWeighted/weightingPower rule from the file metadata, not from a fixed
  assumption; a file whose `position` exceeds one cell while `unitSI` is a
  cell size is non-standard and must be rejected with an actionable error.
