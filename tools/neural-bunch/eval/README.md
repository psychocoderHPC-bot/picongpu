# Frozen field-gate metric

`field_gate.py` evaluates the preregistered accuracy gate that PR #24 must be
judged against. It turns the symbolic formula in the agent guide into a runnable
check and records every ingredient (norms, `S0`, denominator, pass/fail) in a
JSON report.

## The gate

For each physical case and each component group `G` in `{E, cB}`:

```
epsilon_G = ||A_candidate - A_reference||_2 / max(||A_reference||_2, S0_G)
gate:     epsilon_G <= 1e-3
```

The guide states the formula at `picongpu-scalable-field-initialization-agent-guide.md:490-497`
(also `:805`) and `issues.md:15-16`, `issues.md:33`.

### What `A` is

* Point set: the 256 spatial output points of one `8 x 8 x 4` tile.
* Yee positions: at each output point the group is the three staggered
  components of that group, at their own half-cell offsets, not three
  co-located samples. The offsets are `Ex=(1/2,0,0)`, `Ey=(0,1/2,0)`,
  `Ez=(0,0,1/2)`, `Bx=(0,1/2,1/2)`, `By=(1/2,0,1/2)`, `Bz=(1/2,1/2,0)` in cell
  units, pinned in `tools/neural-bunch/reference/conventions.md:40-57`,
  `tools/neural-bunch/reference/geometry.json:24-48`, and verified against
  PIConGPU `YeeCell.hpp:80-82,124-126`.
* Vector: for group `E`, `A = (Ex,Ey,Ez)` at all 256 points, 768 scalars; for
  group `cB`, `A = (cBx,cBy,cBz)` at all 256 points. Canonical layout is
  `[i,j,k,component]` with component fastest and flatten order
  `c + 6*(i + 8*(j + 8*k))` (`tools/neural-bunch/model-format.md:26-29`).
  The norm is invariant to point ordering, so the report also accepts flat
  `(n,1536)` arrays.
* Norm: unweighted Euclidean (L2) over those 768 scalars. No cell-volume
  weight, no per-point normalization, no division by `sqrt(N)`.
* `cB` means the normalized tile magnetic field. In the tile convention the
  stored `B_tilde` already carries the factor `c`
  (`tools/neural-bunch/model-format.md:57-61`), so `cB_tilde == B_tilde`.
  Inputs must already be in this dimensionless normalized convention.
* `||A_reference||_2` is the norm of the reference over the same 768-scalar
  point set of the same case. `S0_G` is likewise per case and per group.
* `E` and `cB` are reported separately; each is a separate pass/fail. There is
  no combined or "worst of" decision in this report, so a caller can apply the
  stricter requirement explicitly. The historical notes often quote the worst
  of E/cB (`tools/neural-bunch/README.md:35-36`).

The comparison is `<=`, as written in the guide. Equality passes.

Near-zero policy: when both `||A_reference||` and `S0_G` are exactly zero, the
denominator is zero. The report then sets `zero_denominator: true`; an exactly
zero candidate gets `relative_l2 = 0.0` and passes, while any nonzero candidate
gets `relative_l2: null` and fails. No epsilon is added to the denominator.
This is the explicit policy the guide asks for at
`picongpu-scalable-field-initialization-agent-guide.md:498` and
`issues.md:251-254` (stationary B).

## What `S0` is

`S0` is not a universal numerical constant. It is a per-case, per-group scale
reconstructed from the reference tile:

```
S0_G = sqrt(N_points) * max_point ||G_point||_2,   N_points = 256
S0_G = 16 * max_cell ||G_cell||
```

`G_point` is the 3-component vector at one staggered output point, so the max is
over the 256 point norms per group. This is the definition used by the training
tooling, not an inference:

* `tools/neural-bunch/training/neuralbunch/reference_data.py:92-105` implements
  `preregistered_scales`: `sqrt(e.size/3) * max ||e_point||` with
  `e.size/3 = 256`, hence the factor 16.
* `tools/neural-bunch/training/README.md:123-139` documents
  `S0_G = sqrt(N_G/3) * max_cell ||G_cell||`, `N_G = 768`, computed before
  training from the reference configuration only.
* `tools/neural-bunch/README.md:35-36` and `docs/neural-bunch/README.md:179-181`
  quote the compact form `S0 = 16 * max_cell ||G||`.
* The dataset stores `s0` per sample/per group
  (`tools/neural-bunch/training/neuralbunch/reference_data.py:153,172,281`).

### Units and scaling

`S0_G` has the same units as `A_G`, i.e. the dimensionless normalized tile
field `E_tilde` and `cB_tilde` of `model-format.md:57-61`
(`E_tilde = E * 4*pi*eps0*dx^2/e`, `B_tilde = B * 4*pi*eps0*c*dx^2/e`). It is
therefore independent of source charge and of `dx` in those normalized units,
but it is *not* a fixed number: it changes with the reference configuration.
For a spatially uniform group field it equals `||A_reference||`; for a peaked
field it is larger, which is exactly the intended floor for weak or cancelling
cases (`reference_data.py:93-102`). Because the factor is `sqrt(256)`, adding
more output points of the same character does not silently change the RMS
meaning; the guide asks for this invariance at
`picongpu-scalable-field-initialization-agent-guide.md:496`.

### Historical use: loss and metric, both

The evidence shows two uses:

1. **Metric denominator** (the acceptance gate):
   `tools/neural-bunch/training/neuralbunch/train.py:178-196` computes
   `denom = max(||A_ref||, S0)` and `rel = ||err||/denom`; the same block labels
   `||A_ref|| < S0` cases `absolute_normalized`.
2. **Loss denominator**: `train.py:138-153` optimizes
   `||A_pred-A_ref||^2 / max(||A_ref||,S0)^2`; `training/tests/test_metrics.py:8-43`
   pins that the loss and metric share the preregistered scale.

A later *training-only* stabilization added a separate absolute floor, first
`0.05` then `5e-2`, giving `max(||A||, S0, floor)` in the loss only
(`tools/neural-bunch/training/neuralbunch/stable_train.py:4-5,74-88`,
`scripts/run_stable.py:44`). That floor is explicitly not part of evaluation
(`stable_train.py:334`). The guide's account is at
`picongpu-scalable-field-initialization-agent-guide.md:874-878`. The frozen
gate therefore uses only `max(||A_ref||, S0)`, never the `0.05` floor.

## Frozen configuration

The config is JSON and **must** declare `s0`. There is no default: a missing or
unknown `s0` block is a hard error, so no accuracy number can be produced with
an undeclared floor.

```json
{
  "gate": 1e-3,
  "s0": {"mode": "reference_tile"},
  "case_ids": ["optional", "per-case labels"]
}
```

Supported modes:

| `s0.mode` | meaning |
| --- | --- |
| `reference_tile` | recompute the historical `S0_G` from the reference dump (frozen definition above) |
| `values` | explicit `values: [[S0_E, S0_cB], ...]`, one pair per case |
| `constant` | one pair `{"E": ..., "cB": ...}` for every case |
| `manifest` | read the `s0` array (shape `(n_cases,2)`) stored in the reference `.npz` |

`gate` defaults to `1e-3`, but a frozen config should state it explicitly.

## Inputs and usage

`candidate` and `reference` may be `.npz` or `.npy`. Accepted shapes:

* `(n, 1536)` flat, `(n, 256, 6)`, `(n, 8, 8, 4, 6)`
* single case: `(1536,)`, `(256, 6)`, `(8, 8, 4, 6)`

For `.npz`, the candidate keys tried are `candidate`, `field`, `fields`; the
reference keys are `reference`, `targets`, `target`, then those same names. Use
`--candidate-key`/`--reference-key` when a file has several arrays. Non-finite
values are rejected.

```bash
python3 tools/neural-bunch/eval/field_gate.py candidate.npz reference.npz \
    --config gate.json --output report.json
```

Exit status: `0` all groups pass, `1` at least one group fails, `2` invalid
input/config (including a missing `s0`). The JSON report contains, per case and
per group:

* `reference_norm`, `error_norm`, `s0`, `denominator`
  (with `denominator_source`: `reference_norm` or `s0_floor`)
* `relative_l2`, `pass`, `absolute_normalized`
* per-component diagnostics (`Ex`, `Ey`, `Ez`/`cBx`, `cBy`, `cBz`) with their
  own norms and the group denominator
* a top-level `summary` with `all_pass`, the failure list and the maximum ratio
* SHA-256 of candidate, reference and config

## Tests

CPU-only, no GPU, no C++ build, no dataset:

```bash
python3 -m unittest tools/neural-bunch/eval/test_field_gate.py
# or, if pytest is installed:
python3 -m pytest -q tools/neural-bunch/eval/test_field_gate.py
```

The synthetic cases pin the analytic value, the `<=` boundary (exact equality
passes, one ULP above fails), the `S0` floor when it dominates, independent
`E`/`cB` reporting, and loud failure when `S0` is absent.

## What the evidence cannot decide

* **No historical numeric `S0` value survives.** `S0` was derived per case
  from the reference tile; the artifacts that stored it are the dataset
  manifests (`s0` arrays) and trained-run reports. If those shards are gone,
  the exact historical number for a given old case cannot be recovered, only
  recomputed from its reference tile under the formula above.
* **The `0.05`/`5e-2` floor's provenance is training-only.** Nothing in the
  evidence makes it part of the acceptance gate; the frozen metric deliberately
  excludes it.
* **`cB` versus `B` at the dump boundary is an assumption.** The tooling
  reports `cB` from the normalized tile `B_tilde`, where `c` is already folded
  in. A raw PIConGPU SI openPMD dump is not accepted directly; converting it
  requires the model-format normalization and the correct time level.
* **The force gate is out of scope here.** The guide defines it separately
  (`picongpu-scalable-field-initialization-agent-guide.md:499-506`); this
  script evaluates the field gate only.
