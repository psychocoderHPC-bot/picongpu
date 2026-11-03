#!/usr/bin/env python3
"""Evaluate the preregistered neural-bunch field gate on field dumps.

The gate is, per case and per component group ``G in {E, cB}``::

    epsilon_G = ||A_candidate - A_reference||_2
                / max(||A_reference||_2, S0_G)  <= 1e-3

``A`` is the 768-scalar vector of one group (256 staggered output points times
three components), in the canonical ``[i, j, k, component]`` tile layout of
``tools/neural-bunch/model-format.md``.  The norm is the unweighted Euclidean
norm over that point set.  ``S0_G`` is not a global constant: it is reconstructed
per case from the reference tile as

    S0_G = sqrt(N_points) * max_point ||G_point||_2,   N_points = 256,

which is the historical preregistered scale documented at
``tools/neural-bunch/training/neuralbunch/reference_data.py:92-105`` and
``tools/neural-bunch/training/README.md:125-139`` (``sqrt(256) = 16``).

The frozen config is a JSON file and MUST declare how ``S0`` is obtained.  A
config without ``s0`` is an error; there is deliberately no implicit default,
because a silently chosen floor makes every accuracy claim unfalsifiable.

Example config::

    {
      "gate": 1e-3,
      "s0": {"mode": "reference_tile"}
    }

Supported ``s0.mode`` values:

* ``reference_tile`` - recompute the historical per-case ``S0`` from the
  reference dump (the frozen definition above).
* ``values`` - explicit per-case ``values``: ``[[S0_E, S0_cB], ...]``.
* ``constant`` - one pair ``{"E": ..., "cB": ...}`` for every case.
* ``manifest`` - read the ``s0`` array (shape ``(n_cases, 2)``) stored next to
  the reference dump, e.g. ``manifest.json``/shard ``s0`` written by the
  dataset tooling.

Usage::

    python3 tools/neural-bunch/eval/field_gate.py CANDIDATE.npz REFERENCE.npz \
        --config gate.json --output report.json

Input arrays may be ``.npz``/``.npy``.  Accepted shapes are ``(n, 1536)``,
``(n, 256, 6)``, ``(n, 8, 8, 4, 6)`` and the single-case forms ``(1536,)``,
``(256, 6)``, ``(8, 8, 4, 6)``.  Values are assumed to already be in the
dimensionless normalized tile convention ``E_tilde`` and ``cB_tilde`` from
``tools/neural-bunch/model-format.md:57-61``; a physical SI dump must be
normalized before it is compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

GATE_DEFAULT = 1e-3
N_POINTS = 8 * 8 * 4
GROUPS = {"E": slice(0, 3), "cB": slice(3, 6)}
COMPONENT_NAMES = {
    "E": ("Ex", "Ey", "Ez"),
    "cB": ("cBx", "cBy", "cBz"),
}
FLAT_SIZE = N_POINTS * 6


class MetricError(RuntimeError):
    """Raised for an ambiguous input, a missing S0 declaration, or bad data."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_arrays(path: str) -> Dict[str, np.ndarray]:
    """Load a ``.npz`` (all arrays) or ``.npy`` (one array) file."""
    p = Path(path)
    if not p.is_file():
        raise MetricError(f"input file does not exist: {path}")
    try:
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        if p.suffix == ".npy":
            return {"array": np.load(p, allow_pickle=False)}
        # np.load also accepts an open file; keep the name list explicit.
        with open(p, "rb") as handle:
            arr = np.load(handle, allow_pickle=False)
            return {"array": arr}
    except MetricError:
        raise
    except Exception as exc:  # noqa: BLE001 - report the concrete loader error
        raise MetricError(f"cannot read {path!r}: {exc}") from exc


def pick_array(
    arrays: Mapping[str, np.ndarray],
    path: str,
    role: str,
    key: str | None = None,
) -> np.ndarray:
    """Select the field array from a loaded file."""
    if key is not None:
        if key not in arrays:
            raise MetricError(
                f"{role} file {path!r} has no array {key!r}; available: "
                f"{sorted(arrays)}"
            )
        return arrays[key]
    for candidate in ("candidate", "field", "fields"):
        if candidate in arrays:
            return arrays[candidate]
    if role == "reference":
        for candidate in ("reference", "targets", "target"):
            if candidate in arrays:
                return arrays[candidate]
    if len(arrays) == 1:
        return next(iter(arrays.values()))
    raise MetricError(
        f"cannot choose the {role} array in {path!r}: pass --{role}-key. "
        f"Available arrays: {sorted(arrays)}"
    )


def canonical_cases(array: np.ndarray, role: str) -> np.ndarray:
    """Return a ``(n_cases, 256, 6)`` view, component fastest."""
    arr = np.asarray(array)
    if arr.dtype.kind not in "fiu":
        raise MetricError(f"{role} array has non-numeric dtype {arr.dtype}")
    if arr.ndim == 1 and arr.size == FLAT_SIZE:
        arr = arr.reshape(1, N_POINTS, 6)
    elif arr.ndim == 2 and arr.shape[1] == FLAT_SIZE:
        arr = arr.reshape(arr.shape[0], N_POINTS, 6)
    elif arr.ndim == 2 and arr.shape == (N_POINTS, 6):
        arr = arr.reshape(1, N_POINTS, 6)
    elif arr.ndim == 3 and arr.shape[1:] == (N_POINTS, 6):
        pass
    elif arr.ndim == 4 and arr.shape == (8, 8, 4, 6):
        arr = arr.reshape(1, N_POINTS, 6)
    elif arr.ndim == 5 and arr.shape[1:] == (8, 8, 4, 6):
        arr = arr.reshape(arr.shape[0], N_POINTS, 6)
    else:
        raise MetricError(
            f"{role} array has shape {arr.shape}; expected (n,1536), (n,256,6), "
            f"(n,8,8,4,6), or a single-case form"
        )
    arr = arr.astype(np.float64, copy=False)
    if arr.shape[0] == 0:
        raise MetricError(f"{role} array contains no cases")
    if not np.all(np.isfinite(arr)):
        raise MetricError(f"{role} array contains NaN or Inf")
    return arr


def reference_tile_s0(reference_case: np.ndarray) -> Dict[str, float]:
    """Frozen historical S0: ``sqrt(N_points) * max_point ||G_point||``."""
    scale = math.sqrt(N_POINTS)
    out: Dict[str, float] = {}
    for group, sl in GROUPS.items():
        vectors = reference_case[:, sl]
        point_norms = np.sqrt(np.sum(vectors * vectors, axis=1))
        out[group] = float(scale * np.max(point_norms))
    return out


def _finite_nonnegative(value: Any, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricError(f"{where} must be a number, got {value!r}") from exc
    if not math.isfinite(result) or result < 0.0:
        raise MetricError(f"{where} must be finite and >= 0, got {value!r}")
    return result


def _pair_from_mapping(mapping: Mapping[str, Any], where: str) -> Dict[str, float]:
    missing = [group for group in GROUPS if group not in mapping]
    if missing:
        raise MetricError(f"{where} is missing {missing}; expected 'E' and 'cB'")
    return {group: _finite_nonnegative(mapping[group], f"{where}.{group}") for group in GROUPS}


def read_s0_spec(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the config's ``s0`` block, failing loudly when absent."""
    if "s0" not in config:
        raise MetricError(
            "frozen config does not specify 's0'; the gate is unfalsifiable "
            "without it. Use {\"s0\": {\"mode\": \"reference_tile\"}} or "
            "explicit per-case/constant values."
        )
    spec = config["s0"]
    if not isinstance(spec, Mapping):
        raise MetricError("config 's0' must be an object with a 'mode'")
    mode = spec.get("mode")
    allowed = {"reference_tile", "values", "constant", "manifest"}
    if mode not in allowed:
        raise MetricError(
            f"config 's0.mode' must be one of {sorted(allowed)}, got {mode!r}"
        )
    return spec


def resolve_s0(
    spec: Mapping[str, Any],
    reference: np.ndarray,
    reference_arrays: Mapping[str, np.ndarray],
) -> List[Dict[str, float]]:
    """Return one ``{"E": ..., "cB": ...}`` per case."""
    mode = spec["mode"]
    n_cases = reference.shape[0]
    if mode == "reference_tile":
        return [reference_tile_s0(reference[i]) for i in range(n_cases)]
    if mode == "constant":
        pair = _pair_from_mapping(spec, "s0.constant") if "E" in spec else _pair_from_mapping(
            spec.get("constant", {}), "s0.constant"
        )
        return [dict(pair) for _ in range(n_cases)]
    if mode == "values":
        values = spec.get("values")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise MetricError("config 's0.values' must be a sequence of [E, cB] pairs")
        if len(values) != n_cases:
            raise MetricError(
                f"config has {len(values)} s0.values but the data has {n_cases} cases"
            )
        resolved = []
        for i, pair in enumerate(values):
            if isinstance(pair, Mapping):
                resolved.append(_pair_from_mapping(pair, f"s0.values[{i}]"))
            elif isinstance(pair, Sequence) and len(pair) == 2:
                resolved.append(
                    {
                        group: _finite_nonnegative(pair[j], f"s0.values[{i}][{j}]")
                        for j, group in enumerate(GROUPS)
                    }
                )
            else:
                raise MetricError(
                    f"s0.values[{i}] must be [S0_E, S0_cB] or an object with E/cB"
                )
        return resolved
    # mode == "manifest"
    if "s0" not in reference_arrays:
        raise MetricError(
            "s0.mode is 'manifest' but the reference file has no 's0' array"
        )
    stored = np.asarray(reference_arrays["s0"], dtype=np.float64)
    if stored.shape != (n_cases, 2):
        raise MetricError(
            f"manifest s0 has shape {stored.shape}; expected ({n_cases}, 2)"
        )
    resolved = []
    for i in range(n_cases):
        resolved.append(
            {
                group: _finite_nonnegative(stored[i, j], f"manifest s0[{i}][{j}]")
                for j, group in enumerate(GROUPS)
            }
        )
    return resolved


def _group_metric(
    candidate: np.ndarray,
    reference: np.ndarray,
    s0: float,
    gate: float,
    group: str,
) -> Dict[str, Any]:
    sl = GROUPS[group]
    pred = candidate[:, sl]
    ref = reference[:, sl]
    err = pred - ref
    ref_norm = float(np.sqrt(np.sum(ref * ref)))
    err_norm = float(np.sqrt(np.sum(err * err)))
    denominator = max(ref_norm, s0)
    degenerate = denominator <= 0.0
    if degenerate:
        # Exact-zero reference group (for example stationary B) with no S0.
        # An identical zero candidate is exactly correct; any nonzero candidate
        # is an infinite relative error.  This is the explicit near-zero policy
        # the guide asks for; it is not a tolerance added to the denominator.
        relative = 0.0 if err_norm == 0.0 else None
    else:
        relative = err_norm / denominator
    passed = bool(relative is not None and relative <= gate)
    components: Dict[str, Any] = {}
    for j, name in enumerate(COMPONENT_NAMES[group]):
        c_err = float(np.sqrt(np.sum(err[:, j] * err[:, j])))
        c_ref = float(np.sqrt(np.sum(ref[:, j] * ref[:, j])))
        components[name] = {
            "reference_norm": c_ref,
            "error_norm": c_err,
            "relative_l2": None if degenerate else c_err / denominator,
        }
    return {
        "reference_norm": ref_norm,
        "error_norm": err_norm,
        "s0": s0,
        "denominator": denominator,
        "denominator_source": (
            "zero_denominator"
            if degenerate
            else ("s0_floor" if s0 > ref_norm else "reference_norm")
        ),
        "relative_l2": relative,
        "pass": passed,
        "absolute_normalized": bool(ref_norm < s0),
        "zero_denominator": degenerate,
        "components": components,
    }


def evaluate(
    candidate_array: np.ndarray,
    reference_array: np.ndarray,
    config: Mapping[str, Any],
    reference_arrays: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute the frozen metric report.  Raises :class:`MetricError` on misuse."""
    spec = read_s0_spec(config)  # raises loudly when S0 is not declared
    gate = _finite_nonnegative(config.get("gate", GATE_DEFAULT), "config.gate")
    if gate <= 0.0:
        raise MetricError("config.gate must be > 0")
    candidate = canonical_cases(candidate_array, "candidate")
    reference = canonical_cases(reference_array, "reference")
    if candidate.shape != reference.shape:
        raise MetricError(
            f"candidate has {candidate.shape[0]} cases with shape {candidate.shape[1:]}, "
            f"reference has {reference.shape[0]} with {reference.shape[1:]}"
        )
    s0_values = resolve_s0(spec, reference, reference_arrays)

    case_ids = config.get("case_ids")
    if case_ids is not None:
        if len(case_ids) != reference.shape[0]:
            raise MetricError(
                f"config has {len(case_ids)} case_ids but data has {reference.shape[0]} cases"
            )
    else:
        case_ids = list(range(reference.shape[0]))

    cases = []
    for i in range(reference.shape[0]):
        entry: Dict[str, Any] = {"index": i, "case": case_ids[i]}
        for group in GROUPS:
            entry[group] = _group_metric(
                candidate[i], reference[i], s0_values[i][group], gate, group
            )
        cases.append(entry)

    failures = [
        (case["case"], group)
        for case in cases
        for group in GROUPS
        if not case[group]["pass"]
    ]
    finite_ratios = [
        case[group]["relative_l2"]
        for case in cases
        for group in GROUPS
        if case[group]["relative_l2"] is not None
    ]
    return {
        "metric": {
            "name": "picongpu-neural-bunch-field-gate",
            "version": 1,
            "gate": gate,
            "comparison": "pass iff relative_l2 <= gate (the guide uses <=)",
            "groups": ["E", "cB"],
            "vector": (
                "for each group: the 3 components at all 256 staggered output "
                "points, canonical [i,j,k,component] layout; 768 scalars"
            ),
            "norm": "unweighted Euclidean (L2) over those 768 scalars",
            "s0_definition": (
                "S0_G = sqrt(N_points) * max_point ||G_point||_2, N_points=256 "
                "(historically written S0 = 16 * max_cell ||G||)"
            ),
            "units": (
                "same as the input, i.e. dimensionless normalized tile field "
                "E_tilde and cB_tilde; inputs must be pre-normalized"
            ),
            "reference_tile_source": "tools/neural-bunch/training/neuralbunch/reference_data.py:92-105",
        },
        "s0_specification": dict(spec),
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "n_groups": len(cases) * len(GROUPS),
            "n_failures": len(failures),
            "all_pass": not failures,
            "failures": [{"case": c, "group": g} for c, g in failures],
            "n_undefined_zero_denominator": sum(
                1
                for case in cases
                for group in GROUPS
                if case[group]["zero_denominator"]
            ),
            "max_relative_l2": max(finite_ratios) if finite_ratios else None,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the preregistered neural-bunch field gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("candidate", help="candidate field dump (.npz/.npy)")
    parser.add_argument("reference", help="reference field dump (.npz/.npy)")
    parser.add_argument(
        "--config", required=True, help="frozen JSON config; must declare 's0'"
    )
    parser.add_argument("--candidate-key", default=None, help="npz array name")
    parser.add_argument("--reference-key", default=None, help="npz array name")
    parser.add_argument("--output", default=None, help="write the JSON report here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config_path = Path(args.config)
        if not config_path.is_file():
            raise MetricError(f"config file does not exist: {args.config}")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, Mapping):
            raise MetricError("config JSON must be an object")
        candidate_arrays = load_arrays(args.candidate)
        reference_arrays = load_arrays(args.reference)
        candidate = pick_array(
            candidate_arrays, args.candidate, "candidate", args.candidate_key
        )
        reference = pick_array(
            reference_arrays, args.reference, "reference", args.reference_key
        )
        report = evaluate(candidate, reference, config, reference_arrays)
    except MetricError as exc:
        print(f"field_gate: error: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"field_gate: error: invalid config JSON: {exc}", file=sys.stderr)
        return 2

    report["inputs"] = {
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "config": str(args.config),
        "candidate_sha256": _sha256(args.candidate),
        "reference_sha256": _sha256(args.reference),
        "config_sha256": _sha256(args.config),
    }
    text = json.dumps(report, indent=2, sort_keys=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["summary"]["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
