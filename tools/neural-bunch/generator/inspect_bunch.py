#!/usr/bin/env python3
"""Inspect, validate and compare openPMD electron-bunch series.

Decodes PIConGPU-compatible layout exactly:
    physical_position = (position + positionOffset) * unitSI
    u                 = momentum * unitSI / (m_e*c) / w^weightingPower

Reads HDF5 (h5py) or ADIOS2/BP5 (openPMD), prints the first rows, validates the
openPMD conventions, verifies the run manifest and the ``.physical.npz``
sidecar, and compares two series array-for-array.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import nb_common as common


def print_report(path: str, result: dict, summary: dict, problems: list, rows: int) -> None:
    print(f"series: {path}")
    print(f"  backend: {result['backend']}  patches: {summary['has_patches']}")
    print(
        f"  particles: {summary['count']}  total_weight: {summary['total_weight']!r}  "
        f"total_charge: {summary['total_charge_C']!r} C"
    )
    momentum = summary["momentum"]
    print(
        "  |u|: min={u_min:.6g} max={u_max:.6g} mean={u_mean:.6g}  "
        "gamma: min={gamma_min:.6g} max={gamma_max:.6g} mean={gamma_mean:.6g}  "
        "beta_mean={beta_mean:.6g}".format(**momentum)
    )
    print(
        f"  physical position (m): min={summary['physical_position_min_SI']} max={summary['physical_position_max_SI']}"
    )
    print(
        f"  charge per macro: {summary['charge_per_macro_SI']!r} C  "
        f"cellSizeSI: {summary['cellSizeSI']}  grid: {summary['grid']}"
    )

    for record in (
        "position",
        "positionOffset",
        "momentum",
        "weighting",
        "id",
        "charge",
        "mass",
    ):
        meta = result["meta"].get(record)
        if meta is None:
            continue
        units = ", ".join(f"{name}={value:g}" for name, value in sorted(meta["unitSI"].items()))
        dims = [float(value) for value in meta["unit_dimension"]]
        constant = f" constant={meta['constant']}" if meta["constant"] else ""
        print(
            f"  {record}: unitSI[{units}] unitDimension={dims} "
            f"macroWeighted={meta['macroWeighted']} "
            f"weightingPower={meta['weightingPower']}{constant}"
        )

    if result["patches"] is not None:
        patches = result["patches"]
        print(
            f"  patches: n={len(patches['numParticles'])} "
            f"numParticles={patches['numParticles'].tolist()[:4]} "
            f"offset={patches['offset'].tolist()[:2]} extent={patches['extent'].tolist()[:2]}"
        )

    decoded = common.decode_physical(result)
    print(f"  first {rows} rows: idx frac(x,y,z) offset(x,y,z) physical(x,y,z) p_raw(x,y,z) w u(x,y,z)")
    for index in range(min(rows, summary["count"])):
        print(
            "    {:d} ({:.5g},{:.5g},{:.5g}) ({:g},{:g},{:g}) "
            "({:.6g},{:.6g},{:.6g}) ({:.5g},{:.5g},{:.5g}) w={:.6g} "
            "u=({:.5g},{:.5g},{:.5g})".format(
                index,
                *result["data"]["position"][index],
                *result["data"]["positionOffset"][index],
                *decoded["position"][index],
                *result["data"]["momentum"][index],
                result["data"]["weighting"][index],
                *decoded["u"][index],
            )
        )

    if problems:
        print("  VALIDATION FAILED:")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("  validation: OK")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series", help="series file (.h5) or directory (.bp5)")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--species", default=common.DEFAULT_SPECIES)
    parser.add_argument("--compare", default=None, help="second series to compare arrays with")
    parser.add_argument(
        "--manifest", default=None, help="manifest to verify (default: <series>.manifest.json if present)"
    )
    parser.add_argument("--no-manifest", action="store_true", help="skip manifest verification")
    parser.add_argument("--sidecar", default=None, help="sidecar npz (default: <series>.physical.npz)")
    parser.add_argument("--no-sidecar", action="store_true", help="skip sidecar verification")
    parser.add_argument("--rows", type=int, default=3, help="number of first rows to print")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument("--expect-n", type=int, default=None)
    parser.add_argument("--expect-total-weight", type=float, default=None)
    parser.add_argument("--expect-gamma", type=float, default=None)
    args = parser.parse_args(argv)

    result = common.read_series(args.series, args.iteration, args.species)
    summary, problems = common.validate_readback(result)

    if args.expect_n is not None and summary["count"] != args.expect_n:
        problems.append(f"count {summary['count']} != expected {args.expect_n}")
    if args.expect_total_weight is not None:
        if not np.isclose(summary["total_weight"], args.expect_total_weight, rtol=1e-12):
            problems.append(f"total weight {summary['total_weight']} != expected {args.expect_total_weight}")
    if args.expect_gamma is not None:
        gamma = summary["momentum"]
        if not (
            np.isclose(gamma["gamma_min"], args.expect_gamma, rtol=1e-9)
            and np.isclose(gamma["gamma_max"], args.expect_gamma, rtol=1e-9)
        ):
            problems.append(f"gamma [{gamma['gamma_min']}, {gamma['gamma_max']}] != expected {args.expect_gamma}")

    manifest = None
    if not args.no_manifest:
        manifest_path = args.manifest or (args.series + ".manifest.json")
        if os.path.isfile(manifest_path):
            try:
                manifest = common.load_manifest(manifest_path)
                problems.extend(common.check_manifest(manifest, args.series, result["data"]))
                expected_backend = manifest.get("backend")
                expected_format = {"hdf5": "hdf5", "adios2": "bp5"}.get(expected_backend, expected_backend)
                if expected_backend and expected_format != result["format"]:
                    problems.append(f"manifest backend {expected_backend} != read backend {result['format']}")
            except Exception as error:  # pragma: no cover - malformed manifest
                problems.append(f"manifest error: {error}")
        elif args.manifest:
            problems.append(f"manifest not found: {manifest_path}")

    sidecar_file = args.sidecar or common.sidecar_path(args.series)
    if not args.no_sidecar:
        if os.path.isfile(sidecar_file):
            sidecar_problems = common.compare_sidecar(result, sidecar_file)
            problems.extend(sidecar_problems)
            if manifest and manifest.get("sidecar"):
                expected = manifest["sidecar"]
                if os.path.getsize(sidecar_file) != expected.get("bytes"):
                    problems.append("sidecar: file size differs from manifest")
                if common.sha256_file(sidecar_file) != expected.get("sha256"):
                    problems.append("sidecar: sha256 differs from manifest")
            print(
                f"sidecar: {sidecar_file} -> "
                + ("matches decoded positions/u/weights/ids" if not sidecar_problems else "MISMATCH")
            )
        elif manifest and manifest.get("sidecar"):
            problems.append(f"sidecar missing: {sidecar_file}")
        else:
            print(f"sidecar: absent ({sidecar_file}), skipped")

    comparison = None
    if args.compare:
        other = common.read_series(args.compare, args.iteration, args.species)
        other_summary, other_problems = common.validate_readback(other)
        problems.extend(f"compare: {problem}" for problem in other_problems)
        comparison = common.compare_readbacks(result, other)
        problems.extend(f"compare: {problem}" for problem in comparison)
        print(
            f"compare: {args.series} ({result['format']}) vs "
            f"{args.compare} ({other['format']}): "
            + ("IDENTICAL arrays and metadata" if not comparison else "MISMATCH")
        )
        print(
            f"  counts: {summary['count']} vs {other_summary['count']}, "
            f"weights: {summary['total_weight']!r} vs {other_summary['total_weight']!r}, "
            f"charges: {summary['total_charge_C']!r} vs {other_summary['total_charge_C']!r}"
        )

    print_report(args.series, result, summary, problems, args.rows)
    if args.json:
        report = {"series": args.series, "summary": summary, "problems": problems}
        if args.compare:
            report["comparison"] = comparison or []
        print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
