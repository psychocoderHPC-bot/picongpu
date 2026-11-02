"""Validate and print a neural-bunch artifact."""

from __future__ import annotations


import numpy as np

from .export import ArtifactError, load_artifact


def inspect_artifact(prefix) -> dict:
    meta, tensors = load_artifact(prefix)
    checks = {
        "format": meta["format"],
        "version": meta["version"],
        "dtype": meta["dtype"],
        "endianness": meta["endianness"],
        "code_sha": meta["code_sha"],
        "features_dim": meta["features"]["dim"],
        "tile": [meta["tile"]["nx"], meta["tile"]["ny"], meta["tile"]["nz"]],
        "encoder_layers": [[layer["in"], layer["out"]] for layer in meta["encoder"]["layers"]],
        "latent": meta["encoder"]["output_dim"],
        "decoder_matrix_shape": meta["decoder"]["matrix_shape"],
        "bank_entries": len(meta["decoder"]["bank"]),
        "a_y_range": [min(meta["decoder"]["axes"]["a_y"]), max(meta["decoder"]["axes"]["a_y"])],
        "a_z_range": [min(meta["decoder"]["axes"]["a_z"]), max(meta["decoder"]["axes"]["a_z"])],
        "artifact_bytes": meta["artifact"]["bytes"],
        "artifact_sha256": meta["artifact"]["sha256"],
        "tensor_count": len(tensors),
        "tensor_elements": int(sum(a.size for a in tensors.values())),
    }
    # extra content checks
    for name, arr in tensors.items():
        if arr.size == 0:
            raise ArtifactError(f"{name} is empty")
        if not np.all(np.isfinite(arr)):
            raise ArtifactError(f"{name} has non-finite values")
        n = np.linalg.norm(arr)
        if not np.isfinite(n):
            raise ArtifactError(f"{name} has an infinite norm")
    if meta["encoder"]["activation"] != "silu":
        raise ArtifactError("only silu encoder activation is supported")
    if meta["decoder"]["kind"] != "geometry_bank_bilinear":
        raise ArtifactError("only geometry_bank_bilinear decoders are supported")
    if meta["decoder"].get("bias"):
        raise ArtifactError("decoder bias must be false")
    return {"valid": True, "checks": checks, "metadata": meta}


def print_report(report, stream=None):
    import sys

    out = stream or sys.stdout
    print("artifact: VALID", file=out)
    for k, v in report["checks"].items():
        print(f"  {k}: {v}", file=out)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Validate a neural-bunch artifact")
    ap.add_argument("artifact", help="path prefix or .json file")
    args = ap.parse_args(argv)
    try:
        report = inspect_artifact(args.artifact)
    except (ArtifactError, FileNotFoundError) as exc:
        print(f"artifact: INVALID: {exc}")
        return 1
    print_report(report)
    return 0
