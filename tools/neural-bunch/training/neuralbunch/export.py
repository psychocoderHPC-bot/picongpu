"""Write and read MODEL-FORMAT.md version-1 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .features import (BASE_INPUTS, DEFAULT_OCTAVES, DENSE_CELL_OFFSETS, DENSE_KIND,
                       PROBE_CELL_OFFSETS, PROBE_COMPONENTS, PROBE_KIND, PROBE_OCTAVES,
                       feature_dim)
from .model import COMPONENTS, OUT_DIM, TILE

FORMAT = "picongpu-neural-bunch"
VERSION = 1
ALIGN = 64


def compute_code_sha() -> str:
    env = os.environ.get("NEURALBUNCH_CODE_SHA")
    if env:
        return env
    training_dir = Path(__file__).resolve().parent.parent
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(training_dir), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if len(sha) == 40:
            return sha
    except Exception:
        pass
    h = hashlib.sha1()
    pkg = Path(__file__).resolve().parent
    for p in sorted(pkg.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _check_model(cfg, tensors):
    cfg.validate()
    if tuple(TILE) != (8, 8, 4):
        raise ValueError("format v1 fixes tile 8x8x4")
    if cfg.out_dim != OUT_DIM:
        raise ValueError(f"decoder out dimension must be {OUT_DIM}, got {cfg.out_dim}")
    if cfg.feature_kind == PROBE_KIND:
        expected_dim = feature_dim(PROBE_KIND, cfg.probe_octaves)
    elif cfg.feature_kind == DENSE_KIND:
        expected_dim = feature_dim(DENSE_KIND)
    else:
        expected_dim = feature_dim("fourier_v1", cfg.octaves)
    if cfg.feature_dim != expected_dim:
        raise ValueError(f"feature dimension mismatch: {cfg.feature_dim} != {expected_dim}")
    by_name = dict(tensors)
    expect_out = cfg.feature_dim
    n_layers = len(cfg.hidden) + 1
    for i in range(n_layers):
        w = by_name.get(f"enc.{i}.w")
        b = by_name.get(f"enc.{i}.b")
        if w is None or b is None:
            raise ValueError(f"missing encoder tensor enc.{i}")
        out = cfg.hidden[i] if i < len(cfg.hidden) else cfg.latent
        if w.shape != (expect_out, out) or b.shape != (out,):
            raise ValueError(
                f"encoder layer {i} shape mismatch: weight {w.shape} expected {(expect_out, out)}"
            )
        expect_out = out
    nz, ny = len(cfg.a_z_anchors), len(cfg.a_y_anchors)
    for iz in range(nz):
        for iy in range(ny):
            name = f"dec.{iz}.{iy}.w"
            t = by_name.get(name)
            if t is None:
                raise ValueError(f"missing decoder bank tensor {name}")
            if t.shape != (cfg.latent, OUT_DIM):
                raise ValueError(
                    f"{name} shape {t.shape} expected {(cfg.latent, OUT_DIM)}"
                )
    extra = set(by_name) - {
        f"enc.{i}.{k}" for i in range(n_layers) for k in ("w", "b")
    } - {f"dec.{iz}.{iy}.w" for iz in range(nz) for iy in range(ny)}
    if extra:
        raise ValueError(f"unexpected tensors: {sorted(extra)}")


def _features_metadata(cfg):
    if cfg.feature_kind == DENSE_KIND:
        return {
            "kind": DENSE_KIND,
            "dim": int(cfg.feature_dim),
            "inputs": list(BASE_INPUTS),
            "octaves": [],
            "probe": {
                "grid": "one probe at every tile cell centre: x=i+0.5 (i=0..7), y=(j+0.5)*a_y (j=0..7), z=(k+0.5)*a_z (k=0..3), i fastest then j then k",
                "cell_offsets": [list(p) for p in DENSE_CELL_OFFSETS],
                "components": list(PROBE_COMPONENTS),
                "field": "per unit q/e: E_n=4*pi*eps0*dx^2*E/q, B_n=4*pi*eps0*c*dx^2*B/q; stable uniform-motion decomposition (1/gamma^2 direct, static branch)",
            },
            "ordering": "9 base, 256 probes x [Ex,Ey,Ez,Bx,By,Bz] in i-fastest cell order; no Fourier terms",
        }
    if cfg.feature_kind == PROBE_KIND:
        return {
            "kind": PROBE_KIND,
            "dim": int(cfg.feature_dim),
            "inputs": list(BASE_INPUTS),
            "octaves": list(PROBE_OCTAVES),
            "probe": {
                "grid": "4x4x2 tile cell centres: x=2i+0.5 (i=0..3), y=(2j+0.5)*a_y (j=0..3), z=(2k+0.5)*a_z (k=0..1)",
                "cell_offsets": [list(p) for p in PROBE_CELL_OFFSETS],
                "components": list(PROBE_COMPONENTS),
                "field": "per unit q/e: E_n=4*pi*eps0*dx^2*E/q, B_n=4*pi*eps0*c*dx^2*B/q; stable uniform-motion decomposition (1/gamma^2 direct, static branch)",
            },
            "ordering": "9 base, 32 probes x [Ex,Ey,Ez,Bx,By,Bz], 3 axes x octaves -3..1 x (sin,cos)",
        }
    return {
        "kind": "fourier_v1",
        "octaves": list(DEFAULT_OCTAVES),
        "inputs": list(BASE_INPUTS),
        "dim": feature_dim("fourier_v1", DEFAULT_OCTAVES),
    }


def build_metadata(cfg, tensor_specs, file_name, nbytes, sha256, code_sha,
                   reference=None, description=None):
    n_y, n_z = len(cfg.a_y_anchors), len(cfg.a_z_anchors)
    shape_map = {name: shape for name, _off, shape in tensor_specs}
    meta = {
        "format": FORMAT,
        "version": VERSION,
        "dtype": "float32",
        "endianness": "little",
        "created_utc": _now(),
        "code_sha": code_sha,
        "tile": {
            "nx": TILE[0], "ny": TILE[1], "nz": TILE[2],
            "components": list(COMPONENTS),
            "flatten": "c + 6*(i + nx*(j + ny*k))",
        },
        "features": _features_metadata(cfg),
        "encoder": {
            "kind": "mlp",
            "activation": "silu",
            "layers": [
                {"weight": f"enc.{i}.w", "bias": f"enc.{i}.b",
                 "in": int(shape_map[f"enc.{i}.w"][0]),
                 "out": int(shape_map[f"enc.{i}.w"][1])}
                for i in range(len(cfg.hidden) + 1)
            ],
            "output_dim": int(cfg.latent),
            "last_layer_activation": "none",
        },
        "decoder": {
            "kind": "geometry_bank_bilinear",
            "axes": {
                "a_y": [float(x) for x in cfg.a_y_anchors],
                "a_z": [float(x) for x in cfg.a_z_anchors],
            },
            "interpolation": "bilinear in log10(a), clamped at the outer anchors",
            "matrix_shape": [int(cfg.latent), OUT_DIM],
            "bias": False,
            "bank": [f"dec.{iz}.{iy}.w" for iz in range(n_z) for iy in range(n_y)],
            "bank_entry_shape": [int(cfg.latent), OUT_DIM],
        },
        "normalization": {
            "length": "dx",
            "charge": "q_i/e = -w_i",
            "field_E": "E = e/(4*pi*eps0*L^2) * E_tilde",
            "field_B": "B = e/(4*pi*eps0*c*L^2) * B_tilde",
            "latent": "none",
        },
        "domain": {
            "gamma": [1.0, 100.0],
            "a_y": [0.1, 10.0],
            "a_z": [0.1, 10.0],
            "d_min_cells": 4.0,
            "d_max_cells": 512.0,
            "distance_metric": "minimum over the 1536 staggered output points of |x_out - x_source(t)| / dx, with x_out in physical coordinates",
        },
        "reference": reference or {
            "shape": "TSC",
            "shape_support_cells": [2.0, 2.0, 2.0],
            "integrator": "adaptive FP64 quadrature of the constant-velocity point kernel over the lab-frame shape",
            "description": description or "FP64 reference implemented by reference/convkernel.py",
        },
        "fallback": {
            "policy": "per (source, tile, component-group): reference if distance < d_min_cells or > d_max_cells or gamma/aspect outside domain",
            "component_groups": {"E": "Ex,Ey,Ez", "B": "Bx,By,Bz"},
        },
        "validation": {
            "tolerance_field_rel_rms": 0.001,
            "tolerance_force_rel_rms": 0.01,
            "report": "validation_report.json",
        },
        "tensors": [
            {"name": name, "offset_bytes": int(off), "count": int(np.prod(shape)),
             "shape": [int(s) for s in shape], "order": "row_major_in_out"}
            for name, off, shape in tensor_specs
        ],
        "artifact": {"file": file_name, "bytes": int(nbytes), "sha256": sha256},
        "license": "MPL-2.0",
    }
    return meta


def write_artifact(model, out_prefix, code_sha=None, reference=None, description=None):
    """Serialize ``model`` to ``<out_prefix>.json`` + ``<out_prefix>.bin``."""
    out_prefix = Path(out_prefix)
    if out_prefix.suffix in (".json", ".bin"):
        out_prefix = out_prefix.with_suffix("")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    named = [(n, np.asarray(t, dtype=np.float64)) for n, t in model.artifact_tensors()]
    _check_model(cfg, named)
    # canonical encoder-then-bank order
    order = []
    for i in range(len(cfg.hidden) + 1):
        order += [f"enc.{i}.w", f"enc.{i}.b"]
    nz, ny = len(cfg.a_z_anchors), len(cfg.a_y_anchors)
    order += [f"dec.{iz}.{iy}.w" for iz in range(nz) for iy in range(ny)]
    by_name = dict(named)
    out_bin = out_prefix.with_suffix(".bin")
    tmp_bin = out_bin.with_suffix(".bin.tmp")
    specs = []
    offset = 0
    h = hashlib.sha256()
    with open(tmp_bin, "wb") as f:
        for name in order:
            arr = by_name[name]
            pad = (-offset) % ALIGN
            if pad:
                z = b"\x00" * pad
                f.write(z)
                h.update(z)
                offset += pad
            data = np.ascontiguousarray(arr, dtype="<f4").tobytes(order="C")
            f.write(data)
            h.update(data)
            specs.append((name, offset, tuple(int(s) for s in arr.shape)))
            offset += len(data)
    os.replace(tmp_bin, out_bin)
    sha = h.hexdigest()
    nbytes = out_bin.stat().st_size
    meta = build_metadata(
        cfg, specs, out_bin.name, nbytes, sha,
        code_sha or compute_code_sha(), reference=reference, description=description,
    )
    json_path = out_prefix.with_suffix(".json")
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_json, json_path)
    return meta


# --------------------------------------------------------------------------
# loading / validation
# --------------------------------------------------------------------------
class ArtifactError(ValueError):
    pass


def load_artifact(prefix):
    """Verify and load an artifact.  Returns ``(metadata, tensors_dict)``."""
    prefix = Path(prefix)
    if prefix.suffix == ".json":
        json_path = prefix
        prefix = prefix.with_suffix("")
    else:
        json_path = prefix.with_suffix(".json")
    if not json_path.is_file():
        raise ArtifactError(f"missing metadata file {json_path}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    for key in ("format", "version", "dtype", "tile", "features", "encoder",
                "decoder", "tensors", "artifact"):
        if key not in meta:
            raise ArtifactError(f"metadata missing required key {key!r}")
    if meta["format"] != FORMAT:
        raise ArtifactError(f"unknown format {meta['format']!r}")
    if int(meta["version"]) != VERSION:
        raise ArtifactError(f"unsupported version {meta['version']}")
    if meta["dtype"] != "float32" or meta["endianness"] != "little":
        raise ArtifactError("only little-endian float32 is supported")
    if int(meta["tile"]["nx"]) != TILE[0] or int(meta["tile"]["ny"]) != TILE[1] or int(meta["tile"]["nz"]) != TILE[2]:
        raise ArtifactError("unsupported tile geometry")
    _kind = meta["features"].get("kind", "fourier_v1")
    _dim = int(meta["features"]["dim"])
    if _kind == "fourier_v1":
        _ok = _dim == 69
    elif _kind == PROBE_KIND:
        _ok = _dim == 231
    elif _kind == DENSE_KIND:
        _ok = _dim == 1545
    else:
        _ok = False
    if not _ok:
        raise ArtifactError("unsupported feature specification")
    bin_path = json_path.parent / meta["artifact"]["file"]
    if not bin_path.is_file():
        raise ArtifactError(f"missing payload {bin_path}")
    size = bin_path.stat().st_size
    if size != int(meta["artifact"]["bytes"]):
        raise ArtifactError(f"byte count mismatch: {size} != {meta['artifact']['bytes']}")
    h = hashlib.sha256()
    with open(bin_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != meta["artifact"]["sha256"]:
        raise ArtifactError("payload sha256 mismatch")
    raw = np.fromfile(bin_path, dtype="<f4")
    tensors = {}
    last_end = 0
    for spec in meta["tensors"]:
        name = spec["name"]
        off = int(spec["offset_bytes"])
        shape = tuple(int(s) for s in spec["shape"])
        count = int(spec["count"])
        if off % ALIGN != 0:
            raise ArtifactError(f"{name}: offset {off} is not {ALIGN}-byte aligned")
        if off < last_end:
            raise ArtifactError(f"{name}: overlapping or non-monotonic offset")
        if count != int(np.prod(shape)):
            raise ArtifactError(f"{name}: count/shape mismatch")
        if off + count * 4 > size:
            raise ArtifactError(f"{name}: tensor exceeds payload")
        tensors[name] = raw[off // 4: off // 4 + count].reshape(shape).astype(np.float64)
        last_end = off + count * 4
    # geometry consistency
    layers = meta["encoder"]["layers"]
    expect = int(meta["features"]["dim"])
    for i, layer in enumerate(layers):
        if int(layer["in"]) != expect:
            raise ArtifactError(f"encoder layer {i}: input dim {layer['in']} != {expect}")
        if tensors[layer["weight"]].shape != (int(layer["in"]), int(layer["out"])):
            raise ArtifactError(f"encoder layer {i}: weight shape mismatch")
        expect = int(layer["out"])
    if expect != int(meta["encoder"]["output_dim"]):
        raise ArtifactError("encoder output_dim mismatch")
    dec = meta["decoder"]
    if list(dec["matrix_shape"]) != [int(meta["encoder"]["output_dim"]), OUT_DIM]:
        raise ArtifactError("decoder matrix_shape mismatch")
    ay = np.asarray(dec["axes"]["a_y"], dtype=np.float64)
    az = np.asarray(dec["axes"]["a_z"], dtype=np.float64)
    if len(dec["bank"]) != len(ay) * len(az):
        raise ArtifactError("decoder bank length does not match axis anchors")
    if not np.all(np.diff(np.log10(ay)) > 0) or not np.all(np.diff(np.log10(az)) > 0):
        raise ArtifactError("decoder anchors are not increasing")
    if ay[0] < 0.1 - 1e-9 or ay[-1] > 10 + 1e-9 or az[0] < 0.1 - 1e-9 or az[-1] > 10 + 1e-9:
        raise ArtifactError("decoder anchors outside [0.1, 10]")
    for name, arr in tensors.items():
        if not np.all(np.isfinite(arr)):
            raise ArtifactError(f"{name} contains non-finite values")
    return meta, tensors
