"""Shared constants and readers for the deterministic openPMD electron-bunch generator.

No PIConGPU code is used.  On-disk conventions (measured against PIConGPU PR#17
output on fwk394):

* ``position``       raw in-cell fraction in [0,1), per-component unitSI = cell size.
* ``positionOffset`` raw integer cell index (float64 here, int32 in PIConGPU),
                     per-component unitSI = cell size.
* physical position  = (position + positionOffset) * unitSI  (component-wise).
* ``momentum``       macro momentum in units of m_e*c, unitSI = m_e*c,
                     macroWeighted=1, weightingPower=1; raw = u * w.
* ``weighting``      raw w, unitSI=1, macroWeighted=1, weightingPower=1.
* single-electron    u = momentum * unitSI / (m_e*c) / w^weightingPower.
* ``charge``/``mass`` constant records; total charge = raw*unitSI*sum(w^wp).

Readers: HDF5 via h5py; ADIOS2/BP5 via openPMD (every load is followed by
``series_flush()`` because openPMD 0.17.x otherwise returns an undefined buffer).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

ELECTRON_CHARGE = -1.602176634e-19  # C
ELECTRON_MASS = 9.1093837015e-31  # kg
SPEED_OF_LIGHT = 299792458.0  # m/s
MOMENTUM_UNIT_SI = ELECTRON_MASS * SPEED_OF_LIGHT  # kg m/s
DEFAULT_SPECIES = "electron_bunch"

POSITION_OFFSET_NAMES = ("x", "y", "z")
VECTOR_RECORDS = ("position", "positionOffset", "momentum")
SCALAR_RECORDS = ("weighting", "id")
CONSTANT_RECORDS = ("charge", "mass")

ZERO_DIM = {
    "L": 0.0,
    "M": 0.0,
    "T": 0.0,
    "I": 0.0,
    "Theta": 0.0,
    "N": 0.0,
    "J": 0.0,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_checksums(data: dict) -> dict:
    """Canonical per-component SHA-256 over raw little-endian bytes."""
    checksums = {}
    for name, array in data.items():
        array = np.ascontiguousarray(array)
        if array.ndim == 2:
            for axis, component in enumerate(POSITION_OFFSET_NAMES):
                checksums[f"{name}/{component}"] = sha256_bytes(np.ascontiguousarray(array[:, axis]).tobytes())
        else:
            checksums[name] = sha256_bytes(array.tobytes())
    return checksums


def file_manifest(path: str) -> list:
    entries = []

    def add_file(absolute: str, relative: str) -> None:
        entries.append(
            {
                "path": relative,
                "bytes": os.path.getsize(absolute),
                "sha256": sha256_file(absolute),
            }
        )

    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for name in sorted(files):
                absolute = os.path.join(root, name)
                add_file(absolute, os.path.relpath(absolute, path))
        entries.sort(key=lambda item: item["path"])
    else:
        add_file(path, os.path.basename(path))
    return entries


def total_bytes(entries: list) -> int:
    return int(sum(item["bytes"] for item in entries))


def sidecar_path(series: str) -> str:
    """Per-series sidecar: ``<series>.physical.npz`` (backend suffix preserved)."""
    return series + ".physical.npz"


def write_sidecar(path: str, position: np.ndarray, u: np.ndarray, weighting: np.ndarray, ids: np.ndarray) -> dict:
    np.savez(
        path,
        position=np.ascontiguousarray(position, dtype=np.float64),
        u=np.ascontiguousarray(u, dtype=np.float64),
        weighting=np.ascontiguousarray(weighting, dtype=np.float64),
        id=np.ascontiguousarray(ids, dtype=np.uint64),
    )
    return {
        "file": os.path.basename(path),
        "bytes": os.path.getsize(path),
        "sha256": sha256_file(path),
    }


# ---------------------------------------------------------------------------
# HDF5 reader
# ---------------------------------------------------------------------------


def _h5_attr(obj, key, default=None):
    try:
        return obj.attrs[key]
    except KeyError:
        return default


def _h5_unit_dimension(obj) -> list:
    raw = _h5_attr(obj, "unitDimension")
    if raw is None:
        return [0.0] * 7
    return [float(value) for value in np.asarray(raw).ravel()]


def _h5_meta(obj, units: dict, constants: dict) -> dict:
    return {
        "unit_dimension": _h5_unit_dimension(obj),
        "unitSI": units,
        "constant": constants,
        "macroWeighted": int(_h5_attr(obj, "macroWeighted", 0)),
        "weightingPower": float(_h5_attr(obj, "weightingPower", 0.0)),
    }


def _h5_read_record(record, count: int) -> tuple[dict, dict, dict]:
    """Return (components, metadata, constants) for one record."""
    import h5py

    constants = {}
    if isinstance(record, h5py.Dataset):
        name = record.name.rsplit("/", 1)[-1]
        components = {name: np.asarray(record[()]).copy()}
        units = {name: float(_h5_attr(record, "unitSI", 1.0))}
        return components, _h5_meta(record, units, constants), constants

    # constant record stored as the record group itself
    if "value" in record.attrs and "shape" in record.attrs:
        name = record.name.rsplit("/", 1)[-1]
        value = float(record.attrs["value"])
        shape = int(np.asarray(record.attrs["shape"]).ravel()[0])
        size = count if count > 0 else shape
        constants[name] = value
        components = {name: np.full(size, value, dtype=np.float64)}
        units = {name: float(_h5_attr(record, "unitSI", 1.0))}
        return components, _h5_meta(record, units, constants), constants

    components = {}
    units = {}
    for name, child in record.items():
        if isinstance(child, h5py.Dataset):
            components[name] = np.asarray(child[()]).copy()
            units[name] = float(_h5_attr(child, "unitSI", 1.0))
        else:
            value = float(child.attrs["value"])
            shape = int(np.asarray(child.attrs["shape"]).ravel()[0])
            size = count if count > 0 else shape
            components[name] = np.full(size, value, dtype=np.float64)
            units[name] = float(_h5_attr(child, "unitSI", 1.0))
            constants[name] = value
    return components, _h5_meta(record, units, constants), constants


def read_hdf5(path: str, iteration: int, species: str) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        group = handle[f"data/{iteration}/particles/{species}"]

        patches = None
        patch_group = group.get("particlePatches")
        if patch_group is not None:
            patches = {
                "numParticles": np.asarray(patch_group["numParticles"][()]).copy(),
                "numParticlesOffset": np.asarray(patch_group["numParticlesOffset"][()]).copy(),
                "offset": np.stack(
                    [np.asarray(patch_group["offset"][axis][()]) for axis in "xyz"],
                    axis=1,
                ),
                "extent": np.stack(
                    [np.asarray(patch_group["extent"][axis][()]) for axis in "xyz"],
                    axis=1,
                ),
            }

        count = int(np.sum(patches["numParticles"])) if patches is not None else 0

        data = {}
        meta = {}
        constants = {}
        for record_name in sorted(group):
            if record_name == "particlePatches":
                continue
            components, record_meta, record_constants = _h5_read_record(group[record_name], count)
            if count == 0:
                count = len(next(iter(components.values())))
            meta[record_name] = record_meta
            for name, value in record_constants.items():
                constants.setdefault(record_name, {})[name] = value
            if record_name in VECTOR_RECORDS and all(axis in components for axis in POSITION_OFFSET_NAMES):
                data[record_name] = np.stack([components[axis] for axis in POSITION_OFFSET_NAMES], axis=1)
            else:
                data[record_name] = next(iter(components.values()))

        attrs = {
            "grid": [int(v) for v in np.asarray(_h5_attr(group, "grid", []))]
            if _h5_attr(group, "grid") is not None
            else None,
            "cellSizeSI": [float(v) for v in np.asarray(_h5_attr(group, "cellSizeSI", []))]
            if _h5_attr(group, "cellSizeSI") is not None
            else None,
        }
        if attrs["grid"] is None and patches is not None:
            attrs["grid"] = (patches["offset"] + patches["extent"]).max(axis=0).astype(int).tolist()
        if attrs["cellSizeSI"] is None and "position" in meta:
            attrs["cellSizeSI"] = [float(meta["position"]["unitSI"].get(axis, 1.0)) for axis in POSITION_OFFSET_NAMES]

        return {
            "backend": "HDF5",
            "format": "hdf5",
            "data": data,
            "meta": meta,
            "constants": constants,
            "patches": patches,
            "attrs": attrs,
        }


# ---------------------------------------------------------------------------
# openPMD / ADIOS2 reader
# ---------------------------------------------------------------------------


def read_openpmd(path: str, iteration: int, species: str) -> dict:
    import openpmd_api as io

    series = io.Series(path, io.Access.read_only)
    it = series.iterations[iteration]
    it.open()
    sp = it.particles[species]
    n = int(sp["position"]["x"].shape[0])

    def load(component):
        array = component.load_chunk([0], [n])
        series.series_flush()
        return np.asarray(array).copy()

    data = {}
    meta = {}
    constants = {}
    for record_name in sorted(sp):
        if record_name == "particlePatches":
            continue
        record = sp[record_name]
        record_meta = {
            "unit_dimension": [float(v) for v in record.unit_dimension],
            "unitSI": {},
            "constant": {},
            "macroWeighted": int(record.get_attribute("macroWeighted")) if "macroWeighted" in record.attributes else 0,
            "weightingPower": float(record.get_attribute("weightingPower"))
            if "weightingPower" in record.attributes
            else 0.0,
        }
        components = {}
        for component_name in record:
            component = record[component_name]
            record_meta["unitSI"][component_name] = float(component.unit_SI)
            if component.constant:
                value = float(component.get_attribute("value"))
                constants.setdefault(record_name, {})[component_name] = value
                record_meta["constant"][component_name] = value
                components[component_name] = np.full(n, value, dtype=np.float64)
            else:
                components[component_name] = load(component)
        meta[record_name] = record_meta
        if record_name in VECTOR_RECORDS:
            data[record_name] = np.stack([components[axis] for axis in POSITION_OFFSET_NAMES], axis=1)
        else:
            data[record_name] = next(iter(components.values()))

    patches = None
    patch_records = sp.particle_patches
    if len(list(patch_records)) > 0:

        def load_patch(component):
            array = component.load_chunk([0], [1])
            series.series_flush()
            return np.asarray(array).copy()

        patches = {
            "numParticles": load_patch(patch_records["numParticles"][io.Record.SCALAR]),
            "numParticlesOffset": load_patch(patch_records["numParticlesOffset"][io.Record.SCALAR]),
            "offset": np.stack([load_patch(patch_records["offset"][axis]) for axis in "xyz"], axis=1),
            "extent": np.stack([load_patch(patch_records["extent"][axis]) for axis in "xyz"], axis=1),
        }

    attrs = {"grid": None, "cellSizeSI": None}
    try:
        if "grid" in sp.attributes:
            attrs["grid"] = [int(v) for v in np.asarray(sp.get_attribute("grid")).ravel()]
        if "cellSizeSI" in sp.attributes:
            attrs["cellSizeSI"] = [float(v) for v in np.asarray(sp.get_attribute("cellSizeSI")).ravel()]
    except Exception:
        pass
    if attrs["grid"] is None and patches is not None:
        attrs["grid"] = (patches["offset"] + patches["extent"]).max(axis=0).astype(int).tolist()
    if attrs["cellSizeSI"] is None and "position" in meta:
        attrs["cellSizeSI"] = [float(meta["position"]["unitSI"].get(axis, 1.0)) for axis in POSITION_OFFSET_NAMES]

    backend = str(series.backend)
    return {
        "backend": backend,
        "format": "bp5" if "ADIOS" in backend.upper() else backend.lower(),
        "data": data,
        "meta": meta,
        "constants": constants,
        "patches": patches,
        "attrs": attrs,
    }


def read_series(path: str, iteration: int = 0, species: str = DEFAULT_SPECIES) -> dict:
    if path.endswith((".h5", ".hdf5")):
        return read_hdf5(path, iteration, species)
    return read_openpmd(path, iteration, species)


# ---------------------------------------------------------------------------
# Decoding and validation
# ---------------------------------------------------------------------------


def _unit_vector(meta_record) -> np.ndarray:
    units = meta_record["unitSI"]
    return np.asarray(
        [float(units.get(axis, 1.0)) for axis in POSITION_OFFSET_NAMES],
        dtype=np.float64,
    )


def decode_physical(result: dict) -> dict:
    """Decode physical positions (m) and single-electron u from the raw records."""
    data = result["data"]
    meta = result["meta"]
    position_units = _unit_vector(meta["position"])
    offset_units = _unit_vector(meta["positionOffset"])
    physical = (data["position"] + data["positionOffset"]) * position_units

    momentum_units = np.asarray(
        [float(meta["momentum"]["unitSI"].get(axis, 1.0)) for axis in POSITION_OFFSET_NAMES],
        dtype=np.float64,
    )
    weighting_power = float(meta["weighting"]["weightingPower"])
    weight_factor = np.power(data["weighting"], weighting_power)[:, None]
    u = data["momentum"] * momentum_units / MOMENTUM_UNIT_SI / weight_factor
    return {
        "position": physical,
        "position_units": position_units,
        "offset_units": offset_units,
        "u": u,
    }


def momentum_stats(data: dict, u: np.ndarray | None = None) -> dict:
    if u is None:
        u = data["momentum"] / (ELECTRON_MASS * SPEED_OF_LIGHT)
    umag = np.linalg.norm(u, axis=1)
    gamma = np.sqrt(1.0 + umag * umag)
    return {
        "u_min": float(umag.min()),
        "u_max": float(umag.max()),
        "u_mean": float(umag.mean()),
        "gamma_min": float(gamma.min()),
        "gamma_max": float(gamma.max()),
        "gamma_mean": float(gamma.mean()),
        "beta_mean": float((umag / gamma).mean()),
    }


def total_charge(result: dict) -> tuple[float, float]:
    data = result["data"]
    record = result["meta"]["charge"]
    raw = result["constants"]["charge"]["charge"]
    unit = record["unitSI"].get("charge", 1.0)
    power = float(record["weightingPower"])
    physical = raw * unit
    total = physical * float(np.sum(np.power(data["weighting"], power)))
    return physical, total


def validate_readback(result: dict) -> tuple[dict, list]:
    problems = []
    data = result["data"]
    meta = result["meta"]
    n = int(data["position"].shape[0])
    if n < 1:
        problems.append("no particles")
    for record, array in data.items():
        if not np.all(np.isfinite(array)):
            problems.append(f"{record}: non-finite values")

    weights = data["weighting"]
    if np.any(weights <= 0.0):
        problems.append("non-positive weights")
    if "id" in data:
        ids = data["id"]
        if ids.dtype != np.uint64:
            problems.append(f"id dtype {ids.dtype} is not uint64")
        if len(np.unique(ids)) != len(ids):
            problems.append("duplicate ids")

    decoded = decode_physical(result)
    position = decoded["position"]
    fraction = data["position"]
    offsets = data["positionOffset"]
    if np.any(fraction < 0.0) or np.any(fraction >= 1.0 + 1e-6):
        problems.append("position raw values are not in-cell fractions [0,1)")
    if not np.allclose(offsets, np.round(offsets), rtol=0.0, atol=1e-9):
        problems.append("positionOffset is not an integer cell index")
    if not np.allclose(decoded["position_units"], decoded["offset_units"], rtol=0.0, atol=0.0):
        problems.append("position and positionOffset unitSI differ")

    # openPMD conventions required by this interface
    if "position" in meta and meta["position"]["macroWeighted"] != 0:
        problems.append("position: macroWeighted != 0")
    if "positionOffset" in meta and meta["positionOffset"]["macroWeighted"] != 0:
        problems.append("positionOffset: macroWeighted != 0")
    if "momentum" in meta:
        if meta["momentum"]["macroWeighted"] != 1:
            problems.append("momentum: macroWeighted != 1")
        if meta["momentum"]["weightingPower"] != 1.0:
            problems.append("momentum: weightingPower != 1.0")
    if "weighting" in meta:
        if meta["weighting"]["macroWeighted"] != 1:
            problems.append("weighting: macroWeighted != 1")
        if meta["weighting"]["weightingPower"] != 1.0:
            problems.append("weighting: weightingPower != 1.0")
    if "charge" in meta:
        dim = meta["charge"]["unit_dimension"]
        if dim[3] != 1.0 or dim[2] != 1.0:
            problems.append("charge: unitDimension is not I=1,T=1")
    if "mass" in meta:
        dim = meta["mass"]["unit_dimension"]
        if dim[1] != 1.0:
            problems.append("mass: unitDimension is not M=1")

    if result["patches"] is not None:
        num = result["patches"]["numParticles"]
        if int(np.sum(num)) != n:
            problems.append(f"patch numParticles sum {int(np.sum(num))} != particle count {n}")

    grid = result["attrs"].get("grid")
    cell = result["attrs"].get("cellSizeSI")
    if grid and cell:
        extent = np.asarray(grid, dtype=np.float64) * np.asarray(cell, dtype=np.float64)
        if np.any(position < -1e-6 * np.asarray(cell)) or np.any(position >= extent + 1e-6 * np.asarray(cell)):
            problems.append("decoded physical positions outside the admitted volume")

    charge_per_macro, charge_total = total_charge(result)
    u = decoded["u"]
    stats = momentum_stats(data, u)
    summary = {
        "count": n,
        "total_weight": float(weights.sum()),
        "total_charge_C": charge_total,
        "charge_per_macro_SI": charge_per_macro,
        "momentum": stats,
        "grid": grid,
        "cellSizeSI": cell,
        "backend": result["backend"],
        "has_patches": result["patches"] is not None,
        "physical_position_min_SI": [float(v) for v in position.min(axis=0)],
        "physical_position_max_SI": [float(v) for v in position.max(axis=0)],
    }
    return summary, problems


def compare_readbacks(left: dict, right: dict) -> list:
    problems = []
    for record in VECTOR_RECORDS:
        lhs = left["data"].get(record)
        rhs = right["data"].get(record)
        if lhs is None or rhs is None:
            problems.append(f"missing record {record}")
            continue
        if lhs.shape != rhs.shape:
            problems.append(f"{record}: shape {lhs.shape} != {rhs.shape}")
            continue
        if not np.array_equal(lhs, rhs):
            bad = np.argwhere(lhs != rhs)
            problems.append(f"{record}: {len(bad)} differing values (first {bad[0]})")
    for record in SCALAR_RECORDS:
        lhs = left["data"].get(record)
        rhs = right["data"].get(record)
        if lhs is None or rhs is None:
            problems.append(f"missing record {record}")
            continue
        if lhs.dtype != rhs.dtype or lhs.shape != rhs.shape:
            problems.append(f"{record}: dtype/shape {lhs.dtype}{lhs.shape} != {rhs.dtype}{rhs.shape}")
            continue
        if not np.array_equal(lhs, rhs):
            problems.append(f"{record}: arrays differ")
    for record in CONSTANT_RECORDS:
        lhs = left["constants"].get(record, {})
        rhs = right["constants"].get(record, {})
        if lhs != rhs:
            problems.append(f"{record}: constants {lhs} != {rhs}")

    for record in left["meta"]:
        if record not in right["meta"]:
            problems.append(f"metadata missing record {record}")
            continue
        lhs = left["meta"][record]
        rhs = right["meta"][record]
        for key in ("unit_dimension", "macroWeighted", "weightingPower", "unitSI"):
            if lhs.get(key) != rhs.get(key):
                problems.append(f"{record}.{key}: {lhs.get(key)} != {rhs.get(key)}")
    return problems


def compare_sidecar(result: dict, sidecar_file: str) -> list:
    problems = []
    if not os.path.isfile(sidecar_file):
        return [f"sidecar not found: {sidecar_file}"]
    loaded = np.load(sidecar_file)
    decoded = decode_physical(result)
    data = result["data"]
    checks = (
        ("position", decoded["position"], loaded["position"], 1e-12),
        ("u", decoded["u"], loaded["u"], 1e-12),
        ("weighting", data["weighting"], loaded["weighting"], 0.0),
    )
    for name, actual, expected, tolerance in checks:
        if actual.shape != expected.shape:
            problems.append(f"sidecar {name}: shape {actual.shape} != {expected.shape}")
            continue
        if tolerance == 0.0:
            equal = np.array_equal(actual, expected)
        else:
            equal = np.allclose(actual, expected, rtol=tolerance, atol=0.0)
        if not equal:
            diff = np.max(np.abs(actual - expected))
            problems.append(f"sidecar {name}: mismatch (max |diff| {diff:g})")
    if "id" in data and "id" in loaded:
        if not np.array_equal(data["id"], loaded["id"]):
            problems.append("sidecar id: mismatch")
    return problems


def check_manifest(manifest: dict, series_path: str, data: dict) -> list:
    problems = []
    expected = manifest.get("array_sha256", {})
    actual = array_checksums(data)
    for key, value in expected.items():
        if actual.get(key) != value:
            problems.append(f"array checksum mismatch: {key}")
    for record, values in manifest.get("constant_values", {}).items():
        found = data.get(record)
        if found is None:
            problems.append(f"constant record missing after read: {record}")
        elif not np.all(found == values.get(record, values)):
            problems.append(f"constant record mismatch: {record}")

    entries = manifest.get("files", [])
    for entry in entries:
        absolute = os.path.join(series_path, entry["path"]) if os.path.isdir(series_path) else series_path
        if not os.path.isfile(absolute):
            problems.append(f"manifest file missing: {entry['path']}")
            continue
        if os.path.getsize(absolute) != entry["bytes"]:
            problems.append(f"file size mismatch: {entry['path']}")
        if sha256_file(absolute) != entry["sha256"]:
            problems.append(f"file checksum mismatch: {entry['path']}")
    return problems


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
