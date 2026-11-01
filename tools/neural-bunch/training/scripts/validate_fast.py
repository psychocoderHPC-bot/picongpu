#!/usr/bin/env python3
"""Validate the compiled C++ reference against the Python golden tiles.

Compares the pybind11 extension ``_nbref_cpp`` (built from
``reference/cpp/neural_reference.hpp``) to the checked-in Python-generated
golden vectors in ``reference/cpp/testdata/*.tile`` and, when importable, to a
live ``reference/convkernel.py`` evaluation of the same cases.

Exit code 0 only if every comparison is <= 1e-9 relative.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent.parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from neuralbunch.fast_reference import load_fast_reference  # noqa: E402


def _read_tile(path: Path):
    lines = path.read_text().splitlines()
    dx = dy = dz = None
    x_tile = None
    sources = []
    settings = {}
    flat = None
    want_tile = False
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if want_tile:
            flat = np.array([float(v) for v in parts], dtype=np.float64)
            want_tile = False
            continue
        if tag == "dx":
            dx, dy, dz = (float(v) for v in parts[1:4])
        elif tag == "x_tile":
            x_tile = tuple(float(v) for v in parts[1:4])
        elif tag == "src":
            vals = [float(v) for v in parts[1:8]]
            sources.append(
                {"x": np.array(vals[0:3]), "u": np.array(vals[3:6]), "w": vals[6]}
            )
        elif tag == "order":
            settings["order"] = int(parts[1])
        elif tag == "tol":
            settings["tol"] = float(parts[1])
        elif tag == "max_level":
            settings["max_level"] = int(parts[1])
        elif tag == "max_order":
            settings["max_order"] = int(parts[1])
        elif tag == "tile":
            want_tile = True
    return x_tile, sources, dx, dy, dz, settings, flat


def _load_python_reference(path: Path):
    try:
        import scipy  # noqa: F401
    except Exception:
        return None
    spec = importlib.util.spec_from_file_location("_golden_convkernel", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdata", default=str(TRAIN_DIR.parent / "reference" / "cpp" / "testdata"))
    ap.add_argument("--python-reference", default=str(TRAIN_DIR.parent / "reference" / "convkernel.py"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=float, default=1e-9)
    args = ap.parse_args(argv)

    mod = load_fast_reference()
    data_dir = Path(args.testdata)
    pyref = _load_python_reference(Path(args.python_reference))
    rows = []
    worst = 0.0
    worst_py = 0.0
    for tile_path in sorted(data_dir.glob("*.tile")):
        x_tile, sources, dx, dy, dz, settings, golden = _read_tile(tile_path)
        got = np.asarray(
            mod.tile_field_flat(x_tile, sources, dx, dy, dz, **settings),
            dtype=np.float64,
        )
        rel = float(np.linalg.norm(got - golden) / max(np.linalg.norm(golden), 1e-300))
        py_rel = None
        if pyref is not None:
            py_tile = pyref.flatten_tile(
                pyref.tile_field(
                    np.asarray(x_tile),
                    [{"x": s["x"], "u": s["u"], "w": s["w"]} for s in sources],
                    dx,
                    dy,
                    dz,
                    **settings,
                )
            )
            py_rel = float(
                np.linalg.norm(got - py_tile) / max(np.linalg.norm(py_tile), 1e-300)
            )
            worst_py = max(worst_py, py_rel)
        worst = max(worst, rel)
        rows.append(
            {
                "case": tile_path.stem,
                "rel_vs_golden": rel,
                "rel_vs_live_python": py_rel,
            }
        )

    result = {
        "extension": str(getattr(mod, "__file__", "?")),
        "cases": rows,
        "max_rel_vs_golden": worst,
        "max_rel_vs_live_python": worst_py if pyref is not None else None,
        "limit": float(args.limit),
        "python_reference_available": pyref is not None,
        "passed": worst <= float(args.limit)
        and (pyref is None or worst_py <= float(args.limit)),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
