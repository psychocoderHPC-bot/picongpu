"""Load the compiled C++ FP64 reference (build with cpp/build_fast.sh)."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent


def find_extension(root=None) -> Path:
    root = Path(root or os.environ.get("NEURALBUNCH_FAST_ROOT", DEFAULT_ROOT))
    matches = sorted(root.glob("_nbref_cpp*.so"))
    if not matches:
        raise FileNotFoundError(
            f"compiled _nbref_cpp extension not found under {root}; "
            "run cpp/build_fast.sh"
        )
    return matches[0]


def load_fast_reference(path=None):
    """Import the pybind11 extension as a module with ``tile_field``."""
    if path is None:
        try:
            import _nbref_cpp as mod  # type: ignore

            return mod
        except ImportError:
            path = find_extension()
    path = Path(path)
    loader = importlib.machinery.ExtensionFileLoader("_nbref_cpp", str(path))
    spec = importlib.util.spec_from_file_location("_nbref_cpp", str(path), loader=loader)
    if spec is None:
        raise ImportError(f"cannot load extension {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "tile_field"):
        raise ImportError(f"{path} has no tile_field")
    return mod
