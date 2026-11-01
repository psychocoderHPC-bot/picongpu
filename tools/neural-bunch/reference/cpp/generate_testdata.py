"""Generate C++ golden tiles from the validated Python FP64 reference.

Run from this directory:  python3 generate_testdata.py
Output: testdata/<case>.tile (text, 1536 values in flatten order).
"""

from __future__ import annotations

import math
import sys

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import convkernel as ck  # noqa: E402

SETTINGS = {
    "order": 4,
    "tol": 1e-7,
    "max_level": 5,
    "max_order": 16,
}

CASES = [
    {
        "name": "far_iso_gamma1_stationary",
        "x_tile": (0.0, 0.0, 0.0),
        "cells": (1.0, 1.0, 1.0),
        "sources": [((-8.0, 0.5, 0.5), (0.0, 0.0, 0.0), 1.0)],
    },
    {
        "name": "far_iso_gamma2",
        "x_tile": (0.0, 0.0, 0.0),
        "cells": (1.0, 1.0, 1.0),
        "sources": [((-8.0, 0.5, 0.5), (math.sqrt(3.0), 0.0, 0.0), 1.0)],
    },
    {
        "name": "far_iso_gamma10",
        "x_tile": (0.0, 0.0, 0.0),
        "cells": (1.0, 1.0, 1.0),
        "sources": [((-8.0, 0.5, 0.5), (math.sqrt(99.0), 0.0, 0.0), 1.0)],
    },
    {
        "name": "far_iso_gamma100",
        "x_tile": (0.0, 0.0, 0.0),
        "cells": (1.0, 1.0, 1.0),
        "sources": [((-12.0, 0.3, -0.4), (math.sqrt(9999.0), 0.0, 0.0), 1.0)],
    },
    {
        "name": "far_anisotropic",
        "x_tile": (0.2, -0.3, 0.5),
        "cells": (0.7, 1.3, 2.1),
        "sources": [
            (
                (0.2 - 8.0 * 0.7, -0.3 + 0.2 * 1.3, 0.5 - 0.1 * 2.1),
                (0.4, -0.2, 0.1),
                1.7,
            )
        ],
    },
    {
        "name": "far_aspect10_d128",
        "x_tile": (0.0, 0.0, 0.0),
        "cells": (1.0, 10.0, 10.0),
        "sources": [((-128.0, 2.0, -3.0), (math.sqrt(99.0), 0.0, 0.0), 1.3)],
    },
    {
        "name": "far_aspect0p1_d128",
        "x_tile": (0.0, 0.0, 0.0),
        "cells": (1.0, 0.1, 0.1),
        "sources": [((-128.0, 0.02, -0.03), (math.sqrt(99.0), 0.0, 0.0), 0.9)],
    },
    {
        "name": "far_aniso_independent_d32",
        "x_tile": (0.0, 0.0, 0.0),
        "cells": (1.0, 2.0, 0.5),
        "sources": [((-32.0, 0.4, -0.15), (0.2, -0.3, 0.1), -1.2)],
    },
    {
        "name": "near_just_outside_support",
        "x_tile": (0.0, 0.0, 0.0),
        "cells": (1.0, 1.0, 1.0),
        "sources": [((9.01, -1.4, -1.4), (0.1, 0.0, 0.0), 1.0)],
    },
    {
        "name": "mixed_sign_weights",
        "x_tile": (0.0, 0.0, 0.0),
        "cells": (1.0, 1.0, 1.0),
        "sources": [
            ((-10.0, 0.2, -0.3), (0.2, 0.1, 0.0), 1.5),
            ((-8.0, -1.0, 1.2), (-0.3, 0.2, 0.1), -0.7),
            ((12.0, 0.4, 0.6), (0.5, -0.1, 0.2), 2.2),
        ],
    },
]


def main():
    out_dir = HERE / "testdata"
    out_dir.mkdir(exist_ok=True)
    for case in CASES:
        x_tile = case["x_tile"]
        dx, dy, dz = case["cells"]
        particles = [
            {"x": x, "u": u, "w": w} for (x, u, w) in case["sources"]
        ]
        tile = ck.tile_field(
            x_tile,
            particles,
            dx,
            dy,
            dz,
            order=SETTINGS["order"],
            tol=SETTINGS["tol"],
            max_level=SETTINGS["max_level"],
            max_order=SETTINGS["max_order"],
        )
        flat = ck.flatten_tile(tile)
        lines = [
            f"name {case['name']}",
            f"dx {dx!r} {dy!r} {dz!r}",
            f"x_tile {x_tile[0]!r} {x_tile[1]!r} {x_tile[2]!r}",
            f"nsrc {len(case['sources'])}",
        ]
        for x, u, w in case["sources"]:
            lines.append(
                "src "
                + " ".join(repr(float(v)) for v in (*x, *u, float(w)))
            )
        lines += [
            f"order {SETTINGS['order']}",
            f"tol {SETTINGS['tol']!r}",
            f"max_level {SETTINGS['max_level']}",
            f"max_order {SETTINGS['max_order']}",
            "tile",
        ]
        lines.append(" ".join(repr(float(v)) for v in flat))
        path = out_dir / f"{case['name']}.tile"
        path.write_text("\n".join(lines) + "\n")
        print("wrote", path, "flat", flat.shape)

    # Shared aspect-corner shape_field vectors for the C++ test.
    aspects = [0.1, 0.5, 1.0, 2.0, 10.0]
    pairs = (
        [(a, a) for a in aspects]
        + [(a, 1.0) for a in aspects]
        + [(1.0, a) for a in aspects]
    )
    lines = ["# dx dy dz x y z ux uy uz Ex Ey Ez Bx By Bz"]
    for dy, dz in pairs:
        for d in (4.0, 8.0, 32.0, 128.0, 512.0):
            for gamma in (1.0, 10.0, 100.0):
                cells = np.array([1.0, dy, dz])
                R0 = np.array([d, 0.3, -0.2]) * cells
                u = np.array(
                    [0.0 if gamma == 1.0 else math.sqrt(gamma * gamma - 1.0),
                     0.0, 0.0]
                )
                E, B = ck.shape_field(
                    R0, np.zeros(3), u, *cells, order=4
                )
                values = [*cells, *R0, *u, *E, *B]
                lines.append(" ".join(repr(float(v)) for v in values))
    aspect_path = out_dir / "aspect_cases.txt"
    aspect_path.write_text("\n".join(lines) + "\n")
    print("wrote", aspect_path, "cases", len(lines) - 1)

    # Near set outside the support (Python is converged there), for the
    # Python/C++ <=1e-9 cross-check.
    src = np.array([4.0, 4.0, 2.0])
    u = np.array([0.3, 0.1, 0.0])
    pos = ck.output_positions(np.zeros(3), 1.0, 1.0, 1.0)
    candidates = []
    for i in range(8):
        for j in range(8):
            for k in range(4):
                for c in range(6):
                    R0 = pos[i, j, k, c] - src
                    excess = np.maximum(np.abs(R0) - 1.5, 0.0)
                    d = float(np.linalg.norm(excess))
                    if 0.0 < d <= 3.0:
                        candidates.append((d, i, j, k, c))
    candidates.sort()
    stride = max(1, len(candidates) // 24)
    selected = candidates[::stride][:24]
    near_lines = ["# dx dy dz x y z ux uy uz Ex Ey Ez Bx By Bz"]
    for d, i, j, k, c in selected:
        p = pos[i, j, k, c]
        E, B = ck.shape_field(p, src, u, 1.0, 1.0, 1.0)
        values = [1.0, 1.0, 1.0, *(p - src), *u, *E, *B]
        near_lines.append(" ".join(repr(float(v)) for v in values))
    near_path = out_dir / "near_cases.txt"
    near_path.write_text("\n".join(near_lines) + "\n")
    print("wrote", near_path, "cases", len(near_lines) - 1)


if __name__ == "__main__":
    main()
