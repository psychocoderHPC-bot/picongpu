"""Case definitions shared by the CLI and the comparison driver.

Each case describes one source and one Yee tile.  The source shape is the
lab-frame TSC shape (no Lorentz contraction); ``u`` is the dimensionless
momentum ``p/(m_e c)`` so ``gamma = sqrt(1+|u|**2)``.
"""

from __future__ import annotations

import numpy as np


def _u(gamma, direction):
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    return np.sqrt(gamma * gamma - 1.0) * d


def default_cases():
    """The four comparison cases required by the T13 brief."""
    return [
        {
            "name": "stationary_center",
            "x_tile": [0.0, 0.0, 0.0],
            "cells": [1.0, 1.0, 1.0],
            "source": [3.5, 3.5, 1.5],
            "u": [0.0, 0.0, 0.0],
            "w": 1.0,
        },
        {
            "name": "gamma10_oblique",
            "x_tile": [0.0, 0.0, 0.0],
            "cells": [1.0, 1.0, 1.0],
            "source": [3.7, 3.2, 1.4],
            "u": _u(10.0, [0.6, 0.5, 0.62]).tolist(),
            "w": 1.0,
        },
        {
            "name": "aspect10",
            "x_tile": [0.0, 0.0, 0.0],
            "cells": [1.0, 10.0, 0.1],
            "source": [3.5, 8.0, 0.15],
            "u": [0.0, 0.0, 0.0],
            "w": 1.0,
        },
        {
            # Source exactly on the Ex staggered position of cell (1,1,1).
            "name": "source_at_output",
            "x_tile": [0.0, 0.0, 0.0],
            "cells": [1.0, 1.0, 1.0],
            "source": [1.5, 1.0, 1.0],
            "u": _u(10.0, [0.6, 0.5, 0.62]).tolist(),
            "w": 1.0,
        },
    ]


def particles_of(case):
    return [{"x": np.asarray(case["source"], dtype=np.float64),
             "u": np.asarray(case["u"], dtype=np.float64),
             "w": float(case.get("w", 1.0))}]
