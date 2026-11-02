#!/usr/bin/env python3
"""Order and partition-refinement convergence study for the reference.

For each probe point the script evaluates the same tensor integral while

* raising the Gauss-Legendre order on a fixed partition, and
* bisecting every transformed ``t`` segment (partition refinement).

Observed orders are derived from the errors against the finest run in each
sequence.  No value is fitted or assumed.

Run::

    python convergence.py [--output convergence.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import indfield  # noqa: E402

U10 = np.sqrt(99.0) * np.array([0.6, 0.5, 0.62]) / np.linalg.norm([0.6, 0.5, 0.62])


def _err(a, b):
    a = np.concatenate([np.asarray(a[0]).ravel(), np.asarray(a[1]).ravel()])
    b = np.concatenate([np.asarray(b[0]).ravel(), np.asarray(b[1]).ravel()])
    return float(np.max(np.abs(a - b)))


def _run(point, order, refine):
    e, b = indfield.integrate_point(point["x_obs"], point["x_src"],
                                    point["u"], *point["cells"],
                                    order=order, refine=refine)
    return e, b


def _study(point):
    name = point["name"]
    print(f"[{name}] s*={(np.asarray(point['x_obs'])-np.asarray(point['x_src']))/np.asarray(point['cells'])}",
          flush=True)

    # Partition refinement at fixed order.
    pref = _run(point, 8, 3)
    part = []
    for r in (0, 1, 2, 3):
        val = _run(point, 8, r)
        part.append({"refine": r, "error": _err(val, pref)})
    print("  partition:", [f"{p['error']:.3e}" for p in part], flush=True)
    for i in range(len(part) - 1):
        part[i]["observed_order"] = (
            float(np.log(part[i]["error"] / part[i + 1]["error"]) / np.log(2.0))
            if part[i + 1]["error"] > 0 and part[i]["error"] > 0 else None
        )
    part[-1]["observed_order"] = None

    # Order refinement on the refined partition.
    oref = _run(point, 16, 2)
    orders = []
    for o in (6, 8, 10, 12, 14, 16):
        val = _run(point, o, 2)
        orders.append({"order": o, "error": _err(val, oref)})
    print("  order:    ", [f"{p['error']:.3e}" for p in orders], flush=True)
    for i in range(len(orders) - 1):
        orders[i]["observed_order"] = (
            float(np.log(orders[i]["error"] / orders[i + 1]["error"])
                  / np.log(orders[i + 1]["order"] / orders[i]["order"]))
            if orders[i + 1]["error"] > 0 and orders[i]["error"] > 0 else None
        )
    orders[-1]["observed_order"] = None

    return {"name": name, "partition": part, "order": orders,
            "error_floor_indicator": part[-1]["error"]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    points = [
        {"name": "stationary_inside", "x_obs": [0.2, 0.3, 0.4],
         "x_src": [0.0, 0.0, 0.0], "u": np.zeros(3), "cells": [1.0, 1.0, 1.0]},
        {"name": "stationary_edge", "x_obs": [0.5, 0.5, 0.5],
         "x_src": [0.0, 0.0, 0.0], "u": np.zeros(3), "cells": [1.0, 1.0, 1.0]},
        {"name": "gamma10_aspect_oblique", "x_obs": [0.2, 0.35, 0.4],
         "x_src": [0.0, 0.0, 0.0], "u": U10, "cells": [1.0, 0.7, 1.3]},
    ]
    t0 = time.time()
    report = {"transform_power": indfield.TRANSFORM_POWER,
              "points": [_study(p) for p in points],
              "wall_seconds": None}
    report["wall_seconds"] = round(time.time() - t0, 1)

    for p in report["points"]:
        porders = [r["observed_order"] for r in p["partition"][:-1]]
        oorders = [r["observed_order"] for r in p["order"][:-1]]
        print(f"SUMMARY {p['name']}: partition orders={porders} "
              f"order-refine orders={oorders} floor~{p['error_floor_indicator']:.2e}")

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(f"wall {report['wall_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
