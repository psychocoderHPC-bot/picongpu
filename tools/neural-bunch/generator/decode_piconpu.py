#!/usr/bin/env python3
"""Decode a real PIConGPU HDF5 particle dump with the same code path as
``inspect_bunch.py`` (``nb_common.read_series`` + ``decode_physical``).

Proves the reader is not tailored to the synthetic generator output.
The default path is the relative ``simData_000000.h5``; override it with
``--series`` or the ``PICONGPU_SERIES`` environment variable.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import nb_common as common

DEFAULT_PATH = os.environ.get("PICONGPU_SERIES", "simData_000000.h5")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default=DEFAULT_PATH)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--species", default="e")
    parser.add_argument("--rows", type=int, default=3)
    args = parser.parse_args(argv)

    result = common.read_series(args.series, args.iteration, args.species)
    summary, problems = common.validate_readback(result)
    decoded = common.decode_physical(result)
    physical = decoded["position"]
    u = decoded["u"]
    umag = np.linalg.norm(u, axis=1)

    print(f"series: {args.series}")
    print(f"  backend={result['backend']} species={args.species} particles={summary['count']}")
    print(f"  cellSizeSI={summary['cellSizeSI']} grid={summary['grid']}")
    print(
        "  physical position (m): "
        f"min={physical.min(axis=0).tolist()} max={physical.max(axis=0).tolist()}"
    )
    print(
        f"  u: min={float(umag.min()):.6g} max={float(umag.max()):.6g} "
        f"mean={float(umag.mean()):.6g} gamma_min={float(np.sqrt(1+umag.min()**2)):.6g} "
        f"gamma_max={float(np.sqrt(1+umag.max()**2)):.6g}"
    )
    print(f"  total_weight={summary['total_weight']!r} total_charge={summary['total_charge_C']!r} C")
    print("  first rows: frac(x,y,z) offset(x,y,z) physical(x,y,z) p_raw(x,y,z) w u(x,y,z)")
    for index in range(min(args.rows, summary["count"])):
        frac = result["data"]["position"][index]
        offset = result["data"]["positionOffset"][index]
        p = result["data"]["momentum"][index]
        w = result["data"]["weighting"][index]
        print(
            "    {:d} ({:.5g},{:.5g},{:.5g}) ({:g},{:g},{:g}) "
            "({:.6g},{:.6g},{:.6g}) ({:.5g},{:.5g},{:.5g}) w={:.6g} "
            "u=({:.5g},{:.5g},{:.5g})".format(index, *frac, *offset, *physical[index], *p, w, *u[index])
        )
    if problems:
        print("  warnings:")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("  validation: OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
