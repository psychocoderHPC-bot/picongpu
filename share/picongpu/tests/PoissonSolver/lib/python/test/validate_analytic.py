#!/usr/bin/env python3
#
# Copyright 2026-2026 PIConGPU contributors
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

"""Analytic regression test for the Poisson solver.

Three closed-form cases are checked against the electrostatic potential written
to openPMD by the Poisson stage:

  A  fully periodic sine mode          (continuum eigenpair, O(h^2) error)
  B  grounded Dirichlet polynomial     (exact for the 7-point stencil)
  C  uniform-density slab, Dirichlet x (exact for the 7-point stencil)

Metrics (never divide per point by phi or rho):

  E_max = max|V - phi| / max|phi|
  E_L2  = sqrt(sum|V - phi|^2 / sum|phi|^2)
  R     = ||L_h(V) - rho/eps0||_2 / ||rho/eps0||_2

The discrete operator L_h is the negative 7-point Laplacian used by the solver.
Periodic axes wrap; Dirichlet axes use zero ghost nodes.

Before any field comparison the validator checks the sign and relative L2 error
of the charge density that the solver actually saw. The iteration-0 openPMD
charge-density meshes are not populated, so the effective injected density is
reconstructed as L_h(V) (the same operator used by R). A sign-flipped injection
(rho_injected ~ -rho_analytic, e.g. from swapped species charge ratios) fails
here with a dedicated message instead of only showing up as E_max ~ 2.
"""

import argparse
import sys

import numpy as np
import openpmd_api as io
from scipy.constants import epsilon_0

#! Potential amplitude of Cases A and B and centre potential of Case C [V].
PHI0 = 1.0

CASES = {
    # name: (periodic axes, E_max, E_L2, R)
    "A": ((True, True, True), 5e-3, 2e-3, 1e-6),
    "B": ((False, False, False), 1e-5, 5e-6, 1e-6),
    "C": ((False, True, True), 1e-5, 5e-6, 1e-6),
}


def apply_stencil(V, h, periodic):
    """Apply the solver's negative 7-point Laplacian with case boundary ghosts."""
    N = V.shape
    padded = np.zeros(tuple(n + 2 for n in N), dtype=np.float64)
    padded[1:-1, 1:-1, 1:-1] = V

    for axis in range(3):
        if not periodic[axis]:
            # Homogeneous Dirichlet ghosts stay zero.
            continue
        lower = [slice(1, -1)] * 3
        upper = [slice(1, -1)] * 3
        source_lower = [slice(1, -1)] * 3
        source_upper = [slice(1, -1)] * 3
        lower[axis] = 0
        upper[axis] = N[axis] + 1
        # Interior node i (0-based) lives at padded[i + 1]. The periodic
        # neighbour below node 0 is node N - 1, which sits at padded[N], not at
        # padded[N - 1]. The neighbour above node N - 1 is node 0 at padded[1].
        source_lower[axis] = N[axis]
        source_upper[axis] = 1
        padded[tuple(lower)] = padded[tuple(source_lower)]
        padded[tuple(upper)] = padded[tuple(source_upper)]

    center = padded[1:-1, 1:-1, 1:-1]
    result = np.zeros_like(V)
    for axis in range(3):
        minus = [slice(1, -1)] * 3
        plus = [slice(1, -1)] * 3
        minus[axis] = slice(0, -2)
        plus[axis] = slice(2, None)
        result += (2.0 * center - padded[tuple(minus)] - padded[tuple(plus)]) / h[axis] ** 2
    return result


def analytic_fields(case, N, h):
    """Evaluate the analytic potential and charge density at the node coordinates."""
    periodic = CASES[case][0]
    shift = np.array([0.0 if p else 1.0 for p in periodic])
    L = (np.asarray(N, dtype=np.float64) + shift) * h

    x = (np.arange(N[0]) + shift[0]) * h[0]
    y = (np.arange(N[1]) + shift[1]) * h[1]
    z = (np.arange(N[2]) + shift[2]) * h[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    if case == "A":
        kx = 2.0 * np.pi / L[0]
        ky = 2.0 * np.pi / L[1]
        kz = 2.0 * np.pi / L[2]
        phi = PHI0 * np.sin(kx * X) * np.sin(ky * Y) * np.sin(kz * Z)
        # The sine is an exact discrete eigenvector; use the discrete eigenvalue.
        N = np.asarray(N)
        lambda_discrete = sum(
            4.0 * np.sin(np.pi / N[d]) ** 2 / h[d] ** 2 for d in range(3)
        )
        rho = epsilon_0 * lambda_discrete * phi
    elif case == "B":
        px = X * (L[0] - X)
        py = Y * (L[1] - Y)
        pz = Z * (L[2] - Z)
        norm = L[0] ** 2 * L[1] ** 2 * L[2] ** 2
        phi = 64.0 * PHI0 * px * py * pz / norm
        rho = 128.0 * epsilon_0 * PHI0 * (py * pz + px * pz + px * py) / norm
    else:  # case C
        rho0 = 8.0 * epsilon_0 * PHI0 / L[0] ** 2
        phi = rho0 / (2.0 * epsilon_0) * X * (L[0] - X)
        rho = np.full_like(phi, rho0)

    return phi, rho, periodic


def relative_norm(values, reference):
    return float(np.sqrt(np.sum(values**2) / np.sum(reference**2)))


#! Generous sanity gate for the density check. It only catches gross injection
#! errors; the scientific accuracy is asserted by R below.
RHO_L2_SANITY = 1.0e-1


def check_injected_density(V, rho, h, periodic):
    """Compare the effective injected charge density with the analytic rho.

    The solver output V satisfies L_h(V) = rho_injected / eps0. Reconstructing
    that right-hand side is the only way to inspect the injected density: the
    iteration-0 openPMD charge-density meshes are all zero.

    Returns ``(sign_dot, rel_l2)`` where ``sign_dot`` is the inner product of
    the reconstructed and the analytic right-hand sides. A negative value means
    the injected density has the wrong sign.
    """
    rhs_effective = apply_stencil(V, h, periodic)
    rhs_analytic = rho / epsilon_0
    sign_dot = float(np.sum(rhs_effective * rhs_analytic))
    rel_l2 = relative_norm(rhs_effective - rhs_analytic, rhs_analytic)
    return sign_dot, rel_l2


def main():
    parser = argparse.ArgumentParser(description="Analytic Poisson solver regression test")
    parser.add_argument("-r", dest="path", required=True, help="Path to the simulation results")
    parser.add_argument("-f", dest="file", required=True, help="openPMD file base name")
    parser.add_argument("-c", dest="case", required=True, choices=sorted(CASES), help="Analytic case")
    parser.add_argument(
        "-e", dest="ext", default="bp", help="openPMD file extension (default: bp)"
    )
    args = parser.parse_args()

    filename = f"{args.path}/openPMD/{args.file}_000000.{args.ext}"
    try:
        series = io.Series(filename, io.Access.read_only)
    except Exception as error:  # noqa: BLE001 - report the concrete file path
        print(f"cannot open '{filename}': {error}")
        return 2

    try:
        comp = series.iterations[0].meshes["FieldV"]
        V = np.asarray(comp[:, :, :], dtype=np.float64)
        series.flush()
        unit_si = float(comp.get_attribute("unitSI"))
        h = (
            np.asarray(comp.get_attribute("gridSpacing"), dtype=np.float64)[::-1]
            * float(comp.get_attribute("gridUnitSI"))
        )
    except KeyError:
        print("openPMD file does not contain the 'FieldV' mesh")
        return 2

    V = np.transpose(V) * unit_si
    phi, rho, periodic = analytic_fields(args.case, V.shape, h)

    phi_max = np.max(np.abs(phi))
    if phi_max == 0.0:
        print("analytic potential is identically zero")
        return 2

    # Density check first: a wrong-sign injection must not be reported as a
    # generic field mismatch (it would otherwise appear as E_max ~ 2).
    rho_dot, rho_rel_l2 = check_injected_density(V, rho, h, periodic)
    print(f"case {args.case}: injected density check")
    print(f"  rho relative L2 = {rho_rel_l2:.6e}  (sanity limit {RHO_L2_SANITY:.1e})")
    print(f"  rho sign dot    = {rho_dot:+.6e}  (must be > 0)")
    if rho_dot <= 0.0:
        print(
            "analytic Poisson regression test FAILED: the injected charge density has the "
            "wrong sign relative to the analytic rho (rho_injected ~ -rho_analytic). Check "
            "the species chargeRatio values (BASE_CHARGE_SI is negative) and the "
            "speciesInitialization density-profile wiring."
        )
        return 1
    if rho_rel_l2 > RHO_L2_SANITY:
        print(
            "analytic Poisson regression test FAILED: the injected charge density differs "
            f"from the analytic rho by a relative L2 error of {rho_rel_l2:.6e}, above the "
            f"sanity limit {RHO_L2_SANITY:.1e}."
        )
        return 1

    E_max = float(np.max(np.abs(V - phi)) / phi_max)
    E_L2 = relative_norm(V - phi, phi)

    rhs = rho / epsilon_0
    residual = apply_stencil(V, h, periodic) - rhs
    R = relative_norm(residual, rhs)

    _, E_max_tol, E_L2_tol, R_tol = CASES[args.case]
    print(f"case {args.case}: grid {V.shape}, spacing {h}")
    print(f"  E_max = {E_max:.6e}  (limit {E_max_tol:.1e})")
    print(f"  E_L2  = {E_L2:.6e}  (limit {E_L2_tol:.1e})")
    print(f"  R     = {R:.6e}  (limit {R_tol:.1e})")

    failed = E_max > E_max_tol or E_L2 > E_L2_tol or R > R_tol
    if failed:
        print("analytic Poisson regression test FAILED")
        return 1
    print("analytic Poisson regression test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
