===========================
Test for the Poisson solver
===========================

This is an analytic regression test for the Poisson solver used during
PIConGPU initialization. Three closed-form solutions are compared with the
electrostatic potential ``V`` written to openPMD.

The charge density is injected analytically, not sampled with macroparticles:
one macro particle sits exactly on every grid node (``startPosition::AnalyticNode``)
and ``density.param`` returns the analytic charge density divided by the species
charge and ``BASE_DENSITY_SI``. CIC deposition then puts the full weight on that
node, so the grid charge density is exact up to floating-point rounding.

Analytic cases
==============

All cases use a 32^3 grid and run at ``-s 0``; the Poisson stage executes during
initialization, so no time steps are needed.

Case A: periodic sine mode (``--periodic 1 1 1``)
  ``phi = phi0 sin(2 pi x/L) sin(2 pi y/L) sin(2 pi z/L)`` with ``L = N*h`` and
  node coordinates ``x_i = i*h``. ``rho = eps0 * lambda_discrete * phi`` uses
  the discrete eigenvalue
  ``lambda_discrete = sum_d 4 sin^2(pi/N_d) / h_d^2``, so the sine is an exact
  eigenvector of the solver's 7-point operator. Using the continuum eigenvalue
  instead would leave an O(h^2) difference of 3.2e-3 at N = 32, which dominates
  ``E_L2`` and does not test the solve.

Case B: grounded Dirichlet box (``--periodic 0 0 0``)
  ``phi = 64 phi0 x(L-x) y(L-y) z(L-z) / (Lx^2 Ly^2 Lz^2)`` with
  ``L = (N+1)*h`` and core nodes ``x_i = (i+1)*h``. The 7-point Laplacian of
  this polynomial is exact on the grid.

Case C: uniform-density slab (``--periodic 0 1 1``)
  ``rho = rho0`` constant, ``phi = rho0/(2 eps0) x(Lx-x)``. Dirichlet in ``x``,
  periodic in ``y`` and ``z``; exact for the discrete operator.

Cases B and C select homogeneous Dirichlet ghosts. The test input set defines
``PARAM_POISSON_BOUNDARY_CONDITION_DIRICHLET`` in ``simulation.param``; the
production default remains homogeneous Neumann. Case A is fully periodic and is
not affected by this switch. The potential is added to the openPMD output by
the test-only ``fileOutput.param``.

Metrics and thresholds
======================

The validator never divides per point by ``phi`` or ``rho``:

``E_max = max|V - phi| / max|phi|``

``E_L2  = sqrt(sum|V - phi|^2 / sum|phi|^2)``

``R     = ||L_h(V) - rho/eps0||_2 / ||rho/eps0||_2``

``L_h`` is the solver's negative 7-point Laplacian. Periodic axes wrap; Dirichlet
axes use zero ghost nodes.

The 7-point stencil itself, including the Dirichlet and Neumann mirror-ghost
handling and the gradient/divergence adjoint identity, is covered by
``share/picongpu/unit/non_dimensional_tests/poissonStencilUT.cpp``.

==========  =========  =========  =========
case        E_max      E_L2       R
==========  =========  =========  =========
A           5e-3       2e-3       1e-6
B           1e-5       5e-6       1e-6
C           1e-5       5e-6       1e-6
==========  =========  =========  =========

Running the test
================

Run ``ci.sh`` with the input set and a destination directory:

.. code-block:: bash

   ./picongpu/share/picongpu/tests/PoissonSolver/bin/ci.sh \
       picongpu/share/picongpu/tests/PoissonSolver/ ./run

``ci.sh`` builds once and runs the three cases. Each run is a 32^3 grid with
``-r 2`` and ``-s 0``, so the test stays small.
