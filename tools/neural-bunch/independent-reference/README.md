# Independent TSC field reference

This directory is a self-contained, independently formulated FP64 reference for
the field of TSC-shaped charged sources on the six staggered Yee positions.  It
exists so that a green field test cannot be produced by two implementations
sharing one numerical bug (guide E2).

## Why it is independent

| aspect | in-repo reference (`reference/convkernel.py`) | this implementation |
|---|---|---|
| singularity handling | per-axis geometric grading of Gauss panels | knot split + coordinate power transform `s = c + (b-c) t**p` |
| panels | fixed three knot panels, graded around projection | knot split, then split exactly at the projection |
| rule | Gauss-Legendre order pair + Richardson level | tensor Gauss-Legendre on transformed segments |
| point kernel algebra | `R_par / R_perp` decomposition | `D = |R|²(1-β²) + (R·β)²` |
| error control | order pair + grading + order increase | order refinement and partition bisection, reported |

No node or weight table is imported from the repository.  The TSC shape is the
same physical definition (it is the input contract, not a numerical choice).

## Formulation

For a source at `x_src`, cell sizes `(dx,dy,dz)`, and observation point `x_obs`:

```
E_n(x_obs) = ∫ S(s) K(R0 - s*cell, u) ds³,   R0 = x_obs - x_src,
S(s) = W(s_x) W(s_y) W(s_z),                 s in cell units,
K(R) = (1-β²) R / D^{3/2},  D = |R|²(1-β²)+(R·β)²,  B_n = β × E_n.
```

The TSC support is split at `-1.5,-0.5,0.5,1.5`.  When `s* = R0/cell` lies
inside the box, each panel is split at `s*` and mapped with
`s = c + (b-c) t**p`, `p = 5`.  The Jacobian `(b-c) p t^{p-1}` cancels the
`1/r²` kernel singularity; no epsilon is added anywhere.  The remaining
integrand is finite at the formerly singular corner and partition refinement
shows an observed order close to `p`.

## Files

* `indfield.py` — kernel, quadrature, point integral, tile assembly.
* `cases.py` — the four comparison cases.
* `cli.py` — writes a reference field dump (`reference[n,1536]`, `s0[n,2]`).
* `selfcheck.py` — Coulomb/convolution, `B=0`, translation, length scaling.
* `convergence.py` — measured order and partition refinement.
* `compare_repo.py` — compares with the in-repo reference through the frozen
  `field_gate.py` metric (imported, not reimplemented).

## Usage

```bash
python cli.py --output reference.npz --order 8 --refine-near 2
python selfcheck.py --output selfcheck.json
python convergence.py --output convergence.json
python compare_repo.py --output-dir report/
```

`refine_near` controls the transformed partition for outputs inside the support
box.  `refine_near=2` is the default; `refine_near=3` is slow but reaches the
`1e-8` level on the probe points.

## Limitations

* The full-tile comparison uses the in-repo reference's fixed-rule mode
  (`adaptive=False`) because its default adaptive tile path takes minutes per
  tile.  Point-level checks use the default adaptive path.
* This is an FP64 reference.  It does not replace an FP128/mpmath certificate
  for every point; `selfcheck.py` cross-checks selected points with SciPy
  Gauss-Kronrod cubature.
* CTest registration is intentionally not added; the main build must stay
  untouched.  Suggested invocation is a Python CTest fixture.

## Measured results (T13 run)

Convergence (`convergence.py`, transform power 5):

| probe | partition-refinement observed order | order-refinement observed order |
|---|---|---|
| stationary, inside support | 5.09, 5.05 | 4.6, 5.0, 5.9, 8.1 |
| stationary, support edge | 9.7, 10.4 | 10.4, 8.5, 8.8, 10.7 |
| gamma=10, oblique, aspect 0.7/1.3 | 2.8, 4.0 | non-monotone (-1.2, 5.8, 5.5, 1.8) |

The stationary and edge probes reach the `1e-8` level against SciPy cubature
(`selfcheck.py`: `1.6e-8` inside, `6.8e-9` outside).  The oblique gamma=10 /
anisotropic probe converges more slowly: its partition-refinement error is still
`2.1e-4` between refine 2 and refine 3, so its demonstrated floor is about
`2e-4`, not `1e-6`.  This nonconvergence is reported, not hidden.

Self-checks: stationary `B` is bit-for-bit zero; translation invariance
`8.5e-16` (E) / `7.7e-16` (B); uniform scaling by 3.7 gives `8.3e-15` (E) /
`4.7e-15` (B) after the `1/s**2` factor; all gamma 1..100 and aspect 0.1..10
sweep points stay finite.

### Comparison with the in-repo reference

`compare_repo.py` uses the in-repo fixed-rule tile mode (`adaptive=False,
order=12`) because its adaptive tile path takes about 20 s per near point
(>30 min per tile).  That fixed mode is not error-controlled inside the TSC
support, so the full-tile relative L2 values are dominated by the in-repo near
band:

| case | E relative L2 | cB relative L2 | max abs diff |
|---|---|---|---|
| stationary_center | 3.7e-4 | 0.0 (both exactly zero) | 1.0e-3 |
| gamma10_oblique | 3.9e-2 | 2.6e-2 | 6.3e-1 |
| aspect10 | 1.4e-1 | 0.0 (stationary) | 6.1e-1 |
| source_at_output | 3.3e-3 | 3.3e-3 | 3.2e-2 |

`point_compare.py` checks the in-repo *adaptive* mode at selected points:

| case | component | this reference | repo adaptive | rel diff | repo converged |
|---|---|---|---|---|---|
| stationary_center | Ex | -3.125894e-2 | -3.125894e-2 | 1.5e-14 | yes |
| gamma10_oblique | Ey | 5.283238e-1 | 5.283263e-1 | 1.6e-6 | no |
| aspect10 | Ey | -2.984011e-2 | -2.984705e-2 | 6.7e-6 | no |
| source_at_output (Ex) | Ex | 1.9e-15 | not run (no return) | - | - |

At the gamma=10 point SciPy agrees with this reference to `1.6e-6`.  At the
aspect=10 point, a second component (Ez at `(4, 5, 0.2)`) shows the in-repo
adaptive result is off by about `1.1e-2` relative (`0.127863` vs `0.129277`,
SciPy and this reference agree to `2e-8`, repo reports `converged=False`).
The in-repo reference therefore does not meet the `1e-6` reference target for
`aspect=10` near the support; this is the main finding of the independent check.
