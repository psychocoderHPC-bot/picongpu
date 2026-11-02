# Reference conventions

This directory contains an independent FP64 reference for the neural bunch
field initializer.  It implements the present-time field of uniformly moving
point charges, integrated over the PIConGPU TSC shape in the lab frame, sampled
on a Yee tile.

Sources: `pico-pr17` refers to a PIConGPU checkout (the feature branch base);
`tools/neural-bunch/model-format.md` is the interface contract.  Paths below
are relative to `tools/neural-bunch/`.

## Quantity / convention table

| Quantity | Convention | Source |
|---|---|---|
| Length unit in the reference API | physical (SI) for `x`, `dx`, `dy`, `dz`; cell units for the shape variable `s` | this file |
| Momentum | `u = p / (m_e c)`, dimensionless, `gamma = sqrt(1 + |u|^2)` | MODEL-FORMAT.md:107-108 |
| Per-unit-charge field | `E_n = 4*pi*eps0*E/q`, `B_n = 4*pi*eps0*c*B/q`; `B_n = beta x E_n` | this file |
| Normalized tile E | `E_tilde = E * 4*pi*eps0*L^2/e`, `L = dx` | MODEL-FORMAT.md:57-59 |
| Normalized tile B | `B_tilde = B * 4*pi*eps0*c*L^2/e` | MODEL-FORMAT.md:60 |
| Charge factor | `q_i/e = -w_i`, applied outside the network | MODEL-FORMAT.md:58 |
| Tile shape | `nx=8`, `ny=8`, `nz=4`, 6 components | MODEL-FORMAT.md:26-28 |
| Component order | `Ex, Ey, Ez, Bx, By, Bz` | MODEL-FORMAT.md:28 |
| Serialized flatten order | `c + 6*(i + 8*(j + 8*k))` | MODEL-FORMAT.md:29 |
| Tile origin | `x_tile` is the physical corner of cell `(0,0,0)`; output `x = x_tile + ((i,j,k) + delta) * (dx,dy,dz)` | this file |
| Shape | TSC, support `[-1.5, 1.5]` cells per axis | TSC.hpp:45 |
| Shape normalization | `integral W(s) ds = 1` per axis; `integral S ds^3 = 1` | TSC.hpp:108-131 |
| Shape argument | `s_axis` is in cell units; physical offset is `(sx*dx, sy*dy, sz*dz)` | this file |
| `a_y`, `a_z` | `a_y = dy/dx`, `a_z = dz/dx` (cell aspect ratios) | MODEL-FORMAT.md:107-108 |
| Reference domain | `gamma in [1,100]`, `a_y,a_z in [0.1,10]` | MODEL-FORMAT.md:64-66 |

## Electric and magnetic offsets in the Yee cell

The 3-D `FieldPosition` specializations give, in cell units:

| Component | Offset `delta` | Source |
|---|---|---|
| `Ex` | `(0.5, 0.0, 0.0)` | YeeCell.hpp:80 |
| `Ey` | `(0.0, 0.5, 0.0)` | YeeCell.hpp:81 |
| `Ez` | `(0.0, 0.0, 0.5)` | YeeCell.hpp:82 |
| `Bx` | `(0.0, 0.5, 0.5)` | YeeCell.hpp:124 |
| `By` | `(0.5, 0.0, 0.5)` | YeeCell.hpp:125 |
| `Bz` | `(0.5, 0.5, 0.0)` | YeeCell.hpp:126 |

These values are also encoded in `convkernel.OFFSETS` and `geometry.json`.
Note that `B` and `E` are shifted by half a cell from each other, as described
at YeeCell.hpp:32-35.

## TSC assignment function

PIConGPU uses (`x` in cell units):

```
W(x) = 3/4 - x^2                     for |x| < 1/2
W(x) = 1/2 * (3/2 - |x|)^2           for 1/2 <= |x| < 3/2
W(x) = 0                             otherwise
```

This is stated in the comment at TSC.hpp:110-114 and implemented through
`ff_1st_radius` (TSC.hpp:47-54), `ff_2nd_radius` (TSC.hpp:56-64), and the
branching in `ChargeAssignment::operator()` (TSC.hpp:108-131).  The support is
three cells per axis (`support = 3`, TSC.hpp:45).  `convkernel.tsc_shape`
implements exactly this piecewise polynomial, and `test_reference.py` pins it
plus `integral W = 1`.

The shape is applied in the lab frame without Lorentz contraction:
`S(s) = W(sx) W(sy) W(sz)`, physical offset `(sx*dx, sy*dy, sz*dz)`.

## Point kernel and numerical stability

For `R = x_obs - x_source(t)`, `beta = u/gamma`, the per-unit-charge field is

```
E_n = (1 - beta^2) R / (R^2 - |beta x R|^2)^(3/2)
B_n = beta x E_n
```

`convkernel.point_field` evaluates the algebraically equivalent stable form

```
invGamma2 = 1 / (1 + |u|^2)
R_par     = R . bhat,   bhat = u/|u|
R_perp    = R - R_par*bhat
D         = R_par^2 + R_perp^2 * invGamma2
E_n       = invGamma2 * R / D^(3/2)
```

The `u == 0` case takes an explicit branch, so `B_n == 0` bit-for-bit.  Tests
compare `gamma = 1, 2, 10, 100` against an 80-digit mpmath evaluation of the
direct formula.  No epsilon is added to `D` or to any denominator; there is no
silent clamping.

## Quadrature (error-controlled for every point)

There is no unconditional "far" rule.  Every observation point is checked:

1. Start on the three base panels `[-1.5,-0.5]`, `[-0.5,0.5]`, `[0.5,1.5]` per
   axis with orders `order` and `order+2` (default 4 and 6).
2. The Richardson-style estimate is `max_abs(I_hi - I_lo) / (1 + max_abs)`
   over all six field components.  If it is `<= tol` (default `1e-7`), the
   high-order result is accepted.
3. Otherwise per-axis panels are geometrically graded around the projected
   target and the order pair is raised by two, up to `max_level` (default 5).
4. At `max_level` the order keeps increasing on the same panels, reusing the
   previous high-order result, up to `max_order` (default 16).

The diagnostic near/far flag uses the **physical** distance to the TSC support
box: `excess_i = max(|R_i| - 1.5*cell_i, 0)` and `|excess| < 2*max(dx,dy,dz)`.
This flag never bypasses the error estimate; it is reported only.

The integrand is evaluated at the actual Gauss nodes, including the
near-singular region; it is never regularized.  Memory is bounded by chunking
the tensor product over the x nodes.

## Tile flattening

`tile_field` returns an array indexed `[i, j, k, c]` with shape `(8,8,4,6)`.
For serialization use:

```python
flat = convkernel.flatten_tile(tile)      # length 1536
tile = convkernel.unflatten_tile(flat)
```

`flatten_tile` transposes to `(k,j,i,c)` before `ravel`, which produces exactly
`index = c + 6*(i + 8*(j + 8*k))`.  `tile_field_flat(...)` is a convenience
wrapper.  The array itself cannot have both `[i,j,k,c]` indexing and i-fastest
NumPy ravel order, so the explicit helpers are the serialization contract.
