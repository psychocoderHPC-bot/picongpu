# mesh-conv: standalone far-field mesh convolution (C++/CUDA)

Self-contained implementation of the free-space mesh field used for the far-field
part of the T151 hybrid. It does not touch PIConGPU.

## Formulation

Reference (validated `indfield.py`):

    E_n(x) = sum_p w_p * integral B2^3(s) K_point(x - x_p - s*cells) ds^3

Mesh (this program):

    rho[i] = sum_p w_p * B2^3(i*h - p_cell)                     (TSC deposit)
    F_c[j] = h^3 * sum_i rho[i] * K_c(((j-i)*h + o_c)*cells)   (free-space conv)

`F_c[j]` is the field component at the staggered Yee position
`(j*h + o_c)*cells`.  The point kernel is the same algebraic form as
`indfield.point_field`:

    D = |R|^2/(1+u^2) + (R.u)^2/(1+u^2)^2
    E = (1/(1+u^2)) R / D^1.5,  B = beta x E,  beta = u/sqrt(1+u^2)

The convolution is a circular FFT convolution zero-padded to `M = pad*N`.  With
`pad = 2` the linear convolution is exact for source/target separations `< N`.
`pad = 1` is the deliberate no-padding control.

## Build (rosi_v100, CUDA 12.6)

    PATH=/usr/local/cuda/bin:$PATH nvcc -O3 -arch=sm_70 -std=c++17 \
         meshconv.cu -lcufft -o meshconv

## Modes

    CUDA_VISIBLE_DEVICES=2 ./meshconv --mode validate [--fft double|float]
    CUDA_VISIBLE_DEVICES=2 ./meshconv --mode nopad
    CUDA_VISIBLE_DEVICES=2 ./meshconv --mode bench  [--fft double|float]

* `validate`: grid 48^3, pad 4, one unit source on the centre node, gamma in
  {1,10,100}, aspect y/x in {0.25, 4, 1}, distances d = 2..16 nodes along +x and
  +y.  Prints `gamma,aspect,axis,d,comp,mesh`.
* `compare_ref.py`: evaluates the `indfield.py` reference at the staggered
  positions and reports worst relative error per component with
  `|ref-mesh|/(1+max(|ref|,|mesh|))` (the T139/T142/T149 convention).
* `nopad`: same source, N=32, compares pad=1 (periodic) against pad=2
  (free space) over the whole grid; reports the wraparound artefact.
* `bench`: particle counts 1e3/1e4/1e6/2e6 in a uniform cube filling a 20-cell
  (N=64, M=128) and a 50-cell (N=128, M=256) box, h=1.  Times deposit, kernel
  build, FFT (1 rho + 6 kernel forwards, 6 products, 6 inverses), and read-out
  (compact + device-to-host) separately with CUDA events.

Precision: deposit, kernel build and rho accumulation are double.  `--fft float`
casts only the two FFT inputs and runs cuFFT in single precision.

## Known gaps

* **Single shared velocity.** All particles use one velocity `u`. Per-particle
  velocity is not implemented; a bunch with velocity spread needs one
  convolution per velocity class or a different formulation.
* **Double precision is required.** Deposit, kernel build and FFT stay in
  double by default. A float32 FFT passes only the peak-normalised 3e-4 gate;
  as a fraction of the local field it fails, with roundoff 1.7e-4 at d=8 and up
  to 2.4 at d=16 on the beamed gamma=10/100 cases.
* **Zero-padding to at least 2x per axis is mandatory.** With `pad = 1` the
  periodic wrap contaminates the field. The no-pad control reproduces the
  artefact at 8.947e-3 (`(i,j,k) = (31,13,18)`, Ex). `pad = 2` makes the linear
  convolution exact for separations below N.
* **Timings** (V100-SXM2-32GB, N=2e6, double FFT): 20-cell bunch 8.6 ms,
  50-cell bunch 60.5 ms. The 50-cell figure includes a 100.7 MB host copy;
  keeping the field on device gives about 38 ms. Scaling as M^3 log M to a
  256^3 box projects about 0.32 s. The 256^3 number is extrapolated, not
  measured.
* **Accuracy splits into two bands.** Far field d=12..16: worst per-component
  error 1.4e-4 to 2.9e-4. Near-dominated d=2..16: 3.0e-2 to 7.2e-1. The mesh
  alone is not a near-field method. Inside the shell the near-field table is
  required.

## PIConGPU integration plan

Not implemented here. This section records the placement contract for the
integration task.

* **Field layout.** Write the six components into PIConGPU's Yee field storage
  at the offsets `E_x (0.5,0,0)`, `E_y (0,0.5,0)`, `E_z (0,0,0.5)`,
  `cB_x (0,0.5,0.5)`, `cB_y (0.5,0,0.5)`, `cB_z (0.5,0.5,0)` in cell units,
  from the `traits::FieldPosition<fields::YeeCell, FieldE/FieldB>`
  specialisations (2D uses the corresponding subset). The mesh read-out already
  uses these offsets, so no interpolation is needed at write time.
* **Coverage split.** The mesh handles source-target pairs beyond the shell
  radius. The near-field table handles pairs inside the shell. Sub-cell
  separations use the analytic box field. The paths must partition pairs
  without overlap.
* **Reference path.** Keep the existing exact quadrature path selectable as the
  reference; the mesh path is an alternative, not a replacement.
* **Determinism.** Sum the deposit with a fixed reduction order so that a change
  of MPI layout changes the field only at rounding level. Record the reduction
  order with the implementation.
* **Open question: multi-rank bunches.** When the bunch spans several ranks,
  each rank deposits its local particles and the deposit is all-reduced. Every
  rank then runs the same local FFT on the reduced grid; no global FFT is
  planned. The cost of the full padded-grid all-reduce per step remains to be
  measured.

## Run everything

    ./run_all.sh
