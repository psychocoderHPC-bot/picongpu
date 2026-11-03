# mesh-conv: standalone far-field mesh convolution (C++/CUDA)

Self-contained implementation of the free-space mesh field used for the far-field
part of the T151 hybrid. It does not touch PIConGPU.

## FFT backend

The delivered implementation uses **alpakaVendor**
(https://github.com/psychocoderHPC/alpakaVendor), pinned by PIConGPU
(`include/picongpu/CMakeLists.txt`, `FetchContent`) to
`af24cc7b2bd125056e9889e270d3af7bdf972c6f`. The program calls only the portable
`alpaka::fft::onHost` API:

    auto plan = alpaka::fft::onHost::PlanBuilder<
                    alpaka::fft::Complex_t<T_Real>, alpaka::Vec<uint32_t, 3>>{extents}
                    .c2c().inPlace().build(device);
    alpaka::fft::onHost::executeForward(queue, plan, in, out);
    alpaka::fft::onHost::executeBackward(queue, plan, in, out);

The vendor maps that plan to the platform FFT: cuFFT on an Nvidia GPU, hipFFT
on AMD, oneMKL DFT on SYCL and FFTW on the CPU. The source contains no direct
vendor FFT call. Both `float` and `double` are supported
(`alpaka::fft::RealScalar`), selected by `--fft float|double`.

This replaces the earlier prototype that called the cuFFT API directly.
Both implementations give the same validated numbers and comparable timings
(see Validation and Performance). The alpakaVendor version is the one that
ships because the project requires the vendor FFT for backend portability.

## Portability

The FFT comes from alpakaVendor; on CUDA that resolves to cuFFT, on HIP to
hipFFT, on SYCL to oneMKL and on the CPU to FFTW. The code calls only the
vendor API (`PlanBuilder`, `executeForward`, `executeBackward`), never a
platform FFT function directly. The CUDA build links `libcufft` because
alpakaVendor's CUDA backend maps to cuFFT internally.

## Formulation

Reference (validated `indfield.py`):

    E_n(x) = sum_p w_p * integral B2^3(s) K_point(x - x_p - s*cells) ds^3

Mesh (this program):

    rho[i] = sum_p w_p * B2^3(i*h - p_cell)                     (TSC deposit)
    F_c[j] = h^3 * sum_i rho[i] * K_c(((j-i)*h + o_c)*cells)   (free-space conv)

`F_c[j]` is the field component at the staggered Yee position
`(j*h + o_c)*cells`.  The `h^3` factor is part of the convolution, not only the
FFT normalisation: the TSC deposit sums to the particle charge, so the Riemann
sum needs the cell volume `h^3` even when the kernel is sampled in cell units.  The point kernel is the same algebraic form as
`indfield.point_field`:

    D = |R|^2/(1+u^2) + (R.u)^2/(1+u^2)^2
    E = (1/(1+u^2)) R / D^1.5,  B = beta x E,  beta = u/sqrt(1+u^2)

The convolution is a circular FFT convolution zero-padded to `M = pad*N`. With
`pad = 2` the linear convolution is exact for source/target separations `< N`.
`pad = 1` is the deliberate no-padding control.

## Build (rosi_v100, CUDA 12.6)

    ./run_all.sh

The script clones alpakaVendor at the PIConGPU pin, builds `meshconv.cu` with
`nvcc` and links `libcufft` from the toolkit. The alpaka version is
the one pinned by alpakaVendor (`extern/alpaka3`,
`7b92361468d3280c260200198081ae7e98f0bc56`).

## Modes

    CUDA_VISIBLE_DEVICES=3 ./meshconv --mode info
    CUDA_VISIBLE_DEVICES=3 ./meshconv --mode validate [--fft double|float] [--h H]
    CUDA_VISIBLE_DEVICES=3 ./meshconv --mode refine
    CUDA_VISIBLE_DEVICES=3 ./meshconv --mode nopad
    CUDA_VISIBLE_DEVICES=3 ./meshconv --mode bench  [--fft double|float]

* `info`: prints the FFT backend and the alpakaVendor pin.
* `validate`: grid 48^3, pad 4, one unit source on the centre node, gamma in
  {1,10,100}, aspect y/x in {0.25, 4, 1}, distances d = 2..16 nodes along +x and
  +y.  Prints `gamma,aspect,axis,d,comp,mesh`.  `--h` sets the dimensionless node
  spacing in cell units (default 1); `compare_ref.py --h H` evaluates the
  reference at the matching positions.
* `refine`: self-contained regression that compares the field at the same
  physical source/target separation for h=1 and h=1/2 (gamma 1 and 10).  It
  fails if the `h^3` quadrature measure is missing from the convolution scale.
* `compare_ref.py`: evaluates the `indfield.py` reference at the staggered
  positions and reports worst relative error per component with
  `|ref-mesh|/(1+max(|ref|,|mesh|))` (the T139/T142/T149 convention).
* `nopad`: same source, N=32, compares pad=1 (periodic) against pad=2
  (free space) over the whole grid; reports the wraparound artefact.
* `bench`: particle counts 1e3/1e4/1e6/2e6 in a uniform cube filling a 20-cell
  (N=64, M=128) and a 50-cell (N=128, M=256) box, h=1.  Times deposit, kernel
  build, FFT (1 rho + 6 kernel forwards, 6 products, 6 inverses), and read-out
  (compact + device-to-host) separately with CUDA events on the alpaka queue's
  stream.

Precision: deposit, kernel build and rho accumulation are double.  `--fft float`
casts only the FFT inputs and runs the transform in single precision.

## Validation

Reference: `indfield.py`, sha256
`a135598041a3dbe134667585631ea53fc6bdb7cf7a84419b5a8d762202599519`,
evaluated serially by `compare_ref.py`. The alpakaVendor build reproduces the
earlier direct-FFT prototype **to every printed digit**.

Worst per-component error over the far band d=12..16 (double FFT): 1.4e-4 to
2.9e-4.  Grid-refinement regression (`--mode refine`, h=1 vs h=1/2 at fixed
physical separation): worst 9.1e-4, gate 2e-3.  With the pre-fix `scale=1/M^3`
it is O(1) (factor h^-3 = 8). Over the near-dominated d=2..16 band: 3.0e-2 to 7.2e-1, so the mesh
alone is not a near-field method and needs the table inside the shell.

No-padding control, N=32, pad=1 vs pad=2: the wraparound artefact is
8.947e-3 at `(i,j,k) = (31,13,18)`, Ex. This is the forbidden periodic image
sum; `pad = 2` is the default in both bench and validation.

`--fft float` (same double deposit and kernel): peak-normalised roundoff is at
most 3.968e-6, which passes the 3e-4 gate; as a fraction of the local double
field it reaches 3.26, so float is not usable for the physics.

## Performance

V100-SXM2-32GB, N=2e6, double FFT, same GPU session, h=1, aspect 1:

| implementation | 20-cell (N=64, M=128) | 50-cell (N=128, M=256) |
|---|---|---|
| alpakaVendor (delivered) | 10.0 ms | 105.5 ms |
| direct-FFT prototype | 10.0 ms | 82.8 ms |

The 50-cell total includes a 100.7 MB device-to-host copy; keeping the field on
device gives about 85 ms for the alpakaVendor build. An earlier T151 run on an
idle GPU measured 8.6 ms and 60.5 ms for the direct-FFT prototype; the
same-session pair above is the apples-to-apples comparison. Extrapolating the
measured numbers as M^3 log M to a 256^3 box projects about 0.9 s on device,
2.3 s with the full host copy. That projection is not measured.

## Known gaps

* **Single shared velocity.** All particles use one velocity `u`. Per-particle
  velocity is not implemented; a bunch with velocity spread needs one
  convolution per velocity class or a different formulation.
* **Double precision is required.** Deposit, kernel build and FFT stay in
  double by default. A float32 FFT passes only the peak-normalised 3e-4 gate;
  as a fraction of the local field it fails, with roundoff 1.7e-4 at d=8 and up
  to 2.4 at d=16 on the beamed gamma=10/100 cases.
* **Zero-padding to at least 2x per axis is mandatory.** With `pad = 1` the
  periodic wrap contaminates the field (8.947e-3 artefact above). `pad = 2`
  makes the linear convolution exact for separations below N.

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
