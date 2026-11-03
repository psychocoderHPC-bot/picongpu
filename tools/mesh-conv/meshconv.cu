// meshconv.cu -- standalone far-field mesh convolution for the TSC-convolved
// uniformly-moving point-charge field (free-space, zero padded FFT).
//
// Reference formulation (validated indfield.py):
//   E_n(x) = sum_p w_p int B2^3(s) K_point(x - x_p - s*cells) ds^3
// Mesh formulation:
//   rho[i] = sum_p w_p B2^3((i*h - p_cell))          (TSC deposit)
//   F_c[j] = h^3 sum_i rho[i] K_c(((j-i)*h + o_c)*cells)   (free-space conv)
// so F_c[j] is the component at the STAGGERED position (j*h + o_c)*cells.
//
// All precision-critical work (deposit, kernel build, accumulation of rho) is
// double.  `--fft float` only casts the two FFT inputs and runs the transform in
// single precision, so the float-FFT experiment is isolated from the kernel
// accuracy.
//
// FFT backend: alpakaVendor (https://github.com/psychocoderHPC/alpakaVendor),
// pinned by PIConGPU to af24cc7b2bd125056e9889e270d3af7bdf972c6f.  The program
// only calls the portable alpaka::fft API and never calls a vendor FFT API
// directly.  alpakaVendor maps the plan to the platform FFT: cuFFT on an Nvidia
// GPU, hipFFT on AMD, oneMKL DFT on SYCL and FFTW on the CPU.  The build script
// links libcufft; see run_all.sh.

#include <alpaka/alpaka.hpp>
#include <alpaka/fft.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <type_traits>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
          cudaGetErrorString(e_)); exit(1); } } while (0)

//: alpakaVendor commit pinned by PIConGPU (include/picongpu/CMakeLists.txt).
static constexpr char const* kAlpakaVendorPin =
    "af24cc7b2bd125056e9889e270d3af7bdf972c6f";

// Device + queue used by every FFT.  The queue owns the CUDA stream the raw
// kernels and the alpakaVendor plans are ordered on.
struct Env
{
  using Device = decltype(alpaka::onHost::makeDeviceSelector(
      alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu).makeDevice(0));
  using Queue = decltype(std::declval<Device&>().makeQueue());

  Device device;
  Queue queue;

  explicit Env(Device d) : device{std::move(d)}, queue{device.makeQueue()}
  {
  }
};

static Env makeEnv()
{
  auto selector = alpaka::onHost::makeDeviceSelector(
      alpaka::api::cuda, alpaka::deviceKind::nvidiaGpu);
  if(!selector.isAvailable())
  {
    fprintf(stderr, "no CUDA/NVIDIA device available to alpaka\n");
    exit(1);
  }
  Env env{selector.makeDevice(0)};
  fprintf(stderr, "[env] device=\"%s\" fft=alpakaVendor pin=%s\n",
          env.device.getName().c_str(), kAlpakaVendorPin);
  return env;
}

// ---------------------------------------------------------------------------
// physical constants / conventions
// ---------------------------------------------------------------------------
__constant__ double c_off[6][3];

static const double kOffsets[6][3] = {
    {0.5, 0.0, 0.0},   // Ex
    {0.0, 0.5, 0.0},   // Ey
    {0.0, 0.0, 0.5},   // Ez
    {0.0, 0.5, 0.5},   // Bx
    {0.5, 0.0, 0.5},   // By
    {0.5, 0.5, 0.0}};  // Bz
static const char* kComp[6] = {"Ex", "Ey", "Ez", "Bx", "By", "Bz"};
static double g_h = 1.0;  // grid spacing in cell units (refinement study)

struct Sim {
  int N = 48;          // number of nodes per axis
  int pad = 4;         // FFT padding factor; M = pad*N
  double h = 1.0;      // grid spacing in cell units (node i -> i*h*cells)
  double cell[3] = {1.0, 1.0, 1.0};
  double u[3] = {0.0, 0.0, 0.0};
  int M() const { return pad * N; }
};

struct Phase {
  double dep = 0, kern = 0, fft = 0, read = 0;
};

// ---------------------------------------------------------------------------
// device kernels
// ---------------------------------------------------------------------------
__device__ __forceinline__ void pointField6(double Rx, double Ry, double Rz,
                                            double ux, double uy, double uz,
                                            double* K) {
  // Independent algebraic form, identical to indfield.point_field:
  //   D = |R|^2 (1-beta^2) + (R.beta)^2 = |R|^2/(1+u^2) + (R.u)^2/(1+u^2)^2
  const double uu2 = ux * ux + uy * uy + uz * uz;
  const double invg2 = 1.0 / (1.0 + uu2);
  const double gam = sqrt(1.0 + uu2), ib = 1.0 / gam;
  const double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
  const double ru = Rx * ux + Ry * uy + Rz * uz;
  const double rub = ru * ib;                 // R.beta, beta = u/gamma
  double D = r2 * invg2 + rub * rub;
  double invD15 = 0.0;
  if (D > 0.0) invD15 = 1.0 / (D * sqrt(D));
  const double f = invg2 * invD15;
  const double Ex = f * Rx, Ey = f * Ry, Ez = f * Rz;
  const double bx = ux * ib, by = uy * ib, bz = uz * ib;
  K[0] = Ex; K[1] = Ey; K[2] = Ez;
  K[3] = by * Ez - bz * Ey;
  K[4] = bz * Ex - bx * Ez;
  K[5] = bx * Ey - by * Ex;
  if (uu2 == 0.0) { K[3] = K[4] = K[5] = 0.0; }
}

// Build the six sampled kernels K_c(d) = K_comp[c](((d+o_c)*h)*cells).
// One thread per FFT-grid point; six staggered offsets => six field evaluations.
__global__ void buildKernelK(double* __restrict__ K, int M, double h,
                             double cx, double cy, double cz,
                             double ux, double uy, double uz) {
  const int64_t M3 = (int64_t)M * M * M;
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M3) return;
  const int i = (int)(idx % M);
  const int j = (int)((idx / M) % M);
  const int k = (int)(idx / ((int64_t)M * M));
  const int di = (i < M / 2) ? i : i - M;
  const int dj = (j < M / 2) ? j : j - M;
  const int dk = (k < M / 2) ? k : k - M;
  for (int c = 0; c < 6; ++c) {
    const double Rx = ((double)di * h + c_off[c][0]) * cx;
    const double Ry = ((double)dj * h + c_off[c][1]) * cy;
    const double Rz = ((double)dk * h + c_off[c][2]) * cz;
    double K6[6];
    pointField6(Rx, Ry, Rz, ux, uy, uz, K6);
    double v = K6[c];
    if (!isfinite(v)) v = 0.0;  // exact for far reads; avoids 0*inf in the FFT
    K[(int64_t)c * M3 + idx] = v;
  }
}

__device__ __forceinline__ double tsc(double x) {
  const double ax = fabs(x);
  if (ax < 0.5) return 0.75 - ax * ax;
  if (ax < 1.5) { const double t = 1.5 - ax; return 0.5 * t * t; }
  return 0.0;
}

// TSC deposit into the M^3 padded grid (rho stored with M stride, x fastest).
__global__ void depositK(const double* __restrict__ pos,
                         const double* __restrict__ w, int P,
                         double* __restrict__ rho, int N, int M, double h,
                         double cx, double cy, double cz) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  const double pc[3] = {pos[3 * p] / cx, pos[3 * p + 1] / cy,
                        pos[3 * p + 2] / cz};  // source in cell units
  const double q = w[p];
  int base[3];
  double frac[3];
  int lo[3], hi[3];
  for (int a = 0; a < 3; ++a) {
    const double t = pc[a] / h;
    base[a] = (int)floor(t);
    frac[a] = pc[a] - base[a] * h;               // in [0,h)
    lo[a] = (int)floor((frac[a] - 1.5) / h) + 1;
    hi[a] = (int)ceil((frac[a] + 1.5) / h) - 1;
  }
  for (int da = lo[0]; da <= hi[0]; ++da) {
    const int ia = base[0] + da;
    if (ia < 0 || ia >= N) continue;
    const double wa = tsc(da * h - frac[0]);
    if (wa == 0.0) continue;
    for (int db = lo[1]; db <= hi[1]; ++db) {
      const int ib = base[1] + db;
      if (ib < 0 || ib >= N) continue;
      const double wb = wa * tsc(db * h - frac[1]);
      if (wb == 0.0) continue;
      for (int dc = lo[2]; dc <= hi[2]; ++dc) {
        const int ic = base[2] + dc;
        if (ic < 0 || ic >= N) continue;
        const double wc = wb * tsc(dc * h - frac[2]);
        if (wc == 0.0) continue;
        atomicAdd(&rho[(int64_t)ia + (int64_t)M * (ib + (int64_t)M * ic)],
                  wc * q);
      }
    }
  }
}

// The alpakaVendor Complex<T> is layout-compatible with the backend complex
// type (the vendor reinterprets the pointer internally).  The kernels therefore
// write/read the real and imaginary scalars directly and avoid any device-side
// construction of the host-oriented Complex type.
template<typename T_Complex>
__global__ void realToComplex(const double* __restrict__ r,
                              T_Complex* __restrict__ c, int64_t n) {
  using R = typename T_Complex::value_type;
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  R* p = reinterpret_cast<R*>(c) + 2 * i;
  p[0] = static_cast<R>(r[i]);
  p[1] = R(0);
}

template<typename T_Complex>
__global__ void mulComplex(T_Complex* __restrict__ a,
                           const T_Complex* __restrict__ b, int64_t n) {
  using R = typename T_Complex::value_type;
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  R* pa = reinterpret_cast<R*>(a) + 2 * i;
  const R* pb = reinterpret_cast<const R*>(b) + 2 * i;
  const R ar = pa[0], ai = pa[1], br = pb[0], bi = pb[1];
  pa[0] = ar * br - ai * bi;
  pa[1] = ar * bi + ai * br;
}

template<typename T_Complex>
__global__ void extract(const T_Complex* __restrict__ src, int M,
                        double* __restrict__ out, int N, int64_t N3,
                        double scale) {
  using R = typename T_Complex::value_type;
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N3) return;
  const int i = (int)(idx % N);
  const int j = (int)((idx / N) % N);
  const int k = (int)(idx / ((int64_t)N * N));
  const int64_t src_idx = i + (int64_t)M * (j + (int64_t)M * k);
  out[idx] = static_cast<double>(reinterpret_cast<const R*>(src)[2 * src_idx]) * scale;
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
static double nowEvents(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, a, b));
  return (double)ms;
}

// Runs deposit + kernel build + alpakaVendor FFT + readout.  particles already
// on device.  Returns F[c*N3 + i + N*(j + N*k)] on the host.
//
// The plan is built once per call from pinned alpakaVendor and reused for all 13
// transforms (1 rho + 6 kernels forward, 6 inverse).  All transforms are C2C and
// in-place; every plan input is converted from the double-precision rho/K arrays
// by the cast kernel.  T_Real selects the FFT precision.
template<typename T_Real>
static std::vector<double> runMeshImpl(Env& env,
                                       const Sim& s,
                                       const double* d_pos,
                                       const double* d_w,
                                       int P, Phase* ph, bool verbose)
{
  using Complex = alpaka::fft::Complex_t<T_Real>;
  using Extents = alpaka::Vec<uint32_t, 3u>;

  const int N = s.N, M = s.M();
  const int64_t M3 = (int64_t)M * M * M;
  const int64_t N3 = (int64_t)N * N * N;

  cudaStream_t stream = env.queue.getNativeHandle();
  CK(cudaMemcpyToSymbol(c_off, kOffsets, sizeof(kOffsets)));

  double* d_rho = nullptr;
  double* d_K = nullptr;
  CK(cudaMalloc(&d_rho, sizeof(double) * M3));
  CK(cudaMalloc(&d_K, sizeof(double) * 6 * M3));

  // Complex FFT working buffers.  rho_hat: FFT(rho); kbuf: reused for each
  // component (FFT(K) -> multiply by rhoHat -> inverse -> compact extract).
  // Both transforms are in-place, so only two M^3 complex buffers are needed.
  Complex* d_rho_hat = nullptr;
  Complex* d_kbuf = nullptr;
  CK(cudaMalloc(&d_rho_hat, sizeof(Complex) * M3));
  CK(cudaMalloc(&d_kbuf, sizeof(Complex) * M3));

  // Non-owning alpaka views over the raw contiguous allocations.  Extents are
  // (M, M, M); alpaka index 0 is the fastest axis, matching the kernels and the
  // plan's default contiguous strides.
  auto extents = Extents::fill(static_cast<uint32_t>(M));
  auto rhoHat = alpaka::makeView(env.device, d_rho_hat, extents);
  auto kbuf = alpaka::makeView(env.device, d_kbuf, extents);

  // One portable plan, 3-D C2C, in-place, no normalization (vendor default).
  auto plan = alpaka::fft::onHost::PlanBuilder<Complex, Extents>{extents}
                  .c2c()
                  .inPlace()
                  .build(env.device);
  static_assert(
      std::is_same_v<decltype(alpaka::getApi(env.queue)), alpaka::api::Cuda>,
      "this tool instantiates the alpakaVendor CUDA (cuFFT) FFT backend");

  std::vector<double> hostF((size_t)6 * N3, 0.0);
  double* d_out = nullptr;
  CK(cudaMalloc(&d_out, sizeof(double) * 6 * N3));

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));

  // ---- deposit -----------------------------------------------------------
  CK(cudaEventRecord(e0, stream));
  CK(cudaMemsetAsync(d_rho, 0, sizeof(double) * M3, stream));
  if(P > 0)
  {
    const int thr = 256, blk = (P + thr - 1) / thr;
    depositK<<<blk, thr, 0, stream>>>(d_pos, d_w, P, d_rho, N, M, s.h,
                                      s.cell[0], s.cell[1], s.cell[2]);
    CK(cudaGetLastError());
  }
  CK(cudaEventRecord(e1, stream));
  CK(cudaEventSynchronize(e1));
  ph->dep = nowEvents(e0, e1);

  // ---- kernel build ------------------------------------------------------
  CK(cudaEventRecord(e0, stream));
  {
    const int thr = 256;
    const int64_t blk = (M3 + thr - 1) / thr;
    buildKernelK<<<(int)blk, thr, 0, stream>>>(
        d_K, M, s.h, s.cell[0], s.cell[1], s.cell[2], s.u[0], s.u[1], s.u[2]);
    CK(cudaGetLastError());
  }
  CK(cudaEventRecord(e1, stream));
  CK(cudaEventSynchronize(e1));
  ph->kern = nowEvents(e0, e1);

  // ---- FFT convolution (alpakaVendor) ------------------------------------
  // The 13 transforms run in-place on the alpaka queue's stream.  Each group is
  // bracketed by CUDA events so the pure transform time is reported separately
  // from the compact read-out kernels.
  double fftMs = 0.0;
  double extractMs = 0.0;
  {
    const int thr = 256;
    const int64_t blk = (M3 + thr - 1) / thr;
    const int64_t blkN = (N3 + thr - 1) / thr;
    // F_c[j] = h^3 sum_i rho[i] K_c(...): the TSC deposit has unit
    // partition sum, so the quadrature measure h^3 is required in addition to
    // the inverse FFT normalization 1/M^3 (alpakaVendor has none).  `h` is the
    // dimensionless node spacing in cell units; `cell` is already folded into
    // the sampled kernel, so no cell factor belongs here.
    const double scale = s.h * s.h * s.h / (double)M3;

    CK(cudaEventRecord(e0, stream));
    realToComplex<Complex><<<(int)blk, thr, 0, stream>>>(d_rho, d_rho_hat, M3);
    CK(cudaGetLastError());
    alpaka::fft::onHost::executeForward(env.queue, plan, rhoHat, rhoHat);
    CK(cudaEventRecord(e1, stream));
    CK(cudaEventSynchronize(e1));
    fftMs += nowEvents(e0, e1);

    for(int c = 0; c < 6; ++c)
    {
      CK(cudaEventRecord(e0, stream));
      realToComplex<Complex><<<(int)blk, thr, 0, stream>>>(
          d_K + (int64_t)c * M3, d_kbuf, M3);
      CK(cudaGetLastError());
      alpaka::fft::onHost::executeForward(env.queue, plan, kbuf, kbuf);
      mulComplex<Complex><<<(int)blk, thr, 0, stream>>>(d_kbuf, d_rho_hat, M3);
      CK(cudaGetLastError());
      alpaka::fft::onHost::executeBackward(env.queue, plan, kbuf, kbuf);
      CK(cudaEventRecord(e1, stream));
      CK(cudaEventSynchronize(e1));
      fftMs += nowEvents(e0, e1);

      CK(cudaEventRecord(e0, stream));
      extract<Complex><<<(int)blkN, thr, 0, stream>>>(
          d_kbuf, M, d_out + (int64_t)c * N3, N, N3, scale);
      CK(cudaGetLastError());
      CK(cudaEventRecord(e1, stream));
      CK(cudaEventSynchronize(e1));
      extractMs += nowEvents(e0, e1);
    }
  }
  ph->fft = fftMs;

  // ---- readout (device-to-host copy of the compact field) ----------------
  CK(cudaEventRecord(e0, stream));
  CK(cudaMemcpyAsync(hostF.data(), d_out, sizeof(double) * 6 * N3,
                     cudaMemcpyDeviceToHost, stream));
  CK(cudaEventRecord(e1, stream));
  CK(cudaEventSynchronize(e1));
  ph->read = extractMs + nowEvents(e0, e1);

  CK(cudaFree(d_rho_hat));
  CK(cudaFree(d_kbuf));
  CK(cudaFree(d_out));
  CK(cudaFree(d_rho));
  CK(cudaFree(d_K));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  if(verbose)
    fprintf(stderr, "[run] N=%d M=%d P=%d dep=%.2fms kern=%.2fms "
                    "fft=%.2fms read=%.2fms\n",
            N, M, P, ph->dep, ph->kern, ph->fft, ph->read);
  return hostF;
}

// Runtime precision dispatch (deposit/kernel/readout kernels shared).
static std::vector<double> runMesh(Env& env,
                                   const Sim& s,
                                   const double* d_pos,
                                   const double* d_w,
                                   int P, bool useFloat, Phase* ph,
                                   bool verbose)
{
  if(useFloat)
    return runMeshImpl<float>(env, s, d_pos, d_w, P, ph, verbose);
  return runMeshImpl<double>(env, s, d_pos, d_w, P, ph, verbose);
}

static void vel(int gamma, double u[3]) {
  if (gamma <= 1) { u[0] = u[1] = u[2] = 0.0; return; }
  const double g = (double)gamma;
  const double beta = sqrt(1.0 - 1.0 / (g * g));
  const double d[3] = {1.0, 2.0, 1.0};
  const double n = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  for (int a = 0; a < 3; ++a) u[a] = g * beta * d[a] / n;
}

// Upload one/alloc particles (setup, excluded from phase timers).
struct DevPart {
  double* pos = nullptr;
  double* w = nullptr;
  int P = 0;
  ~DevPart() { if (pos) cudaFree(pos); if (w) cudaFree(w); }
};

static void upload(DevPart& dp, const std::vector<double>& pos,
                   const std::vector<double>& w) {
  dp.P = (int)w.size();
  if (dp.P == 0) return;
  CK(cudaMalloc(&dp.pos, sizeof(double) * 3 * dp.P));
  CK(cudaMalloc(&dp.w, sizeof(double) * dp.P));
  CK(cudaMemcpy(dp.pos, pos.data(), sizeof(double) * 3 * dp.P,
                cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dp.w, w.data(), sizeof(double) * dp.P,
                cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// modes
// ---------------------------------------------------------------------------
static int modeValidate(Env& env, bool useFloat) {
  const int N = 48, pad = 4;
  const int gammas[3] = {1, 10, 100};
  const double aspects[3] = {0.25, 4.0, 1.0};
  const int center = N / 2;
  double cell[3] = {1.0, 1.0, 1.0};
  double u[3] = {0, 0, 0};
  printf("gamma,aspect,axis,d,comp,mesh\n");
  for (int gi = 0; gi < 3; ++gi) {
    for (int ai = 0; ai < 3; ++ai) {
      Sim s;
      s.N = N; s.pad = pad; s.h = g_h;
      s.cell[0] = cell[0];
      s.cell[1] = aspects[ai];
      s.cell[2] = cell[2];
      vel(gammas[gi], u);
      s.u[0] = u[0]; s.u[1] = u[1]; s.u[2] = u[2];
      std::vector<double> pos = {(double)center * s.h * s.cell[0],
                                 (double)center * s.h * s.cell[1],
                                 (double)center * s.h * s.cell[2]};
      std::vector<double> w = {1.0};
      DevPart dp;
      upload(dp, pos, w);
      Phase ph;
      std::vector<double> F = runMesh(env, s, dp.pos, dp.w, 1, useFloat, &ph, true);
      const int64_t N3 = (int64_t)N * N * N;
      for (int ax = 0; ax < 2; ++ax) {  // 0 = x axis, 1 = y axis
        for (int d = 2; d <= 16; ++d) {
          int ti = center, tj = center, tk = center;
          if (ax == 0) ti += d; else tj += d;
          const int64_t idx = ti + (int64_t)N * (tj + (int64_t)N * tk);
          for (int c = 0; c < 6; ++c)
            printf("%d,%.4g,%s,%d,%s,%.17g\n", gammas[gi], aspects[ai],
                   ax == 0 ? "x" : "y", d, kComp[c],
                   F[(size_t)c * N3 + idx]);
        }
      }
    }
  }
  return 0;
}

static int modeNopad(Env& env) {
  const int N = 32, center = N / 2;
  Sim s;
  s.N = N; s.h = 1.0;
  s.cell[0] = s.cell[1] = s.cell[2] = 1.0;
  s.u[0] = s.u[1] = s.u[2] = 0.0;
  std::vector<double> pos = {0.375 * N, 0.42 * N, 0.55 * N};
  std::vector<double> w = {1.0};
  DevPart dp;
  upload(dp, pos, w);
  Phase p1, p2;
  s.pad = 1;
  std::vector<double> F1 = runMesh(env, s, dp.pos, dp.w, 1, false, &p1, false);
  s.pad = 2;
  std::vector<double> F2 = runMesh(env, s, dp.pos, dp.w, 1, false, &p2, false);
  const int64_t N3 = (int64_t)N * N * N;
  double worst = 0.0;
  int wi = -1, wj = -1, wk = -1, wc = -1;
  for (int c = 0; c < 6; ++c)
    for (int k = 0; k < N; ++k)
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) {
          const int64_t idx = i + (int64_t)N * (j + (int64_t)N * k);
          const double a = F1[(size_t)c * N3 + idx];
          const double b = F2[(size_t)c * N3 + idx];
          const double e = fabs(a - b) / (1.0 + fmax(fabs(a), fabs(b)));
          if (e > worst) { worst = e; wi = i; wj = j; wk = k; wc = c; }
        }
  printf("N=%d pad=1(M=%d) vs pad=2(M=%d), stationary gamma=1, one source at "
         "(%.3f,%.3f,%.3f) cells (T139/T142 position)\n", N, N, 2 * N,
         pos[0], pos[1], pos[2]);
  printf("GLOBAL max |F_periodic-F_free|/(1+max) = %.6e at (i,j,k)=(%d,%d,%d) %s\n",
         worst, wi, wj, wk, kComp[wc]);
  printf("free-space F at that point = %.6e, periodic = %.6e\n",
         F2[(size_t)wc * N3 + wi + (int64_t)N * (wj + (int64_t)N * wk)],
         F1[(size_t)wc * N3 + wi + (int64_t)N * (wj + (int64_t)N * wk)]);
  return 0;
}

// Refinement regression: the same physical source/target separation must give
// the same field at h and h/2.  This is self-contained (no indfield.py) and
// fails by a factor h^-3 if the quadrature measure is missing from `scale`.
static int modeRefine(Env& env) {
  const int N = 48, pad = 4, center = N / 2;
  const int64_t N3 = (int64_t)N * N * N;
  const int gammas[2] = {1, 10};
  const int Ds[3] = {4, 6, 8};
  double worst = 0.0;
  printf("gamma,D,comp,h1,h05,rel\n");
  for (int gi = 0; gi < 2; ++gi) {
    double u[3];
    vel(gammas[gi], u);
    std::vector<double> F[2];
    for (int r = 0; r < 2; ++r) {
      Sim s;
      s.N = N; s.pad = pad; s.h = (r == 0) ? 1.0 : 0.5;
      s.cell[0] = s.cell[1] = s.cell[2] = 1.0;
      s.u[0] = u[0]; s.u[1] = u[1]; s.u[2] = u[2];
      const double p = (double)center * s.h;
      std::vector<double> pos = {p, p, p};
      std::vector<double> w = {1.0};
      DevPart dp;
      upload(dp, pos, w);
      Phase ph;
      F[r] = runMesh(env, s, dp.pos, dp.w, 1, false, &ph, false);
    }
    for (int di = 0; di < 3; ++di) {
      const int D = Ds[di];
      const int i1 = center + D;          // h=1 node at distance D
      const int i05 = center + 2 * D;     // h=1/2 node at the same distance
      const int64_t idx1 = i1 + (int64_t)N * (center + (int64_t)N * center);
      const int64_t idx05 = i05 + (int64_t)N * (center + (int64_t)N * center);
      for (int c = 0; c < 6; ++c) {
        const double a = F[0][(size_t)c * N3 + idx1];
        const double b = F[1][(size_t)c * N3 + idx05];
        const double e = fabs(a - b) / (1.0 + fmax(fabs(a), fabs(b)));
        worst = fmax(worst, e);
        printf("%d,%d,%s,%.17g,%.17g,%.3e\n", gammas[gi], D, kComp[c],
               a, b, e);
      }
    }
  }
  printf("REFINE worst |h1-h05|/(1+max) = %.3e (gate 2e-3): %s\n", worst,
         worst < 2e-3 ? "PASS" : "FAIL");
  return worst < 2e-3 ? 0 : 1;
}

static int modeBench(Env& env, bool useFloat) {
  const int Ns[2] = {64, 128};          // 20-cell and 50-cell bunch containers
  const double bunch[2] = {20.0, 50.0};
  const int Ps[4] = {1000, 10000, 1000000, 2000000};
  printf("fft=%s\n", useFloat ? "float" : "double");
  printf("P,N,M,deposit_ms,kernel_ms,fft_ms,readout_ms,total_ms\n");
  for (int gidx = 0; gidx < 2; ++gidx) {
    const int N = Ns[gidx];
    const double L = bunch[gidx];
    for (int pi = 0; pi < 4; ++pi) {
      const int P = Ps[pi];
      Sim s;
      s.N = N; s.pad = 2; s.h = 1.0;
      s.cell[0] = s.cell[1] = s.cell[2] = 1.0;
      double u[3];
      vel(10, u);
      s.u[0] = u[0]; s.u[1] = u[1]; s.u[2] = u[2];
      // uniform cube of side L centred in the N-node box
      std::vector<double> pos((size_t)3 * P), w((size_t)P);
      const double lo = 0.5 * (N - L);
      uint64_t st = 0x9E3779B97F4A7C15ULL ^ (uint64_t)P ^ (uint64_t)N;
      auto rnd = [&]() {
        st ^= st >> 12; st ^= st << 25; st ^= st >> 27;
        return (double)((st * 0x2545F4914F6CDD1DULL) >> 11) *
               (1.0 / 9007199254740992.0);
      };
      for (int p = 0; p < P; ++p) {
        pos[3 * p + 0] = lo + L * rnd();
        pos[3 * p + 1] = lo + L * rnd();
        pos[3 * p + 2] = lo + L * rnd();
        w[p] = 1.0 / (double)P;
      }
      DevPart dp;
      upload(dp, pos, w);
      Phase ph;
      std::vector<double> F = runMesh(env, s, dp.pos, dp.w, P, useFloat, &ph, false);
      const double tot = ph.dep + ph.kern + ph.fft + ph.read;
      printf("%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n", P, N, s.M(), ph.dep,
             ph.kern, ph.fft, ph.read, tot);
      fflush(stdout);
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  std::string mode = "validate";
  bool useFloat = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--fft" && i + 1 < argc) useFloat = (std::string(argv[++i]) == "float");
    else if (a == "--h" && i + 1 < argc) g_h = std::atof(argv[++i]);
    else if (a == "--help") {
      printf("usage: %s --mode {validate,refine,nopad,bench,info} [--fft double|float] [--h H]\n",
             argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (mode == "info")
  {
    printf("fft_backend=alpakaVendor\nalpakaVendor_pin=%s\n", kAlpakaVendorPin);
    printf("transform=3D C2C, in-place, Normalization::none\n");
    printf("precision=float|double (PlanBuilder<Complex<{float,double}>,Vec<uint32_t,3>>)\n");
    return 0;
  }
  Env env = makeEnv();
  if (mode == "validate") return modeValidate(env, useFloat);
  if (mode == "refine") return modeRefine(env);
  if (mode == "nopad") return modeNopad(env);
  if (mode == "bench") return modeBench(env, useFloat);
  fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
