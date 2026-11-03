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
// double.  `--fft float` only casts the two FFT inputs and runs cuFFT in single
// precision, so the float-FTT experiment is isolated from the kernel accuracy.
//
// Build (rosi_v100):
//   PATH=/usr/local/cuda/bin:$PATH nvcc -O3 -arch=sm_70 -std=c++17 \
//        meshconv.cu -lcufft -o meshconv

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <cufft.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
          cudaGetErrorString(e_)); exit(1); } } while (0)
#define FK(x) do { cufftResult r_ = (x); if (r_ != CUFFT_SUCCESS) { \
  fprintf(stderr, "cuFFT error %s:%d: %d\n", __FILE__, __LINE__, (int)r_); \
  exit(1); } } while (0)

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

__global__ void realToComplexD(const double* __restrict__ r,
                               cufftDoubleComplex* __restrict__ c, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  c[i].x = r[i];
  c[i].y = 0.0;
}
__global__ void realToComplexF(const double* __restrict__ r,
                               cufftComplex* __restrict__ c, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  c[i].x = (float)r[i];
  c[i].y = 0.0f;
}
__global__ void mulD(cufftDoubleComplex* __restrict__ a,
                     const cufftDoubleComplex* __restrict__ b, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double ar = a[i].x, ai = a[i].y, br = b[i].x, bi = b[i].y;
  a[i].x = ar * br - ai * bi;
  a[i].y = ar * bi + ai * br;
}
__global__ void mulF(cufftComplex* __restrict__ a,
                     const cufftComplex* __restrict__ b, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float ar = a[i].x, ai = a[i].y, br = b[i].x, bi = b[i].y;
  a[i].x = ar * br - ai * bi;
  a[i].y = ar * bi + ai * br;
}
__global__ void extractD(const cufftDoubleComplex* __restrict__ src, int M,
                         double* __restrict__ out, int N, int64_t N3,
                         double scale) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N3) return;
  const int i = (int)(idx % N);
  const int j = (int)((idx / N) % N);
  const int k = (int)(idx / ((int64_t)N * N));
  out[idx] = src[i + (int64_t)M * (j + (int64_t)M * k)].x * scale;
}
__global__ void extractF(const cufftComplex* __restrict__ src, int M,
                         float* __restrict__ out, int N, int64_t N3,
                         float scale) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N3) return;
  const int i = (int)(idx % N);
  const int j = (int)((idx / N) % N);
  const int k = (int)(idx / ((int64_t)N * N));
  out[idx] = src[i + (int64_t)M * (j + (int64_t)M * k)].x * scale;
}

// ---------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------
static double nowEvents(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, a, b));
  return (double)ms;
}

// Runs deposit + kernel build + FFT + readout.  particles already on device.
// Returns F[c*N3 + i + N*(j + N*k)] on the host.
static std::vector<double> runMesh(const Sim& s,
                                   const double* d_pos, const double* d_w,
                                   int P, bool useFloat, Phase* ph,
                                   bool verbose) {
  const int N = s.N, M = s.M();
  const int64_t M3 = (int64_t)M * M * M;
  const int64_t N3 = (int64_t)N * N * N;
  const double h3 = s.h * s.h * s.h;

  CK(cudaMemcpyToSymbol(c_off, kOffsets, sizeof(kOffsets)));

  double* d_rho = nullptr;
  double* d_K = nullptr;
  CK(cudaMalloc(&d_rho, sizeof(double) * M3));
  CK(cudaMalloc(&d_K, sizeof(double) * 6 * M3));

  std::vector<double> hostF((size_t)6 * N3, 0.0);
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));

  // ---- deposit -----------------------------------------------------------
  CK(cudaEventRecord(e0));
  CK(cudaMemset(d_rho, 0, sizeof(double) * M3));
  if (P > 0) {
    const int thr = 256, blk = (P + thr - 1) / thr;
    depositK<<<blk, thr>>>(d_pos, d_w, P, d_rho, N, M, s.h, s.cell[0],
                           s.cell[1], s.cell[2]);
    CK(cudaGetLastError());
  }
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  ph->dep = nowEvents(e0, e1);

  // ---- kernel build ------------------------------------------------------
  CK(cudaEventRecord(e0));
  {
    const int thr = 256;
    const int64_t blk = (M3 + thr - 1) / thr;
    buildKernelK<<<(int)blk, thr>>>(d_K, M, s.h, s.cell[0], s.cell[1],
                                    s.cell[2], s.u[0], s.u[1], s.u[2]);
    CK(cudaGetLastError());
  }
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  ph->kern = nowEvents(e0, e1);

  if (!useFloat) {
    cufftDoubleComplex* d_rhoHat = nullptr;
    CK(cudaMalloc(&d_rhoHat, sizeof(cufftDoubleComplex) * M3));
    cufftDoubleComplex* hbuf[6];
    for (int c = 0; c < 6; ++c)
      CK(cudaMalloc(&hbuf[c], sizeof(cufftDoubleComplex) * M3));
    cufftHandle plan;
    FK(cufftPlan3d(&plan, M, M, M, CUFFT_Z2Z));
    CK(cudaEventRecord(e0));
    const int thr = 256;
    const int64_t blk = (M3 + thr - 1) / thr;
    realToComplexD<<<(int)blk, thr>>>(d_rho, d_rhoHat, M3);
    CK(cudaGetLastError());
    FK(cufftExecZ2Z(plan, d_rhoHat, d_rhoHat, CUFFT_FORWARD));
    for (int c = 0; c < 6; ++c) {
      realToComplexD<<<(int)blk, thr>>>(d_K + (int64_t)c * M3, hbuf[c], M3);
      CK(cudaGetLastError());
      FK(cufftExecZ2Z(plan, hbuf[c], hbuf[c], CUFFT_FORWARD));
      mulD<<<(int)blk, thr>>>(hbuf[c], d_rhoHat, M3);
      CK(cudaGetLastError());
      FK(cufftExecZ2Z(plan, hbuf[c], hbuf[c], CUFFT_INVERSE));
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    ph->fft = nowEvents(e0, e1);

    // ---- readout ---------------------------------------------------------
    double* d_out = nullptr;
    CK(cudaMalloc(&d_out, sizeof(double) * 6 * N3));
    CK(cudaEventRecord(e0));
    const double scale = 1.0 / (double)M3;  // cuFFT inverse is unnormalised
    const int64_t blkN = (N3 + thr - 1) / thr;
    for (int c = 0; c < 6; ++c)
      extractD<<<(int)blkN, thr>>>(hbuf[c], M, d_out + (int64_t)c * N3, N, N3,
                                   scale);
    CK(cudaGetLastError());
    CK(cudaMemcpy(hostF.data(), d_out, sizeof(double) * 6 * N3,
                  cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    ph->read = nowEvents(e0, e1);

    for (int c = 0; c < 6; ++c) CK(cudaFree(hbuf[c]));
    CK(cudaFree(d_rhoHat));
    CK(cudaFree(d_out));
    FK(cufftDestroy(plan));
  } else {
    cufftComplex *d_rhoHat = nullptr;
    CK(cudaMalloc(&d_rhoHat, sizeof(cufftComplex) * M3));
    cufftComplex* hbuf[6];
    for (int c = 0; c < 6; ++c)
      CK(cudaMalloc(&hbuf[c], sizeof(cufftComplex) * M3));
    cufftHandle plan;
    FK(cufftPlan3d(&plan, M, M, M, CUFFT_C2C));
    CK(cudaEventRecord(e0));
    const int thr = 256;
    const int64_t blk = (M3 + thr - 1) / thr;
    realToComplexF<<<(int)blk, thr>>>(d_rho, d_rhoHat, M3);
    CK(cudaGetLastError());
    FK(cufftExecC2C(plan, d_rhoHat, d_rhoHat, CUFFT_FORWARD));
    for (int c = 0; c < 6; ++c) {
      realToComplexF<<<(int)blk, thr>>>(d_K + (int64_t)c * M3, hbuf[c], M3);
      CK(cudaGetLastError());
      FK(cufftExecC2C(plan, hbuf[c], hbuf[c], CUFFT_FORWARD));
      mulF<<<(int)blk, thr>>>(hbuf[c], d_rhoHat, M3);
      CK(cudaGetLastError());
      FK(cufftExecC2C(plan, hbuf[c], hbuf[c], CUFFT_INVERSE));
    }
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    ph->fft = nowEvents(e0, e1);

    float* d_out = nullptr;
    CK(cudaMalloc(&d_out, sizeof(float) * 6 * N3));
    std::vector<float> tmp((size_t)6 * N3);
    CK(cudaEventRecord(e0));
    const float scale = 1.0f / (float)M3;
    const int64_t blkN = (N3 + thr - 1) / thr;
    for (int c = 0; c < 6; ++c)
      extractF<<<(int)blkN, thr>>>(hbuf[c], M, d_out + (int64_t)c * N3, N, N3,
                                   scale);
    CK(cudaGetLastError());
    CK(cudaMemcpy(tmp.data(), d_out, sizeof(float) * 6 * N3,
                  cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < 6 * N3; ++i) hostF[i] = (double)tmp[i];
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    ph->read = nowEvents(e0, e1);

    for (int c = 0; c < 6; ++c) CK(cudaFree(hbuf[c]));
    CK(cudaFree(d_rhoHat));
    CK(cudaFree(d_out));
    FK(cufftDestroy(plan));
  }

  CK(cudaFree(d_rho));
  CK(cudaFree(d_K));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  if (verbose)
    fprintf(stderr, "[run] N=%d M=%d P=%d dep=%.2fms kern=%.2fms "
                    "fft=%.2fms read=%.2fms\n",
            N, M, P, ph->dep, ph->kern, ph->fft, ph->read);
  return hostF;
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
static int modeValidate(bool useFloat) {
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
      s.N = N; s.pad = pad; s.h = 1.0;
      s.cell[0] = cell[0];
      s.cell[1] = aspects[ai];
      s.cell[2] = cell[2];
      vel(gammas[gi], u);
      s.u[0] = u[0]; s.u[1] = u[1]; s.u[2] = u[2];
      std::vector<double> pos = {(double)center * s.cell[0],
                                 (double)center * s.cell[1],
                                 (double)center * s.cell[2]};
      std::vector<double> w = {1.0};
      DevPart dp;
      upload(dp, pos, w);
      Phase ph;
      std::vector<double> F = runMesh(s, dp.pos, dp.w, 1, useFloat, &ph, true);
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

static int modeNopad() {
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
  std::vector<double> F1 = runMesh(s, dp.pos, dp.w, 1, false, &p1, false);
  s.pad = 2;
  std::vector<double> F2 = runMesh(s, dp.pos, dp.w, 1, false, &p2, false);
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

static int modeBench(bool useFloat) {
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
      std::vector<double> F = runMesh(s, dp.pos, dp.w, P, useFloat, &ph, false);
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
    else if (a == "--help") {
      printf("usage: %s --mode {validate,nopad,bench} [--fft double|float]\n",
             argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (mode == "validate") return modeValidate(useFloat);
  if (mode == "nopad") return modeNopad();
  if (mode == "bench") return modeBench(useFloat);
  fprintf(stderr, "unknown mode %s\n", mode.c_str());
  return 2;
}
