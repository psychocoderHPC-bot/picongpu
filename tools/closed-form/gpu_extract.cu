// gpu_extract.cu — CF1: singularity-extraction in-support near-field kernel + benchmark.
//
// Run pr24-20261102, task CF1 (restart of the stopped closed-form work).
//
// Quantity (same as gpu_near.cu / S1-algorithm-spec.md sec.1): for one
// (source, target staggered point, component) pair the TSC-convolved
// uniformly-moving point-charge field
//
//   F_c = integral_{[-1.5,1.5]^3} W(s) Kpoint_c(R0 - s) ds^3 ,  R0 = x_obs - x_src
//
// W = Wx Wy Wz separable TSC (support [-1.5,1.5]/axis), Kpoint the per-unit-charge
// field of one uniformly moving charge (stationary: Coulomb).
//
// Two rules:
//   * sing: the GPU1 power-transform scheme (knot split + s* anchor split, t^5 map),
//     order N per panel. Copied unchanged from gpu_near.cu; kept as the validated
//     baseline and for cross-checks.
//   * extract: the CF1 singularity extraction (stationary P2 config of cf1_step2.py,
//     moving-axial config of cf1_step3.py):
//
//     F_c = W(s*) * Ebox_c(s*) + R_c ,   s* = staggered R0
//
//     Ebox = integral over the unit-density support box (exact 8x T closed form,
//       O(1), 24 T calls; T = int_0^A int_0^B 1/sqrt(u^2+X^2+Y^2) dX dY with the
//       |u| -> 0 branch fixed to fire on abs(u) < 1e-300).
//     Moving-axial (beta || z): Ebox = Coulomb 8xT of the gamma-stretched box
//       (stretch about the field point in d = s - s* coordinates) with the axial
//       component divided by gamma; Bbox = beta x Ebox.
//     R_c = integral (S(s) - W(s*)) Kpoint_c(s* - s) ds^3  (smooth-ish remainder)
//       by plain tensor Gauss-Legendre of `order` per panel over the knot-split
//       panels {-1.5,-0.5,0.5,1.5}, with each axis additionally split at s*_a
//       where -1.5 < s*_a < 1.5 (matches cf1_core P2; the split helps even
//       off-support by resolving the nearest-approach variation of K).
//       For non-singular pairs W(s*) = 0: no box term, plain GL.
//
// One thread per (source, node set, component) pair; all precision work in double.
// Device code receives constant tables (GL nodes/weights, Yee offsets) as kernel
// parameters (global constexpr arrays are not visible in CUDA device code).

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr double kKnots[4] = {-1.5, -0.5, 0.5, 1.5};
    // Yee offsets in cell units, order (Ex,Ey,Ez,Bx,By,Bz), row-major 6x3.
    constexpr double kDelta[6][3] = {
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};

    // PIConGPU TSC (quadratic B-spline), argument in cell units.
    ALPAKA_FN_ACC auto tsc(double x) -> double
    {
        double ax = x < 0.0 ? -x : x;
        if(ax < 0.5)
            return 0.75 - ax * ax;
        if(ax < 1.5)
        {
            double t = 1.5 - ax;
            return 0.5 * t * t;
        }
        return 0.0;
    }

    // int_0^A int_0^B 1/sqrt(u^2 + X^2 + Y^2) dX dY  (NS3-verified; CF1 fix: the
    // u->0 branch fires on abs(u), the form is even in u; A, B > 0 in this kernel).
    ALPAKA_FN_ACC auto Tfun(double A, double B, double u) -> double
    {
        if(std::abs(u) < 1e-300)
            return A * std::asinh(B / A) + B * std::asinh(A / B);
        return (A * std::asinh(B / std::sqrt(u * u + A * A))
                + B * std::asinh(A / std::sqrt(u * u + B * B))
                - u * std::atan(A * B / (u * std::sqrt(A * A + u * u + B * B))));
    }

    // Coulomb field component c at the ORIGIN of a unit-density box [lo,hi]^3,
    // 8 T calls. (Field at p of the support box == field at origin of the box
    // expressed in d = s - p coordinates.)
    ALPAKA_FN_ACC auto boxField8T(double const lo[3], double const hi[3], int c) -> double
    {
        double const x0 = hi[0], x1 = -lo[0];
        double const y0 = hi[1], y1 = -lo[1];
        double const z0 = hi[2], z1 = -lo[2];
        double E = 0.0;
        if(c == 0)
        {
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                {
                    double const sy = a ? y1 : y0;
                    double const sz = b ? z1 : z0;
                    E += Tfun(sy, sz, x0) - Tfun(sy, sz, x1);
                }
        }
        else if(c == 1)
        {
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                {
                    double const sx = a ? x1 : x0;
                    double const sz = b ? z1 : z0;
                    E += Tfun(sx, sz, y0) - Tfun(sx, sz, y1);
                }
        }
        else
        {
            for(int a = 0; a < 2; ++a)
                for(int b = 0; b < 2; ++b)
                {
                    double const sx = a ? x1 : x0;
                    double const sy = b ? y1 : y0;
                    E += Tfun(sx, sy, z0) - Tfun(sx, sy, z1);
                }
        }
        return E;
    }

    // Per-pair extract evaluation (host + device). Returns component c.
    ALPAKA_FN_ACC auto extractPair(
        double const r0[3],
        int c,
        double const delta[18],
        double const xi[],
        double const wi[],
        int order,
        double bx,
        double by,
        double bz,
        double invg2,
        bool stationary,
        int stretchAx,
        double gammaStretch) -> double
    {
        double const ox = r0[0] + delta[3 * c + 0];
        double const oy = r0[1] + delta[3 * c + 1];
        double const oz = r0[2] + delta[3 * c + 2];
        double const p[3] = {ox, oy, oz};
        double const Wstar = tsc(ox) * tsc(oy) * tsc(oz);
        bool const singular = Wstar > 0.0;
        double const kK[4] = {-1.5, -0.5, 0.5, 1.5};  // local: global constexpr arrays are
                                                       // not visible in CUDA device code

        // exact box term (only needed in-support)
        double Ebox[3] = {0.0, 0.0, 0.0};
        if(singular)
        {
            double lo[3], hi[3];
            for(int a = 0; a < 3; ++a)
            {
                lo[a] = -1.5 - p[a];
                hi[a] = 1.5 - p[a];
            }
            if(stretchAx >= 0)
            {
                lo[stretchAx] *= gammaStretch;
                hi[stretchAx] *= gammaStretch;
            }
            Ebox[0] = boxField8T(lo, hi, 0);
            Ebox[1] = boxField8T(lo, hi, 1);
            Ebox[2] = boxField8T(lo, hi, 2);
            if(stretchAx >= 0)
                Ebox[stretchAx] /= gammaStretch;
        }

        // P2 panels: knot split per axis + s* split on interior axes (matches
        // cf1_core._panels_axis: the split helps even off-support, where it
        // resolves the nearest-approach variation of K).
        double plo[3][4], phi[3][4];
        int npan[3];
        for(int a = 0; a < 3; ++a)
        {
            if(p[a] > -1.5 && p[a] < 1.5)
            {
                int m = 0;
                for(int k = 0; k < 3; ++k)
                {
                    if(p[a] > kK[k] && p[a] < kK[k + 1])
                    {
                        plo[a][m] = kK[k];
                        phi[a][m] = p[a];
                        ++m;
                        plo[a][m] = p[a];
                        phi[a][m] = kK[k + 1];
                        ++m;
                    }
                    else
                    {
                        plo[a][m] = kK[k];
                        phi[a][m] = kK[k + 1];
                        ++m;
                    }
                }
                npan[a] = m;
            }
            else
            {
                for(int k = 0; k < 3; ++k)
                {
                    plo[a][k] = kK[k];
                    phi[a][k] = kK[k + 1];
                }
                npan[a] = 3;
            }
        }

        int const NX = npan[0] * order, NY = npan[1] * order, NZ = npan[2] * order;
        double Er[3] = {0.0, 0.0, 0.0};
        for(int ii = 0; ii < NX; ++ii)
        {
            int const gx = ii % order, px = ii / order;
            double const mx = 0.5 * (phi[0][px] - plo[0][px]);
            double const sx = 0.5 * (plo[0][px] + phi[0][px]) + mx * xi[gx];
            double const wx = mx * wi[gx];
            for(int jj = 0; jj < NY; ++jj)
            {
                int const gy = jj % order, py = jj / order;
                double const my = 0.5 * (phi[1][py] - plo[1][py]);
                double const sy = 0.5 * (plo[1][py] + phi[1][py]) + my * xi[gy];
                double const wy = my * wi[gy];
                double const Sxy = tsc(sx) * tsc(sy);
                double const wxy = wx * wy;
                for(int kk = 0; kk < NZ; ++kk)
                {
                    int const gz = kk % order, pz = kk / order;
                    double const mz = 0.5 * (phi[2][pz] - plo[2][pz]);
                    double const sz = 0.5 * (plo[2][pz] + phi[2][pz]) + mz * xi[gz];
                    double const S = Sxy * tsc(sz);
                    double const wc = (S - Wstar) * wxy * mz * wi[gz];
                    double const Rx = ox - sx, Ry = oy - sy, Rz = oz - sz;
                    if(stationary)
                    {
                        double r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                        if(r2 < 1e-28)
                            r2 = 1e-28;
                        double const inv = 1.0 / (r2 * std::sqrt(r2));
                        Er[0] += wc * Rx * inv;
                        Er[1] += wc * Ry * inv;
                        Er[2] += wc * Rz * inv;
                    }
                    else
                    {
                        double const r2 = Rx * Rx + Ry * Ry + Rz * Rz;
                        double const rb = Rx * bx + Ry * by + Rz * bz;
                        double D = r2 * invg2 + rb * rb;
                        if(D < 1e-28)
                            D = 1e-28;
                        double const f = invg2 / (D * std::sqrt(D));
                        Er[0] += wc * f * Rx;
                        Er[1] += wc * f * Ry;
                        Er[2] += wc * f * Rz;
                    }
                }
            }
        }

        if(c < 3)
            return Wstar * Ebox[c] + Er[c];
        double const Brem[3] = {
            by * Er[2] - bz * Er[1],
            bz * Er[0] - bx * Er[2],
            bx * Er[1] - by * Er[0]};
        double const Bbox[3] = {
            by * Ebox[2] - bz * Ebox[1],
            bz * Ebox[0] - bx * Ebox[2],
            bx * Ebox[1] - by * Ebox[0]};
        return Wstar * Bbox[c - 3] + Brem[c - 3];
    }

    // Host Gauss-Legendre nodes/weights of order n on [-1,1], ascending (Newton on P_n).
    void hostLeggauss(int n, double* x, double* w)
    {
        int const m = (n + 1) / 2;
        for(int i = 0; i < m; ++i)
        {
            double z = std::cos(M_PI * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5));
            double pp = 0.0;
            for(int it = 0; it < 100; ++it)
            {
                double const z1 = z;
                double p0 = 1.0, p1 = z1, p = 0.0;
                for(int j = 2; j <= n; ++j)
                {
                    p = ((2.0 * j - 1.0) * z1 * p1 - (j - 1.0) * p0) / j;
                    p0 = p1;
                    p1 = p;
                }
                pp = static_cast<double>(n) * (z1 * p1 - p0) / (z1 * z1 - 1.0);
                z = z1 - p1 / pp;
                if(std::abs(z - z1) < 1e-15)
                    break;
            }
            x[i] = -z;
            x[n - 1 - i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
            w[n - 1 - i] = w[i];
        }
    }

    // Per-unit-charge field of one uniformly moving charge (independent-reference form).
    ALPAKA_FN_ACC auto pointField6dev(
        double Rx,
        double Ry,
        double Rz,
        double bx,
        double by,
        double bz,
        double invg2,
        bool stationary,
        double E[3],
        double B[3]) -> void
    {
        double const r2 = Rx * Rx + Ry * Ry + Rz * Rz;
        if(stationary)
        {
            double const inv = 1.0 / (r2 * std::sqrt(r2));
            E[0] = Rx * inv;
            E[1] = Ry * inv;
            E[2] = Rz * inv;
            B[0] = B[1] = B[2] = 0.0;
            return;
        }
        double const rb = Rx * bx + Ry * by + Rz * bz;
        double const D = r2 * invg2 + rb * rb;
        double const D15 = D * std::sqrt(D);
        double const f = invg2 / D15;
        E[0] = f * Rx;
        E[1] = f * Ry;
        E[2] = f * Rz;
        B[0] = by * E[2] - bz * E[1];
        B[1] = bz * E[0] - bx * E[2];
        B[2] = bx * E[1] - by * E[0];
    }
}  // namespace

// ---------------------------------------------------------------------------
// EXTRACT kernel: one thread per (source, component) pair; closed-form box term
// + P2 plain-GL remainder.
// ---------------------------------------------------------------------------
struct ExtractKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::IMdSpan auto out,  // (nPairs,)
        alpaka::concepts::IDataSource auto const& r0,  // (3*nPairs,)
        alpaka::concepts::IDataSource auto const& comp,  // (nPairs,)
        alpaka::concepts::IDataSource auto const& xi,
        alpaka::concepts::IDataSource auto const& wi,
        alpaka::concepts::IDataSource auto const& delta,  // (18,)
        double bx,
        double by,
        double bz,
        double invg2,
        bool stationary,
        int order,
        int stretchAx,
        double gammaStretch,
        std::uint32_t nPairs) const
    {
        for(auto [idx] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{alpaka::Vec<std::uint32_t, 1u>{nPairs}}))
        {
            std::uint32_t const i = idx;
            double r0v[3] = {r0[3 * i + 0], r0[3 * i + 1], r0[3 * i + 2]};
            double delt[18];
            for(int k = 0; k < 18; ++k)
                delt[k] = delta[k];
            double xiv[16], wiv[16];
            for(int k = 0; k < 16; ++k)
            {
                xiv[k] = xi[k];
                wiv[k] = wi[k];
            }
            out[i] = extractPair(r0v, static_cast<int>(comp[i]), delt, xiv, wiv, order,
                                 bx, by, bz, invg2, stationary, stretchAx, gammaStretch);
        }
    }
};

// ---------------------------------------------------------------------------
// SING kernel: GPU1 power-transform scheme (knot split + s* anchor split, t^5),
// copied from gpu_near.cu.
// ---------------------------------------------------------------------------
struct SingKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::IMdSpan auto out,  // (nPairs,)
        alpaka::concepts::IDataSource auto const& r0,  // (3*nPairs,)
        alpaka::concepts::IDataSource auto const& comp,  // (nPairs,)
        alpaka::concepts::IDataSource auto const& xi,  // (8,) GL nodes [-1,1]
        alpaka::concepts::IDataSource auto const& wi,  // (8,) GL weights
        alpaka::concepts::IDataSource auto const& knots,  // (4,)
        alpaka::concepts::IDataSource auto const& delta,  // (18,)
        double bx,
        double by,
        double bz,
        double invg2,
        bool stationary,
        int order,
        int refine,
        std::uint32_t nPairs) const
    {
        constexpr double kP = 5.0;
        int const nseg = 1 << refine;
        int const po = order * nseg;
        for(auto [idx] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{alpaka::Vec<std::uint32_t, 1u>{nPairs}}))
        {
            std::uint32_t const i = idx;
            int const c = static_cast<int>(comp[i]);
            double const ox = r0[3 * i + 0] + delta[3 * c + 0];
            double const oy = r0[3 * i + 1] + delta[3 * c + 1];
            double const oz = r0[3 * i + 2] + delta[3 * c + 2];
            double const rr[3] = {ox, oy, oz};
            bool const singular = (std::abs(ox) < 1.5) && (std::abs(oy) < 1.5) && (std::abs(oz) < 1.5);
            double pc[3][4];
            double pb[3][4];
            double pab[3][4];
            int pn[3];
            for(int a = 0; a < 3; ++a)
            {
                int m = 0;
                for(int p = 0; p < 3; ++p)
                {
                    double const lo = knots[p];
                    double const hi = knots[p + 1];
                    auto add = [&](double a0, double b0)
                    {
                        double cn, on;
                        if(singular)
                        {
                            if(std::abs(a0 - rr[a]) <= std::abs(b0 - rr[a]))
                            {
                                cn = a0;
                                on = b0;
                            }
                            else
                            {
                                cn = b0;
                                on = a0;
                            }
                        }
                        else
                        {
                            cn = a0;
                            on = b0;
                        }
                        pc[a][m] = cn;
                        pb[a][m] = on;
                        pab[a][m] = std::abs(on - cn);
                        ++m;
                    };
                    if(singular && lo < rr[a] && rr[a] < hi)
                    {
                        add(lo, rr[a]);
                        add(rr[a], hi);
                    }
                    else
                    {
                        add(lo, hi);
                    }
                }
                pn[a] = m;
            }
            double sum = 0.0;
            int const NX = pn[0] * po, NY = pn[1] * po, NZ = pn[2] * po;
            if(!singular)
            {
                for(int ii = 0; ii < NX; ++ii)
                    for(int jj = 0; jj < NY; ++jj)
                        for(int kk = 0; kk < NZ; ++kk)
                        {
                            double sx, sy, sz, wx, wy, wz;
                            {
                                int const g = ii % order, seg = (ii / order) % nseg, p = ii / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                sx = pc[0][p] + (pb[0][p] - pc[0][p]) * t;
                                wx = wi[g] * (0.5 / nseg) * pab[0][p] * tsc(sx);
                            }
                            {
                                int const g = jj % order, seg = (jj / order) % nseg, p = jj / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                sy = pc[1][p] + (pb[1][p] - pc[1][p]) * t;
                                wy = wi[g] * (0.5 / nseg) * pab[1][p] * tsc(sy);
                            }
                            {
                                int const g = kk % order, seg = (kk / order) % nseg, p = kk / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                sz = pc[2][p] + (pb[2][p] - pc[2][p]) * t;
                                wz = wi[g] * (0.5 / nseg) * pab[2][p] * tsc(sz);
                            }
                            double E[3];
                            double B[3];
                            pointField6dev(ox - sx, oy - sy, oz - sz, bx, by, bz, invg2, stationary, E, B);
                            sum += (wx * wy * wz) * ((c < 3) ? E[c] : B[c - 3]);
                        }
            }
            else
            {
                for(int ii = 0; ii < NX; ++ii)
                    for(int jj = 0; jj < NY; ++jj)
                        for(int kk = 0; kk < NZ; ++kk)
                        {
                            double sx, sy, sz, wx, wy, wz;
                            {
                                int const g = ii % order, seg = (ii / order) % nseg, p = ii / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                double const t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
                                sx = pc[0][p] + (pb[0][p] - pc[0][p]) * t5;
                                wx = wi[g] * (0.5 / nseg) * pab[0][p] * kP * t4 * tsc(sx);
                            }
                            {
                                int const g = jj % order, seg = (jj / order) % nseg, p = jj / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                double const t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
                                sy = pc[1][p] + (pb[1][p] - pc[1][p]) * t5;
                                wy = wi[g] * (0.5 / nseg) * pab[1][p] * kP * t4 * tsc(sy);
                            }
                            {
                                int const g = kk % order, seg = (kk / order) % nseg, p = kk / po;
                                double const t = (static_cast<double>(seg) + 0.5 + 0.5 * xi[g]) / nseg;
                                double const t2 = t * t, t4 = t2 * t2, t5 = t4 * t;
                                sz = pc[2][p] + (pb[2][p] - pc[2][p]) * t5;
                                wz = wi[g] * (0.5 / nseg) * pab[2][p] * kP * t4 * tsc(sz);
                            }
                            double E[3];
                            double B[3];
                            pointField6dev(ox - sx, oy - sy, oz - sz, bx, by, bz, invg2, stationary, E, B);
                            sum += (wx * wy * wz) * ((c < 3) ? E[c] : B[c - 3]);
                        }
            }
            out[i] = sum;
        }
    }
};

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------
namespace
{
    struct Args
    {
        const char* kernel = "extract";  // extract | sing
        const char* band = "in";  // in | band
        int order = 5;  // GL order per panel
        int refine = 0;  // panel bisections (sing rule only)
        std::uint32_t npairs = 1u << 20;
        int iters = 4;
        double gamma = 1.0;
        std::uint64_t seed = 0x51ed1234ULL;
        bool bench = true;
        bool check = true;
        bool xcheck = false;  // extract(12) vs sing(8,2) on the first 32 in-support pairs
        bool sample = false;
        const char* outpath = "cf1_sample";
    };

    void parseArgs(int argc, char** argv, Args& a)
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string s = argv[i];
            auto next = [&](const char* def) -> const char*
            {
                return (i + 1 < argc) ? argv[++i] : def;
            };
            if(s == "--kernel")
                a.kernel = next("extract");
            else if(s == "--band")
                a.band = next("in");
            else if(s == "--order")
                a.order = std::stoi(next("5"));
            else if(s == "--refine")
                a.refine = std::stoi(next("0"));
            else if(s == "--npairs")
                a.npairs = static_cast<std::uint32_t>(std::stoul(next("0")));
            else if(s == "--iters")
                a.iters = std::stoi(next("4"));
            else if(s == "--gamma")
                a.gamma = std::stod(next("1"));
            else if(s == "--seed")
                a.seed = std::stoull(next("0x51ed1234"));
            else if(s == "--nocheck")
                a.check = false;
            else if(s == "--xcheck")
                a.xcheck = true;
            else if(s == "--sample")
                a.sample = true;
            else if(s == "--out")
                a.outpath = next("cf1_sample");
        }
    }

    void velocity(double gamma, double beta[3], double& invg2, bool& stationary)
    {
        if(gamma <= 1.0)
        {
            beta[0] = beta[1] = beta[2] = 0.0;
            invg2 = 1.0;
            stationary = true;
            return;
        }
        double const u2 = std::sqrt(gamma * gamma - 1.0);
        beta[0] = beta[1] = 0.0;
        beta[2] = u2 / gamma;
        invg2 = 1.0 / (1.0 + u2 * u2);
        stationary = false;
    }

    void genPairs(
        std::uint32_t n,
        const char* band,
        std::uint64_t seed,
        std::vector<double>& rel,
        std::vector<int>& comp)
    {
        rel.resize(3 * static_cast<std::size_t>(n));
        comp.resize(n);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        std::uniform_real_distribution<double> uni01(0.0, 1.0);
        std::normal_distribution<double> gauss(0.0, 1.0);
        for(std::uint32_t i = 0; i < n; ++i)
        {
            int c = static_cast<int>(i % 6);
            bool inBand;
            if(std::string(band) == "in")
                inBand = true;
            else if(std::string(band) == "band")
                inBand = false;
            else
                inBand = (i % 2 == 0);
            double rx, ry, rz;
            if(inBand)
            {
                rx = 1.5 * uni(rng);
                ry = 1.5 * uni(rng);
                rz = 1.5 * uni(rng);
            }
            else
            {
                // 1.5-4 cell band OUTSIDE the support box (|rel|_inf >= 1.5)
                for(int attempt = 0; attempt < 64; ++attempt)
                {
                    double gx = gauss(rng), gy = gauss(rng), gz = gauss(rng);
                    double len = std::sqrt(gx * gx + gy * gy + gz * gz);
                    double rad = 1.5 + 2.5 * uni01(rng);
                    rx = rad * gx / len;
                    ry = rad * gy / len;
                    rz = rad * gz / len;
                    if(std::max({std::abs(rx), std::abs(ry), std::abs(rz)}) >= 1.5)
                        break;
                }
            }
            rel[3 * i] = rx;
            rel[3 * i + 1] = ry;
            rel[3 * i + 2] = rz;
            comp[i] = c;
        }
    }

    double flatDelta[18];
}  // namespace

int main(int argc, char** argv)
{
    using namespace alpaka;
    Args a;
    parseArgs(argc, argv, a);

    for(int c = 0; c < 6; ++c)
        for(int t = 0; t < 3; ++t)
            flatDelta[3 * c + t] = kDelta[c][t];

    double beta[3];
    double invg2;
    bool stationary;
    velocity(a.gamma, beta, invg2, stationary);
    int const stretchAx = (a.gamma > 1.0) ? 2 : -1;
    double const gammaStretch = (a.gamma > 1.0) ? std::sqrt(1.0 / invg2) : 1.0;

    bool const isExtract = std::string(a.kernel) == "extract";
    bool const isSing = std::string(a.kernel) == "sing";
    if(!isExtract && !isSing)
    {
        std::fprintf(stderr, "unknown kernel %s (extract|sing)\n", a.kernel);
        return 1;
    }
    int const nseg = 1 << a.refine;
    int const nodesPerPair = isExtract
        ? ((std::string(a.band) == "band") ? 48 : 64) * a.order * a.order * a.order
        : (std::string(a.band) == "band" ? 3 * nseg * a.order : 4 * nseg * a.order)
            * (std::string(a.band) == "band" ? 3 * nseg * a.order : 4 * nseg * a.order)
            * (std::string(a.band) == "band" ? 3 * nseg * a.order : 4 * nseg * a.order);

    std::printf("gpu_extract kernel=%s band=%s order=%d refine=%d gamma=%.4g npairs=%u iters=%d seed=0x%llx\n",
                a.kernel, a.band, a.order, a.refine, a.gamma, a.npairs, a.iters, static_cast<unsigned long long>(a.seed));
    std::printf("  beta=(%.4g,%.4g,%.4g) invg2=%.6g stationary=%d stretchAx=%d nodes/pair~%d\n",
                beta[0], beta[1], beta[2], invg2, (int)stationary, stretchAx, nodesPerPair);

    std::vector<double> rel;
    std::vector<int> comp;
    genPairs(a.npairs, a.band, a.seed, rel, comp);
    std::vector<double> hOut(a.npairs, 0.0);

    auto const deviceSpec = onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
    auto devSel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSel.makeDevice(0);
    std::printf("  device: %s\n", onHost::getName(device).c_str());
    onHost::Queue queue = device.makeQueue();

    auto dRel = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{3 * a.npairs});
    auto dComp = onHost::alloc<int>(device, Vec<std::uint32_t, 1u>{a.npairs});
    auto dOut = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{a.npairs});
    auto hRel = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{3 * a.npairs});
    auto hComp = onHost::allocHost<int>(Vec<std::uint32_t, 1u>{a.npairs});
    auto hOutV = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{a.npairs});
    for(std::uint32_t i = 0; i < 3 * a.npairs; ++i)
        hRel[i] = rel[i];
    for(std::uint32_t i = 0; i < a.npairs; ++i)
        hComp[i] = comp[i];
    onHost::memcpy(queue, dRel, hRel);
    onHost::memcpy(queue, dComp, hComp);

    auto dDelta = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{18});
    auto hDelta = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{18});
    for(int i = 0; i < 18; ++i)
        hDelta[i] = flatDelta[i];
    onHost::memcpy(queue, dDelta, hDelta);

    // GL tables: GL nodes of different orders are unrelated sets, so each order
    // used gets its own table (main run: a.order; xcheck: extract 12, sing 8).
    auto makeTable = [&](int ord) -> std::pair<std::vector<double>, std::vector<double>>
    {
        std::vector<double> hx(ord), hw(ord);
        hostLeggauss(ord, hx.data(), hw.data());
        return {hx, hw};
    };
    auto upTable = [&](auto const& htab, auto dX, auto dW)
    {
        auto hX = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(htab.first.size())});
        auto hW = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(htab.first.size())});
        for(std::size_t i = 0; i < htab.first.size(); ++i)
        {
            hX[i] = htab.first[i];
            hW[i] = htab.second[i];
        }
        onHost::memcpy(queue, dX, hX);
        onHost::memcpy(queue, dW, hW);
    };
    auto tMain = makeTable(a.order);
    auto dXiMain = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(a.order)});
    auto dWiMain = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{static_cast<std::uint32_t>(a.order)});
    upTable(tMain, dXiMain, dWiMain);
    auto t12 = makeTable(12);
    auto dXi12 = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{12});
    auto dWi12 = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{12});
    upTable(t12, dXi12, dWi12);
    auto t8 = makeTable(8);
    auto dXi8 = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{8});
    auto dWi8 = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{8});
    upTable(t8, dXi8, dWi8);
    auto dKnots = onHost::alloc<double>(device, Vec<std::uint32_t, 1u>{4});
    auto hKnots = onHost::allocHost<double>(Vec<std::uint32_t, 1u>{4});
    for(int i = 0; i < 4; ++i)
        hKnots[i] = kKnots[i];
    onHost::memcpy(queue, dKnots, hKnots);

    auto const exec = exec::gpuCuda;
    auto const frame = onHost::FrameSpec{divCeil(a.npairs, 256u), 256u, exec};

    auto launch = [&](int ord, int ref, bool useExtract, int sAx, double gStr)
    {
        if(ord == a.order)
        {
            if(useExtract)
                queue.enqueue(frame, ExtractKernel{}, dOut, dRel, dComp, dXiMain, dWiMain, dDelta,
                              beta[0], beta[1], beta[2], invg2, stationary, ord, sAx, gStr, a.npairs);
            else
                queue.enqueue(frame, SingKernel{}, dOut, dRel, dComp, dXiMain, dWiMain, dKnots, dDelta,
                              beta[0], beta[1], beta[2], invg2, stationary, ord, ref, a.npairs);
        }
        else if(ord == 12)
        {
            if(useExtract)
                queue.enqueue(frame, ExtractKernel{}, dOut, dRel, dComp, dXi12, dWi12, dDelta,
                              beta[0], beta[1], beta[2], invg2, stationary, ord, sAx, gStr, a.npairs);
            else
                queue.enqueue(frame, SingKernel{}, dOut, dRel, dComp, dXi12, dWi12, dKnots, dDelta,
                              beta[0], beta[1], beta[2], invg2, stationary, ord, ref, a.npairs);
        }
        else  // ord == 8
        {
            if(useExtract)
                queue.enqueue(frame, ExtractKernel{}, dOut, dRel, dComp, dXi8, dWi8, dDelta,
                              beta[0], beta[1], beta[2], invg2, stationary, ord, sAx, gStr, a.npairs);
            else
                queue.enqueue(frame, SingKernel{}, dOut, dRel, dComp, dXi8, dWi8, dKnots, dDelta,
                              beta[0], beta[1], beta[2], invg2, stationary, ord, ref, a.npairs);
        }
    };

    // warm-up
    launch(a.order, a.refine, isExtract, stretchAx, gammaStretch);
    onHost::wait(queue);
    onHost::memcpy(queue, hOutV, dOut);
    onHost::wait(queue);
    for(std::uint32_t i = 0; i < a.npairs; ++i)
        hOut[i] = hOutV[i];

    if(a.check && isExtract)
    {
        // kernel vs CPU same-scheme (extractPair on host)
        double delt[18];
        for(int i = 0; i < 18; ++i)
            delt[i] = flatDelta[i];
        std::uint32_t const nc = std::min<std::uint32_t>(64, a.npairs);
        double maxerr = 0.0;
        for(std::uint32_t i = 0; i < nc; ++i)
        {
            double r0[3] = {rel[3 * i], rel[3 * i + 1], rel[3 * i + 2]};
            double ref = extractPair(r0, comp[i], delt, tMain.first.data(), tMain.second.data(), a.order,
                                     beta[0], beta[1], beta[2], invg2, stationary, stretchAx, gammaStretch);
            double scale = 1.0 + std::max(std::abs(ref), std::abs(hOut[i]));
            maxerr = std::max(maxerr, std::abs(ref - hOut[i]) / scale);
        }
        std::printf("  self-check (kernel vs CPU extract, %u pairs): max rel err = %.3e\n", nc, maxerr);
    }

    if(a.xcheck)
    {
        if(std::string(a.band) != "in")
            std::printf("  xcheck: skipped (needs --band in)\n");
        else
        {
            // extract(order 12) vs sing(order 8, refine 2) on the first 32 in-support pairs
            launch(12, 0, true, stretchAx, gammaStretch);
            onHost::wait(queue);
            onHost::memcpy(queue, hOutV, dOut);
            onHost::wait(queue);
            std::vector<double> hExt(32);
            for(int i = 0; i < 32; ++i)
                hExt[i] = hOutV[i];
            launch(8, 2, false, stretchAx, gammaStretch);
            onHost::wait(queue);
            onHost::memcpy(queue, hOutV, dOut);
            onHost::wait(queue);
            double maxerr = 0.0;
            for(int i = 0; i < 32; ++i)
            {
                double const ref = hOutV[i];
                double scale = 1.0 + std::max(std::abs(ref), std::abs(hExt[i]));
                maxerr = std::max(maxerr, std::abs(ref - hExt[i]) / scale);
            }
            std::printf("  xcheck (extract o12 vs sing 8,2 on 32 in-support pairs): max rel diff = %.3e\n", maxerr);
        }
    }

    double pairsPerSec = 0.0;
    if(a.bench)
    {
        auto const t0 = std::chrono::steady_clock::now();
        for(int it = 0; it < a.iters; ++it)
            launch(a.order, a.refine, isExtract, stretchAx, gammaStretch);
        onHost::wait(queue);
        auto const t1 = std::chrono::steady_clock::now();
        double const secs = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-9;
        pairsPerSec = static_cast<double>(a.npairs) * a.iters / secs;
        double const nodeRate = pairsPerSec * nodesPerPair;
        std::printf("  bench: %u pairs x %d iters in %.4f s -> %.4e pairs/s (%.4e nodes/s)  [%.3f us/pair]\n",
                    a.npairs, a.iters, secs, pairsPerSec, nodeRate, 1e6 / (pairsPerSec ? pairsPerSec : 1.0));
    }

    if(a.sample)
    {
        std::string base = a.outpath;
        std::ofstream b(base + ".bin", std::ios::binary);
        b.write(reinterpret_cast<char const*>(rel.data()), 3 * static_cast<std::size_t>(a.npairs) * sizeof(double));
        b.write(reinterpret_cast<char const*>(comp.data()), static_cast<std::size_t>(a.npairs) * sizeof(int));
        b.write(reinterpret_cast<char const*>(hOut.data()), static_cast<std::size_t>(a.npairs) * sizeof(double));
        std::ofstream m(base + ".meta.txt");
        m << "kernel " << a.kernel << "\nband " << a.band << "\nnpairs " << a.npairs << "\ngamma " << a.gamma << "\n";
        m << "beta " << beta[0] << " " << beta[1] << " " << beta[2] << "\n";
        std::printf("  sample -> %s.bin (+.meta.txt), %u pairs\n", base.c_str(), a.npairs);
    }

    std::printf("DONE kernel=%s band=%s gamma=%.4g pairsPerSec=%.4e\n", a.kernel, a.band, a.gamma, pairsPerSec);
    return 0;
}
