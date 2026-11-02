/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Device port of the validated TSC shape-integrated quadrature (reference/cpp/neural_reference.hpp).
 *
 * Smooth outputs use a fixed-order tensor Gauss rule over the three TSC sub-panels on every axis. The order is
 * selected from the distance to the support (4 cells -> order 4, ~1e-8). In-support outputs use the same
 * box/pyramid decomposition as the CPU reference: 27 TSC sub-boxes, the containing box split at the singularity
 * into corner boxes, each covered by three pyramids with the singularity as apex. Two fixed-order evaluations give
 * a per-output Richardson error estimate; there is no adaptive host loop.
 *
 * Accumulation is FP64 by default. The FP32 path keeps the point-field physics in FP64 but accumulates the
 * quadrature in float with Kahan compensation.
 */

#include "Features.hpp"
#include "Inference.hpp"
#include "PointField.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace picongpu::particles::neuralBunch
{
    inline constexpr int kExactMaxOrder = 32;
    inline constexpr int kExactOrderCount = 7; // orders 4,8,12,16,20,24,32

    ALPAKA_FN_HOST_ACC inline constexpr int exactOrderIndex(int order)
    {
        switch(order)
        {
        case 4:
            return 0;
        case 8:
            return 1;
        case 12:
            return 2;
        case 16:
            return 3;
        case 20:
            return 4;
        case 24:
            return 5;
        case 32:
            return 6;
        default:
            return -1;
        }
    }

    struct ExactSettings
    {
        int smoothOrderFar = 8; // distance >= 4 cells
        int smoothOrderMid = 12; // 2 <= distance < 4
        int smoothOrderClose = 24; // 1 <= distance < 2
        int smoothOrderVeryClose = 32; // 0 < distance < 1
        int nearAngularOrder = 24;
        int nearAngularOrderFine = 32;
        int nearRadialOrder = 4;
        int nearBoxOrder = 12;
        int nearBoxOrderFine = 16;
        double tol = 1.0e-7;
    };

    /** Distance policy for the exact device path.
     *
     * ``defaultCells`` is the safe configured cutoff. With ``aspectScaling`` the cutoff grows with the cell aspect
     * from ``aspect2Cells`` at aspect <= 2 to ``aspect10Cells`` at aspect >= 10, following the measured
     * point-field thresholds in research/BANDS.md (about 48 cells for aspect <= 2, up to 512 for aspect 10).
     */
    struct ExactDistancePolicy
    {
        double defaultCells = 128.0;
        bool aspectScaling = false;
        double aspect2Cells = 48.0;
        double aspect10Cells = 512.0;
    };

    inline double exactDistanceLimit(double aY, double aZ, ExactDistancePolicy const& policy)
    {
        if(!policy.aspectScaling)
            return policy.defaultCells;
        double const aspect = std::max(aY, aZ);
        if(aspect <= 2.0)
            return policy.aspect2Cells;
        if(aspect >= 10.0)
            return policy.aspect10Cells;
        double const t = std::log(aspect / 2.0) / std::log(10.0 / 2.0);
        return std::exp(std::log(policy.aspect2Cells) + t * (std::log(policy.aspect10Cells) - std::log(policy.aspect2Cells)));
    }

    /** Gauss-Legendre nodes/weights for the fixed orders, usable as a kernel argument. */
    struct GaussTable
    {
        double nodes[kExactOrderCount][kExactMaxOrder]{};
        double weights[kExactOrderCount][kExactMaxOrder]{};

        static void gaussLegendre(int n, double* node, double* weight)
        {
            int const m = (n + 1) / 2;
            for(int i = 0; i < m; ++i)
            {
                double z = std::cos(3.141592653589793238462643383279502884 * (i + 0.75) / (n + 0.5));
                double pp = 0.0;
                for(int iteration = 0; iteration < 100; ++iteration)
                {
                    double p1 = 1.0;
                    double p2 = 0.0;
                    for(int j = 1; j <= n; ++j)
                    {
                        double const p3 = p2;
                        p2 = p1;
                        p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j;
                    }
                    pp = n * (z * p1 - p2) / (z * z - 1.0);
                    double const z1 = z;
                    z = z1 - p1 / pp;
                    if(std::abs(z - z1) <= 1e-15)
                        break;
                }
                node[i] = -z;
                node[n - 1 - i] = z;
                double const w = 2.0 / ((1.0 - z * z) * pp * pp);
                weight[i] = w;
                weight[n - 1 - i] = w;
            }
        }

        static GaussTable create()
        {
            GaussTable table;
            constexpr int orders[kExactOrderCount]{4, 8, 12, 16, 20, 24, 32};
            for(int i = 0; i < kExactOrderCount; ++i)
                gaussLegendre(orders[i], table.nodes[i], table.weights[i]);
            return table;
        }
    };

    ALPAKA_FN_HOST_ACC inline double exactTscShape(double x)
    {
        double const ax = alpaka::math::abs(x);
        if(ax < 0.5)
            return 0.75 - ax * ax;
        if(ax < 1.5)
            return 0.5 * (1.5 - ax) * (1.5 - ax);
        return 0.0;
    }

    template<typename T_Type>
    struct ExactAccumulator6
    {
        // The FP32 consumer path keeps the quadrature sum in a widened compensated accumulator; only the final
        // per-output value is stored as float. This is the documented "mixed with compensated accumulation" path.
        using Storage = std::conditional_t<std::is_same_v<T_Type, float>, double, T_Type>;
        Storage sum[6]{};
        Storage compensation[6]{};

        ALPAKA_FN_HOST_ACC void add(int component, double value)
        {
            if constexpr(std::is_same_v<T_Type, float> || std::is_same_v<T_Type, double>)
            {
                Storage const y = static_cast<Storage>(value) - compensation[component];
                Storage const t = sum[component] + y;
                compensation[component] = (t - sum[component]) - y;
                sum[component] = t;
            }
            else
            {
                sum[component] += static_cast<Storage>(value);
            }
        }

        ALPAKA_FN_HOST_ACC void addPointField(Vec3d const& E, Vec3d const& B, double weight)
        {
            add(0, weight * E.x);
            add(1, weight * E.y);
            add(2, weight * E.z);
            add(3, weight * B.x);
            add(4, weight * B.y);
            add(5, weight * B.z);
        }
    };

    ALPAKA_FN_HOST_ACC inline double exactSupportDistance(Vec3d const& R0, Vec3d const& cells)
    {
        double const ex = alpaka::math::max(alpaka::math::abs(R0.x) - 1.5 * cells.x, 0.0);
        double const ey = alpaka::math::max(alpaka::math::abs(R0.y) - 1.5 * cells.y, 0.0);
        double const ez = alpaka::math::max(alpaka::math::abs(R0.z) - 1.5 * cells.z, 0.0);
        return alpaka::math::sqrt(ex * ex + ey * ey + ez * ez);
    }

    ALPAKA_FN_HOST_ACC inline int exactSmoothOrder(double supportDistance, ExactSettings const& settings)
    {
        if(supportDistance >= 4.0)
            return settings.smoothOrderFar;
        if(supportDistance >= 2.0)
            return settings.smoothOrderMid;
        if(supportDistance >= 1.0)
            return settings.smoothOrderClose;
        return settings.smoothOrderVeryClose;
    }

    template<typename T_Type>
    ALPAKA_FN_HOST_ACC void exactSmoothIntegral(
        Vec3d const& R0,
        Vec3d const& u,
        Vec3d const& cells,
        int order,
        GaussTable const& table,
        ExactAccumulator6<T_Type>& accumulator)
    {
        int const index = exactOrderIndex(order);
        constexpr double panels[3][2]{{-1.5, -0.5}, {-0.5, 0.5}, {0.5, 1.5}};
        for(int px = 0; px < 3; ++px)
        {
            double const halfX = 0.5 * (panels[px][1] - panels[px][0]);
            double const midX = 0.5 * (panels[px][1] + panels[px][0]);
            for(int nx = 0; nx < order; ++nx)
            {
                double const sx = halfX * table.nodes[index][nx] + midX;
                double const wx = halfX * table.weights[index][nx] * exactTscShape(sx);
                for(int py = 0; py < 3; ++py)
                {
                    double const halfY = 0.5 * (panels[py][1] - panels[py][0]);
                    double const midY = 0.5 * (panels[py][1] + panels[py][0]);
                    for(int ny = 0; ny < order; ++ny)
                    {
                        double const sy = halfY * table.nodes[index][ny] + midY;
                        double const wxy = wx * halfY * table.weights[index][ny] * exactTscShape(sy);
                        for(int pz = 0; pz < 3; ++pz)
                        {
                            double const halfZ = 0.5 * (panels[pz][1] - panels[pz][0]);
                            double const midZ = 0.5 * (panels[pz][1] + panels[pz][0]);
                            for(int nz = 0; nz < order; ++nz)
                            {
                                double const sz = halfZ * table.nodes[index][nz] + midZ;
                                double const weight
                                    = wxy * halfZ * table.weights[index][nz] * exactTscShape(sz);
                                if(weight == 0.0)
                                    continue;
                                Vec3d const R{
                                    R0.x - sx * cells.x,
                                    R0.y - sy * cells.y,
                                    R0.z - sz * cells.z};
                                auto const field = pointFieldDx(R, u);
                                accumulator.addPointField(field.E, field.B, weight);
                            }
                        }
                    }
                }
            }
        }
    }

    //! Worst case per axis: one containing panel (2*level+1 pieces) plus two neighbouring panels (level+1 each),
    //! i.e. 4*level+3 = 23 pieces for level 5. 32 leaves room up to level 7 and keeps the per-thread local
    //! footprint small: CUDA reserves the local frame for a whole launch, and PIConGPU leaves only ~350 MiB free.
    inline constexpr int kExactMaxGradedPanels = 32;
    inline constexpr int kExactGradedOrder = 8;

    /** Integration interval bound.
     *
     * Stored in single precision deliberately: three axes times 32 panels of an unaligned double pair enlarged the
     * per-thread local frame to the point where CUDA could not reserve it on a device where PIConGPU had already
     * claimed the free memory. The bounds are only interval endpoints, the nodes, weights and all accumulation stay
     * double precision.
     */
    struct ExactPanel
    {
        float a{};
        float b{};
    };

    ALPAKA_FN_HOST_ACC inline void exactBuildGradedPanels(
        double target,
        int level,
        ExactPanel* panels,
        int& count)
    {
        constexpr double baseA[3]{-1.5, -0.5, 0.5};
        constexpr double baseB[3]{-0.5, 0.5, 1.5};
        count = 0;
        for(int p = 0; p < 3; ++p)
        {
            double const a = baseA[p];
            double const b = baseB[p];
            double const h = b - a;
            if(target < a - h || target > b + h)
            {
                if(count < kExactMaxGradedPanels)
                    panels[count++] = {static_cast<float>(a), static_cast<float>(b)};
                continue;
            }
            double points[2 + 2 * 8];
            int pointCount = 0;
            points[pointCount++] = a;
            points[pointCount++] = b;
            if(a <= target && target <= b)
            {
                for(int k = 0; k <= level; ++k)
                {
                    double const off = h * 0.5 * alpaka::math::pow(2.0, -static_cast<double>(k));
                    points[pointCount++] = alpaka::math::min(b, alpaka::math::max(a, target - off));
                    points[pointCount++] = alpaka::math::min(b, alpaka::math::max(a, target + off));
                }
            }
            else if(target < a)
            {
                for(int k = 0; k <= level; ++k)
                    points[pointCount++] = alpaka::math::min(b, a + h * alpaka::math::pow(2.0, -static_cast<double>(k)));
            }
            else
            {
                for(int k = 0; k <= level; ++k)
                    points[pointCount++] = alpaka::math::max(a, b - h * alpaka::math::pow(2.0, -static_cast<double>(k)));
            }
            // insertion sort
            for(int i = 1; i < pointCount; ++i)
            {
                double value = points[i];
                int j = i - 1;
                while(j >= 0 && points[j] > value)
                {
                    points[j + 1] = points[j];
                    --j;
                }
                points[j + 1] = value;
            }
            for(int i = 0; i + 1 < pointCount; ++i)
                if(points[i + 1] > points[i] && count < kExactMaxGradedPanels)
                    panels[count++] = {static_cast<float>(points[i]), static_cast<float>(points[i + 1])};
        }
    }

    template<typename T_Type>
    ALPAKA_FN_HOST_ACC void exactGradedSmoothIntegral(
        Vec3d const& R0,
        Vec3d const& u,
        Vec3d const& cells,
        int level,
        int order,
        GaussTable const& table,
        ExactAccumulator6<T_Type>& accumulator)
    {
        int const index = exactOrderIndex(order);
        double const target[3]{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
        ExactPanel panels[3][kExactMaxGradedPanels]{};
        int panelCount[3]{};
        for(int axis = 0; axis < 3; ++axis)
            exactBuildGradedPanels(target[axis], level, panels[axis], panelCount[axis]);

        // The per-axis nodes and weights are generated inline instead of being expanded into three flat arrays.
        // The expanded form needed two 3.7 KiB per-thread local arrays, and CUDA reserves local memory for the
        // whole launch; PIConGPU keeps only a few hundred MiB free, so that variant failed to launch.
        // The arithmetic and the summation order are unchanged.
        double const* const gaussNodes = table.nodes[index];
        double const* const gaussWeights = table.weights[index];
        for(int px = 0; px < panelCount[0]; ++px)
        {
            double const aX = static_cast<double>(panels[0][px].a);
            double const bX = static_cast<double>(panels[0][px].b);
            double const halfX = 0.5 * (bX - aX);
            double const midX = 0.5 * (bX + aX);
            for(int ix = 0; ix < order; ++ix)
            {
                double const sx = halfX * gaussNodes[ix] + midX;
                double const wx = halfX * gaussWeights[ix] * exactTscShape(sx);
                if(wx == 0.0)
                    continue;
                for(int py = 0; py < panelCount[1]; ++py)
                {
                    double const aY = static_cast<double>(panels[1][py].a);
                    double const bY = static_cast<double>(panels[1][py].b);
                    double const halfY = 0.5 * (bY - aY);
                    double const midY = 0.5 * (bY + aY);
                    for(int iy = 0; iy < order; ++iy)
                    {
                        double const sy = halfY * gaussNodes[iy] + midY;
                        double const wxy = wx * (halfY * gaussWeights[iy] * exactTscShape(sy));
                        if(wxy == 0.0)
                            continue;
                        for(int pz = 0; pz < panelCount[2]; ++pz)
                        {
                            double const aZ = static_cast<double>(panels[2][pz].a);
                            double const bZ = static_cast<double>(panels[2][pz].b);
                            double const halfZ = 0.5 * (bZ - aZ);
                            double const midZ = 0.5 * (bZ + aZ);
                            for(int iz = 0; iz < order; ++iz)
                            {
                                double const sz = halfZ * gaussNodes[iz] + midZ;
                                double const weight
                                    = wxy * (halfZ * gaussWeights[iz] * exactTscShape(sz));
                                if(weight == 0.0)
                                    continue;
                                Vec3d const R{
                                    R0.x - sx * cells.x,
                                    R0.y - sy * cells.y,
                                    R0.z - sz * cells.z};
                                auto const field = pointFieldDx(R, u);
                                accumulator.addPointField(field.E, field.B, weight);
                            }
                        }
                    }
                }
            }
        }
    }

    template<typename T_Type>
    ALPAKA_FN_HOST_ACC void exactBoxIntegral(
        Vec3d const& R0,
        Vec3d const& u,
        Vec3d const& cells,
        int nang,
        int nr,
        int ng,
        GaussTable const& table,
        ExactAccumulator6<T_Type>& accumulator)
    {
        int const ia = exactOrderIndex(nang);
        int const ir = exactOrderIndex(nr);
        int const ig = exactOrderIndex(ng);
        double const t[3]{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
        double const boxLo[3]{-1.5, -1.5, -1.5};

        for(int ix = 0; ix < 3; ++ix)
            for(int iy = 0; iy < 3; ++iy)
                for(int iz = 0; iz < 3; ++iz)
                {
                    double const lo[3]{boxLo[0] + ix, boxLo[1] + iy, boxLo[2] + iz};
                    double const hi[3]{lo[0] + 1.0, lo[1] + 1.0, lo[2] + 1.0};
                    bool contains = true;
                    for(int a = 0; a < 3; ++a)
                        contains = contains && lo[a] <= t[a] && t[a] <= hi[a];

                    if(!contains)
                    {
                        for(int a0 = 0; a0 < ng; ++a0)
                        {
                            double const s0 = 0.5 * (lo[0] + hi[0]) + 0.5 * (hi[0] - lo[0]) * table.nodes[ig][a0];
                            double const w0 = 0.5 * (hi[0] - lo[0]) * table.weights[ig][a0];
                            for(int a1 = 0; a1 < ng; ++a1)
                            {
                                double const s1
                                    = 0.5 * (lo[1] + hi[1]) + 0.5 * (hi[1] - lo[1]) * table.nodes[ig][a1];
                                double const w1 = 0.5 * (hi[1] - lo[1]) * table.weights[ig][a1];
                                for(int a2 = 0; a2 < ng; ++a2)
                                {
                                    double const s2
                                        = 0.5 * (lo[2] + hi[2]) + 0.5 * (hi[2] - lo[2]) * table.nodes[ig][a2];
                                    double const w2 = 0.5 * (hi[2] - lo[2]) * table.weights[ig][a2];
                                    double const W = exactTscShape(s0) * exactTscShape(s1) * exactTscShape(s2);
                                    if(W == 0.0)
                                        continue;
                                    Vec3d const R{
                                        R0.x - s0 * cells.x,
                                        R0.y - s1 * cells.y,
                                        R0.z - s2 * cells.z};
                                    auto const field = pointFieldDx(R, u);
                                    accumulator.addPointField(field.E, field.B, w0 * w1 * w2 * W);
                                }
                            }
                        }
                        continue;
                    }

                    double segLo[3][2]{};
                    double segHi[3][2]{};
                    double segSign[3][2]{};
                    int segCount[3]{};
                    for(int a = 0; a < 3; ++a)
                    {
                        if(lo[a] < t[a])
                        {
                            segLo[a][segCount[a]] = lo[a];
                            segHi[a][segCount[a]] = t[a];
                            segSign[a][segCount[a]] = -1.0;
                            ++segCount[a];
                        }
                        if(t[a] < hi[a])
                        {
                            segLo[a][segCount[a]] = t[a];
                            segHi[a][segCount[a]] = hi[a];
                            segSign[a][segCount[a]] = 1.0;
                            ++segCount[a];
                        }
                    }
                    for(int cx = 0; cx < segCount[0]; ++cx)
                        for(int cy = 0; cy < segCount[1]; ++cy)
                            for(int cz = 0; cz < segCount[2]; ++cz)
                            {
                                double const L[3]{
                                    segHi[0][cx] - segLo[0][cx],
                                    segHi[1][cy] - segLo[1][cy],
                                    segHi[2][cz] - segLo[2][cz]};
                                if(L[0] <= 0.0 || L[1] <= 0.0 || L[2] <= 0.0)
                                    continue;
                                double const sgn[3]{segSign[0][cx], segSign[1][cy], segSign[2][cz]};
                                double const V = L[0] * L[1] * L[2];
                                for(int face = 0; face < 3; ++face)
                                {
                                    int const o0 = (face + 1) % 3;
                                    int const o1 = (face + 2) % 3;
                                    for(int iu = 0; iu < nang; ++iu)
                                    {
                                        double const uu = 0.5 * (table.nodes[ia][iu] + 1.0);
                                        double const wu = 0.5 * table.weights[ia][iu];
                                        for(int iv = 0; iv < nang; ++iv)
                                        {
                                            double const vv = 0.5 * (table.nodes[ia][iv] + 1.0);
                                            double const wv = 0.5 * table.weights[ia][iv];
                                            double q[3]{0.0, 0.0, 0.0};
                                            q[face] = sgn[face] * L[face];
                                            q[o0] = sgn[o0] * uu * L[o0];
                                            q[o1] = sgn[o1] * vv * L[o1];
                                            for(int irr = 0; irr < nr; ++irr)
                                            {
                                                double const r = 0.5 * (table.nodes[ir][irr] + 1.0);
                                                double const weight
                                                    = 0.5 * table.weights[ir][irr] * wu * wv * V * r * r;
                                                Vec3d const s{
                                                    t[0] + r * q[0],
                                                    t[1] + r * q[1],
                                                    t[2] + r * q[2]};
                                                double const W
                                                    = exactTscShape(s.x) * exactTscShape(s.y) * exactTscShape(s.z);
                                                if(W == 0.0)
                                                    continue;
                                                Vec3d const R{
                                                    -r * q[0] * cells.x,
                                                    -r * q[1] * cells.y,
                                                    -r * q[2] * cells.z};
                                                auto const field = pointFieldDx(R, u);
                                                accumulator.addPointField(field.E, field.B, weight * W);
                                            }
                                        }
                                    }
                                }
                            }
                }
    }

    ALPAKA_FN_HOST_ACC inline double exactRichardsonError(
        double const* a,
        double const* b)
    {
        double maxDiff = 0.0;
        double maxVal = 0.0;
        for(int i = 0; i < 6; ++i)
        {
            maxDiff = alpaka::math::max(maxDiff, alpaka::math::abs(a[i] - b[i]));
            maxVal = alpaka::math::max(maxVal, alpaka::math::abs(a[i]));
            maxVal = alpaka::math::max(maxVal, alpaka::math::abs(b[i]));
        }
        return maxDiff / (1.0 + maxVal);
    }

    struct ExactPointResult
    {
        Vec3d E{};
        Vec3d B{};
        double error = 0.0;
        bool inSupport = false;
        bool converged = true;
    };

    template<typename T_Type>
    ALPAKA_FN_HOST_ACC ExactPointResult exactShapeFieldPoint(
        Vec3d const& R0,
        Vec3d const& u,
        Vec3d const& cells,
        ExactSettings const& settings,
        GaussTable const& table)
    {
        ExactPointResult result;
        double const distance = exactSupportDistance(R0, cells);
        if(distance == 0.0)
        {
            result.inSupport = true;
            ExactAccumulator6<T_Type> coarse;
            ExactAccumulator6<T_Type> fine;
            exactBoxIntegral(
                R0,
                u,
                cells,
                settings.nearAngularOrder,
                settings.nearRadialOrder,
                settings.nearBoxOrder,
                table,
                coarse);
            exactBoxIntegral(
                R0,
                u,
                cells,
                settings.nearAngularOrderFine,
                settings.nearRadialOrder,
                settings.nearBoxOrderFine,
                table,
                fine);
            double a[6], b[6];
            for(int i = 0; i < 6; ++i)
            {
                a[i] = static_cast<double>(coarse.sum[i]);
                b[i] = static_cast<double>(fine.sum[i]);
            }
            result.error = exactRichardsonError(a, b);
            result.converged = result.error <= settings.tol;
            result.E = {static_cast<double>(fine.sum[0]), static_cast<double>(fine.sum[1]), static_cast<double>(fine.sum[2])};
            result.B = {static_cast<double>(fine.sum[3]), static_cast<double>(fine.sum[4]), static_cast<double>(fine.sum[5])};
        }
        else
        {
            ExactAccumulator6<T_Type> accumulator;
            int const level = distance >= 4.0 ? 0 : (distance >= 2.0 ? 2 : 5);
            exactGradedSmoothIntegral(R0, u, cells, level, kExactGradedOrder, table, accumulator);
            result.E = {static_cast<double>(accumulator.sum[0]), static_cast<double>(accumulator.sum[1]), static_cast<double>(accumulator.sum[2])};
            result.B = {static_cast<double>(accumulator.sum[3]), static_cast<double>(accumulator.sum[4]), static_cast<double>(accumulator.sum[5])};
        }
        return result;
    }

    struct ExactSample
    {
        Vec3d R0{};
        Vec3d u{};
        Vec3d cells{1.0, 1.0, 1.0};
    };

    struct ExactSampleResult
    {
        std::vector<double> E; // [N,3]
        std::vector<double> B; // [N,3]
        std::vector<double> error; // [N]
        std::vector<unsigned char> inSupport; // [N]
    };

    namespace detail
    {
        template<typename T_Type>
        struct ExactSampleKernel
        {
            ExactSettings settings;
            GaussTable table;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto sampleData, auto Eout, auto Bout, auto errors, auto inSupport)
                const
            {
                auto const count = static_cast<std::size_t>(sampleData.getExtents()[0]);
                for(auto index :
                    alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{count}}))
                {
                    auto const i = static_cast<std::size_t>(index[0]);
                    auto value = [&](std::size_t column)
                    { return sampleData[alpaka::Vec<std::size_t, 2>{i, column}]; };
                    auto const result = exactShapeFieldPoint<T_Type>(
                        Vec3d{value(0u), value(1u), value(2u)},
                        Vec3d{value(3u), value(4u), value(5u)},
                        Vec3d{value(6u), value(7u), value(8u)},
                        settings,
                        table);
                    Eout[alpaka::Vec<std::size_t, 2>{i, 0u}] = result.E.x;
                    Eout[alpaka::Vec<std::size_t, 2>{i, 1u}] = result.E.y;
                    Eout[alpaka::Vec<std::size_t, 2>{i, 2u}] = result.E.z;
                    Bout[alpaka::Vec<std::size_t, 2>{i, 0u}] = result.B.x;
                    Bout[alpaka::Vec<std::size_t, 2>{i, 1u}] = result.B.y;
                    Bout[alpaka::Vec<std::size_t, 2>{i, 2u}] = result.B.z;
                    errors[alpaka::Vec<std::size_t, 1>{i}] = result.error;
                    inSupport[alpaka::Vec<std::size_t, 1>{i}] = result.inSupport ? 1u : 0u;
                }
            }
        };
    } // namespace detail

    template<typename T_Type = double, typename TQueue, typename TExec>
    ExactSampleResult exactShapeFieldSamplesBatched(
        TQueue& queue,
        TExec exec,
        std::span<ExactSample const> samples,
        ExactSettings const& settings = {})
    {
        ExactSampleResult result;
        result.E.assign(samples.size() * 3u, 0.0);
        result.B.assign(samples.size() * 3u, 0.0);
        result.error.assign(samples.size(), 0.0);
        result.inSupport.assign(samples.size(), 0u);
        if(samples.empty())
            return result;

        auto hostSamples = alpaka::onHost::allocHost<double>(alpaka::Vec{samples.size(), std::size_t{9}});
        for(std::size_t i = 0u; i < samples.size(); ++i)
        {
            double const values[9]{
                samples[i].R0.x,
                samples[i].R0.y,
                samples[i].R0.z,
                samples[i].u.x,
                samples[i].u.y,
                samples[i].u.z,
                samples[i].cells.x,
                samples[i].cells.y,
                samples[i].cells.z};
            for(std::size_t c = 0u; c < 9u; ++c)
                hostSamples[alpaka::Vec<std::size_t, 2>{i, c}] = values[c];
        }
        auto deviceSamples = alpaka::onHost::allocLike(queue.getDevice(), hostSamples);
        alpaka::onHost::memcpy(queue, deviceSamples, hostSamples);
        hostSamples.keepAlive(queue);
        auto deviceE = alpaka::onHost::alloc<double>(queue.getDevice(), alpaka::Vec{samples.size(), std::size_t{3}});
        auto deviceB = alpaka::onHost::alloc<double>(queue.getDevice(), alpaka::Vec{samples.size(), std::size_t{3}});
        auto deviceError = alpaka::onHost::alloc<double>(queue.getDevice(), alpaka::Vec{samples.size()});
        auto deviceInSupport = alpaka::onHost::alloc<unsigned char>(queue.getDevice(), alpaka::Vec{samples.size()});

        using Vec = alpaka::Vec<std::uint32_t, 1>;
        auto const threads = Vec{32u};
        auto const frames = Vec{static_cast<std::uint32_t>((samples.size() + 31u) / 32u)};
        queue.enqueue(
            alpaka::onHost::FrameSpec{frames, threads, exec},
            alpaka::KernelBundle{
                detail::ExactSampleKernel<T_Type>{settings, GaussTable::create()},
                deviceSamples,
                deviceE,
                deviceB,
                deviceError,
                deviceInSupport});
        auto hostE = alpaka::onHost::allocHost<double>(alpaka::Vec{samples.size(), std::size_t{3}});
        auto hostB = alpaka::onHost::allocHost<double>(alpaka::Vec{samples.size(), std::size_t{3}});
        auto hostError = alpaka::onHost::allocHost<double>(alpaka::Vec{samples.size()});
        auto hostInSupport = alpaka::onHost::allocHost<unsigned char>(alpaka::Vec{samples.size()});
        alpaka::onHost::memcpy(queue, hostE, deviceE);
        alpaka::onHost::memcpy(queue, hostB, deviceB);
        alpaka::onHost::memcpy(queue, hostError, deviceError);
        alpaka::onHost::memcpy(queue, hostInSupport, deviceInSupport);
        alpaka::onHost::wait(queue);
        for(std::size_t i = 0u; i < samples.size(); ++i)
        {
            for(std::size_t c = 0u; c < 3u; ++c)
            {
                result.E[i * 3u + c] = hostE[alpaka::Vec<std::size_t, 2>{i, c}];
                result.B[i * 3u + c] = hostB[alpaka::Vec<std::size_t, 2>{i, c}];
            }
            result.error[i] = hostError[alpaka::Vec<std::size_t, 1>{i}];
            result.inSupport[i] = hostInSupport[alpaka::Vec<std::size_t, 1>{i}];
        }
        return result;
    }

    struct ExactBatchedResult
    {
        std::vector<float> values; // [S,1536]
        std::vector<float> errors; // [S,1536]
    };

    namespace detail
    {
        template<typename T_Type>
        struct ExactTileKernel
        {
            ExactSettings settings;
            GaussTable table;
            Vec3d cells;
            std::size_t sourceOffset = 0u;
            std::size_t batchCount = 0u;

            ALPAKA_FN_ACC void operator()(auto const& acc, auto sourceData, auto values, auto errors) const
            {
                auto const batches
                    = batchCount != 0u ? batchCount : static_cast<std::size_t>(sourceData.getExtents()[0]);
                auto const total = batches * kTileSlotCount;
                for(auto index :
                    alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{total}}))
                {
                    auto const flat = static_cast<std::size_t>(index[0]);
                    auto const batch = flat / kTileSlotCount;
                    auto const slot = flat % kTileSlotCount;
                    auto const row = sourceOffset + batch;
                    auto const x = sourceData[alpaka::Vec<std::size_t, 2>{row, 0u}];
                    auto const y = sourceData[alpaka::Vec<std::size_t, 2>{row, 1u}];
                    auto const z = sourceData[alpaka::Vec<std::size_t, 2>{row, 2u}];
                    auto const ux = sourceData[alpaka::Vec<std::size_t, 2>{row, 3u}];
                    auto const uy = sourceData[alpaka::Vec<std::size_t, 2>{row, 4u}];
                    auto const uz = sourceData[alpaka::Vec<std::size_t, 2>{row, 5u}];
                    auto const position = staggeredPosition(slot, cells);
                    Vec3d const R0{position.x - x, position.y - y, position.z - z};
                    auto const result = exactShapeFieldPoint<T_Type>(R0, Vec3d{ux, uy, uz}, cells, settings, table);
                    auto const component = slot % 6u;
                    double const value = component < 3u
                                             ? (component == 0u ? result.E.x : (component == 1u ? result.E.y : result.E.z))
                                             : (component == 3u ? result.B.x : (component == 4u ? result.B.y : result.B.z));
                    values[alpaka::Vec<std::size_t, 2>{batch, slot}] = static_cast<float>(value);
                    errors[alpaka::Vec<std::size_t, 2>{batch, slot}] = static_cast<float>(result.error);
                }
            }
        };
    } // namespace detail

    /** Batched device quadrature for a set of sources (r,u per source) at all 1536 outputs. */
    template<typename T_Type = double, typename TQueue, typename TExec>
    ExactBatchedResult exactShapeFieldTileBatched(
        TQueue& queue,
        TExec exec,
        std::span<Source const> sources,
        Vec3d const& cells,
        ExactSettings const& settings = {},
        std::size_t maxSources = 4096u)
    {
        if(sources.size() > maxSources)
            throw std::invalid_argument{"exactShapeFieldTileBatched: source bound exceeded."};
        ExactBatchedResult result;
        result.values.assign(sources.size() * kTileSlotCount, 0.0f);
        result.errors.assign(sources.size() * kTileSlotCount, 0.0f);
        if(sources.empty())
            return result;

        auto hostSources = alpaka::onHost::allocHost<double>(alpaka::Vec{sources.size(), std::size_t{6}});
        for(std::size_t i = 0u; i < sources.size(); ++i)
            for(std::size_t c = 0u; c < 3u; ++c)
            {
                hostSources[alpaka::Vec<std::size_t, 2>{i, c}] = sources[i].r[c];
                hostSources[alpaka::Vec<std::size_t, 2>{i, c + 3u}] = sources[i].u[c];
            }
        auto deviceSources = alpaka::onHost::allocLike(queue.getDevice(), hostSources);
        alpaka::onHost::memcpy(queue, deviceSources, hostSources);
        hostSources.keepAlive(queue);

        auto deviceValues
            = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{sources.size(), kTileSlotCount});
        auto deviceErrors
            = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{sources.size(), kTileSlotCount});

        using Vec = alpaka::Vec<std::uint32_t, 1>;
        auto const total = static_cast<std::uint32_t>(sources.size() * kTileSlotCount);
        auto const threads = Vec{128u};
        auto const frames = Vec{(total + 127u) / 128u};
        queue.enqueue(
            alpaka::onHost::FrameSpec{frames, threads, exec},
            alpaka::KernelBundle{
                detail::ExactTileKernel<T_Type>{settings, GaussTable::create(), cells},
                deviceSources,
                deviceValues,
                deviceErrors});

        auto hostValues = alpaka::onHost::allocHost<float>(alpaka::Vec{sources.size(), kTileSlotCount});
        auto hostErrors = alpaka::onHost::allocHost<float>(alpaka::Vec{sources.size(), kTileSlotCount});
        alpaka::onHost::memcpy(queue, hostValues, deviceValues);
        alpaka::onHost::memcpy(queue, hostErrors, deviceErrors);
        alpaka::onHost::wait(queue);
        for(std::size_t i = 0u; i < sources.size(); ++i)
            for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
            {
                result.values[i * kTileSlotCount + slot]
                    = hostValues[alpaka::Vec<std::size_t, 2>{i, slot}];
                result.errors[i * kTileSlotCount + slot]
                    = hostErrors[alpaka::Vec<std::size_t, 2>{i, slot}];
            }
        return result;
    }

    namespace detail
    {
        struct ExactZeroKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto accumulator) const
            {
                for(auto index :
                    alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{kTileSlotCount}}))
                    accumulator[alpaka::Vec<std::size_t, 1>{static_cast<std::size_t>(index[0])}] = 0.0;
            }
        };

        /** Deterministic per-slot accumulation: one thread per output, sources in fixed order. */
        struct ExactAccumulateKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto values, auto weights, auto accumulator) const
            {
                auto const sources = static_cast<std::size_t>(values.getExtents()[0]);
                for(auto index :
                    alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{kTileSlotCount}}))
                {
                    auto const slot = static_cast<std::size_t>(index[0]);
                    double sum = 0.0;
                    for(std::size_t b = 0u; b < sources; ++b)
                        sum += static_cast<double>(weights[alpaka::Vec<std::size_t, 1>{b}])
                               * static_cast<double>(values[alpaka::Vec<std::size_t, 2>{b, slot}]);
                    accumulator[alpaka::Vec<std::size_t, 1>{slot}] += sum;
                }
            }
        };
    } // namespace detail

    template<typename TQueue, typename TExec, typename TAccumulator>
    void enqueueExactZero(TQueue& queue, TExec exec, TAccumulator& accumulator)
    {
        using Vec = alpaka::Vec<std::uint32_t, 1>;
        constexpr std::uint32_t threads = 64u;
        constexpr auto frames = static_cast<std::uint32_t>((kTileSlotCount + threads - 1u) / threads);
        queue.enqueue(
            alpaka::onHost::FrameSpec{Vec{frames}, Vec{threads}, exec},
            alpaka::KernelBundle{detail::ExactZeroKernel{}, accumulator});
    }

    //! Upper bound on sources per device batch: 512 x 1536 float values = 3 MiB of scratch per buffer.
    inline constexpr std::size_t kExactBatchSources = 512u;

    /** Enqueue exact device quadrature for a batch and accumulate weighted results into a device buffer.
     *
     * @p deviceSources is [S,6] with (r, u), @p deviceWeights is [S] with q/e = -weight, @p accumulator is a
     * 1-D FP64 buffer of 1536 outputs. Sources are processed in slices of @p kExactBatchSources so scratch memory
     * is bounded; every slice accumulates into the same device buffer and nothing is copied to the host. The
     * caller syncs once at the end.
     *
     * @p maxThreads caps the resident thread count. The in-support quadrature needs about 21 KiB of local stack
     * per thread, and the stack is allocated from device memory at launch, so a large grid can exhaust a device
     * whose free memory is already taken by the particle heap. The work distribution is a grid stride, so the
     * cap only changes occupancy, never the result.
     */
    template<typename T_Type = double, typename TQueue, typename TExec, typename TSources, typename TWeights, typename TAccumulator>
    void enqueueExactShapeFieldTileAccumulate(
        TQueue& queue,
        TExec exec,
        TSources const& deviceSources,
        TWeights const& deviceWeights,
        std::size_t sourceCount,
        Vec3d const& cells,
        ExactSettings const& settings,
        TAccumulator& accumulator,
        std::size_t maxThreads = std::numeric_limits<std::size_t>::max())
    {
        if(sourceCount == 0u)
            return;
        using Vec = alpaka::Vec<std::uint32_t, 1>;
        constexpr std::uint32_t threads = 64u;
        constexpr auto frames = static_cast<std::uint32_t>((kTileSlotCount + threads - 1u) / threads);
        constexpr std::uint32_t blockSize = 128u;
        for(std::size_t begin = 0u; begin < sourceCount; begin += kExactBatchSources)
        {
            std::size_t const end = std::min(begin + kExactBatchSources, sourceCount);
            auto const count = end - begin;
            auto deviceValues
                = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{count, kTileSlotCount});
            auto deviceErrors
                = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{count, kTileSlotCount});
            auto const total = static_cast<std::uint64_t>(count) * kTileSlotCount;
            auto const wantedBlocks = (total + blockSize - 1u) / blockSize;
            auto const maxBlocks = std::max<std::uint64_t>(1u, maxThreads / blockSize);
            auto const blocks = static_cast<std::uint32_t>(std::min(wantedBlocks, maxBlocks));
            queue.enqueue(
                alpaka::onHost::FrameSpec{Vec{blocks}, Vec{blockSize}, exec},
                alpaka::KernelBundle{
                    detail::ExactTileKernel<T_Type>{settings, GaussTable::create(), cells, begin, count},
                    deviceSources,
                    deviceValues,
                    deviceErrors});
            queue.enqueue(
                alpaka::onHost::FrameSpec{Vec{frames}, Vec{threads}, exec},
                alpaka::KernelBundle{detail::ExactAccumulateKernel{}, deviceValues, deviceWeights, accumulator});
            deviceValues.keepAlive(queue);
            deviceErrors.keepAlive(queue);
        }
    }

    template<typename T_Type = double, typename TQueue, typename TExec>
    ExactBatchedResult exactShapeFieldTile(
        TQueue& queue,
        TExec exec,
        Source const& source,
        Vec3d const& cells,
        ExactSettings const& settings = {})
    {
        std::span<Source const> one{&source, 1u};
        return exactShapeFieldTileBatched<T_Type>(queue, exec, one, cells, settings);
    }
} // namespace picongpu::particles::neuralBunch
