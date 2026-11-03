/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Precomputed single-source near-field table: metadata and device-side lookup (T150).
 *
 * The table stores the TSC shape-integrated per-unit-charge field of one uniformly moving
 * point charge,
 *
 *   K(r) = int_{[-1.5, 1.5]^3} W(s) Kpoint((r - s) cells, u) ds^3,
 *
 * on a symmetric graded 3D displacement grid in cell units (r_i = (x_obs - x_src)_i / cell_i),
 * for one exact velocity u and one cell aspect cells. The component order is [Ex Ey Ez Bx By
 * Bz] and the on-disk nktable/v1 layout is AoS with the fastest spatial index over x:
 * flat index = ((ix * n + iy) * n + iz) * 6 + c. Stored values use the dx-independent
 * per-unit-charge normalization E_n = 4 pi eps0 E / q, B_n = 4 pi eps0 c B / q, while the
 * exact quadrature path in ExactShapeField.hpp returns the dx-normalized value; a lookup
 * that replaces the quadrature therefore scales the result by cells.x ** 2 (see the
 * in-support branch of exactShapeFieldPoint).
 *
 * The table data lives in global device memory. Only the pointer, the fixed-size grid and
 * the scalars are carried in the copyable metadata struct, so the alpaka kernel argument
 * stays bounded, no per-slot allocation happens, and the lookup adds a handful of registers
 * of local state per thread (trilinear interpolation on the graded tensor grid). The table
 * is exact for one velocity only: callers must fall back to the quadrature path wherever
 * the displacement is outside the grid or the velocity/aspect does not match the table.
 */

#include "Features.hpp"

#include <alpaka/alpaka.hpp>

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace picongpu::particles::neuralBunch
{
    //! Per-axis grid-size bound carried by value in the metadata struct (shipped tables use 77).
    inline constexpr int kNearTableMaxGrid = 128;

    //! Metadata plus device pointer of one near-field table (nktable/v1).
    struct NearFieldTable
    {
        float const* data = nullptr; //!> device global memory, [n, n, n, 6] float32, x-fastest
        double grid[kNearTableMaxGrid]{}; //!> sorted per-axis grid in cell units
        int n = 0; //!> number of grid points per axis
        double rmax = 0.0; //!> per-axis |r| coverage in cell units
        Vec3d u{}; //!> exact velocity of the table (dimensionless momentum)
        Vec3d cells{}; //!> exact cell aspect of the table
        double gamma = 1.0; //!> sqrt(1 + |u|**2)

        ALPAKA_FN_HOST_ACC constexpr bool active() const
        {
            return data != nullptr && n >= 2 && n <= kNearTableMaxGrid;
        }
    };

    //! Trilinear interpolation of the table at displacement @p r in cell units.
    //!
    //! Returns false (and leaves @p out untouched) if the table is inactive or @p r lies
    //! outside the grid on any axis. The grid is non-uniform per axis, so the per-axis
    //! coordinate is the position inside the bracketing interval of the sorted 1D grid; the
    //! interpolation is exact at the grid nodes.
    ALPAKA_FN_HOST_ACC inline bool nearTableLookup(
        NearFieldTable const& table,
        Vec3d const& r,
        double out[6])
    {
        if(!table.active())
            return false;
        int const n = table.n;
        double const rAxis[3]{r.x, r.y, r.z};
        int index[3]{};
        double t[3]{};
        for(int a = 0; a < 3; ++a)
        {
            double const v = rAxis[a];
            if(v < table.grid[0] || v > table.grid[n - 1])
                return false;
            // Binary search: largest i with grid[i] <= v, clamped to [0, n - 2].
            int lo = 0;
            int hi = n - 2;
            while(lo < hi)
            {
                int const mid = (lo + hi + 1) >> 1;
                if(table.grid[mid] <= v)
                    lo = mid;
                else
                    hi = mid - 1;
            }
            index[a] = lo;
            double const span = table.grid[lo + 1] - table.grid[lo];
            t[a] = span > 0.0 ? (v - table.grid[lo]) / span : 0.0;
        }
        int const strideX = n * n;
        int const strideY = n;
        int const base = index[0] * strideX + index[1] * strideY + index[2];
        for(int c = 0; c < 6; ++c)
        {
            double sum = 0.0;
            for(int cx = 0; cx <= 1; ++cx)
                for(int cy = 0; cy <= 1; ++cy)
                    for(int cz = 0; cz <= 1; ++cz)
                    {
                        int const off = base + cx * strideX + cy * strideY + cz;
                        double const w
                            = (cx ? t[0] : 1.0 - t[0]) * (cy ? t[1] : 1.0 - t[1])
                              * (cz ? t[2] : 1.0 - t[2]);
                        sum += w * static_cast<double>(table.data[off * 6 + c]);
                    }
            out[c] = sum;
        }
        return true;
    }

    //! True when the table answers (R0, u, cells): the displacement in cell units is inside
    //! the grid on every axis, and the velocity and cell aspect match the table to rounding
    //! tolerance (the table is exact for one velocity only). Anything else must fall back to
    //! the quadrature path.
    ALPAKA_FN_HOST_ACC inline bool nearTableCovers(
        NearFieldTable const& table,
        Vec3d const& R0,
        Vec3d const& u,
        Vec3d const& cells)
    {
        if(!table.active())
            return false;
        double const r[3]{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
        for(int a = 0; a < 3; ++a)
            if(alpaka::math::abs(r[a]) > table.rmax)
                return false;
        auto const close
            = [](double a, double b)
            { return alpaka::math::abs(a - b) <= 1e-12 * (1.0 + alpaka::math::abs(a) + alpaka::math::abs(b)); };
        if(!close(u.x, table.u.x) || !close(u.y, table.u.y) || !close(u.z, table.u.z))
            return false;
        if(!close(cells.x, table.cells.x) || !close(cells.y, table.cells.y) || !close(cells.z, table.cells.z))
            return false;
        return true;
    }

    //! Host utility: upload a float32 table to device memory and fill the metadata.
    //!
    //! @p hostData is an alpaka host buffer holding the AoS x-fastest table [n, n, n, 6] in
    //! float; its lifetime is bound to @p queue like the other alpaka host buffers of this
    //! header. The returned struct is copyable and carries no per-slot state.
    template<typename TQueue>
    NearFieldTable uploadNearFieldTable(
        TQueue& queue,
        auto hostData,
        std::vector<double> const& grid,
        Vec3d const& u,
        Vec3d const& cells,
        double const gamma)
    {
        NearFieldTable table;
        int const n = static_cast<int>(grid.size());
        if(n < 2 || n > kNearTableMaxGrid
           || hostData.getExtents()[0] != static_cast<std::size_t>(n) * n * n * 6u)
            throw std::invalid_argument{"uploadNearFieldTable: inconsistent table size."};
        for(int i = 0; i < n; ++i)
            table.grid[i] = grid[i];
        table.n = n;
        table.rmax = std::abs(grid.back());
        table.u = u;
        table.cells = cells;
        table.gamma = gamma;
        auto deviceData = alpaka::onHost::allocLike(queue.getDevice(), hostData);
        alpaka::onHost::memcpy(queue, deviceData, hostData);
        hostData.keepAlive(queue);
        table.data = deviceData.data();
        return table;
    }
} // namespace picongpu::particles::neuralBunch
