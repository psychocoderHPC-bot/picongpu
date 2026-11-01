/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Small, PIConGPU-independent helpers for the neural bunch initializer.
 *
 * Kept free of alpaka/PMacc headers so the tile enumeration, cell clipping and
 * the normalized-to-internal field conversion can be unit tested with a plain
 * host compiler.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace picongpu::particles::neuralBunch::tileGeometry
{
    inline constexpr int tileNx = 8;
    inline constexpr int tileNy = 8;
    inline constexpr int tileNz = 4;
    inline constexpr int tileCells = tileNx * tileNy * tileNz;
    inline constexpr int fieldComponents = 6;
    inline constexpr int outputDimension = tileCells * fieldComponents;

    struct Tile
    {
        //! Origin of the tile in the rank's total-domain cell coordinates.
        std::array<int, 3> origin{};
    };

    struct TileGrid
    {
        std::array<int, 3> offset{};
        std::array<int, 3> size{};
        std::array<int, 3> count{};
        std::vector<Tile> tiles;
    };

    inline constexpr std::array<int, 3> tileSize{tileNx, tileNy, tileNz};

    /** Enumerate rank-local tiles.
     *
     * Tile origins are ``offset + i * tileSize`` and are deliberately not
     * aligned to a global multiple of the tile size. The last tile on each axis
     * is clipped by @ref insideLocalDomain when storing, the model still
     * evaluates all 1536 outputs.
     */
    inline TileGrid enumerateTiles(std::array<int, 3> const& offset, std::array<int, 3> const& size)
    {
        TileGrid grid;
        grid.offset = offset;
        grid.size = size;
        for(int axis = 0; axis < 3; ++axis)
            grid.count[axis] = size[axis] <= 0 ? 0 : (size[axis] + tileSize[axis] - 1) / tileSize[axis];
        grid.tiles.reserve(
            static_cast<std::size_t>(grid.count[0]) * static_cast<std::size_t>(grid.count[1])
            * static_cast<std::size_t>(grid.count[2]));
        for(int k = 0; k < grid.count[2]; ++k)
            for(int j = 0; j < grid.count[1]; ++j)
                for(int i = 0; i < grid.count[0]; ++i)
                    grid.tiles.push_back(Tile{{offset[0] + i * tileNx, offset[1] + j * tileNy, offset[2] + k * tileNz}});
        return grid;
    }

    /** Half-open ownership test in the rank's total-domain coordinates. */
    inline bool insideLocalDomain(std::array<int, 3> const& cell, std::array<int, 3> const& offset, std::array<int, 3> const& size)
    {
        for(int axis = 0; axis < 3; ++axis)
            if(cell[axis] < offset[axis] || cell[axis] >= offset[axis] + size[axis])
                return false;
        return true;
    }

    /** Flat cell index inside one tile, matching MODEL-FORMAT's flatten order. */
    inline constexpr int tileCellIndex(int i, int j, int k)
    {
        return i + tileNx * (j + tileNy * k);
    }

    /** Flat output index, MODEL-FORMAT ``c + 6*(i + nx*(j + ny*k))``. */
    inline constexpr int outputIndex(int i, int j, int k, int component)
    {
        return component + fieldComponents * tileCellIndex(i, j, k);
    }

    inline constexpr double elementaryCharge = 1.602176634e-19;
    inline constexpr double vacuumPermittivity = 8.8541878128e-12;
    inline constexpr double speedOfLight = 299792458.0;
    inline constexpr double pi = 3.141592653589793238462643383279502884;

    /** MODEL-FORMAT normalized E to SI: E_SI = E_tilde * e / (4 pi eps0 dx^2). */
    inline double normalizedEToSI(double eTilde, double dxMeters)
    {
        return eTilde * elementaryCharge / (4.0 * pi * vacuumPermittivity * dxMeters * dxMeters);
    }

    /** MODEL-FORMAT normalized B to SI. */
    inline double normalizedBToSI(double bTilde, double dxMeters)
    {
        return normalizedEToSI(bTilde, dxMeters) / speedOfLight;
    }

    /** SI field to PIConGPU internal units. */
    inline double siToInternal(double fieldSI, double unitField)
    {
        return fieldSI / unitField;
    }

    /** Combined normalized E/B to internal units (the conversion used by the stage). */
    inline double normalizedEToInternal(double eTilde, double dxMeters, double unitE)
    {
        return siToInternal(normalizedEToSI(eTilde, dxMeters), unitE);
    }

    inline double normalizedBToInternal(double bTilde, double dxMeters, double unitB)
    {
        return siToInternal(normalizedBToSI(bTilde, dxMeters), unitB);
    }
} // namespace picongpu::particles::neuralBunch::tileGeometry

namespace picongpu::particles::neuralBunch::tileGeometry
{
    /** Anisotropic cell sizes (physical, same length unit as the source positions). */
    struct CellSizes
    {
        double dx = 1.0;
        double dy = 1.0;
        double dz = 1.0;

        double aspectY() const
        {
            return dy / dx;
        }

        double aspectZ() const
        {
            return dz / dx;
        }
    };

    /** Model domain limits copied from the artifact metadata. */
    struct DomainBounds
    {
        double gammaMin = 1.0;
        double gammaMax = 100.0;
        double aYMin = 0.1;
        double aYMax = 10.0;
        double aZMin = 0.1;
        double aZMax = 10.0;
        double dMinCells = 4.0;
        double dMaxCells = 512.0;
    };

    enum class PairRoute
    {
        neural,
        referenceGamma,
        referenceAspect,
        referenceNear,
        referenceFar,
        zeroWeight
    };

    inline char const* routeName(PairRoute route)
    {
        switch(route)
        {
        case PairRoute::neural:
            return "neural";
        case PairRoute::referenceGamma:
            return "reference:gamma";
        case PairRoute::referenceAspect:
            return "reference:aspect";
        case PairRoute::referenceNear:
            return "reference:near";
        case PairRoute::referenceFar:
            return "reference:far";
        case PairRoute::zeroWeight:
            return "skip:zero-weight";
        }
        return "unknown";
    }

    inline constexpr std::array<std::array<double, 3>, 6> componentOffsets{{
        {{0.5, 0.0, 0.0}},
        {{0.0, 0.5, 0.0}},
        {{0.0, 0.0, 0.5}},
        {{0.0, 0.5, 0.5}},
        {{0.5, 0.0, 0.5}},
        {{0.5, 0.5, 0.0}},
    }};

    namespace detail
    {
        /** Minimum absolute distance, in cells of one axis, between a coordinate and
         * the allowed output coordinates `idx + offset` for idx in [0, count). */
        inline double minimumAxisDistance(double sourceCells, double offset, int count)
        {
            double const target = sourceCells - offset;
            double index = std::floor(target + 0.5);
            index = std::clamp(index, 0.0, static_cast<double>(count - 1));
            return target - index;
        }
    } // namespace detail

    /** MODEL-FORMAT distance metric with anisotropic cells.
     *
     * `min over the 1536 staggered output points of |x_out - x_source| / dx`,
     * evaluated exactly without enumerating the points: the output grids of one
     * component are Cartesian products, so the minimum Euclidean distance is
     * the root of the sum of the per-axis minima. Returned in units of dx.
     */
    inline double minimumStaggeredDistanceCells(
        std::array<double, 3> const& source,
        std::array<double, 3> const& tileOrigin,
        CellSizes const& cells)
    {
        double const cell[3]{cells.dx, cells.dy, cells.dz};
        int const count[3]{tileNx, tileNy, tileNz};
        double best2 = std::numeric_limits<double>::infinity();
        for(int component = 0; component < 6; ++component)
        {
            double sum2 = 0.0;
            for(int axis = 0; axis < 3; ++axis)
            {
                double const sourceCells = (source[axis] - tileOrigin[axis]) / cell[axis];
                double const diff
                    = detail::minimumAxisDistance(sourceCells, componentOffsets[component][axis], count[axis]);
                sum2 += (diff * cell[axis] / cells.dx) * (diff * cell[axis] / cells.dx);
            }
            best2 = std::min(best2, sum2);
        }
        return std::sqrt(best2);
    }

    /** Minimum distance to the TSC support boxes (half-width 1.5 cells) of the
     * staggered output points, in units of dx. Diagnostic only; the deterministic
     * neural/reference split uses @ref minimumStaggeredDistanceCells, matching
     * MODEL-FORMAT and the training-time `d_min_cells`. */
    inline double minimumSupportDistanceCells(
        std::array<double, 3> const& source,
        std::array<double, 3> const& tileOrigin,
        CellSizes const& cells)
    {
        double const cell[3]{cells.dx, cells.dy, cells.dz};
        int const count[3]{tileNx, tileNy, tileNz};
        double best2 = std::numeric_limits<double>::infinity();
        for(int component = 0; component < 6; ++component)
        {
            double sum2 = 0.0;
            for(int axis = 0; axis < 3; ++axis)
            {
                double const sourceCells = (source[axis] - tileOrigin[axis]) / cell[axis];
                double const diff
                    = std::abs(detail::minimumAxisDistance(sourceCells, componentOffsets[component][axis], count[axis]));
                double const excess = std::max(diff - 1.5, 0.0) * cell[axis] / cells.dx;
                sum2 += excess * excess;
            }
            best2 = std::min(best2, sum2);
        }
        return std::sqrt(best2);
    }

    /** Deterministic routing exactly as MODEL-FORMAT's fallback policy. */
    inline PairRoute classify(
        double weight,
        double gamma,
        double aspectY,
        double aspectZ,
        double distanceCells,
        DomainBounds const& domain)
    {
        if(weight == 0.0)
            return PairRoute::zeroWeight;
        if(gamma < domain.gammaMin || gamma > domain.gammaMax)
            return PairRoute::referenceGamma;
        if(aspectY < domain.aYMin || aspectY > domain.aYMax || aspectZ < domain.aZMin
           || aspectZ > domain.aZMax)
            return PairRoute::referenceAspect;
        if(distanceCells < domain.dMinCells)
            return PairRoute::referenceNear;
        if(distanceCells > domain.dMaxCells)
            return PairRoute::referenceFar;
        return PairRoute::neural;
    }

    inline bool aspectCovered(CellSizes const& cells, DomainBounds const& domain)
    {
        return cells.aspectY() >= domain.aYMin && cells.aspectY() <= domain.aYMax
               && cells.aspectZ() >= domain.aZMin && cells.aspectZ() <= domain.aZMax;
    }
} // namespace picongpu::particles::neuralBunch::tileGeometry
