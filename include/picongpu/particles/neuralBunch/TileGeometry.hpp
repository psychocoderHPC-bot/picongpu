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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
