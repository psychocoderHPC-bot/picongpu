/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Standalone unit test for the neural bunch tile geometry and unit conversion.
 * Build and run without PIConGPU/alpaka:
 *
 *   g++ -std=c++20 -I <repo>/include test/neuralBunch/tileGeometryTest.cpp -o /tmp/tileGeometryTest
 *   /tmp/tileGeometryTest
 */

#include "picongpu/particles/neuralBunch/TileGeometry.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace tg = picongpu::particles::neuralBunch::tileGeometry;

namespace
{
    int failures = 0;

    void check(bool const condition, char const* message)
    {
        if(!condition)
        {
            std::cerr << "FAIL: " << message << '\n';
            ++failures;
        }
    }

    void checkClose(double const actual, double const expected, double const tolerance, char const* message)
    {
        if(!(std::abs(actual - expected) <= tolerance))
        {
            std::cerr << "FAIL: " << message << " expected " << expected << " got " << actual << '\n';
            ++failures;
        }
    }
} // namespace

int main()
{
    // Rank-local domain offset 32, size 20x9x5. Tile origins start at the local
    // offset, not at a global multiple of 8x8x4.
    std::array<int, 3> const offset{32, 0, 0};
    std::array<int, 3> const size{20, 9, 5};
    auto const grid = tg::enumerateTiles(offset, size);

    check(grid.count[0] == 3, "tile count x");
    check(grid.count[1] == 2, "tile count y");
    check(grid.count[2] == 2, "tile count z");
    check(grid.tiles.size() == 12, "total tile count");

    // First tile starts at the local offset.
    check(grid.tiles[0].origin[0] == 32 && grid.tiles[0].origin[1] == 0 && grid.tiles[0].origin[2] == 0, "first tile");
    // x index 1 -> 32 + 8 = 40; last x tile -> 48.
    check(grid.tiles[1].origin[0] == 40, "second x tile");
    check(grid.tiles[2].origin[0] == 48, "third x tile");
    // Flatten order: x fastest, then y, then z. y tile 1 -> 8, z tile 1 -> 4.
    check(grid.tiles[3 * 2 + 3].origin[1] == 8, "y tile origin");
    check(grid.tiles[3 * 2 * 2 - 1].origin[2] == 4, "z tile origin");

    // Clipping is done when storing, not when enumerating.
    check(tg::insideLocalDomain({48, 0, 0}, offset, size), "inside first clipped cell");
    check(tg::insideLocalDomain({51, 8, 4}, offset, size), "inside last clipped cell");
    check(!tg::insideLocalDomain({52, 0, 0}, offset, size), "outside clipped x edge");
    check(!tg::insideLocalDomain({32, 9, 0}, offset, size), "outside clipped y edge");
    check(!tg::insideLocalDomain({32, 0, 5}, offset, size), "outside clipped z edge");
    check(!tg::insideLocalDomain({31, 0, 0}, offset, size), "outside lower edge");

    // MODEL-FORMAT flatten order c + 6*(i + 8*(j + 8*k)).
    check(tg::outputIndex(0, 0, 0, 0) == 0, "flatten origin Ex");
    check(tg::outputIndex(0, 0, 0, 5) == 5, "flatten origin Bz");
    check(tg::outputIndex(1, 0, 0, 0) == 6, "flatten i stride");
    check(tg::outputIndex(0, 1, 0, 0) == 6 * 8, "flatten j stride");
    check(tg::outputIndex(0, 0, 1, 0) == 6 * 64, "flatten k stride");
    check(tg::outputIndex(7, 7, 3, 5) == tg::outputDimension - 1, "flatten last element");

    // Hand-computed values with e = 1.602176634e-19 C,
    // eps0 = 8.8541878128e-12 F/m, c = 299792458 m/s, dx = 1e-6 m:
    //   E_SI = 2 * e/(4 pi eps0 dx^2) = 2.8799290956851346e3 V/m
    //   B_SI = 3 * e/(4 pi eps0 c dx^2) = 1.4409614145555663e-5 T
    double const dx = 1.0e-6;
    checkClose(tg::normalizedEToSI(2.0, dx), 2.8799290956851346e3, 1.0e-9, "E normalized to SI");
    checkClose(tg::normalizedBToSI(3.0, dx), 1.4409614145555663e-5, 1.0e-15, "B normalized to SI");

    // Internal conversion divides by the PIConGPU units; use hand values:
    // unitE = 1.0e5 V/m -> 0.028799290956851346 internal units.
    checkClose(tg::normalizedEToInternal(2.0, dx, 1.0e5), 2.8799290956851346e-2, 1.0e-13, "E to internal");
    checkClose(tg::normalizedBToInternal(3.0, dx, 1.0e-3), 1.4409614145555663e-2, 1.0e-13, "B to internal");

    if(failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "tileGeometryTest: OK\n";
    return EXIT_SUCCESS;
}
