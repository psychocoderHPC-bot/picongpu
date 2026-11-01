/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Host-only checks for the ported band policy, point field and Taylor basis.
 */

#include "picongpu/particles/neuralBunch/BandPolicy.hpp"
#include "picongpu/particles/neuralBunch/PointField.hpp"
#include "picongpu/particles/neuralBunch/TaylorField.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace nb = picongpu::particles::neuralBunch;

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
} // namespace

int main()
{
    using nb::Band;
    nb::BandPolicy const policy{};
    check(policy.classifyDistance(0.0) == Band::near, "distance 0 is near");
    check(policy.classifyDistance(3.999) == Band::near, "distance just below 4 is near");
    check(policy.classifyDistance(4.0) == Band::mid, "distance 4 is mid");
    check(policy.classifyDistance(48.0) == Band::mid, "distance 48 is mid");
    check(policy.classifyDistance(48.001) == Band::far, "distance above 48 is far");

    nb::BandPolicy const custom{{2.0, 10.0}};
    check(custom.classifyDistance(2.5) == Band::mid, "custom thresholds classify mid");
    bool threw = false;
    try
    {
        nb::BandPolicy const invalid{{10.0, 4.0}};
        (void) invalid;
    }
    catch(std::invalid_argument const&)
    {
        threw = true;
    }
    check(threw, "invalid thresholds are rejected");

    nb::Vec3d const cells{1.0, 2.0, 0.5};
    auto const tile = nb::pointFieldTileHost(nb::Vec3d{1.0, 2.0, 1.0}, nb::Vec3d{0.1, 0.0, 0.0}, cells);
    check(tile.size() == nb::kTileSlotCount, "point field has 1536 slots");
    bool finite = true;
    for(auto const value : tile)
        finite = finite && std::isfinite(value);
    check(finite, "point field values are finite");

    // A static charge at the tile centre: Ex/By parity and non-zero magnitude.
    auto const staticTile = nb::pointFieldTileHost(nb::Vec3d{4.0, 4.0, 2.0}, nb::Vec3d{0.0, 0.0, 0.0}, cells);
    check(std::abs(staticTile[0]) > 0.0, "static point field is non-zero");

    auto const order = 2;
    auto const matrix = nb::taylorMatrixHost(order, cells, nb::TaylorOrigin::tileCentre);
    auto const K = nb::taylorBasisSize(order);
    check(matrix.size() == nb::kTileSlotCount * K, "Taylor matrix has 1536*K entries");
    std::vector<double> coefficients(K, 0.0);
    coefficients[0] = 1.0;
    auto const values = nb::taylorEvaluateHost(matrix, order, coefficients);
    check(values.size() == nb::kTileSlotCount, "Taylor evaluation has 1536 outputs");
    bool taylorFinite = true;
    for(auto const value : values)
        taylorFinite = taylorFinite && std::isfinite(value);
    check(taylorFinite, "Taylor values are finite");

    if(failures != 0)
    {
        std::cerr << failures << " banded policy checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "banded policy checks passed\n";
    return EXIT_SUCCESS;
}
