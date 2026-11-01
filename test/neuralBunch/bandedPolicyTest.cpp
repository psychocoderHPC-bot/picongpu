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

    // Band routing with the model-use options.
    nb::BandedModelUse const nearOff{};
    nb::BandedModelUse const nearOn{true, {}, false};
    nb::BandedModelUse const midOn{false, true, false};
    nb::BandedModelUse const midOff{false, false, false};
    nb::BandedModelUse const correctionOn{false, {}, true};
    using nb::BandRoute;
    check(
        nb::routeBandedBand(nb::Band::near, nearOff, false, false, false, true) == BandRoute::reference,
        "near defaults to reference");
    check(
        nb::routeBandedBand(nb::Band::near, nearOff, false, false, false, false) == BandRoute::error,
        "near without fallback and without model errors");
    check(
        nb::routeBandedBand(nb::Band::near, nearOn, false, false, false, true) == BandRoute::error,
        "near model requested without artifact errors");
    check(
        nb::routeBandedBand(nb::Band::near, nearOn, false, false, true, true) == BandRoute::model,
        "near model uses the primary artifact");
    check(
        nb::routeBandedBand(nb::Band::near, nearOn, true, false, false, true) == BandRoute::model,
        "near model uses a near_model section");
    check(
        nb::routeBandedBand(nb::Band::mid, nearOff, false, false, true, true) == BandRoute::analytic,
        "mid defaults to the analytic point field even with a model");
    check(
        nb::routeBandedBand(nb::Band::mid, nearOff, false, false, false, true) == BandRoute::analytic,
        "mid defaults to the analytic point field without a model");
    check(
        nb::routeBandedBand(nb::Band::mid, midOff, false, false, true, true) == BandRoute::analytic,
        "explicit mid model off uses the point field");
    check(
        nb::routeBandedBand(nb::Band::mid, midOn, false, false, true, true) == BandRoute::model,
        "explicit mid model uses the primary artifact");
    check(
        nb::routeBandedBand(nb::Band::mid, midOn, false, false, false, true) == BandRoute::error,
        "mid model requested without artifact errors");
    check(
        nb::routeBandedBand(nb::Band::mid, correctionOn, false, true, false, true) == BandRoute::model,
        "mid correction is opt-in");
    check(
        nb::routeBandedBand(nb::Band::mid, correctionOn, false, false, false, true) == BandRoute::error,
        "mid correction requested without section errors");
    check(
        nb::routeBandedBand(nb::Band::far, nearOff, false, false, false, false) == BandRoute::analytic,
        "far is analytic point field data");
    check(
        nb::needsUnvalidatedModelWarning(nb::Band::near, nearOn, false, false, true),
        "near model emits the warning");
    check(
        !nb::needsUnvalidatedModelWarning(nb::Band::near, nearOff, true, false, true),
        "near without model does not emit the warning");
    check(
        !nb::needsUnvalidatedModelWarning(nb::Band::mid, nearOff, false, false, true),
        "default mid point field does not emit the warning");
    check(
        nb::needsUnvalidatedModelWarning(nb::Band::mid, midOn, false, false, true),
        "mid primary model emits the warning");
    check(
        nb::needsUnvalidatedModelWarning(nb::Band::mid, correctionOn, false, true, false),
        "mid correction emits the warning");
    check(
        !nb::needsUnvalidatedModelWarning(nb::Band::mid, midOff, false, false, true),
        "explicit mid model off does not emit the warning");

    if(failures != 0)
    {
        std::cerr << failures << " banded policy checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "banded policy checks passed\n";
    return EXIT_SUCCESS;
}
