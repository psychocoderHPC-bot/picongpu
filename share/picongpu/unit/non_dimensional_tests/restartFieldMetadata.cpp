/* Copyright 2026 Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <catch2/catch_test_macros.hpp>

#include <picongpu/plugins/openPMD/restart/RestartFieldMetadata.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using namespace picongpu::openPMD::restartMetadata;

    double const deckUnit = 8850761452125.0;

} // namespace

TEST_CASE("restart metadata converts a factor-2 unitSI checkpoint", "[restart metadata]")
{
    double const fileUnitSI = 2.0 * deckUnit;
    double const factor = toInternalFactor("E/x", fileUnitSI, deckUnit);
    REQUIRE(factor == 2.0);
    REQUIRE(1.5 * factor == 3.0);

    // A PIConGPU-written checkpoint has the deck unit and keeps values bit-identical.
    REQUIRE(toInternalFactor("E/x", deckUnit, deckUnit) == 1.0);
}

TEST_CASE("restart metadata rejects invalid unitSI", "[restart metadata]")
{
    REQUIRE_THROWS_AS(
        toInternalFactor("E/x", std::numeric_limits<double>::quiet_NaN(), deckUnit),
        std::runtime_error);
    REQUIRE_THROWS_AS(toInternalFactor("E/x", 0.0, deckUnit), std::runtime_error);
    REQUIRE_THROWS_AS(toInternalFactor("E/x", -1.0, deckUnit), std::runtime_error);

    try
    {
        toInternalFactor("E/x", std::numeric_limits<double>::infinity(), deckUnit);
    }
    catch(std::runtime_error const& error)
    {
        REQUIRE(std::string(error.what()).find("E/x") != std::string::npos);
        REQUIRE(std::string(error.what()).find("unitSI") != std::string::npos);
    }
}

TEST_CASE("restart metadata rejects a mismatched gridSpacing", "[restart metadata]")
{
    std::vector<double> const deckSpacing{1.7320507764816284, 1.7320507764816284, 1.7320507764816284};

    REQUIRE_NOTHROW(checkGridSpacing("E", deckSpacing, deckSpacing));

    std::vector<double> const wrongSpacing{1.7320507764816284, 3.4641015529632568, 1.7320507764816284};
    try
    {
        checkGridSpacing("E", wrongSpacing, deckSpacing);
        FAIL("expected a mismatched gridSpacing to throw");
    }
    catch(std::runtime_error const& error)
    {
        std::string const message = error.what();
        REQUIRE(message.find("'E'") != std::string::npos);
        REQUIRE(message.find("gridSpacing[1]") != std::string::npos);
        REQUIRE(message.find("3.46") != std::string::npos);
        REQUIRE(message.find("1.73") != std::string::npos);
    }
}

TEST_CASE("restart metadata rejects a wrong in-cell position", "[restart metadata]")
{
    std::vector<double> const deckPosition{0.5, 0.0, 0.0};
    REQUIRE_NOTHROW(checkPosition("E/x", deckPosition, deckPosition));
    try
    {
        checkPosition("E/x", std::vector<double>{0.0, 0.5, 0.0}, deckPosition);
        FAIL("expected a wrong position to throw");
    }
    catch(std::runtime_error const& error)
    {
        REQUIRE(std::string(error.what()).find("E/x") != std::string::npos);
        REQUIRE(std::string(error.what()).find("position") != std::string::npos);
    }
}

TEST_CASE("restart metadata rejects inconsistent mesh metadata", "[restart metadata]")
{
    REQUIRE_NOTHROW(
        checkAxisLabels("E", std::vector<std::string>{"z", "y", "x"}, std::vector<std::string>{"z", "y", "x"}));
    REQUIRE_THROWS_AS(
        checkAxisLabels("E", std::vector<std::string>{"x", "y", "z"}, std::vector<std::string>{"z", "y", "x"}),
        std::runtime_error);

    std::vector<double> const dimension{1.0, 1.0, -3.0, -1.0, 0.0, 0.0, 0.0};
    REQUIRE_NOTHROW(checkUnitDimension("E", dimension, dimension));
    REQUIRE_THROWS_AS(
        checkUnitDimension("E", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, dimension),
        std::runtime_error);

    REQUIRE_NOTHROW(checkTimeOffset("E", 0.0));
    REQUIRE_THROWS_AS(checkTimeOffset("E", 1.0), std::runtime_error);

    std::vector<double> const offset{0.0, 0.0, 0.0};
    REQUIRE_NOTHROW(checkGridGlobalOffset("E", offset, offset));
    REQUIRE_THROWS_AS(
        checkGridGlobalOffset("E", std::vector<double>{1.0, 0.0, 0.0}, offset),
        std::runtime_error);
}
