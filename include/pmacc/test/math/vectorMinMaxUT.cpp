/* Copyright 2026 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/math/Vector.hpp>

#include <catch2/catch_test_macros.hpp>

#include <type_traits>

/** @file
 *
 *  This file is testing element-wise vector minimum and maximum.
 */

TEST_CASE("vector min/max with same scalar type", "[vector][minmax]")
{
    using namespace pmacc::math;

#if TEST_DIM == 2
    {
        auto const a = Vector<float, 2u>(1.f, 4.f);
        auto const b = Vector<float, 2u>(3.f, 2.f);

        auto const min = pmacc::math::min(a, b);
        auto const max = pmacc::math::max(a, b);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(min)>, Vector<float, 2u>>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(max)>, Vector<float, 2u>>);
        REQUIRE(min == Vector<float, 2u>(1.f, 2.f));
        REQUIRE(max == Vector<float, 2u>(3.f, 4.f));

        auto const ad = Vector<double, 2u>(1., 4.);
        auto const bd = Vector<double, 2u>(3., 2.);

        auto const minDouble = pmacc::math::min(ad, bd);
        auto const maxDouble = pmacc::math::max(ad, bd);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(minDouble)>, Vector<double, 2u>>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(maxDouble)>, Vector<double, 2u>>);
        REQUIRE(minDouble == Vector<double, 2u>(1., 2.));
        REQUIRE(maxDouble == Vector<double, 2u>(3., 4.));
    }
#else
    static_assert(TEST_DIM == 3);
    {
        auto const a = Vector<float, 3u>(1.f, 4.f, 2.f);
        auto const b = Vector<float, 3u>(3.f, 2.f, 5.f);

        auto const min = pmacc::math::min(a, b);
        auto const max = pmacc::math::max(a, b);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(min)>, Vector<float, 3u>>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(max)>, Vector<float, 3u>>);
        REQUIRE(min == Vector<float, 3u>(1.f, 2.f, 2.f));
        REQUIRE(max == Vector<float, 3u>(3.f, 4.f, 5.f));

        auto const ad = Vector<double, 3u>(1., 4., 2.);
        auto const bd = Vector<double, 3u>(3., 2., 5.);

        auto const minDouble = pmacc::math::min(ad, bd);
        auto const maxDouble = pmacc::math::max(ad, bd);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(minDouble)>, Vector<double, 3u>>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(maxDouble)>, Vector<double, 3u>>);
        REQUIRE(minDouble == Vector<double, 3u>(1., 2., 2.));
        REQUIRE(maxDouble == Vector<double, 3u>(3., 4., 5.));
    }
#endif
}

TEST_CASE("vector min/max with mixed scalar types", "[vector][minmax]")
{
    using namespace pmacc::math;

#if TEST_DIM == 2
    {
        auto const a = Vector<float, 2u>(1.f, 4.f);
        auto const b = Vector<double, 2u>(3., 2.);

        auto const min = pmacc::math::min(a, b);
        auto const max = pmacc::math::max(a, b);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(min)>, Vector<double, 2u>>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(max)>, Vector<double, 2u>>);
        REQUIRE(min == Vector<double, 2u>(1., 2.));
        REQUIRE(max == Vector<double, 2u>(3., 4.));

        auto const minReversed = pmacc::math::min(b, a);
        auto const maxReversed = pmacc::math::max(b, a);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(minReversed)>, Vector<double, 2u>>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(maxReversed)>, Vector<double, 2u>>);
        REQUIRE(minReversed == Vector<double, 2u>(1., 2.));
        REQUIRE(maxReversed == Vector<double, 2u>(3., 4.));
    }
#else
    static_assert(TEST_DIM == 3);
    {
        auto const a = Vector<float, 3u>(1.f, 4.f, 2.f);
        auto const b = Vector<double, 3u>(3., 2., 5.);

        auto const min = pmacc::math::min(a, b);
        auto const max = pmacc::math::max(a, b);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(min)>, Vector<double, 3u>>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(max)>, Vector<double, 3u>>);
        REQUIRE(min == Vector<double, 3u>(1., 2., 2.));
        REQUIRE(max == Vector<double, 3u>(3., 4., 5.));

        auto const minReversed = pmacc::math::min(b, a);
        auto const maxReversed = pmacc::math::max(b, a);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(minReversed)>, Vector<double, 3u>>);
        STATIC_REQUIRE(std::is_same_v<std::decay_t<decltype(maxReversed)>, Vector<double, 3u>>);
        REQUIRE(minReversed == Vector<double, 3u>(1., 2., 2.));
        REQUIRE(maxReversed == Vector<double, 3u>(3., 4., 5.));
    }
#endif
}
