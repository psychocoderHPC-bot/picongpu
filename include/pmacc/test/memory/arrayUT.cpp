/* Copyright 2026-2026 Rene Widera
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

#include <pmacc/boost_workaround.hpp>

#include <pmacc/memory/Array.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

#include <catch2/catch_test_macros.hpp>

namespace
{
    /** Element type with user-provided copy/move construction and assignment.
     *
     * Every operation increments a dedicated counter, so a byte-wise copy can
     * be distinguished from element-wise construction and assignment. The
     * destructor stays trivial because Array does not destroy its elements.
     */
    struct CountingElement
    {
        int value;

        static inline std::size_t copyConstructed = 0;
        static inline std::size_t moveConstructed = 0;
        static inline std::size_t copyAssigned = 0;
        static inline std::size_t moveAssigned = 0;

        explicit CountingElement(int const v) : value(v)
        {
        }

        CountingElement(CountingElement const& other) : value(other.value)
        {
            ++copyConstructed;
        }

        CountingElement(CountingElement&& other) noexcept : value(other.value)
        {
            ++moveConstructed;
        }

        CountingElement& operator=(CountingElement const& other)
        {
            value = other.value;
            ++copyAssigned;
            return *this;
        }

        CountingElement& operator=(CountingElement&& other) noexcept
        {
            value = other.value;
            ++moveAssigned;
            return *this;
        }

        static void resetCounters() noexcept
        {
            copyConstructed = 0;
            moveConstructed = 0;
            copyAssigned = 0;
            moveAssigned = 0;
        }
    };

    static_assert(
        std::is_trivially_destructible_v<CountingElement>,
        "Array requires a trivially destructible element type");

    using TestArray = pmacc::memory::Array<CountingElement, 3>;

    void requireValues(TestArray const& array, int const expected)
    {
        REQUIRE(array[0].value == expected);
        REQUIRE(array[1].value == expected);
        REQUIRE(array[2].value == expected);
    }

    void requireNoElementOperation()
    {
        REQUIRE(CountingElement::copyConstructed == 0);
        REQUIRE(CountingElement::moveConstructed == 0);
        REQUIRE(CountingElement::copyAssigned == 0);
        REQUIRE(CountingElement::moveAssigned == 0);
    }
} // namespace

TEST_CASE("Array copy and move semantics", "[memory][Array]")
{
    SECTION("copy construction uses the element copy constructor")
    {
        CountingElement::resetCounters();
        TestArray source(7);
        requireValues(source, 7);

        CountingElement::resetCounters();
        TestArray copy(source);

        REQUIRE(CountingElement::copyConstructed == 3);
        REQUIRE(CountingElement::moveConstructed == 0);
        REQUIRE(CountingElement::copyAssigned == 0);
        REQUIRE(CountingElement::moveAssigned == 0);
        requireValues(copy, 7);
        REQUIRE(&copy[0] != &source[0]);

        copy[1].value = 42;
        REQUIRE(source[1].value == 7);
        REQUIRE(copy[1].value == 42);
    }

    SECTION("move construction uses the element move constructor")
    {
        CountingElement::resetCounters();
        TestArray source(9);
        requireValues(source, 9);

        CountingElement::resetCounters();
        TestArray moved(std::move(source));

        REQUIRE(CountingElement::copyConstructed == 0);
        REQUIRE(CountingElement::moveConstructed == 3);
        REQUIRE(CountingElement::copyAssigned == 0);
        REQUIRE(CountingElement::moveAssigned == 0);
        requireValues(moved, 9);
        REQUIRE(&moved[0] != &source[0]);
    }

    SECTION("copy assignment uses the element copy assignment")
    {
        CountingElement::resetCounters();
        TestArray lhs(1);
        TestArray rhs(2);

        CountingElement::resetCounters();
        lhs = rhs;

        REQUIRE(CountingElement::copyConstructed == 0);
        REQUIRE(CountingElement::moveConstructed == 0);
        REQUIRE(CountingElement::copyAssigned == 3);
        REQUIRE(CountingElement::moveAssigned == 0);
        requireValues(lhs, 2);
        requireValues(rhs, 2);

        lhs[0].value = 5;
        REQUIRE(rhs[0].value == 2);
    }

    SECTION("move assignment uses the element move assignment")
    {
        CountingElement::resetCounters();
        TestArray lhs(1);
        TestArray rhs(6);

        CountingElement::resetCounters();
        lhs = std::move(rhs);

        REQUIRE(CountingElement::copyConstructed == 0);
        REQUIRE(CountingElement::moveConstructed == 0);
        REQUIRE(CountingElement::copyAssigned == 0);
        REQUIRE(CountingElement::moveAssigned == 3);
        requireValues(lhs, 6);
    }

    SECTION("self assignment is a value-preserving no-op")
    {
        /* Invariant: assigning an Array to itself neither changes any element
         * value nor runs an element-level copy or move operation, so the
         * operation counters stay at zero.
         */
        CountingElement::resetCounters();
        TestArray array(4);
        array[1].value = 8;

        CountingElement::resetCounters();
        array = array;
        REQUIRE(array[0].value == 4);
        REQUIRE(array[1].value == 8);
        REQUIRE(array[2].value == 4);
        requireNoElementOperation();

        CountingElement::resetCounters();
        array = std::move(array);
        REQUIRE(array[0].value == 4);
        REQUIRE(array[1].value == 8);
        REQUIRE(array[2].value == 4);
        requireNoElementOperation();
    }
}
