/* Copyright 2024-2024 Rene Widera
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

/* Golden vector test for the ported Philox/AlpakaRand CPU fallback stream.
 *
 * The expected values are the alpaka 2.2.0 `createDefault` -> Philox4x32x10
 * stream as consumed by PMacc's AlpakaRand and its distributions. The values
 * are byte-identical between alpaka 2.2.0 and alpaka3 (see the R5 design).
 */

#include "pmacc/random/distributions/Normal.hpp"
#include "pmacc/random/distributions/Uniform.hpp"
#include "pmacc/random/methods/AlpakaRand.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cstdint>
#include <limits>

namespace
{
    //! Minimal worker stand-in: AlpakaRand and its distributions only need the worker for device/GPU methods.
    struct DummyWorker
    {
    };

    using Engine = ::alpaka::rand::engine::Philox4x32x10;

    void requireWords(
        std::uint64_t seed,
        std::uint64_t subsequence,
        std::uint32_t w0,
        std::uint32_t w1,
        std::uint32_t w2,
        std::uint32_t w3)
    {
        Engine engine{seed, subsequence, 0u};
        REQUIRE(engine() == w0);
        REQUIRE(engine() == w1);
        REQUIRE(engine() == w2);
        REQUIRE(engine() == w3);
    }
} // namespace

TEST_CASE("alpaka Philox engine matches the alpaka2 golden vector", "[rng]")
{
    // (seed, subsequence) -> first four 32 bit words
    requireWords(0x00000000u, 0u, 0x801c7a10u, 0x7853bee3u, 0xceb0e774u, 0x305162aau);
    requireWords(0x00000000u, 1u, 0x364fb449u, 0x02435475u, 0x3d971bdcu, 0x5aef1bbfu);
    requireWords(0x00000000u, 2u, 0x9368b9acu, 0xb50ca022u, 0x26796834u, 0x9baf3e7bu);
    requireWords(0x00000000u, 3u, 0x29b8658du, 0xd0a8f755u, 0x528da98au, 0x876a0977u);

    requireWords(0x00000001u, 0u, 0x9ad4c8ecu, 0x3777f27bu, 0x09e7ed88u, 0x890f9946u);
    requireWords(0x00000001u, 1u, 0xda691da3u, 0xb7164727u, 0x33f9b3ffu, 0x4dd1ebf8u);
    requireWords(0x00000001u, 2u, 0xa2c8aebau, 0x2eff2293u, 0xdcb02a14u, 0xd9e79daeu);
    requireWords(0x00000001u, 3u, 0xb78fc8dbu, 0x37bb73f0u, 0x5d5e9241u, 0x1efb9c82u);

    requireWords(0x0000002Au, 0u, 0x234f11bcu, 0x9eefe31au, 0x6c456183u, 0xb462ea09u);
    requireWords(0x0000002Au, 1u, 0xe5cdaff9u, 0x2446945cu, 0xa0512236u, 0xc47cd2a1u);
    requireWords(0x0000002Au, 2u, 0x8095918fu, 0x9522a1e7u, 0x615d3665u, 0xa48b0e92u);
    requireWords(0x0000002Au, 3u, 0x5340888au, 0x600e4617u, 0xa055eaa5u, 0x402067e6u);

    requireWords(0xFFFFFFFFu, 0u, 0x700ca1ecu, 0xcf2678e0u, 0x47267f99u, 0x5edca267u);
}

TEST_CASE("AlpakaRand get64 packs two 32 bit words", "[rng]")
{
    DummyWorker worker;
    pmacc::random::methods::AlpakaRand<> method;
    pmacc::random::methods::AlpakaRand<>::StateType state;

    method.init(worker, state, 0u, 0u);
    std::uint32_t const w0 = method.get32Bits(worker, state);
    std::uint32_t const w1 = method.get32Bits(worker, state);
    std::uint64_t const expected = (static_cast<std::uint64_t>(w0) << 32) | static_cast<std::uint64_t>(w1);
    REQUIRE(expected == 0x801c7a107853bee3ull);

    method.init(worker, state, 0u, 0u);
    std::uint64_t const packed = method.get64Bits(worker, state);
    CHECK(packed == expected);
}

TEST_CASE("PMacc distributions reproduce the alpaka2 golden values", "[rng]")
{
    DummyWorker worker;
    pmacc::random::methods::AlpakaRand<> method;

    struct Golden
    {
        std::uint32_t w0;
        std::uint32_t w1;
        std::uint32_t ufBits;
        double uniformDouble;
        double normal;
    };

    // clang-format off
    Golden const golden[] = {
        {0x234f11bcu, 0x9eefe31au, 0x3ed88ac3u, 0.70463431097281193, 0.14442362386634666},
        {0xe5cdaff9u, 0x2446945cu, 0x3f205122u, 0.76752964441541505, 0.18820126753160099},
        {0x8095918fu, 0x9522a1e7u, 0x3ec2ba6du, 0.64274684026869577, 0.029271608144279235},
    };
    // clang-format on

    for(std::uint32_t subseq = 0u; subseq < 3u; ++subseq)
    {
        auto const& expected = golden[subseq];
        pmacc::random::methods::AlpakaRand<>::StateType state;
        method.init(worker, state, 42u, subseq);

        CHECK(method.get32Bits(worker, state) == expected.w0);
        CHECK(method.get32Bits(worker, state) == expected.w1);

        float const uniformFloat
            = pmacc::random::distributions::Uniform<
                pmacc::random::distributions::uniform::ExcludeZero<float>,
                pmacc::random::methods::AlpakaRand<>>{}(worker, state);
        CHECK(std::bit_cast<std::uint32_t>(uniformFloat) == expected.ufBits);

        double const uniformDouble
            = pmacc::random::distributions::Uniform<
                pmacc::random::distributions::uniform::ExcludeZero<double>,
                pmacc::random::methods::AlpakaRand<>>{}(worker, state);
        CHECK(std::bit_cast<std::uint64_t>(uniformDouble) == std::bit_cast<std::uint64_t>(expected.uniformDouble));

        double const normal = pmacc::random::distributions::Normal<
            double,
            pmacc::random::methods::AlpakaRand<>>{}(worker, state);
        CHECK(normal == Catch::Approx(expected.normal).epsilon(1.0e-14));
    }
}
