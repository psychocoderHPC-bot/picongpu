/* Copyright 2015-2024 Alexander Grund, Rene Widera
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

#pragma once

#include "pmacc/random/distributions/Normal.hpp"
#include "pmacc/types.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            namespace detail
            {
                //! Returns a normally distributed floating point with value with mean 0.0 and standard deviation 1.0
                template<typename T_Type, typename T_RNGMethod>
                struct Normal<T_Type, T_RNGMethod, void>
                {
                    using RNGMethod = T_RNGMethod;
                    using StateType = typename RNGMethod::StateType;

                public:
                    using result_type = T_Type;

                    /** Box-Muller normal distribution.
                     *
                     * This reproduces alpaka2's `rand::distribution::gpu::NormalReal` exactly:
                     * the uniform value uses the low-order bits of the engine words
                     * (`b = word0 | (word1 << 32)` for double) and divides by 2^digits.
                     * The engine is called directly (two 32 bit words per uniform).
                     */
                    template<typename T_Worker>
                    DINLINE result_type operator()(T_Worker const&, StateType& state)
                    {
                        auto uniform = [&state]() -> T_Type
                        {
                            if constexpr(std::is_same_v<T_Type, double>)
                            {
                                // Sequence the two engine draws explicitly: the first draw is the low
                                // word and the second the high word (matching alpaka2's NormalReal).
                                std::uint64_t const low = static_cast<std::uint64_t>(state());
                                std::uint64_t const high = static_cast<std::uint64_t>(state());
                                std::uint64_t const b = low | (high << 32u);
                                constexpr std::uint64_t limit = std::uint64_t{1} << 53u;
                                return static_cast<T_Type>(b & (limit - 1u)) / static_cast<T_Type>(limit);
                            }
                            else
                            {
                                std::uint32_t const b = state();
                                constexpr std::uint32_t limit = std::uint32_t{1} << 24u;
                                return static_cast<T_Type>(b & (limit - 1u)) / static_cast<T_Type>(limit);
                            }
                        };

                        constexpr auto sigma = T_Type{1};
                        constexpr auto mu = T_Type{0};

                        T_Type u1, u2;
                        do
                        {
                            u1 = uniform();
                            u2 = uniform();
                        } while(u1 <= std::numeric_limits<T_Type>::epsilon());

                        T_Type const mag = sigma * ::alpaka::math::sqrt(static_cast<T_Type>(-2.) * ::alpaka::math::log(u1));
                        constexpr T_Type twoPi = static_cast<T_Type>(2. * ::alpaka::math::constants::pi);
                        return mag * static_cast<T_Type>(::alpaka::math::sin(twoPi * u2)) + mu;
                    }
                };

            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
