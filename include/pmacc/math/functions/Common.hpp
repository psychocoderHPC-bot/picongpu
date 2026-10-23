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


#pragma once

#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/dimensions/Definition.hpp"

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace pmacc::math
{
    /* alpaka3 dispatches the math implementation internally via alpaka::thisApi(), therefore
     * the PMacc wrappers only need to forward to the free alpaka::math functions. The macro
     * signatures are kept to avoid touching every function header.
     */
#define ALPAKA_UNARY_MATH_FN_DETAIL(functionName, ...)                                                                \
    /**                                                                                                               \
     * @tparam T_Type argument type                                                                                   \
     * @param arg input argument                                                                                      \
     */                                                                                                               \
    template<typename T_Type>                                                                                       \
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto functionName(T_Type const& arg)->decltype(::alpaka::math::functionName(  \
        arg))                                                                                                         \
    {                                                                                                                 \
        return ::alpaka::math::functionName(arg);                                                                     \
    }

#define ALPAKA_BINARY_MATH_FN_DETAIL(functionName, ...)                                                               \
    /**                                                                                                               \
     * @tparam T_Type1 first argument type                                                                            \
     * @tparam T_Type2 second argument type                                                                           \
     * @param arg1 first input argument                                                                               \
     * @param arg2 second input argument                                                                              \
     */                                                                                                               \
    template<typename T_Type1, typename T_Type2>                                                                   \
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto functionName(T_Type1 const& arg1, T_Type2 const& arg2)                   \
        ->decltype(::alpaka::math::functionName(arg1, arg2))                                                          \
    {                                                                                                                 \
        return ::alpaka::math::functionName(arg1, arg2);                                                             \
    }

#define ALPAKA_UNARY_MATH_FN(functionName, alpakaMathConcept, alpakaMathTrait)                                        \
    ALPAKA_UNARY_MATH_FN_DETAIL(functionName, alpakaMathConcept, alpakaMathTrait)
#define ALPAKA_BINARY_MATH_FN(functionName, alpakaMathConcept, alpakaMathTrait)                                       \
    ALPAKA_BINARY_MATH_FN_DETAIL(functionName, alpakaMathConcept, alpakaMathTrait)

} // namespace pmacc::math
