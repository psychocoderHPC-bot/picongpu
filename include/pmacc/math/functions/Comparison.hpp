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

#include "pmacc/math/functions/Common.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <type_traits>

namespace pmacc::math
{
    template<typename T_Type, uint32_t T_dim, typename T_Storage>
    class Vector;

    namespace detail
    {
        template<typename T>
        struct IsVector : std::false_type
        {
        };

        template<typename T_Type, uint32_t T_dim, typename T_Storage>
        struct IsVector<Vector<T_Type, T_dim, T_Storage>> : std::true_type
        {
        };
    } // namespace detail

    //! Calculates the smaller value of two arguments.
    template<typename T_Type1, typename T_Type2>
        requires(!detail::IsVector<std::decay_t<T_Type1>>::value && !detail::IsVector<std::decay_t<T_Type2>>::value)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto min(T_Type1 const& arg1, T_Type2 const& arg2)
        ->decltype(::alpaka::math::min(arg1, arg2))
    {
        return ::alpaka::math::min(arg1, arg2);
    }

    //! Calculates the larger value of two arguments.
    template<typename T_Type1, typename T_Type2>
        requires(!detail::IsVector<std::decay_t<T_Type1>>::value && !detail::IsVector<std::decay_t<T_Type2>>::value)
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto max(T_Type1 const& arg1, T_Type2 const& arg2)
        ->decltype(::alpaka::math::max(arg1, arg2))
    {
        return ::alpaka::math::max(arg1, arg2);
    }
} // namespace pmacc::math
