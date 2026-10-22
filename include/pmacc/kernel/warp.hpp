/* Copyright 2015-2024 Rene Widera, Alexander Grund
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

#include "pmacc/types.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <type_traits>
#include <tuple>

namespace pmacc
{
    namespace kernel
    {
        /** get lane id of a thread within a warp
         *
         * id is in range [0,WARPSIZE-1]
         */
        template<typename T_Acc>
        DINLINE uint32_t getLaneId(T_Acc const& acc)
        {
            return ::alpaka::onAcc::warp::getLaneIdx(acc);
        }

        /** broadcast data within a warp without using shared memory
         *
         * @param acc alpaka accelerator
         * @param data value to broadcast
         * @param srcLaneId lane id of the source thread
         * @return value send by the source thread
         */
        template<typename T_Acc, typename T_DataType>
        DINLINE T_DataType warpBroadcast(T_Acc const& acc, T_DataType const& data, int const srcLaneId)
        {
            return ::alpaka::onAcc::warp::shfl(acc, data, static_cast<uint32_t>(srcLaneId));
        }

        /** check if T is equal to at least one type in the tuple list
         *
         * @param tuple of types, each type is checked against T
         * @return true if T is equal to at least one type from the tuple list, else false
         */
        template<typename T, typename... T_TypeList>
        inline constexpr bool matchAnyType(std::tuple<T_TypeList...>)
        {
            return std::disjunction_v<std::is_same<T, T_TypeList>...>;
        }

    } // namespace kernel
} // namespace pmacc
