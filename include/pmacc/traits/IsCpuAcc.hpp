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

#include <alpaka/alpaka.hpp>

#include <type_traits>
#include <utility>

namespace pmacc::traits
{
    namespace detail
    {
        /** Obtain the device kind of an accelerator or a lockstep worker.
         *
         * PIConGPU functors are invoked with the PMacc lockstep worker while some
         * code passes the raw alpaka accelerator. Both spellings are supported so
         * that a single trait can replace the legacy `alpaka::Dev<T_Acc>` checks.
         */
        template<typename T_Acc>
        constexpr auto getDeviceKind()
        {
            if constexpr(requires(T_Acc const& acc) { acc.getAcc().getDeviceKind(); })
                return std::declval<T_Acc const&>().getAcc().getDeviceKind();
            else
                return std::declval<T_Acc const&>().getDeviceKind();
        }
    } // namespace detail

    //! true if the accelerator/worker executes on the alpaka3 CPU backend
    template<typename T_Acc>
    inline constexpr bool isCpuAcc_v
        = std::is_same_v<ALPAKA_TYPEOF(detail::getDeviceKind<T_Acc>()), ::alpaka::deviceKind::Cpu>;
} // namespace pmacc::traits
