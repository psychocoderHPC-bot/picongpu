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

namespace pmacc::traits
{
    /** Whether the device provides host mapped memory allocations.
     *
     * alpaka3 dropped the runtime query of the legacy `hasMappedBufSupport`. All
     * device backends shipped by alpaka3 (host, CUDA, HIP, SYCL) provide the
     * `AllocMapped` operation, so the capability is always available.
     */
    template<typename T_Device>
    inline constexpr bool isMappedMemorySupported_v = true;
} // namespace pmacc::traits
