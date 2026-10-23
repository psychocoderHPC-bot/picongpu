/* Copyright 2017-2024 Rene Widera
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

#include <type_traits>

namespace pmacc
{
    namespace traits
    {
        /** Get number of workers
         *
         * the number of workers for a kernel depending on the used accelerator
         *
         * alpaka3 CPU executors only support one thread per block, therefore the number of
         * workers is clamped to one for all non-GPU backends.
         *
         * @tparam T_maxWorkers the maximum number of workers
         * @tparam T_Acc the accelerator type
         * @return @p ::value number of workers
         */
        template<uint32_t T_maxWorkers, typename T_Acc = Acc<DIM1>>
        struct GetNumWorkers
        {
            static constexpr uint32_t value = T_Acc::isGpu ? T_maxWorkers : 1u;
        };
    } // namespace traits
} // namespace pmacc
