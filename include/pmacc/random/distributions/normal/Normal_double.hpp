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

#ifndef ALPAKA_DISABLE_VENDOR_RNG
#    include "pmacc/algorithms/math.hpp"
#    include "pmacc/random/distributions/Normal.hpp"
#    include "pmacc/random/distributions/Uniform.hpp"
#    include "pmacc/random/distributions/misc/MullerBox.hpp"
#    include "pmacc/random/methods/MRG32k3aMin.hpp"
#    include "pmacc/random/methods/XorMin.hpp"
#    include "pmacc/types.hpp"

#    include <type_traits>

namespace pmacc
{
    namespace random
    {
        namespace distributions
        {
            namespace detail
            {
/* XorMin and MRG32k3aMin may alias the alpaka RNG when a CPU accelerator is used,
 * therefore we must not add a specialization for those RNG methods in that case.
 * XorMin is a distinct vendor type on CUDA and HIP, while MRG32k3aMin is only a
 * distinct vendor type on CUDA. On HIP MRG32k3aMin aliases AlpakaRand (see
 * MRG32k3aMin.hpp), so a specialization would also match AlpakaRand and shadow
 * Normal_generic, changing the host RNG stream away from the alpaka2-identical
 * values. Keep the MRG32k3aMin specialization CUDA only.
 */
#    if (ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED)
                //! specialization for XorMin
                template<typename T_Acc>
                struct Normal<double, methods::XorMin<T_Acc>, void> : public MullerBox<double, methods::XorMin<T_Acc>>
                {
                };
#    endif
#    if (ALPAKA_ACC_GPU_CUDA_ENABLED)
                //! specialization for MRG32k3aMin
                template<typename T_Acc>
                struct Normal<double, methods::MRG32k3aMin<T_Acc>, void>
                    : public MullerBox<double, methods::MRG32k3aMin<T_Acc>>
                {
                };
#    endif
            } // namespace detail
        } // namespace distributions
    } // namespace random
} // namespace pmacc
#endif
