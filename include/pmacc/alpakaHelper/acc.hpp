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

#include "pmacc/traits/IsCpuAcc.hpp"
#include "pmacc/traits/IsMappedMemorySupported.hpp"

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace pmacc
{
    using IdxType = uint32_t;
    using MemIdxType = size_t;

    //! alpaka API/device kind/executor used for the compute device selected at build time
#if defined(ALPAKA_LANG_CUDA) && ALPAKA_LANG_CUDA
    using ComputeApi = ::alpaka::api::Cuda;
    using ComputeDeviceKind = ::alpaka::deviceKind::NvidiaGpu;
    inline constexpr auto ComputeExecutor = ::alpaka::exec::gpuCuda;
#elif defined(ALPAKA_LANG_HIP) && ALPAKA_LANG_HIP
    using ComputeApi = ::alpaka::api::Hip;
    using ComputeDeviceKind = ::alpaka::deviceKind::AmdGpu;
    inline constexpr auto ComputeExecutor = ::alpaka::exec::gpuHip;
#else
    using ComputeApi = ::alpaka::api::Host;
    using ComputeDeviceKind = ::alpaka::deviceKind::Cpu;
    inline constexpr auto ComputeExecutor = ::alpaka::exec::cpuOmpBlocks;
#endif

    using HostApi = ::alpaka::api::Host;
    using HostDeviceKind = ::alpaka::deviceKind::Cpu;

    using HostDevice = ::alpaka::onHost::Device<HostApi, HostDeviceKind>;
    using ComputeDevice = ::alpaka::onHost::Device<ComputeApi, ComputeDeviceKind>;

    //! plain executor type, safe to use in function signatures (unlike ALPAKA_TYPEOF)
    using ComputeExecutorType = ALPAKA_TYPEOF(ComputeExecutor);

    /** Lockstep accelerator tag.
     *
     * alpaka3 no longer has per-dimensionality accelerator types. This tag carries the
     * executor which is used whenever a compile time accelerator property is required.
     */
    template<uint32_t T_dim>
    struct Acc
    {
        static constexpr auto executor = ComputeExecutor;

        //! true if kernels are executed on a GPU backend
        static constexpr bool isGpu = !std::is_same_v<ComputeApi, ::alpaka::api::Host>;
    };

#if defined(PMACC_USE_ASYNC_QUEUES) && PMACC_USE_ASYNC_QUEUES == 1
    inline constexpr auto ComputeQueuePolicies = ::alpaka::onHost::QueuePolicyList{::alpaka::queueKind::nonBlocking};
#else
    inline constexpr auto ComputeQueuePolicies = ::alpaka::onHost::QueuePolicyList{::alpaka::queueKind::blocking};
#endif

    using ComputeDeviceQueue = decltype(
        std::declval<ComputeDevice&>().makeQueue(std::declval<decltype(ComputeQueuePolicies) const&>()));

    using ComputeDeviceEvent = decltype(std::declval<ComputeDevice&>().makeEvent());

    /** Concrete owning buffer type returned by an alpaka3 device allocation. */
    template<typename T_Type, uint32_t T_dim>
    using ComputeBuffer = decltype(::alpaka::onHost::alloc<T_Type>(
        std::declval<ComputeDevice const&>(),
        std::declval<::alpaka::Vec<MemIdxType, T_dim>>()));

    /** Concrete owning host-mapped buffer type returned by an alpaka3 device allocation. */
    template<typename T_Type, uint32_t T_dim>
    using ComputeMappedBuffer = decltype(::alpaka::onHost::allocMapped<T_Type>(
        std::declval<ComputeDevice const&>(),
        std::declval<::alpaka::Vec<MemIdxType, T_dim>>()));

    /*! device compile flag
     *
     * Enabled if the compiler processes currently a separate compile path for the device code
     *
     * @attention value is always 0 for alpaka CPU accelerators
     *
     * Value is 1 if device path is compiled else 0
     */
#if defined(__CUDA_ARCH__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1 && defined(__HIP__))
#    define PMACC_DEVICE_COMPILE 1
#else
#    define PMACC_DEVICE_COMPILE 0
#endif

} // namespace pmacc
