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

#include <alpaka/alpaka.hpp>

#include <cassert>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pmacc
{
    namespace detail
    {
        /** Extract the alpaka API and device kind from an onHost::Device type. */
        template<typename T_Device>
        struct DeviceSpecOf;

        template<typename T_Api, typename T_DeviceKind>
        struct DeviceSpecOf<::alpaka::onHost::Device<T_Api, T_DeviceKind>>
        {
            using Api = T_Api;
            using DeviceKind = T_DeviceKind;
        };
    } // namespace detail

    namespace manager
    {
        template<typename T_DeviceType>
        struct Device
        {
            using DeviceType = T_DeviceType;
            using Api = typename detail::DeviceSpecOf<DeviceType>::Api;
            using DeviceKind = typename detail::DeviceSpecOf<DeviceType>::DeviceKind;

            /** Handle to the alpaka device.
             *
             * The handle is intentionally leaked by giving it a no-op deleter. alpaka3's
             * unifiedCudaHip::Device destructor calls hipDeviceReset()/cudaDeviceReset().
             * PMacc keeps the handle in this function-local static, so without the leak the
             * last reference is released from QueueController's static destructor while the
             * CUDA/HIP runtime's own atexit cleanup is still pending. hipDeviceReset() then
             * tears down the context underneath that cleanup and the process aborts with a
             * double free in libamdhip64 during __cxa_finalize (seen on ROCm 7.2).
             * The backend runtime owns context teardown and releases the leaked resources at
             * process exit, so the no-op deleter is safe for both HIP and CUDA.
             */
            std::shared_ptr<DeviceType> m_device;

            int m_devIdx = -1;

            static Device& get()
            {
                static Device instance;
                return instance;
            }

            static auto makeSelector()
            {
                return ::alpaka::onHost::makeDeviceSelector(Api{}, DeviceKind{});
            }

            auto device(int idx = 0) -> DeviceType&
            {
                if(m_devIdx != -1)
                {
                    if(m_devIdx == idx)
                        return *m_device;
                    else
                        throw std::runtime_error(
                            std::string("Device with id '") + std::to_string(m_devIdx)
                            + "' is already selected, changing the device is not allowed.");
                }
                else
                {
                    int const numDevices = count();
                    if(idx >= numDevices)
                    {
                        std::stringstream err;
                        err << "Unable to return device " << idx << ". There are only " << numDevices << " devices!";
                        throw std::runtime_error(err.str());
                    }

                    std::optional<DeviceType> dev;

                    auto selector = makeSelector();
                    try
                    {
                        dev = std::make_optional<DeviceType>(selector.makeDevice(static_cast<uint32_t>(idx)));
                    }
                    catch(std::runtime_error const& e)
                    {
                        throw std::runtime_error(e.what());
                    }
                    /* Intentionally never destroy the handle, see the m_device comment. */
                    m_device = std::shared_ptr<DeviceType>(
                        new DeviceType(std::move(*dev)),
                        [](DeviceType*) { /* no-op deleter: intentionally leak the handle */ });
                    m_devIdx = idx;
                    return *m_device;
                }
            }

            /**! reset the current device
             *
             * streams, memory and events on the current device must be
             * deleted at first by the user else the call will throw an error
             */
            void reset()
            {
                if(!m_device)
                {
                    std::stringstream err;
                    err << "device " << m_devIdx << " can not destroyed (was never created) ";
                    throw std::runtime_error(err.str());
                }
                else
                {
                    m_devIdx = -1;
                    /* Drop our reference but keep the handle alive: the no-op deleter above
                     * must not run at static-destruction time. */
                    m_device.reset();
                }
            }

            auto id() -> int
            {
                return m_devIdx;
            }

            auto current() -> DeviceType&
            {
                assert(m_device);
                return *m_device;
            }

            auto count() -> int
            {
                auto selector = makeSelector();
                return static_cast<int>(selector.getDeviceCount());
            }

        protected:
            Device()
            {
            }
        };

    } // namespace manager
} // namespace pmacc
