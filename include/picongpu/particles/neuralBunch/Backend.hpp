/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Runtime selection of the alpaka backend that was enabled at configure time. */

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace picongpu::particles::neuralBunch
{
    /** Invoke @p function(device, exec) on the first available device whose API matches ``requested``.
     *
     * ``requested`` is ``"host"`` or ``"cuda"``. The backend list is fixed at compile time, so an unavailable
     * backend produces a clear runtime error instead of silently falling back.
     */
    template<typename TFunction>
    void dispatchBackend(std::string const& requested, TFunction&& function)
    {
        auto const backends
            = alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors);

        bool invoked = false;
        std::apply(
            [&](auto const&... configuration)
            {
                auto tryConfiguration = [&](auto const& cfg)
                {
                    using Api = std::decay_t<decltype(cfg[alpaka::object::api])>;
                    bool const match = (requested == "host" && std::same_as<Api, alpaka::api::Host>)
                                       || (requested == "cuda" && std::same_as<Api, alpaka::api::Cuda>);
                    if(!match || invoked)
                        return;
                    auto const spec = alpaka::onHost::makeDeviceSpec(cfg);
                    auto selector = alpaka::onHost::makeDeviceSelector(spec);
                    if(!selector.isAvailable())
                        return;
                    auto device = selector.makeDevice(0);
                    auto exec = cfg[alpaka::object::exec];
                    invoked = true;
                    function(device, exec);
                };
                (void) tryConfiguration;
                (tryConfiguration(configuration), ...);
            },
            backends);

        if(!invoked)
            throw std::runtime_error{
                "no available alpaka device for backend '" + requested + "' (was it enabled at configure time?)"};
    }
} // namespace picongpu::particles::neuralBunch
