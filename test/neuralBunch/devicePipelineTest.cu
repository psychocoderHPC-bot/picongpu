/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Device-resident primitive checks: batched GEMV, weighted tile accumulation and the batched point field.
 * These are the building blocks of the host-free PIConGPU initializer; the test keeps every intermediate on
 * the accelerator and compares only the final small result with an independent host computation.
 */

#include "picongpu/particles/neuralBunch/Backend.hpp"
#include "picongpu/particles/neuralBunch/DeviceBanded.hpp"
#include "picongpu/particles/neuralBunch/Inference.hpp"
#include "picongpu/particles/neuralBunch/PointField.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/matrix/gemv.hpp>
#include <alpaka/nn/onHost/ops/weighted_add.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace nb = picongpu::particles::neuralBunch;

namespace
{
    int failures = 0;

    void check(bool const condition, char const* message, double const detail = 0.0)
    {
        if(!condition)
        {
            std::cerr << "FAIL: " << message << " (" << detail << ")\n";
            ++failures;
        }
    }

    float pseudoRandom(std::size_t& state)
    {
        state = state * 6'364'136'223'846'793'005u + 1'442'695'040'888'963'407u;
        return static_cast<float>((state >> 33) % 2001u) / 1000.0f - 1.0f;
    }
} // namespace

int main(int argc, char** argv)
{
    std::string const backend = argc > 1 ? argv[1] : "cuda";
    nb::dispatchBackend(
        backend,
        [&](auto& device, auto const& exec)
        {
            auto queue = device.makeQueue();
            constexpr std::size_t latent = 8u;
            constexpr std::size_t batches = 4u;
            constexpr std::size_t slots = nb::kOutDim;
            std::size_t state = 42u;

            // 1) gemvBatched: coefficients [B,latent] x matrix [latent,slots].
            std::vector<float> hostCoefficients(batches * latent);
            std::vector<float> hostMatrix(latent * slots);
            for(auto& value : hostCoefficients)
                value = pseudoRandom(state);
            for(auto& value : hostMatrix)
                value = pseudoRandom(state);

            auto deviceCoefficients = alpaka::onHost::alloc<float>(device, alpaka::Vec{batches, latent});
            auto hostCoefficientsBuffer = alpaka::onHost::allocHost<float>(alpaka::Vec{batches, latent});
            for(std::size_t b = 0u; b < batches; ++b)
                for(std::size_t k = 0u; k < latent; ++k)
                    hostCoefficientsBuffer[alpaka::Vec<std::size_t, 2>{b, k}] = hostCoefficients[b * latent + k];
            alpaka::onHost::memcpy(queue, deviceCoefficients, hostCoefficientsBuffer);
            hostCoefficientsBuffer.keepAlive(queue);

            auto deviceMatrix = alpaka::onHost::alloc<float>(device, alpaka::Vec{latent, slots});
            auto hostMatrixBuffer = alpaka::onHost::allocHost<float>(alpaka::Vec{latent, slots});
            for(std::size_t k = 0u; k < latent; ++k)
                for(std::size_t slot = 0u; slot < slots; ++slot)
                    hostMatrixBuffer[alpaka::Vec<std::size_t, 2>{k, slot}] = hostMatrix[k * slots + slot];
            alpaka::onHost::memcpy(queue, deviceMatrix, hostMatrixBuffer);
            hostMatrixBuffer.keepAlive(queue);

            auto deviceResult = alpaka::onHost::alloc<float>(device, alpaka::Vec{batches, slots});
            alpaka::nn::onHost::gemvBatched<float>(queue, deviceCoefficients, deviceMatrix, deviceResult);
            auto hostResult = alpaka::onHost::allocHost<float>(alpaka::Vec{batches, slots});
            alpaka::onHost::memcpy(queue, hostResult, deviceResult);
            alpaka::onHost::wait(queue);

            double gemvMaxError = 0.0;
            for(std::size_t b = 0u; b < batches; ++b)
                for(std::size_t slot = 0u; slot < slots; ++slot)
                {
                    double expected = 0.0;
                    for(std::size_t k = 0u; k < latent; ++k)
                        expected += static_cast<double>(hostCoefficients[b * latent + k])
                                    * static_cast<double>(hostMatrix[k * slots + slot]);
                    auto const actual = hostResult[alpaka::Vec<std::size_t, 2>{b, slot}];
                    gemvMaxError = std::max(gemvMaxError, std::abs(actual - expected));
                }
            check(gemvMaxError < 1.0e-3, "gemvBatched matches the host dot products", gemvMaxError);

            // 2) weightedAdd into a tile accumulator, fixed ascending row order.
            std::vector<float> hostWeights{0.5f, -1.25f, 0.75f, 2.0f};
            auto deviceWeights = alpaka::onHost::alloc<float>(device, alpaka::Vec{batches});
            auto hostWeightsBuffer = alpaka::onHost::allocHost<float>(alpaka::Vec{batches});
            for(std::size_t b = 0u; b < batches; ++b)
                hostWeightsBuffer[alpaka::Vec<std::size_t, 1>{b}] = hostWeights[b];
            alpaka::onHost::memcpy(queue, deviceWeights, hostWeightsBuffer);
            hostWeightsBuffer.keepAlive(queue);

            auto deviceTile = alpaka::onHost::alloc<float>(device, alpaka::Vec{slots});
            alpaka::nn::onHost::ops::fill(queue, exec, deviceTile, 0.0f);
            alpaka::nn::onHost::ops::weightedAdd<float>(queue, exec, deviceResult, deviceWeights, deviceTile);
            auto hostTile = alpaka::onHost::allocHost<float>(alpaka::Vec{slots});
            alpaka::onHost::memcpy(queue, hostTile, deviceTile);
            alpaka::onHost::wait(queue);

            std::vector<double> expectedTile(slots, 0.0);
            for(std::size_t b = 0u; b < batches; ++b)
                for(std::size_t slot = 0u; slot < slots; ++slot)
                    expectedTile[slot] += static_cast<double>(hostWeights[b])
                                          * static_cast<double>(hostResult[alpaka::Vec<std::size_t, 2>{b, slot}]);
            double weightedMaxError = 0.0;
            for(std::size_t slot = 0u; slot < slots; ++slot)
                weightedMaxError = std::max(
                    weightedMaxError,
                    std::abs(hostTile[alpaka::Vec<std::size_t, 1>{slot}] - expectedTile[slot]));
            check(weightedMaxError < 1.0e-2, "weightedAdd accumulates the batch deterministically", weightedMaxError);

            // 3) batched point field matches the host per-source evaluation.
            std::vector<nb::Source> sources(3u);
            for(std::size_t i = 0u; i < sources.size(); ++i)
            {
                sources[i].r[0] = 0.2 + 0.3 * static_cast<double>(i);
                sources[i].r[1] = 1.1;
                sources[i].r[2] = 2.4;
                sources[i].u[0] = 0.1;
                sources[i].u[1] = -0.2;
                sources[i].u[2] = 0.05;
                sources[i].weight = 0.5 + static_cast<double>(i);
                sources[i].aY = 1.3;
                sources[i].aZ = 0.7;
            }
            nb::Vec3d const cells{1.0, 1.3, 0.7};
            auto deviceSources = nb::deviceDetail::packSourceGeometry(queue, std::span<nb::Source const>{sources});
            auto devicePointField = alpaka::onHost::alloc<float>(device, alpaka::Vec{sources.size(), slots});
            nb::enqueuePointFieldBatched(queue, exec, deviceSources, cells, devicePointField);
            auto hostPointField = alpaka::onHost::allocHost<float>(alpaka::Vec{sources.size(), slots});
            alpaka::onHost::memcpy(queue, hostPointField, devicePointField);
            alpaka::onHost::wait(queue);

            double pointFieldMaxError = 0.0;
            for(std::size_t i = 0u; i < sources.size(); ++i)
            {
                auto const reference = nb::pointFieldTileHost(
                    nb::Vec3d{sources[i].r[0], sources[i].r[1], sources[i].r[2]},
                    nb::Vec3d{sources[i].u[0], sources[i].u[1], sources[i].u[2]},
                    cells);
                for(std::size_t slot = 0u; slot < slots; ++slot)
                    pointFieldMaxError = std::max(
                        pointFieldMaxError,
                        static_cast<double>(
                            std::abs(hostPointField[alpaka::Vec<std::size_t, 2>{i, slot}] - reference[slot])));
            }
            check(pointFieldMaxError < 1.0e-4, "batched point field matches the host reference", pointFieldMaxError);
        });

    if(failures != 0)
    {
        std::cerr << failures << " device pipeline check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "device pipeline checks passed\n";
    return EXIT_SUCCESS;
}
