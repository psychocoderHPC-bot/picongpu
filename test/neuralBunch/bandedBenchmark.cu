/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Optional throughput benchmark for the banded evaluator. Not a ctest.
 *
 * Build target ``neuralBunchBandedBenchmark`` and run
 * ``neuralBunchBandedBenchmark <model-prefix> [cuda|host] [sources] [repeat]``.
 */

#include "picongpu/particles/neuralBunch/Backend.hpp"
#include "picongpu/particles/neuralBunch/BandPolicy.hpp"
#include "picongpu/particles/neuralBunch/Inference.hpp"
#include "picongpu/particles/neuralBunch/Model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace picongpu::particles::neuralBunch;

namespace
{
    std::vector<Source> makeSources(std::size_t count, double radialOffset)
    {
        std::vector<Source> sources(count);
        for(std::size_t i = 0u; i < count; ++i)
        {
            auto& source = sources[i];
            source.r[0] = radialOffset + 0.01 * static_cast<double>(i % 13u);
            source.r[1] = 4.0 + 0.02 * static_cast<double>(i % 11u);
            source.r[2] = 2.0 + 0.01 * static_cast<double>(i % 7u);
            source.u[0] = 0.01 * static_cast<double>(static_cast<int>(i % 5u) - 2);
            source.u[1] = 0.5;
            source.u[2] = 0.0;
            source.weight = 0.5 + 0.01 * static_cast<double>(i % 9u);
            source.aY = 1.0;
            source.aZ = 1.0;
        }
        return sources;
    }
} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <model-prefix> [cuda|host] [sources] [repeat]\n";
        return EXIT_FAILURE;
    }
    std::string const backend = argc > 2 ? argv[2] : "cuda";
    auto const sources = argc > 3 ? static_cast<std::size_t>(std::stoull(argv[3])) : 4096u;
    auto const repeat = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 5u;

    dispatchBackend(
        backend,
        [&](auto& device, auto const& exec)
        {
            auto queue = device.makeQueue();
            auto const model = Model::load(argv[1]);
            auto const& meta = model.metadata();
            auto const tensors = model.loadTensors(queue);
            BandPolicy const policy{};
            FieldNormalization const normalization{1.0e-7};

            auto run = [&](std::vector<Source> const& batch, char const* name)
            {
                // Warm-up.
                (void) evaluateTileBanded(
                    queue,
                    exec,
                    meta,
                    tensors,
                    batch,
                    policy,
                    {},
                    1.0,
                    1.0,
                    normalization,
                    batch.size(),
                    64u);
                double best = 1.0e30;
                for(std::size_t r = 0u; r < repeat; ++r)
                {
                    auto const start = std::chrono::steady_clock::now();
                    (void) evaluateTileBanded(
                        queue,
                        exec,
                        meta,
                        tensors,
                        batch,
                        policy,
                        {},
                        1.0,
                        1.0,
                        normalization,
                        batch.size(),
                        64u);
                    best = std::min(
                        best,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
                }
                double const pairs = static_cast<double>(batch.size());
                std::cout << name << "_pairs=" << batch.size() << " " << name << "_seconds=" << best << " " << name
                          << "_pairs_per_second=" << pairs / best << " " << name
                          << "_ns_per_pair=" << best / pairs * 1.0e9 << "\n";
            };

            run(makeSources(sources, 100.0), "far_point_field");
            run(makeSources(sources, 20.0), "mid_model");
        });
    return EXIT_SUCCESS;
}
