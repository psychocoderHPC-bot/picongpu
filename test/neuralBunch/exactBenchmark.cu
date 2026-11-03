/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Exact device quadrature parity and throughput benchmark. The "check" mode is a ctest; the default mode reports
 * per-pair timings used for the production cost model.
 */

#include "picongpu/particles/neuralBunch/Backend.hpp"
#include "picongpu/particles/neuralBunch/ExactShapeField.hpp"
#include "picongpu/particles/neuralBunch/Inference.hpp"
#include "picongpu/particles/neuralBunch/PointField.hpp"
#include "picongpu/particles/neuralBunch/ReferenceKernel.hpp"
#include "picongpu/particles/neuralBunch/ReferenceTile.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace picongpu::particles::neuralBunch;

namespace
{
    std::vector<Source> makeSources(std::size_t count, Vec3d const& r, double weight = 1.0)
    {
        std::vector<Source> sources(count);
        for(auto& source : sources)
        {
            source.r[0] = r.x;
            source.r[1] = r.y;
            source.r[2] = r.z;
            source.u[0] = 0.2;
            source.u[1] = 0.1;
            source.weight = weight;
        }
        return sources;
    }

    std::vector<reference::Source> toReference(std::vector<Source> const& sources)
    {
        std::vector<reference::Source> out(sources.size());
        for(std::size_t i = 0; i < sources.size(); ++i)
        {
            out[i].x = {sources[i].r[0], sources[i].r[1], sources[i].r[2]};
            out[i].u = {sources[i].u[0], sources[i].u[1], sources[i].u[2]};
            // The host reference applies q/e = -w itself, the device accumulator is fed -weight, so the source
            // weight maps one to one.
            out[i].w = sources[i].weight;
        }
        return out;
    }

    struct HostTile
    {
        std::vector<double> values;
        double maxAbs = 0.0;
    };

    /** Exactly the runtime path: zero, accumulate on device in batches, one host copy per tile. */
    template<typename TQueue, typename TExec>
    HostTile tileExactDevice(
        TQueue& queue,
        TExec exec,
        std::vector<Source> const& sources,
        Vec3d const& cells,
        std::size_t maxThreads = std::numeric_limits<std::size_t>::max())
    {
        auto const count = sources.size();
        auto hostSources = alpaka::onHost::allocHost<double>(alpaka::Vec{count, std::size_t{6}});
        auto hostWeights = alpaka::onHost::allocHost<double>(alpaka::Vec{count});
        for(std::size_t i = 0; i < count; ++i)
        {
            for(std::size_t c = 0; c < 3u; ++c)
            {
                hostSources[alpaka::Vec<std::size_t, 2>{i, c}] = sources[i].r[c];
                hostSources[alpaka::Vec<std::size_t, 2>{i, c + 3u}] = sources[i].u[c];
            }
            hostWeights[alpaka::Vec<std::size_t, 1>{i}] = -sources[i].weight;
        }
        auto deviceSources = alpaka::onHost::allocLike(queue.getDevice(), hostSources);
        auto deviceWeights = alpaka::onHost::allocLike(queue.getDevice(), hostWeights);
        alpaka::onHost::memcpy(queue, deviceSources, hostSources);
        alpaka::onHost::memcpy(queue, deviceWeights, hostWeights);
        hostSources.keepAlive(queue);
        hostWeights.keepAlive(queue);
        auto accumulator = alpaka::onHost::alloc<double>(queue.getDevice(), alpaka::Vec{kOutDim});
        enqueueExactZero(queue, exec, accumulator);
        enqueueExactShapeFieldTileAccumulate<double>(
            queue,
            exec,
            deviceSources,
            deviceWeights,
            count,
            cells,
            ExactSettings{},
            accumulator,
            maxThreads);
        auto hostAccumulator = alpaka::onHost::allocHost<double>(alpaka::Vec{kOutDim});
        alpaka::onHost::memcpy(queue, hostAccumulator, accumulator);
        alpaka::onHost::wait(queue);
        HostTile tile;
        tile.values.resize(kOutDim);
        for(std::size_t k = 0; k < kOutDim; ++k)
        {
            tile.values[k] = hostAccumulator[alpaka::Vec<std::size_t, 1>{k}];
            tile.maxAbs = std::max(tile.maxAbs, std::abs(tile.values[k]));
        }
        return tile;
    }

    double maxRelative(std::vector<double> const& actual, std::vector<double> const& expected)
    {
        double maxDiff = 0.0;
        double scale = 1.0;
        for(std::size_t i = 0; i < expected.size(); ++i)
        {
            maxDiff = std::max(maxDiff, std::abs(actual[i] - expected[i]));
            scale = std::max(scale, std::abs(expected[i]));
        }
        return maxDiff / scale;
    }

    int runCheck(std::string const& backend)
    {
        int failures = 0;
        auto check = [&](char const* name, bool ok, double measured)
        {
            std::cout << (ok ? "PASS " : "FAIL ") << name << " value=" << measured << "\n";
            if(!ok)
                ++failures;
        };
        check(
            "distance_limit_default",
            exactDistanceLimit(1.0, 1.0, ExactDistancePolicy{}) == 128.0,
            exactDistanceLimit(1.0, 1.0, ExactDistancePolicy{}));
        ExactDistancePolicy const scaled{128.0, true, 48.0, 512.0};
        check("distance_limit_aspect2", exactDistanceLimit(1.0, 2.0, scaled) == 48.0, exactDistanceLimit(1.0, 2.0, scaled));
        check("distance_limit_aspect10", exactDistanceLimit(1.0, 10.0, scaled) == 512.0, exactDistanceLimit(1.0, 10.0, scaled));

        dispatchBackend(
            backend,
            [&](auto& device, auto const& exec)
            {
                auto queue = device.makeQueue();
                struct Case
                {
                    char const* name;
                    Vec3d r;
                    Vec3d cells;
                    double tolerance;
                };
                for(auto const& c : {Case{"smooth", {20.0, 4.0, 2.0}, {1.0, 1.0, 1.0}, 5.0e-6},
                                     Case{"in_support", {4.0, 4.0, 2.0}, {1.0, 1.0, 1.0}, 1.0e-3},
                                     Case{"smooth_aniso", {20.0, 4.0, 2.0}, {1.0, 2.0, 0.5}, 5.0e-6},
                                     Case{"in_support_aniso", {4.0, 4.0, 2.0}, {1.0, 2.0, 0.5}, 1.0e-3}})
                {
                    auto const sources = makeSources(4u, c.r);
                    auto const deviceTile = tileExactDevice(queue, exec, sources, c.cells);
                    auto const reference = reference::tileFieldParallel(
                        {0.0, 0.0, 0.0},
                        toReference(sources),
                        c.cells.x,
                        c.cells.y,
                        c.cells.z);
                    double const dx2 = c.cells.x * c.cells.x;
                    std::vector<double> expected(reference.size());
                    for(std::size_t k = 0; k < expected.size(); ++k)
                        expected[k] = reference[k] / dx2;
                    double const rel = maxRelative(deviceTile.values, expected);
                    check(c.name, rel <= c.tolerance && deviceTile.maxAbs < 1.0e3, rel);
                    bool finite = true;
                    for(double v : deviceTile.values)
                        finite = finite && std::isfinite(v);
                    check("finite", finite, deviceTile.maxAbs);
                }
            });
        return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    void runBenchmark(std::string const& backend, std::size_t batch, std::size_t maxThreads)
    {
        Vec3d const cells{1.0, 1.0, 1.0};
        dispatchBackend(
            backend,
            [&](auto& device, auto const& exec)
            {
                auto queue = device.makeQueue();
                for(auto const& [name, r] : {std::pair<char const*, Vec3d>{"smooth", {20.0, 4.0, 2.0}},
                                             std::pair<char const*, Vec3d>{"in_support", {4.0, 4.0, 2.0}}})
                {
                    auto const sources = makeSources(batch, r);
                    std::size_t const repeats = 3u;
                    (void) tileExactDevice(queue, exec, sources, cells, maxThreads);
                    auto const begin = std::chrono::steady_clock::now();
                    for(std::size_t i = 0; i < repeats; ++i)
                        (void) tileExactDevice(queue, exec, sources, cells, maxThreads);
                    double const seconds
                        = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
                          / static_cast<double>(repeats);
                    std::cout << "exact " << name << " batch=" << batch << " maxthreads=" << maxThreads
                              << " seconds=" << seconds
                              << " us_per_pair=" << seconds / static_cast<double>(batch) * 1.0e6
                              << " pairs_per_second=" << static_cast<double>(batch) / seconds << "\n";
                }

                // Analytic point field: the far-band cost per (source, tile) pair.
                for(std::size_t const pointBatch : {std::size_t{1024}})
                {
                    auto const sources = makeSources(pointBatch, {120.0, 60.0, 30.0});
                    auto hostSources = alpaka::onHost::allocHost<double>(alpaka::Vec{pointBatch, std::size_t{6}});
                    for(std::size_t i = 0; i < pointBatch; ++i)
                        for(std::size_t c = 0; c < 3u; ++c)
                        {
                            hostSources[alpaka::Vec<std::size_t, 2>{i, c}] = sources[i].r[c];
                            hostSources[alpaka::Vec<std::size_t, 2>{i, c + 3u}] = sources[i].u[c];
                        }
                    auto deviceSources = alpaka::onHost::allocLike(queue.getDevice(), hostSources);
                    auto deviceOut
                        = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{pointBatch, kTileSlotCount});
                    alpaka::onHost::memcpy(queue, deviceSources, hostSources);
                    hostSources.keepAlive(queue);
                    std::size_t const repeats = 5u;
                    auto const begin = std::chrono::steady_clock::now();
                    for(std::size_t i = 0; i < repeats; ++i)
                        enqueuePointFieldBatched(queue, exec, deviceSources, cells, deviceOut);
                    alpaka::onHost::wait(queue);
                    double const seconds
                        = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
                          / static_cast<double>(repeats);
                    double const pairs = static_cast<double>(pointBatch);
                    std::cout << "point smooth batch=" << pointBatch << " seconds=" << seconds
                              << " ns_per_pair=" << seconds / pairs * 1.0e9
                              << " pairs_per_second=" << pairs / seconds << "\n";
                }
            });
    }
} // namespace

int main(int argc, char** argv)
{
    std::string const mode = argc > 1 ? argv[1] : "bench";
    std::string const backend = argc > 2 ? argv[2] : "cuda";
    std::size_t const batch = argc > 3 ? static_cast<std::size_t>(std::stoull(argv[3])) : 64u;
    std::size_t const maxThreads
        = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : std::numeric_limits<std::size_t>::max();
    if(mode == "check")
        return runCheck(backend);
    runBenchmark(backend, batch, maxThreads);
    return EXIT_SUCCESS;
}
