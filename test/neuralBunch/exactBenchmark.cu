/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Exact device quadrature parity and throughput benchmark. The "check" mode is a ctest; the
 * "bench" mode reports the per-pair quadrature timings used for the production cost model;
 * the "bench-table" mode (T150) reports the same per-pair timings through the precomputed
 * near-field table path (settings.useNearTable) for a table loaded from nktable/v1 files,
 * with all sources at the table velocity (the single-velocity operator case).
 */

#include "picongpu/particles/neuralBunch/Backend.hpp"
#include "picongpu/particles/neuralBunch/ExactShapeField.hpp"
#include "picongpu/particles/neuralBunch/Inference.hpp"
#include "picongpu/particles/neuralBunch/NearFieldTable.hpp"
#include "picongpu/particles/neuralBunch/PointField.hpp"
#include "picongpu/particles/neuralBunch/ReferenceKernel.hpp"
#include "picongpu/particles/neuralBunch/ReferenceTile.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
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

    std::vector<Source> makeTableSources(std::size_t count, Vec3d const& r, Vec3d const& u)
    {
        std::vector<Source> sources(count);
        for(auto& source : sources)
        {
            source.r[0] = r.x;
            source.r[1] = r.y;
            source.r[2] = r.z;
            source.u[0] = u.x;
            source.u[1] = u.y;
            source.u[2] = u.z;
        }
        return sources;
    }

    std::vector<reference::Source> toReference(std::vector<Source> const& sources)
    {
        std::vector<reference::Source> out(sources.size());
        for(std::size_t i = 0u; i < sources.size(); ++i)
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
        ExactSettings const& settings = {},
        std::size_t maxThreads = std::numeric_limits<std::size_t>::max())
    {
        auto const count = sources.size();
        auto hostSources = alpaka::onHost::allocHost<double>(alpaka::Vec{count, std::size_t{6}});
        auto hostWeights = alpaka::onHost::allocHost<double>(alpaka::Vec{count});
        for(std::size_t i = 0u; i < count; ++i)
        {
            for(std::size_t c = 0u; c < 3u; ++c)
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
        auto accumulator = alpaka::onHost::alloc<double>(queue.getDevice(), alpaka::Vec{kOutDim});
        enqueueExactZero(queue, exec, accumulator);
        enqueueExactShapeFieldTileAccumulate<double>(
            queue,
            exec,
            deviceSources,
            deviceWeights,
            count,
            cells,
            settings,
            accumulator,
            maxThreads);
        auto hostAccumulator = alpaka::onHost::allocHost<double>(alpaka::Vec{kOutDim});
        alpaka::onHost::memcpy(queue, hostAccumulator, accumulator);
        alpaka::onHost::wait(queue);
        HostTile tile;
        tile.values.resize(kOutDim);
        for(std::size_t k = 0u; k < kOutDim; ++k)
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
        for(std::size_t i = 0u; i < expected.size(); ++i)
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
                    for(std::size_t k = 0u; k < expected.size(); ++k)
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
                for(auto const& [name, r] : {std::pair<char const*, Vec3d>{
                    "smooth",
                    {20.0, 4.0, 2.0}},
                    std::pair<char const*, Vec3d>{
                        "in_support",
                        {4.0, 4.0, 2.0}}})
                {
                    auto const sources = makeSources(batch, r);
                    std::size_t const repeats = 3u;
                    (void) tileExactDevice(queue, exec, sources, cells, ExactSettings{}, maxThreads);
                    auto const begin = std::chrono::steady_clock::now();
                    for(std::size_t i = 0u; i < repeats; ++i)
                        (void) tileExactDevice(queue, exec, sources, cells, ExactSettings{}, maxThreads);
                    double const seconds
                        = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
                          / static_cast<double>(repeats);
                    std::cout << "exact " << name << " batch=" << batch << " maxthreads=" << maxThreads
                              << " seconds=" << seconds
                              << " us_per_pair=" << seconds / static_cast<double>(batch) * 1.0e6
                              << " pairs_per_second=" << static_cast<double>(batch) / seconds
                              << " quadrature_nodes_per_slot=" << 64ull * (12ull * 12ull * 12ull + 16ull * 16ull * 16ull)
                              << "\n";
                }

                // Analytic point field: the far-band cost per (source, tile) pair.
                for(std::size_t const pointBatch : {std::size_t{1024}})
                {
                    auto const sources = makeSources(pointBatch, {120.0, 60.0, 30.0});
                    auto hostSources = alpaka::onHost::allocHost<double>(alpaka::Vec{pointBatch, std::size_t{6}});
                    for(std::size_t i = 0u; i < pointBatch; ++i)
                        for(std::size_t c = 0u; c < 3u; ++c)
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
                    for(std::size_t i = 0u; i < repeats; ++i)
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

    struct LoadedTable
    {
        std::vector<double> grid;
        Vec3d u{};
        Vec3d cells{};
        double gamma = 1.0;
        std::vector<float> hostData;
    };

    //! Read an nktable/v1 binary + JSON metadata; the data is converted to the float32
    //! device layout the lookup path uses.
    LoadedTable loadTableFile(std::string const& binPath)
    {
        LoadedTable t;
        std::string const jsonPath = binPath.substr(0, binPath.size() - 4) + ".json";
        std::ifstream jin(jsonPath);
        if(!jin)
            throw std::runtime_error{"cannot open table metadata " + jsonPath};
        nlohmann::json meta;
        jin >> meta;
        int n = meta.at("n").get<int>();
        if(n < 2 || n > kNearTableMaxGrid)
            throw std::runtime_error{"table grid size out of bounds"};
        t.grid.resize(n);
        for(int i = 0; i < n; ++i)
            t.grid[i] = meta.at("grid").at(i).get<double>();
        auto const vec3 = [&](char const* key)
        {
            return Vec3d{
                meta.at(key).at(0).get<double>(),
                meta.at(key).at(1).get<double>(),
                meta.at(key).at(2).get<double>()};
        };
        t.u = vec3("u_cell");
        t.cells = vec3("cells");
        t.gamma = meta.at("gamma").get<double>();
        std::ifstream bin(binPath, std::ios::binary);
        if(!bin)
            throw std::runtime_error{"cannot open table binary " + binPath};
        std::size_t const count = static_cast<std::size_t>(n) * n * n * 6u;
        char const dtype = meta.value("dtype", std::string{"float64"}).front();
        bool const f64 = dtype != 'f';
        std::vector<char> raw(count * (f64 ? 8u : 4u));
        bin.read(raw.data(), static_cast<std::streamsize>(raw.size()));
        if(bin.gcount() != static_cast<std::streamsize>(raw.size()))
            throw std::runtime_error{"short table read for " + binPath};
        t.hostData.resize(count);
        if(f64)
        {
            auto const* d = reinterpret_cast<double const*>(raw.data());
            for(std::size_t i = 0u; i < count; ++i)
                t.hostData[i] = static_cast<float>(d[i]);
        }
        else
        {
            auto const* f = reinterpret_cast<float const*>(raw.data());
            t.hostData.assign(f, f + count);
        }
        return t;
    }

    void runBenchmarkTable(std::string const& tablePath, std::string const& backend, std::size_t batch, std::size_t maxThreads)
    {
        LoadedTable const t = loadTableFile(tablePath);
        dispatchBackend(
            backend,
            [&](auto& device, auto const& exec)
            {
                auto queue = device.makeQueue();
                auto hostData = alpaka::onHost::allocHost<float>(alpaka::Vec{t.hostData.size()});
                for(std::size_t i = 0u; i < t.hostData.size(); ++i)
                    hostData[alpaka::Vec<std::size_t, 1>{i}] = t.hostData[i];
                NearFieldTable const table = uploadNearFieldTable(queue, hostData, t.grid, t.u, t.cells, t.gamma);
                std::cout << "table " << tablePath << " gamma=" << t.gamma << " n=" << t.grid.size()
                          << " u=[" << t.u.x << ", " << t.u.y << ", " << t.u.z << "]"
                          << " cells=[" << t.cells.x << ", " << t.cells.y << ", " << t.cells.z << "]\n";
                ExactSettings settings;
                settings.useNearTable = true;
                settings.nearTable = table;
                Vec3d const cells = t.cells;
                for(auto const& [name, r] : {std::pair<char const*, Vec3d>{
                    "smooth",
                    {20.0, 4.0, 2.0}},
                    std::pair<char const*, Vec3d>{
                        "in_support",
                        {4.0, 4.0, 2.0}}})
                {
                    // All sources at the table velocity: the single-velocity operator case.
                    auto const sources = makeTableSources(batch, r, t.u);
                    std::size_t const repeats = 3u;
                    (void) tileExactDevice(queue, exec, sources, cells, settings, maxThreads);
                    auto const begin = std::chrono::steady_clock::now();
                    for(std::size_t i = 0u; i < repeats; ++i)
                        (void) tileExactDevice(queue, exec, sources, cells, settings, maxThreads);
                    double const seconds
                        = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()
                          / static_cast<double>(repeats);
                    std::cout << "exact-table " << name << " batch=" << batch << " maxthreads=" << maxThreads
                              << " seconds=" << seconds
                              << " us_per_pair=" << seconds / static_cast<double>(batch) * 1.0e6
                              << " pairs_per_second=" << static_cast<double>(batch) / seconds
                              << " quadrature_nodes_per_slot=" << 64ull * (12ull * 12ull * 12ull + 16ull * 16ull * 16ull)
                              << "\n";
                }
            });
    }
} // namespace

int main(int argc, char** argv)
{
    std::string const mode = argc > 1 ? argv[1] : "bench";
    std::string tablePath;
    int ai = 2;
    if(mode == "bench-table")
    {
        if(argc <= 2)
        {
            std::cerr << "usage: exactBenchmark bench-table <table.bin> [backend] [batch] [maxThreads]\n";
            return 2;
        }
        tablePath = argv[ai++];
    }
    std::string const backend = argc > ai ? argv[ai++] : "cuda";
    std::size_t const batch = argc > ai ? static_cast<std::size_t>(std::stoull(argv[ai++])) : 64u;
    std::size_t const maxThreads
        = argc > ai ? static_cast<std::size_t>(std::stoull(argv[ai])) : std::numeric_limits<std::size_t>::max();
    if(mode == "check")
        return runCheck(backend);
    if(mode == "bench-table")
    {
        runBenchmarkTable(tablePath, backend, batch, maxThreads);
        return EXIT_SUCCESS;
    }
    runBenchmark(backend, batch, maxThreads);
    return EXIT_SUCCESS;
}
