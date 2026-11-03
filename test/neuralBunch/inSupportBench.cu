/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Stationary in-support measurement bench for the exact device quadrature.
 *
 * Measures, on the device, the per-slot and per-(source, tile)-pair cost of the
 * near-exact pass for stationary (u = 0) sources whose position lies inside the
 * TSC support box (the "in-support" case). The same binary runs unchanged on the
 * production tree (where in-support uses the coarse/fine box+pyramid quadrature)
 * and on the analytic-subtraction tree (where in-support stationary uses the
 * surface integral + full-support remainder), so the two runs give the measured
 * before/after ratio without any inherited numbers.
 *
 * Modes:
 *   slot   [backend] [threads] [evals]   isolated in-support per-slot cost
 *   pair   [backend] [tiles]   production per-tile loop, one source per tile
 *   batch  [backend] [count] [batches] [maxthreads]
 *          single tile, count sources in 512-source slices (production enqueue)
 *   grid   [backend] [out.csv] host + device values on the azimuth/position grid
 *
 * All positions are in cell units with cells = (1, 1, 1) and the source at the
 * tile origin. The source position for the cost modes is (0.3, -0.2, 0.1), an
 * interior point (8 sub-boxes in the containing cell, the worst case for the
 * sub-box count).
 */

#include "picongpu/particles/neuralBunch/Backend.hpp"
#include "picongpu/particles/neuralBunch/ExactShapeField.hpp"
#include "picongpu/particles/neuralBunch/PointField.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace picongpu::particles::neuralBunch;

namespace
{
    Vec3d const kCostPos{0.3, -0.2, 0.1};

    double secondsBetween(std::chrono::steady_clock::time_point const& a,
                          std::chrono::steady_clock::time_point const& b)
    {
        return std::chrono::duration<double>(b - a).count();
    }

    /** Isolated per-slot cost: every thread evaluates in-support points in a loop. */
    struct SlotKernel
    {
        Vec3d pos;

        ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
        {
            auto const tid = static_cast<std::size_t>(alpaka::onAcc::getThreadIdx<0>(acc));
            auto const nThreads = static_cast<std::size_t>(alpaka::onAcc::getThreadCount<0>(acc));
            auto const evals = static_cast<std::size_t>(alpaka::onAcc::getGridBlockCount<0>(acc));
            auto const total = nThreads * evals;
            GaussTable const table = GaussTable::create();
            ExactSettings const settings;
            double accSum = 0.0;
            for(std::size_t i = tid; i < total; i += nThreads)
            {
                auto const result
                    = exactShapeFieldPoint<double>(pos, Vec3d{0.0, 0.0, 0.0}, Vec3d{1.0, 1.0, 1.0}, settings, table);
                accSum += result.E.x + result.E.y + result.E.z + result.B.x + result.B.y + result.B.z;
            }
            out[alpaka::Vec<std::size_t, 1>{tid}] = accSum;
        }
    };

    void runSlotBench(std::string const& backend, std::size_t threads, std::size_t evalsPerThread)
    {
        dispatchBackend(
            backend,
            [&](auto& device, auto const& exec)
            {
                auto queue = device.makeQueue();
                using Vec = alpaka::Vec<std::uint32_t, 1>;
                auto const out = alpaka::onHost::alloc<double>(device.getDevice(), alpaka::Vec{threads});
                std::size_t const repeats = 3u;
                double best = 1.0e30;
                for(std::size_t r = 0u; r < repeats; ++r)
                {
                    auto const t0 = std::chrono::steady_clock::now();
                    queue.enqueue(
                        alpaka::onHost::FrameSpec{Vec{evalsPerThread}, Vec{threads}, exec},
                        alpaka::KernelBundle{SlotKernel{kCostPos}, out});
                    alpaka::onHost::wait(queue);
                    best = std::min(best, secondsBetween(t0, std::chrono::steady_clock::now()));
                }
                double const totalEvals = static_cast<double>(threads) * static_cast<double>(evalsPerThread);
                std::cout << "slot pos=0.3,-0.2,0.1 threads=" << threads << " evals=" << totalEvals << " seconds="
                          << best << " us_per_slot=" << best / totalEvals * 1.0e6
                          << " slots_per_millisecond=" << totalEvals / (best * 1.0e3) << "\n";
            });
    }

    /** Production per-tile loop with one source per tile (exactly the NeuralBunchInit pattern). */
    void runPairBench(std::string const& backend, std::size_t tiles, std::size_t maxThreads)
    {
        dispatchBackend(
            backend,
            [&](auto& device, auto const& exec)
            {
                auto queue = device.makeQueue();
                auto accumulator = alpaka::onHost::alloc<double>(device.getDevice(), alpaka::Vec{kTileSlotCount});
                enqueueExactZero(queue, exec, accumulator);
                alpaka::onHost::wait(queue);
                std::size_t const repeats = 3u;
                double best = 1.0e30;
                for(std::size_t r = 0u; r < repeats; ++r)
                {
                    auto const t0 = std::chrono::steady_clock::now();
                    for(std::size_t tile = 0u; tile < tiles; ++tile)
                    {
                        auto hostSources = alpaka::onHost::allocHost<double>(alpaka::Vec{std::size_t{1}, std::size_t{6}});
                        for(int c = 0; c < 3; ++c)
                        {
                            hostSources[alpaka::Vec<std::size_t, 2>{0, static_cast<std::size_t>(c)}]
                                = kCostPos[c];
                            hostSources[alpaka::Vec<std::size_t, 2>{0, static_cast<std::size_t>(3 + c)}] = 0.0;
                        }
                        auto hostWeights = alpaka::onHost::allocHost<double>(alpaka::Vec{std::size_t{1}});
                        hostWeights[alpaka::Vec<std::size_t, 1>{0}] = -1.0;
                        auto deviceSources = alpaka::onHost::allocLike(device.getDevice(), hostSources);
                        auto deviceWeights = alpaka::onHost::allocLike(device.getDevice(), hostWeights);
                        alpaka::onHost::memcpy(queue, deviceSources, hostSources);
                        alpaka::onHost::memcpy(queue, deviceWeights, hostWeights);
                        hostSources.keepAlive(queue);
                        hostWeights.keepAlive(queue);
                        enqueueExactShapeFieldTileAccumulate<double>(
                            queue,
                            exec,
                            deviceSources,
                            deviceWeights,
                            1u,
                            Vec3d{1.0, 1.0, 1.0},
                            ExactSettings{},
                            accumulator,
                            maxThreads);
                        alpaka::onHost::wait(queue);
                    }
                    best = std::min(best, secondsBetween(t0, std::chrono::steady_clock::now()));
                }
                std::cout << "pair tiles=" << tiles << " maxthreads=" << maxThreads << " seconds=" << best
                          << " ms_per_pair=" << best / static_cast<double>(tiles) * 1.0e3
                          << " pairs_per_second=" << static_cast<double>(tiles) / best << "\n";
            });
    }

    /** Single tile, count sources processed through the production 512-source slices. */
    void runBatchBench(std::string const& backend, std::size_t count, std::size_t batches, std::size_t maxThreads)
    {
        dispatchBackend(
            backend,
            [&](auto& device, auto const& exec)
            {
                auto queue = device.makeQueue();
                auto hostSources = alpaka::onHost::allocHost<double>(alpaka::Vec{count, std::size_t{6}});
                for(std::size_t i = 0u; i < count; ++i)
                    for(int c = 0; c < 3; ++c)
                    {
                        hostSources[alpaka::Vec<std::size_t, 2>{i, static_cast<std::size_t>(c)}] = kCostPos[c];
                        hostSources[alpaka::Vec<std::size_t, 2>{i, static_cast<std::size_t>(3 + c)}] = 0.0;
                    }
                auto hostWeights = alpaka::onHost::allocHost<double>(alpaka::Vec{count});
                for(std::size_t i = 0u; i < count; ++i)
                    hostWeights[alpaka::Vec<std::size_t, 1>{i}] = -1.0;
                auto deviceSources = alpaka::onHost::allocLike(device.getDevice(), hostSources);
                auto deviceWeights = alpaka::onHost::allocLike(device.getDevice(), hostWeights);
                alpaka::onHost::memcpy(queue, deviceSources, hostSources);
                alpaka::onHost::memcpy(queue, deviceWeights, hostWeights);
                hostSources.keepAlive(queue);
                hostWeights.keepAlive(queue);
                auto accumulator = alpaka::onHost::alloc<double>(device.getDevice(), alpaka::Vec{kTileSlotCount});
                enqueueExactZero(queue, exec, accumulator);
                alpaka::onHost::wait(queue);
                std::size_t const repeats = 3u;
                double best = 1.0e30;
                for(std::size_t r = 0u; r < repeats; ++r)
                {
                    auto const t0 = std::chrono::steady_clock::now();
                    for(std::size_t b = 0u; b < batches; ++b)
                    {
                        enqueueExactShapeFieldTileAccumulate<double>(
                            queue,
                            exec,
                            deviceSources,
                            deviceWeights,
                            count,
                            Vec3d{1.0, 1.0, 1.0},
                            ExactSettings{},
                            accumulator,
                            maxThreads);
                    }
                    alpaka::onHost::wait(queue);
                    best = std::min(best, secondsBetween(t0, std::chrono::steady_clock::now()));
                }
                double const pairs = static_cast<double>(batches) * static_cast<double>(count);
                std::cout << "batch count=" << count << " batches=" << batches << " maxthreads=" << maxThreads
                          << " pairs=" << pairs << " seconds=" << best
                          << " ms_per_pair=" << best / pairs * 1.0e3
                          << " pairs_per_second=" << pairs / best << "\n";
            });
    }

    struct GridPoint
    {
        std::string tag;
        double x, y, z;
    };

    std::vector<GridPoint> makeGrid()
    {
        std::vector<GridPoint> pts;
        auto add = [&](char const* tag, double x, double y, double z)
        { pts.push_back(GridPoint{tag, x, y, z}); };
        double const az[5]{0.0, 30.0, 45.0, 60.0, 90.0};
        double const rad[5]{0.25, 0.5, 0.75, 1.0, 1.25};
        char buf[64];
        for(double const r : rad)
        {
            for(double const aDeg : az)
            {
                double const a = aDeg * 3.14159265358979323846 / 180.0;
                std::snprintf(buf, sizeof(buf), "in_r%.2f_az%.0f", r, aDeg);
                add(buf, r * std::cos(a), r * std::sin(a), 0.0);
            }
            std::snprintf(buf, sizeof(buf), "in_r%.2f_z", r);
            add(buf, 0.0, 0.0, r);
            double const d = r / std::sqrt(3.0);
            std::snprintf(buf, sizeof(buf), "in_r%.2f_diag", r);
            add(buf, d, d, d);
        }
        add("in_center", 0.0, 0.0, 0.0);
        // On the support boundary (distance == 0, some coordinate at +-1.5).
        add("face_x0", 1.5, 0.0, 0.0);
        add("face_x_in", 1.5, 0.5, 0.3);
        add("face_x_neg", 1.5, -0.4, 0.7);
        add("face_xy", 1.5, 1.5, 0.3);
        add("face_xyn", 1.5, 1.5, -1.2);
        add("face_corner", 1.5, 1.5, 1.5);
        add("face_y_in", 0.5, 1.5, 0.3);
        add("face_z_in", 0.3, -1.5, 1.5);
        // Just outside the support (distance > 0): graded-panel path, unchanged.
        double const dOut[5]{0.01, 0.05, 0.1, 0.25, 0.5};
        for(double const d : dOut)
        {
            std::snprintf(buf, sizeof(buf), "out_ax_%.2f", d);
            add(buf, 1.5 + d, 0.0, 0.0);
            std::snprintf(buf, sizeof(buf), "out_in_%.2f", d);
            add(buf, 1.5 + d, 0.5, 0.3);
        }
        double const dOutCorner[3]{0.01, 0.1, 0.5};
        for(double const d : dOutCorner)
        {
            std::snprintf(buf, sizeof(buf), "out_corner_%.2f", d);
            add(buf, 1.5 + d, 1.5 + d, 1.5 + d);
        }
        return pts;
    }

    void runGrid(std::string const& backend, std::string const& outCsv)
    {
        auto const pts = makeGrid();
        std::vector<double> hostE(pts.size() * 3u, 0.0);
        std::vector<double> hostB(pts.size() * 3u, 0.0);
        {
            GaussTable const table = GaussTable::create();
            ExactSettings const settings;
            for(std::size_t i = 0u; i < pts.size(); ++i)
            {
                auto const result = exactShapeFieldPoint<double>(
                    Vec3d{pts[i].x, pts[i].y, pts[i].z},
                    Vec3d{0.0, 0.0, 0.0},
                    Vec3d{1.0, 1.0, 1.0},
                    settings,
                    table);
                hostE[i * 3u + 0u] = result.E.x;
                hostE[i * 3u + 1u] = result.E.y;
                hostE[i * 3u + 2u] = result.E.z;
                hostB[i * 3u + 0u] = result.B.x;
                hostB[i * 3u + 1u] = result.B.y;
                hostB[i * 3u + 2u] = result.B.z;
            }
        }
        std::vector<double> devE(pts.size() * 3u, 0.0);
        std::vector<double> devB(pts.size() * 3u, 0.0);
        std::vector<double> devErr(pts.size(), 0.0);
        std::vector<unsigned char> devIn(pts.size(), 0u);
        {
            std::vector<ExactSample> samples(pts.size());
            for(std::size_t i = 0u; i < pts.size(); ++i)
            {
                samples[i].R0 = {pts[i].x, pts[i].y, pts[i].z};
                samples[i].u = {0.0, 0.0, 0.0};
                samples[i].cells = {1.0, 1.0, 1.0};
            }
            dispatchBackend(
                backend,
                [&](auto& device, auto const& exec)
                {
                    auto queue = device.makeQueue();
                    auto const res = exactShapeFieldSamplesBatched<double>(queue, exec, samples);
                    for(std::size_t i = 0u; i < pts.size(); ++i)
                    {
                        for(int c = 0; c < 3; ++c)
                        {
                            devE[i * 3u + c] = res.E[i * 3u + c];
                            devB[i * 3u + c] = res.B[i * 3u + c];
                        }
                        devErr[i] = res.error[i];
                        devIn[i] = res.inSupport[i];
                    }
                });
        }
        std::ofstream out(outCsv);
        out << "tag,x,y,z,host_E_x,host_E_y,host_E_z,host_B_x,host_B_y,host_B_z,"
               "dev_E_x,dev_E_y,dev_E_z,dev_B_x,dev_B_y,dev_B_z,dev_error,dev_inSupport\n";
        for(std::size_t i = 0u; i < pts.size(); ++i)
        {
            out << pts[i].tag << "," << pts[i].x << "," << pts[i].y << "," << pts[i].z << ","
                << hostE[i * 3u + 0u] << "," << hostE[i * 3u + 1u] << "," << hostE[i * 3u + 2u] << ","
                << hostB[i * 3u + 0u] << "," << hostB[i * 3u + 1u] << "," << hostB[i * 3u + 2u] << ","
                << devE[i * 3u + 0u] << "," << devE[i * 3u + 1u] << "," << devE[i * 3u + 2u] << ","
                << devB[i * 3u + 0u] << "," << devB[i * 3u + 1u] << "," << devB[i * 3u + 2u] << ","
                << devErr[i] << "," << static_cast<int>(devIn[i]) << "\n";
        }
        std::cout << "grid points=" << pts.size() << " csv=" << outCsv << "\n";
    }
} // namespace

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <slot|pair|batch|grid> [backend] [args...]\n";
        return EXIT_FAILURE;
    }
    std::string const mode = argv[1];
    std::string const backend = argc > 2 ? argv[2] : "cuda";
    if(mode == "slot")
    {
        std::size_t const threads = argc > 3 ? std::stoull(argv[3]) : 1024u;
        std::size_t const evals = argc > 4 ? std::stoull(argv[4]) : 2048u;
        runSlotBench(backend, threads, evals);
    }
    else if(mode == "pair")
    {
        std::size_t const tiles = argc > 3 ? std::stoull(argv[3]) : 64u;
        std::size_t const maxThreads = argc > 4 ? std::stoull(argv[4]) : std::numeric_limits<std::size_t>::max();
        runPairBench(backend, tiles, maxThreads);
    }
    else if(mode == "batch")
    {
        std::size_t const count = argc > 3 ? std::stoull(argv[3]) : 512u;
        std::size_t const batches = argc > 4 ? std::stoull(argv[4]) : 8u;
        std::size_t const maxThreads = argc > 5 ? std::stoull(argv[5]) : std::numeric_limits<std::size_t>::max();
        runBatchBench(backend, count, batches, maxThreads);
    }
    else if(mode == "grid")
    {
        if(argc < 4)
        {
            std::cerr << "grid mode needs an output csv path\n";
            return EXIT_FAILURE;
        }
        runGrid(backend, argv[3]);
    }
    else
    {
        std::cerr << "unknown mode " << mode << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
