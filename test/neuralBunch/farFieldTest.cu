/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Standalone far-field proof for the neural-bunch tile field.
 *
 * Two modes:
 *
 *   neuralBunchFarFieldTest                 small host case: direct vs far
 *                                           per-component relative L2 at
 *                                           expansion orders 2/4/6, plus the
 *                                           interaction counts.
 *   neuralBunchFarFieldTest bench [S]       device timing of the direct tile
 *                                           kernel against the far evaluator
 *                                           (default S=100000, 128^3 grid).
 *
 * This target is NOT wired into NeuralBunchInit.x.cpp; it only exercises the
 * new ``farfield`` header and the existing point-field kernel.
 */

#include "picongpu/particles/neuralBunch/Backend.hpp"
#include "picongpu/particles/neuralBunch/TileWideField.hpp"
#include "picongpu/particles/neuralBunch/farfield/FarField.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace nb = picongpu::particles::neuralBunch;
namespace ff = picongpu::particles::neuralBunch::farfield;

namespace
{
    int failures = 0;

    void check(bool const condition, char const* message)
    {
        if(!condition)
        {
            std::cerr << "FAIL: " << message << '\n';
            ++failures;
        }
    }

    struct CloudBuffers
    {
        std::vector<double> positions;
        std::vector<double> velocities;
        std::vector<double> charges;

        ff::SourceCloud cloud() const
        {
            return {positions.data(), velocities.data(), charges.data(), charges.size()};
        }
    };

    /** Compact Gaussian bunch at the domain centre plus a small velocity spread. */
    CloudBuffers makeBunch(
        std::size_t count,
        std::array<int, 3> const& tileCounts,
        double bunchRadius,
        double velocitySpread,
        std::uint32_t seed)
    {
        CloudBuffers buffers;
        buffers.positions.resize(3 * count);
        buffers.velocities.resize(3 * count);
        buffers.charges.resize(count);
        std::mt19937 rng(seed);
        std::normal_distribution<double> gaussian(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        double const extent[3]{
            0.5 * tileCounts[0] * nb::tileGeometry::tileNx,
            0.5 * tileCounts[1] * nb::tileGeometry::tileNy,
            0.5 * tileCounts[2] * nb::tileGeometry::tileNz};
        for(std::size_t i = 0; i < count; ++i)
        {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            do
            {
                x = gaussian(rng);
                y = gaussian(rng);
                z = gaussian(rng);
            } while(x * x + y * y + z * z > 9.0);
            buffers.positions[3 * i + 0] = extent[0] + bunchRadius * x;
            buffers.positions[3 * i + 1] = extent[1] + bunchRadius * y;
            buffers.positions[3 * i + 2] = extent[2] + bunchRadius * z;
            buffers.velocities[3 * i + 0] = 0.5 + velocitySpread * uniform(rng);
            buffers.velocities[3 * i + 1] = 0.2 + velocitySpread * uniform(rng);
            buffers.velocities[3 * i + 2] = -0.1 + velocitySpread * uniform(rng);
            buffers.charges[i] = (i % 7u == 0u) ? -0.5 : -1.0;
        }
        return buffers;
    }

    struct Metric
    {
        double groupE = 0.0;
        double groupB = 0.0;
        double component[6]{};
    };

    /** field_gate.py definition: per group, ||cand-ref||_2 / max(||ref||_2, S0),
     * S0 = sqrt(256) * max_point ||G_point||. Reported as the maximum over tiles.
     */
    Metric compareFields(std::vector<float> const& candidate, std::vector<float> const& reference, std::size_t tiles)
    {
        constexpr std::size_t slots = nb::tileGeometry::outputDimension;
        constexpr std::size_t points = slots / 6u;
        Metric result;
        for(std::size_t tile = 0; tile < tiles; ++tile)
        {
            double error2[2]{0.0, 0.0};
            double reference2[2]{0.0, 0.0};
            double errorComponent[6]{};
            double maximumPoint[2]{0.0, 0.0};
            for(std::size_t point = 0; point < points; ++point)
            {
                double pointNorm[2]{0.0, 0.0};
                for(std::size_t component = 0; component < 6; ++component)
                {
                    std::size_t const index = tile * slots + point * 6u + component;
                    double const value = candidate[index];
                    double const target = reference[index];
                    double const difference = value - target;
                    std::size_t const group = component / 3u;
                    error2[group] += difference * difference;
                    reference2[group] += target * target;
                    errorComponent[component] += difference * difference;
                    pointNorm[group] += target * target;
                }
                for(std::size_t group = 0; group < 2; ++group)
                    maximumPoint[group] = std::max(maximumPoint[group], std::sqrt(pointNorm[group]));
            }
            double denominator[2];
            for(std::size_t group = 0; group < 2; ++group)
            {
                double const s0 = std::sqrt(static_cast<double>(points)) * maximumPoint[group];
                denominator[group] = std::max(std::sqrt(reference2[group]), s0);
            }
            result.groupE = std::max(
                result.groupE,
                denominator[0] > 0.0 ? std::sqrt(error2[0]) / denominator[0] : 0.0);
            result.groupB = std::max(
                result.groupB,
                denominator[1] > 0.0 ? std::sqrt(error2[1]) / denominator[1] : 0.0);
            for(std::size_t component = 0; component < 6; ++component)
            {
                std::size_t const group = component / 3u;
                double const value
                    = denominator[group] > 0.0 ? std::sqrt(errorComponent[component]) / denominator[group] : 0.0;
                result.component[component] = std::max(result.component[component], value);
            }
        }
        return result;
    }

    char const* componentName(std::size_t component)
    {
        static constexpr char const* names[6]{"Ex", "Ey", "Ez", "cBx", "cBy", "cBz"};
        return names[component];
    }

    template<int Order>
    std::vector<float> evaluateFarHost(
        ff::InteractionLists const& lists,
        ff::SourceCloud const& cloud,
        nb::Vec3d const& cells,
        std::array<int, 3> const& offset,
        std::array<int, 3> const& tileCounts)
    {
        std::size_t const tiles = lists.statistics.tileCount;
        std::vector<float> values(tiles * nb::tileGeometry::outputDimension, 0.0f);
        for(std::size_t tile = 0; tile < tiles; ++tile)
            ff::evaluateTileFar<Order>(
                lists,
                cloud,
                cells,
                offset,
                tileCounts,
                tile,
                values.data() + tile * nb::tileGeometry::outputDimension);
        return values;
    }

    void accuracyCase()
    {
        std::size_t const sourceCount = 400;
        std::array<int, 3> const tileCounts{2, 2, 2};
        std::array<int, 3> const offset{0, 0, 0};
        nb::Vec3d const cells{1.0, 1.0, 1.0};
        auto const buffers = makeBunch(sourceCount, tileCounts, 2.0, 0.02, 20261102u);
        auto const cloud = buffers.cloud();
        std::size_t const tiles = static_cast<std::size_t>(tileCounts[0]) * tileCounts[1] * tileCounts[2];
        std::size_t const slots = nb::tileGeometry::outputDimension;

        std::vector<float> direct(tiles * slots, 0.0f);
        for(std::size_t tile = 0; tile < tiles; ++tile)
            ff::evaluateTileDirect(
                cloud,
                cells,
                offset,
                tileCounts,
                tile,
                direct.data() + tile * slots);

        ff::TreeParameters parameters;
        parameters.leafCapacity = 16;
        parameters.separationFactor = 2.0;
        auto const lists = ff::buildInteractionLists(cloud, parameters, cells, offset, tileCounts);

        std::cout << "sources " << sourceCount << " tiles " << tiles << " slots " << slots << " nodes "
                  << lists.nodes.size() << '\n';
        std::cout << "far_node_tile_interactions " << lists.statistics.farNodeInteractions
                  << " near_source_pairs " << lists.statistics.nearSourceEvaluations
                  << " direct_source_pairs " << lists.statistics.directSourceEvaluations
                  << " max_far_nodes_per_tile " << lists.statistics.maxFarNodesPerTile
                  << " max_near_sources_per_tile " << lists.statistics.maxNearSourcesPerTile << '\n';

        double const gate = 1.0e-3;
        for(int const order : {2, 4, 6})
        {
            std::vector<float> far;
            if(order == 2)
                far = evaluateFarHost<2>(lists, cloud, cells, offset, tileCounts);
            else if(order == 4)
                far = evaluateFarHost<4>(lists, cloud, cells, offset, tileCounts);
            else
                far = evaluateFarHost<6>(lists, cloud, cells, offset, tileCounts);
            auto const metric = compareFields(far, direct, tiles);
            std::cout << "order " << order << " relL2_E " << metric.groupE << " relL2_cB " << metric.groupB;
            for(std::size_t component = 0; component < 6; ++component)
                std::cout << ' ' << componentName(component) << "_relL2 " << metric.component[component];
            std::cout << " work_units " << lists.statistics.totalWorkUnits(order)
                      << " gate_pass " << ((metric.groupE <= gate && metric.groupB <= gate) ? 1 : 0) << '\n';
        }
        check(lists.statistics.farNodeInteractions > 0, "at least one far interaction");
        check(lists.statistics.nearSourceEvaluations < lists.statistics.directSourceEvaluations, "near is a subset");
    }

    template<int Order>
    struct FarTileKernel
    {
        nb::Vec3d cells;
        std::array<int, 3> offset;
        std::array<int, 3> tileCounts;
        std::size_t tileCount;
        std::size_t slotCount;

        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            auto nodes,
            auto farOffsets,
            auto farIndices,
            auto nearOffsets,
            auto nearIndices,
            auto positions,
            auto velocities,
            auto charges,
            auto tileValues) const
        {
            std::size_t const total = tileCount * slotCount;
            for(auto index : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{total}}))
            {
                std::size_t const flat = static_cast<std::size_t>(index[0]);
                std::size_t const tile = flat / slotCount;
                std::size_t const slot = flat % slotCount;
                auto const i0 = static_cast<int>(tile % static_cast<std::size_t>(tileCounts[0]));
                auto const j0 = static_cast<int>(
                    (tile / static_cast<std::size_t>(tileCounts[0])) % static_cast<std::size_t>(tileCounts[1]));
                auto const k0 = static_cast<int>(
                    tile / (static_cast<std::size_t>(tileCounts[0])
                            * static_cast<std::size_t>(tileCounts[1])));
                nb::Vec3d const origin{
                    static_cast<double>(offset[0] + i0 * nb::tileGeometry::tileNx),
                    static_cast<double>(offset[1] + j0 * nb::tileGeometry::tileNy) * cells.y,
                    static_cast<double>(offset[2] + k0 * nb::tileGeometry::tileNz) * cells.z};
                nb::Vec3d const staggered = nb::staggeredPosition(slot, cells);
                int const component = static_cast<int>(slot % 6u);
                double sum = 0.0;
                int const farBegin = farOffsets[alpaka::Vec<std::size_t, 1>{tile}];
                int const farEnd = farOffsets[alpaka::Vec<std::size_t, 1>{tile + 1u}];
                for(int k = farBegin; k < farEnd; ++k)
                {
                    int const node = farIndices[alpaka::Vec<std::size_t, 1>{static_cast<std::size_t>(k)}];
                    sum += ff::nodeFieldComponent<Order>(
                        nodes[alpaka::Vec<std::size_t, 1>{static_cast<std::size_t>(node)}],
                        origin.x + staggered.x,
                        origin.y + staggered.y,
                        origin.z + staggered.z,
                        component);
                }
                int const nearBegin = nearOffsets[alpaka::Vec<std::size_t, 1>{tile}];
                int const nearEnd = nearOffsets[alpaka::Vec<std::size_t, 1>{tile + 1u}];
                for(int k = nearBegin; k < nearEnd; ++k)
                {
                    std::size_t const source
                        = nearIndices[alpaka::Vec<std::size_t, 1>{static_cast<std::size_t>(k)}];
                    nb::Vec3d const sourceR{
                        positions[alpaka::Vec<std::size_t, 2>{source, 0u}] - origin.x,
                        positions[alpaka::Vec<std::size_t, 2>{source, 1u}] - origin.y,
                        positions[alpaka::Vec<std::size_t, 2>{source, 2u}] - origin.z};
                    nb::Vec3d const u{
                        velocities[alpaka::Vec<std::size_t, 2>{source, 0u}],
                        velocities[alpaka::Vec<std::size_t, 2>{source, 1u}],
                        velocities[alpaka::Vec<std::size_t, 2>{source, 2u}]};
                    sum += charges[alpaka::Vec<std::size_t, 1>{source}] * nb::pointFieldSlot(slot, sourceR, u, cells);
                }
                tileValues[alpaka::Vec<std::size_t, 2>{tile, slot}] += static_cast<float>(sum);
            }
        }
    };

    template<int Order>
    void enqueueFarTiles(
        auto& queue,
        auto const& exec,
        nb::Vec3d const& cells,
        std::array<int, 3> const& offset,
        std::array<int, 3> const& tileCounts,
        std::size_t tileCount,
        auto const& deviceNodes,
        auto const& deviceFarOffsets,
        auto const& deviceFarIndices,
        auto const& deviceNearOffsets,
        auto const& deviceNearIndices,
        auto const& devicePositions,
        auto const& deviceVelocities,
        auto const& deviceCharges,
        auto& deviceTileValues)
    {
        std::size_t const total = tileCount * nb::tileGeometry::outputDimension;
        using Vec = alpaka::Vec<std::uint32_t, 1>;
        queue.enqueue(
            alpaka::onHost::FrameSpec{Vec{static_cast<std::uint32_t>(total)}, Vec{256u}, exec},
            alpaka::KernelBundle{
                FarTileKernel<Order>{cells, offset, tileCounts, tileCount, nb::tileGeometry::outputDimension},
                deviceNodes,
                deviceFarOffsets,
                deviceFarIndices,
                deviceNearOffsets,
                deviceNearIndices,
                devicePositions,
                deviceVelocities,
                deviceCharges,
                deviceTileValues});
    }

    double now()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void benchmarkCase(std::size_t sourceCount, std::array<int, 3> const& tileCounts)
    {
        std::array<int, 3> const offset{0, 0, 0};
        nb::Vec3d const cells{1.0, 1.0, 1.0};
        std::size_t const tiles = static_cast<std::size_t>(tileCounts[0]) * tileCounts[1] * tileCounts[2];
        std::size_t const slots = nb::tileGeometry::outputDimension;

        auto const buffers = makeBunch(sourceCount, tileCounts, 2.0, 0.02, 424242u);
        auto const cloud = buffers.cloud();
        ff::TreeParameters parameters;
        parameters.leafCapacity = 16;
        parameters.separationFactor = 2.0;
        double const buildStart = now();
        auto const lists = ff::buildInteractionLists(cloud, parameters, cells, offset, tileCounts);
        double const buildSeconds = now() - buildStart;
        std::cout << "bench_sources " << sourceCount << " tiles " << tiles << " nodes " << lists.nodes.size()
                  << " far_node_tile_interactions " << lists.statistics.farNodeInteractions
                  << " near_source_pairs " << lists.statistics.nearSourceEvaluations
                  << " direct_source_pairs " << lists.statistics.directSourceEvaluations
                  << " max_far_nodes_per_tile " << lists.statistics.maxFarNodesPerTile
                  << " max_near_sources_per_tile " << lists.statistics.maxNearSourcesPerTile
                  << " host_build_seconds " << buildSeconds << '\n';

        nb::dispatchBackend(
            "cuda",
            [&](auto& device, auto const& exec)
            {
                auto queue = device.makeQueue();
                auto hostPositions = alpaka::onHost::allocHost<double>(alpaka::Vec{sourceCount, std::size_t{3}});
                auto hostVelocities = alpaka::onHost::allocHost<double>(alpaka::Vec{sourceCount, std::size_t{3}});
                auto hostCharges = alpaka::onHost::allocHost<double>(alpaka::Vec{sourceCount});
                for(std::size_t i = 0; i < sourceCount; ++i)
                {
                    for(int axis = 0; axis < 3; ++axis)
                    {
                        hostPositions[alpaka::Vec<std::size_t, 2>{i, static_cast<std::size_t>(axis)}]
                            = buffers.positions[3 * i + axis];
                        hostVelocities[alpaka::Vec<std::size_t, 2>{i, static_cast<std::size_t>(axis)}]
                            = buffers.velocities[3 * i + axis];
                    }
                    hostCharges[alpaka::Vec<std::size_t, 1>{i}] = buffers.charges[i];
                }
                auto devicePositions = alpaka::onHost::allocLike(device, hostPositions);
                auto deviceVelocities = alpaka::onHost::allocLike(device, hostVelocities);
                auto deviceCharges = alpaka::onHost::allocLike(device, hostCharges);
                alpaka::onHost::memcpy(queue, devicePositions, hostPositions);
                alpaka::onHost::memcpy(queue, deviceVelocities, hostVelocities);
                alpaka::onHost::memcpy(queue, deviceCharges, hostCharges);
                hostPositions.keepAlive(queue);
                hostVelocities.keepAlive(queue);
                hostCharges.keepAlive(queue);

                auto hostNodes = alpaka::onHost::allocHost<ff::NodeAggregate>(alpaka::Vec{lists.nodes.size()});
                for(std::size_t i = 0; i < lists.nodes.size(); ++i)
                    hostNodes[alpaka::Vec<std::size_t, 1>{i}] = lists.nodes[i];
                auto deviceNodes = alpaka::onHost::allocLike(device, hostNodes);
                alpaka::onHost::memcpy(queue, deviceNodes, hostNodes);
                hostNodes.keepAlive(queue);

                auto hostFarOffsets = alpaka::onHost::allocHost<int>(alpaka::Vec{lists.farOffsets.size()});
                auto hostFarIndices = alpaka::onHost::allocHost<int>(alpaka::Vec{lists.farNodes.size()});
                auto hostNearOffsets = alpaka::onHost::allocHost<int>(alpaka::Vec{lists.nearOffsets.size()});
                auto hostNearIndices
                    = alpaka::onHost::allocHost<std::uint32_t>(alpaka::Vec{lists.nearSources.size()});
                for(std::size_t i = 0; i < lists.farOffsets.size(); ++i)
                    hostFarOffsets[alpaka::Vec<std::size_t, 1>{i}] = lists.farOffsets[i];
                for(std::size_t i = 0; i < lists.farNodes.size(); ++i)
                    hostFarIndices[alpaka::Vec<std::size_t, 1>{i}] = lists.farNodes[i];
                for(std::size_t i = 0; i < lists.nearOffsets.size(); ++i)
                    hostNearOffsets[alpaka::Vec<std::size_t, 1>{i}] = lists.nearOffsets[i];
                for(std::size_t i = 0; i < lists.nearSources.size(); ++i)
                    hostNearIndices[alpaka::Vec<std::size_t, 1>{i}] = lists.nearSources[i];
                auto deviceFarOffsets = alpaka::onHost::allocLike(device, hostFarOffsets);
                auto deviceFarIndices = alpaka::onHost::allocLike(device, hostFarIndices);
                auto deviceNearOffsets = alpaka::onHost::allocLike(device, hostNearOffsets);
                auto deviceNearIndices = alpaka::onHost::allocLike(device, hostNearIndices);
                alpaka::onHost::memcpy(queue, deviceFarOffsets, hostFarOffsets);
                alpaka::onHost::memcpy(queue, deviceFarIndices, hostFarIndices);
                alpaka::onHost::memcpy(queue, deviceNearOffsets, hostNearOffsets);
                alpaka::onHost::memcpy(queue, deviceNearIndices, hostNearIndices);
                hostFarOffsets.keepAlive(queue);
                hostFarIndices.keepAlive(queue);
                hostNearOffsets.keepAlive(queue);
                hostNearIndices.keepAlive(queue);

                auto deviceTileValues
                    = alpaka::onHost::alloc<float>(device, alpaka::Vec{tiles, slots});
                auto deviceBandCounts
                    = alpaka::onHost::alloc<std::uint32_t>(device, alpaka::Vec{tiles, std::size_t{3}});
                auto hostZero = alpaka::onHost::allocHost<float>(alpaka::Vec{tiles, slots});
                auto hostTileValues = alpaka::onHost::allocHost<float>(alpaka::Vec{tiles, slots});
                for(std::size_t tile = 0; tile < tiles; ++tile)
                    for(std::size_t slot = 0; slot < slots; ++slot)
                        hostZero[alpaka::Vec<std::size_t, 2>{tile, slot}] = 0.0f;

                auto runDirect = [&]()
                {
                    alpaka::onHost::memcpy(queue, deviceTileValues, hostZero);
                    alpaka::onHost::wait(queue);
                    double const start = now();
                    nb::enqueueTileWidePointField(
                        queue,
                        exec,
                        devicePositions,
                        deviceVelocities,
                        deviceCharges,
                        sourceCount,
                        cells,
                        offset,
                        tileCounts,
                        0.0,
                        4.0,
                        48.0,
                        false,
                        deviceTileValues,
                        deviceBandCounts);
                    alpaka::onHost::wait(queue);
                    return now() - start;
                };

                auto fetch = [&]()
                {
                    alpaka::onHost::memcpy(queue, hostTileValues, deviceTileValues);
                    alpaka::onHost::wait(queue);
                    std::vector<float> values(tiles * slots);
                    for(std::size_t tile = 0; tile < tiles; ++tile)
                        for(std::size_t slot = 0; slot < slots; ++slot)
                            values[tile * slots + slot] = hostTileValues[alpaka::Vec<std::size_t, 2>{tile, slot}];
                    return values;
                };

                (void) runDirect();
                double directSeconds = 1.0e300;
                for(int rep = 0; rep < 2; ++rep)
                    directSeconds = std::min(directSeconds, runDirect());
                auto const direct = fetch();
                std::cout << "direct_seconds " << directSeconds << '\n';

                auto runFar = [&](auto orderTag)
                {
                    constexpr int order = decltype(orderTag)::value;
                    alpaka::onHost::memcpy(queue, deviceTileValues, hostZero);
                    alpaka::onHost::wait(queue);
                    double const start = now();
                    enqueueFarTiles<order>(
                        queue,
                        exec,
                        cells,
                        offset,
                        tileCounts,
                        tiles,
                        deviceNodes,
                        deviceFarOffsets,
                        deviceFarIndices,
                        deviceNearOffsets,
                        deviceNearIndices,
                        devicePositions,
                        deviceVelocities,
                        deviceCharges,
                        deviceTileValues);
                    alpaka::onHost::wait(queue);
                    return now() - start;
                };

                auto timeOrder = [&](auto orderTag)
                {
                    constexpr int order = decltype(orderTag)::value;
                    (void) runFar(orderTag);
                    double best = 1.0e300;
                    for(int rep = 0; rep < 2; ++rep)
                        best = std::min(best, runFar(orderTag));
                    auto const far = fetch();
                    auto const metric = compareFields(far, direct, tiles);
                    std::cout << "far_order " << order << " seconds " << best << " speedup " << directSeconds / best
                              << " relL2_E " << metric.groupE << " relL2_cB " << metric.groupB;
                    for(std::size_t component = 0; component < 6; ++component)
                        std::cout << ' ' << componentName(component) << "_relL2 " << metric.component[component];
                    std::cout << " work_units " << lists.statistics.totalWorkUnits(order) << '\n';
                };
                timeOrder(std::integral_constant<int, 2>{});
                timeOrder(std::integral_constant<int, 4>{});
            });
    }
} // namespace

int main(int argc, char** argv)
{
    bool bench = argc > 1 && std::string(argv[1]) == "bench";
    if(!bench)
    {
        accuracyCase();
        if(failures != 0)
        {
            std::cerr << failures << " check(s) failed\n";
            return EXIT_FAILURE;
        }
        std::cout << "neuralBunchFarFieldTest: OK\n";
        return EXIT_SUCCESS;
    }

    std::size_t const sources = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 100000u;
    std::array<int, 3> const tileCounts{16, 16, 32}; // 128 x 128 x 128 cells
    benchmarkCase(sources, tileCounts);
    return EXIT_SUCCESS;
}
