/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Device evaluation of the far-field interaction lists (PROTOTYPE).
 *
 * One block per tile, one thread per staggered output, exactly like
 * ``TileWideField.hpp``.  Each thread walks its tile's far-node list, evaluates
 * every accepted node once with the Cartesian multipole expansion, then walks
 * the near-source list and evaluates those sources with the exact point field.
 * The two lists are disjoint and exhaustive (see ``Tree.hpp``), so every source
 * contributes exactly once to every tile.
 */

#include "Multipole.hpp"
#include "Tree.hpp"

#include "../PointField.hpp"
#include "../TileGeometry.hpp"

#include <alpaka/alpaka.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace picongpu::particles::neuralBunch::farfield
{
    namespace detail
    {
        inline constexpr std::uint32_t blockSize = 256u;

        template<int Order>
        struct FarFieldKernel
        {
            Vec3d cells{};
            std::array<int, 3> offset{};
            std::array<int, 3> tileCounts{};

            ALPAKA_FN_ACC void operator()(
                auto const& acc,
                auto nodeCenter,
                auto nodeVelocity,
                auto nodeMoments,
                auto farOffsets,
                auto farNodes,
                auto nearOffsets,
                auto nearSources,
                auto positions,
                auto velocities,
                auto charges,
                auto tileValues) const
            {
                auto const blockIdx = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::blocks);
                auto const threadIdx = acc.getIdxWithin(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads);
                std::size_t const globalBlock = static_cast<std::size_t>(blockIdx[0]);
                std::size_t const blocksPerTile
                    = (static_cast<std::size_t>(tileGeometry::outputDimension) + blockSize - 1u) / blockSize;
                std::size_t const tileIndex = globalBlock / blocksPerTile;
                std::size_t const subBlock = globalBlock % blocksPerTile;
                std::size_t const thread = static_cast<std::size_t>(threadIdx[0]);
                auto const countX = static_cast<std::size_t>(tileCounts[0]);
                auto const countY = static_cast<std::size_t>(tileCounts[1]);
                auto const countZ = static_cast<std::size_t>(tileCounts[2]);
                if(tileIndex >= countX * countY * countZ)
                    return;
                std::size_t const slot = subBlock * blockSize + thread;
                if(slot >= static_cast<std::size_t>(tileGeometry::outputDimension))
                    return;

                auto const i0 = tileIndex % countX;
                auto const j0 = (tileIndex / countX) % countY;
                auto const k0 = tileIndex / (countX * countY);
                Vec3d const originDx{
                    static_cast<double>(offset[0] + static_cast<int>(i0) * tileGeometry::tileNx) * cells.x,
                    static_cast<double>(offset[1] + static_cast<int>(j0) * tileGeometry::tileNy) * cells.y,
                    static_cast<double>(offset[2] + static_cast<int>(k0) * tileGeometry::tileNz) * cells.z};
                Vec3d const stagger = staggeredPosition(slot, cells);
                Vec3d const target{originDx.x + stagger.x, originDx.y + stagger.y, originDx.z + stagger.z};
                int const component = static_cast<int>(slot % 6u);

                constexpr int count = momentCount(Order);
                double sum = 0.0;

                for(std::uint32_t entry = farOffsets[alpaka::Vec<std::size_t, 1>{tileIndex}];
                    entry < farOffsets[alpaka::Vec<std::size_t, 1>{tileIndex + 1u}];
                    ++entry)
                {
                    std::size_t const node = farNodes[alpaka::Vec<std::size_t, 1>{entry}];
                    Vec3d const center{
                        nodeCenter[alpaka::Vec<std::size_t, 2>{node, 0u}],
                        nodeCenter[alpaka::Vec<std::size_t, 2>{node, 1u}],
                        nodeCenter[alpaka::Vec<std::size_t, 2>{node, 2u}]};
                    Vec3d const u{
                        nodeVelocity[alpaka::Vec<std::size_t, 2>{node, 0u}],
                        nodeVelocity[alpaka::Vec<std::size_t, 2>{node, 1u}],
                        nodeVelocity[alpaka::Vec<std::size_t, 2>{node, 2u}]};
                    std::array<double, count> moments{};
                    for(int index = 0; index < count; ++index)
                        moments[index] = nodeMoments[alpaka::Vec<std::size_t, 2>{node, static_cast<std::size_t>(index)}];
                    Vec3d E{};
                    Vec3d B{};
                    multipoleFarField<Order>(
                        Vec3d{target.x - center.x, target.y - center.y, target.z - center.z},
                        u,
                        moments.data(),
                        E,
                        B);
                    sum += component < 3 ? (component == 0 ? E.x : (component == 1 ? E.y : E.z))
                                         : (component == 3 ? B.x : (component == 4 ? B.y : B.z));
                }

                for(std::uint32_t entry = nearOffsets[alpaka::Vec<std::size_t, 1>{tileIndex}];
                    entry < nearOffsets[alpaka::Vec<std::size_t, 1>{tileIndex + 1u}];
                    ++entry)
                {
                    std::size_t const source = nearSources[alpaka::Vec<std::size_t, 1>{entry}];
                    Vec3d const relativeDx{
                        positions[alpaka::Vec<std::size_t, 2>{source, 0u}] - originDx.x,
                        positions[alpaka::Vec<std::size_t, 2>{source, 1u}] - originDx.y,
                        positions[alpaka::Vec<std::size_t, 2>{source, 2u}] - originDx.z};
                    Vec3d const u{
                        velocities[alpaka::Vec<std::size_t, 2>{source, 0u}],
                        velocities[alpaka::Vec<std::size_t, 2>{source, 1u}],
                        velocities[alpaka::Vec<std::size_t, 2>{source, 2u}]};
                    Vec3d const R{
                        stagger.x - relativeDx.x,
                        stagger.y - relativeDx.y,
                        stagger.z - relativeDx.z};
                    auto const field = pointFieldDx(R, u);
                    double const charge = charges[alpaka::Vec<std::size_t, 1>{source}];
                    sum += charge
                           * (component < 3 ? (component == 0 ? field.E.x : (component == 1 ? field.E.y : field.E.z))
                                            : (component == 3 ? field.B.x
                                                              : (component == 4 ? field.B.y : field.B.z)));
                }

                tileValues[alpaka::Vec<std::size_t, 2>{tileIndex, slot}] += static_cast<float>(sum);
            }
        };

        template<typename TQueue, typename TExec, int Order, typename... TArgs>
        void dispatch(
            TQueue& queue,
            TExec exec,
            Vec3d const& cells,
            std::array<int, 3> const& offset,
            std::array<int, 3> const& tileCounts,
            TArgs&... args)
        {
            auto const tileCount = static_cast<std::uint64_t>(tileCounts[0]) * static_cast<std::uint64_t>(tileCounts[1])
                                   * static_cast<std::uint64_t>(tileCounts[2]);
            if(tileCount == 0u)
                return;
            using Vec = alpaka::Vec<std::uint32_t, 1>;
            auto const blocksPerTile
                = (static_cast<std::uint64_t>(tileGeometry::outputDimension) + blockSize - 1u) / blockSize;
            queue.enqueue(
                alpaka::onHost::FrameSpec{
                    Vec{static_cast<std::uint32_t>(tileCount * blocksPerTile)},
                    Vec{blockSize},
                    exec},
                alpaka::KernelBundle{FarFieldKernel<Order>{cells, offset, tileCounts}, args...});
        }
    } // namespace detail

    /** Enqueue the far-field evaluation for all tiles of one chunk.
     *
     * ``order`` must be in [0, maxEvaluationOrder]; values above are clamped.
     */
    template<typename TQueue, typename TExec, typename TTileValues>
    void enqueueFarField(
        TQueue& queue,
        TExec exec,
        auto const& nodeCenter,
        auto const& nodeVelocity,
        auto const& nodeMoments,
        auto const& farOffsets,
        auto const& farNodes,
        auto const& nearOffsets,
        auto const& nearSources,
        auto const& positions,
        auto const& velocities,
        auto const& charges,
        Vec3d const& cells,
        std::array<int, 3> const& offset,
        std::array<int, 3> const& tileCounts,
        int order,
        TTileValues& tileValues)
    {
        if(order < 0)
            order = 0;
        if(order > maxEvaluationOrder)
            order = maxEvaluationOrder;
        switch(order)
        {
        case 0:
            detail::dispatch<decltype(queue), decltype(exec), 0>(
                queue,
                exec,
                cells,
                offset,
                tileCounts,
                nodeCenter,
                nodeVelocity,
                nodeMoments,
                farOffsets,
                farNodes,
                nearOffsets,
                nearSources,
                positions,
                velocities,
                charges,
                tileValues);
            break;
        case 1:
            detail::dispatch<decltype(queue), decltype(exec), 1>(
                queue,
                exec,
                cells,
                offset,
                tileCounts,
                nodeCenter,
                nodeVelocity,
                nodeMoments,
                farOffsets,
                farNodes,
                nearOffsets,
                nearSources,
                positions,
                velocities,
                charges,
                tileValues);
            break;
        case 2:
            detail::dispatch<decltype(queue), decltype(exec), 2>(
                queue,
                exec,
                cells,
                offset,
                tileCounts,
                nodeCenter,
                nodeVelocity,
                nodeMoments,
                farOffsets,
                farNodes,
                nearOffsets,
                nearSources,
                positions,
                velocities,
                charges,
                tileValues);
            break;
        default:
            detail::dispatch<decltype(queue), decltype(exec), 3>(
                queue,
                exec,
                cells,
                offset,
                tileCounts,
                nodeCenter,
                nodeVelocity,
                nodeMoments,
                farOffsets,
                farNodes,
                nearOffsets,
                nearSources,
                positions,
                velocities,
                charges,
                tileValues);
            break;
        }
    }

} // namespace picongpu::particles::neuralBunch::farfield
