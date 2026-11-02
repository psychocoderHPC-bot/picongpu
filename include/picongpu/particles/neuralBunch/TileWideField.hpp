/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Tile-centric analytic point-field kernel.
 *
 * The legacy banded path classifies every (source, tile) pair on the host and assembles one source batch per
 * tile. That costs a few microseconds of host work per pair and dominates on large grids. This kernel moves the
 * whole loop to the device:
 *
 *  - one block per tile, one thread per (tile, staggered output) group;
 *  - each thread walks the chunk sources in ascending order, so every output accumulates deterministically;
 *  - the MODEL-FORMAT minimum staggered distance is evaluated on the device and drives the band counters and the
 *    optional near threshold, so no pair is classified on the host;
 *  - the validated ``pointFieldDx`` formula produces the normalised field and ``q/e = -weight`` weights it.
 *
 * Scratch is bounded by the chunk (the source arrays are uploaded once per chunk) and the tile accumulator.
 * Contributions are added to the caller's ``[tiles, 1536]`` device buffer, which the existing store kernel writes
 * into the PIConGPU field boxes.
 */

#include "Features.hpp"
#include "PointField.hpp"
#include "TileGeometry.hpp"

#include <alpaka/alpaka.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace picongpu::particles::neuralBunch
{
    namespace tileWideDetail
    {
        constexpr std::uint32_t blockSize = 256u;

        ALPAKA_FN_HOST_ACC inline double minimumAxisDistance(double sourceCells, double offset, int count)
        {
            double const target = sourceCells - offset;
            double index = alpaka::math::floor(target + 0.5);
            index = alpaka::math::min(alpaka::math::max(index, 0.0), static_cast<double>(count - 1));
            return target - index;
        }

        /** Exact MODEL-FORMAT minimum distance, in dx units, from a tile-relative source position. */
        ALPAKA_FN_HOST_ACC inline double minimumStaggeredDistance(Vec3d const& relativeDx, Vec3d const& cellsDx)
        {
            double const cell[3]{cellsDx.x, cellsDx.y, cellsDx.z};
            int const count[3]{tileGeometry::tileNx, tileGeometry::tileNy, tileGeometry::tileNz};
            double const offset[6][3]{
                {0.5, 0.0, 0.0},
                {0.0, 0.5, 0.0},
                {0.0, 0.0, 0.5},
                {0.0, 0.5, 0.5},
                {0.5, 0.0, 0.5},
                {0.5, 0.5, 0.0}};
            double const sourceDx[3]{relativeDx.x, relativeDx.y, relativeDx.z};
            double best2 = 1.0e300;
            for(int component = 0; component < 6; ++component)
            {
                double sum2 = 0.0;
                for(int axis = 0; axis < 3; ++axis)
                {
                    double const sourceCells = sourceDx[axis] / cell[axis];
                    double const diff = minimumAxisDistance(sourceCells, offset[component][axis], count[axis]);
                    double const diffDx = diff * cell[axis];
                    sum2 += diffDx * diffDx;
                }
                best2 = alpaka::math::min(best2, sum2);
            }
            return alpaka::math::sqrt(best2);
        }

        /** Cheap per-(source, tile) near test: the source lies inside the tile's cell box expanded by
         * @p threshold (dx units), i.e. ``relativeDx`` is within ``[-threshold, tileSize + threshold]`` per axis.
         *
         * This is a super-set of the exact ``minimumStaggeredDistance < threshold`` test (an L-infinity box around the
         * tile region versus an L2 ball around the nearest staggered point), so every target inside a source's TSC
         * support -- and more generally every pair the near band would flag -- is covered. The kernel uses this to
         * skip those (source, tile) pairs out of the singular point field, and the stage's per-tile exact-quadrature
         * near list uses the *identical* test, which is what keeps the point-field and exact partitions disjoint and
         * exhaustive (no pair is dropped, none is double-counted). The short-circuit form keeps the per-slot cost to a
         * couple of comparisons for the far pairs that dominate a production grid, so the default path keeps the
         * per-pair work of the analytic kernel (the exact test would be ~25x pricier when evaluated by all 1536
         * output threads). ``cellsDx`` is ``(1, a_y, a_z)`` as elsewhere in this file.
         */
        ALPAKA_FN_HOST_ACC inline bool sourceNearBox(Vec3d const& relativeDx, Vec3d const& cellsDx, double threshold)
        {
            // The near band is `threshold` cells of margin on every axis, in that axis's own cell units:
            // `threshold * cellsDx.a` dx on axis a (cellsDx = (1, a_y, a_z)). This keeps the TSC support
            // (half-width 1.5 cells/axis) inside the band for anisotropic cells: a fixed dx margin covers
            // only threshold/cellsDx.a cells on axis a, which under-covers the long axes (aspect 10 needs a
            // 1.5-cell y-margin of 15 dx, not 4). For isotropic cells (cellsDx == 1) this is exactly the old
            // fixed-dx box, so an explicitly set threshold is unchanged there.
            double const thrX = threshold * cellsDx.x;
            double const thrY = threshold * cellsDx.y;
            double const thrZ = threshold * cellsDx.z;
            double const maxX = static_cast<double>(tileGeometry::tileNx) * cellsDx.x;
            double const maxY = static_cast<double>(tileGeometry::tileNy) * cellsDx.y;
            double const maxZ = static_cast<double>(tileGeometry::tileNz) * cellsDx.z;
            return relativeDx.x >= -thrX && relativeDx.x <= maxX + thrX && relativeDx.y >= -thrY
                   && relativeDx.y <= maxY + thrY && relativeDx.z >= -thrZ
                   && relativeDx.z <= maxZ + thrZ;
        }

        struct TileWideKernel
        {
            Vec3d cellsDx{};
            std::array<int, 3> offset{};
            std::array<int, 3> tileCounts{};
            std::size_t sourceCount{};
            double nearThresholdCells{};
            double bandNearCells{};
            double bandFarCells{};
            bool computeBands{};

            ALPAKA_FN_ACC void operator()(
                auto const& acc,
                auto positions,
                auto velocities,
                auto charges,
                auto tileValues,
                auto bandCounts) const
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
                    static_cast<double>(offset[0] + static_cast<int>(i0) * tileGeometry::tileNx),
                    static_cast<double>(offset[1] + static_cast<int>(j0) * tileGeometry::tileNy) * cellsDx.y,
                    static_cast<double>(offset[2] + static_cast<int>(k0) * tileGeometry::tileNz) * cellsDx.z};
                Vec3d const stagger = staggeredPosition(slot, cellsDx);
                int const component = static_cast<int>(slot % 6u);

                double sum = 0.0;
                std::uint32_t nearCount = 0u;
                std::uint32_t midCount = 0u;
                std::uint32_t farCount = 0u;

                for(std::size_t source = 0u; source < sourceCount; ++source)
                {
                    auto const value
                        = [&](std::size_t column) { return positions[alpaka::Vec<std::size_t, 2>{source, column}]; };
                    Vec3d const relativeDx{value(0u) - originDx.x, value(1u) - originDx.y, value(2u) - originDx.z};
                    double const charge = charges[alpaka::Vec<std::size_t, 1>{source}];
                    bool const contributes = charge != 0.0;
                    // The band counters live in the single owning thread (subBlock == 0, thread == 0). With no near
                    // threshold nothing else consumes the exact minimum distance, so evaluate it in that thread only
                    // instead of in all 1536 output threads (the T8 trim).
                    double minimum = -1.0;
                    bool const bandOwner = subBlock == 0u && thread == 0u;
                    if(computeBands && bandOwner && contributes)
                        minimum = minimumStaggeredDistance(relativeDx, cellsDx);
                    if(computeBands && bandOwner && contributes)
                    {
                        if(minimum < bandNearCells)
                            ++nearCount;
                        else if(minimum < bandFarCells)
                            ++midCount;
                        else
                            ++farCount;
                    }
                    // Near exclusion. The singular point kernel is never correct inside a source's TSC support (NaN at
                    // coincidence, ~1-4x overestimate in the support), so when a near threshold is active the default
                    // path routes those (source, tile) pairs to the exact device quadrature instead. ``sourceNearBox``
                    // is a cheap per-slot super-set of the exact distance test and is evaluated by every output
                    // thread, so the pair is skipped out of the point field for all 1536 slots. The stage builds the
                    // per-tile near list with this identical test and feeds it to
                    // ``enqueueExactShapeFieldTileAccumulate``, which makes the point-field and exact partitions
                    // disjoint and exhaustive.
                    if(!contributes
                       || (nearThresholdCells > 0.0 && sourceNearBox(relativeDx, cellsDx, nearThresholdCells)))
                        continue;

                    Vec3d const u{
                        velocities[alpaka::Vec<std::size_t, 2>{source, 0u}],
                        velocities[alpaka::Vec<std::size_t, 2>{source, 1u}],
                        velocities[alpaka::Vec<std::size_t, 2>{source, 2u}]};
                    // Host pointFieldSlot uses R = x_output - x_source.
                    Vec3d const R{stagger.x - relativeDx.x, stagger.y - relativeDx.y, stagger.z - relativeDx.z};
                    auto const field = pointFieldDx(R, u);
                    double const contribution
                        = component < 3 ? (component == 0 ? field.E.x : (component == 1 ? field.E.y : field.E.z))
                                        : (component == 3 ? field.B.x : (component == 4 ? field.B.y : field.B.z));
                    sum += charge * contribution;
                }

                tileValues[alpaka::Vec<std::size_t, 2>{tileIndex, slot}] += static_cast<float>(sum);
                if(subBlock == 0u && thread == 0u)
                {
                    bandCounts[alpaka::Vec<std::size_t, 2>{tileIndex, 0u}] += nearCount;
                    bandCounts[alpaka::Vec<std::size_t, 2>{tileIndex, 1u}] += midCount;
                    bandCounts[alpaka::Vec<std::size_t, 2>{tileIndex, 2u}] += farCount;
                }
            }
        };
    } // namespace tileWideDetail

    /** Enqueue the tile-wide analytic point field for one chunk.
     *
     * @param positions ``[S,3]`` source positions in dx units (world coordinates divided by dx).
     * @param velocities ``[S,3]`` ``u = p/(m_e c)``.
     * @param charges ``[S]`` signed charge ``q/e = -weight``; zero-weight sources are skipped.
     * @param tileValues ``[T,1536]`` accumulator, added in place.
     * @param bandCounts ``[T,3]`` near/mid/far pair counters, added in place.
     */
    template<typename TQueue, typename TExec>
    void enqueueTileWidePointField(
        TQueue& queue,
        TExec exec,
        auto const& positions,
        auto const& velocities,
        auto const& charges,
        std::size_t sourceCount,
        Vec3d const& cellsDx,
        std::array<int, 3> const& offset,
        std::array<int, 3> const& tileCounts,
        double nearThresholdCells,
        double bandNearCells,
        double bandFarCells,
        bool computeBands,
        auto& tileValues,
        auto& bandCounts)
    {
        (void) sourceCount;
        auto const tileCount = static_cast<std::uint64_t>(tileCounts[0]) * static_cast<std::uint64_t>(tileCounts[1])
                               * static_cast<std::uint64_t>(tileCounts[2]);
        if(tileCount == 0u)
            return;
        using Vec = alpaka::Vec<std::uint32_t, 1>;
        auto const blocksPerTile
            = (static_cast<std::uint64_t>(tileGeometry::outputDimension) + tileWideDetail::blockSize - 1u)
              / tileWideDetail::blockSize;
        queue.enqueue(
            alpaka::onHost::FrameSpec{
                Vec{static_cast<std::uint32_t>(tileCount * blocksPerTile)},
                Vec{tileWideDetail::blockSize},
                exec},
            alpaka::KernelBundle{
                tileWideDetail::TileWideKernel{
                    cellsDx,
                    offset,
                    tileCounts,
                    sourceCount,
                    nearThresholdCells,
                    bandNearCells,
                    bandFarCells,
                    computeBands},
                positions,
                velocities,
                charges,
                tileValues,
                bandCounts});
    }

} // namespace picongpu::particles::neuralBunch
