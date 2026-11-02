/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Far-field source grouping for the neural-bunch initializer (PROTOTYPE).
 *
 * This is the additive, default-off hook that turns the O(N_sources * tiles)
 * point-field evaluation into a source-hierarchy traversal.  It does not modify
 * ``TileWideField.hpp``; the caller opts in from ``NeuralBunchInit.x.cpp`` when
 * ``picongpu::neuralBunch::farFieldParam.enabled`` is true.
 *
 * Pipeline per streamed chunk:
 *   1. host: build the adaptive source tree over the chunk (``Tree.hpp``);
 *   2. host: traverse the tree for every tile and emit disjoint far-node and
 *      near-source interaction lists;
 *   3. device: upload the mnemonics and evaluate (``Device.hpp``).
 *
 * All approximations live in the far branch.  The near branch is the existing
 * exact point field, and the sum is linear in the sources, so the result is a
 * well-defined approximation of the direct path whose error is measured by
 * ``tools/neural-bunch/eval/field_gate.py``.
 *
 * Known prototype limitations (documented, not hidden):
 *   - the tree, moments and interaction lists are rebuilt per streamed chunk and
 *     per rank; there is no MPI source ownership or tree exchange;
 *   - moments are recomputed from the members at every used node instead of
 *     transferred from children (O(N log N) build, not O(N));
 *   - the far field uses one representative velocity per node, bounded by
 *     ``velocityTolerance``; a broad velocity distribution falls back to direct;
 *   - orders are limited to 0..3 (monopole, dipole, quadrupole, octupole).
 */

#include "Device.hpp"
#include "Tree.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace picongpu::particles::neuralBunch::farfield
{
    namespace detail
    {
        template<typename T, typename TDevice, typename TQueue>
        auto makeDeviceCopy(
            TQueue& queue,
            TDevice const& device,
            std::vector<T> const& host,
            std::size_t minSize)
        {
            std::size_t const size = std::max<std::size_t>(host.size(), minSize);
            auto buffer = alpaka::onHost::alloc<T>(device, alpaka::Vec{size});
            auto hostBuffer = alpaka::onHost::allocHost<T>(alpaka::Vec{size});
            for(std::size_t i = 0; i < size; ++i)
                hostBuffer[alpaka::Vec<std::size_t, 1>{i}] = i < host.size() ? host[i] : T{};
            alpaka::onHost::memcpy(queue, buffer, hostBuffer);
            hostBuffer.keepAlive(queue);
            return buffer;
        }

    } // namespace detail

    /** Build the tree and interaction lists for one chunk and enqueue the device evaluation.
     *
     * @param hostPositions ``[sourceCount,3]`` host buffer, dx units.
     * @param hostVelocities ``[sourceCount,3]`` host buffer, u = p/(m_e c).
     * @param hostCharges ``[sourceCount]`` host buffer, q/e.
     * @param devicePositions/deviceVelocities/deviceCharges matching device buffers.
     * @param tileValues ``[tiles,1536]`` device accumulator, added in place.
     */
    template<
        typename TQueue,
        typename TExec,
        typename THostPositions,
        typename THostVelocities,
        typename THostCharges,
        typename TDevicePositions,
        typename TDeviceVelocities,
        typename TDeviceCharges,
        typename TTileValues>
    Statistics enqueueFarFieldChunk(
        TQueue& queue,
        TExec exec,
        THostPositions const& hostPositions,
        THostVelocities const& hostVelocities,
        THostCharges const& hostCharges,
        TDevicePositions& devicePositions,
        TDeviceVelocities& deviceVelocities,
        TDeviceCharges& deviceCharges,
        std::size_t sourceCount,
        Vec3d const& cells,
        std::array<int, 3> const& offset,
        std::array<int, 3> const& tileCounts,
        int order,
        double tolerance,
        double velocityTolerance,
        int leafMax,
        int maxDepth,
        TTileValues& tileValues)
    {
        std::vector<double> positions(3 * sourceCount);
        std::vector<double> velocities(3 * sourceCount);
        std::vector<double> charges(sourceCount);
        for(std::size_t source = 0; source < sourceCount; ++source)
        {
            for(std::size_t axis = 0; axis < 3; ++axis)
            {
                positions[3 * source + axis]
                    = hostPositions[alpaka::Vec<std::size_t, 2>{source, axis}];
                velocities[3 * source + axis]
                    = hostVelocities[alpaka::Vec<std::size_t, 2>{source, axis}];
            }
            charges[source] = hostCharges[alpaka::Vec<std::size_t, 1>{source}];
        }

        TreeParameters parameters{};
        parameters.order = std::clamp(order, 0, maxEvaluationOrder);
        parameters.tolerance = tolerance;
        parameters.velocityTolerance = velocityTolerance;
        parameters.leafMax = std::max(1, leafMax);
        parameters.maxDepth = std::max(1, maxDepth);

        SourceCloud cloud{positions.data(), velocities.data(), charges.data(), sourceCount};
        InteractionLists lists = buildInteractionLists(cloud, parameters, cells, offset, tileCounts);

        auto const& device = queue.getDevice();

        // Node aggregates are indexed [node, component] on the device.
        std::size_t const nodeCount = std::max<std::size_t>(lists.aggregates.size(), 1u);
        auto hostCenter = alpaka::onHost::allocHost<double>(alpaka::Vec{nodeCount, std::size_t{3}});
        auto hostVelocity = alpaka::onHost::allocHost<double>(alpaka::Vec{nodeCount, std::size_t{3}});
        auto hostMoments = alpaka::onHost::allocHost<double>(
            alpaka::Vec{nodeCount, static_cast<std::size_t>(maximalMomentCount)});
        for(std::size_t node = 0; node < lists.aggregates.size(); ++node)
        {
            for(std::size_t axis = 0; axis < 3; ++axis)
            {
                hostCenter[alpaka::Vec<std::size_t, 2>{node, axis}]
                    = lists.aggregates[node].center[axis];
                hostVelocity[alpaka::Vec<std::size_t, 2>{node, axis}]
                    = lists.aggregates[node].velocity[axis];
            }
            for(int index = 0; index < maximalMomentCount; ++index)
                hostMoments[alpaka::Vec<std::size_t, 2>{node, static_cast<std::size_t>(index)}]
                    = lists.aggregates[node].moments[index];
        }
        auto deviceCenter = alpaka::onHost::allocLike(device, hostCenter);
        auto deviceVelocity = alpaka::onHost::allocLike(device, hostVelocity);
        auto deviceMoments = alpaka::onHost::allocLike(device, hostMoments);
        alpaka::onHost::memcpy(queue, deviceCenter, hostCenter);
        alpaka::onHost::memcpy(queue, deviceVelocity, hostVelocity);
        alpaka::onHost::memcpy(queue, deviceMoments, hostMoments);
        hostCenter.keepAlive(queue);
        hostVelocity.keepAlive(queue);
        hostMoments.keepAlive(queue);

        auto deviceFarOffsets = detail::makeDeviceCopy<std::uint32_t>(queue, device, lists.farOffsets, 1);
        auto deviceFarNodes = detail::makeDeviceCopy<std::uint32_t>(queue, device, lists.farNodes, 1);
        auto deviceNearOffsets = detail::makeDeviceCopy<std::uint32_t>(queue, device, lists.nearOffsets, 1);
        auto deviceNearSources = detail::makeDeviceCopy<std::uint32_t>(queue, device, lists.nearSources, 1);

        enqueueFarField(
            queue,
            exec,
            deviceCenter,
            deviceVelocity,
            deviceMoments,
            deviceFarOffsets,
            deviceFarNodes,
            deviceNearOffsets,
            deviceNearSources,
            devicePositions,
            deviceVelocities,
            deviceCharges,
            cells,
            offset,
            tileCounts,
            parameters.order,
            tileValues);

        return lists.statistics;
    }

    /** One-line summary for the initializer log. */
    inline std::string statisticsLine(Statistics const& stats)
    {
        char buffer[512];
        std::snprintf(
            buffer,
            sizeof(buffer),
            "farfield: orderTruncationEstimateMax=%g mean=%g farPairs=%zu nearPairs=%zu "
            "acceptedFarFraction=%.6f usedNodes=%zu nodeTileInteractions=%zu maxFarNodesPerTile=%zu "
            "maxNearSourcesPerTile=%zu pairAccountingMismatchTiles=%zu",
            stats.maxTruncationEstimate,
            stats.meanTruncationEstimate(),
            stats.farPairs,
            stats.nearPairs,
            stats.acceptedFarFraction(),
            stats.usedNodes,
            stats.acceptedNodeTilePairs,
            stats.maxFarNodesPerTile,
            stats.maxNearSourcesPerTile,
            stats.mismatchedTiles);
        return std::string{buffer};
    }

} // namespace picongpu::particles::neuralBunch::farfield
