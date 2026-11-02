/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Host-side source tree and interaction-list construction (PROTOTYPE).
 *
 * Builds an adaptive binary spatial tree ("octree-style" median splits) over
 * the sources of one streamed chunk and, for every tile, produces a disjoint
 * and exhaustive split into
 *
 *   - accepted far nodes (evaluated by their Cartesian multipoles), and
 *   - near sources (evaluated exactly by the point field).
 *
 * Every source belongs to exactly one leaf, and a traversal either accepts a
 * node (all of its sources become far) or descends into its children.  A leaf
 * that is not accepted contributes all of its sources to the near list.  The
 * invariant ``farPairs + nearPairs == sourceCount`` per tile is checked and
 * reported.
 *
 * Admissibility uses the guide's velocity-aware metric
 *
 *   d_gamma(R) = sqrt(R^T (I + u u^T) R) = |A(u) R|,
 *
 * with A = I + (gamma - 1) n n^T, evaluated for the node's representative
 * velocity u.  The node radius and the tile extent are both measured after the
 * same transform and subtracted before the multipole acceptance ratio is
 * applied.  A node is accepted only if
 *
 *   d = |A (x_tile - y_node)|,  d_min = d - radius_node' - radius_tile',
 *   d_min > 0  and  radius_node' <= theta * d_min,
 *
 * with theta = tolerance^(1/(order+1)).  This rejects usable nodes rather than
 * risking an unusable one.
 *
 * Velocity rule (conservative): a node is never evaluated with a velocity that
 * differs from a member's velocity by more than
 *
 *   |u_i - u_C| <= velocityTolerance * (1 + |u_C|) / gamma_C.
 *
 * ``(1 + |u|)/gamma`` is 1 for ultra-relativistic and sqrt(2) for slow motion,
 * so the rule bounds the angular kernel shift |du|/|u| * gamma ~ |du| for large
 * gamma (the kernel's angular width is 1/gamma).  Nodes violating the rule are
 * split by velocity; leaves that still violate it are evaluated exactly as near
 * sources.  A broad velocity distribution therefore disables grouping instead of
 * corrupting the result.
 */

#include "Multipole.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace picongpu::particles::neuralBunch::farfield
{
    struct TreeParameters
    {
        //! Cartesian multipole order, 0 (monopole) .. maxEvaluationOrder.
        int order = 3;
        //! Target relative truncation tolerance; sets theta = tolerance^(1/(order+1)).
        double tolerance = 1.0e-3;
        //! Maximum allowed relative velocity spread inside a far node.
        double velocityTolerance = 1.0e-3;
        //! Leaf source count at which splitting stops.
        int leafMax = 8;
        //! Hard depth limit.
        int maxDepth = 24;
    };

    struct SourceCloud
    {
        double const* positions = nullptr; //!< [n,3] in dx units
        double const* velocities = nullptr; //!< [n,3] u = p/(m_e c)
        double const* charges = nullptr; //!< [n] q/e
        std::size_t size = 0;
    };

    struct Node
    {
        std::size_t begin{};
        std::size_t end{};
        int depth{};
        int children[2]{-1, -1};
        double totalCharge{};
        Vec3d center{};
        Vec3d velocity{};
        //! max_i |A(u)(y_i - center)| in dx.
        double radius{};
        //! max_i |u_i - velocity|.
        double maxVelocityDeviation{};
        std::array<double, 3> bboxMin{};
        std::array<double, 3> bboxMax{};
        std::array<double, 3> velocityMin{};
        std::array<double, 3> velocityMax{};
        //! Charge moments in the node's transformed frame (|a| <= maximalOrder).
        std::array<double, maximalMomentCount> moments{};
        //! |M_a|, used for the p+1 truncation estimate.
        std::array<double, maximalMomentCount> absoluteMoments{};
        bool momentsReady{false};
    };

    /** One far node as uploaded to the device. */
    struct NodeAggregate
    {
        std::array<double, 3> center{};
        std::array<double, 3> velocity{};
        std::array<double, maximalMomentCount> moments{};
    };

    struct Statistics
    {
        std::size_t sourceCount{};
        std::size_t tileCount{};
        std::size_t farPairs{};
        std::size_t nearPairs{};
        std::size_t acceptedNodeTilePairs{};
        std::size_t usedNodes{};
        std::size_t mismatchedTiles{};
        std::size_t maxFarNodesPerTile{};
        std::size_t maxNearSourcesPerTile{};
        double maxTruncationEstimate{};
        double sumTruncationEstimate{};
        std::size_t truncationSamples{};

        double acceptedFarFraction() const
        {
            double const total = static_cast<double>(sourceCount) * static_cast<double>(tileCount);
            return total > 0.0 ? static_cast<double>(farPairs) / total : 0.0;
        }

        double meanTruncationEstimate() const
        {
            return truncationSamples ? sumTruncationEstimate / static_cast<double>(truncationSamples) : 0.0;
        }
    };

    struct InteractionLists
    {
        std::vector<std::uint32_t> farOffsets;
        std::vector<std::uint32_t> farNodes;
        std::vector<std::uint32_t> nearOffsets;
        std::vector<std::uint32_t> nearSources;
        std::vector<NodeAggregate> aggregates;
        Statistics statistics;
    };

    namespace detail
    {
        inline Vec3d loadPosition(SourceCloud const& cloud, std::size_t index)
        {
            return Vec3d{
                cloud.positions[3 * index],
                cloud.positions[3 * index + 1],
                cloud.positions[3 * index + 2]};
        }

        inline Vec3d loadVelocity(SourceCloud const& cloud, std::size_t index)
        {
            return Vec3d{
                cloud.velocities[3 * index],
                cloud.velocities[3 * index + 1],
                cloud.velocities[3 * index + 2]};
        }

        inline bool velocityAdmissible(Node const& node, double relativeTolerance)
        {
            auto const frame = makeVelocityFrame(node.velocity);
            double const speed = std::sqrt(
                node.velocity.x * node.velocity.x + node.velocity.y * node.velocity.y
                + node.velocity.z * node.velocity.z);
            double const allowed = relativeTolerance * (1.0 + speed) / frame.gamma;
            return node.maxVelocityDeviation <= allowed;
        }
    } // namespace detail

    class SourceTree
    {
    public:
        SourceTree(SourceCloud const& cloud, TreeParameters parameters)
            : cloud(cloud)
            , parameters(parameters)
        {
            order.resize(cloud.size);
            for(std::size_t i = 0; i < cloud.size; ++i)
                order[i] = i;
            nodes.reserve(2 * cloud.size / std::max(1, parameters.leafMax) + 16);
            if(cloud.size)
                build(0, cloud.size, 0);
        }

        std::vector<Node> const& getNodes() const
        {
            return nodes;
        }

        std::size_t getRoot() const
        {
            return nodes.empty() ? 0 : 0;
        }

        /** Compute and cache the charge moments of one node. */
        void computeMoments(std::size_t nodeIndex)
        {
            Node& node = nodes[nodeIndex];
            if(node.momentsReady)
                return;
            auto const frame = makeVelocityFrame(node.velocity);
            node.moments.fill(0.0);
            node.absoluteMoments.fill(0.0);
            std::array<double, maximalOrder + 1> powerX{};
            std::array<double, maximalOrder + 1> powerY{};
            std::array<double, maximalOrder + 1> powerZ{};
            for(std::size_t i = node.begin; i < node.end; ++i)
            {
                std::size_t const source = order[i];
                double const charge = cloud.charges[source];
                if(charge == 0.0)
                    continue;
                Vec3d const displacement = applyA(frame, difference(detail::loadPosition(cloud, source), node.center));
                powerX[0] = powerY[0] = powerZ[0] = 1.0;
                for(int exponent = 1; exponent <= maximalOrder; ++exponent)
                {
                    powerX[exponent] = powerX[exponent - 1] * displacement.x;
                    powerY[exponent] = powerY[exponent - 1] * displacement.y;
                    powerZ[exponent] = powerZ[exponent - 1] * displacement.z;
                }
                for(int index = 0; index < maximalMomentCount; ++index)
                {
                    auto const m = multiIndices[index];
                    double const monomial = powerX[m.i] * powerY[m.j] * powerZ[m.k];
                    node.moments[index] += charge * monomial;
                    node.absoluteMoments[index] += std::abs(charge) * monomial;
                }
            }
            node.momentsReady = true;
        }

        std::array<double, 3> const& getBoundingMin(std::size_t index) const
        {
            return nodes[index].bboxMin;
        }

        std::array<double, 3> const& getBoundingMax(std::size_t index) const
        {
            return nodes[index].bboxMax;
        }

    private:
        void updateStatistics(std::size_t begin, std::size_t end, Node& node) const
        {
            double chargeSum = 0.0;
            Vec3d weightedPosition{};
            Vec3d weightedVelocity{};
            for(int axis = 0; axis < 3; ++axis)
            {
                node.bboxMin[axis] = std::numeric_limits<double>::max();
                node.bboxMax[axis] = std::numeric_limits<double>::lowest();
                node.velocityMin[axis] = std::numeric_limits<double>::max();
                node.velocityMax[axis] = std::numeric_limits<double>::lowest();
            }
            std::size_t const count = end - begin;
            for(std::size_t i = begin; i < end; ++i)
            {
                std::size_t const source = order[i];
                Vec3d const position = detail::loadPosition(cloud, source);
                Vec3d const velocity = detail::loadVelocity(cloud, source);
                double const charge = cloud.charges[source];
                chargeSum += charge;
                weightedPosition.x += charge * position.x;
                weightedPosition.y += charge * position.y;
                weightedPosition.z += charge * position.z;
                weightedVelocity.x += charge * velocity.x;
                weightedVelocity.y += charge * velocity.y;
                weightedVelocity.z += charge * velocity.z;
                double const p[3]{position.x, position.y, position.z};
                double const u[3]{velocity.x, velocity.y, velocity.z};
                for(int axis = 0; axis < 3; ++axis)
                {
                    node.bboxMin[axis] = std::min(node.bboxMin[axis], p[axis]);
                    node.bboxMax[axis] = std::max(node.bboxMax[axis], p[axis]);
                    node.velocityMin[axis] = std::min(node.velocityMin[axis], u[axis]);
                    node.velocityMax[axis] = std::max(node.velocityMax[axis], u[axis]);
                }
            }
            node.totalCharge = chargeSum;
            // Charge-weighted centroids; fall back to the plain mean for a
            // neutral node so the geometry stays well defined.
            double const weight = std::abs(chargeSum) > 1.0e-30 ? chargeSum : static_cast<double>(count);
            if(std::abs(chargeSum) > 1.0e-30)
            {
                node.center = Vec3d{
                    weightedPosition.x / weight,
                    weightedPosition.y / weight,
                    weightedPosition.z / weight};
                node.velocity = Vec3d{
                    weightedVelocity.x / weight,
                    weightedVelocity.y / weight,
                    weightedVelocity.z / weight};
            }
            else
            {
                Vec3d plainPosition{};
                Vec3d plainVelocity{};
                for(std::size_t i = begin; i < end; ++i)
                {
                    std::size_t const source = order[i];
                    Vec3d const position = detail::loadPosition(cloud, source);
                    Vec3d const velocity = detail::loadVelocity(cloud, source);
                    plainPosition.x += position.x;
                    plainPosition.y += position.y;
                    plainPosition.z += position.z;
                    plainVelocity.x += velocity.x;
                    plainVelocity.y += velocity.y;
                    plainVelocity.z += velocity.z;
                }
                node.center = Vec3d{
                    plainPosition.x / weight,
                    plainPosition.y / weight,
                    plainPosition.z / weight};
                node.velocity = Vec3d{
                    plainVelocity.x / weight,
                    plainVelocity.y / weight,
                    plainVelocity.z / weight};
            }
            auto const frame = makeVelocityFrame(node.velocity);
            node.maxVelocityDeviation = 0.0;
            node.radius = 0.0;
            for(std::size_t i = begin; i < end; ++i)
            {
                std::size_t const source = order[i];
                double const deviation
                    = length(difference(detail::loadVelocity(cloud, source), node.velocity));
                node.maxVelocityDeviation = std::max(node.maxVelocityDeviation, deviation);
                Vec3d const transformed
                    = applyA(frame, difference(detail::loadPosition(cloud, source), node.center));
                node.radius = std::max(
                    node.radius,
                    std::sqrt(
                        transformed.x * transformed.x + transformed.y * transformed.y
                        + transformed.z * transformed.z));
            }
        }

        void build(std::size_t begin, std::size_t end, int depth)
        {
            Node node{};
            node.begin = begin;
            node.end = end;
            node.depth = depth;
            updateStatistics(begin, end, node);
            std::size_t const nodeIndex = nodes.size();
            nodes.push_back(node);

            std::size_t const count = end - begin;
            if(count <= 1 || depth >= parameters.maxDepth)
                return;
            if(count <= static_cast<std::size_t>(std::max(1, parameters.leafMax)))
                return;

            double const spatialExtent[3]{
                node.bboxMax[0] - node.bboxMin[0],
                node.bboxMax[1] - node.bboxMin[1],
                node.bboxMax[2] - node.bboxMin[2]};
            double const velocityExtent[3]{
                node.velocityMax[0] - node.velocityMin[0],
                node.velocityMax[1] - node.velocityMin[1],
                node.velocityMax[2] - node.velocityMin[2]};

            bool const velocityOk = detail::velocityAdmissible(node, parameters.velocityTolerance);
            double chosenExtent = 0.0;
            int axis = 0;
            bool splitByVelocity = false;
            if(!velocityOk)
            {
                splitByVelocity = true;
                for(int candidate = 0; candidate < 3; ++candidate)
                    if(velocityExtent[candidate] > chosenExtent)
                    {
                        chosenExtent = velocityExtent[candidate];
                        axis = candidate;
                    }
            }
            if(!splitByVelocity || chosenExtent <= 0.0)
            {
                splitByVelocity = false;
                chosenExtent = 0.0;
                for(int candidate = 0; candidate < 3; ++candidate)
                    if(spatialExtent[candidate] > chosenExtent)
                    {
                        chosenExtent = spatialExtent[candidate];
                        axis = candidate;
                    }
            }
            if(chosenExtent <= 0.0)
                return; // degenerate node, handled exactly as a leaf

            std::size_t const middle = begin + count / 2;
            if(splitByVelocity)
            {
                auto const& cloudRef = cloud;
                std::nth_element(
                    order.begin() + begin,
                    order.begin() + middle,
                    order.begin() + end,
                    [&](std::size_t a, std::size_t b)
                    {
                        auto const value = [&](std::size_t source)
                        {
                            Vec3d const u = detail::loadVelocity(cloudRef, source);
                            return axis == 0 ? u.x : (axis == 1 ? u.y : u.z);
                        };
                        return value(a) < value(b);
                    });
            }
            else
            {
                auto const& cloudRef = cloud;
                std::nth_element(
                    order.begin() + begin,
                    order.begin() + middle,
                    order.begin() + end,
                    [&](std::size_t a, std::size_t b)
                    {
                        auto const value = [&](std::size_t source)
                        {
                            Vec3d const p = detail::loadPosition(cloudRef, source);
                            return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
                        };
                        return value(a) < value(b);
                    });
            }

            // Depth-first allocation: the left subtree starts right after this
            // node and the right subtree starts when the left one is complete.
            build(begin, middle, depth + 1);
            std::size_t const leftRoot = nodeIndex + 1;
            std::size_t const rightRoot = nodes.size();
            build(middle, end, depth + 1);
            nodes[nodeIndex].children[0] = static_cast<int>(leftRoot);
            nodes[nodeIndex].children[1] = static_cast<int>(rightRoot);
        }

    public:
        SourceCloud cloud;
        TreeParameters parameters;
        std::vector<std::size_t> order;
        std::vector<Node> nodes;
    };

    /** Conservative tile radius used by the admissibility test, in dx (lab frame). */
    inline double tileEuclideanRadius(Vec3d const& cells)
    {
        // Targets span the 8 x 8 x 4 cell block plus the half-cell Yee offsets.
        double const halfX = 4.5 * cells.x;
        double const halfY = 4.5 * cells.y;
        double const halfZ = 2.5 * cells.z;
        return std::sqrt(halfX * halfX + halfY * halfY + halfZ * halfZ);
    }

    inline Vec3d tileCenter(
        std::size_t tileIndex,
        std::array<int, 3> const& offset,
        std::array<int, 3> const& tileCounts,
        Vec3d const& cells)
    {
        std::size_t const countX = static_cast<std::size_t>(tileCounts[0]);
        std::size_t const countY = static_cast<std::size_t>(tileCounts[1]);
        std::size_t const i0 = tileIndex % countX;
        std::size_t const j0 = (tileIndex / countX) % countY;
        std::size_t const k0 = tileIndex / (countX * countY);
        return Vec3d{
            (static_cast<double>(offset[0]) + (static_cast<double>(i0) + 3.5) * 8.0) * cells.x,
            (static_cast<double>(offset[1]) + (static_cast<double>(j0) + 3.5) * 8.0) * cells.y,
            (static_cast<double>(offset[2]) + (static_cast<double>(k0) + 1.5) * 4.0) * cells.z};
    }

    /** Build the per-tile far/near interaction lists.
     *
     * The returned lists are disjoint and exhaustive by construction; the
     * per-tile and global invariants are recorded in the statistics.
     */
    inline InteractionLists buildInteractionLists(
        SourceCloud const& cloud,
        TreeParameters const& parameters,
        Vec3d const& cells,
        std::array<int, 3> const& offset,
        std::array<int, 3> const& tileCounts)
    {
        InteractionLists lists;
        Statistics& stats = lists.statistics;
        stats.sourceCount = cloud.size;
        stats.tileCount = static_cast<std::size_t>(tileCounts[0]) * static_cast<std::size_t>(tileCounts[1])
                          * static_cast<std::size_t>(tileCounts[2]);
        lists.farOffsets.reserve(stats.tileCount + 1);
        lists.nearOffsets.reserve(stats.tileCount + 1);
        if(cloud.size == 0 || stats.tileCount == 0)
        {
            lists.farOffsets.assign(stats.tileCount + 1, 0u);
            lists.nearOffsets.assign(stats.tileCount + 1, 0u);
            return lists;
        }

        SourceTree tree(cloud, parameters);
        auto const& nodes = tree.getNodes();
        double const theta
            = std::pow(parameters.tolerance, 1.0 / static_cast<double>(parameters.order + 1));
        double const tileRadius = tileEuclideanRadius(cells);

        std::vector<int> nodeToAggregate(nodes.size(), -1);
        std::vector<std::size_t> stack;

        for(std::size_t tile = 0; tile < stats.tileCount; ++tile)
        {
            Vec3d const center = tileCenter(tile, offset, tileCounts, cells);
            std::size_t farPairsTile = 0;
            std::size_t nearPairsTile = 0;
            std::size_t farNodesTile = 0;
            std::size_t nearSourcesTile = 0;

            stack.clear();
            stack.push_back(0);
            while(!stack.empty())
            {
                std::size_t const nodeIndex = static_cast<std::size_t>(stack.back());
                stack.pop_back();
                Node const& node = nodes[nodeIndex];
                std::size_t const count = node.end - node.begin;
                if(count == 0)
                    continue;

                bool const velocityOk = detail::velocityAdmissible(node, parameters.velocityTolerance);
                bool accepted = false;
                if(velocityOk)
                {
                    auto const frame = makeVelocityFrame(node.velocity);
                    Vec3d const transformed = applyA(frame, difference(center, node.center));
                    double const distance = length(transformed);
                    double const effectiveTileRadius = maxStretch(frame) * tileRadius;
                    double const minimumSeparation = distance - node.radius - effectiveTileRadius;
                    accepted = minimumSeparation > 0.0 && node.radius <= theta * minimumSeparation;
                }

                if(accepted)
                {
                    if(nodeToAggregate[nodeIndex] < 0)
                    {
                        tree.computeMoments(nodeIndex);
                        NodeAggregate aggregate{};
                        aggregate.center = {node.center.x, node.center.y, node.center.z};
                        aggregate.velocity = {node.velocity.x, node.velocity.y, node.velocity.z};
                        aggregate.moments = node.moments;
                        nodeToAggregate[nodeIndex] = static_cast<int>(lists.aggregates.size());
                        lists.aggregates.push_back(aggregate);
                    }
                    lists.farNodes.push_back(static_cast<std::uint32_t>(nodeToAggregate[nodeIndex]));
                    farPairsTile += count;
                    ++farNodesTile;
                    ++stats.acceptedNodeTilePairs;

                    auto const frame = makeVelocityFrame(node.velocity);
                    Vec3d const X = applyA(frame, difference(center, node.center));
                    auto const truncation = multipoleTruncation<maxEvaluationOrder>(X, node.absoluteMoments.data());
                    double const magnitude = std::sqrt(
                        truncation.x * truncation.x + truncation.y * truncation.y
                        + truncation.z * truncation.z);
                    stats.maxTruncationEstimate = std::max(stats.maxTruncationEstimate, magnitude);
                    stats.sumTruncationEstimate += magnitude;
                    ++stats.truncationSamples;
                }
                else if(node.children[0] >= 0 || node.children[1] >= 0)
                {
                    for(int childIndex = 0; childIndex < 2; ++childIndex)
                        if(nodes[nodeIndex].children[childIndex] >= 0)
                            stack.push_back(static_cast<std::size_t>(nodes[nodeIndex].children[childIndex]));
                }
                else
                {
                    for(std::size_t i = node.begin; i < node.end; ++i)
                        lists.nearSources.push_back(static_cast<std::uint32_t>(tree.order[i]));
                    nearPairsTile += count;
                    nearSourcesTile += count;
                }
            }

            if(farPairsTile + nearPairsTile != cloud.size)
                ++stats.mismatchedTiles;
            stats.farPairs += farPairsTile;
            stats.nearPairs += nearPairsTile;
            stats.maxFarNodesPerTile = std::max(stats.maxFarNodesPerTile, farNodesTile);
            stats.maxNearSourcesPerTile = std::max(stats.maxNearSourcesPerTile, nearSourcesTile);
            lists.farOffsets.push_back(static_cast<std::uint32_t>(lists.farNodes.size()));
            lists.nearOffsets.push_back(static_cast<std::uint32_t>(lists.nearSources.size()));
        }

        stats.usedNodes = lists.aggregates.size();
        return lists;
    }

} // namespace picongpu::particles::neuralBunch::farfield
