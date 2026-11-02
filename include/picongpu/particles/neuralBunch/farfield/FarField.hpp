/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Standalone far-field prototype for the neural-bunch tile field.
 *
 * This header is deliberately NOT wired into the production initializer. It
 * provides an octree over the source particles, per-node charge / centroid /
 * charge-weighted velocity / Cartesian moments, a velocity-aware admissibility
 * test, and an expansion evaluator that reuses the existing analytic point
 * kernel (``pointFieldSlot``) for the near path.
 *
 * The far expansion is a Cartesian Taylor expansion of the *same* relativistic
 * point-charge field used by ``PointField.hpp``. For one node the mean velocity
 * ``u`` fixes ``gamma`` and the direction ``bhat``; with ``s = R - d + (gamma-1)
 * ((R-d).bhat) bhat`` the field is ``gamma * s / |s|^3`` (and ``B = beta x E``),
 * exactly the decomposition in ``Features.hpp:pointFieldDx``. The Taylor series
 * in the source offset ``d`` is generated with a small three-variable jet type,
 * then contracted with the charge-weighted moments ``sum q d^alpha``.
 *
 * The near evaluation calls ``pointFieldSlot`` unchanged.
 */

#include "picongpu/particles/neuralBunch/PointField.hpp"
#include "picongpu/particles/neuralBunch/TileGeometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace picongpu::particles::neuralBunch::farfield
{
    inline constexpr int kMaxOrder = 6;
    inline constexpr int kMaxTerms = 84; // (6+1)(6+2)(6+3)/6

    constexpr int termCount(int order)
    {
        return (order + 1) * (order + 2) * (order + 3) / 6;
    }

    /** Decode a term index back into its multi-index.
     *
     * Computed arithmetically instead of a namespace-scope table so the same
     * code is usable from device kernels without a device-side global.
     */
    constexpr std::array<int, kMaxTerms> makeTermA()
    {
        std::array<int, kMaxTerms> result{};
        int term = 0;
        for(int total = 0; total <= kMaxOrder; ++total)
            for(int aa = 0; aa <= total; ++aa)
                for(int bb = 0; bb <= total - aa; ++bb)
                    result[term++] = aa;
        return result;
    }

    constexpr std::array<int, kMaxTerms> makeTermB()
    {
        std::array<int, kMaxTerms> result{};
        int term = 0;
        for(int total = 0; total <= kMaxOrder; ++total)
            for(int aa = 0; aa <= total; ++aa)
                for(int bb = 0; bb <= total - aa; ++bb)
                    result[term++] = bb;
        return result;
    }

    constexpr std::array<int, kMaxTerms> makeTermC()
    {
        std::array<int, kMaxTerms> result{};
        int term = 0;
        for(int total = 0; total <= kMaxOrder; ++total)
            for(int aa = 0; aa <= total; ++aa)
                for(int bb = 0; bb <= total - aa; ++bb)
                    result[term++] = total - aa - bb;
        return result;
    }

    constexpr std::array<int, kMaxTerms> makeTermDegree()
    {
        std::array<int, kMaxTerms> result{};
        int term = 0;
        for(int total = 0; total <= kMaxOrder; ++total)
            for(int aa = 0; aa <= total; ++aa)
                for(int bb = 0; bb <= total - aa; ++bb)
                    result[term++] = total;
        return result;
    }

    constexpr std::array<int, kMaxTerms> kTermAHost = makeTermA();
    constexpr std::array<int, kMaxTerms> kTermBHost = makeTermB();
    constexpr std::array<int, kMaxTerms> kTermCHost = makeTermC();
    constexpr std::array<int, kMaxTerms> kTermDegreeHost = makeTermDegree();
#if defined(__CUDACC__)
    __device__ constexpr std::array<int, kMaxTerms> kTermADevice = makeTermA();
    __device__ constexpr std::array<int, kMaxTerms> kTermBDevice = makeTermB();
    __device__ constexpr std::array<int, kMaxTerms> kTermCDevice = makeTermC();
    __device__ constexpr std::array<int, kMaxTerms> kTermDegreeDevice = makeTermDegree();
#endif

    ALPAKA_FN_HOST_ACC inline constexpr int termDegreeOf(int index)
    {
#if defined(__CUDA_ARCH__)
        return kTermDegreeDevice[index];
#else
        return kTermDegreeHost[index];
#endif
    }

    ALPAKA_FN_HOST_ACC inline constexpr int termAOf(int index)
    {
#if defined(__CUDA_ARCH__)
        return kTermADevice[index];
#else
        return kTermAHost[index];
#endif
    }

    ALPAKA_FN_HOST_ACC inline constexpr int termBOf(int index)
    {
#if defined(__CUDA_ARCH__)
        return kTermBDevice[index];
#else
        return kTermBHost[index];
#endif
    }

    ALPAKA_FN_HOST_ACC inline constexpr int termCOf(int index)
    {
#if defined(__CUDA_ARCH__)
        return kTermCDevice[index];
#else
        return kTermCHost[index];
#endif
    }

    ALPAKA_FN_HOST_ACC inline constexpr int termIndex(int aa, int bb, int cc)
    {
        int const total = aa + bb + cc;
        int result = 0;
        for(int t = 0; t < total; ++t)
            result += (t + 1) * (t + 2) / 2;
        for(int x = 0; x < aa; ++x)
            result += total - x + 1;
        return result + bb;
    }

    ALPAKA_FN_HOST_ACC inline constexpr int flatTermIndex(int key)
    {
        int const a = key / 49;
        int const b = (key / 7) % 7;
        int const c = key % 7;
        return termIndex(a, b, c);
    }

    /** Truncated three-variable Taylor jet. Terms are ordered by total degree, then by the x exponent, then y. */
    template<int Order>
    struct Jet3
    {
        static_assert(Order >= 0 && Order <= kMaxOrder);
        double v[termCount(Order)]{};

        ALPAKA_FN_HOST_ACC static Jet3 constant(double value)
        {
            Jet3 result;
            result.v[0] = value;
            return result;
        }

        ALPAKA_FN_HOST_ACC static Jet3 variable(int axis)
        {
            Jet3 result;
            result.v[termIndex(axis == 0 ? 1 : 0, axis == 1 ? 1 : 0, axis == 2 ? 1 : 0)] = 1.0;
            return result;
        }
    };

    template<int Order>
    ALPAKA_FN_HOST_ACC inline Jet3<Order> add(Jet3<Order> const& x, Jet3<Order> const& y)
    {
        Jet3<Order> result;
        constexpr int terms = termCount(Order);
        for(int i = 0; i < terms; ++i)
            result.v[i] = x.v[i] + y.v[i];
        return result;
    }

    template<int Order>
    ALPAKA_FN_HOST_ACC inline Jet3<Order> subtract(Jet3<Order> const& x, Jet3<Order> const& y)
    {
        Jet3<Order> result;
        constexpr int terms = termCount(Order);
        for(int i = 0; i < terms; ++i)
            result.v[i] = x.v[i] - y.v[i];
        return result;
    }

    template<int Order>
    ALPAKA_FN_HOST_ACC inline Jet3<Order> scale(Jet3<Order> const& x, double factor)
    {
        Jet3<Order> result;
        constexpr int terms = termCount(Order);
        for(int i = 0; i < terms; ++i)
            result.v[i] = x.v[i] * factor;
        return result;
    }

    template<int Order>
    ALPAKA_FN_HOST_ACC inline Jet3<Order> multiply(Jet3<Order> const& x, Jet3<Order> const& y)
    {
        Jet3<Order> result;
        constexpr int terms = termCount(Order);
        for(int i = 0; i < terms; ++i)
        {
            if(x.v[i] == 0.0)
                continue;
            int const ai = termAOf(i);
            int const bi = termBOf(i);
            int const ci = termCOf(i);
            int const di = termDegreeOf(i);
            for(int j = 0; j < terms; ++j)
            {
                if(y.v[j] == 0.0)
                    continue;
                if(di + termDegreeOf(j) > Order)
                    continue;
                int const ia = ai + termAOf(j);
                int const ib = bi + termBOf(j);
                int const ic = ci + termCOf(j);
                result.v[flatTermIndex(ia * 49 + ib * 7 + ic)] += x.v[i] * y.v[j];
            }
        }
        return result;
    }

    /** 1 / f, exact through the requested order (requires f(0) != 0). */
    template<int Order>
    ALPAKA_FN_HOST_ACC inline Jet3<Order> inverse(Jet3<Order> const& f)
    {
        Jet3<Order> result;
        constexpr int terms = termCount(Order);
        result.v[0] = 1.0 / f.v[0];
        for(int i = 1; i < terms; ++i)
        {
            int const ai = termAOf(i);
            int const bi = termBOf(i);
            int const ci = termCOf(i);
            int const di = termDegreeOf(i);
            double sum = 0.0;
            for(int j = 1; j < terms; ++j)
            {
                if(termDegreeOf(j) > di)
                    continue;
                int const ja = termAOf(j);
                int const jb = termBOf(j);
                int const jc = termCOf(j);
                if(ja > ai || jb > bi || jc > ci)
                    continue;
                int const k = flatTermIndex((ai - ja) * 49 + (bi - jb) * 7 + (ci - jc));
                sum += f.v[j] * result.v[k];
            }
            result.v[i] = -sum / f.v[0];
        }
        return result;
    }

    /** sqrt(f), exact through the requested order (requires f(0) > 0). */
    template<int Order>
    ALPAKA_FN_HOST_ACC inline Jet3<Order> squareRoot(Jet3<Order> const& f)
    {
        Jet3<Order> result;
        constexpr int terms = termCount(Order);
        result.v[0] = alpaka::math::sqrt(f.v[0]);
        for(int i = 1; i < terms; ++i)
        {
            int const ai = termAOf(i);
            int const bi = termBOf(i);
            int const ci = termCOf(i);
            int const di = termDegreeOf(i);
            double sum = 0.0;
            for(int j = 1; j < terms; ++j)
            {
                int const dj = termDegreeOf(j);
                if(dj >= di)
                    continue;
                int const ja = termAOf(j);
                int const jb = termBOf(j);
                int const jc = termCOf(j);
                if(ja > ai || jb > bi || jc > ci)
                    continue;
                int const k = flatTermIndex((ai - ja) * 49 + (bi - jb) * 7 + (ci - jc));
                sum += result.v[j] * result.v[k];
            }
            result.v[i] = (f.v[i] - sum) / (2.0 * result.v[0]);
        }
        return result;
    }

    /** Per-node aggregate. POD so it can be copied to a device unchanged. */
    struct NodeAggregate
    {
        double center[3]{};
        double radius{};
        double charge{};
        double meanU[3]{};
        double gamma{1.0};
        double moments[kMaxTerms]{};
    };

    struct Source
    {
        double r[3]{};
        double u[3]{};
        double charge{};
    };

    struct SourceCloud
    {
        double const* positions{}; // [S,3], interleaved
        double const* velocities{}; // [S,3], interleaved
        double const* charges{}; // [S]
        std::size_t count{};
    };

    struct TreeParameters
    {
        int leafCapacity = 16;
        int maxDepth = 12;
        /** Accept a far node only if d_gamma > separationFactor * radius + tileRadius. */
        double separationFactor = 1.0;
        /** Minimum accepted d_gamma; guards degenerate all-coincident nodes. */
        double minimumSeparation = 0.5;
    };

    struct Statistics
    {
        std::size_t farNodeInteractions = 0; // accepted (node, tile) pairs
        std::size_t nearSourceEvaluations = 0; // exact (source, tile) pairs
        std::size_t directSourceEvaluations = 0; // baseline (source, tile) pairs
        std::size_t tileCount = 0;
        std::size_t nodeCount = 0;
        std::size_t usedFarNodes = 0;
        std::size_t maxFarNodesPerTile = 0;
        std::size_t maxNearSourcesPerTile = 0;

        std::size_t farSourceEquivalent() const
        {
            return farNodeInteractions;
        }
        /** Work units at the chosen expansion order: far node/tile times
         * (order+1)(order+2)(order+3)/6 moments, near source/tile exactly. */
        std::size_t totalWorkUnits(int order) const
        {
            return farNodeInteractions * static_cast<std::size_t>(termCount(order)) + nearSourceEvaluations;
        }
    };

    struct InteractionLists
    {
        std::vector<NodeAggregate> nodes;
        std::vector<int> farOffsets; // tileCount + 1
        std::vector<int> farNodes; // node indices
        std::vector<int> nearOffsets; // tileCount + 1
        std::vector<std::uint32_t> nearSources; // source indices
        std::vector<std::uint8_t> tileIsFarEmpty; // tileCount
        Statistics statistics;
    };

    namespace detail
    {
        struct HostNode
        {
            NodeAggregate aggregate{};
            int child[8]{-1, -1, -1, -1, -1, -1, -1, -1};
            std::size_t begin{};
            std::size_t end{};
            bool leaf{true};
        };

        inline double dot(double const a[3], double const b[3])
        {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }

        /** Charge / centroid / charge-weighted velocity / radius / moments over [begin,end). */
        inline NodeAggregate aggregateRange(
            SourceCloud const& cloud,
            std::vector<std::uint32_t> const& perm,
            std::size_t begin,
            std::size_t end,
            double const fallbackCenter[3])
        {
            NodeAggregate node;
            double chargeSum = 0.0;
            double weighted[3]{0.0, 0.0, 0.0};
            double velocitySum[3]{0.0, 0.0, 0.0};
            double absWeight = 0.0;
            for(std::size_t slot = begin; slot < end; ++slot)
            {
                std::size_t const source = perm[slot];
                double const charge = cloud.charges[source];
                chargeSum += charge;
                for(int axis = 0; axis < 3; ++axis)
                {
                    weighted[axis] += charge * cloud.positions[3 * source + axis];
                    velocitySum[axis] += charge * cloud.velocities[3 * source + axis];
                }
                absWeight += std::abs(charge);
            }
            node.charge = chargeSum;
            bool const useCharge = std::abs(chargeSum) > 1.0e-30 * std::max(absWeight, 1.0);
            for(int axis = 0; axis < 3; ++axis)
            {
                node.center[axis]
                    = useCharge ? weighted[axis] / chargeSum : fallbackCenter[axis];
                node.meanU[axis] = useCharge && absWeight > 0.0
                                       ? velocitySum[axis] / chargeSum
                                       : 0.0;
            }
            if(!useCharge && absWeight > 0.0)
            {
                // Fall back to an absolute-charge weighted velocity for neutral nodes.
                double sum[3]{0.0, 0.0, 0.0};
                for(std::size_t slot = begin; slot < end; ++slot)
                {
                    std::size_t const source = perm[slot];
                    double const weight = std::abs(cloud.charges[source]);
                    for(int axis = 0; axis < 3; ++axis)
                        sum[axis] += weight * cloud.velocities[3 * source + axis];
                }
                for(int axis = 0; axis < 3; ++axis)
                    node.meanU[axis] = sum[axis] / absWeight;
            }
            double const uu = dot(node.meanU, node.meanU);
            node.gamma = std::sqrt(1.0 + uu);

            double radius2 = 0.0;
            for(std::size_t slot = begin; slot < end; ++slot)
            {
                std::size_t const source = perm[slot];
                double delta[3];
                for(int axis = 0; axis < 3; ++axis)
                    delta[axis] = cloud.positions[3 * source + axis] - node.center[axis];
                radius2 = std::max(radius2, dot(delta, delta));
            }
            node.radius = std::sqrt(radius2);

            // Moments sum q * d^alpha for |alpha| <= kMaxOrder.
            for(std::size_t slot = begin; slot < end; ++slot)
            {
                std::size_t const source = perm[slot];
                double const charge = cloud.charges[source];
                if(charge == 0.0)
                    continue;
                double powers[3][kMaxOrder + 1];
                for(int axis = 0; axis < 3; ++axis)
                {
                    double const delta = cloud.positions[3 * source + axis] - node.center[axis];
                    powers[axis][0] = 1.0;
                    for(int exponent = 1; exponent <= kMaxOrder; ++exponent)
                        powers[axis][exponent] = powers[axis][exponent - 1] * delta;
                }
                for(int term = 0; term < kMaxTerms; ++term)
                    node.moments[term] += charge * powers[0][termAOf(term)] * powers[1][termBOf(term)]
                                          * powers[2][termCOf(term)];
            }
            return node;
        }

        inline void buildTree(
            SourceCloud const& cloud,
            TreeParameters const& parameters,
            std::vector<std::uint32_t>& perm,
            std::vector<HostNode>& nodes,
            int nodeIndex,
            std::size_t begin,
            std::size_t end,
            double const cubeCenter[3],
            double const cubeHalf,
            int depth)
        {
            nodes[static_cast<std::size_t>(nodeIndex)].begin = begin;
            nodes[static_cast<std::size_t>(nodeIndex)].end = end;
            std::size_t const count = end - begin;
            bool const canSplit = count > static_cast<std::size_t>(parameters.leafCapacity)
                                  && depth < parameters.maxDepth && count > 8u;
            if(!canSplit)
            {
                nodes[static_cast<std::size_t>(nodeIndex)].leaf = true;
                nodes[static_cast<std::size_t>(nodeIndex)].aggregate
                    = aggregateRange(cloud, perm, begin, end, cubeCenter);
                return;
            }

            // Stable partition into the eight octants.
            std::array<std::size_t, 8> counts{};
            std::array<std::vector<std::uint32_t>, 8> buckets;
            for(std::size_t slot = begin; slot < end; ++slot)
            {
                std::uint32_t const source = perm[slot];
                int octant = 0;
                for(int axis = 0; axis < 3; ++axis)
                    if(cloud.positions[3 * source + axis] >= cubeCenter[axis])
                        octant |= 1 << axis;
                buckets[octant].push_back(source);
                ++counts[octant];
            }
            std::array<std::size_t, 8> offsets{};
            offsets[0] = begin;
            for(int octant = 1; octant < 8; ++octant)
                offsets[octant] = offsets[octant - 1] + counts[octant - 1];
            for(int octant = 0; octant < 8; ++octant)
                std::copy(buckets[octant].begin(), buckets[octant].end(), perm.begin() + offsets[octant]);

            // Degenerate partition (all sources on one side): keep the node as a leaf.
            for(int octant = 0; octant < 8; ++octant)
            {
                if(counts[octant] == count)
                {
                    nodes[static_cast<std::size_t>(nodeIndex)].leaf = true;
                    nodes[static_cast<std::size_t>(nodeIndex)].aggregate
                        = aggregateRange(cloud, perm, begin, end, cubeCenter);
                    return;
                }
            }

            nodes[static_cast<std::size_t>(nodeIndex)].leaf = false;
            double const childHalf = 0.5 * cubeHalf;
            for(int octant = 0; octant < 8; ++octant)
            {
                if(counts[octant] == 0)
                    continue;
                int const childIndex = static_cast<int>(nodes.size());
                nodes.emplace_back();
                nodes[static_cast<std::size_t>(nodeIndex)].child[octant] = childIndex;
                double childCenter[3];
                for(int axis = 0; axis < 3; ++axis)
                    childCenter[axis] = cubeCenter[axis] + ((octant >> axis) & 1 ? childHalf : -childHalf);
                buildTree(
                    cloud,
                    parameters,
                    perm,
                    nodes,
                    childIndex,
                    offsets[octant],
                    offsets[octant] + counts[octant],
                    childCenter,
                    childHalf,
                    depth + 1);
            }
            nodes[static_cast<std::size_t>(nodeIndex)].aggregate
                = aggregateRange(cloud, perm, begin, end, cubeCenter);
        }

        inline void computeRootCube(SourceCloud const& cloud, double center[3], double& half)
        {
            double low[3]{1.0e300, 1.0e300, 1.0e300};
            double high[3]{-1.0e300, -1.0e300, -1.0e300};
            bool any = false;
            for(std::size_t source = 0; source < cloud.count; ++source)
            {
                if(cloud.charges[source] == 0.0)
                    continue;
                any = true;
                for(int axis = 0; axis < 3; ++axis)
                {
                    low[axis] = std::min(low[axis], cloud.positions[3 * source + axis]);
                    high[axis] = std::max(high[axis], cloud.positions[3 * source + axis]);
                }
            }
            if(!any)
            {
                for(int axis = 0; axis < 3; ++axis)
                {
                    center[axis] = 0.0;
                    low[axis] = high[axis] = 0.0;
                }
            }
            double extent = 0.0;
            for(int axis = 0; axis < 3; ++axis)
            {
                center[axis] = 0.5 * (low[axis] + high[axis]);
                extent = std::max(extent, high[axis] - low[axis]);
            }
            half = 0.5 * extent + 1.0e-9;
        }

        inline double tileHalfDiagonal(Vec3d const& cells)
        {
            double const x = 0.5 * tileGeometry::tileNx * cells.x;
            double const y = 0.5 * tileGeometry::tileNy * cells.y;
            double const z = 0.5 * tileGeometry::tileNz * cells.z;
            return std::sqrt(x * x + y * y + z * z);
        }

        inline void collectTile(
            std::vector<HostNode> const& nodes,
            int nodeIndex,
            double const tileCenter[3],
            double const tileRadius,
            TreeParameters const& parameters,
            std::vector<std::uint32_t> const& perm,
            std::vector<int>& farNodes,
            std::vector<std::uint32_t>& nearSources,
            std::vector<std::uint8_t>& usedNode,
            Statistics& statistics)
        {
            HostNode const& node = nodes[static_cast<std::size_t>(nodeIndex)];
            NodeAggregate const& aggregate = node.aggregate;
            double relative[3];
            for(int axis = 0; axis < 3; ++axis)
                relative[axis] = tileCenter[axis] - aggregate.center[axis];
            double const along = dot(relative, aggregate.meanU);
            double const dGamma
                = std::sqrt(dot(relative, relative) + along * along);
            double const threshold
                = parameters.separationFactor * aggregate.radius + tileRadius;
            bool const admitted = dGamma > threshold && dGamma > parameters.minimumSeparation;

            if(admitted)
            {
                farNodes.push_back(nodeIndex);
                usedNode[static_cast<std::size_t>(nodeIndex)] = 1u;
                ++statistics.farNodeInteractions;
                return;
            }
            if(node.leaf)
            {
                for(std::size_t slot = node.begin; slot < node.end; ++slot)
                    nearSources.push_back(perm[slot]);
                return;
            }
            for(int octant = 0; octant < 8; ++octant)
                if(node.child[octant] >= 0)
                    collectTile(
                        nodes,
                        node.child[octant],
                        tileCenter,
                        tileRadius,
                        parameters,
                        perm,
                        farNodes,
                        nearSources,
                        usedNode,
                        statistics);
        }
    } // namespace detail

    /** Build the octree plus per-tile near/far interaction lists.
     *
     * The near list stores original source indices, so the evaluator does not
     * depend on the internal permutation used while building the tree.
     */
    inline InteractionLists buildInteractionLists(
        SourceCloud const& cloud,
        TreeParameters const& parameters,
        Vec3d const& cells,
        std::array<int, 3> const& offset,
        std::array<int, 3> const& tileCounts)
    {
        InteractionLists lists;
        std::size_t const tileCount = static_cast<std::size_t>(tileCounts[0])
                                      * static_cast<std::size_t>(tileCounts[1])
                                      * static_cast<std::size_t>(tileCounts[2]);
        lists.statistics.tileCount = tileCount;
        lists.statistics.directSourceEvaluations = cloud.count * tileCount;
        lists.farOffsets.assign(tileCount + 1, 0);
        lists.nearOffsets.assign(tileCount + 1, 0);
        lists.tileIsFarEmpty.assign(tileCount, 1);

        std::vector<std::uint32_t> perm(cloud.count);
        for(std::size_t source = 0; source < cloud.count; ++source)
            perm[source] = static_cast<std::uint32_t>(source);

        std::vector<detail::HostNode> hostNodes;
        hostNodes.reserve(std::max<std::size_t>(1, cloud.count / 4));
        double rootCenter[3];
        double rootHalf = 1.0;
        detail::computeRootCube(cloud, rootCenter, rootHalf);
        hostNodes.emplace_back();
        detail::buildTree(cloud, parameters, perm, hostNodes, 0, 0, cloud.count, rootCenter, rootHalf, 0);

        lists.statistics.nodeCount = hostNodes.size();
        lists.nodes.resize(hostNodes.size());
        for(std::size_t i = 0; i < hostNodes.size(); ++i)
            lists.nodes[i] = hostNodes[i].aggregate;
        std::vector<std::uint8_t> usedNode(hostNodes.size(), 0);

        double const tileRadius = detail::tileHalfDiagonal(cells);
        std::size_t farTotal = 0;
        std::size_t nearTotal = 0;
        for(std::size_t tile = 0; tile < tileCount; ++tile)
        {
            auto const i0 = static_cast<int>(tile % static_cast<std::size_t>(tileCounts[0]));
            auto const j0 = static_cast<int>(
                (tile / static_cast<std::size_t>(tileCounts[0])) % static_cast<std::size_t>(tileCounts[1]));
            auto const k0 = static_cast<int>(tile / (static_cast<std::size_t>(tileCounts[0])
                                                      * static_cast<std::size_t>(tileCounts[1])));
            double tileCenter[3]{
                static_cast<double>(offset[0] + i0 * tileGeometry::tileNx) + 0.5 * tileGeometry::tileNx * cells.x,
                (static_cast<double>(offset[1] + j0 * tileGeometry::tileNy) + 0.5 * tileGeometry::tileNy)
                    * cells.y,
                (static_cast<double>(offset[2] + k0 * tileGeometry::tileNz) + 0.5 * tileGeometry::tileNz)
                    * cells.z};

            std::vector<int> farNodes;
            std::vector<std::uint32_t> nearSources;
            if(cloud.count > 0)
                detail::collectTile(
                    hostNodes,
                    0,
                    tileCenter,
                    tileRadius,
                    parameters,
                    perm,
                    farNodes,
                    nearSources,
                    usedNode,
                    lists.statistics);
            farTotal += farNodes.size();
            nearTotal += nearSources.size();
            lists.statistics.maxFarNodesPerTile = std::max(lists.statistics.maxFarNodesPerTile, farNodes.size());
            lists.statistics.maxNearSourcesPerTile
                = std::max(lists.statistics.maxNearSourcesPerTile, nearSources.size());
            lists.tileIsFarEmpty[tile] = farNodes.empty() ? 1u : 0u;
            for(int index : farNodes)
                lists.farNodes.push_back(index);
            for(std::uint32_t index : nearSources)
                lists.nearSources.push_back(index);
            lists.farOffsets[tile + 1] = static_cast<int>(farTotal);
            lists.nearOffsets[tile + 1] = static_cast<int>(nearTotal);
        }
        lists.statistics.nearSourceEvaluations = nearTotal;
        for(std::uint8_t used : usedNode)
            lists.statistics.usedFarNodes += used;

        return lists;
    }

    /** Far-field expansion for one node at one target point, for one component.
     *
     * @p component is ``slot % 6`` (0..2 E, 3..5 B). Reuses the exact
     * relativistic decomposition of ``pointFieldDx``.
     */
    template<int Order>
    ALPAKA_FN_HOST_ACC inline double nodeFieldComponent(
        NodeAggregate const& node,
        double targetX,
        double targetY,
        double targetZ,
        int component)
    {
        double const R[3]{targetX - node.center[0], targetY - node.center[1], targetZ - node.center[2]};
        double const uu = node.meanU[0] * node.meanU[0] + node.meanU[1] * node.meanU[1]
                          + node.meanU[2] * node.meanU[2];
        double const invG2 = 1.0 / (1.0 + uu);
        double const gamma = alpaka::math::sqrt(1.0 + uu);
        double const speed = alpaka::math::sqrt(uu);
        double const safe = speed > 0.0 ? speed : 1.0;
        double const bhat[3]{node.meanU[0] / safe, node.meanU[1] / safe, node.meanU[2] / safe};
        double const beta[3]{node.meanU[0] / gamma, node.meanU[1] / gamma, node.meanU[2] / gamma};
        (void) invG2;

        // t = R - d
        Jet3<Order> t[3];
        for(int axis = 0; axis < 3; ++axis)
            t[axis] = subtract(Jet3<Order>::constant(R[axis]), Jet3<Order>::variable(axis));

        // s = t + (gamma - 1) (t . bhat) bhat
        Jet3<Order> along = add(
            add(scale(t[0], bhat[0]), scale(t[1], bhat[1])),
            scale(t[2], bhat[2]));
        Jet3<Order> s[3];
        for(int axis = 0; axis < 3; ++axis)
            s[axis] = add(t[axis], scale(along, (gamma - 1.0) * bhat[axis]));

        Jet3<Order> s2 = add(add(multiply(s[0], s[0]), multiply(s[1], s[1])), multiply(s[2], s[2]));
        Jet3<Order> r = squareRoot(s2);
        Jet3<Order> r3 = multiply(multiply(r, r), r);
        Jet3<Order> invR3 = inverse(r3);

        // The field is gamma * R / |stretched R|^3: the numerator uses the
        // unstretched offset, only the denominator is Lorentz-stretched.
        Jet3<Order> Ejet[3];
        for(int axis = 0; axis < 3; ++axis)
            Ejet[axis] = scale(multiply(t[axis], invR3), gamma);

        Jet3<Order> field;
        if(component < 3)
        {
            field = Ejet[component];
        }
        else
        {
            // B = beta x E
            int const a = component - 3;
            int const i = (a + 1) % 3;
            int const j = (a + 2) % 3;
            field = subtract(scale(Ejet[j], beta[i]), scale(Ejet[i], beta[j]));
        }
        double sum = 0.0;
        constexpr int terms = termCount(Order);
        for(int term = 0; term < terms; ++term)
            sum += field.v[term] * node.moments[term];
        return sum;
    }

    /** Host evaluation of one tile through the interaction lists. */
    template<int Order>
    inline void evaluateTileFar(
        InteractionLists const& lists,
        SourceCloud const& cloud,
        Vec3d const& cells,
        std::array<int, 3> const& offset,
        std::array<int, 3> const& tileCounts,
        std::size_t tile,
        float* output)
    {
        auto const i0 = static_cast<int>(tile % static_cast<std::size_t>(tileCounts[0]));
        auto const j0 = static_cast<int>(
            (tile / static_cast<std::size_t>(tileCounts[0])) % static_cast<std::size_t>(tileCounts[1]));
        auto const k0 = static_cast<int>(
            tile / (static_cast<std::size_t>(tileCounts[0]) * static_cast<std::size_t>(tileCounts[1])));
        Vec3d const originDx{
            static_cast<double>(offset[0] + i0 * tileGeometry::tileNx),
            static_cast<double>(offset[1] + j0 * tileGeometry::tileNy) * cells.y,
            static_cast<double>(offset[2] + k0 * tileGeometry::tileNz) * cells.z};

        for(std::size_t slot = 0; slot < tileGeometry::outputDimension; ++slot)
        {
            double sum = 0.0;
            for(int k = lists.farOffsets[tile]; k < lists.farOffsets[tile + 1]; ++k)
            {
                NodeAggregate const& node = lists.nodes[static_cast<std::size_t>(lists.farNodes[k])];
                Vec3d const staggered = staggeredPosition(slot, cells);
                sum += nodeFieldComponent<Order>(
                    node,
                    originDx.x + staggered.x,
                    originDx.y + staggered.y,
                    originDx.z + staggered.z,
                    static_cast<int>(slot % 6u));
            }
            for(int k = lists.nearOffsets[tile]; k < lists.nearOffsets[tile + 1]; ++k)
            {
                std::size_t const source = lists.nearSources[static_cast<std::size_t>(k)];
                Vec3d const sourceR{
                    cloud.positions[3 * source] - originDx.x,
                    cloud.positions[3 * source + 1] - originDx.y,
                    cloud.positions[3 * source + 2] - originDx.z};
                Vec3d const u{
                    cloud.velocities[3 * source],
                    cloud.velocities[3 * source + 1],
                    cloud.velocities[3 * source + 2]};
                sum += cloud.charges[source] * pointFieldSlot(slot, sourceR, u, cells);
            }
            output[slot] += static_cast<float>(sum);
        }
    }

    /** Direct host baseline: every source at every staggered output. */
    inline void evaluateTileDirect(
        SourceCloud const& cloud,
        Vec3d const& cells,
        std::array<int, 3> const& offset,
        std::array<int, 3> const& tileCounts,
        std::size_t tile,
        float* output)
    {
        auto const i0 = static_cast<int>(tile % static_cast<std::size_t>(tileCounts[0]));
        auto const j0 = static_cast<int>(
            (tile / static_cast<std::size_t>(tileCounts[0])) % static_cast<std::size_t>(tileCounts[1]));
        auto const k0 = static_cast<int>(
            tile / (static_cast<std::size_t>(tileCounts[0]) * static_cast<std::size_t>(tileCounts[1])));
        Vec3d const originDx{
            static_cast<double>(offset[0] + i0 * tileGeometry::tileNx),
            static_cast<double>(offset[1] + j0 * tileGeometry::tileNy) * cells.y,
            static_cast<double>(offset[2] + k0 * tileGeometry::tileNz) * cells.z};
        for(std::size_t slot = 0; slot < tileGeometry::outputDimension; ++slot)
        {
            double sum = 0.0;
            for(std::size_t source = 0; source < cloud.count; ++source)
            {
                double const charge = cloud.charges[source];
                if(charge == 0.0)
                    continue;
                Vec3d const sourceR{
                    cloud.positions[3 * source] - originDx.x,
                    cloud.positions[3 * source + 1] - originDx.y,
                    cloud.positions[3 * source + 2] - originDx.z};
                Vec3d const u{
                    cloud.velocities[3 * source],
                    cloud.velocities[3 * source + 1],
                    cloud.velocities[3 * source + 2]};
                sum += charge * pointFieldSlot(slot, sourceR, u, cells);
            }
            output[slot] += static_cast<float>(sum);
        }
    }
} // namespace picongpu::particles::neuralBunch::farfield
