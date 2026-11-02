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

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace nb = picongpu::particles::neuralBunch;
namespace ff = picongpu::particles::neuralBunch::farfield;

namespace
{
    constexpr double kPi = 3.14159265358979323846;
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

    /** Velocity distribution profiles for the bunch.
     *
     * Positions are always the same compact Gaussian (radius ``bunchRadius`` at the
     * domain centre); only the velocity field changes per profile, so the geometry
     * is comparable across profiles. ``u`` is the normalised momentum (gamma*beta,
     * c = 1), so a profile's mean gamma is ``sqrt(1 + |u|^2)``.
     */
    enum class Profile
    {
        Default, // meanU (0.5, 0.2, -0.1), isotropic spread (the legacy cold case)
        Axial,   // single velocity along +z at the requested gamma
        Oblique, // single velocity at 45 deg from +z (in the x-z plane) at the requested gamma
        Broad    // gamma uniform in [gammaMin, gammaMax], direction within coneDeg of +z
    };

    struct BunchSpec
    {
        Profile profile{Profile::Default};
        double gamma{10.0}; // Axial/Oblique target gamma
        double spread{0.02}; // Default isotropic delta-u
        double coneDeg{10.0}; // Broad: half-angle of the direction cone
        double gammaMin{1.0}; // Broad
        double gammaMax{100.0}; // Broad
    };

    /** Compact Gaussian bunch at the domain centre with a profile-driven velocity. */
    CloudBuffers makeBunch(
        std::size_t count,
        std::array<int, 3> const& tileCounts,
        double bunchRadius,
        BunchSpec const& spec,
        std::uint32_t seed)
    {
        CloudBuffers buffers;
        buffers.positions.resize(3 * count);
        buffers.velocities.resize(3 * count);
        buffers.charges.resize(count);
        std::mt19937 rng(seed);
        std::normal_distribution<double> gaussian(0.0, 1.0);
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);
        double const extent[3]{
            0.5 * tileCounts[0] * nb::tileGeometry::tileNx,
            0.5 * tileCounts[1] * nb::tileGeometry::tileNy,
            0.5 * tileCounts[2] * nb::tileGeometry::tileNz};
        double const singleSpeed = std::sqrt(spec.gamma * spec.gamma - 1.0);
        double const coneRad = spec.coneDeg * kPi / 180.0;
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
            switch(spec.profile)
            {
                case Profile::Axial:
                    buffers.velocities[3 * i + 0] = 0.0;
                    buffers.velocities[3 * i + 1] = 0.0;
                    buffers.velocities[3 * i + 2] = singleSpeed;
                    break;
                case Profile::Oblique:
                {
                    double const inv = 1.0 / std::sqrt(2.0);
                    buffers.velocities[3 * i + 0] = singleSpeed * inv;
                    buffers.velocities[3 * i + 1] = 0.0;
                    buffers.velocities[3 * i + 2] = singleSpeed * inv;
                    break;
                }
                case Profile::Broad:
                {
                    double const g = spec.gammaMin + (spec.gammaMax - spec.gammaMin) * uniform01(rng);
                    double const speed = std::sqrt(g * g - 1.0);
                    double const theta = coneRad * std::sqrt(uniform01(rng));
                    double const phi = 2.0 * kPi * uniform01(rng);
                    double const st = std::sin(theta);
                    buffers.velocities[3 * i + 0] = speed * st * std::cos(phi);
                    buffers.velocities[3 * i + 1] = speed * st * std::sin(phi);
                    buffers.velocities[3 * i + 2] = speed * std::cos(theta);
                    break;
                }
                case Profile::Default:
                default:
                    buffers.velocities[3 * i + 0] = 0.5 + spec.spread * uniform(rng);
                    buffers.velocities[3 * i + 1] = 0.2 + spec.spread * uniform(rng);
                    buffers.velocities[3 * i + 2] = -0.1 + spec.spread * uniform(rng);
                    break;
            }
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
        auto const buffers = makeBunch(sourceCount, tileCounts, 2.0, BunchSpec{}, 20261102u);
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

    __global__ void directTileKernel(
        int slotCount,
        nb::Vec3d cells,
        int const* offset,
        int const* tileCounts,
        int sourceCount,
        double const* positions,
        double const* velocities,
        double const* charges,
        float* tileValues)
    {
        int const flat = blockIdx.x * blockDim.x + threadIdx.x;
        int const tile = flat / slotCount;
        int const slot = flat % slotCount;
        if(tile >= tileCounts[0] * tileCounts[1] * tileCounts[2])
            return;
        int const i0 = tile % tileCounts[0];
        int const j0 = (tile / tileCounts[0]) % tileCounts[1];
        int const k0 = tile / (tileCounts[0] * tileCounts[1]);
        nb::Vec3d const origin{
            static_cast<double>(offset[0] + i0 * nb::tileGeometry::tileNx),
            static_cast<double>(offset[1] + j0 * nb::tileGeometry::tileNy) * cells.y,
            static_cast<double>(offset[2] + k0 * nb::tileGeometry::tileNz) * cells.z};
        int const component = slot % 6;
        double sum = 0.0;
        for(int source = 0; source < sourceCount; ++source)
        {
            double const charge = charges[source];
            if(charge == 0.0)
                continue;
            nb::Vec3d const sourceR{
                positions[3 * source] - origin.x,
                positions[3 * source + 1] - origin.y,
                positions[3 * source + 2] - origin.z};
            nb::Vec3d const u{
                velocities[3 * source],
                velocities[3 * source + 1],
                velocities[3 * source + 2]};
            sum += charge * nb::pointFieldSlot(static_cast<std::size_t>(slot), sourceR, u, cells);
        }
        (void) component;
        tileValues[static_cast<std::size_t>(tile) * slotCount + slot] += static_cast<float>(sum);
    }

    template<int Order>
    __global__ void farTileKernel(
        int slotCount,
        nb::Vec3d cells,
        int const* offset,
        int const* tileCounts,
        ff::NodeAggregate const* nodes,
        int const* farOffsets,
        int const* farIndices,
        int const* nearOffsets,
        std::uint32_t const* nearIndices,
        double const* positions,
        double const* velocities,
        double const* charges,
        float* tileValues)
    {
        int const flat = blockIdx.x * blockDim.x + threadIdx.x;
        int const tile = flat / slotCount;
        int const slot = flat % slotCount;
        if(tile >= tileCounts[0] * tileCounts[1] * tileCounts[2])
            return;
        int const i0 = tile % tileCounts[0];
        int const j0 = (tile / tileCounts[0]) % tileCounts[1];
        int const k0 = tile / (tileCounts[0] * tileCounts[1]);
        nb::Vec3d const origin{
            static_cast<double>(offset[0] + i0 * nb::tileGeometry::tileNx),
            static_cast<double>(offset[1] + j0 * nb::tileGeometry::tileNy) * cells.y,
            static_cast<double>(offset[2] + k0 * nb::tileGeometry::tileNz) * cells.z};
        nb::Vec3d const staggered = nb::staggeredPosition(static_cast<std::size_t>(slot), cells);
        int const component = slot % 6;
        double sum = 0.0;
        for(int k = farOffsets[tile]; k < farOffsets[tile + 1]; ++k)
        {
            int const node = farIndices[k];
            sum += ff::nodeFieldComponent<Order>(
                nodes[node],
                origin.x + staggered.x,
                origin.y + staggered.y,
                origin.z + staggered.z,
                component);
        }
        for(int k = nearOffsets[tile]; k < nearOffsets[tile + 1]; ++k)
        {
            std::size_t const source = nearIndices[k];
            nb::Vec3d const sourceR{
                positions[3 * source] - origin.x,
                positions[3 * source + 1] - origin.y,
                positions[3 * source + 2] - origin.z};
            nb::Vec3d const u{
                velocities[3 * source],
                velocities[3 * source + 1],
                velocities[3 * source + 2]};
            sum += charges[source] * nb::pointFieldSlot(static_cast<std::size_t>(slot), sourceR, u, cells);
        }
        tileValues[static_cast<std::size_t>(tile) * slotCount + slot] += static_cast<float>(sum);
    }

    double now()
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    /** Persist the direct field (the reference for every far configuration of a
     * case) so the O(S*tiles) direct kernel runs once per case, not once per run. */
    bool loadDirectReference(std::string const& file, double& directSeconds, std::vector<float>& values)
    {
        std::ifstream in(file, std::ios::binary);
        if(!in)
            return false;
        double ds = 0.0;
        in.read(reinterpret_cast<char*>(&ds), sizeof(double));
        in.seekg(0, std::ios::end);
        std::streamoff const size = in.tellg();
        in.seekg(8, std::ios::beg);
        std::size_t const nvals = size > static_cast<std::streamoff>(8)
                                       ? static_cast<std::size_t>(size - 8) / sizeof(float)
                                       : 0;
        values.assign(nvals, 0.0f);
        if(values.empty())
            return false;
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
        if(!in)
            return false;
        directSeconds = ds;
        return true;
    }

    void saveDirectReference(std::string const& file, double directSeconds, std::vector<float> const& values)
    {
        std::ofstream out(file, std::ios::binary);
        out.write(reinterpret_cast<char const*>(&directSeconds), sizeof(double));
        out.write(reinterpret_cast<char const*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    }

    char const* profileName(Profile p)
    {
        switch(p)
        {
            case Profile::Axial:
                return "axial";
            case Profile::Oblique:
                return "oblique";
            case Profile::Broad:
                return "broad";
            case Profile::Default:
            default:
                return "default";
        }
    }

    void benchmarkCase(
        std::size_t sourceCount,
        std::array<int, 3> const& tileCounts,
        std::string const& refFile,
        double sep,
        double splitFrac,
        ff::VelocitySplitMode splitMode,
        BunchSpec const& spec,
        std::vector<int> const& orders)
    {
        std::cout << std::unitbuf;
        std::array<int, 3> const offset{0, 0, 0};
        nb::Vec3d const cells{1.0, 1.0, 1.0};
        std::size_t const tiles = static_cast<std::size_t>(tileCounts[0]) * tileCounts[1] * tileCounts[2];
        std::size_t const slots = nb::tileGeometry::outputDimension;

        auto const buffers = makeBunch(sourceCount, tileCounts, 2.0, spec, 424242u);
        auto const cloud = buffers.cloud();
        ff::TreeParameters parameters;
        parameters.leafCapacity = 16;
        parameters.separationFactor = sep;
        parameters.velocitySplitFraction = splitFrac;
        parameters.velocitySplitMode = splitMode;
        double const buildStart = now();
        auto const lists = ff::buildInteractionLists(cloud, parameters, cells, offset, tileCounts);
        double const buildSeconds = now() - buildStart;
        double const farFraction
            = lists.statistics.directSourceEvaluations > 0
                ? 1.0 - static_cast<double>(lists.statistics.nearSourceEvaluations)
                          / static_cast<double>(lists.statistics.directSourceEvaluations)
                : 0.0;
        double maxVelDev = 0.0;
        for(auto const& node : lists.nodes)
            maxVelDev = std::max(maxVelDev, node.maxVelDev);
        std::cout << "profile " << profileName(spec.profile) << " gamma " << spec.gamma
                  << " spread " << spec.spread << " cone " << spec.coneDeg << " gmin " << spec.gammaMin
                  << " gmax " << spec.gammaMax << '\n';
        std::cout << "bench_sources " << sourceCount << " tiles " << tiles << " nodes " << lists.nodes.size()
                  << " far_node_tile_interactions " << lists.statistics.farNodeInteractions
                  << " near_source_pairs " << lists.statistics.nearSourceEvaluations
                  << " direct_source_pairs " << lists.statistics.directSourceEvaluations
                  << " max_far_nodes_per_tile " << lists.statistics.maxFarNodesPerTile
                  << " max_near_sources_per_tile " << lists.statistics.maxNearSourcesPerTile
                  << " host_build_seconds " << buildSeconds << " far_fraction " << farFraction
                  << " max_vel_dev " << maxVelDev
                  << " velocity_split_nodes " << lists.statistics.velocitySplitNodes
                  << " disjoint_max_dev " << lists.statistics.disjointMaxDeviation
                  << " disjoint_tiles " << lists.statistics.disjointTilesChecked
                  << " sep " << sep << " split " << splitFrac
                  << " split_mode " << static_cast<int>(splitMode) << '\n';

        auto checkCuda = [](cudaError_t status, char const* what)
        {
            if(status != cudaSuccess)
            {
                std::cerr << "CUDA error in " << what << ": " << cudaGetErrorString(status) << '\n';
                std::exit(EXIT_FAILURE);
            }
        };

        double* devicePositions = nullptr;
        double* deviceVelocities = nullptr;
        double* deviceCharges = nullptr;
        ff::NodeAggregate* deviceNodes = nullptr;
        int* deviceFarOffsets = nullptr;
        int* deviceFarIndices = nullptr;
        int* deviceNearOffsets = nullptr;
        std::uint32_t* deviceNearIndices = nullptr;
        int* deviceOffset = nullptr;
        int* deviceTileCounts = nullptr;
        float* deviceTileValues = nullptr;

        checkCuda(cudaMalloc(&devicePositions, 3 * sourceCount * sizeof(double)), "positions");
        checkCuda(cudaMalloc(&deviceVelocities, 3 * sourceCount * sizeof(double)), "velocities");
        checkCuda(cudaMalloc(&deviceCharges, sourceCount * sizeof(double)), "charges");
        checkCuda(cudaMemcpy(
                     devicePositions,
                     buffers.positions.data(),
                     3 * sourceCount * sizeof(double),
                     cudaMemcpyHostToDevice),
                 "positions copy");
        checkCuda(cudaMemcpy(
                     deviceVelocities,
                     buffers.velocities.data(),
                     3 * sourceCount * sizeof(double),
                     cudaMemcpyHostToDevice),
                 "velocities copy");
        checkCuda(cudaMemcpy(
                     deviceCharges,
                     buffers.charges.data(),
                     sourceCount * sizeof(double),
                     cudaMemcpyHostToDevice),
                 "charges copy");
        checkCuda(
            cudaMalloc(&deviceNodes, std::max<std::size_t>(1, lists.nodes.size()) * sizeof(ff::NodeAggregate)),
            "nodes");
        checkCuda(cudaMalloc(&deviceFarOffsets, std::max<std::size_t>(1, lists.farOffsets.size()) * sizeof(int)), "farOffsets");
        checkCuda(cudaMalloc(&deviceFarIndices, std::max<std::size_t>(1, lists.farNodes.size()) * sizeof(int)), "farIndices");
        checkCuda(cudaMalloc(&deviceNearOffsets, std::max<std::size_t>(1, lists.nearOffsets.size()) * sizeof(int)), "nearOffsets");
        checkCuda(
            cudaMalloc(
                &deviceNearIndices,
                std::max<std::size_t>(1, lists.nearSources.size()) * sizeof(std::uint32_t)),
            "nearIndices");
        checkCuda(cudaMalloc(&deviceOffset, 3 * sizeof(int)), "offset");
        checkCuda(cudaMalloc(&deviceTileCounts, 3 * sizeof(int)), "tileCounts");
        checkCuda(cudaMalloc(&deviceTileValues, tiles * slots * sizeof(float)), "tileValues");
        checkCuda(cudaMemcpy(
                     deviceNodes,
                     lists.nodes.data(),
                     lists.nodes.size() * sizeof(ff::NodeAggregate),
                     cudaMemcpyHostToDevice),
                 "nodes copy");
        checkCuda(cudaMemcpy(
                     deviceFarOffsets,
                     lists.farOffsets.data(),
                     lists.farOffsets.size() * sizeof(int),
                     cudaMemcpyHostToDevice),
                 "farOffsets copy");
        if(!lists.farNodes.empty())
            checkCuda(cudaMemcpy(
                         deviceFarIndices,
                         lists.farNodes.data(),
                         lists.farNodes.size() * sizeof(int),
                         cudaMemcpyHostToDevice),
                     "farIndices copy");
        checkCuda(cudaMemcpy(
                     deviceNearOffsets,
                     lists.nearOffsets.data(),
                     lists.nearOffsets.size() * sizeof(int),
                     cudaMemcpyHostToDevice),
                 "nearOffsets copy");
        if(!lists.nearSources.empty())
            checkCuda(cudaMemcpy(
                         deviceNearIndices,
                         lists.nearSources.data(),
                         lists.nearSources.size() * sizeof(std::uint32_t),
                         cudaMemcpyHostToDevice),
                     "nearIndices copy");
        checkCuda(cudaMemcpy(deviceOffset, offset.data(), 3 * sizeof(int), cudaMemcpyHostToDevice), "offset copy");
        checkCuda(
            cudaMemcpy(deviceTileCounts, tileCounts.data(), 3 * sizeof(int), cudaMemcpyHostToDevice),
            "tileCounts copy");

        int const block = 256;
        int const total = static_cast<int>(tiles * slots);
        int const grid = (total + block - 1) / block;
        auto hostFetch = [&]()
        {
            std::vector<float> values(tiles * slots);
            checkCuda(
                cudaMemcpy(
                    values.data(),
                    deviceTileValues,
                    tiles * slots * sizeof(float),
                    cudaMemcpyDeviceToHost),
                "fetch");
            return values;
        };
        auto runDirect = [&]()
        {
            checkCuda(cudaMemset(deviceTileValues, 0, tiles * slots * sizeof(float)), "zero");
            checkCuda(cudaDeviceSynchronize(), "zero sync");
            double const start = now();
            directTileKernel<<<grid, block, 0, 0>>>(
                static_cast<int>(slots),
                cells,
                deviceOffset,
                deviceTileCounts,
                static_cast<int>(sourceCount),
                devicePositions,
                deviceVelocities,
                deviceCharges,
                deviceTileValues);
            checkCuda(cudaDeviceSynchronize(), "direct");
            return now() - start;
        };
        // The direct field is the reference for every far configuration of this
        // (source, tile) case, so it is computed once and cached to `refFile`.
        std::vector<float> direct;
        double directSeconds = 1.0e300;
        int const directReps = sourceCount >= 50000u ? 1 : 2;
        bool const haveRef = !refFile.empty() && loadDirectReference(refFile, directSeconds, direct);
        if(!haveRef)
        {
            // The direct kernel is O(S * tiles); large cases get a single timed
            // run (no warm-up repetition) so the proof stays inside a sane wall time.
            if(sourceCount < 50000u)
                (void) runDirect();
            for(int rep = 0; rep < directReps; ++rep)
                directSeconds = std::min(directSeconds, runDirect());
            direct = hostFetch();
            if(!refFile.empty())
                saveDirectReference(refFile, directSeconds, direct);
        }
        if(direct.size() != tiles * slots)
        {
            std::cerr << "direct reference size mismatch: " << direct.size() << " != " << tiles * slots
                      << " (wrong --ref for this case?)\n";
            std::exit(EXIT_FAILURE);
        }
        std::cout << "direct_seconds " << directSeconds << " direct_reps " << directReps
                  << " direct_cached " << (haveRef ? 1 : 0) << "\n";

        auto runFar = [&](int order)
        {
            checkCuda(cudaMemset(deviceTileValues, 0, tiles * slots * sizeof(float)), "zero");
            checkCuda(cudaDeviceSynchronize(), "zero sync");
            double const start = now();
            if(order == 2)
                farTileKernel<2><<<grid, block, 0, 0>>>(
                    static_cast<int>(slots),
                    cells,
                    deviceOffset,
                    deviceTileCounts,
                    deviceNodes,
                    deviceFarOffsets,
                    deviceFarIndices,
                    deviceNearOffsets,
                    deviceNearIndices,
                    devicePositions,
                    deviceVelocities,
                    deviceCharges,
                    deviceTileValues);
            else if(order == 4)
                farTileKernel<4><<<grid, block, 0, 0>>>(
                    static_cast<int>(slots),
                    cells,
                    deviceOffset,
                    deviceTileCounts,
                    deviceNodes,
                    deviceFarOffsets,
                    deviceFarIndices,
                    deviceNearOffsets,
                    deviceNearIndices,
                    devicePositions,
                    deviceVelocities,
                    deviceCharges,
                    deviceTileValues);
            else
                farTileKernel<6><<<grid, block, 0, 0>>>(
                    static_cast<int>(slots),
                    cells,
                    deviceOffset,
                    deviceTileCounts,
                    deviceNodes,
                    deviceFarOffsets,
                    deviceFarIndices,
                    deviceNearOffsets,
                    deviceNearIndices,
                    devicePositions,
                    deviceVelocities,
                    deviceCharges,
                    deviceTileValues);
            checkCuda(cudaDeviceSynchronize(), "far");
            return now() - start;
        };

        for(int const order : orders)
        {
            (void) runFar(order);
            double best = 1.0e300;
            for(int rep = 0; rep < 2; ++rep)
                best = std::min(best, runFar(order));
            auto const far = hostFetch();
            auto const metric = compareFields(far, direct, tiles);
            std::cout << "far_order " << order << " seconds " << best << " speedup " << directSeconds / best
                      << " relL2_E " << metric.groupE << " relL2_cB " << metric.groupB;
            for(std::size_t component = 0; component < 6; ++component)
                std::cout << " _relL2_" << componentName(component) << ' ' << metric.component[component];
            std::cout << " work_units " << lists.statistics.totalWorkUnits(order) << '\n';
        }

        checkCuda(cudaFree(devicePositions), "free");
        checkCuda(cudaFree(deviceVelocities), "free");
        checkCuda(cudaFree(deviceCharges), "free");
        checkCuda(cudaFree(deviceNodes), "free");
        checkCuda(cudaFree(deviceFarOffsets), "free");
        checkCuda(cudaFree(deviceFarIndices), "free");
        checkCuda(cudaFree(deviceNearOffsets), "free");
        checkCuda(cudaFree(deviceNearIndices), "free");
        checkCuda(cudaFree(deviceOffset), "free");
        checkCuda(cudaFree(deviceTileCounts), "free");
        checkCuda(cudaFree(deviceTileValues), "free");
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

    // Default = best measured configuration (T28): tightening the admissibility
    // factor to 3.0 brings the cB gate under 1e-3 on both the dense 1e5 and
    // 2000-source cases while keeping E well under the gate, and order 2 is the
    // lowest-error order (higher orders worsen E on the dense case). Override
    // with --sep / --split / --orders to reproduce the full experiment grid.
    std::size_t sources = 100000u;
    std::string refFile;
    double sep = 3.0;
    double splitFrac = 0.0;
    ff::VelocitySplitMode splitMode = ff::VelocitySplitMode::Frac;
    BunchSpec spec;
    std::vector<int> orders{2};
    for(int i = 2; i < argc; ++i)
    {
        std::string const arg = argv[i];
        auto const next = [&]() -> std::string
        {
            if(i + 1 >= argc)
            {
                std::cerr << "missing value for " << arg << '\n';
                std::exit(EXIT_FAILURE);
            }
            ++i;
            return argv[i];
        };
        if(arg == "--ref")
            refFile = next();
        else if(arg == "--sep")
            sep = std::stod(next());
        else if(arg == "--split")
            splitFrac = std::stod(next());
        else if(arg == "--split-mode")
        {
            std::string const m = next();
            if(m == "frac")
                splitMode = ff::VelocitySplitMode::Frac;
            else if(m == "abs")
                splitMode = ff::VelocitySplitMode::Abs;
            else if(m == "gamma")
                splitMode = ff::VelocitySplitMode::Gamma;
            else
            {
                std::cerr << "unknown --split-mode " << m << " (frac|abs|gamma)\n";
                std::exit(EXIT_FAILURE);
            }
        }
        else if(arg == "--profile")
        {
            std::string const p = next();
            if(p == "default")
                spec.profile = Profile::Default;
            else if(p == "axial")
                spec.profile = Profile::Axial;
            else if(p == "oblique")
                spec.profile = Profile::Oblique;
            else if(p == "broad")
                spec.profile = Profile::Broad;
            else
            {
                std::cerr << "unknown --profile " << p << " (default|axial|oblique|broad)\n";
                std::exit(EXIT_FAILURE);
            }
        }
        else if(arg == "--gamma")
            spec.gamma = std::stod(next());
        else if(arg == "--spread")
            spec.spread = std::stod(next());
        else if(arg == "--cone")
            spec.coneDeg = std::stod(next());
        else if(arg == "--gmin")
            spec.gammaMin = std::stod(next());
        else if(arg == "--gmax")
            spec.gammaMax = std::stod(next());
        else if(arg == "--orders")
        {
            orders.clear();
            std::stringstream ss(next());
            std::string item;
            while(std::getline(ss, item, ','))
                if(!item.empty())
                    orders.push_back(std::stoi(item));
        }
        else
            sources = static_cast<std::size_t>(std::stoull(arg));
    }
    std::array<int, 3> const tileCounts{16, 16, 32}; // 128 x 128 x 128 cells
    benchmarkCase(sources, tileCounts, refFile, sep, splitFrac, splitMode, spec, orders);
    return EXIT_SUCCESS;
}
