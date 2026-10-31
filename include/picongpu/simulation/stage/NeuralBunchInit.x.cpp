/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include "picongpu/simulation/stage/NeuralBunchInit.hpp"

#include "picongpu/defines.hpp"

#if defined(PIC_ENABLE_NEURAL_BUNCH) && PIC_ENABLE_NEURAL_BUNCH

#    include "picongpu/particles/param.hpp"
#    include "picongpu/param/neuralBunch.param"
#    include "picongpu/fields/FieldB.hpp"
#    include "picongpu/fields/FieldE.hpp"
#    include "picongpu/particles/neuralBunch/Inference.hpp"
#    include "picongpu/particles/neuralBunch/Model.hpp"
#    include "picongpu/particles/neuralBunch/SourceFile.hpp"
#    include "picongpu/particles/neuralBunch/TileGeometry.hpp"
#    include "picongpu/plugins/output/WriteSpeciesCommon.hpp"

#    include <pmacc/alpakaHelper/Device.hpp>
#    include <pmacc/alpakaHelper/acc.hpp>
#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/Environment.hpp>
#    include <pmacc/mappings/simulation/SubGrid.hpp>
#    include <pmacc/meta/ForEach.hpp>
#    include <pmacc/meta/conversion/MakeSeq.hpp>
#    include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#    include <pmacc/particles/meta/FindByNameOrType.hpp>
#    include <pmacc/particles/operations/splitIntoListOfFrames.kernel>
#    include <pmacc/particles/ParticleDescription.hpp>
#    include <pmacc/traits/HasIdentifier.hpp>
#    include <pmacc/traits/Resolve.hpp>

#    include <boost/mpl/placeholders.hpp>

#    include <algorithm>
#    include <array>
#    include <chrono>
#    include <cmath>
#    include <cstdint>
#    include <limits>
#    include <sstream>
#    include <stdexcept>
#    include <string>
#    include <vector>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            namespace neuralBunchDetail
            {
                using namespace picongpu::particles::neuralBunch;
                using picongpu::neuralBunch::NeuralBunchParam;
                namespace tg = picongpu::particles::neuralBunch::tileGeometry;

                static_assert(
                    tg::tileNx == static_cast<int>(kTileNx) && tg::tileNy == static_cast<int>(kTileNy)
                        && tg::tileNz == static_cast<int>(kTileNz),
                    "neuralBunch tile geometry must match MODEL-FORMAT v1.");

                constexpr double electronMassSI = 9.1093837015e-31;
                constexpr double speedOfLightSI = 299792458.0;

                std::string num(double const value)
                {
                    std::ostringstream stream;
                    stream.precision(12);
                    stream << value;
                    return stream.str();
                }

                struct Statistics
                {
                    std::uint64_t globalParticles = 0;
                    std::uint64_t neuralPairs = 0;
                    std::uint64_t referencePairs = 0;
                    std::uint64_t localParticles = 0;
                    double globalWeight = 0.0;
                    double localWeight = 0.0;
                    double globalCharge = 0.0;
                    double decodedBytes = 0.0;
                    double ioSeconds = 0.0;
                    double inferenceSeconds = 0.0;
                    double fieldStoreSeconds = 0.0;
                    double particleSeconds = 0.0;
                    double maxLatent = 0.0;
                    double fieldSums[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    std::uint64_t nonFiniteFields = 0;
                };

                constexpr double decodedBytesPerParticle = 3.0 * 8.0 + 3.0 * 8.0 + 8.0 + 8.0 + 8.0 + 8.0 + 1.0;

                /** Build tile-local sources for one chunk. */
                std::vector<Source> makeSources(
                    ParticleChunk const& chunk,
                    std::array<double, 3> const& tileOriginPhysical,
                    std::array<double, 3> const& cellSizeSI)
                {
                    std::vector<Source> sources;
                    sources.reserve(chunk.size());
                    for(std::size_t i = 0; i < chunk.size(); ++i)
                    {
                        if(!chunk.valid[i])
                            continue;
                        Source source;
                        for(int axis = 0; axis < 3; ++axis)
                        {
                            source.r[axis] = (chunk.position[3 * i + axis] - tileOriginPhysical[axis])
                                             / cellSizeSI[axis];
                            source.u[axis] = chunk.u[3 * i + axis];
                        }
                        source.weight = chunk.weight[i];
                        // Cubic PIConGPU cells in Phase A, matching the model's dx/dy/dz spacing.
                        source.aY = 1.0;
                        source.aZ = 1.0;
                        sources.push_back(source);
                    }
                    return sources;
                }

                template<typename T_Species>
                struct ParticleInserter
                {
                    using SpeciesType = T_Species;
                    using FrameType = typename SpeciesType::FrameType;
                    using ParticleDescription = typename FrameType::ParticleDescription;
                    using ParticleAttributeList = typename FrameType::ValueTypeSeq;

                    using TypesToDelete = pmacc::mp_list<multiMask, localCellIdx>;
                    using ParticleCleanedAttributeList
                        = typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type;
                    using ParticleNewAttributeList = MakeSeq_t<ParticleCleanedAttributeList, totalCellIdx>;
                    using NewParticleDescription
                        = typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type;

                    using LoadFrame = Frame<OperatorCreateVectorBox, NewParticleDescription>;
                    using LoadBuffer = Frame<OperatorCreateAlpakaBuffer, NewParticleDescription>;

                    using PositionKey = position<position_pic>;
                    using PositionValue = typename pmacc::traits::Resolve<PositionKey>::type::type;
                    using MomentumValue = typename pmacc::traits::Resolve<momentum>::type::type;
                    using WeightingValue = typename pmacc::traits::Resolve<weighting>::type::type;
                    using TotalCellValue = typename pmacc::traits::Resolve<totalCellIdx>::type::type;

                    static constexpr bool hasId
                        = pmacc::mp_contains<typename NewParticleDescription::ValueTypeSeq, particleId>::value;
                    using IdValue = typename pmacc::traits::Resolve<particleId>::type::type;

                    static void insertChunk(
                        ParticleChunk const& chunk,
                        std::array<int, 3> const& localOffset,
                        std::array<int, 3> const& localSize,
                        std::array<double, 3> const& cellSizeSI,
                        double const momentumFactor,
                        MappingDesc const& cellDescription,
                        Statistics& stats)
                    {
                        std::size_t accepted = 0;
                        for(std::size_t i = 0; i < chunk.size(); ++i)
                        {
                            if(!chunk.valid[i])
                                continue;
                            bool inside = true;
                            for(int axis = 0; axis < 3; ++axis)
                            {
                                double const cellIndex = std::floor(
                                    chunk.position[3 * i + axis] / cellSizeSI[axis]);
                                if(!std::isfinite(cellIndex)
                                   || cellIndex < static_cast<double>(std::numeric_limits<int>::min())
                                   || cellIndex > static_cast<double>(std::numeric_limits<int>::max())
                                   || static_cast<int>(cellIndex) < localOffset[axis]
                                   || static_cast<int>(cellIndex) >= localOffset[axis] + localSize[axis])
                                    inside = false;
                            }
                            if(inside)
                                ++accepted;
                        }
                        if(accepted == 0)
                            return;

                        DataConnector& dc = Environment<>::get().DataConnector();
                        auto species = dc.get<SpeciesType>(FrameType::getName());
                        auto const& subGrid = Environment<simDim>::get().SubGrid();
                        DataSpace<simDim> const cellOffsetToTotalDomain
                            = subGrid.getLocalDomain().offset + subGrid.getGlobalDomain().offset;

                        LoadBuffer buffers;
                        LoadFrame frame;
                        pmacc::meta::ForEach<
                            typename NewParticleDescription::ValueTypeSeq,
                            MallocMappedMemory<boost::mpl::_1>>
                            mallocMem;
                        mallocMem(buffers, frame, accepted);

                        auto* positionPtr = frame.getIdentifier(PositionKey{}).getPointer();
                        auto* momentumPtr = frame.getIdentifier(momentum{}).getPointer();
                        auto* weightingPtr = frame.getIdentifier(weighting{}).getPointer();
                        auto* totalCellPtr = frame.getIdentifier(totalCellIdx{}).getPointer();
                        IdValue* id = nullptr;
                        if constexpr(hasId)
                            id = frame.getIdentifier(particleId{}).getPointer();

                        std::size_t target = 0;
                        for(std::size_t i = 0; i < chunk.size(); ++i)
                        {
                            if(!chunk.valid[i])
                                continue;
                            std::array<int, 3> cell{};
                            bool inside = true;
                            for(int axis = 0; axis < 3; ++axis)
                            {
                                double const positionSI = chunk.position[3 * i + axis];
                                double const cellIndex = std::floor(positionSI / cellSizeSI[axis]);
                                if(!std::isfinite(cellIndex)
                                   || cellIndex < static_cast<double>(std::numeric_limits<int>::min())
                                   || cellIndex > static_cast<double>(std::numeric_limits<int>::max()))
                                {
                                    inside = false;
                                    break;
                                }
                                cell[axis] = static_cast<int>(cellIndex);
                                if(cell[axis] < localOffset[axis]
                                   || cell[axis] >= localOffset[axis] + localSize[axis])
                                    inside = false;
                            }
                            if(!inside)
                                continue;

                            PositionValue inCell;
                            for(int axis = 0; axis < 3; ++axis)
                            {
                                double const positionSI = chunk.position[3 * i + axis];
                                double fraction
                                    = positionSI / cellSizeSI[axis] - static_cast<double>(cell[axis]);
                                if(fraction >= 1.0)
                                    fraction = 0.0;
                                if(fraction < 0.0)
                                    fraction = 0.0;
                                inCell[axis] = static_cast<float_X>(fraction);
                            }
                            positionPtr[target] = inCell;

                            double const weight = chunk.weight[i];
                            momentumPtr[target] = MomentumValue(
                                static_cast<float_X>(chunk.u[3 * i + 0] * weight * momentumFactor),
                                static_cast<float_X>(chunk.u[3 * i + 1] * weight * momentumFactor),
                                static_cast<float_X>(chunk.u[3 * i + 2] * weight * momentumFactor));
                            weightingPtr[target] = static_cast<WeightingValue>(weight);

                            TotalCellValue totalCellIndex;
                            for(int axis = 0; axis < 3; ++axis)
                                totalCellIndex[axis] = cell[axis];
                            totalCellPtr[target] = totalCellIndex;

                            if constexpr(hasId)
                                id[target] = chunk.id.empty()
                                                 ? static_cast<IdValue>(chunk.begin + i + 1u)
                                                 : static_cast<IdValue>(chunk.id[i]);
                            ++target;
                        }

                        if(target != accepted)
                            throw std::runtime_error(
                                "neuralBunch: accepted particle count changed between passes.");

                        pmacc::particles::operations::splitIntoListOfFrames(
                            *species,
                            frame,
                            static_cast<uint32_t>(accepted),
                            cellOffsetToTotalDomain,
                            totalCellIdx{},
                            cellDescription,
                            picLog::PHYSICS());

                        stats.localParticles += accepted;
                        for(std::size_t i = 0; i < chunk.size(); ++i)
                        {
                            if(!chunk.valid[i])
                                continue;
                            bool inside = true;
                            for(int axis = 0; axis < 3; ++axis)
                            {
                                double const cellIndex = std::floor(
                                    chunk.position[3 * i + axis] / cellSizeSI[axis]);
                                if(static_cast<int>(cellIndex) < localOffset[axis]
                                   || static_cast<int>(cellIndex) >= localOffset[axis] + localSize[axis])
                                    inside = false;
                            }
                            if(inside)
                                stats.localWeight += chunk.weight[i];
                        }
                    }
                };

                template<typename T_Species>
                void readAndInsertParticles(
                    SourceReader& reader,
                    std::uint64_t const chunkSize,
                    std::array<int, 3> const& localOffset,
                    std::array<int, 3> const& localSize,
                    std::array<double, 3> const& cellSizeSI,
                    MappingDesc const& cellDescription,
                    Statistics& stats)
                {
                    double const momentumFactor
                        = electronMassSI * speedOfLightSI / (sim.unit.mass() * sim.unit.speed());
                    auto const start = std::chrono::steady_clock::now();
                    reader.streamAll(
                        [&](ParticleChunk const& chunk)
                        {
                            stats.decodedBytes += chunk.size() * decodedBytesPerParticle;
                            double chunkWeight = 0.0;
                            for(std::size_t i = 0; i < chunk.size(); ++i)
                                if(chunk.valid[i])
                                    chunkWeight += chunk.weight[i];
                            stats.globalWeight += chunkWeight;
                            ParticleInserter<T_Species>::insertChunk(
                                chunk,
                                localOffset,
                                localSize,
                                cellSizeSI,
                                momentumFactor,
                                cellDescription,
                                stats);
                        },
                        chunkSize);
                    stats.particleSeconds += std::chrono::duration<double>(
                                                 std::chrono::steady_clock::now() - start)
                                                 .count();
                }

                template<typename T_Species>
                Statistics run(
                    NeuralBunchParam const& params,
                    uint32_t const step,
                    MappingDesc const& cellDescription)
                {
                    Statistics stats;

                    auto const& subGrid = Environment<simDim>::get().SubGrid();
                    DataSpace<simDim> const localDomainOffset
                        = subGrid.getLocalDomain().offset + subGrid.getGlobalDomain().offset;
                    DataSpace<simDim> const localDomainSize = subGrid.getLocalDomain().size;
                    std::array<int, 3> offset{};
                    std::array<int, 3> size{};
                    std::array<double, 3> cellSizeSI{};
                    for(uint32_t axis = 0; axis < simDim; ++axis)
                    {
                        offset[axis] = localDomainOffset[axis];
                        size[axis] = localDomainSize[axis];
                        cellSizeSI[axis] = sim.pic.getCellSize()[axis] * sim.unit.length();
                    }
                    if constexpr(simDim != 3)
                        throw std::runtime_error(
                            "neuralBunch: the neural bunch initializer requires a 3D simulation.");

                    double const dxMeters = cellSizeSI[0];
                    double const unitE = sim.unit.eField();
                    double const unitB = sim.unit.bField();

                    Model const model = Model::load(params.modelPath);
                    ModelMetadata const& meta = model.metadata();

                    auto& device = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();
                    auto queue = device.makeQueue(pmacc::ComputeQueuePolicies);
                    auto const exec = pmacc::ComputeExecutor;
                    auto const tensors = model.loadTensors(queue);

                    auto const grid = tg::enumerateTiles(offset, size);
                    std::vector<std::vector<double>> latents(
                        grid.tiles.size(),
                        std::vector<double>(meta.latent, 0.0));

                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto fieldE = dc.get<FieldE>(FieldE::getName());
                    auto fieldB = dc.get<FieldB>(FieldB::getName());
                    auto hostE = fieldE->getHostDataBox().shift(fieldE->getGridBuffer().getGridLayout().guardSizeND());
                    auto hostB = fieldB->getHostDataBox().shift(fieldB->getGridBuffer().getGridLayout().guardSizeND());

                    MPI_Comm const comm
                        = Environment<simDim>::get().GridController().getCommunicator().getMPIComm();
                    int const rank = Environment<simDim>::get().GridController().getGlobalRank();

                    {
                        SourceReader reader(params.filename, params.iteration, params.species, comm);
                        SourceMetadata const& sourceMeta = reader.metadata();
                        if(params.timePolicy != 0)
                            throw std::runtime_error(
                                "neuralBunch: Phase A requires timePolicy == 0 (source time equals injection "
                                "time).");
                        if(std::abs(sourceMeta.time - 0.0) > 1.0e-12 * (1.0 + std::abs(sourceMeta.time)))
                            throw std::runtime_error(
                                "neuralBunch: timePolicy 0 requires a source iteration time of zero for this "
                                "phase, got "
                                + num(sourceMeta.time) + " s.");

                        auto const ioStart = std::chrono::steady_clock::now();
                        reader.streamAll(
                            [&](ParticleChunk const& chunk)
                            {
                                stats.decodedBytes += chunk.size() * decodedBytesPerParticle;
                                stats.neuralPairs += chunk.size() * grid.tiles.size();
                                for(std::size_t tileIndex = 0; tileIndex < grid.tiles.size(); ++tileIndex)
                                {
                                    std::array<double, 3> tileOriginPhysical{};
                                    for(int axis = 0; axis < 3; ++axis)
                                        tileOriginPhysical[axis]
                                            = static_cast<double>(grid.tiles[tileIndex].origin[axis])
                                              * cellSizeSI[axis];
                                    auto sources = makeSources(chunk, tileOriginPhysical, cellSizeSI);
                                    if(sources.empty())
                                        continue;
                                    auto const encodeStart = std::chrono::steady_clock::now();
                                    auto hidden = runEncoder(queue, exec, meta, tensors, sources);
                                    stats.inferenceSeconds += std::chrono::duration<double>(
                                                                  std::chrono::steady_clock::now() - encodeStart)
                                                                  .count();
                                    auto& latent = latents[tileIndex];
                                    for(std::size_t i = 0; i < sources.size(); ++i)
                                        for(std::size_t k = 0; k < meta.latent; ++k)
                                        {
                                            double const contribution
                                                = -sources[i].weight
                                                  * static_cast<double>(
                                                      hidden[i * meta.latent + k]);
                                            latent[k] += contribution;
                                            stats.maxLatent = std::max(stats.maxLatent, std::abs(latent[k]));
                                        }
                                }
                            },
                            params.chunkSize);
                        stats.ioSeconds += std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() - ioStart)
                                               .count();
                        stats.globalParticles = sourceMeta.globalCount;
                    }

                    auto const storeStart = std::chrono::steady_clock::now();
                    for(std::size_t tileIndex = 0; tileIndex < grid.tiles.size(); ++tileIndex)
                    {
                        auto const& tile = grid.tiles[tileIndex];
                        std::vector<float> const latentFloat(
                            latents[tileIndex].begin(),
                            latents[tileIndex].end());
                        auto const values = decodeFromLatent(
                            queue,
                            exec,
                            meta,
                            tensors,
                            latentFloat,
                            1.0,
                            1.0,
                            FieldNormalization{dxMeters});
                        for(int k = 0; k < tg::tileNz; ++k)
                            for(int j = 0; j < tg::tileNy; ++j)
                                for(int i = 0; i < tg::tileNx; ++i)
                                {
                                    std::array<int, 3> cell{
                                        tile.origin[0] + i,
                                        tile.origin[1] + j,
                                        tile.origin[2] + k};
                                    if(!tg::insideLocalDomain(cell, offset, size))
                                        continue;
                                    DataSpace<simDim> localCell;
                                    for(int axis = 0; axis < 3; ++axis)
                                        localCell[axis] = cell[axis] - offset[axis];
                                    auto& e = hostE(localCell);
                                    auto& b = hostB(localCell);
                                    for(int component = 0; component < 3; ++component)
                                    {
                                        e[component] += static_cast<float_X>(tg::normalizedEToInternal(
                                            values[tg::outputIndex(i, j, k, component)],
                                            dxMeters,
                                            unitE));
                                        b[component] += static_cast<float_X>(tg::normalizedBToInternal(
                                            values[tg::outputIndex(i, j, k, component + 3)],
                                            dxMeters,
                                            unitB));
                                    }
                                }
                    }
                    fieldE->syncToDevice();
                    fieldB->syncToDevice();
                    stats.fieldStoreSeconds += std::chrono::duration<double>(
                                                   std::chrono::steady_clock::now() - storeStart)
                                                   .count();

                    // Local field diagnostics over owned cells only.
                    for(int k = 0; k < size[2]; ++k)
                        for(int j = 0; j < size[1]; ++j)
                            for(int i = 0; i < size[0]; ++i)
                            {
                                DataSpace<simDim> localCell;
                                localCell[0] = i;
                                localCell[1] = j;
                                localCell[2] = k;
                                auto const& e = hostE(localCell);
                                auto const& b = hostB(localCell);
                                for(int component = 0; component < 3; ++component)
                                {
                                    double const ev = static_cast<double>(e[component]);
                                    double const bv = static_cast<double>(b[component]);
                                    stats.fieldSums[component] += ev;
                                    stats.fieldSums[component + 3] += bv;
                                    if(!std::isfinite(ev) || !std::isfinite(bv))
                                        ++stats.nonFiniteFields;
                                }
                            }

                    // Pass 2: reopen and insert only locally owned particles.
                    {
                        SourceReader reader(params.filename, params.iteration, params.species, comm);
                        auto const start = std::chrono::steady_clock::now();
                        readAndInsertParticles<typename neuralBunch::Species>(
                            reader,
                            params.chunkSize,
                            offset,
                            size,
                            cellSizeSI,
                            cellDescription,
                            stats);
                        stats.ioSeconds += std::chrono::duration<double>(
                                               std::chrono::steady_clock::now() - start)
                                               .count();
                    }

                    log<picLog::PHYSICS>(
                        "neuralBunch: rank=%1% localFieldSums=[%2% %3% %4% %5% %6% %7%] localNonFinite=%8%")
                        % rank % num(stats.fieldSums[0]) % num(stats.fieldSums[1]) % num(stats.fieldSums[2])
                        % num(stats.fieldSums[3]) % num(stats.fieldSums[4]) % num(stats.fieldSums[5])
                        % stats.nonFiniteFields;

                    unsigned long long localCount = stats.localParticles;
                    unsigned long long totalCount = 0;
                    MPI_Allreduce(&localCount, &totalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
                    double localWeight = stats.localWeight;
                    double totalWeight = 0.0;
                    MPI_Allreduce(&localWeight, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, comm);
                    double fieldSums[6]{};
                    MPI_Allreduce(stats.fieldSums, fieldSums, 6, MPI_DOUBLE, MPI_SUM, comm);
                    unsigned long long nonFinite = stats.nonFiniteFields;
                    unsigned long long totalNonFinite = 0;
                    MPI_Allreduce(&nonFinite, &totalNonFinite, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);

                    double const chargePerWeight = -1.602176634e-19;
                    double const globalCharge = chargePerWeight * stats.globalWeight;
                    log<picLog::PHYSICS>(
                        "neuralBunch: rank=%1% step=%2% tiles=%3% neuralPairs=%4% referencePairs=%5% "
                        "globalParticles=%6% localInserted=%7% totalInserted=%8% globalWeight=%9%")
                        % rank % step % grid.tiles.size() % stats.neuralPairs % stats.referencePairs
                        % stats.globalParticles % stats.localParticles % totalCount % num(stats.globalWeight);
                    log<picLog::PHYSICS>(
                        "neuralBunch: rank=%1% totalInsertedWeight=%2% globalCharge=%3% "
                        "fieldSums=[%4% %5% %6% %7% %8% %9%]")
                        % rank % num(totalWeight) % num(globalCharge) % num(fieldSums[0]) % num(fieldSums[1])
                        % num(fieldSums[2]) % num(fieldSums[3]) % num(fieldSums[4]) % num(fieldSums[5]);
                    log<picLog::PHYSICS>(
                        "neuralBunch: rank=%1% nonFiniteFields=%2% maxLatent=%3%")
                        % rank % totalNonFinite % num(stats.maxLatent);
                    log<picLog::PHYSICS>(
                        "neuralBunch: rank=%1% decodedBytes=%2% ioSeconds=%3% inferenceSeconds=%4% "
                        "fieldStoreSeconds=%5% particleSeconds=%6%")
                        % rank % num(stats.decodedBytes) % num(stats.ioSeconds) % num(stats.inferenceSeconds)
                        % num(stats.fieldStoreSeconds) % num(stats.particleSeconds);

                    if(totalCount != stats.globalParticles)
                        throw std::runtime_error(
                            "neuralBunch: inserted " + std::to_string(totalCount) + " particles but the source "
                            "file contains " + std::to_string(stats.globalParticles) + ".");
                    double const weightTolerance
                        = 1.0e-9 * std::max(1.0, std::abs(stats.globalWeight));
                    if(std::abs(totalWeight - stats.globalWeight) > weightTolerance)
                        throw std::runtime_error(
                            "neuralBunch: inserted weight sum " + num(totalWeight) + " differs from the source "
                            "sum " + num(stats.globalWeight) + ".");
                    if(totalNonFinite != 0)
                        throw std::runtime_error(
                            "neuralBunch: non-finite field values detected after injection.");
                    return stats;
                }
            } // namespace neuralBunchDetail

            void NeuralBunchInit::operator()(uint32_t const step, MappingDesc const& cellDescription) const
            {
                using SpeciesType
                    = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, neuralBunch::Species>;
                neuralBunchDetail::run<SpeciesType>(neuralBunch::neuralBunchParam, step, cellDescription);
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu

#else

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            //! Feature compiled out: the stage is a no-op and adds no dependency.
            void NeuralBunchInit::operator()(uint32_t const, MappingDesc const&) const
            {
            }
        } // namespace stage
    } // namespace simulation
} // namespace picongpu

#endif
