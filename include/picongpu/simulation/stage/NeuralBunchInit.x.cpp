/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include "picongpu/simulation/stage/NeuralBunchInit.hpp"

#include "picongpu/defines.hpp"

#if defined(PIC_ENABLE_NEURAL_BUNCH) && PIC_ENABLE_NEURAL_BUNCH

#    include "picongpu/fields/FieldB.hpp"
#    include "picongpu/fields/FieldE.hpp"
#    include "picongpu/param/neuralBunch.param"
#    include "picongpu/particles/neuralBunch/Inference.hpp"
#    include "picongpu/particles/neuralBunch/Model.hpp"
#    include "picongpu/particles/neuralBunch/ReferenceKernel.hpp"
#    include "picongpu/particles/neuralBunch/ReferenceTile.hpp"
#    include "picongpu/particles/neuralBunch/SourceFile.hpp"
#    include "picongpu/particles/neuralBunch/TileGeometry.hpp"
#    include "picongpu/particles/param.hpp"
#    include "picongpu/plugins/output/WriteSpeciesCommon.hpp"

#    include <pmacc/Environment.hpp>
#    include <pmacc/alpakaHelper/Device.hpp>
#    include <pmacc/alpakaHelper/acc.hpp>
#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/mappings/simulation/SubGrid.hpp>
#    include <pmacc/meta/ForEach.hpp>
#    include <pmacc/meta/conversion/MakeSeq.hpp>
#    include <pmacc/meta/conversion/RemoveFromSeq.hpp>
#    include <pmacc/particles/IdProvider.hpp>
#    include <pmacc/particles/ParticleDescription.hpp>
#    include <pmacc/particles/meta/FindByNameOrType.hpp>
#    include <pmacc/particles/operations/splitIntoListOfFrames.kernel>
#    include <pmacc/traits/HasIdentifier.hpp>
#    include <pmacc/traits/Resolve.hpp>

#    include <boost/mpl/placeholders.hpp>

#    include <algorithm>
#    include <array>
#    include <chrono>
#    include <cmath>
#    include <cstdint>
#    include <cstdlib>
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
                namespace nbRef = picongpu::particles::neuralBunch::reference;

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
                    std::uint64_t referenceGammaPairs = 0;
                    std::uint64_t referenceAspectPairs = 0;
                    std::uint64_t referenceNearPairs = 0;
                    std::uint64_t referenceFarPairs = 0;
                    std::uint64_t zeroWeightPairs = 0;
                    std::uint64_t forcedReferencePairs = 0;
                    std::uint64_t unvalidatedPairs = 0;
                    std::uint64_t synthesizedIds = 0;
                    std::vector<std::uint64_t> localIds;
                    std::uint64_t localParticles = 0;
                    double globalWeight = 0.0;
                    double localWeight = 0.0;
                    double globalCharge = 0.0;
                    double decodedBytes = 0.0;
                    double ioSeconds = 0.0;
                    double inferenceSeconds = 0.0;
                    double referenceSeconds = 0.0;
                    double referenceMaxAbs = 0.0;
                    double referenceL1 = 0.0;
                    double diagHybridVsReference = -1.0;
                    double diagHybridVsNeural = -1.0;
                    double diagReferenceScale = 0.0;
                    bool diagAvailable = false;
                    char const* idMode = "none";
                    bool idUnique = true;
                    double fieldStoreSeconds = 0.0;
                    double particleSeconds = 0.0;
                    double maxLatent = 0.0;
                    double fieldSums[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                    std::uint64_t nonFiniteFields = 0;
                };

                constexpr double decodedBytesPerParticle = 3.0 * 8.0 + 3.0 * 8.0 + 8.0 + 8.0 + 8.0 + 8.0 + 1.0;

                template<typename T_Species>
                struct ParticleInserter
                {
                    using SpeciesType = T_Species;
                    using FrameType = typename SpeciesType::FrameType;
                    using ParticleDescription = typename FrameType::ParticleDescription;
                    using ParticleAttributeList = typename FrameType::ValueTypeSeq;

                    using TypesToDelete = pmacc::mp_list<multiMask, localCellIdx>;
                    using ParticleCleanedAttributeList =
                        typename RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type;
                    using ParticleNewAttributeList = MakeSeq_t<ParticleCleanedAttributeList, totalCellIdx>;
                    using NewParticleDescription =
                        typename ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type;

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
                                double const cellIndex = std::floor(chunk.position[3 * i + axis] / cellSizeSI[axis]);
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
                        pmacc::meta::
                            ForEach<typename NewParticleDescription::ValueTypeSeq, MallocMappedMemory<boost::mpl::_1>>
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
                                if(cell[axis] < localOffset[axis] || cell[axis] >= localOffset[axis] + localSize[axis])
                                    inside = false;
                            }
                            if(!inside)
                                continue;

                            PositionValue inCell;
                            for(int axis = 0; axis < 3; ++axis)
                            {
                                double const positionSI = chunk.position[3 * i + axis];
                                double fraction = positionSI / cellSizeSI[axis] - static_cast<double>(cell[axis]);
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
                            {
                                if(chunk.id.empty())
                                {
                                    auto idProvider = dc.get<IdProvider>("globalId");
                                    id[target] = static_cast<IdValue>(idProvider->getNewIdHost());
                                    ++stats.synthesizedIds;
                                }
                                else
                                    id[target] = static_cast<IdValue>(chunk.id[i]);
                                stats.localIds.push_back(static_cast<std::uint64_t>(id[target]));
                            }
                            ++target;
                        }

                        if(target != accepted)
                            throw std::runtime_error("neuralBunch: accepted particle count changed between passes.");

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
                                double const cellIndex = std::floor(chunk.position[3 * i + axis] / cellSizeSI[axis]);
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
                    stats.particleSeconds
                        += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                }

                template<typename T_Species>
                Statistics run(NeuralBunchParam const& params, uint32_t const step, MappingDesc const& cellDescription)
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

                    tg::CellSizes const cells{cellSizeSI[0], cellSizeSI[1], cellSizeSI[2]};
                    double const aspectY = cells.aspectY();
                    double const aspectZ = cells.aspectZ();
                    tg::DomainBounds const domain{
                        meta.gammaMin,
                        meta.gammaMax,
                        meta.aYMin,
                        meta.aYMax,
                        meta.aZMin,
                        meta.aZMax,
                        meta.dMinCells,
                        meta.dMaxCells};

                    if(!params.enableReferenceFallback)
                        throw std::runtime_error(
                            "neuralBunch: enableReferenceFallback=false is not supported; the Phase B hybrid "
                            "policy always requires the reference fallback.");
                    tg::ValidatedBand validatedBand;
                    tg::ValidatedBand const* band = nullptr;
                    if(meta.validatedBand)
                    {
                        auto const& source = *meta.validatedBand;
                        validatedBand.gammaMin = source.gammaMin;
                        validatedBand.gammaMax = source.gammaMax;
                        validatedBand.aYMin = source.aYMin;
                        validatedBand.aYMax = source.aYMax;
                        validatedBand.aZMin = source.aZMin;
                        validatedBand.aZMax = source.aZMax;
                        validatedBand.dMinCells = source.dMinCells;
                        validatedBand.dMaxCells = source.dMaxCells;
                        validatedBand.measuredMaxRelRmsE = source.measuredMaxRelRmsE;
                        validatedBand.measuredMaxRelRmsCb = source.measuredMaxRelRmsCb;
                        validatedBand.measuredMinForcePassShare = source.measuredMinForcePassShare;
                        band = &validatedBand;
                    }
                    bool const neuralEnabled = params.allowUnvalidatedNeural || (band != nullptr);
                    if(neuralEnabled && !tg::aspectCovered(cells, domain))
                        throw std::runtime_error(
                            "neuralBunch: simulation cell aspect a_y=" + num(aspectY) + ", a_z=" + num(aspectZ)
                            + " is outside the model domain a_y=[" + num(domain.aYMin) + "," + num(domain.aYMax)
                            + "], a_z=[" + num(domain.aZMin) + "," + num(domain.aZMax)
                            + "]. Refusing to clamp the model domain.");

                    auto& device = pmacc::manager::Device<pmacc::ComputeDevice>::get().current();
                    auto queue = device.makeQueue(pmacc::ComputeQueuePolicies);
                    auto const exec = pmacc::ComputeExecutor;
                    auto const tensors = model.loadTensors(queue);

                    auto const grid = tg::enumerateTiles(offset, size);
                    std::vector<std::vector<double>> latents(grid.tiles.size(), std::vector<double>(meta.latent, 0.0));
                    std::vector<std::vector<double>> referenceTiles(
                        grid.tiles.size(),
                        std::vector<double>(tg::outputDimension, 0.0));

                    DataConnector& dc = Environment<>::get().DataConnector();
                    auto fieldE = dc.get<FieldE>(FieldE::getName());
                    auto fieldB = dc.get<FieldB>(FieldB::getName());
                    auto hostE = fieldE->getHostDataBox().shift(fieldE->getGridBuffer().getGridLayout().guardSizeND());
                    auto hostB = fieldB->getHostDataBox().shift(fieldB->getGridBuffer().getGridLayout().guardSizeND());

                    MPI_Comm const comm = Environment<simDim>::get().GridController().getCommunicator().getMPIComm();
                    int const rank = Environment<simDim>::get().GridController().getGlobalRank();

                    if(params.allowUnvalidatedNeural)
                        log<picLog::PHYSICS>(
                            "neuralBunch: WARNING rank=%1% TEST-ONLY allowUnvalidatedNeural=true ignores the "
                            "validated band; neural results are not production-validated.")
                            % rank;
                    else if(band == nullptr)
                        log<picLog::PHYSICS>(
                            "neuralBunch: rank=%1% no validation.validated_band in the artifact; neural path "
                            "DISABLED, all pairs use the FP64 reference. This is the production default.")
                            % rank;
                    else
                        log<picLog::PHYSICS>(
                            "neuralBunch: rank=%1% validated band d=[%2%,%3%] gamma=[%4%,%5%] "
                            "a_y=[%6%,%7%] a_z=[%8%,%9%] measured E=%10% cB=%11% forcePass=%12% source='%13%'")
                            % rank % num(band->dMinCells) % num(band->dMaxCells) % num(band->gammaMin)
                            % num(band->gammaMax) % num(band->aYMin) % num(band->aYMax) % num(band->aZMin)
                            % num(band->aZMax) % num(band->measuredMaxRelRmsE) % num(band->measuredMaxRelRmsCb)
                            % num(band->measuredMinForcePassShare) % meta.validatedBand->source;


                    // Validation-only switch: force the reference path for every
                    // non-zero-weight pair on every rank (same environment on all ranks).
                    bool const forceReference = std::getenv("NEURAL_BUNCH_FORCE_REFERENCE") != nullptr;

                    bool diagnosticChosen = false;
                    std::size_t diagnosticTile = 0;
                    std::array<double, 3> diagnosticOrigin{};
                    std::vector<Source> diagnosticSources;
                    std::vector<nbRef::Source> diagnosticReferenceSources;
                    std::vector<std::vector<nbRef::Source>> referenceWork(grid.tiles.size());

                    {
                        SourceReader reader(params.filename, params.iteration, params.species, comm);
                        SourceMetadata const& sourceMeta = reader.metadata();
                        if(params.timePolicy != 0)
                            throw std::runtime_error(
                                "neuralBunch: Phase B requires timePolicy == 0 (source time equals injection "
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
                                for(std::size_t tileIndex = 0; tileIndex < grid.tiles.size(); ++tileIndex)
                                {
                                    std::array<double, 3> tileOriginPhysical{};
                                    for(int axis = 0; axis < 3; ++axis)
                                        tileOriginPhysical[axis]
                                            = static_cast<double>(grid.tiles[tileIndex].origin[axis])
                                              * cellSizeSI[axis];

                                    std::vector<Source> allSources;
                                    std::vector<nbRef::Source> allReferenceSources;
                                    std::vector<Source> neuralSources;
                                    std::vector<nbRef::Source> referenceSources;
                                    allSources.reserve(chunk.size());
                                    allReferenceSources.reserve(chunk.size());
                                    for(std::size_t i = 0; i < chunk.size(); ++i)
                                    {
                                        if(!chunk.valid[i])
                                            continue;
                                        Source source;
                                        std::array<double, 3> position{};
                                        for(int axis = 0; axis < 3; ++axis)
                                        {
                                            position[axis] = chunk.position[3 * i + axis];
                                            source.r[axis] = (position[axis] - tileOriginPhysical[axis]) / cells.dx;
                                            source.u[axis] = chunk.u[3 * i + axis];
                                        }
                                        source.weight = chunk.weight[i];
                                        source.aY = aspectY;
                                        source.aZ = aspectZ;
                                        allSources.push_back(source);

                                        nbRef::Source referenceSource;
                                        referenceSource.x = {position[0], position[1], position[2]};
                                        referenceSource.u = {source.u[0], source.u[1], source.u[2]};
                                        referenceSource.w = source.weight;
                                        allReferenceSources.push_back(referenceSource);

                                        double const gamma = std::sqrt(
                                            1.0 + source.u[0] * source.u[0] + source.u[1] * source.u[1]
                                            + source.u[2] * source.u[2]);
                                        double const distanceCells
                                            = tg::minimumStaggeredDistanceCells(position, tileOriginPhysical, cells);
                                        if(forceReference && source.weight != 0.0)
                                        {
                                            referenceSources.push_back(referenceSource);
                                            ++stats.forcedReferencePairs;
                                            continue;
                                        }
                                        switch(tg::classify(
                                            source.weight,
                                            gamma,
                                            aspectY,
                                            aspectZ,
                                            distanceCells,
                                            domain,
                                            band,
                                            params.allowUnvalidatedNeural))
                                        {
                                        case tg::PairRoute::neural:
                                            neuralSources.push_back(source);
                                            break;
                                        case tg::PairRoute::referenceGamma:
                                            referenceSources.push_back(referenceSource);
                                            ++stats.referenceGammaPairs;
                                            break;
                                        case tg::PairRoute::referenceAspect:
                                            referenceSources.push_back(referenceSource);
                                            ++stats.referenceAspectPairs;
                                            break;
                                        case tg::PairRoute::referenceNear:
                                            referenceSources.push_back(referenceSource);
                                            ++stats.referenceNearPairs;
                                            break;
                                        case tg::PairRoute::referenceFar:
                                            referenceSources.push_back(referenceSource);
                                            ++stats.referenceFarPairs;
                                            break;
                                        case tg::PairRoute::referenceUnvalidated:
                                            referenceSources.push_back(referenceSource);
                                            ++stats.unvalidatedPairs;
                                            break;
                                        case tg::PairRoute::zeroWeight:
                                            ++stats.zeroWeightPairs;
                                            break;
                                        }
                                    }

                                    if(!neuralSources.empty())
                                    {
                                        auto const encodeStart = std::chrono::steady_clock::now();
                                        auto hidden = runEncoder(queue, exec, meta, tensors, neuralSources);
                                        stats.inferenceSeconds += std::chrono::duration<double>(
                                                                      std::chrono::steady_clock::now() - encodeStart)
                                                                      .count();
                                        auto& latent = latents[tileIndex];
                                        for(std::size_t i = 0; i < neuralSources.size(); ++i)
                                            for(std::size_t k = 0; k < meta.latent; ++k)
                                            {
                                                double const contribution
                                                    = -neuralSources[i].weight
                                                      * static_cast<double>(hidden[i * meta.latent + k]);
                                                latent[k] += contribution;
                                                stats.maxLatent = std::max(stats.maxLatent, std::abs(latent[k]));
                                            }
                                        stats.neuralPairs += neuralSources.size();
                                    }

                                    if(!referenceSources.empty())
                                    {
                                        // Reference work is evaluated after the tile loop, in parallel.
                                        referenceWork[tileIndex] = std::move(referenceSources);
                                        stats.referencePairs += referenceWork[tileIndex].size();
                                    }

                                    if(!diagnosticChosen && !neuralSources.empty()
                                       && !referenceWork[tileIndex].empty())
                                    {
                                        // Compare the three policies on a bounded subset of this tile:
                                        // up to four neural and four reference sources. This keeps the
                                        // reference-only diagnostic affordable while using the same
                                        // production code paths.
                                        diagnosticChosen = true;
                                        diagnosticTile = tileIndex;
                                        diagnosticOrigin = tileOriginPhysical;
                                        std::size_t const neuralCount
                                            = std::min<std::size_t>(4u, neuralSources.size());
                                        std::size_t const referenceCount
                                            = std::min<std::size_t>(4u, referenceWork[tileIndex].size());
                                        for(std::size_t k = 0; k < neuralCount; ++k)
                                        {
                                            nbRef::Source reference;
                                            for(int axis = 0; axis < 3; ++axis)
                                            {
                                                (&reference.x.x)[axis]
                                                    = neuralSources[k].r[axis] * cells.dx + tileOriginPhysical[axis];
                                                (&reference.u.x)[axis] = neuralSources[k].u[axis];
                                            }
                                            reference.w = neuralSources[k].weight;
                                            diagnosticReferenceSources.push_back(reference);
                                        }
                                        for(std::size_t k = 0; k < referenceCount; ++k)
                                            diagnosticReferenceSources.push_back(referenceWork[tileIndex][k]);

                                        for(auto const& reference : diagnosticReferenceSources)
                                        {
                                            Source source;
                                            for(int axis = 0; axis < 3; ++axis)
                                            {
                                                source.r[axis]
                                                    = ((&reference.x.x)[axis] - tileOriginPhysical[axis]) / cells.dx;
                                                source.u[axis] = (&reference.u.x)[axis];
                                            }
                                            source.weight = reference.w;
                                            source.aY = aspectY;
                                            source.aZ = aspectZ;
                                            diagnosticSources.push_back(source);
                                        }
                                    }
                                }

                                // Host FP64 reference fallback. Tiles are visited in order;
                                // each tile parallelises its 1536 independent output points.
                                auto const referenceStart = std::chrono::steady_clock::now();
                                std::uint64_t referenceMaxErrors = 0;
                                for(std::size_t tileIndex = 0; tileIndex < grid.tiles.size(); ++tileIndex)
                                {
                                    if(referenceWork[tileIndex].empty())
                                        continue;
                                    std::array<double, 3> tileOriginPhysical{};
                                    for(int axis = 0; axis < 3; ++axis)
                                        tileOriginPhysical[axis]
                                            = static_cast<double>(grid.tiles[tileIndex].origin[axis])
                                              * cellSizeSI[axis];
                                    nbRef::TileReport report;
                                    auto const referenceValues = nbRef::tileFieldParallel(
                                        {tileOriginPhysical[0], tileOriginPhysical[1], tileOriginPhysical[2]},
                                        referenceWork[tileIndex],
                                        cells.dx,
                                        cells.dy,
                                        cells.dz,
                                        &report);
                                    auto& referenceTile = referenceTiles[tileIndex];
                                    double localMax = 0.0;
                                    double localL1 = 0.0;
                                    for(std::size_t k = 0; k < referenceValues.size(); ++k)
                                    {
                                        referenceTile[k] += referenceValues[k];
                                        localMax = std::max(localMax, std::abs(referenceValues[k]));
                                        localL1 += std::abs(referenceValues[k]);
                                    }
#    pragma omp critical(neuralBunchReferenceStats)
                                    {
                                        stats.referenceMaxAbs = std::max(stats.referenceMaxAbs, localMax);
                                        stats.referenceL1 += localL1;
                                        if(!report.all_converged)
                                        {
                                            ++referenceMaxErrors;
                                            log<picLog::PHYSICS>(
                                                "neuralBunch: rank=%1% tile=%2% reference quadrature not "
                                                "converged (max error %3%)")
                                                % rank % tileIndex % num(report.max_error);
                                        }
                                    }
                                    referenceWork[tileIndex].clear();
                                }
                                stats.referenceSeconds
                                    += std::chrono::duration<double>(std::chrono::steady_clock::now() - referenceStart)
                                           .count();
                                if(referenceMaxErrors != 0)
                                    log<picLog::PHYSICS>(
                                        "neuralBunch: rank=%1% reference quadrature not converged for %2% "
                                        "tile(s) in this chunk")
                                        % rank % referenceMaxErrors;
                            },
                            params.chunkSize);
                        stats.ioSeconds
                            += std::chrono::duration<double>(std::chrono::steady_clock::now() - ioStart).count();
                        stats.globalParticles = sourceMeta.globalCount;
                    }

                    if(stats.neuralPairs + stats.referencePairs + stats.zeroWeightPairs
                       != stats.globalParticles * grid.tiles.size())
                        throw std::runtime_error(
                            "neuralBunch: pair partition is not exhaustive: neural "
                            + std::to_string(stats.neuralPairs) + " + reference "
                            + std::to_string(stats.referencePairs) + " + zero-weight "
                            + std::to_string(stats.zeroWeightPairs)
                            + " != " + std::to_string(stats.globalParticles * grid.tiles.size()) + " pairs.");

                    // Hybrid-vs-forced comparison for one small tile where both routes occur.
                    if(diagnosticChosen)
                    {
                        auto const referenceAll = nbRef::tileFieldParallel(
                            {diagnosticOrigin[0], diagnosticOrigin[1], diagnosticOrigin[2]},
                            diagnosticReferenceSources,
                            cells.dx,
                            cells.dy,
                            cells.dz);
                        std::vector<double> referenceAllSI(tg::outputDimension, 0.0);
                        for(std::size_t k = 0; k < referenceAllSI.size(); ++k)
                            referenceAllSI[k] = (k % 6) < 3 ? tg::normalizedEToSI(referenceAll[k], cells.dx)
                                                            : tg::normalizedBToSI(referenceAll[k], cells.dx);

                        auto hidden = runEncoder(queue, exec, meta, tensors, diagnosticSources);
                        std::vector<double> neuralAllLatent(meta.latent, 0.0);
                        for(std::size_t i = 0; i < diagnosticSources.size(); ++i)
                            for(std::size_t k = 0; k < meta.latent; ++k)
                                neuralAllLatent[k]
                                    -= diagnosticSources[i].weight * static_cast<double>(hidden[i * meta.latent + k]);
                        std::vector<float> const neuralAllFloat(neuralAllLatent.begin(), neuralAllLatent.end());
                        auto const neuralAll = decodeFromLatent(
                            queue,
                            exec,
                            meta,
                            tensors,
                            neuralAllFloat,
                            aspectY,
                            aspectZ,
                            FieldNormalization{cells.dx});

                        std::vector<float> const neuralSubFloat(
                            latents[diagnosticTile].begin(),
                            latents[diagnosticTile].end());
                        auto const neuralSub = decodeFromLatent(
                            queue,
                            exec,
                            meta,
                            tensors,
                            neuralSubFloat,
                            aspectY,
                            aspectZ,
                            FieldNormalization{cells.dx});

                        double hybridVsReference = 0.0;
                        double hybridVsNeural = 0.0;
                        double scale = 0.0;
                        for(std::size_t k = 0; k < tg::outputDimension; ++k)
                        {
                            double const referenceSubSI
                                = (k % 6) < 3 ? tg::normalizedEToSI(referenceTiles[diagnosticTile][k], cells.dx)
                                              : tg::normalizedBToSI(referenceTiles[diagnosticTile][k], cells.dx);
                            double const hybrid = static_cast<double>(neuralSub[k]) + referenceSubSI;
                            hybridVsReference = std::max(hybridVsReference, std::abs(hybrid - referenceAllSI[k]));
                            hybridVsNeural
                                = std::max(hybridVsNeural, std::abs(hybrid - static_cast<double>(neuralAll[k])));
                            scale = std::max(
                                scale,
                                std::max(std::abs(referenceAllSI[k]), std::abs(static_cast<double>(neuralAll[k]))));
                        }
                        stats.diagHybridVsReference = hybridVsReference / std::max(scale, 1e-300);
                        stats.diagHybridVsNeural = hybridVsNeural / std::max(scale, 1e-300);
                        stats.diagReferenceScale = scale;
                        stats.diagAvailable = true;
                    }

                    auto const storeStart = std::chrono::steady_clock::now();
                    for(std::size_t tileIndex = 0; tileIndex < grid.tiles.size(); ++tileIndex)
                    {
                        auto const& tile = grid.tiles[tileIndex];
                        std::vector<float> const latentFloat(latents[tileIndex].begin(), latents[tileIndex].end());
                        auto const values = decodeFromLatent(
                            queue,
                            exec,
                            meta,
                            tensors,
                            latentFloat,
                            aspectY,
                            aspectZ,
                            FieldNormalization{cells.dx});
                        auto const& referenceTile = referenceTiles[tileIndex];
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
                                        std::size_t const index = tg::outputIndex(i, j, k, component);
                                        std::size_t const indexB = tg::outputIndex(i, j, k, component + 3);
                                        double const eSI = static_cast<double>(values[index])
                                                           + tg::normalizedEToSI(referenceTile[index], cells.dx);
                                        double const bSI = static_cast<double>(values[indexB])
                                                           + tg::normalizedBToSI(referenceTile[indexB], cells.dx);
                                        e[component] += static_cast<float_X>(eSI / unitE);
                                        b[component] += static_cast<float_X>(bSI / unitB);
                                    }
                                }
                    }
                    fieldE->syncToDevice();
                    fieldB->syncToDevice();
                    stats.fieldStoreSeconds
                        += std::chrono::duration<double>(std::chrono::steady_clock::now() - storeStart).count();

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
                        stats.ioSeconds
                            += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
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

                    // Particle-ID materialisation and global uniqueness check.
                    if constexpr(ParticleInserter<typename neuralBunch::Species>::hasId)
                    {
                        stats.idMode = stats.synthesizedIds == 0u ? "file" : "synthesized";
                        unsigned long long localIds = stats.localIds.size();
                        unsigned long long totalIds = 0;
                        MPI_Allreduce(&localIds, &totalIds, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
                        if(totalIds != totalCount)
                            throw std::runtime_error(
                                "neuralBunch: materialized " + std::to_string(totalIds) + " particle IDs but inserted "
                                + std::to_string(totalCount) + " particles.");
                        int localCount = static_cast<int>(stats.localIds.size());
                        std::vector<int> counts;
                        if(rank == 0)
                            counts.resize(Environment<simDim>::get().GridController().getGlobalSize(), 0);
                        MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
                        std::vector<int> displacements(counts.size(), 0);
                        if(rank == 0)
                            for(std::size_t i = 1; i < counts.size(); ++i)
                                displacements[i] = displacements[i - 1] + counts[i - 1];
                        std::vector<std::uint64_t> allIds(rank == 0 ? static_cast<std::size_t>(totalIds) : 0u);
                        MPI_Gatherv(
                            stats.localIds.data(),
                            localCount,
                            MPI_UINT64_T,
                            allIds.data(),
                            counts.data(),
                            displacements.data(),
                            MPI_UINT64_T,
                            0,
                            comm);
                        int duplicates = 0;
                        if(rank == 0)
                        {
                            std::sort(allIds.begin(), allIds.end());
                            duplicates
                                = static_cast<int>(std::adjacent_find(allIds.begin(), allIds.end()) != allIds.end());
                        }
                        MPI_Bcast(&duplicates, 1, MPI_INT, 0, comm);
                        stats.idUnique = duplicates == 0;
                    }

                    double const chargePerWeight = -1.602176634e-19;
                    double const globalCharge = chargePerWeight * stats.globalWeight;
                    log<picLog::PHYSICS>("neuralBunch: rank=%1% step=%2% tiles=%3% neuralPairs=%4% referencePairs=%5% "
                                         "globalParticles=%6% localInserted=%7% totalInserted=%8% globalWeight=%9%")
                        % rank % step % grid.tiles.size() % stats.neuralPairs % stats.referencePairs
                        % stats.globalParticles % stats.localParticles % totalCount % num(stats.globalWeight);
                    log<picLog::PHYSICS>("neuralBunch: rank=%1% totalInsertedWeight=%2% globalCharge=%3% "
                                         "fieldSums=[%4% %5% %6% %7% %8% %9%]")
                        % rank % num(totalWeight) % num(globalCharge) % num(fieldSums[0]) % num(fieldSums[1])
                        % num(fieldSums[2]) % num(fieldSums[3]) % num(fieldSums[4]) % num(fieldSums[5]);
                    log<picLog::PHYSICS>("neuralBunch: rank=%1% nonFiniteFields=%2% maxLatent=%3%") % rank
                        % totalNonFinite % num(stats.maxLatent);
                    log<picLog::PHYSICS>("neuralBunch: rank=%1% decodedBytes=%2% ioSeconds=%3% inferenceSeconds=%4% "
                                         "referenceSeconds=%5% fieldStoreSeconds=%6% particleSeconds=%7%")
                        % rank % num(stats.decodedBytes) % num(stats.ioSeconds) % num(stats.inferenceSeconds)
                        % num(stats.referenceSeconds) % num(stats.fieldStoreSeconds) % num(stats.particleSeconds);
                    unsigned long long neuralPairs = stats.neuralPairs;
                    unsigned long long referencePairs = stats.referencePairs;
                    unsigned long long nearPairs = stats.referenceNearPairs;
                    unsigned long long farPairs = stats.referenceFarPairs;
                    unsigned long long gammaPairs = stats.referenceGammaPairs;
                    unsigned long long aspectPairs = stats.referenceAspectPairs;
                    unsigned long long zeroPairs = stats.zeroWeightPairs;
                    unsigned long long totalPairs = neuralPairs + referencePairs + zeroPairs;
                    MPI_Allreduce(MPI_IN_PLACE, &totalPairs, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
                    log<picLog::PHYSICS>("neuralBunch: rank=%1% hybrid reasons near=%2% far=%3% gamma=%4% aspect=%5% "
                                         "zeroWeight=%6% unvalidated=%7% forced=%8% referenceCoverage=%9%")
                        % rank % nearPairs % farPairs % gammaPairs % aspectPairs % zeroPairs % stats.unvalidatedPairs
                        % stats.forcedReferencePairs
                        % num(totalPairs == 0u ? 0.0 : double(referencePairs) / double(totalPairs));
                    log<picLog::PHYSICS>("neuralBunch: rank=%1% referenceMaxAbs=%2% referenceL1=%3% "
                                         "hybridVsReference=%4% hybridVsNeural=%5% diagScale=%6% diagAvailable=%7%")
                        % rank % num(stats.referenceMaxAbs) % num(stats.referenceL1) % num(stats.diagHybridVsReference)
                        % num(stats.diagHybridVsNeural) % num(stats.diagReferenceScale)
                        % (stats.diagAvailable ? 1 : 0);
                    log<picLog::PHYSICS>("neuralBunch: rank=%1% ids=%2% unique=%3% synthesized=%4%") % rank
                        % stats.idMode % (stats.idUnique ? 1 : 0) % stats.synthesizedIds;

                    if(totalCount != stats.globalParticles)
                        throw std::runtime_error(
                            "neuralBunch: inserted " + std::to_string(totalCount)
                            + " particles but the source "
                              "file contains "
                            + std::to_string(stats.globalParticles) + ".");
                    double const weightTolerance = 1.0e-9 * std::max(1.0, std::abs(stats.globalWeight));
                    if(std::abs(totalWeight - stats.globalWeight) > weightTolerance)
                        throw std::runtime_error(
                            "neuralBunch: inserted weight sum " + num(totalWeight)
                            + " differs from the source "
                              "sum "
                            + num(stats.globalWeight) + ".");
                    if(totalNonFinite != 0)
                        throw std::runtime_error("neuralBunch: non-finite field values detected after injection.");
                    if(!stats.idUnique)
                        throw std::runtime_error("neuralBunch: duplicate particle IDs after insertion.");
                    return stats;
                }
            } // namespace neuralBunchDetail

            void NeuralBunchInit::operator()(uint32_t const step, MappingDesc const& cellDescription) const
            {
                using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, neuralBunch::Species>;
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
