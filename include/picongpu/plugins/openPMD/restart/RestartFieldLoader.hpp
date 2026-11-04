/* Copyright 2014-2026 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera
 *                     Benjamin Worpitz, Franz Poeschel, Alexander Debus
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/defines.hpp"
#    include "picongpu/fields/absorber/pml/Field.hpp"
#    include "picongpu/plugins/misc/ComponentNames.hpp"
#    include "picongpu/plugins/openPMD/openPMDWriter.def"
#    include "picongpu/plugins/openPMD/restart/RestartDatatype.hpp"
#    include "picongpu/plugins/openPMD/restart/RestartFieldMetadata.hpp"
#    include "picongpu/simulation/control/MovingWindow.hpp"
#    include "picongpu/traits/IsFieldDomainBound.hpp"
#    include "picongpu/traits/IsFieldOutputOptional.hpp"

#    include <pmacc/Environment.hpp>
#    include <pmacc/communication/manager_common.hpp>
#    include <pmacc/dataManagement/DataConnector.hpp>
#    include <pmacc/dimensions/DataSpace.hpp>
#    include <pmacc/dimensions/GridLayout.hpp>
#    include <pmacc/particles/frame_types.hpp>
#    include <pmacc/types.hpp>

#    include <iostream>
#    include <optional>
#    include <sstream>
#    include <stdexcept>
#    include <string>
#    include <type_traits>

#    include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace openPMD
    {
        /** Metadata expected by the deck for one checkpoint field. */
        struct FieldMetadata
        {
            /// Absolute unit of each component in PIConGPU internal units.
            std::vector<float_64> unitSI;
            /// In-cell position of each component.
            std::vector<std::vector<double>> position;
            /// Physical dimension of the field.
            std::vector<float_64> unitDimension;
        };

        /**
         * Helper class for openPMD plugin to load fields from parallel openPMD
         * storages.
         */
        class RestartFieldLoader
        {
        private:
            static void warnMissingMetadata(std::string const& record, char const* attribute)
            {
                log<picLog::INPUT_OUTPUT>(
                    "openPMD: WARNING record '%1%' has no %2% metadata; using the backwards-compatible reading")
                    % record % attribute;
            }

            /** Validate the mesh-level metadata that defines the field geometry.
             *
             * Attributes that are not present keep the historical behaviour and
             * only produce a warning, so existing checkpoints remain readable.
             */
            static void validateMeshMetadata(
                ::openPMD::Mesh& mesh,
                std::string const& objectName,
                ThreadParams* params,
                uint32_t const currentStep,
                FieldMetadata const& expected)
            {
                auto const cellSize = sim.pic.getCellSize();

                std::vector<double> deckGridSpacing(simDim);
                for(uint32_t d = 0; d < simDim; ++d)
                    deckGridSpacing[simDim - 1 - d] = static_cast<double>(cellSize[d]);

                DataSpace<simDim> globalSlideOffset;
                uint32_t const numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
                auto const localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                globalSlideOffset.y() += numSlides * localDomain.size.y();
                std::vector<double> deckGlobalOffset(simDim);
                for(uint32_t d = 0; d < simDim; ++d)
                    deckGlobalOffset[simDim - 1 - d]
                        = static_cast<double>(cellSize[d])
                          * static_cast<double>(params->window.globalDimensions.offset[d] + globalSlideOffset[d]);

                if(mesh.containsAttribute("gridSpacing"))
                    restartMetadata::checkGridSpacing(
                        objectName,
                        mesh.gridSpacing<double>(),
                        deckGridSpacing);
                else
                    warnMissingMetadata(objectName, "gridSpacing");

                if(mesh.containsAttribute("gridGlobalOffset"))
                    restartMetadata::checkGridGlobalOffset(
                        objectName,
                        mesh.gridGlobalOffset(),
                        deckGlobalOffset);
                else
                    warnMissingMetadata(objectName, "gridGlobalOffset");

                if(mesh.containsAttribute("axisLabels"))
                {
                    std::vector<std::string> deckAxisLabels;
                    if constexpr(simDim == DIM3)
                        deckAxisLabels = {"z", "y", "x"};
                    else
                        deckAxisLabels = {"y", "x"};
                    restartMetadata::checkAxisLabels(objectName, mesh.axisLabels(), deckAxisLabels);
                }
                else
                    warnMissingMetadata(objectName, "axisLabels");

                if(mesh.containsAttribute("unitDimension"))
                {
                    auto const fileDimension
                        = mesh.getAttribute("unitDimension").get<std::vector<double>>();
                    std::vector<double> deckDimension(expected.unitDimension.begin(), expected.unitDimension.end());
                    restartMetadata::checkUnitDimension(objectName, fileDimension, deckDimension);
                }
                else
                    warnMissingMetadata(objectName, "unitDimension");

                if(mesh.containsAttribute("timeOffset"))
                    restartMetadata::checkTimeOffset(objectName, mesh.timeOffset<double>());
                else
                    warnMissingMetadata(objectName, "timeOffset");
            }

        public:
            template<class Data>
            static void loadField(
                Data& field,
                uint32_t const numComponents,
                std::string objectName,
                ThreadParams* params,
                uint32_t const currentStep,
                bool const isDomainBound,
                FieldMetadata const& expected,
                std::optional<DataSpace<simDim>> const& domainOffset = std::nullopt,
                std::optional<DataSpace<simDim>> const& localDomainSize = std::nullopt,
                std::optional<DataSpace<simDim>> const& destinationBufferOffset = std::nullopt)
            {
                log<picLog::INPUT_OUTPUT>("Begin loading field '%1%'") % objectName;

                auto const name_lookup_tpl = plugins::misc::getComponentNames(numComponents);
                DataSpace<simDim> const field_guard = field.getGridLayout().guardSizeND();

                pmacc::Selection<simDim> const localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();

                using ValueType = typename Data::ValueType;
                field.getHostBuffer().setValue(ValueType::create(0.0));

                ::pmacc::math::Vector<uint64_t, simDim> domain_offset = localDomain.offset;
                DataSpace<simDim> local_domain_size = params->window.localDimensions.size;
                bool const useCustomReadLayout = domainOffset.has_value() && localDomainSize.has_value();
                DataSpace<simDim> destination_buffer_offset = field_guard;
                if(useCustomReadLayout)
                {
                    domain_offset = *domainOffset;
                    local_domain_size = *localDomainSize;
                    destination_buffer_offset = destinationBufferOffset.value_or(field_guard);
                }

                auto const numDataPoints = local_domain_size.productOfComponents();

                if(numDataPoints == 0)
                {
                    // avoid deadlock between not finished pmacc tasks and mpi blocking collectives
                    eventSystem::getTransactionEvent().waitForFinished();
                    // Agree on the number of flush operations
                    for(uint32_t d = 0; d < numComponents; d++)
                    {
                        params->openPMDSeries->flush(PreferredFlushTarget::Disk);
                    }
                    return;
                }

                ::openPMD::Series& series = *params->openPMDSeries;
                ::openPMD::Mesh& mesh = series.iterations[currentStep].open().meshes[objectName];

                validateMeshMetadata(mesh, objectName, params, currentStep, expected);

                auto destBox = field.getHostBuffer().getDataBox();
                for(uint32_t n = 0; n < numComponents; ++n)
                {
                    // Read the subdomain which belongs to our mpi position.
                    // The total grid size must match the grid size of the stored
                    // data.
                    log<picLog::INPUT_OUTPUT>("openPMD: Read from domain: offset=%1% size=%2%") % domain_offset
                        % local_domain_size;
                    ::openPMD::MeshRecordComponent rc = mesh;
                    if(numComponents > 1)
                        rc = mesh[name_lookup_tpl[n]];
                    std::string const recordName
                        = numComponents > 1 ? objectName + "/" + name_lookup_tpl[n] : objectName;

                    log<picLog::INPUT_OUTPUT>("openPMD: Read from field '%1%'") % objectName;

                    double conversionFactor = 1.0;
                    if(rc.containsAttribute("unitSI"))
                    {
                        conversionFactor = restartMetadata::toInternalFactor(
                            recordName,
                            rc.unitSI(),
                            expected.unitSI[n]);
                    }
                    else
                        warnMissingMetadata(recordName, "unitSI");

                    if(rc.containsAttribute("position"))
                    {
                        auto const filePosition = rc.position<double>();
                        restartMetadata::checkPosition(recordName, filePosition, expected.position[n]);
                    }
                    else
                        warnMissingMetadata(recordName, "position");

                    ::openPMD::Offset start
                        = asStandardVector<::pmacc::math::Vector<uint64_t, simDim>&, ::openPMD::Offset>(domain_offset);
                    ::openPMD::Extent count
                        = asStandardVector<DataSpace<simDim>&, ::openPMD::Extent>(local_domain_size);

                    log<picLog::INPUT_OUTPUT>("openPMD: Allocate %1% elements")
                        % local_domain_size.productOfComponents();

                    // avoid deadlock between not finished pmacc tasks and mpi calls
                    // in openPMD backends
                    eventSystem::getTransactionEvent().waitForFinished();

                    /*
                     * @todo float_X should be some kind of gridBuffer's
                     *       GetComponentsType<ValueType>::type
                     */
                    bool converted = false;
                    std::shared_ptr<float_X> field_container
                        = restartDatatype::loadChunkConverted<float_X>(
                            rc,
                            recordName,
                            start,
                            count,
                            [&mesh]() { mesh.seriesFlush(); },
                            converted);
                    if(converted && Environment<simDim>::get().GridController().getGlobalRank() == 0)
                        std::cerr << "openPMD: WARNING record '" << recordName << "' stores "
                                  << ::openPMD::datatypeToString(rc.getDatatype()) << " and is converted to "
                                  << ::openPMD::datatypeToString(::openPMD::determineDatatype<float_X>())
                                  << " during chunk loading" << std::endl;

                    /* start a blocking read of all scheduled variables */
                    mesh.seriesFlush();


#    pragma omp parallel for simd
                    for(int linearId = 0; linearId < numDataPoints; ++linearId)
                    {
                        DataSpace<simDim> destIdx;
                        if(useCustomReadLayout)
                        {
                            destIdx = pmacc::math::mapToND(local_domain_size, linearId);
                            destIdx += destination_buffer_offset;
                        }
                        else
                        {
                            /* calculate index inside the moving window domain which
                             * is located on the local grid*/
                            destIdx = pmacc::math::mapToND(params->window.localDimensions.size, linearId);
                            /* jump over guard and local sliding window offset*/
                            destIdx += field_guard + params->localWindowToDomainOffset;
                        }

                        destBox(destIdx)[n] = static_cast<float_X>(
                            static_cast<double>(field_container.get()[linearId]) * conversionFactor);
                    }
                }

                field.hostToDevice();

                eventSystem::getTransactionEvent().waitForFinished();

                log<picLog::INPUT_OUTPUT>("openPMD: Read from domain: offset=%1% size=%2%") % domain_offset
                    % local_domain_size;
                log<picLog::INPUT_OUTPUT>("openPMD: Finished loading field '%1%'") % objectName;
            }
        };

        /**
         * Helper class for openPMDWriter (forEach operator) to load a field from
         * openPMD
         *
         * @tparam T_Field field class to load
         */
        template<typename T_Field>
        struct LoadFields
        {
        private:
            static constexpr bool isPmlSlabField = std::is_same_v<T_Field, fields::absorber::pml::FieldE>
                                                   || std::is_same_v<T_Field, fields::absorber::pml::FieldB>;

            static std::string getPmlSlabName(std::string const& fieldName, uint32_t const slabIdx)
            {
                auto const slabSuffix = [&]()
                {
                    switch(slabIdx)
                    {
                    case 0u:
                        return "_xneg";
                    case 1u:
                        return "_xpos";
                    case 2u:
                        return "_yneg";
                    case 3u:
                        return "_ypos";
                    case 4u:
                        if constexpr(simDim == DIM3)
                            return "_zneg";
                        break;
                    case 5u:
                        if constexpr(simDim == DIM3)
                            return "_zpos";
                        break;
                    default:
                        break;
                    }
                    throw std::runtime_error("Invalid PML slab index for naming: " + std::to_string(slabIdx));
                }();
                return fieldName + slabSuffix;
            }

            static FieldMetadata getMetadata()
            {
                FieldMetadata metadata;
                auto const unit = T_Field::getUnit();
                for(uint32_t n = 0; n < static_cast<uint32_t>(T_Field::numComponents); ++n)
                    metadata.unitSI.push_back(std::abs(static_cast<float_64>(unit[n])));

                traits::FieldPosition<fields::YeeCell, T_Field> const fieldPosition;
                auto const positions = fieldPosition();
                for(uint32_t n = 0; n < static_cast<uint32_t>(T_Field::numComponents); ++n)
                {
                    std::vector<double> position(simDim);
                    for(uint32_t d = 0; d < simDim; ++d)
                        position[d] = static_cast<double>(positions[n][d]);
                    metadata.position.push_back(std::move(position));
                }

                metadata.unitDimension = T_Field::getUnitDimension();
                return metadata;
            }

        public:
            HINLINE void operator()(ThreadParams* params, uint32_t const restartStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                ThreadParams* tp = params;

                // Skip optional fields
                if(traits::IsFieldOutputOptional<T_Field>::value && !dc.hasId(T_Field::getName()))
                    return;

                /* load field without copying data to host */
                auto field = dc.get<T_Field>(T_Field::getName());

                /* load from openPMD */
                bool const isDomainBound = traits::IsFieldDomainBound<T_Field>::value;

                FieldMetadata const metadata = getMetadata();

                if constexpr(isPmlSlabField)
                {
                    auto const localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
                    auto& meshes = params->openPMDSeries->iterations[restartStep].open().meshes;
                    for(uint32_t slabIdx = 0u; slabIdx < T_Field::getNumSlabs(); ++slabIdx)
                    {
                        auto const slabName = getPmlSlabName(T_Field::getName(), slabIdx);
                        if(!meshes.contains(slabName))
                        {
                            log<picLog::INPUT_OUTPUT>(
                                "openPMD: WARNING checkpoint has no PML record '%1%'; leaving it at zero")
                                % slabName;
                            continue;
                        }
                        auto const slabBegin = field->getSlabBegin(slabIdx);
                        auto const slabViewBegin = field->getSlabViewBegin(slabIdx);
                        auto const slabViewSize = field->getSlabViewSize(slabIdx);
                        auto const destinationBufferOffset
                            = field->getGridBuffer(slabIdx).getGridLayout().guardSizeND()
                              + (slabViewBegin - slabBegin);
                        auto& slabBuffer = field->getGridBuffer(slabIdx);
                        RestartFieldLoader::loadField(
                            slabBuffer,
                            static_cast<uint32_t>(T_Field::numComponents),
                            slabName,
                            tp,
                            restartStep,
                            isDomainBound,
                            metadata,
                            localDomain.offset + slabViewBegin,
                            slabViewSize,
                            destinationBufferOffset);
                    }
                }
                else
                {
                    RestartFieldLoader::loadField(
                        field->getGridBuffer(),
                        static_cast<uint32_t>(T_Field::numComponents),
                        T_Field::getName(),
                        tp,
                        restartStep,
                        isDomainBound,
                        metadata);
                }
            }
        };

        using namespace pmacc;

    } /* namespace openPMD */
} /* namespace picongpu */

#endif
