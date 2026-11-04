/* Copyright 2026 Rene Widera
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

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace picongpu
{
    namespace openPMD
    {
        namespace restartMetadata
        {
            /** Format a vector for error messages. */
            inline std::string toString(std::vector<double> const& values)
            {
                std::ostringstream stream;
                stream << "[";
                for(size_t i = 0; i < values.size(); ++i)
                    stream << (i == 0 ? "" : ", ") << values[i];
                stream << "]";
                return stream.str();
            }

            /** Check that a value can be used as an openPMD unitSI conversion factor.
             *
             * unitSI is the factor from stored to SI values and must be positive
             * and finite for a physical field record.
             */
            inline void requireValidUnitSI(std::string const& record, double unitSI)
            {
                if(!std::isfinite(unitSI) || unitSI <= 0.0)
                    throw std::runtime_error(
                        "openPMD: record '" + record + "' has invalid unitSI " + std::to_string(unitSI)
                        + "; expected a positive finite value");
            }

            /** Conversion factor from stored values to PIConGPU internal values.
             *
             * A PIConGPU-written checkpoint stores unitSI equal to the absolute
             * value of the deck unit, so the factor is exactly 1.0 and the restart
             * stays bit-identical. External files with a different unit system are
             * converted to internal units.
             */
            inline double toInternalFactor(std::string const& record, double fileUnitSI, double deckUnitSI)
            {
                requireValidUnitSI(record, fileUnitSI);
                if(!std::isfinite(deckUnitSI) || deckUnitSI <= 0.0)
                    throw std::runtime_error(
                        "openPMD: internal unit for record '" + record + "' is invalid: "
                        + std::to_string(deckUnitSI));
                return fileUnitSI / deckUnitSI;
            }

            /** Compare two values with a relative tolerance. */
            inline bool nearlyEqual(double left, double right, double relativeTolerance = 1e-6)
            {
                double const scale = std::max({1.0, std::abs(left), std::abs(right)});
                return std::abs(left - right) <= relativeTolerance * scale;
            }

            /** Validate the openPMD gridSpacing against the deck cell size.
             *
             * The vectors use openPMD axis order, which is reversed with respect
             * to PIConGPU's internal x, y, z order.
             */
            inline void checkGridSpacing(
                std::string const& record,
                std::vector<double> const& fileSpacing,
                std::vector<double> const& deckSpacing)
            {
                if(fileSpacing.size() != deckSpacing.size())
                    throw std::runtime_error(
                        "openPMD: record '" + record + "' has gridSpacing with "
                        + std::to_string(fileSpacing.size()) + " entries, the deck has "
                        + std::to_string(deckSpacing.size()));
                for(size_t i = 0; i < fileSpacing.size(); ++i)
                {
                    if(!nearlyEqual(fileSpacing[i], deckSpacing[i]))
                        throw std::runtime_error(
                            "openPMD: record '" + record + "' gridSpacing[" + std::to_string(i) + "]="
                            + std::to_string(fileSpacing[i]) + " differs from the deck value "
                            + std::to_string(deckSpacing[i]));
                }
            }

            /** Validate an in-cell component position against the deck field layout. */
            inline void checkPosition(
                std::string const& record,
                std::vector<double> const& filePosition,
                std::vector<double> const& deckPosition)
            {
                if(filePosition.size() != deckPosition.size())
                    throw std::runtime_error(
                        "openPMD: record '" + record + "' has position with "
                        + std::to_string(filePosition.size()) + " entries, the deck has "
                        + std::to_string(deckPosition.size()));
                for(size_t i = 0; i < filePosition.size(); ++i)
                {
                    if(!nearlyEqual(filePosition[i], deckPosition[i]))
                        throw std::runtime_error(
                            "openPMD: record '" + record + "' position[" + std::to_string(i) + "]="
                            + std::to_string(filePosition[i]) + " differs from the deck value "
                            + std::to_string(deckPosition[i]));
                }
            }


            /** Validate the openPMD gridGlobalOffset against the deck window origin. */
            inline void checkGridGlobalOffset(
                std::string const& record,
                std::vector<double> const& fileOffset,
                std::vector<double> const& deckOffset)
            {
                if(fileOffset.size() != deckOffset.size())
                    throw std::runtime_error(
                        "openPMD: record '" + record + "' has gridGlobalOffset with "
                        + std::to_string(fileOffset.size()) + " entries, the deck has "
                        + std::to_string(deckOffset.size()));
                for(size_t i = 0; i < fileOffset.size(); ++i)
                {
                    if(!nearlyEqual(fileOffset[i], deckOffset[i]))
                        throw std::runtime_error(
                            "openPMD: record '" + record + "' gridGlobalOffset[" + std::to_string(i) + "]="
                            + std::to_string(fileOffset[i]) + " differs from the deck value "
                            + std::to_string(deckOffset[i]));
                }
            }

            /** Validate the openPMD axis labels. */
            inline void checkAxisLabels(
                std::string const& record,
                std::vector<std::string> const& fileLabels,
                std::vector<std::string> const& deckLabels)
            {
                if(fileLabels != deckLabels)
                {
                    std::ostringstream file;
                    std::ostringstream deck;
                    for(size_t i = 0; i < fileLabels.size(); ++i)
                        file << (i == 0 ? "" : ",") << fileLabels[i];
                    for(size_t i = 0; i < deckLabels.size(); ++i)
                        deck << (i == 0 ? "" : ",") << deckLabels[i];
                    throw std::runtime_error(
                        "openPMD: record '" + record + "' axisLabels='" + file.str()
                        + "' differ from the deck layout '" + deck.str() + "'");
                }
            }

            /** Validate the openPMD unitDimension against the deck field dimension. */
            inline void checkUnitDimension(
                std::string const& record,
                std::vector<double> const& fileDimension,
                std::vector<double> const& deckDimension)
            {
                if(fileDimension.size() != deckDimension.size())
                    throw std::runtime_error(
                        "openPMD: record '" + record + "' has unitDimension with "
                        + std::to_string(fileDimension.size()) + " entries, the deck has "
                        + std::to_string(deckDimension.size()));
                for(size_t i = 0; i < fileDimension.size(); ++i)
                {
                    if(!nearlyEqual(fileDimension[i], deckDimension[i], 1e-12))
                        throw std::runtime_error(
                            "openPMD: record '" + record + "' unitDimension=" + toString(fileDimension)
                            + " differs from the deck value " + toString(deckDimension));
                }
            }

            /** Validate the openPMD timeOffset.
             *
             * The restart path has no way to apply a time offset, so a non-zero
             * value would silently start the simulation at the wrong time.
             */
            inline void checkTimeOffset(std::string const& record, double timeOffset)
            {
                if(!nearlyEqual(timeOffset, 0.0, 1e-12))
                    throw std::runtime_error(
                        "openPMD: record '" + record + "' has timeOffset " + std::to_string(timeOffset)
                        + ", the restart path requires 0");
            }
        } // namespace restartMetadata
    } // namespace openPMD
} // namespace picongpu
