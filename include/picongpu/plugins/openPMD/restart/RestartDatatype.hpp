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

#if (ENABLE_OPENPMD == 1)

#    include <cstddef>
#    include <memory>
#    include <stdexcept>
#    include <string>
#    include <type_traits>

#    include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace openPMD
    {
        namespace restartDatatype
        {
            /** Datatypes that can be read and converted to the expected type. */
            inline bool isConvertibleFloatingPoint(::openPMD::Datatype const datatype)
            {
                return datatype == ::openPMD::Datatype::FLOAT || datatype == ::openPMD::Datatype::DOUBLE;
            }

            inline std::size_t numElements(::openPMD::Extent const& extent)
            {
                std::size_t result = 1;
                for(auto const value : extent)
                    result *= static_cast<std::size_t>(value);
                return result;
            }

            /** Load a record component into a buffer of the type PIConGPU expects.
             *
             * A matching datatype is read directly. A stored FLOAT or DOUBLE
             * loaded into a different floating-point type is converted
             * explicitly. The conversion happens after `synchronize` has made
             * the backend read visible, so the returned buffer is ready to use.
             * Any other mismatch, for example an integer dataset where a
             * floating-point value is expected, fails with the record name and
             * both datatypes instead of relying on a backend conversion error.
             *
             * @param converted set to true when a datatype conversion happened;
             *        the caller is expected to warn about it.
             */
            template<typename T_Value, typename T_Synchronize>
            std::shared_ptr<T_Value> loadChunkConverted(
                ::openPMD::RecordComponent& recordComponent,
                std::string const& recordName,
                ::openPMD::Offset const& start,
                ::openPMD::Extent const& count,
                T_Synchronize&& synchronize,
                bool& converted)
            {
                converted = false;
                auto const storedDatatype = recordComponent.getDatatype();
                auto const expectedDatatype = ::openPMD::determineDatatype<T_Value>();
                if(storedDatatype == expectedDatatype)
                    return recordComponent.loadChunk<T_Value>(start, count);

                if(!isConvertibleFloatingPoint(storedDatatype) || !std::is_floating_point_v<T_Value>)
                    throw std::runtime_error(
                        "openPMD: record '" + recordName + "' stores "
                        + ::openPMD::datatypeToString(storedDatatype) + " but the restart path requires "
                        + ::openPMD::datatypeToString(expectedDatatype)
                        + "; only FLOAT and DOUBLE checkpoint data can be converted");

                auto const elements = numElements(count);
                std::shared_ptr<T_Value> result(
                    new T_Value[elements],
                    [](T_Value* pointer) { delete[] pointer; });

                auto convert = [&](auto const* source)
                {
                    using SourceType = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
                    for(std::size_t i = 0; i < elements; ++i)
                    {
                        if constexpr(std::is_same_v<SourceType, T_Value>)
                            result.get()[i] = source[i];
                        else
                            result.get()[i] = static_cast<T_Value>(source[i]);
                    }
                };

                if(storedDatatype == ::openPMD::Datatype::FLOAT)
                {
                    auto source = recordComponent.loadChunk<float>(start, count);
                    synchronize();
                    convert(source.get());
                }
                else // DOUBLE
                {
                    auto source = recordComponent.loadChunk<double>(start, count);
                    synchronize();
                    convert(source.get());
                }
                converted = true;
                return result;
            }
        } // namespace restartDatatype
    } // namespace openPMD
} // namespace picongpu

#endif
