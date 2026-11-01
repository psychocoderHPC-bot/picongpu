/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */
#include "picongpu/particles/neuralBunch/SourceFile.hpp"

#if defined(PIC_ENABLE_NEURAL_BUNCH) && PIC_ENABLE_NEURAL_BUNCH

#    include <algorithm>
#    include <array>
#    include <cmath>
#    include <complex>
#    include <cstring>
#    include <limits>
#    include <optional>
#    include <sstream>
#    include <type_traits>
#    include <variant>

#    include <openPMD/openPMD.hpp>

namespace picongpu::particles::neuralBunch
{
    namespace
    {
        constexpr double electronMassSI = 9.1093837015e-31; // kg
        constexpr double speedOfLightSI = 299792458.0; // m/s
        constexpr double positionFractionTolerance = 1.0e-5;

        template<typename T>
        struct IsComplex : std::false_type
        {
        };

        template<typename T>
        struct IsComplex<std::complex<T>> : std::true_type
        {
        };

        using Variant = openPMD::RecordComponent::shared_ptr_dataset_types;

        [[noreturn]] void fail(std::string const& message)
        {
            throw SourceError(message);
        }

        std::string num(double value)
        {
            std::ostringstream os;
            os.precision(17);
            os << value;
            return os.str();
        }

        /** Collective failure propagation: if any rank recorded an error, every
         * rank throws the same message. The number and order of collectives is
         * independent of which rank failed. */
        void collectiveFail(std::string const& localError, MPI_Comm comm)
        {
            int rank = 0;
            int size = 1;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
            int const localHas = localError.empty() ? 0 : 1;
            int anyHas = 0;
            MPI_Allreduce(&localHas, &anyHas, 1, MPI_INT, MPI_MAX, comm);
            if(!anyHas)
                return;
            int const candidate = localError.empty() ? size : rank;
            int root = size;
            MPI_Allreduce(&candidate, &root, 1, MPI_INT, MPI_MIN, comm);
            unsigned long long length = (rank == root) ? static_cast<unsigned long long>(localError.size()) : 0ull;
            MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, root, comm);
            std::string message(static_cast<std::size_t>(length), '\0');
            if(rank == root && length > 0)
                std::memcpy(message.data(), localError.data(), localError.size());
            if(length > 0)
                MPI_Bcast(message.data(), static_cast<int>(length), MPI_CHAR, root, comm);
            throw SourceError(message);
        }

        bool dimensionsMatch(
            openPMD::unit_representations::AsArray const& actual,
            std::array<double, 7> const& expected)
        {
            for(std::size_t i = 0; i < 7; ++i)
                if(std::fabs(actual[i] - expected[i]) > 1.0e-12)
                    return false;
            return true;
        }

        std::string dimensionsToString(openPMD::unit_representations::AsArray const& d)
        {
            std::ostringstream os;
            os << "[" << d[0] << "," << d[1] << "," << d[2] << "," << d[3] << "," << d[4] << "," << d[5] << "," << d[6]
               << "]";
            return os.str();
        }

        double checkedUnitSI(
            openPMD::RecordComponent const& component,
            openPMD::Record const& record,
            std::array<double, 7> const& expectedDimensions,
            std::string const& label)
        {
            auto const dimensions = record.unitDimension();
            bool const dimensionsOk = dimensionsMatch(dimensions, expectedDimensions);
            bool const unitSIPresent = component.containsAttribute("unitSI");
            double const unitSI = unitSIPresent ? component.getAttribute("unitSI").get<double>() : 1.0;
            if(!std::isfinite(unitSI) || unitSI <= 0.0)
                fail(label + ": invalid unitSI " + num(unitSI) + " (must be finite and positive)");
            if(!dimensionsOk)
                fail(
                    label + ": unitDimension " + dimensionsToString(dimensions)
                    + " does not match expected L,M,T,I,theta,N,J = " + dimensionsToString(expectedDimensions)
                    + (unitSIPresent ? ""
                                     : " (unitSI is absent; default 1 is only allowed for matching dimensionality)"));
            return unitSI;
        }

        int attributeInt(
            openPMD::Attributable const& object,
            std::string const& key,
            int defaultValue,
            std::string const& label)
        {
            if(!object.containsAttribute(key))
                return defaultValue;
            double const value = object.getAttribute(key).get<double>();
            if(!std::isfinite(value))
                fail(label + ": attribute '" + key + "' is not finite");
            double const rounded = std::floor(value + 0.5);
            if(std::fabs(value - rounded) > 1.0e-9)
                fail(label + ": attribute '" + key + "' is not an integer");
            return static_cast<int>(rounded);
        }

        /** openPMD stores scalar records either as a child component with the
         * record's name or as a dataset directly on the record. Prefer the
         * named child: its extent/unitSI/constant state are fully loaded. */
        openPMD::RecordComponent scalarComponent(openPMD::Record& record, std::string const& name)
        {
            if(record.contains(name))
                return openPMD::RecordComponent(record[name]);
            return openPMD::RecordComponent(record);
        }

        double constantValue(openPMD::RecordComponent const& component, std::string const& label)
        {
            if(!component.containsAttribute("value"))
                fail(label + ": constant record component has no 'value' attribute");
            try
            {
                return component.getAttribute("value").get<double>();
            }
            catch(std::exception const& error)
            {
                fail(label + ": cannot read constant 'value': " + error.what());
            }
        }

        void convertDouble(Variant const& variant, double* destination, std::uint64_t count, std::string const& label)
        {
            std::visit(
                [&](auto const& pointer)
                {
                    using Element = typename std::decay_t<decltype(pointer)>::element_type;
                    if constexpr(IsComplex<Element>::value)
                    {
                        fail(label + ": complex-valued particle records are not supported");
                    }
                    else
                    {
                        if(!pointer && count > 0)
                            fail(label + ": backend returned a null data pointer");
                        for(std::uint64_t i = 0; i < count; ++i)
                            destination[i] = static_cast<double>(pointer.get()[i]);
                    }
                },
                variant);
        }

        void convertId(
            Variant const& variant,
            std::uint64_t* destination,
            std::uint64_t count,
            std::string const& label)
        {
            std::visit(
                [&](auto const& pointer)
                {
                    using Element = typename std::decay_t<decltype(pointer)>::element_type;
                    if constexpr(IsComplex<Element>::value)
                    {
                        fail(label + ": complex-valued id records are not supported");
                    }
                    else if constexpr(std::is_floating_point_v<Element>)
                    {
                        for(std::uint64_t i = 0; i < count; ++i)
                        {
                            long double const value = static_cast<long double>(pointer.get()[i]);
                            if(!std::isfinite(value) || value < 0.0L
                               || value > static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
                                fail(label + ": id at local index " + std::to_string(i) + " is not a valid uint64");
                            auto const converted = static_cast<std::uint64_t>(value);
                            if(static_cast<long double>(converted) != value)
                                fail(label + ": id at local index " + std::to_string(i) + " is not integral");
                            destination[i] = converted;
                        }
                    }
                    else if constexpr(std::is_integral_v<Element>)
                    {
                        for(std::uint64_t i = 0; i < count; ++i)
                        {
                            if constexpr(std::is_signed_v<Element>)
                                if(pointer.get()[i] < 0)
                                    fail(label + ": id at local index " + std::to_string(i) + " is negative");
                            destination[i] = static_cast<std::uint64_t>(pointer.get()[i]);
                        }
                    }
                    else
                    {
                        fail(label + ": unsupported id datatype");
                    }
                },
                variant);
        }

        /** One batch of double-valued components: schedule all backend reads,
         * flush once, then convert into the caller's bounded buffers. */
        struct DoubleRequest
        {
            openPMD::RecordComponent component;
            std::vector<double>* destination;
            std::string label;
        };

        void loadDoubleBatch(
            openPMD::Series& series,
            std::uint64_t begin,
            std::uint64_t count,
            std::vector<DoubleRequest>& requests)
        {
            std::vector<Variant> pending;
            std::vector<std::size_t> pendingIndex;
            for(std::size_t i = 0; i < requests.size(); ++i)
            {
                auto& request = requests[i];
                request.destination->resize(count);
                if(request.component.constant())
                {
                    double const value = constantValue(request.component, request.label);
                    std::fill(request.destination->begin(), request.destination->end(), value);
                }
                else if(count == 0)
                {
                    // no backend request needed
                }
                else
                {
                    pending.push_back(
                        request.component.loadChunkVariant(openPMD::Offset{begin}, openPMD::Extent{count}));
                    pendingIndex.push_back(i);
                }
            }
            if(!pending.empty())
                series.flush();
            for(std::size_t k = 0; k < pending.size(); ++k)
            {
                auto const& request = requests[pendingIndex[k]];
                convertDouble(pending[k], request.destination->data(), count, request.label);
            }
        }

        struct MetaWire
        {
            std::uint64_t globalCount;
            std::uint64_t iteration;
            double time;
            double dt;
            double timeUnitSI;
            double positionUnitSI[3];
            double positionOffsetUnitSI[3];
            double momentumUnitSI[3];
            double weightingUnitSI;
            double chargeUnitSI;
            double massUnitSI;
            double charge;
            double mass;
            double cellSizeSI[3];
            int momentumMacroWeighted;
            int momentumWeightingPower;
            int weightingMacroWeighted;
            int weightingWeightingPower;
            int hasPositionOffset;
            int hasIds;
            int hasCharge;
            int hasMass;
            int hasCellSize;
            int positionAbsolute;
        };

        void encode(SourceMetadata const& m, MetaWire& w)
        {
            std::memset(&w, 0, sizeof(w));
            w.globalCount = m.globalCount;
            w.iteration = m.iteration;
            w.time = m.time;
            w.dt = m.dt;
            w.timeUnitSI = m.timeUnitSI;
            for(int i = 0; i < 3; ++i)
            {
                w.positionUnitSI[i] = m.positionUnitSI[i];
                w.positionOffsetUnitSI[i] = m.positionOffsetUnitSI[i];
                w.momentumUnitSI[i] = m.momentumUnitSI[i];
                w.cellSizeSI[i] = m.cellSizeSI[i];
            }
            w.weightingUnitSI = m.weightingUnitSI;
            w.chargeUnitSI = m.chargeUnitSI;
            w.massUnitSI = m.massUnitSI;
            w.charge = m.charge;
            w.mass = m.mass;
            w.momentumMacroWeighted = m.momentumMacroWeighted;
            w.momentumWeightingPower = m.momentumWeightingPower;
            w.weightingMacroWeighted = m.weightingMacroWeighted;
            w.weightingWeightingPower = m.weightingWeightingPower;
            w.hasPositionOffset = m.hasPositionOffset;
            w.hasIds = m.hasIds;
            w.hasCharge = m.hasCharge;
            w.hasMass = m.hasMass;
            w.hasCellSize = m.hasCellSize;
            w.positionAbsolute = m.positionAbsolute;
        }

        void decode(MetaWire const& w, SourceMetadata& m)
        {
            m.globalCount = w.globalCount;
            m.iteration = w.iteration;
            m.time = w.time;
            m.dt = w.dt;
            m.timeUnitSI = w.timeUnitSI;
            for(int i = 0; i < 3; ++i)
            {
                m.positionUnitSI[i] = w.positionUnitSI[i];
                m.positionOffsetUnitSI[i] = w.positionOffsetUnitSI[i];
                m.momentumUnitSI[i] = w.momentumUnitSI[i];
                m.cellSizeSI[i] = w.cellSizeSI[i];
            }
            m.weightingUnitSI = w.weightingUnitSI;
            m.chargeUnitSI = w.chargeUnitSI;
            m.massUnitSI = w.massUnitSI;
            m.charge = w.charge;
            m.mass = w.mass;
            m.momentumMacroWeighted = w.momentumMacroWeighted;
            m.momentumWeightingPower = w.momentumWeightingPower;
            m.weightingMacroWeighted = w.weightingMacroWeighted;
            m.weightingWeightingPower = w.weightingWeightingPower;
            m.hasPositionOffset = w.hasPositionOffset != 0;
            m.hasIds = w.hasIds != 0;
            m.hasCharge = w.hasCharge != 0;
            m.hasMass = w.hasMass != 0;
            m.hasCellSize = w.hasCellSize != 0;
            m.positionAbsolute = w.positionAbsolute != 0;
            m.positionConvention = m.positionAbsolute ? "absolute" : "additive";
        }
    } // namespace

    struct SourceReader::Impl
    {
        MPI_Comm comm = MPI_COMM_NULL;
        int rank = 0;
        int size = 1;
        std::string path;
        std::string species;
        SourceMetadata meta;

        std::unique_ptr<openPMD::Series> series;

        std::optional<openPMD::RecordComponent> position[3];
        std::optional<openPMD::RecordComponent> positionOffset[3];
        std::optional<openPMD::RecordComponent> momentum[3];
        std::optional<openPMD::RecordComponent> weighting;
        std::optional<openPMD::RecordComponent> id;
        std::optional<openPMD::RecordComponent> charge;
        std::optional<openPMD::RecordComponent> mass;
    };

    SourceReader::SourceReader(
        std::string const& path,
        std::uint64_t iteration,
        std::string const& species,
        MPI_Comm comm)
        : impl_(std::make_unique<Impl>())
        , comm_(comm)
    {
        impl_->comm = comm;
        impl_->path = path;
        impl_->species = species;
        MPI_Comm_rank(comm, &impl_->rank);
        MPI_Comm_size(comm, &impl_->size);

        std::string localError;
        try
        {
            impl_->series = std::make_unique<openPMD::Series>(path, openPMD::Access::READ_ONLY, comm);
            auto& series = *impl_->series;
            if(!series.iterations.contains(iteration))
            {
                std::ostringstream os;
                os << path << ": iteration " << iteration << " not present (available iterations:";
                for(auto const& entry : series.iterations)
                    os << " " << entry.first;
                os << ")";
                fail(os.str());
            }
            auto iterationHandle = series.iterations[iteration];
            if(!iterationHandle.particles.contains(species))
            {
                std::ostringstream os;
                os << path << ": species '" << species << "' not present in iteration " << iteration
                   << " (available species:";
                for(auto const& entry : iterationHandle.particles)
                    os << " " << entry.first;
                os << ")";
                fail(os.str());
            }
            auto particles = iterationHandle.particles[species];

            SourceMetadata& meta = impl_->meta;
            meta.species = species;
            meta.iteration = iteration;
            meta.time = iterationHandle.time<double>();
            meta.dt = iterationHandle.dt<double>();
            meta.timeUnitSI = iterationHandle.timeUnitSI();
            if(!std::isfinite(meta.timeUnitSI) || meta.timeUnitSI <= 0.0)
                fail(path + ": invalid iteration timeUnitSI " + num(meta.timeUnitSI));

            auto requireRecord = [&](std::string const& name) -> openPMD::Record
            {
                if(!particles.contains(name))
                    fail(path + ": species '" + species + "' has no required record '" + name + "'");
                return particles[name];
            };

            static constexpr std::array<double, 7> lengthDimensions = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            static constexpr std::array<double, 7> momentumDimensions = {1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0};
            static constexpr std::array<double, 7> dimensionless = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            // --- position (required, vector) ---
            openPMD::Record positionRecord = requireRecord("position");
            if(positionRecord.empty())
                fail(path + ": record 'position' has no x/y/z components");
            for(int axis = 0; axis < 3; ++axis)
            {
                std::string const name = std::string(1, "xyz"[axis]);
                if(!positionRecord.contains(name))
                    fail(path + ": record 'position' is missing component '" + name + "'");
                impl_->position[axis] = positionRecord[name];
                meta.positionUnitSI[axis] = checkedUnitSI(
                    *impl_->position[axis],
                    positionRecord,
                    lengthDimensions,
                    path + ": position/" + name);
            }

            // --- positionOffset (required by this reader, see README) ---
            openPMD::Record offsetRecord = requireRecord("positionOffset");
            if(offsetRecord.empty())
                fail(path + ": record 'positionOffset' has no x/y/z components");
            for(int axis = 0; axis < 3; ++axis)
            {
                std::string const name = std::string(1, "xyz"[axis]);
                if(!offsetRecord.contains(name))
                    fail(path + ": record 'positionOffset' is missing component '" + name + "'");
                impl_->positionOffset[axis] = offsetRecord[name];
                meta.positionOffsetUnitSI[axis] = checkedUnitSI(
                    *impl_->positionOffset[axis],
                    offsetRecord,
                    lengthDimensions,
                    path + ": positionOffset/" + name);
            }
            meta.hasPositionOffset = true;

            // --- momentum (required, vector) ---
            openPMD::Record momentumRecord = requireRecord("momentum");
            if(momentumRecord.empty())
                fail(path + ": record 'momentum' has no x/y/z components");
            for(int axis = 0; axis < 3; ++axis)
            {
                std::string const name = std::string(1, "xyz"[axis]);
                if(!momentumRecord.contains(name))
                    fail(path + ": record 'momentum' is missing component '" + name + "'");
                impl_->momentum[axis] = momentumRecord[name];
                meta.momentumUnitSI[axis] = checkedUnitSI(
                    *impl_->momentum[axis],
                    momentumRecord,
                    momentumDimensions,
                    path + ": momentum/" + name);
            }
            meta.momentumMacroWeighted = attributeInt(momentumRecord, "macroWeighted", 1, path + ": momentum");
            meta.momentumWeightingPower = attributeInt(momentumRecord, "weightingPower", 1, path + ": momentum");
            if(meta.momentumWeightingPower < 0)
                fail(path + ": momentum/weightingPower must be non-negative");

            // --- weighting (required, scalar) ---
            openPMD::Record weightingRecord = requireRecord("weighting");
            impl_->weighting = scalarComponent(weightingRecord, "weighting");
            meta.weightingUnitSI
                = checkedUnitSI(*impl_->weighting, weightingRecord, dimensionless, path + ": weighting");
            meta.weightingMacroWeighted = attributeInt(weightingRecord, "macroWeighted", 1, path + ": weighting");
            meta.weightingWeightingPower = attributeInt(weightingRecord, "weightingPower", 1, path + ": weighting");

            // --- optional id ---
            if(particles.contains("id"))
            {
                openPMD::Record idRecord = particles["id"];
                impl_->id = scalarComponent(idRecord, "id");
                meta.hasIds = true;
            }

            // --- optional constant charge/mass ---
            if(particles.contains("charge"))
            {
                openPMD::Record chargeRecord = particles["charge"];
                impl_->charge = scalarComponent(chargeRecord, "charge");
                meta.chargeUnitSI = chargeRecord.containsAttribute("unitSI")
                                        ? chargeRecord.getAttribute("unitSI").get<double>()
                                        : 1.0;
                if(!std::isfinite(meta.chargeUnitSI) || meta.chargeUnitSI <= 0.0)
                    fail(path + ": invalid charge unitSI");
                if(impl_->charge->constant())
                    meta.charge = constantValue(*impl_->charge, path + ": charge") * meta.chargeUnitSI;
                else
                {
                    auto const chargeExtent = impl_->charge->getExtent();
                    if(chargeExtent.size() != 1 || chargeExtent[0] != impl_->position[0]->getExtent()[0])
                        fail(path + ": charge extent does not match position/x");
                }
                meta.hasCharge = true;
            }
            if(particles.contains("mass"))
            {
                openPMD::Record massRecord = particles["mass"];
                impl_->mass = scalarComponent(massRecord, "mass");
                meta.massUnitSI
                    = massRecord.containsAttribute("unitSI") ? massRecord.getAttribute("unitSI").get<double>() : 1.0;
                if(!std::isfinite(meta.massUnitSI) || meta.massUnitSI <= 0.0)
                    fail(path + ": invalid mass unitSI");
                if(impl_->mass->constant())
                    meta.mass = constantValue(*impl_->mass, path + ": mass") * meta.massUnitSI;
                else
                {
                    auto const massExtent = impl_->mass->getExtent();
                    if(massExtent.size() != 1 || massExtent[0] != impl_->position[0]->getExtent()[0])
                        fail(path + ": mass extent does not match position/x");
                }
                meta.hasMass = true;
            }

            // --- dataset extent is authoritative for the global count ---
            auto const extent = impl_->position[0]->getExtent();
            if(extent.size() != 1)
                fail(path + ": position/x is not a one-dimensional record");
            meta.globalCount = extent[0];
            for(int axis = 0; axis < 3; ++axis)
            {
                auto const positionExtent = impl_->position[axis]->getExtent();
                auto const offsetExtent = impl_->positionOffset[axis]->getExtent();
                auto const momentumExtent = impl_->momentum[axis]->getExtent();
                if(positionExtent.size() != 1 || positionExtent[0] != meta.globalCount)
                    fail(
                        path + ": position component " + std::string(1, "xyz"[axis])
                        + " extent does not match position/x");
                if(offsetExtent.size() != 1 || offsetExtent[0] != meta.globalCount)
                    fail(
                        path + ": positionOffset component " + std::string(1, "xyz"[axis])
                        + " extent does not match position/x");
                if(momentumExtent.size() != 1 || momentumExtent[0] != meta.globalCount)
                    fail(
                        path + ": momentum component " + std::string(1, "xyz"[axis])
                        + " extent does not match position/x");
            }
            auto const weightingExtent = impl_->weighting->getExtent();
            if(weightingExtent.size() != 1 || weightingExtent[0] != meta.globalCount)
                fail(path + ": weighting extent does not match position/x");
            if(impl_->id)
            {
                auto const idExtent = impl_->id->getExtent();
                if(idExtent.size() != 1 || idExtent[0] != meta.globalCount)
                    fail(path + ": id extent does not match position/x");
            }

            // --- decide the position convention from metadata only ---
            bool sameUnits = true;
            for(int axis = 0; axis < 3; ++axis)
            {
                double const p = meta.positionUnitSI[axis];
                double const o = meta.positionOffsetUnitSI[axis];
                if(std::fabs(p - o) > 1.0e-12 * std::max(1.0, std::max(std::fabs(p), std::fabs(o))))
                    sameUnits = false;
            }
            meta.positionAbsolute = !sameUnits;
            meta.positionConvention = meta.positionAbsolute ? "absolute" : "additive";
            if(!meta.positionAbsolute)
            {
                // OpenPMD/PIConGPU: position unitSI is the cell size. In this
                // representation position must stay inside one cell.
                for(int axis = 0; axis < 3; ++axis)
                    meta.cellSizeSI[axis] = meta.positionUnitSI[axis];
            }
            else
            {
                // Absolute-coordinate variant: positionOffset unitSI is the
                // cell size, position is already the physical coordinate.
                for(int axis = 0; axis < 3; ++axis)
                {
                    meta.cellSizeSI[axis] = meta.positionOffsetUnitSI[axis];
                    if(!std::isfinite(meta.cellSizeSI[axis]) || meta.cellSizeSI[axis] <= 0.0)
                        fail(path + ": positionOffset unitSI is not a positive cell size");
                }
            }
            meta.hasCellSize = true;

            if(meta.globalCount > std::numeric_limits<std::size_t>::max() / 128u)
                fail(path + ": particle count does not fit the host address space");
        }
        catch(std::exception const& error)
        {
            localError = std::string("SourceFile(") + path + ", species " + species + ", iteration "
                         + std::to_string(iteration) + "): " + error.what();
        }

        collectiveFail(localError, comm);

        MetaWire wire{};
        if(impl_->rank == 0)
            encode(impl_->meta, wire);
        MPI_Bcast(&wire, static_cast<int>(sizeof(MetaWire)), MPI_BYTE, 0, comm);
        decode(wire, impl_->meta);
        impl_->meta.species = species;
    }

    SourceReader::~SourceReader() = default;

    SourceReader::SourceReader(SourceReader&& other) noexcept : impl_(std::move(other.impl_)), comm_(other.comm_)
    {
        other.comm_ = MPI_COMM_NULL;
    }

    SourceReader& SourceReader::operator=(SourceReader&& other) noexcept
    {
        impl_ = std::move(other.impl_);
        comm_ = other.comm_;
        other.comm_ = MPI_COMM_NULL;
        return *this;
    }

    SourceMetadata const& SourceReader::metadata() const
    {
        return impl_->meta;
    }

    std::uint64_t SourceReader::globalCount() const
    {
        return impl_->meta.globalCount;
    }

    void SourceReader::readChunk(std::uint64_t begin, std::uint64_t count, ParticleChunk& out)
    {
        Impl& d = *impl_;
        if(d.rank != 0)
            fail("SourceReader::readChunk is a rank-0 operation; use readChunkCollective/streamAll");
        if(begin > d.meta.globalCount)
            fail(
                d.path + ": chunk begin " + std::to_string(begin) + " exceeds global count "
                + std::to_string(d.meta.globalCount));
        if(count > std::numeric_limits<std::uint64_t>::max() - begin)
            fail(d.path + ": chunk begin + count overflows uint64");
        std::uint64_t const remaining = d.meta.globalCount - begin;
        std::uint64_t const n = std::min(count, remaining);
        if(n > std::numeric_limits<std::size_t>::max() / 128u)
            fail(d.path + ": chunk size does not fit the host address space");

        out.begin = begin;
        out.resize(static_cast<std::size_t>(n));
        if(n == 0)
            return;

        std::vector<double> rawPosition[3];
        std::vector<double> rawOffset[3];
        std::vector<double> rawMomentum[3];
        std::vector<double> rawWeight;
        std::vector<double> rawCharge;
        std::vector<double> rawMass;
        std::vector<std::uint64_t> rawId;

        std::vector<DoubleRequest> requests;
        for(int axis = 0; axis < 3; ++axis)
        {
            std::string const suffix = std::string("position/") + "xyz"[axis];
            requests.push_back({*d.position[axis], &rawPosition[axis], d.path + ": " + suffix});
        }
        for(int axis = 0; axis < 3; ++axis)
        {
            std::string const suffix = std::string("positionOffset/") + "xyz"[axis];
            requests.push_back({*d.positionOffset[axis], &rawOffset[axis], d.path + ": " + suffix});
        }
        for(int axis = 0; axis < 3; ++axis)
        {
            std::string const suffix = std::string("momentum/") + "xyz"[axis];
            requests.push_back({*d.momentum[axis], &rawMomentum[axis], d.path + ": " + suffix});
        }
        requests.push_back({*d.weighting, &rawWeight, d.path + ": weighting"});
        if(d.charge)
            requests.push_back({*d.charge, &rawCharge, d.path + ": charge"});
        else
            rawCharge.assign(static_cast<std::size_t>(n), 0.0);
        if(d.mass)
            requests.push_back({*d.mass, &rawMass, d.path + ": mass"});
        else
            rawMass.assign(static_cast<std::size_t>(n), 0.0);

        loadDoubleBatch(*d.series, begin, n, requests);

        // charge/mass records are stored in their record units; expose SI.
        for(std::uint64_t i = 0; i < n; ++i)
        {
            rawCharge[static_cast<std::size_t>(i)] *= d.meta.chargeUnitSI;
            rawMass[static_cast<std::size_t>(i)] *= d.meta.massUnitSI;
        }

        if(d.id)
        {
            rawId.resize(static_cast<std::size_t>(n));
            if(d.id->constant())
            {
                std::uint64_t const value = static_cast<std::uint64_t>(constantValue(*d.id, d.path + ": id"));
                std::fill(rawId.begin(), rawId.end(), value);
            }
            else
            {
                auto variant = d.id->loadChunkVariant(openPMD::Offset{begin}, openPMD::Extent{n});
                d.series->flush();
                convertId(variant, rawId.data(), n, d.path + ": id");
            }
        }
        else
        {
            rawId.resize(static_cast<std::size_t>(n));
            for(std::uint64_t i = 0; i < n; ++i)
                rawId[i] = begin + i;
        }

        SourceMetadata const& meta = d.meta;
        int const momentumEffectivePower = meta.momentumMacroWeighted != 0 ? meta.momentumWeightingPower : 0;

        std::size_t firstInvalid = std::numeric_limits<std::size_t>::max();
        std::string firstInvalidField;
        for(std::uint64_t i = 0; i < n; ++i)
        {
            double const weight = rawWeight[static_cast<std::size_t>(i)] * meta.weightingUnitSI;
            double const factor = momentumEffectivePower == 0 ? 1.0 : std::pow(weight, meta.momentumWeightingPower);
            double const charge
                = d.charge ? rawCharge[static_cast<std::size_t>(i)] : (meta.hasCharge ? meta.charge : 0.0);
            double const mass = d.mass ? rawMass[static_cast<std::size_t>(i)] : (meta.hasMass ? meta.mass : 0.0);
            double const restMass = mass > 0.0 ? mass : electronMassSI;
            bool valid = true;
            auto const flag = [&](double value, std::string const& field)
            {
                if(!std::isfinite(value))
                {
                    valid = false;
                    if(firstInvalid == std::numeric_limits<std::size_t>::max())
                    {
                        firstInvalid = static_cast<std::size_t>(i);
                        firstInvalidField = field;
                    }
                }
            };

            flag(weight, "weighting");
            flag(charge, "charge");
            flag(mass, "mass");
            if(momentumEffectivePower != 0 && weight <= 0.0)
            {
                valid = false;
                if(firstInvalid == std::numeric_limits<std::size_t>::max())
                {
                    firstInvalid = static_cast<std::size_t>(i);
                    firstInvalidField = "weighting (non-positive with weightingPower != 0)";
                }
            }

            out.weight[static_cast<std::size_t>(i)] = weight;
            out.id[static_cast<std::size_t>(i)] = rawId[static_cast<std::size_t>(i)];
            out.charge[static_cast<std::size_t>(i)] = charge;
            out.mass[static_cast<std::size_t>(i)] = mass;

            for(int axis = 0; axis < 3; ++axis)
            {
                double const positionRaw = rawPosition[axis][static_cast<std::size_t>(i)];
                double const offsetRaw = rawOffset[axis][static_cast<std::size_t>(i)];
                double physical = 0.0;
                if(meta.positionAbsolute)
                {
                    physical = positionRaw * meta.positionUnitSI[axis];
                    double const cellSize = meta.cellSizeSI[axis];
                    double const ratio = physical / cellSize;
                    double const fraction = ratio - offsetRaw;
                    if(fraction < -1.0e-6 || fraction >= 1.0 + 1.0e-6)
                        fail(
                            d.path + ": positionOffset/x at global index " + std::to_string(begin + i)
                            + " is inconsistent with floor(position/cellSize) for the absolute-position "
                              "convention");
                }
                else
                {
                    if(positionRaw < -positionFractionTolerance || positionRaw > 1.0 + positionFractionTolerance)
                        fail(
                            d.path + ": position/" + std::string(1, "xyz"[axis]) + " at global index "
                            + std::to_string(begin + i) + " is " + num(positionRaw)
                            + " but unitSI equals the positionOffset cell size; the additive openPMD "
                              "convention requires position inside [0,1). Rejecting non-standard file.");
                    physical = (positionRaw + offsetRaw) * meta.positionUnitSI[axis];
                }
                double const u = rawMomentum[axis][static_cast<std::size_t>(i)] * meta.momentumUnitSI[axis]
                                 / (restMass * speedOfLightSI * factor);
                out.position[3 * static_cast<std::size_t>(i) + axis] = physical;
                out.u[3 * static_cast<std::size_t>(i) + axis] = u;
                flag(physical, "position");
                flag(u, "momentum");
            }
            out.valid[static_cast<std::size_t>(i)] = valid ? 1u : 0u;
        }

        if(firstInvalid != std::numeric_limits<std::size_t>::max())
            fail(
                d.path + ": non-finite or invalid " + firstInvalidField + " at global particle index "
                + std::to_string(begin + firstInvalid) + " (species " + d.meta.species + ")");
    }

    void SourceReader::readChunkCollective(std::uint64_t begin, std::uint64_t count, ParticleChunk& out)
    {
        Impl& d = *impl_;
        std::string localError;
        if(d.rank == 0)
        {
            try
            {
                readChunk(begin, count, out);
            }
            catch(std::exception const& error)
            {
                localError = error.what();
            }
        }
        collectiveFail(localError, comm_);

        unsigned long long size = (d.rank == 0) ? static_cast<unsigned long long>(out.size()) : 0ull;
        MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG_LONG, 0, comm_);
        auto const n = static_cast<std::size_t>(size);
        if(n > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 3u)
            fail("SourceReader::readChunkCollective: chunk is too large for an MPI_Bcast count");
        if(d.rank != 0)
            out.resize(n);
        out.begin = begin;
        if(n == 0)
            return;
        MPI_Bcast(out.position.data(), static_cast<int>(3 * n), MPI_DOUBLE, 0, comm_);
        MPI_Bcast(out.u.data(), static_cast<int>(3 * n), MPI_DOUBLE, 0, comm_);
        MPI_Bcast(out.weight.data(), static_cast<int>(n), MPI_DOUBLE, 0, comm_);
        MPI_Bcast(out.id.data(), static_cast<int>(n), MPI_UNSIGNED_LONG_LONG, 0, comm_);
        MPI_Bcast(out.charge.data(), static_cast<int>(n), MPI_DOUBLE, 0, comm_);
        MPI_Bcast(out.mass.data(), static_cast<int>(n), MPI_DOUBLE, 0, comm_);
        MPI_Bcast(out.valid.data(), static_cast<int>(n), MPI_UNSIGNED_CHAR, 0, comm_);
    }

    void SourceReader::streamAll(ChunkCallback const& callback, std::uint64_t chunkSize)
    {
        Impl& d = *impl_;
        if(chunkSize == 0)
            fail("SourceReader::streamAll requires chunkSize > 0");
        std::uint64_t const total = d.meta.globalCount;
        ParticleChunk chunk;
        for(std::uint64_t begin = 0; begin < total;)
        {
            std::uint64_t const count = std::min(chunkSize, total - begin);
            readChunkCollective(begin, count, chunk);
            callback(chunk);
            begin += count;
        }
    }

} // namespace picongpu::particles::neuralBunch

#endif // PIC_ENABLE_NEURAL_BUNCH
