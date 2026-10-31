/* Copyright 2026
 *
 * Bounded-memory openPMD particle source reader for the neural bunch
 * initializer (standalone; later ported into PIConGPU).
 *
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <mpi.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace picongpu::particles::neuralBunch
{

    /** Actionable error raised by the source reader.
     *
     * Every message names the file, record, and (where applicable) the global
     * particle index that failed validation, so callers can report it verbatim.
     */
    class SourceError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    /** Validated metadata of one particle species in one openPMD iteration.
     *
     * All unitSI values are the values found in the file (or the openPMD
     * default of 1 when a unitSI attribute is genuinely absent and the record
     * dimensionality matches). Physical quantities below are already in SI.
     */
    struct SourceMetadata
    {
        std::uint64_t globalCount = 0; //!< dataset extent of the position record
        std::string species;
        std::uint64_t iteration = 0;
        double time = 0.0; //!< iteration time, in units of timeUnitSI
        double dt = 0.0;
        double timeUnitSI = 1.0;

        double positionUnitSI[3] = {1.0, 1.0, 1.0};
        double positionOffsetUnitSI[3] = {1.0, 1.0, 1.0};
        double momentumUnitSI[3] = {1.0, 1.0, 1.0};
        double weightingUnitSI = 1.0;
        double chargeUnitSI = 1.0;
        double massUnitSI = 1.0;

        double charge = 0.0; //!< per-particle constant charge in C (if present)
        double mass = 0.0; //!< per-particle constant mass in kg (if present)

        int momentumMacroWeighted = 1;
        int momentumWeightingPower = 1;
        int weightingMacroWeighted = 1;
        int weightingWeightingPower = 1;

        bool hasPositionOffset = false;
        bool hasIds = false;
        bool hasCharge = false;
        bool hasMass = false;
        bool hasCellSize = false;
        double cellSizeSI[3] = {0.0, 0.0, 0.0};

        /** Position convention detected from metadata, never from the path.
         *
         * - additive: position and positionOffset share a per-axis unitSI, so
         *   physical = (position + positionOffset) * unitSI (openPMD /
         *   PIConGPU convention, MODEL-FORMAT).
         * - absolute: the per-axis unitSI values differ, so position already
         *   carries the physical coordinate: physical = position * unitSI.
         *   positionOffset is range-checked for consistency.
         */
        bool positionAbsolute = false;
        std::string positionConvention = "additive";
    };

    /** Host struct-of-arrays for one bounded chunk. */
    struct ParticleChunk
    {
        std::uint64_t begin = 0; //!< global index of the first particle
        std::vector<double> position; //!< 3 * size(), metres
        std::vector<double> u; //!< 3 * size(), single-electron u = p/(m c)
        std::vector<double> weight; //!< size()
        std::vector<std::uint64_t> id; //!< size()
        std::vector<double> charge; //!< size(), per-particle C
        std::vector<double> mass; //!< size(), per-particle kg
        std::vector<std::uint8_t> valid; //!< size(), 0 flags a non-finite value

        std::size_t size() const
        {
            return weight.size();
        }

        void resize(std::size_t n)
        {
            position.resize(3 * n);
            u.resize(3 * n);
            weight.resize(n);
            id.resize(n);
            charge.resize(n);
            mass.resize(n);
            valid.resize(n);
        }
    };

    /** Collective, bounded-memory openPMD particle reader.
     *
     * Construction opens the series with the supplied MPI communicator in
     * READ_ONLY mode and validates the metadata identically on all ranks. The
     * constructor is collective: if any rank fails validation, every rank
     * receives the same actionable error.
     *
     * readChunk() is a rank-0 operation. streamAll() and
     * readChunkCollective() are collective: rank 0 performs all backend reads,
     * then broadcasts the decoded arrays, so every rank observes the same
     * globally ordered stream and executes the same number of collectives
     * regardless of its own domain or of having zero local particles.
     *
     * There is intentionally no host-side double buffering: a callback may
     * consume, but must not retain, the chunk reference it receives.
     */
    class SourceReader
    {
    public:
        using ChunkCallback = std::function<void(ParticleChunk const&)>;

        SourceReader(
            std::string const& path,
            std::uint64_t iteration,
            std::string const& species,
            MPI_Comm comm);
        ~SourceReader();

        SourceReader(SourceReader const&) = delete;
        SourceReader& operator=(SourceReader const&) = delete;
        SourceReader(SourceReader&&) noexcept;
        SourceReader& operator=(SourceReader&&) noexcept;

        SourceMetadata const& metadata() const;
        std::uint64_t globalCount() const;

        /** Rank-0 only: read [begin, begin + count). Clamped to the dataset.
         *
         * Throws SourceError if called on a non-zero rank, on non-finite
         * decoded data, or on a position convention violation.
         */
        void readChunk(std::uint64_t begin, std::uint64_t count, ParticleChunk& out);

        /** Collective: one locked-step chunk, rank 0 reads then broadcasts. */
        void readChunkCollective(std::uint64_t begin, std::uint64_t count, ParticleChunk& out);

        /** Collective: stream the whole species in fixed-size global chunks.
         *
         * chunkSize == 0 is rejected. For an empty species the loop performs
         * zero iterations and therefore zero collectives on every rank.
         */
        void streamAll(ChunkCallback const& callback, std::uint64_t chunkSize = 1u << 20);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        MPI_Comm comm_ = MPI_COMM_NULL;
    };

} // namespace picongpu::particles::neuralBunch
