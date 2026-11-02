/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Host tests for the openPMD position decode in SourceFile.cpp.
 *
 * The reader must apply the standard openPMD rule component-wise:
 *
 *     x = position * positionUnitSI + positionOffset * positionOffsetUnitSI
 *
 * with no inference from the unit scales and no range restriction on the raw
 * position. Each fixture is a separate openPMD series written by this test.
 */

#include "picongpu/particles/neuralBunch/SourceFile.hpp"

#include <openPMD/openPMD.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <mpi.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace nb = picongpu::particles::neuralBunch;

namespace
{
    int checks = 0;
    int failures = 0;

    void check(bool const ok, std::string const& what)
    {
        ++checks;
        if(!ok)
        {
            ++failures;
            std::cerr << "FAIL: " << what << "\n";
        }
    }

    constexpr std::array<double, 7> lengthDimensions = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    constexpr std::array<double, 7> momentumDimensions = {1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0};
    constexpr std::array<double, 7> dimensionless = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    struct Fixture
    {
        std::string name;
        std::array<std::vector<double>, 3> position;
        std::array<double, 3> positionUnitSI;
        bool hasOffset = true;
        std::array<std::vector<double>, 3> offset;
        std::array<double, 3> offsetUnitSI;
    };

    void writeVector(
        openPMD::Record& record,
        std::string const& component,
        std::vector<double> const& values,
        double unitSI,
        std::array<double, 7> const& dimensions)
    {
        record.setUnitDimension(dimensions);
        auto rc = record[component];
        rc.setUnitSI(unitSI);
        rc.resetDataset(openPMD::Dataset(openPMD::Datatype::DOUBLE, {values.size()}));
        std::shared_ptr<double> data(new double[values.size()], std::default_delete<double[]>());
        std::copy(values.begin(), values.end(), data.get());
        rc.storeChunk(data, openPMD::Offset{0}, openPMD::Extent{values.size()});
    }

    void writeFixture(fs::path const& path, Fixture const& f)
    {
        std::size_t const n = f.position[0].size();
        openPMD::Series series(path.string(), openPMD::Access::CREATE, MPI_COMM_WORLD);
        auto iteration = series.iterations[0];
        auto species = iteration.particles["electrons"];

        for(int axis = 0; axis < 3; ++axis)
        {
            std::string const name(1, "xyz"[axis]);
            writeVector(species["position"], name, f.position[axis], f.positionUnitSI[axis], lengthDimensions);

            std::vector<double> u(n, 0.0);
            writeVector(species["momentum"], name, u, 1.0, momentumDimensions);

            if(f.hasOffset)
                writeVector(
                    species["positionOffset"],
                    name,
                    f.offset[axis],
                    f.offsetUnitSI[axis],
                    lengthDimensions);
        }
        writeVector(species["weighting"], "weighting", std::vector<double>(n, 1.0), 1.0, dimensionless);
        series.flush();
    }

    /** File extension of an available openPMD backend that can store particle data.
     *
     * The test must not assume HDF5: openPMD may be built with ADIOS2 only.
     */
    std::string seriesExtension()
    {
        auto const available = openPMD::getFileExtensions();
        for(std::string const& preferred : {"h5", "bp5", "bp4"})
            if(std::find(available.begin(), available.end(), preferred) != available.end())
                return preferred;
        return available.empty() ? std::string{"h5"} : available.front();
    }

    void runFixture(fs::path const& root, Fixture const& f)
    {
        fs::path const path = root / (f.name + "." + seriesExtension());
        try
        {
            writeFixture(path, f);
            nb::SourceReader reader(path.string(), 0, "electrons", MPI_COMM_WORLD);
            check(reader.globalCount() == f.position[0].size(), f.name + ": global count");

            nb::ParticleChunk chunk;
            reader.readChunk(0, reader.globalCount(), chunk);
            check(chunk.size() == f.position[0].size(), f.name + ": chunk size");

            for(int axis = 0; axis < 3; ++axis)
            {
                check(
                    reader.metadata().positionUnitSI[axis] == f.positionUnitSI[axis],
                    f.name + ": positionUnitSI axis " + std::to_string(axis));
                check(
                    reader.metadata().hasPositionOffset == f.hasOffset,
                    f.name + ": hasPositionOffset axis " + std::to_string(axis));
                for(std::size_t i = 0; i < chunk.size(); ++i)
                {
                    double const p = f.position[axis][i] * f.positionUnitSI[axis];
                    double const o = f.hasOffset ? f.offset[axis][i] * f.offsetUnitSI[axis] : 0.0;
                    double const expected = p + o;
                    double const actual = chunk.position[3 * i + std::size_t(axis)];
                    double const scale = std::max(1.0, std::max(std::fabs(expected), std::fabs(actual)));
                    check(
                        std::fabs(actual - expected) <= 1.0e-12 * scale,
                        f.name + ": axis " + std::to_string(axis) + " particle " + std::to_string(i) + " expected "
                            + std::to_string(expected) + " got " + std::to_string(actual));
                }
            }
        }
        catch(std::exception const& error)
        {
            check(false, f.name + ": unexpected exception: " + error.what());
        }
    }

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    fs::path const root = fs::temp_directory_path() / ("neuralBunchSourceFileTest-" + std::to_string(::getpid()));
    fs::remove_all(root);
    fs::create_directories(root);

    std::vector<Fixture> fixtures;

    // E1-a: in-cell fraction with zero offset and equal unit scales.
    {
        Fixture f;
        f.name = "fractional_zero_offset";
        f.position = {{{0.25, 0.5}, {0.125, 0.75}, {0.9, 0.0}}};
        f.positionUnitSI = {1.0e-6, 2.0e-6, 3.0e-6};
        f.offset = {{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}};
        f.offsetUnitSI = f.positionUnitSI;
        fixtures.push_back(f);
    }

    // E1-b: independent position/Offset units with a nonzero cell offset.
    {
        Fixture f;
        f.name = "independent_units_nonzero_offset";
        f.position = {{{2.0, 0.5}, {3.0, 1.0}, {4.0, 2.0}}};
        f.positionUnitSI = {1.0e-6, 1.0e-6, 1.0e-6};
        f.offset = {{{5.0, -1.0}, {6.0, 0.0}, {7.0, 3.0}}};
        f.offsetUnitSI = {2.0e-6, 3.0e-6, 4.0e-6};
        fixtures.push_back(f);
    }

    // E1-c: absolute SI position, unitSI = 1, zero offset. The raw position is
    // far outside [0,1); it must still be accepted.
    {
        Fixture f;
        f.name = "absolute_si_zero_offset";
        f.position = {{{1.5}, {2.5}, {3.5}}};
        f.positionUnitSI = {1.0, 1.0, 1.0};
        f.offset = {{{0.0}, {0.0}, {0.0}}};
        f.offsetUnitSI = {1.0, 1.0, 1.0};
        fixtures.push_back(f);
    }

    // E1-d: equal unit scales with a nonzero constant offset and a raw
    // position outside [0,1). The old equal-unit branch rejected this.
    {
        Fixture f;
        f.name = "equal_units_nonzero_offset_out_of_range";
        f.position = {{{5.25}, {6.5}, {7.75}}};
        f.positionUnitSI = {1.0e-6, 1.0e-6, 1.0e-6};
        f.offset = {{{1000.0}, {2000.0}, {3000.0}}};
        f.offsetUnitSI = {1.0e-6, 1.0e-6, 1.0e-6};
        fixtures.push_back(f);
    }

    // Per-record offset with fractional and negative values and independent
    // units.
    {
        Fixture f;
        f.name = "per_record_fractional_negative_offset";
        f.position = {{{0.25, 0.5}, {0.125, 0.75}, {0.9, 0.0}}};
        f.positionUnitSI = {1.0e-6, 2.0e-6, 3.0e-6};
        f.offset = {{{-1.5, 0.5}, {2.25, -0.75}, {-0.125, 4.5}}};
        f.offsetUnitSI = {2.0e-6, 3.0e-6, 4.0e-6};
        fixtures.push_back(f);
    }

    // Absent positionOffset record: the offset contributes zero.
    {
        Fixture f;
        f.name = "absent_offset";
        f.position = {{{0.25}, {0.5}, {0.75}}};
        f.positionUnitSI = {1.0e-6, 2.0e-6, 3.0e-6};
        f.hasOffset = false;
        f.offsetUnitSI = {1.0, 1.0, 1.0};
        fixtures.push_back(f);
    }

    for(auto const& f : fixtures)
        runFixture(root, f);

    fs::remove_all(root);

    if(failures != 0)
        std::cerr << failures << " of " << checks << " checks failed\n";
    else
        std::cout << "all " << checks << " checks passed\n";

    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}
