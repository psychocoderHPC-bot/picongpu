/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Host-only tests for the ported FP64 reference kernel and the anisotropic tile
 * geometry. Golden tiles are the checked-in Python-reference vectors.
 */

#include "picongpu/particles/neuralBunch/ReferenceKernel.hpp"

#include "picongpu/particles/neuralBunch/TileGeometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace nb = picongpu::particles::neuralBunch;
namespace ref = picongpu::particles::neuralBunch::reference;
namespace tg = picongpu::particles::neuralBunch::tileGeometry;

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

    struct Case
    {
        std::string name;
        double dx = 1.0, dy = 1.0, dz = 1.0;
        ref::Vec3 xTile{};
        std::vector<ref::Source> sources;
        ref::Settings settings{};
        std::vector<double> tile;
    };

    Case loadCase(fs::path const& path)
    {
        std::ifstream in(path);
        if(!in)
            throw std::runtime_error("cannot open " + path.string());
        Case value;
        std::string token;
        while(in >> token)
        {
            if(token == "name")
                in >> value.name;
            else if(token == "dx")
                in >> value.dx >> value.dy >> value.dz;
            else if(token == "x_tile")
                in >> value.xTile.x >> value.xTile.y >> value.xTile.z;
            else if(token == "nsrc")
            {
                int n = 0;
                in >> n;
                value.sources.resize(n);
                for(auto& source : value.sources)
                {
                    std::string tag;
                    in >> tag;
                    in >> source.x.x >> source.x.y >> source.x.z >> source.u.x >> source.u.y >> source.u.z >> source.w;
                }
            }
            else if(token == "order")
                in >> value.settings.order;
            else if(token == "tol")
                in >> value.settings.tol;
            else if(token == "max_level")
                in >> value.settings.max_level;
            else if(token == "max_order")
                in >> value.settings.max_order;
            else if(token == "near_order")
            {
                int ignored = 0;
                in >> ignored;
            }
            else if(token == "tile")
            {
                value.tile.resize(1536);
                for(double& entry : value.tile)
                    in >> entry;
                break;
            }
        }
        if(value.tile.size() != 1536)
            throw std::runtime_error("short tile in " + path.string());
        return value;
    }

    void goldenTiles(fs::path const& dir)
    {
        std::vector<fs::path> paths;
        for(auto const& entry : fs::directory_iterator(dir))
            if(entry.path().extension() == ".tile")
                paths.push_back(entry.path());
        std::sort(paths.begin(), paths.end());
        check(!paths.empty(), "golden testdata present");

        for(auto const& path : paths)
        {
            Case c = loadCase(path);
            ref::TileReport report;
            auto const got = ref::tile_field(c.xTile, c.sources, c.dx, c.dy, c.dz, c.settings, &report);
            double maxAbs = 0.0;
            double maxRef = 0.0;
            bool finite = true;
            for(std::size_t i = 0; i < got.size(); ++i)
            {
                finite = finite && std::isfinite(got[i]);
                maxAbs = std::max(maxAbs, std::abs(got[i] - c.tile[i]));
                maxRef = std::max(maxRef, std::abs(c.tile[i]));
            }
            double const rel = maxAbs / std::max(maxRef, 1e-300);
            double const limit = c.name.find("near") != std::string::npos ? 1e-7 : 1e-10;
            check(finite, c.name + ": finite");
            check(rel <= limit, c.name + ": golden relative error");
            if(c.name.find("near") == std::string::npos)
                check(report.all_converged, c.name + ": converged");
        }
    }

    void directAnisotropic()
    {
        // Independent direct check at one output point with unequal cells.
        ref::Vec3 const xObs{1.0, 0.5, 0.25};
        ref::Vec3 const xSrc{-7.0, 0.5, 0.125};
        ref::Vec3 const u{1.5, -0.25, 0.75};
        ref::Vec3 E, B;
        ref::shape_field(xObs, xSrc, u, 1.0, 2.0, 0.5, ref::Settings{}, E, B);
        // Recompute the same integral with a tighter settings object: both must agree.
        ref::Vec3 E2, B2;
        ref::Settings tight;
        tight.tol = 1e-10;
        tight.max_order = 20;
        ref::shape_field(xObs, xSrc, u, 1.0, 2.0, 0.5, tight, E2, B2);
        double const scale = std::max({1.0, std::abs(E.x), std::abs(E2.x), std::abs(B.z), std::abs(B2.z)});
        check(std::abs(E.x - E2.x) / scale < 1e-8, "anisotropic E self-agreement");
        check(std::abs(B.z - B2.z) / scale < 1e-8, "anisotropic B self-agreement");

        // Zero-weight and empty sources are exactly zero.
        ref::Source zero;
        zero.x = xSrc;
        zero.u = u;
        zero.w = 0.0;
        auto const zeroTile = ref::tile_field(ref::Vec3{}, {zero}, 1.0, 2.0, 0.5);
        check(
            std::all_of(zeroTile.begin(), zeroTile.end(), [](double v) { return v == 0.0; }),
            "zero weight zero tile");
        auto const emptyTile = ref::tile_field(ref::Vec3{}, {}, 1.0, 2.0, 0.5);
        check(std::all_of(emptyTile.begin(), emptyTile.end(), [](double v) { return v == 0.0; }), "empty zero tile");
    }

    void anisotropicGeometry()
    {
        tg::CellSizes const cells{1.0, 2.0, 0.5};
        check(cells.aspectY() == 2.0, "a_y = dy/dx");
        check(cells.aspectZ() == 0.5, "a_z = dz/dx");

        // Source exactly on the Ex offset of output (0,0,0): origin + (0.5,0,0)*cells.
        std::array<double, 3> const origin{0.0, 0.0, 0.0};
        std::array<double, 3> const onEx{0.5 * cells.dx, 0.0, 0.0};
        check(tg::minimumStaggeredDistanceCells(onEx, origin, cells) == 0.0, "zero distance on staggered point");

        // Brute-force the MODEL-FORMAT minimum over all 1536 staggered points and
        // compare with the factorised implementation for several anisotropic offsets.
        auto bruteForce = [&](std::array<double, 3> const& source)
        {
            double best = std::numeric_limits<double>::infinity();
            double const axisCell[3]{cells.dx, cells.dy, cells.dz};
            for(int component = 0; component < 6; ++component)
                for(int k = 0; k < tg::tileNz; ++k)
                    for(int j = 0; j < tg::tileNy; ++j)
                        for(int i = 0; i < tg::tileNx; ++i)
                        {
                            double sum2 = 0.0;
                            int const idx[3]{i, j, k};
                            for(int axis = 0; axis < 3; ++axis)
                            {
                                double const output
                                    = origin[axis]
                                      + (idx[axis] + tg::componentOffsets[component][axis]) * axisCell[axis];
                                double const diff = (source[axis] - output) / cells.dx;
                                sum2 += diff * diff;
                            }
                            best = std::min(best, std::sqrt(sum2));
                        }
            return best;
        };
        for(auto const& source : std::array<std::array<double, 3>, 4>{
                {{{0.3, 1.7, 0.1}},
                 {{0.0, 1.0 * cells.dy, 0.0}},
                 {{0.0, 0.0, 0.25 * cells.dz}},
                 {{2.5, -3.25, 4.75}}}})
        {
            double const fast = tg::minimumStaggeredDistanceCells(source, origin, cells);
            double const slow = bruteForce(source);
            check(std::abs(fast - slow) < 1.0e-12, "distance matches brute-force staggered minimum");
        }

        // Hand-computed: far along x, the nearest Ex point is at i=7, x=7.5 dx,
        // so the minimum distance is 2.5 dx cells.
        std::array<double, 3> const farX{10.0, 0.0, 0.0};
        check(
            std::abs(tg::minimumStaggeredDistanceCells(farX, origin, cells) - 2.5) < 1.0e-12,
            "anisotropic hand-computed distance");

        // Classification: no validated band disables the neural path.
        tg::DomainBounds domain;
        check(
            tg::classify(1.0, 1.0, 2.0, 0.5, 10.0, domain) == tg::PairRoute::referenceUnvalidated,
            "hybrid: no band routes to reference");
        check(
            tg::classify(1.0, 1.0, 2.0, 0.5, 10.0, domain, nullptr, true) == tg::PairRoute::neural,
            "hybrid: test-only override restores neural");
        tg::ValidatedBand band;
        band.dMinCells = 4.0;
        band.dMaxCells = 100.0;
        band.gammaMin = 1.0;
        band.gammaMax = 100.0;
        band.aYMin = 1.0;
        band.aYMax = 4.0;
        band.aZMin = 0.25;
        band.aZMax = 1.0;
        check(
            tg::classify(1.0, 10.0, 2.0, 0.5, 10.0, domain, &band) == tg::PairRoute::neural,
            "hybrid: in-band pair is neural");
        check(
            tg::classify(1.0, 10.0, 2.0, 0.5, 4.1, domain, &band) == tg::PairRoute::referenceNear,
            "hybrid: near-band margin uses reference");
        check(
            tg::classify(1.0, 10.0, 2.0, 0.5, 96.0, domain, &band) == tg::PairRoute::referenceFar,
            "hybrid: far-band margin uses reference");
        check(
            tg::classify(1.0, 101.0, 2.0, 0.5, 10.0, domain, &band) == tg::PairRoute::referenceGamma,
            "hybrid: gamma outside band uses reference");
        check(
            tg::classify(1.0, 10.0, 3.9, 0.5, 10.0, domain, &band) == tg::PairRoute::referenceAspect,
            "hybrid: aspect outside band margin uses reference");
        check(
            tg::classify(0.0, 1.0, 2.0, 0.5, 1.0, domain, &band) == tg::PairRoute::zeroWeight,
            "hybrid: zero weight skipped");
        check(tg::aspectCovered(cells, domain), "hybrid: anisotropic aspect covered");
        tg::CellSizes bad{1.0, 20.0, 1.0};
        check(!tg::aspectCovered(bad, domain), "hybrid: aspect outside model range");
    }
} // namespace

int main(int argc, char** argv)
{
    fs::path dataDir = argc > 1 ? fs::path(argv[1]) : fs::path{NB_TESTDATA_DIR};
    goldenTiles(dataDir);
    directAnisotropic();
    anisotropicGeometry();

    if(failures != 0)
    {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "referenceKernelTest: " << checks << " checks OK\n";
    return 0;
}
