/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * T150 near-field table path. The test builds a small table on the host by sampling the
 * exact quadrature at the grid nodes, uploads it to the device, and checks the
 * parameter-gated path (settings.useNearTable, default off):
 *
 *  - default off and inactive table: bit-identical to the quadrature path;
 *  - out of domain (|r| beyond rmax, velocity mismatch, aspect mismatch): bit-identical
 *    fallback to the quadrature path;
 *  - in-support on-grid node: the lookup reproduces the stored value;
 *  - in-support off-grid: the trilinear interpolation stays within tolerance of the
 *    quadrature, and the table values are odd in r;
 *  - device tile pass with the table: in-support in-domain slots equal the host trilinear
 *    value of the stored corners (the quadrature cannot reproduce it to 1e-9, which proves
 *    the device path evaluates the table), out-of-support slots are bit-identical to the
 *    quadrature run, and the enlarged kernel argument launches (a local-table regression of
 *    the 20,280 B kind would break the launch).
 */

#include "picongpu/particles/neuralBunch/Backend.hpp"
#include "picongpu/particles/neuralBunch/ExactShapeField.hpp"
#include "picongpu/particles/neuralBunch/NearFieldTable.hpp"

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace nb = picongpu::particles::neuralBunch;

namespace
{
    int failures = 0;
    int checks = 0;

    void check(bool const ok, std::string const& what, double detail = 0.0)
    {
        ++checks;
        if(!ok)
        {
            ++failures;
            std::cerr << "FAIL: " << what << " (" << detail << ")\n";
        }
    }

    double maxRelative6(double const a[6], double const b[6])
    {
        double maxDiff = 0.0;
        double scale = 1.0;
        for(int i = 0; i < 6; ++i)
        {
            maxDiff = std::max(maxDiff, std::abs(a[i] - b[i]));
            scale = std::max(scale, std::abs(b[i]));
        }
        return maxDiff / scale;
    }

    nb::Vec3d scaled(nb::Vec3d const& v, double const s)
    {
        return {v.x * s, v.y * s, v.z * s};
    }

    //! Symmetric grid -rmax .. rmax with spacing h (rmax / h integral).
    std::vector<double> makeGrid(double const h, double const rmax)
    {
        std::size_t const n = static_cast<std::size_t>(std::llround(rmax / h)) * 2u + 1u;
        std::vector<double> grid(n);
        for(std::size_t i = 0u; i < n; ++i)
            grid[i] = (static_cast<double>(i) - static_cast<double>(n / 2u)) * h;
        return grid;
    }

    //! Sample the exact quadrature at every grid node for (u, cells); returns the AoS
    //! x-fastest values [n, n, n, 6] in the quadrature (dx-normalized) convention.
    std::vector<double> sampleTable(
        std::vector<double> const& grid,
        nb::Vec3d const& u,
        nb::Vec3d const& cells,
        nb::ExactSettings const& settings)
    {
        int const n = static_cast<int>(grid.size());
        std::vector<double> out(static_cast<std::size_t>(n) * n * n * 6u);
        nb::GaussTable const gauss = nb::GaussTable::create();
        for(int ix = 0; ix < n; ++ix)
            for(int iy = 0; iy < n; ++iy)
                for(int iz = 0; iz < n; ++iz)
                {
                    nb::Vec3d const R0{grid[ix] * cells.x, grid[iy] * cells.y, grid[iz] * cells.z};
                    auto const result = nb::exactShapeFieldPoint<double>(R0, u, cells, settings, gauss);
                    double const v[6]{result.E.x, result.E.y, result.E.z, result.B.x, result.B.y, result.B.z};
                    std::size_t const off = (static_cast<std::size_t>(ix) * n + iy) * n + iz;
                    for(int c = 0; c < 6; ++c)
                        out[off * 6u + c] = v[c];
                }
        return out;
    }

    nb::NearFieldTable hostTableFrom(std::vector<float> const& data, std::vector<double> const& grid, double rmax)
    {
        nb::NearFieldTable table;
        table.data = data.empty() ? nullptr : data.data();
        int const n = static_cast<int>(grid.size());
        for(int i = 0; i < n; ++i)
            table.grid[i] = grid[i];
        table.n = n;
        table.rmax = rmax;
        return table;
    }

    double quadratureAt(nb::Vec3d const& R0, nb::Vec3d const& u, nb::Vec3d const& cells, double out[6])
    {
        nb::GaussTable const gauss = nb::GaussTable::create();
        auto const result = nb::exactShapeFieldPoint<double>(R0, u, cells, nb::ExactSettings{}, gauss);
        out[0] = result.E.x;
        out[1] = result.E.y;
        out[2] = result.E.z;
        out[3] = result.B.x;
        out[4] = result.B.y;
        out[5] = result.B.z;
        return result.error;
    }

    //! Single-source tile through the exact near pass, mirroring the benchmark's runtime path.
    template<typename TQueue, typename TExec>
    std::vector<double> tilePass(
        TQueue& queue,
        TExec exec,
        nb::Source const& source,
        nb::Vec3d const& cells,
        nb::ExactSettings const& settings)
    {
        auto hostSources = alpaka::onHost::allocHost<double>(alpaka::Vec<std::size_t, 2>{1u, 6u});
        auto hostWeights = alpaka::onHost::allocHost<double>(alpaka::Vec<std::size_t, 1>{1u});
        for(int c = 0; c < 3; ++c)
        {
            hostSources[alpaka::Vec<std::size_t, 2>{0u, static_cast<std::size_t>(c)}] = source.r[c];
            hostSources[alpaka::Vec<std::size_t, 2>{0u, static_cast<std::size_t>(c + 3)}] = source.u[c];
        }
        hostWeights[alpaka::Vec<std::size_t, 1>{0u}] = -source.weight;
        auto deviceSources = alpaka::onHost::allocLike(queue.getDevice(), hostSources);
        auto deviceWeights = alpaka::onHost::allocLike(queue.getDevice(), hostWeights);
        alpaka::onHost::memcpy(queue, deviceSources, hostSources);
        alpaka::onHost::memcpy(queue, deviceWeights, hostWeights);
        hostSources.keepAlive(queue);
        auto accumulator = alpaka::onHost::alloc<double>(queue.getDevice(), alpaka::Vec{nb::kOutDim});
        nb::enqueueExactZero(queue, exec, accumulator);
        nb::enqueueExactShapeFieldTileAccumulate<double>(
            queue,
            exec,
            deviceSources,
            deviceWeights,
            1u,
            cells,
            settings,
            accumulator);
        auto hostAcc = alpaka::onHost::allocHost<double>(alpaka::Vec{nb::kOutDim});
        alpaka::onHost::memcpy(queue, hostAcc, accumulator);
        alpaka::onHost::wait(queue);
        std::vector<double> values(nb::kOutDim);
        for(std::size_t k = 0u; k < nb::kOutDim; ++k)
            values[k] = hostAcc[alpaka::Vec<std::size_t, 1>{k}];
        return values;
    }

    bool runHostChecks(std::vector<double> const& grid, nb::Vec3d const& u, nb::Vec3d const& cells)
    {
        std::vector<float> f32(static_cast<std::size_t>(grid.size()) * grid.size() * grid.size() * 6u);
        auto const f64 = sampleTable(grid, u, cells, nb::ExactSettings{});
        for(std::size_t i = 0u; i < f64.size(); ++i)
            f32[i] = static_cast<float>(f64[i]);
        int const n = static_cast<int>(grid.size());
        double const rmax = grid.back();
        nb::NearFieldTable const table = hostTableFrom(f32, grid, rmax);

        // covers(): in-domain true; |r| beyond rmax, u mismatch, cells mismatch, inactive false.
        nb::Vec3d const rIn{1.0, -0.5, 0.5};
        check(nb::nearTableCovers(table, scaled(rIn, cells.x), u, cells), "covers in-domain", 0.0);
        nb::Vec3d const rOut{rmax + 0.1, 0.0, 0.0};
        check(!nb::nearTableCovers(table, scaled(rOut, cells.x), u, cells), "covers rejects beyond rmax", 0.0);
        nb::Vec3d const uBad{u.x + 1e-3, u.y, u.z};
        check(!nb::nearTableCovers(table, scaled(rIn, cells.x), uBad, cells), "covers rejects u mismatch", 0.0);
        nb::Vec3d const cellsBad{cells.x, cells.y * 1.001, cells.z};
        check(!nb::nearTableCovers(table, scaled(rIn, cells.x), u, cellsBad), "covers rejects cells mismatch", 0.0);
        nb::NearFieldTable const inactive = hostTableFrom(std::vector<float>{}, grid, rmax);
        check(!nb::nearTableCovers(inactive, scaled(rIn, cells.x), u, cells), "covers rejects inactive table", 0.0);

        // On-grid node: the lookup reproduces the stored value; the quadrature at the node
        // produced the stored value (f32 rounding).
        nb::Vec3d const rNode{0.75, 0.75, 0.0};
        double stored[6]{};
        double quad[6]{};
        quadratureAt(scaled(rNode, cells.x), u, cells, quad);
        double lookup[6]{};
        check(nb::nearTableLookup(table, rNode, lookup), "lookup on-grid succeeds", 0.0);
        auto const indexOf = [&](double v)
        {
            for(int i = 0; i < n; ++i)
                if(std::abs(grid[i] - v) < 1e-12)
                    return i;
            return -1;
        };
        int const ix = indexOf(0.75);
        int const iy = indexOf(0.75);
        int const iz = indexOf(0.0);
        check(ix >= 0 && iy >= 0 && iz >= 0, "grid carries the test node", 0.0);
        for(int c = 0; c < 6; ++c)
            stored[c] = static_cast<double>(f32[((std::size_t(ix) * n + iy) * n + iz) * 6u + c]);
        double maxStoredDiff = 0.0;
        for(int c = 0; c < 6; ++c)
            maxStoredDiff = std::max(maxStoredDiff, std::abs(lookup[c] - stored[c]) / (1.0 + std::abs(stored[c])));
        check(maxStoredDiff <= 1e-5, "lookup on-grid equals stored value", maxStoredDiff);
        double maxNodeQuad = 0.0;
        for(int c = 0; c < 6; ++c)
            maxNodeQuad = std::max(maxNodeQuad, std::abs(stored[c] - quad[c]) / (1.0 + std::abs(quad[c])));
        check(maxNodeQuad <= 1e-5, "stored node equals quadrature at the node", maxNodeQuad);

        // Off-grid in-support point: trilinear interpolation stays close to the quadrature.
        nb::Vec3d const rOff{0.4, -0.3, 0.6};
        double quadOff[6]{};
        double lookupOff[6]{};
        quadratureAt(scaled(rOff, cells.x), u, cells, quadOff);
        check(nb::nearTableLookup(table, rOff, lookupOff), "lookup off-grid succeeds", 0.0);
        check(maxRelative6(lookupOff, quadOff) <= 2e-2, "lookup off-grid within interpolation tolerance", maxRelative6(lookupOff, quadOff));
        double lookupOdd[6]{};
        nb::Vec3d const rNodeNeg{-0.75, -0.75, 0.0};
        check(nb::nearTableLookup(table, rNodeNeg, lookupOdd), "lookup at mirrored node succeeds", 0.0);
        double oddness = 0.0;
        for(int c = 0; c < 6; ++c)
            oddness = std::max(oddness, std::abs(lookup[c] + lookupOdd[c]) / (1.0 + std::abs(lookup[c])));
        check(oddness <= 1e-4, "table values are odd in r", oddness);

        // Default off / inactive table / out-of-domain: bit-identical to the quadrature path.
        auto const pointWith = [&](bool const useTable, nb::NearFieldTable const& t, nb::Vec3d const& uq)
        {
            nb::ExactSettings settings;
            settings.useNearTable = useTable;
            settings.nearTable = t;
            nb::GaussTable const gauss = nb::GaussTable::create();
            auto const result = nb::exactShapeFieldPoint<double>(scaled(rOff, cells.x), uq, cells, settings, gauss);
            return std::array<double, 6>{result.E.x, result.E.y, result.E.z, result.B.x, result.B.y, result.B.z};
        };
        auto const defaultOff = pointWith(false, table, u);
        auto const inactiveOn = pointWith(true, inactive, u);
        auto const uMismatch = pointWith(true, table, uBad);
        auto const farPoint = [&, table]
        {
            nb::Vec3d const rFar{rmax * 2.0, 0.0, 0.0};
            nb::ExactSettings settings;
            settings.useNearTable = true;
            settings.nearTable = table;
            nb::GaussTable const gauss = nb::GaussTable::create();
            auto const result = nb::exactShapeFieldPoint<double>(scaled(rFar, cells.x), u, cells, settings, gauss);
            return std::array<double, 6>{result.E.x, result.E.y, result.E.z, result.B.x, result.B.y, result.B.z};
        }();
        auto const bitIdentical = [](std::array<double, 6> const& a, std::array<double, 6> const& b)
        {
            for(int c = 0; c < 6; ++c)
                if(a[c] != b[c])
                    return false;
            return true;
        };
        check(bitIdentical(defaultOff, inactiveOn), "inactive table is bit-identical to the quadrature", 0.0);
        check(bitIdentical(defaultOff, uMismatch), "u mismatch falls back bit-identically", 0.0);
        check(bitIdentical(defaultOff, farPoint), "beyond rmax falls back bit-identically", 0.0);
        return true;
    }

    void runDeviceChecks(std::string const& backend, std::vector<double> const& grid, nb::Vec3d const& u, nb::Vec3d const& cells)
    {
        int const n = static_cast<int>(grid.size());
        double const rmax = grid.back();
        std::vector<float> f32(static_cast<std::size_t>(n) * n * n * 6u);
        auto const f64 = sampleTable(grid, u, cells, nb::ExactSettings{});
        for(std::size_t i = 0u; i < f64.size(); ++i)
            f32[i] = static_cast<float>(f64[i]);
        nb::NearFieldTable const hostRefTable = hostTableFrom(f32, grid, rmax);

        nb::dispatchBackend(
            backend,
            [&](auto& device, auto const& exec)
            {
                auto queue = device.makeQueue();
                auto hostData = alpaka::onHost::allocHost<float>(alpaka::Vec{f32.size()});
                for(std::size_t i = 0u; i < f32.size(); ++i)
                    hostData[alpaka::Vec<std::size_t, 1>{i}] = f32[i];
                nb::NearFieldTable const devTable = nb::uploadNearFieldTable(queue, hostData, grid, u, cells, 1.0);
                double const dx2 = cells.x * cells.x;

                nb::Source source;
                source.r[0] = 4.0;
                source.r[1] = 4.0;
                source.r[2] = 2.0; // tile center of the 8 x 8 x 4 tile
                source.u[0] = u.x;
                source.u[1] = u.y;
                source.u[2] = u.z;

                nb::ExactSettings settingsOff;
                nb::ExactSettings settingsOn;
                settingsOn.useNearTable = true;
                settingsOn.nearTable = devTable;
                // The mixed run: the same data, a smaller rmax, so face slots fall back.
                nb::NearFieldTable devTableSmall = devTable;
                devTableSmall.rmax = 1.0;
                nb::ExactSettings settingsMixed;
                settingsMixed.useNearTable = true;
                settingsMixed.nearTable = devTableSmall;

                auto const valuesOff = tilePass(queue, exec, source, cells, settingsOff);
                auto const valuesOn = tilePass(queue, exec, source, cells, settingsOn);
                auto const valuesMixed = tilePass(queue, exec, source, cells, settingsMixed);
                check(std::all_of(valuesOff.begin(), valuesOff.end(), [](double v) { return std::isfinite(v); }), "quadrature tile finite", 0.0);
                check(std::all_of(valuesOn.begin(), valuesOn.end(), [](double v) { return std::isfinite(v); }), "table tile finite", 0.0);

                int tableSlots = 0;
                int fallbackSlots = 0;
                int offGridTableSlots = 0;
                double maxTableSlotDiff = 0.0; // vs host trilinear of the stored corners
                double maxInterpDiff = 0.0; // table vs quadrature at in-support slots
                double maxFallbackDiff = 0.0; // out-of-support slots: table run vs quadrature run
                double maxDiscriminatingDiff = 0.0; // off-grid in-support slots: table vs quadrature
                for(std::size_t slot = 0u; slot < nb::kOutDim; ++slot)
                {
                    nb::Vec3d const pos = nb::staggeredPosition(slot, cells);
                    nb::Vec3d const R0{pos.x - source.r[0], pos.y - source.r[1], pos.z - source.r[2]};
                    nb::Vec3d const rCells{R0.x / cells.x, R0.y / cells.y, R0.z / cells.z};
                    bool inSupport = true;
                    bool inDomain = true;
                    bool inDomainMixed = true;
                    for(int a = 0; a < 3; ++a)
                    {
                        double const rc = (a == 0) ? rCells.x : ((a == 1) ? rCells.y : rCells.z);
                        if(std::abs(rc) > 1.5 + 1e-12)
                            inSupport = false;
                        if(std::abs(rc) > rmax + 1e-12)
                            inDomain = false;
                        if(std::abs(rc) > 1.0 + 1e-12)
                            inDomainMixed = false;
                    }
                    if(inSupport && inDomain)
                    {
                        ++tableSlots;
                        double hostValue[6]{};
                        bool const ok = nb::nearTableLookup(hostRefTable, rCells, hostValue);
                        check(ok, "host lookup for in-support slot", 0.0);
                        double const expected = (slot % 6u) < 3u ? hostValue[slot % 6u] * dx2 : hostValue[3u + slot % 6u] * dx2;
                        double const d = std::abs(valuesOn[slot] - expected) / (1.0 + std::abs(expected));
                        maxTableSlotDiff = std::max(maxTableSlotDiff, d);
                        maxInterpDiff = std::max(maxInterpDiff, std::abs(valuesOn[slot] - valuesOff[slot]) / (1.0 + std::abs(valuesOff[slot])));
                        // Off-grid slot: the trilinear value of the stored corners is not the
                        // quadrature value; a difference beyond 1e-9 proves the table ran.
                        bool const onGrid = [rCells]
                        {
                            double const rc[3]{rCells.x, rCells.y, rCells.z};
                            for(int a = 0; a < 3; ++a)
                            {
                                double const frac = std::remainder(rc[a], 0.75);
                                if(std::abs(frac) > 1e-12 && std::abs(frac - 0.75) > 1e-12)
                                    return false;
                            }
                            return true;
                        }();
                        if(!onGrid)
                        {
                            ++offGridTableSlots;
                            maxDiscriminatingDiff = std::max(maxDiscriminatingDiff, std::abs(valuesOn[slot] - valuesOff[slot]) / (1.0 + std::abs(valuesOff[slot])));
                        }
                    }
                    else if(inSupport) // in-support but beyond rmax: mixed run must fall back
                    {
                        ++fallbackSlots;
                        double const d = std::abs(valuesMixed[slot] - valuesOff[slot]) / (1.0 + std::abs(valuesOff[slot]));
                        maxFallbackDiff = std::max(maxFallbackDiff, d);
                    }
                    else // out of support: every table-gated run is the quadrature path
                    {
                        ++fallbackSlots;
                        double const dOn = std::abs(valuesOn[slot] - valuesOff[slot]) / (1.0 + std::abs(valuesOff[slot]));
                        double const dMixed = std::abs(valuesMixed[slot] - valuesOff[slot]) / (1.0 + std::abs(valuesOff[slot]));
                        maxFallbackDiff = std::max(maxFallbackDiff, dOn);
                        maxFallbackDiff = std::max(maxFallbackDiff, dMixed);
                    }
                }
                check(tableSlots > 0, "test geometry exercises in-support slots", static_cast<double>(tableSlots));
                check(offGridTableSlots > 0, "test geometry exercises off-grid in-support slots", static_cast<double>(offGridTableSlots));
                check(maxTableSlotDiff <= 1e-6, "device table slots equal the host trilinear value", maxTableSlotDiff);
                check(maxInterpDiff <= 2e-2, "device table slots within interpolation tolerance of the quadrature", maxInterpDiff);
                check(maxDiscriminatingDiff > 1e-9, "device path evaluates the table (off-grid slots differ from the quadrature)", maxDiscriminatingDiff);
                check(maxFallbackDiff == 0.0, "out-of-support and out-of-domain slots are bit-identical to the quadrature", maxFallbackDiff);
                std::cout << "nearTable device: slots=" << valuesOn.size() << " tableSlots=" << tableSlots
                          << " offGridTableSlots=" << offGridTableSlots << " fallbackSlots=" << fallbackSlots
                          << " maxTableSlotDiff=" << maxTableSlotDiff << " maxInterpDiff=" << maxInterpDiff
                          << " maxDiscriminatingDiff=" << maxDiscriminatingDiff << "\n";
            });
    }
} // namespace

int main(int argc, char** argv)
{
    std::string const backend = argc > 1 ? argv[1] : "cuda";

    // Table family of the production runs: gamma 10, oblique direction, iso cells.
    nb::Vec3d const cells{1.0, 1.0, 1.0};
    double const dir[3]{0.6, 0.5, 0.62};
    double const dirNorm = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    double const speed = std::sqrt(10.0 * 10.0 - 1.0);
    nb::Vec3d const u{speed * dir[0] / dirNorm, speed * dir[1] / dirNorm, speed * dir[2] / dirNorm};

    std::vector<double> const grid = makeGrid(0.75, 2.25); // n = 7, all nodes in/out support
    if(!runHostChecks(grid, u, cells))
        std::cerr << "nearTable host checks failed\n";
    runDeviceChecks(backend, grid, u, cells);

    std::cout << "nearTableTest: " << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
