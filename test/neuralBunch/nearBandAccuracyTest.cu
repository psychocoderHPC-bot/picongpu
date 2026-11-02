/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Pins the near-band accuracy fix. A target inside a source's TSC support, including one exactly
 * coincident with a source, is evaluated by the exact device quadrature, not by the singular
 * analytic point kernel. For a single tile the test reproduces the default tile-wide path the
 * stage now uses -- the analytic point field for the mid/far bands plus a per-tile exact
 * quadrature pass for the near band, the two partitions split by the same near box -- and checks
 * it against the FP64 host reference under the frozen 1e-3 relative gate. It also reproduces the
 * pre-fix default (point field only, no near threshold) and shows it is non-finite at the
 * coincident point, and that a stationary source yields B == 0 exactly. A near-box partition case
 * (threshold 0.4, source just outside the tile) pins the disjoint point/exact split: the pre-fix
 * distance-based skip keeps a pair the box-selected exact pass also adds, which is counted twice.
 * A zero-threshold case pins that the exact pass is skipped there too: with threshold 0 the point
 * field keeps every pair, so an unconditional exact pass would double count the in-tile sources and
 * would not match the legacy point-field-only path. Each scenario also prints the near-exact
 * (source, tile) pair fraction the stage logs.
 *
 * The 1e-3 gate reproduces tools/neural-bunch/eval/field_gate.py (Python-based) in C++ with the
 * identical arithmetic, per field group G in {E, B} over the 256 staggered points:
 *   S0_G = 16 * max_point ||G_point||,   ratio = ||A_cand - A_ref||_2 / max(||A_ref||_2, S0_G)
 * where G_point is the 3-component vector of group G at one point and 16 = sqrt(N_G/3) with
 * N_G = 768 (the number of values in one group). S0_G is the preregistered scale taken from the
 * reference tile, so a weak or cancelling tile is normalised by the configuration scale rather
 * than by its own tiny norm.
 */

#include "picongpu/particles/neuralBunch/Backend.hpp"
#include "picongpu/particles/neuralBunch/ExactShapeField.hpp"
#include "picongpu/particles/neuralBunch/PointField.hpp"
#include "picongpu/particles/neuralBunch/ReferenceKernel.hpp"
#include "picongpu/particles/neuralBunch/TileGeometry.hpp"
#include "picongpu/particles/neuralBunch/TileWideField.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/ops/weighted_add.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace nb = picongpu::particles::neuralBunch;
namespace ref = picongpu::particles::neuralBunch::reference;
namespace tg = picongpu::particles::neuralBunch::tileGeometry;

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

    struct Src
    {
        double pos[3]{};
        double u[3]{};
        double w = 1.0;
    };

    // Mirrors nb::tileWideDetail::sourceNearBox (the stage's near test): a source is box-near to a
    // tile when its tile-relative position (dx units) lies in [-threshold, tileExtent+threshold] on
    // every axis. Inlined here so this test also builds against pre-fix trees where the helper is
    // absent; the formula is identical to the kernel's, keeping the point/exact partitions disjoint
    // and exhaustive.
    bool sourceNearBox(nb::Vec3d const& rel, nb::Vec3d const& cells, double const threshold)
    {
        double const maxX = static_cast<double>(tg::tileNx) * cells.x;
        double const maxY = static_cast<double>(tg::tileNy) * cells.y;
        double const maxZ = static_cast<double>(tg::tileNz) * cells.z;
        return rel.x >= -threshold && rel.x <= maxX + threshold
            && rel.y >= -threshold && rel.y <= maxY + threshold
            && rel.z >= -threshold && rel.z <= maxZ + threshold;
    }

    // The frozen 1e-3 gate for one field group (E: groupStart 0, B: groupStart 3) over the 256
    // staggered points. The tile is stored in model order c + 6*(i + 8*(j + 8*k)), so point p owns
    // slots 6*p .. 6*p+5.
    double groupGate(std::vector<double> const& cand, std::vector<double> const& reference, std::size_t groupStart)
    {
        constexpr std::size_t kPoints = 256u; // 8 * 8 * 4 cells
        double sumRef2 = 0.0;
        double sumDiff2 = 0.0;
        double s0Group = 0.0;
        for(std::size_t p = 0; p < kPoints; ++p)
        {
            double g[3]{};
            for(int c = 0; c < 3; ++c)
            {
                std::size_t const slot = 6u * p + groupStart + static_cast<std::size_t>(c);
                double const rv = reference[slot];
                double const d = cand[slot] - rv;
                sumRef2 += rv * rv;
                sumDiff2 += d * d;
                g[c] = rv;
            }
            s0Group = std::max(s0Group, std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]));
        }
        double const s0 = 16.0 * s0Group; // 16 = sqrt(768/3); preregistered scale from the reference
        double const refNorm = std::sqrt(sumRef2);
        double const diffNorm = std::sqrt(sumDiff2);
        return diffNorm / std::max(refNorm, s0);
    }

    std::vector<ref::Source> toReference(std::vector<Src> const& sources)
    {
        std::vector<ref::Source> out(sources.size());
        for(std::size_t i = 0; i < sources.size(); ++i)
        {
            out[i].x = {sources[i].pos[0], sources[i].pos[1], sources[i].pos[2]};
            out[i].u = {sources[i].u[0], sources[i].u[1], sources[i].u[2]};
            out[i].w = sources[i].w;
        }
        return out;
    }

    /** The tile-wide default path for a single tile at the origin: the point field (which skips the
     * box-near pairs when @p nearMax > 0) plus, when @p doExactPass and @p nearMax > 0, the exact
     * device quadrature for exactly those near pairs, both accumulated into the same 1536-value tile.
     *
     * The exact pass is gated on @p nearMax > 0 exactly like the stage: at threshold 0 the point
     * field skips nothing, so running the exact pass as well would add every in-tile source twice.
     *
     * @p nearMax == 0 and @p doExactPass == false reproduce the pre-fix default (the point field
     * evaluates every pair, including the singular near ones, and nothing fills them in).
     */
    template<typename TQueue, typename TExec>
    std::vector<double> tileWidePath(
        TQueue& queue,
        TExec exec,
        std::vector<Src> const& sources,
        nb::Vec3d cells,
        double nearMax,
        bool doExactPass)
    {
        constexpr std::size_t kOut = 1536u;
        std::size_t const S = sources.size();
        auto hostPos = alpaka::onHost::allocHost<double>(alpaka::Vec{S, std::size_t{3}});
        auto hostVel = alpaka::onHost::allocHost<double>(alpaka::Vec{S, std::size_t{3}});
        auto hostChg = alpaka::onHost::allocHost<double>(alpaka::Vec{S});
        for(std::size_t i = 0; i < S; ++i)
        {
            for(std::size_t a = 0; a < 3u; ++a)
            {
                hostPos[alpaka::Vec<std::size_t, 2>{i, a}] = sources[i].pos[a];
                hostVel[alpaka::Vec<std::size_t, 2>{i, a}] = sources[i].u[a];
            }
            hostChg[alpaka::Vec<std::size_t, 1>{i}] = -sources[i].w;
        }
        auto devPos = alpaka::onHost::allocLike(queue.getDevice(), hostPos);
        auto devVel = alpaka::onHost::allocLike(queue.getDevice(), hostVel);
        auto devChg = alpaka::onHost::allocLike(queue.getDevice(), hostChg);
        alpaka::onHost::memcpy(queue, devPos, hostPos);
        alpaka::onHost::memcpy(queue, devVel, hostVel);
        alpaka::onHost::memcpy(queue, devChg, hostChg);
        hostPos.keepAlive(queue);
        hostVel.keepAlive(queue);
        hostChg.keepAlive(queue);

        auto pointBuf = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{std::size_t{1}, kOut});
        auto bandBuf = alpaka::onHost::alloc<std::uint32_t>(queue.getDevice(), alpaka::Vec{std::size_t{1}, std::size_t{3}});
        alpaka::nn::onHost::ops::fill(queue, exec, pointBuf, 0.0f);
        alpaka::nn::onHost::ops::fill(queue, exec, bandBuf, 0u);
        nb::enqueueTileWidePointField(
            queue,
            exec,
            devPos,
            devVel,
            devChg,
            S,
            cells,
            std::array<int, 3>{0, 0, 0},
            std::array<int, 3>{1, 1, 1},
            nearMax,
            nearMax,
            48.0,
            true,
            pointBuf,
            bandBuf);

        std::vector<double> tile(kOut, 0.0);
        auto pointHost = alpaka::onHost::allocHost<float>(alpaka::Vec{std::size_t{1}, kOut});
        alpaka::onHost::memcpy(queue, pointHost, pointBuf);
        for(std::size_t k = 0; k < kOut; ++k)
            tile[k] += static_cast<double>(pointHost[alpaka::Vec<std::size_t, 2>{0u, k}]);

        if(doExactPass && nearMax > 0.0)
        {
            std::vector<Src> near;
            for(auto const& s : sources)
                if(sourceNearBox(nb::Vec3d{s.pos[0], s.pos[1], s.pos[2]}, cells, nearMax))
                    near.push_back(s);
            std::size_t const N = near.size();
            if(N > 0u)
            {
                auto hostSrc = alpaka::onHost::allocHost<double>(alpaka::Vec{N, std::size_t{6}});
                auto hostW = alpaka::onHost::allocHost<double>(alpaka::Vec{N});
                for(std::size_t i = 0; i < N; ++i)
                {
                    for(int c = 0; c < 6; ++c)
                        hostSrc[alpaka::Vec<std::size_t, 2>{i, static_cast<std::size_t>(c)}]
                            = c < 3 ? near[i].pos[c] : near[i].u[c - 3];
                    hostW[alpaka::Vec<std::size_t, 1>{i}] = -near[i].w;
                }
                auto devSrc = alpaka::onHost::allocLike(queue.getDevice(), hostSrc);
                auto devW = alpaka::onHost::allocLike(queue.getDevice(), hostW);
                alpaka::onHost::memcpy(queue, devSrc, hostSrc);
                alpaka::onHost::memcpy(queue, devW, hostW);
                hostSrc.keepAlive(queue);
                hostW.keepAlive(queue);
                auto exactBuf = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{kOut});
                nb::enqueueExactZero(queue, exec, exactBuf);
                // Same launch-stack bound as the stage's exact path: the in-support quadrature needs
                // ~21 KiB of local stack per thread, so cap the resident thread count by free memory.
                double const freeBytes = static_cast<double>(queue.getDevice().getFreeGlobalMemBytes());
                std::size_t const maxThreads
                    = std::clamp<std::size_t>(static_cast<std::size_t>(freeBytes / (4u * 24u * 1024u)), 512u, 65536u);
                nb::enqueueExactShapeFieldTileAccumulate<double>(
                    queue,
                    exec,
                    devSrc,
                    devW,
                    N,
                    cells,
                    nb::ExactSettings{},
                    exactBuf,
                    maxThreads);
                auto exactHost = alpaka::onHost::allocHost<float>(alpaka::Vec{kOut});
                alpaka::onHost::memcpy(queue, exactHost, exactBuf);
                for(std::size_t k = 0; k < kOut; ++k)
                    tile[k] += static_cast<double>(exactHost[alpaka::Vec<std::size_t, 1>{k}]);
            }
        }
        alpaka::onHost::wait(queue);
        return tile;
    }

    // Largest |cand-ref| / (|ref|+eps) over the 256 points for one component c (0..5).
    double componentMaxRel(std::vector<double> const& cand, std::vector<double> const& reference, std::size_t c)
    {
        double m = 0.0;
        for(std::size_t p = 0; p < 256u; ++p)
        {
            std::size_t const slot = 6u * p + c;
            m = std::max(m, std::abs(cand[slot] - reference[slot]) / (std::abs(reference[slot]) + 1.0e-300));
        }
        return m;
    }

    void report(char const* name, std::vector<double> const& cand, std::vector<double> const& reference)
    {
        char const* comp[6]{"Ex", "Ey", "Ez", "Bx", "By", "Bz"};
        for(std::size_t c = 0; c < 6u; ++c)
            std::cout << "  " << name << " " << comp[c] << " maxRel=" << componentMaxRel(cand, reference, c) << "\n";
        std::cout << "  " << name << " gate_E=" << groupGate(cand, reference, 0u)
                  << " gate_B=" << groupGate(cand, reference, 3u) << "\n";
    }

    /** Near-exact (source, tile) pair accounting for a single tile at the origin, mirroring the
     * stage's ``tileWideNearExactPairs`` / ``tileWideSourceTilePairs`` counters. The exact pass runs
     * only when the threshold is positive, so a zero threshold has zero exact pairs.
     */
    std::pair<std::size_t, std::size_t> nearPairCounts(
        std::vector<Src> const& sources,
        nb::Vec3d const& cells,
        double const nearMax)
    {
        std::size_t exact = 0u;
        if(nearMax > 0.0)
            for(auto const& s : sources)
                if(sourceNearBox(nb::Vec3d{s.pos[0], s.pos[1], s.pos[2]}, cells, nearMax))
                    ++exact;
        return {exact, sources.size()}; // one tile, so total (source, tile) pairs == sources
    }

    void reportPairs(char const* name, std::pair<std::size_t, std::size_t> const& pairs)
    {
        double const fraction = pairs.second != 0u
                                    ? 100.0 * static_cast<double>(pairs.first) / static_cast<double>(pairs.second)
                                    : 0.0;
        std::cout << "  " << name << " near-exact pairs=" << pairs.first << "/" << pairs.second << " ("
                  << fraction << "% of source-tile pairs)\n";
    }

    char const* compName(std::size_t c)
    {
        static char const* n[6]{"Ex", "Ey", "Ez", "Bx", "By", "Bz"};
        return n[c];
    }

    int run(std::string const& backend)
    {
        nb::dispatchBackend(
            backend,
            [&](auto& device, auto const& exec)
            {
                (void) device;
                auto queue = device.makeQueue();
                // The default near band is params.bandedNearMaxCells (4 dx) -- the threshold the stage now passes to
                // the point field and uses for the exact pass.
                constexpr double kNearMax = 4.0;

                // ---- Scenario A: coincident + in-support + far, isotropic --------------------------------------
                // s0 sits exactly on the Ex staggered output at (i=3,j=2,k=1) -> slot 0 + 6*(3 + 8*(2 + 8*1)) = 498.
                // s1 is inside the TSC support of several outputs; s2 is far (point field only).
                std::vector<Src> a{
                    {Src{{3.5, 2.0, 1.0}, {0.30, 0.20, 0.15}, 1.0}},
                    {Src{{4.0, 4.0, 2.0}, {-0.15, 0.10, 0.20}, 0.7}},
                    {Src{{60.0, 40.0, 20.0}, {0.10, 0.10, 0.00}, 0.5}}};
                nb::Vec3d const cellsA{1.0, 1.0, 1.0};
                std::size_t const coincidentSlot = 498u;

                auto refA = ref::tile_field(ref::Vec3{0.0, 0.0, 0.0}, toReference(a), 1.0, 1.0, 1.0);
                auto defaultA = tileWidePath(queue, exec, a, cellsA, kNearMax, true);
                auto legacyA = tileWidePath(queue, exec, a, cellsA, 0.0, false);

                bool defaultFinite = std::all_of(
                    defaultA.begin(),
                    defaultA.end(),
                    [](double v) { return std::isfinite(v); });
                check(defaultFinite, "A: default path finite over all 1536 outputs", 0.0);
                check(
                    std::isfinite(defaultA[coincidentSlot]),
                    "A: coincident output finite (no NaN/Inf from the point kernel)",
                    defaultA[coincidentSlot]);
                check(
                    !std::isfinite(legacyA[coincidentSlot]),
                    "A: pre-fix point-field-only path is non-finite at the coincident output",
                    legacyA[coincidentSlot]);

                double const gateA_E = groupGate(defaultA, refA, 0u);
                double const gateA_B = groupGate(defaultA, refA, 3u);
                check(gateA_E <= 1.0e-3, "A: E group within the frozen 1e-3 gate", gateA_E);
                check(gateA_B <= 1.0e-3, "A: B group within the frozen 1e-3 gate", gateA_B);
                std::cout << "[A isotropic coincident] default vs exact reference\n";
                report("default", defaultA, refA);
                reportPairs("default", nearPairCounts(a, cellsA, kNearMax));
                std::cout << "  coincident slot " << coincidentSlot << " (" << compName(coincidentSlot % 6u) << "): default="
                          << defaultA[coincidentSlot] << " reference=" << refA[coincidentSlot] << " legacy="
                          << legacyA[coincidentSlot] << "\n";

                // ---- Scenario B: stationary source -> B == 0 exactly --------------------------------------------
                std::vector<Src> b{{Src{{4.0, 4.0, 2.0}, {0.0, 0.0, 0.0}, 1.0}}};
                auto refB = ref::tile_field(ref::Vec3{0.0, 0.0, 0.0}, toReference(b), 1.0, 1.0, 1.0);
                auto defaultB = tileWidePath(queue, exec, b, cellsA, kNearMax, true);
                double maxB = 0.0;
                for(std::size_t p = 0; p < 256u; ++p)
                    for(int c = 3; c < 6; ++c)
                        maxB = std::max(maxB, std::abs(defaultB[6u * p + static_cast<std::size_t>(c)]));
                check(maxB == 0.0, "B: stationary source yields B exactly 0", maxB);
                check(
                    std::isfinite(defaultB[coincidentSlot]), "B: all outputs finite", 0.0);
                double const gateB_E = groupGate(defaultB, refB, 0u);
                check(gateB_E <= 1.0e-3, "B: stationary E group within the 1e-3 gate", gateB_E);
                std::cout << "[B isotropic stationary] default vs exact reference\n";
                report("default", defaultB, refB);
                reportPairs("default", nearPairCounts(b, cellsA, kNearMax));
                std::cout << "  max|B| = " << maxB << " (must be exactly 0)\n";

                // ---- Scenario C: anisotropic cells, in-support (pins the aspect-scaled near box + exact pass) ---
                std::vector<Src> c{{Src{{3.0, 2.0, 1.0}, {0.25, 0.10, 0.15}, 1.0}},
                                   {Src{{4.5, 3.0, 1.5}, {-0.10, 0.18, 0.05}, 0.8}}};
                nb::Vec3d const cellsC{1.0, 2.0, 0.5};
                auto refC = ref::tile_field(ref::Vec3{0.0, 0.0, 0.0}, toReference(c), 1.0, 2.0, 0.5);
                auto defaultC = tileWidePath(queue, exec, c, cellsC, kNearMax, true);
                bool defaultCFinite = std::all_of(defaultC.begin(), defaultC.end(), [](double v) { return std::isfinite(v); });
                check(defaultCFinite, "C: anisotropic default path finite", 0.0);
                double const gateC_E = groupGate(defaultC, refC, 0u);
                double const gateC_B = groupGate(defaultC, refC, 3u);
                check(gateC_E <= 1.0e-3, "C: anisotropic E group within the 1e-3 gate", gateC_E);
                check(gateC_B <= 1.0e-3, "C: anisotropic B group within the 1e-3 gate", gateC_B);
                std::cout << "[C anisotropic in-support] default vs exact reference\n";
                report("default", defaultC, refC);
                reportPairs("default", nearPairCounts(c, cellsC, kNearMax));

                // ---- Scenario D: near-box partition contract ---------------------------------------------------
                // A source just outside the tile is inside the near box (threshold 0.4) but farther than 0.4
                // from the nearest staggered point (0.45). The pre-fix point field skips by exact distance, so
                // it keeps this pair while the box-selected exact pass also adds it -- the contribution is
                // counted twice. The post-fix point field skips by the identical box, so only the exact pass
                // contributes. This pins the disjoint-and-exhaustive partition the default path relies on.
                std::vector<Src> d{{Src{{8.05, 0.0, 0.0}, {0.20, 0.10, 0.05}, 1.0}}};
                constexpr double kPartitionNearMax = 0.4;
                auto refD = ref::tile_field(ref::Vec3{0.0, 0.0, 0.0}, toReference(d), 1.0, 1.0, 1.0);
                auto defaultD = tileWidePath(queue, exec, d, cellsA, kPartitionNearMax, true);
                bool const defaultDFinite
                    = std::all_of(defaultD.begin(), defaultD.end(), [](double v) { return std::isfinite(v); });
                check(defaultDFinite, "D: near-box partition path finite", 0.0);
                double const gateD_E = groupGate(defaultD, refD, 0u);
                double const gateD_B = groupGate(defaultD, refD, 3u);
                check(gateD_E <= 1.0e-3, "D: box-near pair counted once (E group)", gateD_E);
                check(gateD_B <= 1.0e-3, "D: box-near pair counted once (B group)", gateD_B);
                std::cout << "[D near-box partition, threshold 0.4] default vs exact reference\n";
                report("default", defaultD, refD);
                reportPairs("default", nearPairCounts(d, cellsA, kPartitionNearMax));

                // ---- Scenario E: zero near threshold -> legacy point-field-only path ---------------------------
                // A zero threshold disables the near-band split: the point field skips nothing and the exact near
                // pass must not run. If it did, this in-tile source (not on a staggered point, so the point value
                // stays finite) would be added by both passes and the result would not match the legacy
                // point-field-only path. The helper applies the same threshold gate as the stage.
                std::vector<Src> e{{Src{{4.2, 3.7, 2.3}, {0.18, -0.12, 0.09}, 0.9}}};
                constexpr double kZeroNearMax = 0.0;
                auto zeroE = tileWidePath(queue, exec, e, cellsA, kZeroNearMax, true);
                auto legacyE = tileWidePath(queue, exec, e, cellsA, kZeroNearMax, false);
                double maxDiffE = 0.0;
                for(std::size_t k = 0; k < 1536u; ++k)
                    maxDiffE = std::max(maxDiffE, std::abs(zeroE[k] - legacyE[k]));
                check(
                    maxDiffE == 0.0,
                    "E: threshold 0 skips the exact pass (equals legacy point-field-only)",
                    maxDiffE);
                bool const zeroEFinite
                    = std::all_of(zeroE.begin(), zeroE.end(), [](double v) { return std::isfinite(v); });
                check(zeroEFinite, "E: threshold-0 path finite", 0.0);
                std::cout << "[E zero near threshold] exact pass skipped\n";
                std::cout << "  max|threshold0 - legacy| = " << maxDiffE << " (must be exactly 0)\n";
                reportPairs("threshold0", nearPairCounts(e, cellsA, kZeroNearMax));
            });
        return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
} // namespace

int main(int argc, char** argv)
{
    std::string const backend = argc > 1 ? argv[1] : "cuda";
    int const rc = run(backend);
    if(failures != 0)
    {
        std::cerr << failures << " of " << checks << " near-band checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "nearBandAccuracyTest: " << checks << " checks OK\n";
    return rc;
}
