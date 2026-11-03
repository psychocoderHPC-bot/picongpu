/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Analytic point-charge field at the 1536 staggered tile outputs.
 *
 * For one source and one tile the normalized field is the per-unit-(q/e) point field of a uniformly moving
 * charge evaluated at every Yee-staggered output, in dx units:
 *
 *   E_n = 4*pi*eps0*dx^2 * E/q,   B_n = 4*pi*eps0*c*dx^2 * B/q.
 *
 * Positions are ``x_tile + (i + off_c.x, (j + off_c.y)*a_y, (k + off_c.z)*a_z)`` in dx units, with the six Yee
 * offsets ``Ex(.5,0,0) Ey(0,.5,0) Ez(0,0,.5) Bx(0,.5,.5) By(.5,0,.5) Bz(.5,.5,0)``.  Output slot order is the
 * MODEL-FORMAT flatten order ``c + 6*(i + 8*(j + 8*k))``.  Charge weighting (``q/e = -w``) and the ``dx^2``
 * normalization are the caller's responsibility; this header returns the per-unit-charge, dx-normalized value.
 *
 * The kernel maps one output slot per thread (1536 slots per source), so scratch is bounded by one float.  The
 * host function uses the same ``pointFieldDx`` as the device path.  The stable decomposition (1/gamma^2 direct,
 * parallel/perp split, explicit static branch) is the one in reference/convkernel.py and Features.hpp.
 */

#include "Features.hpp"

#include <alpaka/alpaka.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace picongpu::particles::neuralBunch
{
    inline constexpr std::size_t kTileNxPoint = 8;
    inline constexpr std::size_t kTileNyPoint = 8;
    inline constexpr std::size_t kTileNzPoint = 4;
    inline constexpr std::size_t kComponentsPoint = 6;
    inline constexpr std::size_t kTileSlotCount = kTileNxPoint * kTileNyPoint * kTileNzPoint * kComponentsPoint;

    /** Yee offset of component @p component in cell units. */
    ALPAKA_FN_HOST_ACC inline Vec3d yeeOffset(std::size_t component)
    {
        switch(component % 6u)
        {
        case 0u:
            return {0.5, 0.0, 0.0};
        case 1u:
            return {0.0, 0.5, 0.0};
        case 2u:
            return {0.0, 0.0, 0.5};
        case 3u:
            return {0.0, 0.5, 0.5};
        case 4u:
            return {0.5, 0.0, 0.5};
        default:
            return {0.5, 0.5, 0.0};
        }
    }

    /** Position of output slot @p slot in dx units, relative to the tile cell origin.
     *
     * @p cells is (1, a_y, a_z) for a tile with dx as length unit.
     */
    ALPAKA_FN_HOST_ACC inline Vec3d staggeredPosition(std::size_t slot, Vec3d cells)
    {
        auto const component = slot % 6u;
        auto const cell = slot / 6u;
        auto const i = cell % 8u;
        auto const j = (cell / 8u) % 8u;
        auto const k = cell / 64u;
        auto const offset = yeeOffset(component);
        return {
            (static_cast<double>(i) + offset.x) * cells.x,
            (static_cast<double>(j) + offset.y) * cells.y,
            (static_cast<double>(k) + offset.z) * cells.z};
    }

    /** One normalized point-field component at one staggered slot. */
    ALPAKA_FN_HOST_ACC inline float pointFieldSlot(
        std::size_t slot,
        Vec3d const& sourceR,
        Vec3d const& u,
        Vec3d const& cells)
    {
        auto const position = staggeredPosition(slot, cells);
        auto const field
            = pointFieldDx(Vec3d{position.x - sourceR.x, position.y - sourceR.y, position.z - sourceR.z}, u);
        auto const component = slot % 6u;
        if(component < 3u)
            return static_cast<float>(component == 0u ? field.E.x : (component == 1u ? field.E.y : field.E.z));
        return static_cast<float>(component == 3u ? field.B.x : (component == 4u ? field.B.y : field.B.z));
    }

    /** Host reference: fill 1536 values in flatten order. */
    inline void pointFieldTileHost(float* out, Vec3d const& sourceR, Vec3d const& u, Vec3d const& cells)
    {
        for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
            out[slot] = pointFieldSlot(slot, sourceR, u, cells);
    }

    inline std::vector<float> pointFieldTileHost(Vec3d const& sourceR, Vec3d const& u, Vec3d const& cells)
    {
        std::vector<float> out(kTileSlotCount, 0.0f);
        pointFieldTileHost(out.data(), sourceR, u, cells);
        return out;
    }

    namespace detail
    {
        struct PointFieldTileKernel
        {
            Vec3d sourceR{};
            Vec3d u{};
            Vec3d cells{};

            ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
            {
                for(auto index : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{kTileSlotCount}}))
                {
                    auto const slot = static_cast<std::size_t>(index[0]);
                    out[alpaka::Vec<std::size_t, 1>{slot}] = pointFieldSlot(slot, sourceR, u, cells);
                }
            }
        };

        struct PointFieldBatchedKernel
        {
            Vec3d cells{};

            ALPAKA_FN_ACC void operator()(auto const& acc, auto sources, auto out) const
            {
                auto const sourceCount = static_cast<std::size_t>(sources.getExtents()[0]);
                auto const total = sourceCount * kTileSlotCount;
                for(auto index : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{total}}))
                {
                    auto const flat = static_cast<std::size_t>(index[0]);
                    auto const sourceIndex = flat / kTileSlotCount;
                    auto const slot = flat % kTileSlotCount;
                    auto value = [&](std::size_t column)
                    { return sources[alpaka::Vec<std::size_t, 2>{sourceIndex, column}]; };
                    auto const sourceR = Vec3d{value(0u), value(1u), value(2u)};
                    auto const u = Vec3d{value(3u), value(4u), value(5u)};
                    out[alpaka::Vec<std::size_t, 2>{sourceIndex, slot}] = pointFieldSlot(slot, sourceR, u, cells);
                }
            }
        };
    } // namespace detail

    /** Enqueue one source evaluated at all 1536 outputs into @p deviceOut (a 1-D float buffer/view). */
    template<typename TQueue, typename TExec, typename TOut>
    void enqueuePointFieldTile(
        TQueue& queue,
        TExec exec,
        Vec3d const& sourceR,
        Vec3d const& u,
        Vec3d const& cells,
        TOut& deviceOut)
    {
        using Vec = alpaka::Vec<std::uint32_t, 1>;
        auto const threads = Vec{64u};
        auto const frames = Vec{static_cast<std::uint32_t>((kTileSlotCount + 63u) / 64u)};
        queue.enqueue(
            alpaka::onHost::FrameSpec{frames, threads, exec},
            alpaka::KernelBundle{detail::PointFieldTileKernel{sourceR, u, cells}, deviceOut});
    }

    /** Batched device evaluation: out[S,1536], one work item per (source, slot). */
    template<typename TQueue, typename TExec, typename TSources, typename TOut>
    void enqueuePointFieldBatched(
        TQueue& queue,
        TExec exec,
        TSources const& deviceSources,
        Vec3d const& cells,
        TOut& deviceOut)
    {
        using Vec = alpaka::Vec<std::uint32_t, 1>;
        auto const sources = static_cast<std::size_t>(deviceSources.getExtents()[0]);
        auto const total = sources * kTileSlotCount;
        auto const threads = Vec{128u};
        auto const frames = Vec{static_cast<std::uint32_t>((total + 127u) / 128u)};
        queue.enqueue(
            alpaka::onHost::FrameSpec{frames, threads, exec},
            alpaka::KernelBundle{detail::PointFieldBatchedKernel{cells}, deviceSources, deviceOut});
    }

    /** Host convenience wrapper for a source batch; returns S*1536 values in flatten order. */
    template<typename TQueue, typename TExec, typename TSources>
    std::vector<float> pointFieldBatched(TQueue& queue, TExec exec, TSources const& sources, Vec3d const& cells)
    {
        if(sources.empty())
            return {};
        auto hostSources = alpaka::onHost::allocHost<double>(alpaka::Vec{sources.size(), std::size_t{6}});
        for(std::size_t i = 0u; i < sources.size(); ++i)
        {
            hostSources[alpaka::Vec<std::size_t, 2>{i, 0u}] = sources[i].r[0];
            hostSources[alpaka::Vec<std::size_t, 2>{i, 1u}] = sources[i].r[1];
            hostSources[alpaka::Vec<std::size_t, 2>{i, 2u}] = sources[i].r[2];
            hostSources[alpaka::Vec<std::size_t, 2>{i, 3u}] = sources[i].u[0];
            hostSources[alpaka::Vec<std::size_t, 2>{i, 4u}] = sources[i].u[1];
            hostSources[alpaka::Vec<std::size_t, 2>{i, 5u}] = sources[i].u[2];
        }
        auto deviceSources = alpaka::onHost::allocLike(queue.getDevice(), hostSources);
        alpaka::onHost::memcpy(queue, deviceSources, hostSources);
        hostSources.keepAlive(queue);
        auto deviceOut = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{sources.size(), kTileSlotCount});
        enqueuePointFieldBatched(queue, exec, deviceSources, cells, deviceOut);
        auto hostOut = alpaka::onHost::allocHost<float>(alpaka::Vec{sources.size(), kTileSlotCount});
        alpaka::onHost::memcpy(queue, hostOut, deviceOut);
        alpaka::onHost::wait(queue);
        std::vector<float> result(sources.size() * kTileSlotCount);
        for(std::size_t i = 0u; i < sources.size(); ++i)
            for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
                result[i * kTileSlotCount + slot] = hostOut[alpaka::Vec<std::size_t, 2>{i, slot}];
        return result;
    }

    /** Device evaluation of one source at all 1536 outputs; returns host values in flatten order. */
    template<typename TQueue, typename TExec>
    std::vector<float> pointFieldTile(
        TQueue& queue,
        TExec exec,
        Vec3d const& sourceR,
        Vec3d const& u,
        Vec3d const& cells)
    {
        auto deviceOut = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{kTileSlotCount});
        enqueuePointFieldTile(queue, exec, sourceR, u, cells, deviceOut);
        auto hostOut = alpaka::onHost::allocHost<float>(alpaka::Vec{kTileSlotCount});
        alpaka::onHost::memcpy(queue, hostOut, deviceOut);
        alpaka::onHost::wait(queue);
        std::vector<float> result(kTileSlotCount);
        for(std::size_t i = 0u; i < kTileSlotCount; ++i)
            result[i] = hostOut[alpaka::Vec<std::size_t, 1>{i}];
        return result;
    }
} // namespace picongpu::particles::neuralBunch
