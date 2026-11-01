/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Stand-alone per-tile inference core for the neural bunch-field model.
 *
 * Data flow for one tile and a list of sources (all lengths in ``dx`` units):
 *
 *   1. pack (r, u, a_y, a_z) into a device buffer,
 *   2. build the ``fourier_v1`` feature matrix on the device ([N,69] float32),
 *   3. ``mlpForward`` -> h ([N,latent], SiLU hidden, no activation on the last layer),
 *   4. device reduction z[latent] = sum_i (-w_i) * h_i,
 *   5. interpolate the decoder bank bilinearly in log10(a_y)/log10(a_z) on the device,
 *   6. one tiny GEMM F[1,1536] = z[1,latent] * W_interp[latent,1536],
 *   7. scale E and B components once with the MODEL-FORMAT normalisation and return the
 *      1536 values in the documented flatten order ``c + 6*(i + nx*(j + ny*k))``.
 *
 * Synchronisation and lifetime: every public helper enqueues work on the caller's queue and then calls
 * ``alpaka::onHost::wait(queue)`` before returning a host result or letting a temporary buffer die. A returned
 * device buffer therefore has no pending work that still references destroyed temporaries. The caller owns the
 * queue and must not destroy it while a helper runs. All temporary device buffers are local to the helper, so the
 * peak scratch is bounded by
 *
 *     N*(8 doubles + 69 floats) + N*latent floats + latent*1536 floats + 1536 floats,
 *
 * with ``N <= maxSources`` (default 4096, checked before any allocation).
 *
 * The decoder bank and encoder weights are taken from the buffers returned by ``Model::loadTensors`` (flat 1-D
 * device buffers); 2-D views are built with ``alpaka::makeView`` and do not own the payload, so those buffers must
 * stay alive for the whole evaluator lifetime.
 */

#include "Features.hpp"
#include "Model.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>
#include <alpaka/nn/onHost/nn/mlp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace picongpu::particles::neuralBunch
{
    /** One bunch source in tile-local coordinates.
     *
     * ``r`` is the position relative to the tile cell origin in dx units, ``u = p/(m_e c)``, ``weight`` is the
     * positive macro-particle weight (the model uses ``q/e = -weight``), and ``aY``/``aZ`` are ``dy/dx`` and
     * ``dz/dx`` (the per-source aspect used by the feature map).
     */
    struct Source
    {
        double r[3]{0.0, 0.0, 0.0};
        double u[3]{0.0, 0.0, 0.0};
        double weight = 1.0;
        double aY = 1.0;
        double aZ = 1.0;
    };

    /** Bilinear decoder cell in log10(a) space, clamped at the outer anchors. */
    struct BilinearCell
    {
        std::array<std::size_t, 4> index{};
        std::array<float, 4> weight{};
    };

    /** Reproduces ``model.interp_indices_weights`` from the Python reference exactly. */
    inline BilinearCell bilinearCell(
        std::vector<double> const& anchorsY,
        std::vector<double> const& anchorsZ,
        double aY,
        double aZ)
    {
        auto axisCell = [](double value, std::vector<double> const& anchors)
        {
            auto const n = anchors.size();
            if(n < 2u)
                throw std::invalid_argument{"bilinearCell: at least two anchors are required."};
            auto const low = std::log10(anchors.front());
            auto const high = std::log10(anchors.back());
            double t = (std::log10(value) - low) / (high - low) * static_cast<double>(n - 1u);
            t = std::clamp(t, 0.0, static_cast<double>(n - 1u));
            auto i0 = static_cast<std::size_t>(std::floor(t));
            i0 = std::min(i0, n - 2u);
            auto const i1 = std::min(i0 + 1u, n - 1u);
            return std::array<std::size_t, 2>{i0, i1};
        };
        auto const cellY = axisCell(aY, anchorsY);
        auto const cellZ = axisCell(aZ, anchorsZ);
        auto const iy0 = cellY[0];
        auto const iy1 = cellY[1];
        auto const iz0 = cellZ[0];
        auto const iz1 = cellZ[1];
        // Recompute the fractions; the helper above only returns the integer cell.
        auto fraction = [](double value, std::vector<double> const& anchors, std::size_t i0)
        {
            auto const n = anchors.size();
            auto const low = std::log10(anchors.front());
            auto const high = std::log10(anchors.back());
            double t = (std::log10(value) - low) / (high - low) * static_cast<double>(n - 1u);
            t = std::clamp(t, 0.0, static_cast<double>(n - 1u));
            return t - static_cast<double>(i0);
        };
        auto const fy = fraction(aY, anchorsY, iy0);
        auto const fz = fraction(aZ, anchorsZ, iz0);
        auto const ny = anchorsY.size();

        BilinearCell result;
        result.index = {iz0 * ny + iy0, iz0 * ny + iy1, iz1 * ny + iy0, iz1 * ny + iy1};
        result.weight
            = {static_cast<float>((1.0 - fy) * (1.0 - fz)),
               static_cast<float>(fy * (1.0 - fz)),
               static_cast<float>((1.0 - fy) * fz),
               static_cast<float>(fy * fz)};
        return result;
    }

    namespace detail
    {
        template<typename TExec>
        inline auto makeFrameSpec(std::size_t extent, std::uint32_t maxThreads, TExec exec)
        {
            using Vec = alpaka::Vec<std::uint32_t, 1>;
            auto const threads
                = static_cast<std::uint32_t>(std::max<std::size_t>(1u, std::min<std::size_t>(extent, maxThreads)));
            auto const frames = static_cast<std::uint32_t>((extent + threads - 1u) / threads);
            return alpaka::onHost::FrameSpec{Vec{frames}, Vec{threads}, exec};
        }

        struct FourierV1Kernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto sources, auto features) const
            {
                auto const count = sources.getExtents()[0];
                for(auto i : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{count}}))
                {
                    auto const row = static_cast<std::size_t>(i[0]);
                    auto value = [&](std::size_t column) { return sources[alpaka::Vec<std::size_t, 2>{row, column}]; };
                    FourierV1::fill(
                        &features[alpaka::Vec<std::size_t, 2>{row, 0u}],
                        value(0u),
                        value(1u),
                        value(2u),
                        value(3u),
                        value(4u),
                        value(5u),
                        value(6u),
                        value(7u));
                }
            }
        };

        struct WeightedReduceKernel
        {
            ALPAKA_FN_ACC void operator()(auto const& acc, auto hidden, auto weights, auto latent) const
            {
                auto const latentCount = latent.getExtents()[0];
                auto const sourceCount = hidden.getExtents()[0];
                for(auto i : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{latentCount}}))
                {
                    auto const column = static_cast<std::size_t>(i[0]);
                    float sum = 0.0f;
                    for(std::size_t row = 0u; row < static_cast<std::size_t>(sourceCount); ++row)
                        sum += hidden[alpaka::Vec<std::size_t, 2>{row, column}]
                               * weights[alpaka::Vec<std::size_t, 1>{row}];
                    latent[i] = sum;
                }
            }
        };

        struct InterpolateDecoderKernel
        {
            float w0{};
            float w1{};
            float w2{};
            float w3{};

            ALPAKA_FN_ACC void operator()(auto const& acc, auto bank0, auto bank1, auto bank2, auto bank3, auto out)
                const
            {
                auto const extents = out.getExtents();
                auto const columns = static_cast<std::size_t>(extents[1]);
                auto const total = static_cast<std::size_t>(extents[0]) * columns;
                for(auto i : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{total}}))
                {
                    auto const flat = static_cast<std::size_t>(i[0]);
                    auto const index = alpaka::Vec<std::size_t, 2>{flat / columns, flat % columns};
                    out[index] = w0 * bank0[index] + w1 * bank1[index] + w2 * bank2[index] + w3 * bank3[index];
                }
            }
        };

        struct NormalizeFieldKernel
        {
            double scaleE{};
            double scaleB{};

            ALPAKA_FN_ACC void operator()(auto const& acc, auto field) const
            {
                auto const extents = field.getExtents();
                auto const columns = static_cast<std::size_t>(extents[1]);
                auto const total = static_cast<std::size_t>(extents[0]) * columns;
                for(auto i : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{total}}))
                {
                    auto const flat = static_cast<std::size_t>(i[0]);
                    auto const index = alpaka::Vec<std::size_t, 2>{flat / columns, flat % columns};
                    auto const component = index[1] % 6u;
                    auto const scale = component < 3u ? scaleE : scaleB;
                    field[index] = static_cast<float>(static_cast<double>(field[index]) * scale);
                }
            }
        };

        template<typename TQueue>
        auto packSources(TQueue& queue, std::span<Source const> sources)
        {
            auto const count = sources.size();
            auto hostPacked = alpaka::onHost::allocHost<double>(alpaka::Vec{count, std::size_t{8}});
            for(std::size_t i = 0u; i < count; ++i)
            {
                hostPacked[alpaka::Vec<std::size_t, 2>{i, 0u}] = sources[i].r[0];
                hostPacked[alpaka::Vec<std::size_t, 2>{i, 1u}] = sources[i].r[1];
                hostPacked[alpaka::Vec<std::size_t, 2>{i, 2u}] = sources[i].r[2];
                hostPacked[alpaka::Vec<std::size_t, 2>{i, 3u}] = sources[i].u[0];
                hostPacked[alpaka::Vec<std::size_t, 2>{i, 4u}] = sources[i].u[1];
                hostPacked[alpaka::Vec<std::size_t, 2>{i, 5u}] = sources[i].u[2];
                hostPacked[alpaka::Vec<std::size_t, 2>{i, 6u}] = sources[i].aY;
                hostPacked[alpaka::Vec<std::size_t, 2>{i, 7u}] = sources[i].aZ;
            }
            auto devicePacked = alpaka::onHost::allocLike(queue.getDevice(), hostPacked);
            alpaka::onHost::memcpy(queue, devicePacked, hostPacked);
            hostPacked.keepAlive(queue);
            return devicePacked;
        }

        template<typename TQueue, typename TExec, typename TBuffer, typename THidden>
        void encodeInto(
            TQueue& queue,
            TExec exec,
            ModelMetadata const& meta,
            std::vector<TBuffer> const& tensors,
            std::span<Source const> sources,
            THidden& hidden)
        {
            auto const count = sources.size();
            if(static_cast<std::size_t>(hidden.getExtents()[0]) != count
               || static_cast<std::size_t>(hidden.getExtents()[1]) != meta.latent)
                throw std::invalid_argument{"encodeInto: hidden extent must be [N, latent]."};

            auto deviceSources = packSources(queue, sources);
            auto features = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{count, meta.featureDim});
            queue.enqueue(
                makeFrameSpec(count, 64u, exec),
                alpaka::KernelBundle{FourierV1Kernel{}, deviceSources, features});

            using Weight = alpaka::nn::onHost::nn::WeightView<decltype(alpaka::makeView(
                queue.getDevice(),
                alpaka::onHost::data(tensors.front()),
                alpaka::Vec{std::size_t{1}, std::size_t{1}}))>;
            using Bias = alpaka::nn::onHost::nn::BiasView<decltype(alpaka::makeView(
                queue.getDevice(),
                alpaka::onHost::data(tensors.front()),
                alpaka::Vec{std::size_t{1}}))>;

            std::vector<Weight> weights;
            std::vector<Bias> biases;
            weights.reserve(meta.layers.size());
            biases.reserve(meta.layers.size());
            for(auto const& layer : meta.layers)
            {
                auto const& weightBuffer = tensors[meta.tensorIndex(layer.weight)];
                auto const& biasBuffer = tensors[meta.tensorIndex(layer.bias)];
                auto weightView = alpaka::makeView(
                    queue.getDevice(),
                    alpaka::onHost::data(weightBuffer),
                    alpaka::Vec{layer.in, layer.out});
                auto biasView
                    = alpaka::makeView(queue.getDevice(), alpaka::onHost::data(biasBuffer), alpaka::Vec{layer.out});
                weights.emplace_back(Weight{weightView});
                biases.emplace_back(Bias{biasView});
            }

            alpaka::nn::onHost::nn::mlpForward<float>(
                queue,
                exec,
                features,
                std::span<Weight const>{weights},
                std::span<Bias const>{biases},
                alpaka::nn::onHost::nn::Activation::silu,
                alpaka::nn::onHost::nn::Activation::none,
                hidden);
            alpaka::onHost::wait(queue);
        }
    } // namespace detail

    /** Build the feature matrix on the device and return it on the host (test helper). */
    template<typename TQueue, typename TExec>
    std::vector<float> computeFeatures(
        TQueue& queue,
        TExec exec,
        ModelMetadata const& meta,
        std::span<Source const> sources)
    {
        auto const count = sources.size();
        std::vector<float> result(count * meta.featureDim, 0.0f);
        if(count == 0u)
            return result;

        auto deviceSources = detail::packSources(queue, sources);
        auto features = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{count, meta.featureDim});
        queue.enqueue(
            detail::makeFrameSpec(count, 64u, exec),
            alpaka::KernelBundle{detail::FourierV1Kernel{}, deviceSources, features});
        auto hostFeatures = alpaka::onHost::allocHost<float>(alpaka::Vec{count, meta.featureDim});
        alpaka::onHost::memcpy(queue, hostFeatures, features);
        alpaka::onHost::wait(queue);
        for(std::size_t i = 0u; i < count; ++i)
            for(std::size_t k = 0u; k < meta.featureDim; ++k)
                result[i * meta.featureDim + k] = hostFeatures[alpaka::Vec<std::size_t, 2>{i, k}];
        return result;
    }

    /** Run the encoder and return h[N, latent] on the host (test helper). */
    template<typename TQueue, typename TExec, typename TBuffer>
    std::vector<float> runEncoder(
        TQueue& queue,
        TExec exec,
        ModelMetadata const& meta,
        std::vector<TBuffer> const& tensors,
        std::span<Source const> sources)
    {
        auto const count = sources.size();
        std::vector<float> result(count * meta.latent, 0.0f);
        if(count == 0u)
            return result;

        auto hidden = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{count, meta.latent});
        detail::encodeInto(queue, exec, meta, tensors, sources, hidden);
        auto hostHidden = alpaka::onHost::allocHost<float>(alpaka::Vec{count, meta.latent});
        alpaka::onHost::memcpy(queue, hostHidden, hidden);
        alpaka::onHost::wait(queue);
        for(std::size_t i = 0u; i < count; ++i)
            for(std::size_t k = 0u; k < meta.latent; ++k)
                result[i * meta.latent + k] = hostHidden[alpaka::Vec<std::size_t, 2>{i, k}];
        return result;
    }

    /** Interpolate the bank and apply z on the device; return the 1536 normalised outputs. */
    template<typename TQueue, typename TExec, typename TBuffer>
    std::vector<float> decodeFromLatent(
        TQueue& queue,
        TExec exec,
        ModelMetadata const& meta,
        std::vector<TBuffer> const& tensors,
        std::vector<float> const& latent,
        double aY,
        double aZ,
        FieldNormalization const& normalization = {})
    {
        if(latent.size() != meta.latent)
            throw std::invalid_argument{"decodeFromLatent: latent vector has the wrong length."};
        auto const cell = bilinearCell(meta.aYAnchors, meta.aZAnchors, aY, aZ);

        auto bankView = [&](std::size_t index)
        {
            auto const& buffer = tensors[meta.tensorIndex(meta.bank[index])];
            return alpaka::makeView(
                queue.getDevice(),
                alpaka::onHost::data(buffer),
                alpaka::Vec{meta.latent, kOutDim});
        };
        auto bank0 = bankView(cell.index[0]);
        auto bank1 = bankView(cell.index[1]);
        auto bank2 = bankView(cell.index[2]);
        auto bank3 = bankView(cell.index[3]);

        auto interpolated = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{meta.latent, kOutDim});
        detail::InterpolateDecoderKernel const interp{cell.weight[0], cell.weight[1], cell.weight[2], cell.weight[3]};
        queue.enqueue(
            detail::makeFrameSpec(meta.latent * kOutDim, 256u, exec),
            alpaka::KernelBundle{interp, bank0, bank1, bank2, bank3, interpolated});

        auto hostLatent = alpaka::onHost::allocHost<float>(alpaka::Vec{std::size_t{1}, meta.latent});
        for(std::size_t i = 0u; i < meta.latent; ++i)
            hostLatent[alpaka::Vec<std::size_t, 2>{0u, i}] = latent[i];
        auto deviceLatent = alpaka::onHost::allocLike(queue.getDevice(), hostLatent);
        alpaka::onHost::memcpy(queue, deviceLatent, hostLatent);
        hostLatent.keepAlive(queue);

        auto field = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{std::size_t{1}, kOutDim});
        alpaka::nn::onHost::gemm<float>(queue, deviceLatent, interpolated, field);

        detail::NormalizeFieldKernel const normalize{normalization.scaleE(), normalization.scaleB()};
        queue.enqueue(detail::makeFrameSpec(kOutDim, 256u, exec), alpaka::KernelBundle{normalize, field});

        auto hostField = alpaka::onHost::allocHost<float>(alpaka::Vec{std::size_t{1}, kOutDim});
        alpaka::onHost::memcpy(queue, hostField, field);
        alpaka::onHost::wait(queue);

        std::vector<float> result(kOutDim);
        for(std::size_t i = 0u; i < kOutDim; ++i)
            result[i] = hostField[alpaka::Vec<std::size_t, 2>{0u, i}];
        return result;
    }

    /** Full per-tile evaluation. Returns the 1536 normalised values in MODEL-FORMAT flatten order. */
    template<typename TQueue, typename TExec, typename TBuffer>
    std::vector<float> evaluateTile(
        TQueue& queue,
        TExec exec,
        ModelMetadata const& meta,
        std::vector<TBuffer> const& tensors,
        std::span<Source const> sources,
        double aY,
        double aZ,
        FieldNormalization const& normalization = {},
        std::size_t maxSources = 4096u)
    {
        if(sources.size() > maxSources)
            throw std::invalid_argument{
                "evaluateTile: " + std::to_string(sources.size()) + " sources exceed the configured bound of "
                + std::to_string(maxSources) + "."};
        if(sources.empty())
            return std::vector<float>(kOutDim, 0.0f);

        auto hidden = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{sources.size(), meta.latent});
        detail::encodeInto(queue, exec, meta, tensors, sources, hidden);

        auto hostWeights = alpaka::onHost::allocHost<float>(alpaka::Vec{sources.size()});
        for(std::size_t i = 0u; i < sources.size(); ++i)
            // signed charge q/e = -w
            hostWeights[alpaka::Vec<std::size_t, 1>{i}] = static_cast<float>(-sources[i].weight);
        auto deviceWeights = alpaka::onHost::allocLike(queue.getDevice(), hostWeights);
        alpaka::onHost::memcpy(queue, deviceWeights, hostWeights);
        hostWeights.keepAlive(queue);

        auto deviceLatent = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{meta.latent});
        queue.enqueue(
            detail::makeFrameSpec(meta.latent, 128u, exec),
            alpaka::KernelBundle{detail::WeightedReduceKernel{}, hidden, deviceWeights, deviceLatent});

        auto hostLatent = alpaka::onHost::allocHost<float>(alpaka::Vec{meta.latent});
        alpaka::onHost::memcpy(queue, hostLatent, deviceLatent);
        alpaka::onHost::wait(queue);

        std::vector<float> latent(meta.latent);
        for(std::size_t i = 0u; i < meta.latent; ++i)
            latent[i] = hostLatent[alpaka::Vec<std::size_t, 1>{i}];

        return decodeFromLatent(queue, exec, meta, tensors, latent, aY, aZ, normalization);
    }
} // namespace picongpu::particles::neuralBunch
