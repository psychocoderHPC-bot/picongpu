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

#include "BandPolicy.hpp"
#include "Features.hpp"
#include "Model.hpp"
#include "PointField.hpp"
#include "TaylorField.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>
#include <alpaka/nn/onHost/nn/mlp.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
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

        template<FeatureKind Kind>
        struct FeaturesKernel
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
                    fillFeatures<Kind>(
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

        template<typename TQueue, typename TExec>
        void enqueueFeatures(
            TQueue& queue,
            TExec exec,
            FeatureKind kind,
            std::size_t count,
            auto const& deviceSources,
            auto& features)
        {
            auto launch = [&](auto tag)
            {
                constexpr FeatureKind Kind = decltype(tag)::value;
                queue.enqueue(
                    makeFrameSpec(count, 64u, exec),
                    alpaka::KernelBundle{FeaturesKernel<Kind>{}, deviceSources, features});
            };
            switch(kind)
            {
            case FeatureKind::fourierV1:
                launch(std::integral_constant<FeatureKind, FeatureKind::fourierV1>{});
                break;
            case FeatureKind::fourierProbeV1:
                launch(std::integral_constant<FeatureKind, FeatureKind::fourierProbeV1>{});
                break;
            case FeatureKind::probeDenseV1:
                launch(std::integral_constant<FeatureKind, FeatureKind::probeDenseV1>{});
                break;
            }
        }

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
        void encodeLayers(
            TQueue& queue,
            TExec exec,
            ModelMetadata const& meta,
            FeatureKind featureKind,
            std::size_t featureDim,
            std::vector<EncoderLayer> const& layers,
            std::vector<TBuffer> const& tensors,
            std::span<Source const> sources,
            THidden& hidden)
        {
            auto const count = sources.size();
            if(static_cast<std::size_t>(hidden.getExtents()[0]) != count)
                throw std::invalid_argument{"encodeLayers: hidden row count must match the source count."};
            if(layers.empty())
                throw std::invalid_argument{"encodeLayers: the encoder must have at least one layer."};
            if(static_cast<std::size_t>(hidden.getExtents()[1]) != layers.back().out)
                throw std::invalid_argument{"encodeLayers: hidden column count must match the last layer out."};

            auto deviceSources = packSources(queue, sources);
            auto features = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{count, featureDim});
            enqueueFeatures(queue, exec, featureKind, count, deviceSources, features);

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
            weights.reserve(layers.size());
            biases.reserve(layers.size());
            for(auto const& layer : layers)
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

        template<typename TQueue, typename TExec, typename TBuffer, typename THidden>
        void encodeInto(
            TQueue& queue,
            TExec exec,
            ModelMetadata const& meta,
            std::vector<TBuffer> const& tensors,
            std::span<Source const> sources,
            THidden& hidden)
        {
            if(static_cast<std::size_t>(hidden.getExtents()[0]) != sources.size()
               || static_cast<std::size_t>(hidden.getExtents()[1]) != meta.latent)
                throw std::invalid_argument{"encodeInto: hidden extent must be [N, latent]."};
            encodeLayers(queue, exec, meta, meta.featureKind, meta.featureDim, meta.layers, tensors, sources, hidden);
        }
    } // namespace detail

    /** Build the feature matrix for an explicit kind on the device (test helper). */
    template<typename TQueue, typename TExec>
    std::vector<float> computeFeatures(TQueue& queue, TExec exec, FeatureKind kind, std::span<Source const> sources)
    {
        auto const count = sources.size();
        auto const dim = featureDimension(kind);
        std::vector<float> result(count * dim, 0.0f);
        if(count == 0u)
            return result;

        auto deviceSources = detail::packSources(queue, sources);
        auto features = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{count, dim});
        detail::enqueueFeatures(queue, exec, kind, count, deviceSources, features);
        auto hostFeatures = alpaka::onHost::allocHost<float>(alpaka::Vec{count, dim});
        alpaka::onHost::memcpy(queue, hostFeatures, features);
        alpaka::onHost::wait(queue);
        for(std::size_t i = 0u; i < count; ++i)
            for(std::size_t k = 0u; k < dim; ++k)
                result[i * dim + k] = hostFeatures[alpaka::Vec<std::size_t, 2>{i, k}];
        return result;
    }

    /** Build the feature matrix described by model metadata on the device (test helper). */
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
        detail::enqueueFeatures(queue, exec, meta.featureKind, count, deviceSources, features);
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

    /** Run an arbitrary encoder chain and return h[N, out] on the host. */
    template<typename TQueue, typename TExec, typename TBuffer>
    std::vector<float> runEncoderLayers(
        TQueue& queue,
        TExec exec,
        ModelMetadata const& meta,
        FeatureKind featureKind,
        std::size_t featureDim,
        std::vector<EncoderLayer> const& layers,
        std::vector<TBuffer> const& tensors,
        std::span<Source const> sources)
    {
        auto const count = sources.size();
        auto const outDim = layers.empty() ? std::size_t{0} : layers.back().out;
        std::vector<float> result(count * outDim, 0.0f);
        if(count == 0u)
            return result;
        if(outDim == 0u)
            throw std::invalid_argument{"runEncoderLayers: the encoder must have at least one layer."};

        auto hidden = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{count, outDim});
        detail::encodeLayers(queue, exec, meta, featureKind, featureDim, layers, tensors, sources, hidden);
        auto hostHidden = alpaka::onHost::allocHost<float>(alpaka::Vec{count, outDim});
        alpaka::onHost::memcpy(queue, hostHidden, hidden);
        alpaka::onHost::wait(queue);
        for(std::size_t i = 0u; i < count; ++i)
            for(std::size_t k = 0u; k < outDim; ++k)
                result[i * outDim + k] = hostHidden[alpaka::Vec<std::size_t, 2>{i, k}];
        return result;
    }

    /** Apply a dense linear decoder (weight [in,out], optional bias [out]) without field normalization. */
    template<typename TQueue, typename TExec, typename TBuffer>
    std::vector<float> decodeLinearRaw(
        TQueue& queue,
        TExec exec,
        ModelMetadata const& meta,
        std::vector<TBuffer> const& tensors,
        LinearDecoderSpec const& decoder,
        std::vector<float> const& latent)
    {
        if(latent.size() != decoder.in)
            throw std::invalid_argument{"decodeLinearRaw: latent vector has the wrong length."};
        auto const& weightBuffer = tensors[meta.tensorIndex(decoder.weight)];
        auto weight = alpaka::makeView(
            queue.getDevice(),
            alpaka::onHost::data(weightBuffer),
            alpaka::Vec{decoder.in, decoder.out});

        auto hostLatent = alpaka::onHost::allocHost<float>(alpaka::Vec{std::size_t{1}, decoder.in});
        for(std::size_t i = 0u; i < decoder.in; ++i)
            hostLatent[alpaka::Vec<std::size_t, 2>{0u, i}] = latent[i];
        auto deviceLatent = alpaka::onHost::allocLike(queue.getDevice(), hostLatent);
        alpaka::onHost::memcpy(queue, deviceLatent, hostLatent);
        hostLatent.keepAlive(queue);

        auto out = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{std::size_t{1}, decoder.out});
        alpaka::nn::onHost::gemm<float>(queue, deviceLatent, weight, out);

        if(decoder.hasBias)
        {
            auto const& biasBuffer = tensors[meta.tensorIndex(decoder.bias)];
            auto bias
                = alpaka::makeView(queue.getDevice(), alpaka::onHost::data(biasBuffer), alpaka::Vec{decoder.out});
            alpaka::nn::onHost::ops::biasAdd(queue, exec, out, bias, out);
        }

        auto hostOut = alpaka::onHost::allocHost<float>(alpaka::Vec{std::size_t{1}, decoder.out});
        alpaka::onHost::memcpy(queue, hostOut, out);
        alpaka::onHost::wait(queue);
        std::vector<float> result(decoder.out);
        for(std::size_t i = 0u; i < decoder.out; ++i)
            result[i] = hostOut[alpaka::Vec<std::size_t, 2>{0u, i}];
        return result;
    }

    /** Batched dense linear decoder: latent [S,in] -> [S,out], optional bias, no field normalization. */
    template<typename TQueue, typename TExec, typename TBuffer>
    std::vector<float> decodeLinearBatchedRaw(
        TQueue& queue,
        TExec exec,
        ModelMetadata const& meta,
        std::vector<TBuffer> const& tensors,
        LinearDecoderSpec const& decoder,
        std::vector<float> const& latentBatch,
        std::size_t batches)
    {
        if(latentBatch.size() != batches * decoder.in)
            throw std::invalid_argument{"decodeLinearBatchedRaw: latent batch shape mismatch."};
        auto const& weightBuffer = tensors[meta.tensorIndex(decoder.weight)];
        auto weight = alpaka::makeView(
            queue.getDevice(),
            alpaka::onHost::data(weightBuffer),
            alpaka::Vec{decoder.in, decoder.out});

        auto hostLatent = alpaka::onHost::allocHost<float>(alpaka::Vec{batches, decoder.in});
        for(std::size_t b = 0u; b < batches; ++b)
            for(std::size_t i = 0u; i < decoder.in; ++i)
                hostLatent[alpaka::Vec<std::size_t, 2>{b, i}] = latentBatch[b * decoder.in + i];
        auto deviceLatent = alpaka::onHost::allocLike(queue.getDevice(), hostLatent);
        alpaka::onHost::memcpy(queue, deviceLatent, hostLatent);
        hostLatent.keepAlive(queue);

        auto out = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{batches, decoder.out});
        alpaka::nn::onHost::gemm<float>(queue, deviceLatent, weight, out);
        if(decoder.hasBias)
        {
            auto const& biasBuffer = tensors[meta.tensorIndex(decoder.bias)];
            auto bias
                = alpaka::makeView(queue.getDevice(), alpaka::onHost::data(biasBuffer), alpaka::Vec{decoder.out});
            alpaka::nn::onHost::ops::biasAdd(queue, exec, out, bias, out);
        }
        auto hostOut = alpaka::onHost::allocHost<float>(alpaka::Vec{batches, decoder.out});
        alpaka::onHost::memcpy(queue, hostOut, out);
        alpaka::onHost::wait(queue);
        std::vector<float> result(batches * decoder.out);
        for(std::size_t b = 0u; b < batches; ++b)
            for(std::size_t i = 0u; i < decoder.out; ++i)
                result[b * decoder.out + i] = hostOut[alpaka::Vec<std::size_t, 2>{b, i}];
        return result;
    }

    /** Batched decoder-bank evaluation: latent [S,latent] -> [S,1536], one GEMM, no normalization. */
    template<typename TQueue, typename TExec, typename TBuffer>
    std::vector<float> decodeBankBatchedRaw(
        TQueue& queue,
        TExec exec,
        ModelMetadata const& meta,
        std::vector<TBuffer> const& tensors,
        std::vector<float> const& latentBatch,
        std::size_t batches,
        double aY,
        double aZ)
    {
        if(latentBatch.size() != batches * meta.latent)
            throw std::invalid_argument{"decodeBankBatchedRaw: latent batch shape mismatch."};
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

        auto hostLatent = alpaka::onHost::allocHost<float>(alpaka::Vec{batches, meta.latent});
        for(std::size_t b = 0u; b < batches; ++b)
            for(std::size_t i = 0u; i < meta.latent; ++i)
                hostLatent[alpaka::Vec<std::size_t, 2>{b, i}] = latentBatch[b * meta.latent + i];
        auto deviceLatent = alpaka::onHost::allocLike(queue.getDevice(), hostLatent);
        alpaka::onHost::memcpy(queue, deviceLatent, hostLatent);
        hostLatent.keepAlive(queue);

        auto out = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{batches, kOutDim});
        alpaka::nn::onHost::gemm<float>(queue, deviceLatent, interpolated, out);
        auto hostOut = alpaka::onHost::allocHost<float>(alpaka::Vec{batches, kOutDim});
        alpaka::onHost::memcpy(queue, hostOut, out);
        alpaka::onHost::wait(queue);
        std::vector<float> result(batches * kOutDim);
        for(std::size_t b = 0u; b < batches; ++b)
            for(std::size_t i = 0u; i < kOutDim; ++i)
                result[b * kOutDim + i] = hostOut[alpaka::Vec<std::size_t, 2>{b, i}];
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
        FieldNormalization const& normalization = {},
        bool applyNormalization = true)
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

        if(applyNormalization)
        {
            detail::NormalizeFieldKernel const normalize{normalization.scaleE(), normalization.scaleB()};
            queue.enqueue(detail::makeFrameSpec(kOutDim, 256u, exec), alpaka::KernelBundle{normalize, field});
        }

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

    /** Per-source three-band routing result. */
    struct BandedTileResult
    {
        std::vector<float> values; // 1536 normalised outputs (partial when near/mid are not implemented)
        std::array<std::size_t, 3> sourceCounts{}; // near, mid, far (zero-weight sources excluded)
        std::array<double, 3> bandSeconds{}; // near, mid, far device evaluation time
        bool nearNotImplemented = false;
        bool midNotImplemented = false;
    };

    /** Optional callback for the near band. Returning nullopt means "near model not available". */
    using NearTileEvaluator = std::function<std::optional<std::vector<float>>(Source const&)>;

    /** Three-band tile evaluation.
     *
     * Each non-zero source is classified with @p policy using the tile aspect @p aY/@p aZ and its distance to the
     * nearest staggered output. Per-source contributions are built before charge weighting:
     *   - near: the optional ``near_model`` full-field head, else @p nearEvaluator, else marked incomplete,
     *   - mid: the analytic point field plus the optional ``mid_correction`` head; without a correction the
     *     existing neural encoder/decoder bank is used,
     *   - far: the analytic point field, plus the optional ``far_multipole`` Taylor head when present.
     * Each band is evaluated in batches of at most @p maxBatch sources with one encoder GEMM, one decoder
     * GEMM/GEMV and one batched point-field kernel per batch. Scratch stays bounded by maxBatch. The per-source
     * contributions are multiplied by q/e = -weight and accumulated in original source order, and the field
     * normalization is applied once at the end.
     */
    template<typename TQueue, typename TExec, typename TBuffer>
    BandedTileResult evaluateTileBanded(
        TQueue& queue,
        TExec exec,
        ModelMetadata const& meta,
        std::vector<TBuffer> const& tensors,
        std::span<Source const> sources,
        BandPolicy const& policy,
        NearTileEvaluator nearEvaluator = {},
        double aY = 1.0,
        double aZ = 1.0,
        FieldNormalization const& normalization = {},
        std::size_t maxSources = 4096u,
        std::size_t maxBatch = 64u,
        TaylorBatchBackend farBackend = TaylorBatchBackend::automatic,
        bool usePrimaryModelForNear = false,
        bool useModelForMid = false,
        bool useMidCorrection = false)
    {
        if(sources.size() > maxSources)
            throw std::invalid_argument{
                "evaluateTileBanded: " + std::to_string(sources.size()) + " sources exceed the bound of "
                + std::to_string(maxSources) + "."};
        if(maxBatch == 0u)
            throw std::invalid_argument{"evaluateTileBanded: maxBatch must be positive."};
        if(farBackend == TaylorBatchBackend::automatic)
            farBackend = defaultTaylorBatchBackend(queue.getDevice());

        BandedTileResult result;
        result.values.assign(kOutDim, 0.0f);
        Vec3d const cells{1.0, aY, aZ};

        std::vector<std::size_t> nearIndices;
        std::vector<std::size_t> midIndices;
        std::vector<std::size_t> farIndices;
        for(std::size_t index = 0u; index < sources.size(); ++index)
        {
            auto const& source = sources[index];
            if(source.weight == 0.0)
                continue;
            Band const band = policy.classify(Vec3d{source.r[0], source.r[1], source.r[2]}, cells);
            if(band == Band::near)
            {
                ++result.sourceCounts[0];
                nearIndices.push_back(index);
            }
            else if(band == Band::mid)
            {
                ++result.sourceCounts[1];
                midIndices.push_back(index);
            }
            else
            {
                ++result.sourceCounts[2];
                farIndices.push_back(index);
            }
        }

        // Accumulate each source directly into the tile sum in the fixed band/batch order. This keeps the
        // scratch bounded by maxBatch instead of sources*1536 and is deterministic for a fixed chunking.
        auto accumulate
            = [&](std::vector<std::size_t> const& indices, std::size_t begin, std::vector<float> const& values)
        {
            auto const count = values.size() / kOutDim;
            for(std::size_t b = 0u; b < count; ++b)
            {
                double const charge = -sources[indices[begin + b]].weight;
                for(std::size_t slot = 0u; slot < kOutDim; ++slot)
                    result.values[slot]
                        += static_cast<float>(charge * static_cast<double>(values[b * kOutDim + slot]));
            }
        };

        auto makeBatch = [&](std::vector<std::size_t> const& indices, std::size_t begin, std::size_t end)
        {
            std::vector<Source> batchSources;
            batchSources.reserve(end - begin);
            for(std::size_t i = begin; i < end; ++i)
                batchSources.push_back(sources[indices[i]]);
            return batchSources;
        };

        // near: optional model head, then explicit callback, otherwise mark incomplete.
        if(!nearIndices.empty())
        {
            auto const nearStart = std::chrono::steady_clock::now();
            if(meta.nearModel.present)
            {
                for(std::size_t begin = 0u; begin < nearIndices.size(); begin += maxBatch)
                {
                    auto const end = std::min(begin + maxBatch, nearIndices.size());
                    auto batchSources = makeBatch(nearIndices, begin, end);
                    auto const latent = runEncoderLayers(
                        queue,
                        exec,
                        meta,
                        meta.featureKind,
                        meta.featureDim,
                        meta.nearModel.layers,
                        tensors,
                        batchSources);
                    auto const values = decodeLinearBatchedRaw(
                        queue,
                        exec,
                        meta,
                        tensors,
                        meta.nearModel.decoder,
                        latent,
                        batchSources.size());
                    accumulate(nearIndices, begin, values);
                }
            }
            else if(usePrimaryModelForNear && !meta.layers.empty() && !meta.bank.empty())
            {
                for(std::size_t begin = 0u; begin < nearIndices.size(); begin += maxBatch)
                {
                    auto const end = std::min(begin + maxBatch, nearIndices.size());
                    auto batchSources = makeBatch(nearIndices, begin, end);
                    auto const latent = runEncoderLayers(
                        queue,
                        exec,
                        meta,
                        meta.featureKind,
                        meta.featureDim,
                        meta.layers,
                        tensors,
                        batchSources);
                    auto const values
                        = decodeBankBatchedRaw(queue, exec, meta, tensors, latent, batchSources.size(), aY, aZ);
                    accumulate(nearIndices, begin, values);
                }
            }
            else
            {
                for(auto const index : nearIndices)
                {
                    if(!nearEvaluator)
                    {
                        result.nearNotImplemented = true;
                        continue;
                    }
                    auto tile = nearEvaluator(sources[index]);
                    if(!tile || tile->size() != kOutDim)
                    {
                        result.nearNotImplemented = true;
                        continue;
                    }
                    double const charge = -sources[index].weight;
                    for(std::size_t slot = 0u; slot < kOutDim; ++slot)
                        result.values[slot] += static_cast<float>(charge * static_cast<double>((*tile)[slot]));
                }
            }
            result.bandSeconds[0]
                = std::chrono::duration<double>(std::chrono::steady_clock::now() - nearStart).count();
        }

        // Precompute the far-band evaluation matrix once per tile.
        std::vector<double> farMatrix;
        if(meta.farMultipole.present)
        {
            Vec3d const farCells{1.0, meta.farMultipole.aspectY, meta.farMultipole.aspectZ};
            farMatrix = taylorMatrixHost(meta.farMultipole.order, farCells, meta.farMultipole.origin);
        }

        if(!midIndices.empty())
        {
            auto const midStart = std::chrono::steady_clock::now();
            bool const hasModel = !meta.layers.empty() && !meta.bank.empty();
            if(useModelForMid && !hasModel)
                result.midNotImplemented = true;
            else if(useMidCorrection && !meta.midCorrection.present)
                result.midNotImplemented = true;
            else
            {
                for(std::size_t begin = 0u; begin < midIndices.size(); begin += maxBatch)
                {
                    auto const end = std::min(begin + maxBatch, midIndices.size());
                    auto batchSources = makeBatch(midIndices, begin, end);
                    std::vector<float> contributions;
                    if(useMidCorrection)
                    {
                        contributions = pointFieldBatched(queue, exec, batchSources, cells);
                        auto const latent = runEncoderLayers(
                            queue,
                            exec,
                            meta,
                            meta.featureKind,
                            meta.featureDim,
                            meta.midCorrection.layers,
                            tensors,
                            batchSources);
                        auto const correction = decodeLinearBatchedRaw(
                            queue,
                            exec,
                            meta,
                            tensors,
                            meta.midCorrection.decoder,
                            latent,
                            batchSources.size());
                        for(std::size_t i = 0u; i < contributions.size(); ++i)
                            contributions[i] += correction[i];
                    }
                    else if(useModelForMid)
                    {
                        auto const latent = runEncoderLayers(
                            queue,
                            exec,
                            meta,
                            meta.featureKind,
                            meta.featureDim,
                            meta.layers,
                            tensors,
                            batchSources);
                        contributions
                            = decodeBankBatchedRaw(queue, exec, meta, tensors, latent, batchSources.size(), aY, aZ);
                    }
                    else
                    {
                        contributions = pointFieldBatched(queue, exec, batchSources, cells);
                    }
                    accumulate(midIndices, begin, contributions);
                }
            }
            result.bandSeconds[1] = std::chrono::duration<double>(std::chrono::steady_clock::now() - midStart).count();
        }

        if(!farIndices.empty())
        {
            auto const farStart = std::chrono::steady_clock::now();
            for(std::size_t begin = 0u; begin < farIndices.size(); begin += maxBatch)
            {
                auto const end = std::min(begin + maxBatch, farIndices.size());
                auto batchSources = makeBatch(farIndices, begin, end);
                auto contributions = pointFieldBatched(queue, exec, batchSources, cells);
                if(meta.farMultipole.present)
                {
                    auto const coefficients = runEncoderLayers(
                        queue,
                        exec,
                        meta,
                        meta.featureKind,
                        meta.featureDim,
                        meta.farMultipole.layers,
                        tensors,
                        batchSources);
                    auto const taylor = taylorEvaluateBatched(
                        queue,
                        exec,
                        farMatrix,
                        meta.farMultipole.order,
                        coefficients,
                        batchSources.size(),
                        meta.farMultipole.perComponentScale,
                        farBackend);
                    for(std::size_t i = 0u; i < contributions.size(); ++i)
                        contributions[i] += taylor[i];
                }
                accumulate(farIndices, begin, contributions);
            }
            result.bandSeconds[2] = std::chrono::duration<double>(std::chrono::steady_clock::now() - farStart).count();
        }

        for(std::size_t slot = 0u; slot < kOutDim; ++slot)
        {
            auto const component = slot % 6u;
            auto const scale = component < 3u ? normalization.scaleE() : normalization.scaleB();
            result.values[slot] = static_cast<float>(static_cast<double>(result.values[slot]) * scale);
        }
        return result;
    }

} // namespace picongpu::particles::neuralBunch
