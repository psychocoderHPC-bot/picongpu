/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Device-resident banded tile evaluation.
 *
 * This is the production counterpart of ``evaluateTileBanded``. It keeps every intermediate tensor on the
 * accelerator: the encoder writes ``hidden`` on the device, the per-band contributions are decoded with
 * ``gemvBatched``/``gemm`` on the device and accumulated directly into a caller-owned 1536-element tile
 * accumulator with ``ops::weightedAdd``. No function in this header copies a device result back to the host or
 * waits for the queue. The caller synchronises once, after all tiles and all bands have been enqueued.
 *
 * Accumulation order is fixed: bands are visited near, mid, far; inside a band the source batches keep the
 * original source order; ``weightedAdd`` sums the batch rows in ascending order. The result is therefore
 * bitwise deterministic for a fixed chunking and batch size.
 *
 * Only the source coordinates (openPMD input) are uploaded from the host. That upload is I/O, not a device
 * round-trip.
 */

#include "Inference.hpp"
#include "PointField.hpp"
#include "TaylorField.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/matrix/gemv.hpp>
#include <alpaka/nn/onHost/ops/elementwise.hpp>
#include <alpaka/nn/onHost/ops/weighted_add.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace picongpu::particles::neuralBunch
{
    namespace deviceDetail
    {
        /** Upload the (r,u) geometry of a source batch as a [S,6] device array. */
        template<typename TQueue>
        auto packSourceGeometry(TQueue& queue, std::span<Source const> sources)
        {
            auto host = alpaka::onHost::allocHost<double>(alpaka::Vec{sources.size(), std::size_t{6}});
            for(std::size_t i = 0u; i < sources.size(); ++i)
            {
                for(std::size_t axis = 0u; axis < 3u; ++axis)
                {
                    host[alpaka::Vec<std::size_t, 2>{i, axis}] = sources[i].r[axis];
                    host[alpaka::Vec<std::size_t, 2>{i, axis + 3u}] = sources[i].u[axis];
                }
            }
            auto device = alpaka::onHost::allocLike(queue.getDevice(), host);
            alpaka::onHost::memcpy(queue, device, host);
            host.keepAlive(queue);
            device.keepAlive(queue);
            return device;
        }

        /** Upload the model charge weighting q/e = -weight for a source batch. */
        template<typename TQueue>
        auto packNegativeWeights(TQueue& queue, std::span<Source const> sources)
        {
            auto host = alpaka::onHost::allocHost<float>(alpaka::Vec{sources.size()});
            for(std::size_t i = 0u; i < sources.size(); ++i)
                host[alpaka::Vec<std::size_t, 1>{i}] = static_cast<float>(-sources[i].weight);
            auto device = alpaka::onHost::allocLike(queue.getDevice(), host);
            alpaka::onHost::memcpy(queue, device, host);
            host.keepAlive(queue);
            device.keepAlive(queue);
            return device;
        }

        /** Build a [1, N] device view over a 1-D accumulator slice. */
        template<typename TQueue, typename TBuffer>
        auto rowView(TQueue& queue, TBuffer& buffer, std::size_t offset, std::size_t columns)
        {
            return alpaka::makeView(
                queue.getDevice(),
                alpaka::onHost::data(buffer) + offset,
                alpaka::Vec<std::size_t, 2>{std::size_t{1}, columns});
        }

        /** Interpolate the decoder bank and apply it to a device latent batch: [B,latent] -> [B,1536]. */
        template<typename TQueue, typename TExec, typename TBuffer, typename TLatent, typename TOut>
        void enqueueDecodeBankBatchedRaw(
            TQueue& queue,
            TExec exec,
            ModelMetadata const& meta,
            std::vector<TBuffer> const& tensors,
            TLatent const& deviceLatent,
            double aY,
            double aZ,
            TOut& out)
        {
            auto const cell = bilinearCell(meta.aYAnchors, meta.aZAnchors, aY, aZ);
            auto bankView = [&](std::size_t index)
            {
                auto const& buffer = tensors[meta.tensorIndex(meta.bank[index])];
                return alpaka::makeView(
                    queue.getDevice(),
                    alpaka::onHost::data(buffer),
                    alpaka::Vec{meta.latent, kOutDim});
            };
            auto interpolated = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{meta.latent, kOutDim});
            detail::InterpolateDecoderKernel const interp{
                cell.weight[0],
                cell.weight[1],
                cell.weight[2],
                cell.weight[3]};
            queue.enqueue(
                detail::makeFrameSpec(meta.latent * kOutDim, 256u, exec),
                alpaka::KernelBundle{
                    interp,
                    bankView(cell.index[0]),
                    bankView(cell.index[1]),
                    bankView(cell.index[2]),
                    bankView(cell.index[3]),
                    interpolated});
            interpolated.keepAlive(queue);
            alpaka::nn::onHost::gemvBatched<float>(queue, deviceLatent, interpolated, out);
        }

        /** Apply an explicit linear decoder to a device latent batch: [B,in] -> [B,out]. */
        template<typename TQueue, typename TExec, typename TBuffer, typename TLatent, typename TOut>
        void enqueueDecodeLinearBatchedRaw(
            TQueue& queue,
            TExec exec,
            ModelMetadata const& meta,
            std::vector<TBuffer> const& tensors,
            LinearDecoderSpec const& decoder,
            TLatent const& deviceLatent,
            TOut& out)
        {
            auto const& weightBuffer = tensors[meta.tensorIndex(decoder.weight)];
            auto weight = alpaka::makeView(
                queue.getDevice(),
                alpaka::onHost::data(weightBuffer),
                alpaka::Vec{decoder.in, decoder.out});
            alpaka::nn::onHost::gemvBatched<float>(queue, deviceLatent, weight, out);
            if(decoder.hasBias)
            {
                auto const& biasBuffer = tensors[meta.tensorIndex(decoder.bias)];
                auto bias
                    = alpaka::makeView(queue.getDevice(), alpaka::onHost::data(biasBuffer), alpaka::Vec{decoder.out});
                alpaka::nn::onHost::ops::biasAdd(queue, exec, out, bias, out);
            }
        }

        /** out[slot] += sum_s weights[s] * contributions[s, slot], deterministic ascending row order. */
        template<typename TQueue, typename TExec, typename TContrib, typename TWeights, typename TOut>
        void accumulate(TQueue& queue, TExec exec, TContrib& contributions, TWeights& weights, TOut& out)
        {
            alpaka::nn::onHost::ops::weightedAdd<float>(queue, exec, contributions, weights, out);
        }
    } // namespace deviceDetail

    /** Device-resident three-band evaluation result (no host data). */
    struct DeviceBandedResult
    {
        std::array<std::size_t, 3> sourceCounts{}; //!< near, mid, far (zero-weight sources excluded)
        std::array<double, 3> bandSeconds{}; //!< host enqueue time per band
        bool nearNotImplemented = false;
        bool midNotImplemented = false;
    };

    /** Enqueue the banded evaluation of one tile into a device accumulator.
     *
     * @param target a 1-D device view of 1536 floats; contributions are added to it in place. The buffer is
     *        not normalised here; the caller applies the single E/B normalisation in its store kernel.
     * @param maxBatch maximum number of sources per encoder/decoder batch (bounded scratch).
     *
     * Nothing is copied to the host and no queue wait is performed.
     */
    template<typename TQueue, typename TExec, typename TBuffer, typename TTarget>
    DeviceBandedResult enqueueBandedTileDevice(
        TQueue& queue,
        TExec exec,
        ModelMetadata const& meta,
        std::vector<TBuffer> const& tensors,
        std::span<Source const> sources,
        BandPolicy const& policy,
        double aY,
        double aZ,
        TTarget& target,
        std::size_t maxBatch = 64u,
        TaylorBatchBackend farBackend = TaylorBatchBackend::automatic,
        bool usePrimaryModelForNear = false,
        bool useModelForMid = false,
        bool useMidCorrection = false)
    {
        if(maxBatch == 0u)
            throw std::invalid_argument{"enqueueBandedTileDevice: maxBatch must be positive."};
        if(farBackend == TaylorBatchBackend::automatic)
            farBackend = defaultTaylorBatchBackend(queue.getDevice());

        DeviceBandedResult result;
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

        auto batchSources = [&](std::vector<std::size_t> const& indices, std::size_t begin, std::size_t end)
        {
            std::vector<Source> batch;
            batch.reserve(end - begin);
            for(std::size_t i = begin; i < end; ++i)
                batch.push_back(sources[indices[i]]);
            return batch;
        };

        auto addPointField = [&](std::vector<std::size_t> const& indices, std::size_t begin, std::size_t end)
        {
            auto batch = batchSources(indices, begin, end);
            auto deviceSources = deviceDetail::packSourceGeometry(queue, batch);
            auto contributions = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{batch.size(), kOutDim});
            enqueuePointFieldBatched(queue, exec, deviceSources, cells, contributions);
            deviceSources.keepAlive(queue);
            contributions.keepAlive(queue);
            auto weights = deviceDetail::packNegativeWeights(queue, batch);
            deviceDetail::accumulate(queue, exec, contributions, weights, target);
        };

        auto addModelBank = [&](std::vector<std::size_t> const& indices, std::size_t begin, std::size_t end)
        {
            auto batch = batchSources(indices, begin, end);
            auto hidden
                = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{batch.size(), std::size_t(meta.latent)});
            detail::enqueueEncodeLayers(
                queue,
                exec,
                meta,
                meta.featureKind,
                meta.featureDim,
                meta.layers,
                tensors,
                std::span<Source const>{batch},
                hidden);
            auto contributions = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{batch.size(), kOutDim});
            deviceDetail::enqueueDecodeBankBatchedRaw(queue, exec, meta, tensors, hidden, aY, aZ, contributions);
            hidden.keepAlive(queue);
            contributions.keepAlive(queue);
            auto weights = deviceDetail::packNegativeWeights(queue, batch);
            deviceDetail::accumulate(queue, exec, contributions, weights, target);
        };

        auto addNearModel = [&](std::vector<std::size_t> const& indices, std::size_t begin, std::size_t end)
        {
            auto batch = batchSources(indices, begin, end);
            auto hidden = alpaka::onHost::alloc<float>(
                queue.getDevice(),
                alpaka::Vec{batch.size(), meta.nearModel.layers.back().out});
            detail::enqueueEncodeLayers(
                queue,
                exec,
                meta,
                meta.featureKind,
                meta.featureDim,
                meta.nearModel.layers,
                tensors,
                std::span<Source const>{batch},
                hidden);
            auto contributions = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{batch.size(), kOutDim});
            deviceDetail::enqueueDecodeLinearBatchedRaw(
                queue,
                exec,
                meta,
                tensors,
                meta.nearModel.decoder,
                hidden,
                contributions);
            hidden.keepAlive(queue);
            contributions.keepAlive(queue);
            auto weights = deviceDetail::packNegativeWeights(queue, batch);
            deviceDetail::accumulate(queue, exec, contributions, weights, target);
        };

        auto addMidCorrection = [&](std::vector<std::size_t> const& indices, std::size_t begin, std::size_t end)
        {
            auto batch = batchSources(indices, begin, end);
            auto hidden = alpaka::onHost::alloc<float>(
                queue.getDevice(),
                alpaka::Vec{batch.size(), meta.midCorrection.layers.back().out});
            detail::enqueueEncodeLayers(
                queue,
                exec,
                meta,
                meta.featureKind,
                meta.featureDim,
                meta.midCorrection.layers,
                tensors,
                std::span<Source const>{batch},
                hidden);
            auto contributions = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{batch.size(), kOutDim});
            deviceDetail::enqueueDecodeLinearBatchedRaw(
                queue,
                exec,
                meta,
                tensors,
                meta.midCorrection.decoder,
                hidden,
                contributions);
            hidden.keepAlive(queue);
            contributions.keepAlive(queue);
            auto weights = deviceDetail::packNegativeWeights(queue, batch);
            deviceDetail::accumulate(queue, exec, contributions, weights, target);
        };

        // near: optional near head, then the primary encoder/decoder, otherwise mark incomplete.
        if(!nearIndices.empty())
        {
            auto const nearStart = std::chrono::steady_clock::now();
            if(meta.nearModel.present)
            {
                for(std::size_t begin = 0u; begin < nearIndices.size(); begin += maxBatch)
                    addNearModel(nearIndices, begin, std::min(begin + maxBatch, nearIndices.size()));
            }
            else if(usePrimaryModelForNear && !meta.layers.empty() && !meta.bank.empty())
            {
                for(std::size_t begin = 0u; begin < nearIndices.size(); begin += maxBatch)
                    addModelBank(nearIndices, begin, std::min(begin + maxBatch, nearIndices.size()));
            }
            else
                result.nearNotImplemented = true;
            result.bandSeconds[0]
                = std::chrono::duration<double>(std::chrono::steady_clock::now() - nearStart).count();
        }

        // Precompute and upload the far-band evaluation matrix once per tile when requested.
        std::size_t const farK = meta.farMultipole.present ? taylorBasisSize(meta.farMultipole.order) : 0u;
        auto farMatrixDevice
            = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{std::max<std::size_t>(1u, farK), kOutDim});
        if(meta.farMultipole.present)
        {
            Vec3d const farCells{1.0, meta.farMultipole.aspectY, meta.farMultipole.aspectZ};
            auto const matrix = taylorMatrixHost(meta.farMultipole.order, farCells, meta.farMultipole.origin);
            auto const transposed = taylorMatrixTransposedHost(matrix, kOutDim, farK);
            auto hostMatrix = alpaka::onHost::allocHost<float>(alpaka::Vec{farK, kOutDim});
            for(std::size_t k = 0u; k < farK; ++k)
                for(std::size_t slot = 0u; slot < kOutDim; ++slot)
                {
                    double value = transposed[k * kOutDim + slot];
                    if(!meta.farMultipole.perComponentScale.empty())
                        value *= meta.farMultipole.perComponentScale[slot % 6u];
                    hostMatrix[alpaka::Vec<std::size_t, 2>{k, slot}] = static_cast<float>(value);
                }
            alpaka::onHost::memcpy(queue, farMatrixDevice, hostMatrix);
            hostMatrix.keepAlive(queue);
            farMatrixDevice.keepAlive(queue);
        }

        auto addFar = [&](std::vector<std::size_t> const& indices, std::size_t begin, std::size_t end)
        {
            auto batch = batchSources(indices, begin, end);
            addPointField(indices, begin, end);
            if(meta.farMultipole.present)
            {
                auto coefficients = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{batch.size(), farK});
                detail::enqueueEncodeLayers(
                    queue,
                    exec,
                    meta,
                    meta.featureKind,
                    meta.featureDim,
                    meta.farMultipole.layers,
                    tensors,
                    std::span<Source const>{batch},
                    coefficients);
                auto contributions
                    = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{batch.size(), kOutDim});
                if(farBackend == TaylorBatchBackend::native)
                    taylorBatchedNative<float>(queue, exec, farMatrixDevice, coefficients, contributions);
                else
                    taylorBatchedVendor<float>(queue, exec, farMatrixDevice, coefficients, contributions);
                coefficients.keepAlive(queue);
                contributions.keepAlive(queue);
                auto weights = deviceDetail::packNegativeWeights(queue, batch);
                deviceDetail::accumulate(queue, exec, contributions, weights, target);
            }
        };

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
                    if(useMidCorrection)
                    {
                        addPointField(midIndices, begin, end);
                        addMidCorrection(midIndices, begin, end);
                    }
                    else if(useModelForMid)
                        addModelBank(midIndices, begin, end);
                    else
                        addPointField(midIndices, begin, end);
                }
            }
            result.bandSeconds[1] = std::chrono::duration<double>(std::chrono::steady_clock::now() - midStart).count();
        }

        if(!farIndices.empty())
        {
            auto const farStart = std::chrono::steady_clock::now();
            for(std::size_t begin = 0u; begin < farIndices.size(); begin += maxBatch)
                addFar(farIndices, begin, std::min(begin + maxBatch, farIndices.size()));
            result.bandSeconds[2] = std::chrono::duration<double>(std::chrono::steady_clock::now() - farStart).count();
        }
        return result;
    }

} // namespace picongpu::particles::neuralBunch
