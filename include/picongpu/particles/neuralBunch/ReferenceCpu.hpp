/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Pure host reference implementation of the per-tile forward pass.
 *
 * This file exists only for tests and diagnostics. It uses double arithmetic throughout (weights are the float32
 * payload values widened to double) and the same formulas as the NumPy exporter reference. It shares the artifact
 * metadata but none of the device code.
 */

#include "Features.hpp"
#include "Inference.hpp"
#include "Model.hpp"

#include <cmath>
#include <cstddef>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace picongpu::particles::neuralBunch
{
    struct HostArtifact
    {
        ModelMetadata metadata;
        std::map<std::string, std::vector<float>> tensors;
    };

    inline std::vector<float> referenceFeatures(FeatureKind kind, Source const& source)
    {
        std::vector<float> features(featureDimension(kind));
        featuresHost(
            kind,
            features.data(),
            source.r[0],
            source.r[1],
            source.r[2],
            source.u[0],
            source.u[1],
            source.u[2],
            source.aY,
            source.aZ);
        return features;
    }

    inline std::vector<double> referenceEncode(
        HostArtifact const& artifact,
        std::span<Source const> sources,
        std::vector<float> const* featureOverride = nullptr)
    {
        auto const& meta = artifact.metadata;
        auto const count = sources.size();
        std::vector<double> hidden(count * meta.latent, 0.0);
        std::vector<double> x(meta.featureDim, 0.0);
        for(std::size_t n = 0u; n < count; ++n)
        {
            if(featureOverride != nullptr)
            {
                for(std::size_t k = 0u; k < meta.featureDim; ++k)
                    x[k] = static_cast<double>((*featureOverride)[n * meta.featureDim + k]);
            }
            else
            {
                auto const features = referenceFeatures(meta.featureKind, sources[n]);
                for(std::size_t k = 0u; k < meta.featureDim; ++k)
                    x[k] = static_cast<double>(features[k]);
            }

            std::vector<double> current = x;
            for(std::size_t layerIndex = 0u; layerIndex < meta.layers.size(); ++layerIndex)
            {
                auto const& layer = meta.layers[layerIndex];
                auto const& weight = artifact.tensors.at(layer.weight);
                auto const& bias = artifact.tensors.at(layer.bias);
                std::vector<double> next(layer.out, 0.0);
                for(std::size_t out = 0u; out < layer.out; ++out)
                {
                    double sum = static_cast<double>(bias[out]);
                    for(std::size_t in = 0u; in < layer.in; ++in)
                        sum += current[in] * static_cast<double>(weight[in * layer.out + out]);
                    bool const last = layerIndex + 1u == meta.layers.size();
                    next[out] = last ? sum : sum / (1.0 + std::exp(-sum));
                }
                current = std::move(next);
            }
            for(std::size_t k = 0u; k < meta.latent; ++k)
                hidden[n * meta.latent + k] = current[k];
        }
        return hidden;
    }

    inline std::vector<double> referenceAccumulate(
        HostArtifact const& artifact,
        std::span<Source const> sources,
        std::vector<double> const& hidden)
    {
        auto const& meta = artifact.metadata;
        std::vector<double> latent(meta.latent, 0.0);
        for(std::size_t n = 0u; n < sources.size(); ++n)
        {
            double const charge = -sources[n].weight;
            for(std::size_t k = 0u; k < meta.latent; ++k)
                latent[k] += charge * hidden[n * meta.latent + k];
        }
        return latent;
    }

    /** Interpolated decoder matrix, [latent, 1536] row-major. */
    inline std::vector<double> referenceInterpolatedBank(HostArtifact const& artifact, double aY, double aZ)
    {
        auto const& meta = artifact.metadata;
        auto const cell = bilinearCell(meta.aYAnchors, meta.aZAnchors, aY, aZ);
        std::vector<double> matrix(meta.latent * kOutDim, 0.0);
        for(std::size_t entry = 0u; entry < 4u; ++entry)
        {
            auto const& bank = artifact.tensors.at(meta.bank[cell.index[entry]]);
            double const weight = static_cast<double>(cell.weight[entry]);
            for(std::size_t i = 0u; i < matrix.size(); ++i)
                matrix[i] += weight * static_cast<double>(bank[i]);
        }
        return matrix;
    }

    inline std::vector<float> referenceDecodeNormalized(
        HostArtifact const& artifact,
        std::vector<double> const& latent,
        double aY,
        double aZ,
        FieldNormalization const& normalization = {})
    {
        auto const& meta = artifact.metadata;
        auto const matrix = referenceInterpolatedBank(artifact, aY, aZ);
        std::vector<float> result(kOutDim, 0.0f);
        for(std::size_t out = 0u; out < kOutDim; ++out)
        {
            double sum = 0.0;
            for(std::size_t k = 0u; k < meta.latent; ++k)
                sum += latent[k] * matrix[k * kOutDim + out];
            auto const component = out % 6u;
            auto const scale = component < 3u ? normalization.scaleE() : normalization.scaleB();
            result[out] = static_cast<float>(sum * scale);
        }
        return result;
    }

    inline std::vector<float> referenceEvaluate(
        HostArtifact const& artifact,
        std::span<Source const> sources,
        double aY,
        double aZ,
        FieldNormalization const& normalization = {})
    {
        if(sources.empty())
            return std::vector<float>(kOutDim, 0.0f);
        auto const hidden = referenceEncode(artifact, sources);
        auto const latent = referenceAccumulate(artifact, sources, hidden);
        return referenceDecodeNormalized(artifact, latent, aY, aZ, normalization);
    }

    inline std::vector<double> referenceEncodeLayers(
        HostArtifact const& artifact,
        std::vector<EncoderLayer> const& layers,
        FeatureKind featureKind,
        std::size_t featureDim,
        std::span<Source const> sources)
    {
        auto const count = sources.size();
        auto const outDim = layers.empty() ? std::size_t{0} : layers.back().out;
        std::vector<double> hidden(count * outDim, 0.0);
        for(std::size_t n = 0u; n < count; ++n)
        {
            auto const features = referenceFeatures(featureKind, sources[n]);
            std::vector<double> current(featureDim, 0.0);
            for(std::size_t k = 0u; k < featureDim; ++k)
                current[k] = static_cast<double>(features[k]);
            for(std::size_t layerIndex = 0u; layerIndex < layers.size(); ++layerIndex)
            {
                auto const& layer = layers[layerIndex];
                auto const& weight = artifact.tensors.at(layer.weight);
                auto const& bias = artifact.tensors.at(layer.bias);
                std::vector<double> next(layer.out, 0.0);
                for(std::size_t out = 0u; out < layer.out; ++out)
                {
                    double sum = static_cast<double>(bias[out]);
                    for(std::size_t in = 0u; in < layer.in; ++in)
                        sum += current[in] * static_cast<double>(weight[in * layer.out + out]);
                    bool const last = layerIndex + 1u == layers.size();
                    next[out] = last ? sum : sum / (1.0 + std::exp(-sum));
                }
                current = std::move(next);
            }
            for(std::size_t k = 0u; k < outDim; ++k)
                hidden[n * outDim + k] = current[k];
        }
        return hidden;
    }

    inline std::vector<double> referenceLinearDecode(
        HostArtifact const& artifact,
        LinearDecoderSpec const& decoder,
        std::vector<double> const& latent)
    {
        auto const& weight = artifact.tensors.at(decoder.weight);
        std::vector<double> out(decoder.out, 0.0);
        std::vector<float> const* bias = nullptr;
        if(decoder.hasBias)
            bias = &artifact.tensors.at(decoder.bias);
        for(std::size_t o = 0u; o < decoder.out; ++o)
        {
            double sum = bias != nullptr ? static_cast<double>((*bias)[o]) : 0.0;
            for(std::size_t i = 0u; i < decoder.in; ++i)
                sum += latent[i] * static_cast<double>(weight[i * decoder.out + o]);
            out[o] = sum;
        }
        return out;
    }

    inline BandedTileResult referenceBandedTile(
        HostArtifact const& artifact,
        std::span<Source const> sources,
        BandPolicy const& policy,
        NearTileEvaluator nearEvaluator,
        double aY,
        double aZ,
        FieldNormalization const& normalization = {},
        std::size_t maxBatch = 64u)
    {
        auto const& meta = artifact.metadata;
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

        std::vector<double> perSource(sources.size() * kOutDim, 0.0);

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
            for(std::size_t slot = 0u; slot < kOutDim; ++slot)
                perSource[index * kOutDim + slot] = static_cast<double>((*tile)[slot]);
        }

        for(std::size_t begin = 0u; begin < midIndices.size(); begin += maxBatch)
        {
            auto const end = std::min(begin + maxBatch, midIndices.size());
            std::vector<Source> batchSources;
            for(std::size_t i = begin; i < end; ++i)
                batchSources.push_back(sources[midIndices[i]]);
            std::span<Source const> batch{batchSources};
            std::vector<double> latent;
            if(meta.midCorrection.present)
                latent = referenceEncodeLayers(
                    artifact,
                    meta.midCorrection.layers,
                    meta.featureKind,
                    meta.featureDim,
                    batch);
            auto const latentDim = meta.midCorrection.present ? meta.midCorrection.latent : std::size_t{0};
            for(std::size_t b = 0u; b < batchSources.size(); ++b)
            {
                auto const& source = batchSources[b];
                auto const field = pointFieldTileHost(
                    Vec3d{source.r[0], source.r[1], source.r[2]},
                    Vec3d{source.u[0], source.u[1], source.u[2]},
                    cells);
                for(std::size_t slot = 0u; slot < kOutDim; ++slot)
                    perSource[midIndices[begin + b] * kOutDim + slot] = static_cast<double>(field[slot]);
                if(meta.midCorrection.present)
                {
                    std::vector<double> sourceLatent(
                        latent.begin() + static_cast<std::ptrdiff_t>(b * latentDim),
                        latent.begin() + static_cast<std::ptrdiff_t>((b + 1u) * latentDim));
                    auto const correction = referenceLinearDecode(artifact, meta.midCorrection.decoder, sourceLatent);
                    for(std::size_t slot = 0u; slot < kOutDim; ++slot)
                        perSource[midIndices[begin + b] * kOutDim + slot] += correction[slot];
                }
            }
        }

        for(std::size_t begin = 0u; begin < farIndices.size(); begin += maxBatch)
        {
            auto const end = std::min(begin + maxBatch, farIndices.size());
            std::vector<Source> batchSources;
            for(std::size_t i = begin; i < end; ++i)
                batchSources.push_back(sources[farIndices[i]]);
            std::span<Source const> batch{batchSources};

            if(meta.farMultipole.present)
            {
                auto const coefficients = referenceEncodeLayers(
                    artifact,
                    meta.farMultipole.layers,
                    meta.featureKind,
                    meta.featureDim,
                    batch);
                auto const K = meta.farMultipole.coefficientDim;
                Vec3d const farCells{1.0, meta.farMultipole.aspectY, meta.farMultipole.aspectZ};
                auto const matrix = taylorMatrixHost(meta.farMultipole.order, farCells, meta.farMultipole.origin);
                for(std::size_t b = 0u; b < batchSources.size(); ++b)
                {
                    std::vector<double> sourceCoefficients(
                        coefficients.begin() + static_cast<std::ptrdiff_t>(b * K),
                        coefficients.begin() + static_cast<std::ptrdiff_t>((b + 1u) * K));
                    auto const values = taylorEvaluateHost(
                        matrix,
                        meta.farMultipole.order,
                        sourceCoefficients,
                        meta.farMultipole.perComponentScale);
                    for(std::size_t slot = 0u; slot < kOutDim; ++slot)
                        perSource[farIndices[begin + b] * kOutDim + slot] = values[slot];
                }
            }
            else if(!meta.layers.empty() && !meta.bank.empty())
            {
                auto const hidden
                    = referenceEncodeLayers(artifact, meta.layers, meta.featureKind, meta.featureDim, batch);
                auto const matrix = referenceInterpolatedBank(artifact, aY, aZ);
                for(std::size_t b = 0u; b < batchSources.size(); ++b)
                {
                    auto const row = farIndices[begin + b] * kOutDim;
                    for(std::size_t slot = 0u; slot < kOutDim; ++slot)
                    {
                        double sum = 0.0;
                        for(std::size_t k = 0u; k < meta.latent; ++k)
                            sum += hidden[b * meta.latent + k] * matrix[k * kOutDim + slot];
                        perSource[row + slot] = sum;
                    }
                }
            }
            else
            {
                for(std::size_t b = 0u; b < batchSources.size(); ++b)
                {
                    auto const& source = batchSources[b];
                    auto const field = pointFieldTileHost(
                        Vec3d{source.r[0], source.r[1], source.r[2]},
                        Vec3d{source.u[0], source.u[1], source.u[2]},
                        cells);
                    for(std::size_t slot = 0u; slot < kOutDim; ++slot)
                        perSource[farIndices[begin + b] * kOutDim + slot] = static_cast<double>(field[slot]);
                }
            }
        }

        std::vector<double> accumulated(kOutDim, 0.0);
        for(std::size_t index = 0u; index < sources.size(); ++index)
        {
            double const charge = -sources[index].weight;
            if(charge == 0.0)
                continue;
            for(std::size_t slot = 0u; slot < kOutDim; ++slot)
                accumulated[slot] += charge * perSource[index * kOutDim + slot];
        }
        for(std::size_t slot = 0u; slot < kOutDim; ++slot)
        {
            auto const component = slot % 6u;
            auto const scale = component < 3u ? normalization.scaleE() : normalization.scaleB();
            result.values[slot] = static_cast<float>(accumulated[slot] * scale);
        }
        return result;
    }
} // namespace picongpu::particles::neuralBunch
