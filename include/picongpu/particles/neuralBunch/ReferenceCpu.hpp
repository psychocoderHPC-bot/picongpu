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

    inline std::vector<float> referenceFeatures(Source const& source)
    {
        std::vector<float> features(kFeatureDim);
        fourierV1Host(
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
                auto const features = referenceFeatures(sources[n]);
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
    inline std::vector<double> referenceInterpolatedBank(
        HostArtifact const& artifact,
        double aY,
        double aZ)
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
} // namespace picongpu::particles::neuralBunch
