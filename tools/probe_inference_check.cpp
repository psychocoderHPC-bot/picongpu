/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 *
 * Evaluate one tile of an artifact through the PIConGPU inference path and
 * print the 1536 values as JSON for comparison with the NumPy forward.
 *
 * Usage: probe_inference_check <artifact-prefix> <sources.json> <a_y> <a_z> [host|cuda]
 */

#include "picongpu/particles/neuralBunch/Backend.hpp"
#include "picongpu/particles/neuralBunch/Inference.hpp"
#include "picongpu/particles/neuralBunch/Model.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace picongpu::particles::neuralBunch;

int main(int argc, char** argv)
{
    if(argc < 5)
    {
        std::fprintf(stderr, "usage: %s artifact sources.json a_y a_z [host|cuda]\n", argv[0]);
        return 2;
    }
    std::string const backend = argc > 5 ? argv[5] : "cuda";
    double const aY = std::stod(argv[3]);
    double const aZ = std::stod(argv[4]);

    nlohmann::json sourcesJson;
    {
        std::ifstream stream{argv[2]};
        stream >> sourcesJson;
    }
    std::vector<Source> sources;
    for(auto const& entry : sourcesJson)
    {
        Source source;
        for(int axis = 0; axis < 3; ++axis)
        {
            source.r[axis] = entry.at("r").at(axis).get<double>();
            source.u[axis] = entry.at("u").at(axis).get<double>();
        }
        source.weight = entry.at("w").get<double>();
        source.aY = aY;
        source.aZ = aZ;
        sources.push_back(source);
    }

    try
    {
        auto const model = Model::load(argv[1]);
        auto const& meta = model.metadata();
        std::vector<float> actual;
        dispatchBackend(
            backend,
            [&](auto& device, auto exec)
            {
                auto queue = device.makeQueue();
                auto const tensors = model.loadTensors(queue);
                auto const hidden = runEncoder(queue, exec, meta, tensors, sources);
                std::vector<float> latent(meta.latent, 0.0f);
                for(std::size_t i = 0; i < sources.size(); ++i)
                    for(std::size_t k = 0; k < meta.latent; ++k)
                        latent[k] -= static_cast<float>(sources[i].weight)
                                     * hidden[i * meta.latent + k];
                actual = decodeFromLatent(queue, exec, meta, tensors, latent, aY, aZ);
            });
        std::printf("{\"values\": [");
        for(std::size_t i = 0; i < actual.size(); ++i)
            std::printf("%s%.17g", i == 0 ? "" : ", ", static_cast<double>(actual[i]));
        std::printf("], \"dim\": %zu}\n", meta.featureDim);
        return 0;
    }
    catch(std::exception const& error)
    {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 1;
    }
}
