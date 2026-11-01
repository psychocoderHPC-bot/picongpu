/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include "picongpu/particles/neuralBunch/ReferenceTile.hpp"

#if defined(PIC_ENABLE_NEURAL_BUNCH) && PIC_ENABLE_NEURAL_BUNCH

#    include <omp.h>

#    include <algorithm>
#    include <cstddef>
#    include <vector>

namespace picongpu::particles::neuralBunch::reference
{
    std::vector<double> tileFieldParallel(
        Vec3 const& origin,
        std::vector<Source> const& sources,
        double dx,
        double dy,
        double dz,
        TileReport* report)
    {
        constexpr int tileNx = 8;
        constexpr int tileNy = 8;
        constexpr int tileNz = 4;
        constexpr int outputDimension = tileNx * tileNy * tileNz * 6;
        std::array<std::array<double, 3>, 6> const componentOffsets{{
            {{0.5, 0.0, 0.0}},
            {{0.0, 0.5, 0.0}},
            {{0.0, 0.0, 0.5}},
            {{0.0, 0.5, 0.5}},
            {{0.5, 0.0, 0.5}},
            {{0.5, 0.5, 0.0}},
        }};

        int const threads = std::max(1, omp_get_max_threads());
        std::vector<std::vector<double>> partial(
            static_cast<std::size_t>(threads),
            std::vector<double>(outputDimension, 0.0));
        std::vector<TileReport> reports(static_cast<std::size_t>(threads));

#pragma omp parallel
        {
            int const thread = omp_get_thread_num();
            auto& local = partial[static_cast<std::size_t>(thread)];
            auto& localReport = reports[static_cast<std::size_t>(thread)];
#pragma omp for schedule(static)
            for(int flat = 0; flat < outputDimension; ++flat)
            {
                int const component = flat % 6;
                int const cell = flat / 6;
                int const i = cell % tileNx;
                int const j = (cell / tileNx) % tileNy;
                int const k = cell / (tileNx * tileNy);
                Vec3 const position{
                    origin.x + (i + componentOffsets[component][0]) * dx,
                    origin.y + (j + componentOffsets[component][1]) * dy,
                    origin.z + (k + componentOffsets[component][2]) * dz};
                double sum = 0.0;
                for(auto const& source : sources)
                {
                    if(source.w == 0.0)
                        continue;
                    Vec3 E, B;
                    ShapeResult result;
                    shape_field(position, source.x, source.u, dx, dy, dz, Settings{}, E, B, &result);
                    double const value = component < 3 ? E[component] : B[component - 3];
                    sum += -source.w * dx * dx * value;
                    localReport.max_error = std::max(localReport.max_error, result.error);
                    localReport.all_converged = localReport.all_converged && result.converged;
                    localReport.any_near = localReport.any_near || result.near;
                }
                local[static_cast<std::size_t>(flat)] = sum;
            }
        }

        std::vector<double> tile(outputDimension, 0.0);
        TileReport combined;
        for(int thread = 0; thread < threads; ++thread)
        {
            for(int flat = 0; flat < outputDimension; ++flat)
                tile[static_cast<std::size_t>(flat)]
                    += partial[static_cast<std::size_t>(thread)][static_cast<std::size_t>(flat)];
            combined.max_error = std::max(combined.max_error, reports[static_cast<std::size_t>(thread)].max_error);
            combined.all_converged
                = combined.all_converged && reports[static_cast<std::size_t>(thread)].all_converged;
            combined.any_near = combined.any_near || reports[static_cast<std::size_t>(thread)].any_near;
        }
        if(report)
            *report = combined;
        return tile;
    }
} // namespace picongpu::particles::neuralBunch::reference

#endif
