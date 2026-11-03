/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Host-only OpenMP wrapper around the validated FP64 reference kernel.
 *
 * The kernel itself is serial over the 1536 output points. This wrapper keeps
 * the kernel arithmetic unchanged and parallelises the independent output
 * points of one tile, which makes the near-support fallback tractable.
 */

#include "picongpu/particles/neuralBunch/ReferenceKernel.hpp"

#include <vector>

namespace picongpu::particles::neuralBunch::reference
{
    std::vector<double> tileFieldParallel(
        Vec3 const& origin,
        std::vector<Source> const& sources,
        double dx,
        double dy,
        double dz,
        TileReport* report = nullptr);
} // namespace picongpu::particles::neuralBunch::reference
