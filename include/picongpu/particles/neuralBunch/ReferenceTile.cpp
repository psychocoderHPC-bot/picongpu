/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#include "picongpu/particles/neuralBunch/ReferenceTile.hpp"

#if defined(PIC_ENABLE_NEURAL_BUNCH) && PIC_ENABLE_NEURAL_BUNCH

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
        // The portable kernel already parallelises independent (source, slot)
        // pairs with OpenMP and reduces in source order, so the result is
        // bitwise independent of the thread count. This translation unit is
        // host-only, which keeps the OpenMP pragmas out of device code.
        return tile_field(origin, sources, dx, dy, dz, Settings{}, report);
    }
} // namespace picongpu::particles::neuralBunch::reference

#endif
