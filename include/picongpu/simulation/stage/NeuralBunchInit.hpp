/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "picongpu/defines.hpp"

#include <cstdint>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Neural bunch initializer (Phase A: plumbing and mechanism).
             *
             * Called from ``Simulation::fillSimulation()`` in the non-restart
             * branch after ``ParticleInit`` and before the final field guard
             * exchange. Pass 1 evaluates the neural field model on rank-local
             * tiles and adds all six field components to FieldE/FieldB in
             * internal units. Pass 2 re-reads the source and inserts the
             * particles that belong to this rank's half-open local domain.
             *
             * Restart is handled by the existing ``fillSimulation`` branch: the
             * stage is not called on restart, so fields and particles are never
             * added twice.
             */
            struct NeuralBunchInit
            {
                void operator()(uint32_t const step, MappingDesc const& cellDescription) const;
            };
        } // namespace stage
    } // namespace simulation
} // namespace picongpu
