/* Copyright 2013-2024 Axel Huebl, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/traits/attribute/GetMass.hpp"

namespace picongpu
{
    namespace particles
    {
        namespace manipulators
        {
            namespace unary
            {
                namespace acc
                {
                    /** manipulate the speed
                     *
                     * @tparam T_ParamClass picongpu::particles::manipulators::unary::param::DriftCfg,
                     *                      type with compile configuration
                     * @tparam T_ValueFunctor pmacc::math::operation, binary operator type to reduce current and new
                     * value
                     */
                    template<typename T_ParamClass, typename T_ValueFunctor>
                    struct Drift : private T_ValueFunctor
                    {
                        /** manipulate the speed of the particle
                         *
                         * @tparam T_Particle pmacc::Particle, particle type
                         *
                         * @param particle particle to be manipulated
                         */
                        template<typename T_Particle>
                        HDINLINE void operator()(T_Particle& particle)
                        {
                            using ParamClass = T_ParamClass;
                            using ValueFunctor = T_ValueFunctor;

                            float_X const macroWeighting = particle[weighting_];
                            float_X const macroMass = picongpu::traits::attribute::getMass(macroWeighting, particle);

                            // The result is stored in float_X, so evaluate in float_X as
                            // well. This keeps the kernel free of double arithmetic on
                            // devices without fp64 support.
                            float_X const myGamma = precisionCast<float_X>(ParamClass::gamma);

                            float_X const initFreeBeta
                                = math::sqrt(float_X(1.0) - float_X(1.0) / (myGamma * myGamma));

                            float3_X const driftDirection = ParamClass::driftDirection;
                            float3_X const normDir = driftDirection / pmacc::math::l2norm(driftDirection);

                            float3_X const mom(
                                normDir * (myGamma * initFreeBeta * macroMass * sim.pic.getSpeedOfLight()));

                            ValueFunctor::operator()(particle[momentum_], mom);
                        }
                    };

                } // namespace acc
            } // namespace unary
        } // namespace manipulators
    } // namespace particles
} // namespace picongpu
