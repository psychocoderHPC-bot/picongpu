/* Copyright 2015-2024 Marco Garten, Jakob Trojok, Brian Marre
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
#include "picongpu/particles/ionization/byField/IonizationCurrent/IonizerReturn.hpp"
#include "picongpu/particles/ionization/utilities.hpp"
#include "picongpu/particles/traits/GetAtomicNumbers.hpp"
#include "picongpu/particles/traits/GetIonizationEnergies.hpp"
#include "picongpu/traits/attribute/GetChargeState.hpp"

#include <pmacc/algorithms/math/defines/pi.hpp>
#include <pmacc/algorithms/math/floatMath/floatingPoint.tpp>
#include <pmacc/types.hpp>

/** \file AlgorithmADK.hpp
 *
 * IONIZATION ALGORITHM for the ADK model
 *
 * - implements the calculation of ionization probability and changes charge
 *   states by decreasing the number of bound electrons
 * - is called with the IONIZATION MODEL, specifically by setting the flag in
 *   speciesDefinition.param
 */


namespace picongpu
{
    namespace particles
    {
        namespace ionization
        {
            /** Calculation for the Ammosov-Delone-Krainov tunneling model
             *
             * for either linear or circular laser polarization
             *
             * @tparam T_linPol boolean value that is true for lin. pol. and false for circ. pol.
             */
            template<bool T_linPol>
            struct AlgorithmADK
            {
                /** Functor implementation
                 * @tparam EType type of electric field
                 * @tparam BType type of magnetic field
                 * @tparam ParticleType type of particle to be ionized
                 *
                 * @param bField magnetic field value at t=0
                 * @param eField electric field value at t=0
                 * @param parentIon particle instance to be ionized with position at t=0 and momentum at t=-1/2
                 * @param randNr random number, equally distributed in range [0.:1.0]
                 *
                 * @return ionization energy and number of new macro electrons to be created
                 */
                template<typename EType, typename BType, typename ParticleType>
                HDINLINE IonizerReturn
                operator()(BType const bField, EType const eField, ParticleType& parentIon, float_X randNr)
                {
                    float_X const protonNumber
                        = picongpu::traits::GetAtomicNumbers<ParticleType>::type::numberOfProtons;
                    float_X const chargeState = picongpu::traits::attribute::getChargeState(parentIon);

                    /* verify that ion is not completely ionized */
                    if(chargeState < protonNumber)
                    {
                        uint32_t const cs = pmacc::math::float2int_rd(chargeState);
                        float_X const iEnergy =
                            typename picongpu::traits::GetIonizationEnergies<ParticleType>::type{}[cs];

                        constexpr float_X pi = pmacc::math::Pi<float_X>::value;
                        /* electric field in atomic units - only absolute value */
                        float_X const eInAU = sim.pic.conv().eField2auEField(pmacc::math::l2norm(eField));

                        /* no tunneling without a field or without a bound electron: both make
                         * nEff and dBase diverge (division by zero, 0^Inf). l2norm is >= 0, so
                         * zero is the only reachable edge; inputs from outside are finite. */
                        if(eInAU <= float_X(0.0) || iEnergy <= float_X(0.0))
                        {
                            return IonizerReturn{0.0, 0u};
                        }

                        /* the charge that attracts the electron that is to be ionized:
                         * equals `protonNumber - #allInnerElectrons`
                         */
                        float_X const effectiveCharge = chargeState + float_X(1.0);
                        /* effective principal quantum number (unitless) */
                        float_X const nEff = effectiveCharge / math::sqrt(float_X(2.0) * iEnergy);
                        /* ADK rate in the log domain: pow(dBase, 2 nEff) overflows to Inf while
                         * exp(-2 Zeff^3/(3 nEff^3 eInAU)) underflows to 0 for strong fields, and
                         * 0*Inf is NaN. The logarithm is algebraically identical in the safe regime:
                         *   ln(dBase) = ln(4 e Zeff^3) - ln(eInAU) - 4 ln(nEff)
                         *   ln(rate)  = ln(eInAU) + 2 nEff ln(dBase) - ln(8 pi) - ln(Zeff)
                         *               - 2 Zeff^3/(3 nEff^3 eInAU)
                         * Every log argument is > 0 because of the guard above. */
                        float_X const lnDBase = math::log(float_X(4.0) * math::exp(1._X) * util::cube(effectiveCharge))
                                                - math::log(eInAU) - float_X(4.0) * math::log(nEff);
                        float_X lnRate
                            = math::log(eInAU) + float_X(2.0) * nEff * lnDBase - math::log(float_X(8.0) * pi)
                              - math::log(effectiveCharge)
                              - float_X(2.0) * util::cube(effectiveCharge) / (float_X(3.0) * util::cube(nEff) * eInAU);

                        /* in case of linear polarization the rate is modified by an additional factor */
                        if(T_linPol)
                        {
                            /* factor from averaging over one laser cycle with LINEAR polarization */
                            lnRate += float_X(0.5)
                                      * (math::log(float_X(3.0)) + float_X(3.0) * math::log(nEff) + math::log(eInAU)
                                         - math::log(pi) - float_X(3.0) * math::log(effectiveCharge));
                        }

                        /* simulation time step in atomic units */
                        auto const timeStepAU = sim.pic.conv().time2auTime(sim.pic.getDt());
                        /* ionization probability
                         *
                         * probability = rate * time step
                         * --> for infinitesimal time steps
                         *
                         * the whole ensemble should then follow
                         * P = 1 - exp(-rate * time step) if the laser wavelength is
                         * sampled well enough
                         *
                         * Clamp the linearized probability to [0,1]: rate*dt > 1 already means
                         * certain ionization, and the Monte-Carlo test randNr < prob needs a
                         * probability in that range.
                         */
                        float_X const lnProb = lnRate + math::log(timeStepAU);
                        float_X const probADK = (lnProb >= float_X(0.0)) ? float_X(1.0) : math::exp(lnProb);

                        /* ionization condition */
                        if(randNr < probADK)
                        {
                            /* return ionization energy and number of macro electrons to produce */
                            return IonizerReturn{iEnergy, 1u};
                        }
                    }
                    /* no ionization */
                    return IonizerReturn{0.0, 0u};
                }
            };

        } // namespace ionization
    } // namespace particles
} // namespace picongpu
