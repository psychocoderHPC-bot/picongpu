/* Copyright 2026 Rene Widera
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

//! Tests for the algebraic zero/invalid-input guards of the ionization code paths.

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/SetChargeState.hpp"
#include "picongpu/particles/ionization/byCollision/ThomasFermi/AlgorithmThomasFermi.hpp"
#include "picongpu/particles/ionization/byField/ADK/AlgorithmADK.hpp"
#include "picongpu/particles/ionization/byField/BSI/AlgorithmBSIEffectiveZ.hpp"
#include "picongpu/particles/ionization/byField/IonizationCurrent/JIonizationCalc.hpp"
#include "picongpu/particles/ionization/byField/Keldysh/AlgorithmKeldysh.hpp"
#include "picongpu/particles/ionization/param.hpp"
#include "picongpu/particles/traits/GetIonizationEnergies.hpp"

#include <pmacc/math/ConstVector.hpp>
#include <pmacc/meta/String.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/memory/buffers/ParticlesBuffer.hpp>
#include <pmacc/particles/memory/frames/Frame.hpp>

#include <cstdint>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>

namespace
{
    using namespace picongpu;

    //! One-slot supercell so that it fits a one-particle frame.
    using TestSuperCell = typename pmacc::math::CT::shrinkTo<pmacc::math::CT::Int<1, 1, 1>, simDim>::type;

    //! Element used by the custom tables below.
    struct TestElement
    {
        static constexpr float_X numberOfProtons = 1.0;
        static constexpr float_X numberOfNeutrons = 0.0;
    };

    //! Tables to reach the zero-ionization-energy and zero-effective-charge guards.
    PMACC_CONST_VECTOR(float_X, 1, ZeroEnergyTable, 0.0_X);
    PMACC_CONST_VECTOR(float_X, 1, HalfEnergyTable, 0.5_X);
    PMACC_CONST_VECTOR(float_X, 1, ZeroChargeTable, 0.0_X);

    //! Build a one-particle frame from an attribute-independent flag list.
    template<typename T_Flags>
    struct TestIon
    {
        using Description = pmacc::ParticleDescription<
            PMACC_CSTRING("TestIon"),
            std::integral_constant<uint32_t, 1>,
            TestSuperCell,
            MakeSeq_t<boundElectrons>,
            T_Flags>;
        using Frame = pmacc::Frame<pmacc::detail::OperatorCreatePairStaticArray<1>, Description>;
        using Particle = typename Frame::ParticleType;

        Frame frame;

        explicit TestIon(float_X initialBoundElectrons)
        {
            frame[0][boundElectrons_] = initialBoundElectrons;
        }

        Particle particle()
        {
            return frame[0];
        }
    };

    using HydrogenFlags = MakeSeq_t<
        atomicNumbers<ionization::atomicNumbers::Hydrogen_t>,
        ionizationEnergies<ionization::energies::AU::Hydrogen_t>,
        effectiveNuclearCharge<ionization::effectiveNuclearCharge::Hydrogen_t>>;
    using HydrogenIon = TestIon<HydrogenFlags>;

    using ZeroEnergyFlags = MakeSeq_t<atomicNumbers<TestElement>, ionizationEnergies<ZeroEnergyTable_t>>;
    using ZeroEnergyIon = TestIon<ZeroEnergyFlags>;

    using ZeroChargeFlags = MakeSeq_t<
        atomicNumbers<TestElement>,
        ionizationEnergies<HalfEnergyTable_t>,
        effectiveNuclearCharge<ZeroChargeTable_t>>;
    using ZeroChargeIon = TestIon<ZeroChargeFlags>;

    //! Field in simulation units that maps to eInAU atomic units.
    //! Field along x only, so that l2norm() equals the requested atomic-unit value.
    float3_X fieldForAU(float_X eInAU)
    {
        float3_X field = float3_X::create(0.0_X);
        field.x() = eInAU / sim.pic.conv().eField2auEField(float_X(1.0));
        return field;
    }

    float_X hydrogenIonizationEnergy()
    {
        return picongpu::traits::GetIonizationEnergies<HydrogenIon::Particle>::type{}[0];
    }
} // namespace

TEST_CASE("ADK: zero field and probability bounds", "[ionization][ADK]")
{
    HydrogenIon ion(1.0_X); // neutral hydrogen, charge state 0
    auto particle = ion.particle();
    float3_X const bField = float3_X::create(0.0_X);

    // zero field: no tunneling, and the returned energy must not be NaN.
    auto const zeroField
        = picongpu::particles::ionization::AlgorithmADK<false>{}(bField, float3_X::create(0.0_X), particle, 0.5_X);
    CHECK(zeroField.newMacroElectrons == 0u);
    CHECK(zeroField.ionizationEnergy == 0.0_X);

    // representative field: randNr == 1 only fails the strict '<' test if prob > 1.
    float3_X const eField = fieldForAU(0.1_X);
    auto const bounded = picongpu::particles::ionization::AlgorithmADK<false>{}(bField, eField, particle, 1.0_X);
    CHECK(bounded.newMacroElectrons == 0u);
    CHECK(bounded.ionizationEnergy == 0.0_X);

    // randNr == 0 ionizes for the positive probability, which exercises the rate path.
    auto const ionizes = picongpu::particles::ionization::AlgorithmADK<false>{}(bField, eField, particle, 0.0_X);
    CHECK(ionizes.newMacroElectrons == 1u);
    CHECK(ionizes.ionizationEnergy == hydrogenIonizationEnergy());
}

TEST_CASE("ADK: large field probability is clamped to one", "[ionization][ADK]")
{
    HydrogenIon ion(1.0_X);
    auto particle = ion.particle();
    float3_X const bField = float3_X::create(0.0_X);
    // The hydrogen rate peaks near eInAU = 2/3, where rate*dt is about 8 without the clamp.
    float3_X const eField = fieldForAU(float_X(2.0) / float_X(3.0));

    // Clamped prob == 1 still fails the strict '<' test for randNr == 1.
    auto const notForced = picongpu::particles::ionization::AlgorithmADK<false>{}(bField, eField, particle, 1.0_X);
    CHECK(notForced.newMacroElectrons == 0u);
    // Just below 1 the ionization is certain.
    auto const forced = picongpu::particles::ionization::AlgorithmADK<false>{}(bField, eField, particle, 0.999_X);
    CHECK(forced.newMacroElectrons == 1u);
}

TEST_CASE("Keldysh: zero ionization energy gives no electrons", "[ionization][Keldysh]")
{
    ZeroEnergyIon ion(1.0_X);
    auto particle = ion.particle();
    float3_X const eField = fieldForAU(0.1_X);
    float3_X const bField = float3_X::create(0.0_X);

    auto const result = picongpu::particles::ionization::AlgorithmKeldysh{}(bField, eField, particle, 0.5_X);

    CHECK(result.newMacroElectrons == 0u); // iEnergy = 0 makes the rate Inf or NaN
    CHECK(result.ionizationEnergy == 0.0_X);
}

TEST_CASE("BSI effective charge: zero ZEff gives no electrons", "[ionization][BSI]")
{
    ZeroChargeIon ion(1.0_X);
    auto particle = ion.particle();
    float3_X const eField = fieldForAU(1.0_X);

    auto const result = picongpu::particles::ionization::AlgorithmBSIEffectiveZ{}(eField, particle);

    CHECK(result.newMacroElectrons == 0u); // the critical field 1/(4 ZEff) is undefined
    CHECK(result.ionizationEnergy == 0.0_X);
}

TEST_CASE("JIonizationCalc: zero field gives zero current", "[ionization][current]")
{
    float3_X const eField = float3_X::create(0.0_X);

    auto const current = picongpu::particles::ionization::JIonizationCalc{}(1.0_X, eField);

    CHECK(current.x() == 0.0_X); // avoids the 0/0 of eField/|eField|^2
    CHECK(current.y() == 0.0_X);
    CHECK(current.z() == 0.0_X);
}

TEST_CASE("ThomasFermi: zero density gives no electrons", "[ionization][ThomasFermi]")
{
    HydrogenIon ion(1.0_X);
    auto particle = ion.particle();

    // density 0 is below the cutoff and must not reach kinEnergyDensity / density.
    auto const numElectrons = picongpu::particles::ionization::AlgorithmThomasFermi{}(0.0_X, 0.0_X, particle, 0.5_X);

    CHECK(numElectrons == 0u);
}

TEST_CASE("SetChargeState: clamps to [0, numberOfProtons]", "[ionization][chargeState]")
{
    HydrogenIon ion(1.0_X);
    auto particle = ion.particle();

    picongpu::particles::atomicPhysics::SetChargeState{}(particle, -1.0_X);
    CHECK(particle[boundElectrons_] == 0.0_X); // a negative float underflow clamps to 0

    picongpu::particles::atomicPhysics::SetChargeState{}(particle, 2.0_X);
    CHECK(particle[boundElectrons_] == 1.0_X); // above Z = 1 clamps to Z

    picongpu::particles::atomicPhysics::SetChargeState{}(particle, 0.5_X);
    CHECK(particle[boundElectrons_] == 0.5_X); // in-range values are unchanged
}
