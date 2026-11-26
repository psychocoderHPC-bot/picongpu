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

//! Tests for the ionization input guards, the ADK rate rewrite and the Thomas-Fermi valid regime.

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

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace
{
    using namespace picongpu;
    namespace particleIonization = picongpu::particles::ionization;
    namespace atomicPhysics = picongpu::particles::atomicPhysics;
    namespace particleToGrid = picongpu::particles::particleToGrid;

    using TestSuperCell = typename pmacc::math::CT::shrinkTo<pmacc::math::CT::Int<1, 1, 1>, simDim>::type;

    struct TestElement
    {
        static constexpr float_X numberOfProtons = 1.0;
        static constexpr float_X numberOfNeutrons = 0.0;
    };

    struct TestHelium
    {
        static constexpr float_X numberOfProtons = 2.0;
        static constexpr float_X numberOfNeutrons = 2.0;
    };

    PMACC_CONST_VECTOR(float_X, 1, ZeroEnergyTable, 0.0_X);
    PMACC_CONST_VECTOR(float_X, 1, HalfEnergyTable, 0.5_X);
    PMACC_CONST_VECTOR(float_X, 1, ZeroChargeTable, 0.0_X);
    PMACC_CONST_VECTOR(float_X, 1, HalfChargeTable, 0.5_X);
    PMACC_CONST_VECTOR(float_X, 1, NegativeChargeTable, -0.5_X);

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

    using ElementNumbers = atomicNumbers<TestElement>;
    using ZeroEnergy = ionizationEnergies<ZeroEnergyTable_t>;
    using HalfEnergy = ionizationEnergies<HalfEnergyTable_t>;
    using HalfCharge = effectiveNuclearCharge<HalfChargeTable_t>;

    using HydrogenIon = TestIon<MakeSeq_t<
        atomicNumbers<ionization::atomicNumbers::Hydrogen_t>,
        ionizationEnergies<ionization::energies::AU::Hydrogen_t>,
        effectiveNuclearCharge<ionization::effectiveNuclearCharge::Hydrogen_t>>>;
    using ZeroEnergyIon = TestIon<MakeSeq_t<ElementNumbers, ZeroEnergy>>;
    using ZeroChargeIon = TestIon<MakeSeq_t<ElementNumbers, HalfEnergy, effectiveNuclearCharge<ZeroChargeTable_t>>>;
    using ZeroEnergyHalfChargeIon = TestIon<MakeSeq_t<ElementNumbers, ZeroEnergy, HalfCharge>>;
    using NegativeChargeIon
        = TestIon<MakeSeq_t<ElementNumbers, HalfEnergy, effectiveNuclearCharge<NegativeChargeTable_t>>>;
    using HeliumIon = TestIon<MakeSeq_t<atomicNumbers<TestHelium>>>;

    //! Field in simulation units that maps to eInAU atomic units (x only, so l2norm is eInAU).
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

    template<bool T_linPol>
    bool adkIonizes(float3_X const eField, float_X randNr)
    {
        HydrogenIon ion(1.0_X);
        auto particle = ion.particle();
        return particleIonization::AlgorithmADK<T_linPol>{}(float3_X::create(0.0_X), eField, particle, randNr)
                   .newMacroElectrons
               == 1u;
    }

    //! Recover the ADK probability by bisecting randNr: the ionizer fires iff randNr < probability.
    template<bool T_linPol>
    float_X adkProbability(float3_X const eField)
    {
        float_X lo = 0.0_X; // randNr = 0 fires for any positive probability
        float_X hi = 1.0_X; // randNr = 1 never fires for a probability <= 1
        for(int i = 0; i < 40; ++i)
        {
            float_X const mid = float_X(0.5) * (lo + hi);
            if(adkIonizes<T_linPol>(eField, mid))
                lo = mid;
            else
                hi = mid;
        }
        return float_X(0.5) * (lo + hi);
    }

    //! Circular-polarization ADK probability from the original formula, in double (hydrogen-like).
    double adkReferenceProbability(double eInAU, double iEnergy)
    {
        constexpr double e = 2.718281828459045, pi = 3.141592653589793;
        double const nEff = 1.0 / std::sqrt(2.0 * iEnergy);
        double const rate = eInAU * std::pow(4.0 * e / (eInAU * std::pow(nEff, 4.0)), 2.0 * nEff) / (8.0 * pi)
                            * std::exp(-2.0 / (3.0 * std::pow(nEff, 3.0) * eInAU));
        return rate * sim.pic.conv().time2auTime(sim.pic.getDt<double>());
    }
} // namespace

TEST_CASE("ADK: guards, clamp and log-domain parity", "[ionization][ADK]")
{
    HydrogenIon ion(1.0_X); // neutral hydrogen, charge state 0
    auto particle = ion.particle();
    float3_X const bField = float3_X::create(0.0_X);

    // Zero field is a regression check: the old NaN probability also failed randNr.
    auto const zeroField = particleIonization::AlgorithmADK<false>{}(bField, float3_X::create(0.0_X), particle, 0.5_X);
    CHECK(zeroField.newMacroElectrons == 0u);

    float3_X const eField = fieldForAU(0.1_X);
    CHECK(particleIonization::AlgorithmADK<false>{}(bField, eField, particle, 1.0_X).newMacroElectrons == 0u);
    auto const ionizes = particleIonization::AlgorithmADK<false>{}(bField, eField, particle, 0.0_X);
    CHECK(ionizes.newMacroElectrons == 1u);
    CHECK(ionizes.ionizationEnergy == hydrogenIonizationEnergy());

    double const eInAU = sim.pic.conv().eField2auEField(pmacc::math::l2norm(eField));
    double const reference = adkReferenceProbability(eInAU, hydrogenIonizationEnergy());
    float_X const probability = adkProbability<false>(eField);
    REQUIRE(probability > 0.0_X);
    REQUIRE(probability < 1.0_X);
    CHECK(probability == Catch::Approx(reference).epsilon(1e-4));

    float3_X const strongField = fieldForAU(float_X(2.0) / float_X(3.0));
    auto const notForced = particleIonization::AlgorithmADK<false>{}(bField, strongField, particle, 1.0_X);
    CHECK(notForced.newMacroElectrons == 0u);
    auto const forced = particleIonization::AlgorithmADK<false>{}(bField, strongField, particle, 0.999_X);
    CHECK(forced.newMacroElectrons == 1u);
    // Linear branch clamp: randNr below one fires, randNr = 1 does not.
    CHECK(adkIonizes<true>(strongField, 0.999_X));
    CHECK_FALSE(adkIonizes<true>(strongField, 1.0_X));
}

TEST_CASE("Keldysh: zero ionization energy gives no electrons (regression only)", "[ionization][Keldysh]")
{
    ZeroEnergyIon ion(1.0_X);
    auto particle = ion.particle();
    auto const result
        = particleIonization::AlgorithmKeldysh{}(float3_X::create(0.0_X), fieldForAU(0.1_X), particle, 0.5_X);

    // Regression only: the old Inf/NaN probability also failed the randNr test.
    CHECK(result.newMacroElectrons == 0u);
    CHECK(result.ionizationEnergy == 0.0_X);
}

TEST_CASE("BSI effective charge: guard branches", "[ionization][BSI]")
{
    auto checkNoIonization = [](auto& ion)
    {
        auto particle = ion.particle();
        auto const result = particleIonization::AlgorithmBSIEffectiveZ{}(fieldForAU(1.0_X), particle);
        CHECK(result.newMacroElectrons == 0u);
        CHECK(result.ionizationEnergy == 0.0_X);
    };

    // ZEff == 0 is a regression check: the old critField = Inf also blocked ionization.
    ZeroChargeIon zeroChargeIon(1.0_X);
    checkNoIonization(zeroChargeIon);
    // Behavior change: old iEnergy == 0 gave critField == 0 and ZEff < 0 a negative
    // one, so any field ionized. Both must now return {0, 0}.
    ZeroEnergyHalfChargeIon zeroEnergyIon(1.0_X);
    checkNoIonization(zeroEnergyIon);
    NegativeChargeIon negativeChargeIon(1.0_X);
    checkNoIonization(negativeChargeIon);
}

TEST_CASE("JIonizationCalc: zero field gives zero current (behavior change)", "[ionization][current]")
{
    auto const current = particleIonization::JIonizationCalc{}(1.0_X, float3_X::create(0.0_X));

    // The old code returned NaN here, so these checks fail without the guard.
    CHECK(current.x() == 0.0_X);
    CHECK(current.y() == 0.0_X);
    CHECK(current.z() == 0.0_X);
}

TEST_CASE("ThomasFermi: cutoff and valid regime", "[ionization][ThomasFermi]")
{
    HydrogenIon zeroDensityIon(1.0_X);
    auto zeroDensityParticle = zeroDensityIon.particle();
    // Regression only: the old NaN temperature also failed the temperature cutoff.
    CHECK(particleIonization::AlgorithmThomasFermi{}(0.0_X, 0.0_X, zeroDensityParticle, 0.5_X) == 0u);

    float_64 const densityUnit = particleToGrid::derivedAttributes::Density().getUnit()[0];
    float_64 const energyDensityUnit = particleToGrid::derivedAttributes::EnergyDensity().getUnit()[0];
    float_64 const densityCutoff = particleIonization::thomasFermi::CUTOFF_LOW_DENSITY;
    float_64 const temperatureCutoff = particleIonization::thomasFermi::CUTOFF_LOW_TEMPERATURE_EV;
    float_X const density = static_cast<float_X>(2.0 * densityCutoff / densityUnit);
    constexpr float_64 jouleToEv = sim.si.conv().joule2eV(1.0) * float_64(2.0 / 3.0);
    float_X const kinEnergyDensity
        = static_cast<float_X>(100.0 * temperatureCutoff / jouleToEv * density * densityUnit / energyDensityUnit);

    HeliumIon heliumIon(1.0_X);
    auto heliumParticle = heliumIon.particle();
    // randNr = 0 rounds the fractional charge state up, so ionization must happen.
    CHECK(particleIonization::AlgorithmThomasFermi{}(kinEnergyDensity, density, heliumParticle, 0.0_X) > 0u);
}

TEST_CASE("SetChargeState: exact boundaries clamp into [0, numberOfProtons]", "[ionization][chargeState]")
{
    HydrogenIon ion(1.0_X);
    auto particle = ion.particle();
    constexpr float_X numberOfProtons = 1.0_X;
    float_X const eps = std::numeric_limits<float_X>::epsilon();

    atomicPhysics::SetChargeState{}(particle, 0.0_X);
    CHECK(particle[boundElectrons_] == 0.0_X);
    atomicPhysics::SetChargeState{}(particle, numberOfProtons);
    CHECK(particle[boundElectrons_] == numberOfProtons);
    atomicPhysics::SetChargeState{}(particle, numberOfProtons + eps);
    CHECK(particle[boundElectrons_] == numberOfProtons);
    atomicPhysics::SetChargeState{}(particle, -eps);
    CHECK(particle[boundElectrons_] == 0.0_X);
}
