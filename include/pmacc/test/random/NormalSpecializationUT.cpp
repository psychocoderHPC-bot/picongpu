/* Copyright 2026 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/boost_workaround.hpp>

#include <pmacc/random/distributions/Normal.hpp>
#include <pmacc/random/distributions/misc/MullerBox.hpp>
#include <pmacc/random/methods/AlpakaRand.hpp>
#include <pmacc/random/methods/MRG32k3aMin.hpp>
#include <pmacc/random/methods/XorMin.hpp>

#include <catch2/catch_test_macros.hpp>

#include <type_traits>

namespace
{
    namespace distributions = pmacc::random::distributions;
    namespace methods = pmacc::random::methods;

    //! true if the Normal implementation for T_Method uses the vendor specific MullerBox transform
    template<typename T_Type, typename T_Method>
    inline constexpr bool usesMullerBox
        = std::is_base_of_v<distributions::MullerBox<T_Type, T_Method>, distributions::Normal<T_Type, T_Method>>;

    //! true if the RNG provider is a dedicated vendor type instead of the alpaka fallback
    template<typename T_Method>
    inline constexpr bool isVendorType = !std::is_same_v<T_Method, methods::AlpakaRand<>>;

/* XorMin is a distinct vendor RNG on CUDA and HIP and aliases AlpakaRand on
 * every CPU backend, so its Normal specialization must follow the GPU guards.
 * Its state type only needs the plain XORWOW struct, which instantiates with
 * both curand and hiprand.
 */
#if (ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED)
    static_assert(isVendorType<methods::XorMin<>>, "CUDA/HIP XorMin must be a distinct vendor RNG");
    static_assert(usesMullerBox<float, methods::XorMin<>>, "CUDA/HIP XorMin must use the vendor normal");
    static_assert(usesMullerBox<double, methods::XorMin<>>, "CUDA/HIP XorMin must use the vendor normal");
#else
    static_assert(!isVendorType<methods::XorMin<>>, "non-GPU XorMin aliases AlpakaRand");
    static_assert(!usesMullerBox<float, methods::XorMin<>>, "non-GPU XorMin aliases AlpakaRand");
    static_assert(!usesMullerBox<double, methods::XorMin<>>, "non-GPU XorMin aliases AlpakaRand");
#endif

/* MRG32k3aMin is a dedicated vendor RNG only on CUDA. On HIP and every CPU
 * backend it aliases AlpakaRand (see MRG32k3aMin.hpp), so a vendor
 * specialization would also match AlpakaRand and shadow Normal_generic. That
 * changes the host RNG stream away from the alpaka2-identical values, hence the
 * MRG32k3aMin specialization must be CUDA only.
 *
 * Normal<T, MRG32k3aMin<>> cannot be instantiated with CUDA 12.5: PMacc declares
 * MRG32k3aMin::StateType as double s1[3]/s2[3] while the installed
 * curandStateMRG32k3a uses unsigned int s1[3]/s2[3], so the layout static
 * asserts in MRG32k3aMin fire on the untouched base tree as well. That is a
 * pre-existing defect unrelated to specialization selection. AlpakaRand is not
 * instantiable on CUDA either because its state type comes from a device-only
 * alpaka factory. Therefore CUDA can only verify the precondition at type
 * level (MRG32k3aMin is a distinct vendor RNG) while the real selection is
 * pinned on all backends where the engines compile.
 */
#if (ALPAKA_ACC_GPU_CUDA_ENABLED)
    static_assert(isVendorType<methods::MRG32k3aMin<>>, "CUDA MRG32k3aMin must be a distinct vendor RNG");
#else
    using FloatMrg = distributions::Normal<float, methods::MRG32k3aMin<>>;
    using DoubleMrg = distributions::Normal<double, methods::MRG32k3aMin<>>;

    static_assert(!isVendorType<methods::MRG32k3aMin<>>, "non-CUDA MRG32k3aMin must alias AlpakaRand");
    static_assert(
        !usesMullerBox<float, methods::MRG32k3aMin<>>,
        "non-CUDA MRG32k3aMin aliases AlpakaRand and must use the generic normal");
    static_assert(
        !usesMullerBox<double, methods::MRG32k3aMin<>>,
        "non-CUDA MRG32k3aMin aliases AlpakaRand and must use the generic normal");
    static_assert(!usesMullerBox<float, methods::AlpakaRand<>>, "AlpakaRand must use the generic normal");
    static_assert(!usesMullerBox<double, methods::AlpakaRand<>>, "AlpakaRand must use the generic normal");
#endif
} // namespace

TEST_CASE("Normal distribution selects the expected RNG specialization", "[random]")
{
#if (ALPAKA_ACC_GPU_CUDA_ENABLED)
    // The MRG32k3aMin and AlpakaRand engines cannot be instantiated with CUDA 12.5, see above.
    STATIC_REQUIRE(isVendorType<methods::MRG32k3aMin<>>);
    STATIC_REQUIRE(isVendorType<methods::XorMin<>>);
    STATIC_REQUIRE(std::is_same_v<typename distributions::Normal<float, methods::XorMin<>>::result_type, float>);
    STATIC_REQUIRE(std::is_same_v<typename distributions::Normal<double, methods::XorMin<>>::result_type, double>);
#elif (ALPAKA_ACC_GPU_HIP_ENABLED)
    // MRG32k3aMin aliases AlpakaRand on HIP, XorMin stays a distinct vendor RNG.
    STATIC_REQUIRE(!isVendorType<methods::MRG32k3aMin<>>);
    STATIC_REQUIRE(isVendorType<methods::XorMin<>>);
    STATIC_REQUIRE(std::is_same_v<typename FloatMrg::result_type, float>);
    STATIC_REQUIRE(std::is_same_v<typename DoubleMrg::result_type, double>);
    STATIC_REQUIRE(std::is_same_v<typename distributions::Normal<float, methods::AlpakaRand<>>::result_type, float>);
    STATIC_REQUIRE(std::is_same_v<typename distributions::Normal<double, methods::AlpakaRand<>>::result_type, double>);
#else
    // No GPU backend: both providers alias AlpakaRand.
    STATIC_REQUIRE(!isVendorType<methods::MRG32k3aMin<>>);
    STATIC_REQUIRE(!isVendorType<methods::XorMin<>>);
    STATIC_REQUIRE(std::is_same_v<typename FloatMrg::result_type, float>);
    STATIC_REQUIRE(std::is_same_v<typename DoubleMrg::result_type, double>);
    STATIC_REQUIRE(std::is_same_v<typename distributions::Normal<float, methods::AlpakaRand<>>::result_type, float>);
    STATIC_REQUIRE(std::is_same_v<typename distributions::Normal<double, methods::AlpakaRand<>>::result_type, double>);
#endif
}
