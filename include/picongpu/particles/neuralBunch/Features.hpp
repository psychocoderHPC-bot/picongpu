/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** The frozen ``fourier_v1`` feature map from MODEL-FORMAT.md.
 *
 * Feature order (69 values):
 *   1. the nine base inputs r_x, r_y, r_z, u_x, u_y, u_z, gamma, a_y, a_z,
 *   2. axis-major (x,y,z), octave ascending (-3..6), sin before cos:
 *      sin(pi * 2^j * r_axis), cos(pi * 2^j * r_axis).
 *
 * The angle is evaluated in double and only the final value is narrowed to float. The device and host entry points
 * use the same operations; the host reference uses ``std::sin``/``std::cos``, the device kernel uses
 * ``alpaka::math::sin``/``alpaka::math::cos`` on double, so the float32 result matches the NumPy reference for all
 * values whose double result is not within half an ULP of a float rounding boundary.
 */

#include <alpaka/alpaka.hpp>

#include <array>
#include <cmath>
#include <cstddef>

namespace picongpu::particles::neuralBunch
{
    inline constexpr int kFeatureDim = 69;
    inline constexpr int kOctaveLow = -3;
    inline constexpr int kOctaveHigh = 6;
    inline constexpr double kPi = 3.141592653589793238462643383279502884;

    inline constexpr std::array<char const*, 9>
        kFeatureInputNames{"r_x", "r_y", "r_z", "u_x", "u_y", "u_z", "gamma", "a_y", "a_z"};

    /** 2^j for j = -3 .. 6, indexed by j + 3. */
    inline constexpr std::array<double, 10> kOctaveScale{0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0};

    /** Device-side feature construction. */
    struct FourierV1
    {
        /** Write 69 features for one source. @p out must point at kFeatureDim floats. */
        ALPAKA_FN_ACC static void fill(
            float* out,
            double rX,
            double rY,
            double rZ,
            double uX,
            double uY,
            double uZ,
            double aY,
            double aZ)
        {
            double const gamma = alpaka::math::sqrt(1.0 + uX * uX + uY * uY + uZ * uZ);
            out[0] = static_cast<float>(rX);
            out[1] = static_cast<float>(rY);
            out[2] = static_cast<float>(rZ);
            out[3] = static_cast<float>(uX);
            out[4] = static_cast<float>(uY);
            out[5] = static_cast<float>(uZ);
            out[6] = static_cast<float>(gamma);
            out[7] = static_cast<float>(aY);
            out[8] = static_cast<float>(aZ);

            // A local constexpr array is usable in device code; a namespace-scope std::array constant is not.
            constexpr double octaveScale[10]{0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0};
            double const r[3]{rX, rY, rZ};
            std::size_t k = 9u;
            for(std::size_t axis = 0u; axis < 3u; ++axis)
            {
                for(std::size_t octave = 0u; octave < 10u; ++octave)
                {
                    double const phase = kPi * octaveScale[octave] * r[axis];
                    out[k++] = static_cast<float>(alpaka::math::sin(phase));
                    out[k++] = static_cast<float>(alpaka::math::cos(phase));
                }
            }
        }
    };

    /** Host reference feature construction, used only by tests. */
    inline void fourierV1Host(
        float* out,
        double rX,
        double rY,
        double rZ,
        double uX,
        double uY,
        double uZ,
        double aY,
        double aZ)
    {
        double const gamma = std::sqrt(1.0 + uX * uX + uY * uY + uZ * uZ);
        out[0] = static_cast<float>(rX);
        out[1] = static_cast<float>(rY);
        out[2] = static_cast<float>(rZ);
        out[3] = static_cast<float>(uX);
        out[4] = static_cast<float>(uY);
        out[5] = static_cast<float>(uZ);
        out[6] = static_cast<float>(gamma);
        out[7] = static_cast<float>(aY);
        out[8] = static_cast<float>(aZ);

        double const r[3]{rX, rY, rZ};
        std::size_t k = 9u;
        for(std::size_t axis = 0u; axis < 3u; ++axis)
        {
            for(std::size_t octave = 0u; octave < kOctaveScale.size(); ++octave)
            {
                double const phase = kPi * kOctaveScale[octave] * r[axis];
                out[k++] = static_cast<float>(std::sin(phase));
                out[k++] = static_cast<float>(std::cos(phase));
            }
        }
    }
} // namespace picongpu::particles::neuralBunch
