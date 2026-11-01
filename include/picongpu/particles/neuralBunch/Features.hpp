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
#include <optional>
#include <string_view>

namespace picongpu::particles::neuralBunch
{
    inline constexpr int kFeatureDim = 69;
    inline constexpr int kOctaveLow = -3;
    inline constexpr int kOctaveHigh = 6;
    inline constexpr double kPi = 3.141592653589793238462643383279502884;

    inline constexpr std::array<char const*, 9> kFeatureInputNames{
        "r_x",
        "r_y",
        "r_z",
        "u_x",
        "u_y",
        "u_z",
        "gamma",
        "a_y",
        "a_z"};

    /** 2^j for j = -3 .. 6, indexed by j + 3. */
    inline constexpr std::array<double, 10> kOctaveScale{0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0};

    /** Device-side feature construction. */
    struct FourierV1
    {
        /** Write 69 features for one source. @p out must point at kFeatureDim floats. */
        ALPAKA_FN_HOST_ACC static void fill(
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

    // -----------------------------------------------------------------------
    // Probe feature kinds added by the training package (MODEL-FORMAT v1 extensions)
    //
    //   fourier_probe_v1 : 9 base + 4x4x2 probes x 6 components + low-frequency
    //                      Fourier of r (octaves -3..1) = 231
    //   probe_dense_v1   : 9 base + 8x8x4 probes x 6 components = 1545
    //
    // Probe positions are the per-tile coordinates from the Python reference:
    //   fourier_probe_v1: x=2i+0.5, y=(2j+0.5)*a_y, z=(2k+0.5)*a_z, k outer, i inner
    //   probe_dense_v1  : x=i+0.5,  y=(j+0.5)*a_y,  z=(k+0.5)*a_z,  k outer, i inner
    // Values are the per-unit-(q/e) point-charge field in dx units, using the
    // stable uniform-motion decomposition of features.py:point_field_dx.
    // -----------------------------------------------------------------------
    enum class FeatureKind
    {
        fourierV1,
        fourierProbeV1,
        probeDenseV1
    };

    inline constexpr int kProbeOctaveLow = -3;
    inline constexpr int kProbeOctaveHigh = 1;
    inline constexpr std::size_t kProbeCount = 32;
    inline constexpr std::size_t kDenseCount = 256;
    inline constexpr std::size_t kProbeDim = 231;
    inline constexpr std::size_t kDenseDim = 1545;
    inline constexpr std::size_t kMaxFeatureDim = kDenseDim;

    constexpr std::size_t featureDimension(FeatureKind kind)
    {
        switch(kind)
        {
        case FeatureKind::fourierV1:
            return static_cast<std::size_t>(kFeatureDim);
        case FeatureKind::fourierProbeV1:
            return kProbeDim;
        case FeatureKind::probeDenseV1:
            return kDenseDim;
        }
        return 0u;
    }

    constexpr std::string_view featureKindName(FeatureKind kind)
    {
        switch(kind)
        {
        case FeatureKind::fourierV1:
            return "fourier_v1";
        case FeatureKind::fourierProbeV1:
            return "fourier_probe_v1";
        case FeatureKind::probeDenseV1:
            return "probe_dense_v1";
        }
        return {};
    }

    inline std::optional<FeatureKind> featureKindFromString(std::string_view name)
    {
        if(name == "fourier_v1")
            return FeatureKind::fourierV1;
        if(name == "fourier_probe_v1")
            return FeatureKind::fourierProbeV1;
        if(name == "probe_dense_v1")
            return FeatureKind::probeDenseV1;
        return std::nullopt;
    }

    constexpr std::size_t featureProbeCount(FeatureKind kind)
    {
        switch(kind)
        {
        case FeatureKind::fourierProbeV1:
            return kProbeCount;
        case FeatureKind::probeDenseV1:
            return kDenseCount;
        default:
            return 0u;
        }
    }

    namespace detail
    {
        /** Probe offsets in dx units (x, y, z), ordered i fastest, then j, then k. */
        constexpr std::array<std::array<double, 3>, kProbeCount> probeCellOffsets()
        {
            std::array<std::array<double, 3>, kProbeCount> offsets{};
            std::size_t index = 0u;
            for(int k = 0; k < 2; ++k)
                for(int j = 0; j < 4; ++j)
                    for(int i = 0; i < 4; ++i)
                        offsets[index++] = {2.0 * i + 0.5, 2.0 * j + 0.5, 2.0 * k + 0.5};
            return offsets;
        }

        constexpr std::array<std::array<double, 3>, kDenseCount> denseCellOffsets()
        {
            std::array<std::array<double, 3>, kDenseCount> offsets{};
            std::size_t index = 0u;
            for(int k = 0; k < 4; ++k)
                for(int j = 0; j < 8; ++j)
                    for(int i = 0; i < 8; ++i)
                        offsets[index++] = {i + 0.5, j + 0.5, k + 0.5};
            return offsets;
        }
    } // namespace detail

    inline constexpr auto kProbeCellOffsets = detail::probeCellOffsets();
    inline constexpr auto kDenseCellOffsets = detail::denseCellOffsets();

    struct Vec3d
    {
        double x{};
        double y{};
        double z{};
    };

    struct PointFieldDx
    {
        Vec3d E{};
        Vec3d B{};
    };

    /** Python ``features.point_field_dx``: per-unit-charge field for R in dx units. */
    ALPAKA_FN_HOST_ACC inline PointFieldDx pointFieldDx(Vec3d const& R, Vec3d const& u)
    {
        double const uu2 = u.x * u.x + u.y * u.y + u.z * u.z;
        double const invG2 = 1.0 / (1.0 + uu2);
        double const speed = alpaka::math::sqrt(uu2);
        bool const isStatic = speed == 0.0;
        double const safe = isStatic ? 1.0 : speed;
        Vec3d const bhat{u.x / safe, u.y / safe, u.z / safe};
        double const gamma = alpaka::math::sqrt(1.0 + uu2);
        Vec3d const beta{u.x / gamma, u.y / gamma, u.z / gamma};

        double const rPar = R.x * bhat.x + R.y * bhat.y + R.z * bhat.z;
        Vec3d const rPerp{R.x - rPar * bhat.x, R.y - rPar * bhat.y, R.z - rPar * bhat.z};
        double const perp2 = rPerp.x * rPerp.x + rPerp.y * rPerp.y + rPerp.z * rPerp.z;
        double const denom = alpaka::math::pow(rPar * rPar + perp2 * invG2, 1.5);

        Vec3d const eMov{invG2 * R.x / denom, invG2 * R.y / denom, invG2 * R.z / denom};
        Vec3d const bMov{
            beta.y * eMov.z - beta.z * eMov.y,
            beta.z * eMov.x - beta.x * eMov.z,
            beta.x * eMov.y - beta.y * eMov.x};

        double const r2 = R.x * R.x + R.y * R.y + R.z * R.z;
        double const r23 = alpaka::math::pow(r2, 1.5);
        Vec3d const eStatic{R.x / r23, R.y / r23, R.z / r23};

        PointFieldDx result;
        result.E = isStatic ? eStatic : eMov;
        result.B = isStatic ? Vec3d{} : bMov;
        return result;
    }

    /** Append axis-major Fourier terms (sin then cos per octave) for the given octave range. */
    ALPAKA_FN_HOST_ACC inline void appendFourierTerms(
        float* out,
        double const r[3],
        int octaveLow,
        int octaveHigh)
    {
        constexpr double scales[10]{0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0};
        std::size_t k = 0u;
        for(std::size_t axis = 0u; axis < 3u; ++axis)
        {
            for(int octave = octaveLow; octave <= octaveHigh; ++octave)
            {
                double const phase = kPi * scales[octave + 3] * r[axis];
                out[k++] = static_cast<float>(alpaka::math::sin(phase));
                out[k++] = static_cast<float>(alpaka::math::cos(phase));
            }
        }
    }

    /** Host+device feature construction for all three kinds. fourier_v1 uses the frozen FourierV1 path. */
    template<FeatureKind Kind>
    ALPAKA_FN_HOST_ACC inline void fillFeatures(
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
        if constexpr(Kind == FeatureKind::fourierV1)
        {
            FourierV1::fill(out, rX, rY, rZ, uX, uY, uZ, aY, aZ);
        }
        else
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

            std::size_t write = 9u;
            std::size_t const probes
                = Kind == FeatureKind::fourierProbeV1 ? kProbeCount : kDenseCount;
            for(std::size_t p = 0u; p < probes; ++p)
            {
                // Computed inline: namespace-scope constexpr arrays are not usable in device code.
                double offsetX{};
                double offsetY{};
                double offsetZ{};
                if constexpr(Kind == FeatureKind::fourierProbeV1)
                {
                    auto const i = p % 4u;
                    auto const j = (p / 4u) % 4u;
                    auto const k = p / 16u;
                    offsetX = 2.0 * static_cast<double>(i) + 0.5;
                    offsetY = 2.0 * static_cast<double>(j) + 0.5;
                    offsetZ = 2.0 * static_cast<double>(k) + 0.5;
                }
                else
                {
                    auto const i = p % 8u;
                    auto const j = (p / 8u) % 8u;
                    auto const k = p / 64u;
                    offsetX = static_cast<double>(i) + 0.5;
                    offsetY = static_cast<double>(j) + 0.5;
                    offsetZ = static_cast<double>(k) + 0.5;
                }
                Vec3d const R{offsetX - rX, offsetY * aY - rY, offsetZ * aZ - rZ};
                auto const field = pointFieldDx(R, Vec3d{uX, uY, uZ});
                out[write++] = static_cast<float>(field.E.x);
                out[write++] = static_cast<float>(field.E.y);
                out[write++] = static_cast<float>(field.E.z);
                out[write++] = static_cast<float>(field.B.x);
                out[write++] = static_cast<float>(field.B.y);
                out[write++] = static_cast<float>(field.B.z);
            }
            if constexpr(Kind == FeatureKind::fourierProbeV1)
            {
                double const r[3]{rX, rY, rZ};
                appendFourierTerms(out + write, r, kProbeOctaveLow, kProbeOctaveHigh);
            }
        }
    }

    /** Shared host feature construction for tests. */
    inline void featuresHost(
        FeatureKind kind,
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
        switch(kind)
        {
        case FeatureKind::fourierV1:
            fillFeatures<FeatureKind::fourierV1>(out, rX, rY, rZ, uX, uY, uZ, aY, aZ);
            break;
        case FeatureKind::fourierProbeV1:
            fillFeatures<FeatureKind::fourierProbeV1>(out, rX, rY, rZ, uX, uY, uZ, aY, aZ);
            break;
        case FeatureKind::probeDenseV1:
            fillFeatures<FeatureKind::probeDenseV1>(out, rX, rY, rZ, uX, uY, uZ, aY, aZ);
            break;
        }
    }
} // namespace picongpu::particles::neuralBunch
