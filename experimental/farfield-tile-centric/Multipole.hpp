/* Copyright 2026
 *
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Far-field multipole core for the neural-bunch source hierarchy (PROTOTYPE).
 *
 * Every routine here is host+device and independent of the tree construction.
 * The model is the guide's fixed-velocity Coulomb transform (agent guide
 * section 7, "Fixed-velocity Coulomb transform"):
 *
 *   A(u) = I + (gamma - 1) n n^T,  n = u/|u|,  gamma = sqrt(1 + |u|^2)
 *   E(R, u) = gamma * A^{-1} E_C(A R),   E_C(X) = X / |X|^3
 *   B(R, u) = beta x E(R, u),            beta = u / gamma
 *
 * For a node C with one representative velocity u_C the sources sit at
 * z_i = A_C y_i and the target at X = A_C x.  Expanding the Coulomb field about
 * z_C = A_C y_C gives the Cartesian multipole series
 *
 *   sum_i q_i E_C(X - z_i) ~= sum_{|a| <= p} (-1)^|a| M_a c_a(X - z_C),
 *   M_a = sum_i q_i (z_i - z_C)^a,
 *   c_a = (1/a!) d^a E_C evaluated at X - z_C.
 *
 * The stored node aggregates are therefore the total charge and the Cartesian
 * moments of order 1..p in the node's transformed frame:
 *
 *   p = 0: charge (monopole)
 *   p = 1: + dipole            (3 moments)
 *   p = 2: + quadrupole        (6 moments)
 *   p = 3: + octupole          (10 moments)
 *
 * The Taylor coefficients c_a are generated without symbolic differentiation:
 * with s = |X + d|^2 = s0 + 2 X.d + |d|^2 and w = s^{-3/2},
 *
 *   E_C(X + d) = (X + d) w,   w = s0^{-3/2} (1 + a1 u + a2 u^2 + ...),
 *   u = (2 X.d + |d|^2)/s0,   a1 = -3/2, a2 = 15/8.
 *
 * u and u^2 are sparse quadratics in d, so every coefficient is a closed form
 * and no polynomial algebra is needed.  Because w is the exact Taylor series of
 * s^{-3/2} about d = 0, c_a is exactly (1/a!) d^a E_C, and the moment
 * contraction above needs no extra factorial.
 *
 * Order is capped at 3 (octupole).  The p+1 term is evaluated separately by
 * ``multipoleTruncation`` so callers can report the truncation estimate.
 */

#include "../Features.hpp"

#include <alpaka/alpaka.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

namespace picongpu::particles::neuralBunch::farfield
{
    //! Highest supported Cartesian evaluation order (octupole).
    inline constexpr int maxEvaluationOrder = 3;
    //! Order of the stored multi-index table; one above the evaluation order so
    //! the p+1 truncation term can be reported.
    inline constexpr int maximalOrder = maxEvaluationOrder + 1;
    //! Number of Cartesian multi-indices with |a| <= maximalOrder.
    inline constexpr int maximalMomentCount = 35;

    struct MultiIndex
    {
        int i{};
        int j{};
        int k{};
        int degree{};
    };

    constexpr std::array<MultiIndex, maximalMomentCount> makeMultiIndices()
    {
        std::array<MultiIndex, maximalMomentCount> table{};
        int index = 0;
        for(int degree = 0; degree <= maximalOrder; ++degree)
            for(int i = degree; i >= 0; --i)
                for(int j = degree - i; j >= 0; --j)
                {
                    int const k = degree - i - j;
                    table[index++] = MultiIndex{i, j, k, degree};
                }
        return table;
    }

    //! Multi-indices ordered by total degree, then lexicographically.
    inline constexpr auto multiIndices = makeMultiIndices();

    //! Number of moments stored for an expansion of the given order.
    constexpr int momentCount(int order)
    {
        return (order + 1) * (order + 2) * (order + 3) / 6;
    }

    constexpr std::array<std::array<int, maximalMomentCount>, 3> makeMomentMinusOne()
    {
        std::array<std::array<int, maximalMomentCount>, 3> table{};
        for(int component = 0; component < 3; ++component)
            for(int index = 0; index < maximalMomentCount; ++index)
            {
                auto const m = multiIndices[index];
                int const e[3]{m.i, m.j, m.k};
                if(e[component] == 0)
                {
                    table[component][index] = -1;
                    continue;
                }
                int const want[3]{e[0] - (component == 0), e[1] - (component == 1), e[2] - (component == 2)};
                int found = -1;
                for(int other = 0; other < maximalMomentCount; ++other)
                {
                    auto const o = multiIndices[other];
                    if(o.i == want[0] && o.j == want[1] && o.k == want[2])
                    {
                        found = other;
                        break;
                    }
                }
                table[component][index] = found;
            }
        return table;
    }

    /** Index of the multi-index a - e_component, or -1 when a_component == 0. */
    inline constexpr auto momentMinusOne = makeMomentMinusOne();

    ALPAKA_FN_HOST_ACC inline Vec3d difference(Vec3d const& a, Vec3d const& b)
    {
        return Vec3d{a.x - b.x, a.y - b.y, a.z - b.z};
    }

    ALPAKA_FN_HOST_ACC inline double length(Vec3d const& v)
    {
        return alpaka::math::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    struct VelocityFrame
    {
        //! Unit velocity direction, zero when u == 0.
        Vec3d direction{};
        //! Lorentz factor sqrt(1 + |u|^2).
        double gamma{1.0};
    };

    ALPAKA_FN_HOST_ACC inline VelocityFrame makeVelocityFrame(Vec3d const& u)
    {
        double const u2 = u.x * u.x + u.y * u.y + u.z * u.z;
        VelocityFrame frame{};
        frame.gamma = alpaka::math::sqrt(1.0 + u2);
        if(u2 > 0.0)
        {
            double const inverse = 1.0 / alpaka::math::sqrt(u2);
            frame.direction = Vec3d{u.x * inverse, u.y * inverse, u.z * inverse};
        }
        return frame;
    }

    /** A v = v + (gamma - 1) (n.v) n. */
    ALPAKA_FN_HOST_ACC inline Vec3d applyA(VelocityFrame const& frame, Vec3d const& v)
    {
        double const along = frame.direction.x * v.x + frame.direction.y * v.y + frame.direction.z * v.z;
        double const factor = (frame.gamma - 1.0) * along;
        return Vec3d{
            v.x + factor * frame.direction.x,
            v.y + factor * frame.direction.y,
            v.z + factor * frame.direction.z};
    }

    /** A^{-1} v = v - (gamma - 1)/gamma (n.v) n. */
    ALPAKA_FN_HOST_ACC inline Vec3d applyAInverse(VelocityFrame const& frame, Vec3d const& v)
    {
        double const along = frame.direction.x * v.x + frame.direction.y * v.y + frame.direction.z * v.z;
        double const factor = (frame.gamma - 1.0) / frame.gamma * along;
        return Vec3d{
            v.x - factor * frame.direction.x,
            v.y - factor * frame.direction.y,
            v.z - factor * frame.direction.z};
    }

    /** Largest singular value of A (1 along the transverse axes, gamma along n). */
    ALPAKA_FN_HOST_ACC inline double maxStretch(VelocityFrame const& frame)
    {
        return frame.gamma;
    }

    namespace detail
    {
        ALPAKA_FN_HOST_ACC inline double xComponent(Vec3d const& v, int axis)
        {
            return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
        }

        /** Coefficient of d^a in u = (2 X.d + |d|^2)/s0 (zero unless |a| <= 2). */
        ALPAKA_FN_HOST_ACC inline double uCoefficient(
            MultiIndex const& m,
            Vec3d const& X,
            double s0,
            double inverseS0)
        {
            if(m.degree == 1)
            {
                int const axis = m.i == 1 ? 0 : (m.j == 1 ? 1 : 2);
                return 2.0 * xComponent(X, axis) * inverseS0;
            }
            if(m.degree == 2)
            {
                bool const diagonal = (m.i == 2) || (m.j == 2) || (m.k == 2);
                return diagonal ? inverseS0 : 0.0;
            }
            return 0.0;
        }

        /** Coefficient of d^a in u^2 (zero unless 2 <= |a| <= 4). */
        ALPAKA_FN_HOST_ACC inline double uSquaredCoefficient(
            MultiIndex const& m,
            Vec3d const& X,
            double inverseS0)
        {
            double const inverseS02 = inverseS0 * inverseS0;
            if(m.degree == 2)
            {
                // 4 (X.d)^2 / s0^2
                if(m.i == 2)
                    return 4.0 * X.x * X.x * inverseS02;
                if(m.j == 2)
                    return 4.0 * X.y * X.y * inverseS02;
                if(m.k == 2)
                    return 4.0 * X.z * X.z * inverseS02;
                if(m.i == 1 && m.j == 1)
                    return 8.0 * X.x * X.y * inverseS02;
                if(m.i == 1 && m.k == 1)
                    return 8.0 * X.x * X.z * inverseS02;
                if(m.j == 1 && m.k == 1)
                    return 8.0 * X.y * X.z * inverseS02;
                return 0.0;
            }
            if(m.degree == 3)
            {
                // 4 (X.d) |d|^2 / s0^2
                if(m.i == 3)
                    return 4.0 * X.x * inverseS02;
                if(m.j == 3)
                    return 4.0 * X.y * inverseS02;
                if(m.k == 3)
                    return 4.0 * X.z * inverseS02;
                if(m.i == 2)
                    return 4.0 * xComponent(X, m.j == 1 ? 1 : 2) * inverseS02;
                if(m.j == 2)
                    return 4.0 * xComponent(X, m.i == 1 ? 0 : 2) * inverseS02;
                if(m.k == 2)
                    return 4.0 * xComponent(X, m.i == 1 ? 0 : 1) * inverseS02;
                return 0.0;
            }
            if(m.degree == 4)
            {
                // |d|^4 / s0^2
                if(m.i == 4 || m.j == 4 || m.k == 4)
                    return inverseS02;
                if(m.i == 2 && m.j == 2)
                    return 2.0 * inverseS02;
                if(m.i == 2 && m.k == 2)
                    return 2.0 * inverseS02;
                if(m.j == 2 && m.k == 2)
                    return 2.0 * inverseS02;
                return 0.0;
            }
            return 0.0;
        }

        /** Taylor coefficients w_a of w(d) = |X + d|^{-3} up to |a| <= Order. */
        template<int Order>
        ALPAKA_FN_HOST_ACC inline void fillWeightCoefficients(
            std::array<double, momentCount(Order)>& w,
            Vec3d const& X,
            double s0)
        {
            constexpr int count = momentCount(Order);
            double const inverseS0 = 1.0 / s0;
            double const scale = inverseS0 / alpaka::math::sqrt(s0); // s0^{-3/2}
            constexpr double a1 = -1.5;
            constexpr double a2 = 15.0 / 8.0;
            for(int index = 0; index < count; ++index)
            {
                auto const m = multiIndices[index];
                double coefficient = 0.0;
                if(m.degree == 0)
                    coefficient = 1.0;
                else
                {
                    coefficient += a1 * uCoefficient(m, X, s0, inverseS0);
                    coefficient += a2 * uSquaredCoefficient(m, X, inverseS0);
                }
                w[index] = scale * coefficient;
            }
        }
    } // namespace detail

    /** Multipole E field of one node in its transformed frame.
     *
     * @param X target minus node centre, transformed by the node frame, in dx.
     * @param moments charge, then moments M_a in the same transformed frame.
     */
    template<int Order>
    ALPAKA_FN_HOST_ACC inline Vec3d multipoleFieldTransformed(
        Vec3d const& X,
        double const* moments)
    {
        constexpr int count = momentCount(Order);
        double const s0 = X.x * X.x + X.y * X.y + X.z * X.z;
        if(s0 <= 0.0)
            return Vec3d{};
        std::array<double, count> w{};
        detail::fillWeightCoefficients<Order>(w, X, s0);

        Vec3d result{};
        for(int index = 0; index < count; ++index)
        {
            auto const m = multiIndices[index];
            double const sign = (m.degree & 1) ? -1.0 : 1.0;
            double const weight = sign * moments[index];
            int const minusX = momentMinusOne[0][index];
            int const minusY = momentMinusOne[1][index];
            int const minusZ = momentMinusOne[2][index];
            result.x += weight * (X.x * w[index] + (minusX >= 0 ? w[minusX] : 0.0));
            result.y += weight * (X.y * w[index] + (minusY >= 0 ? w[minusY] : 0.0));
            result.z += weight * (X.z * w[index] + (minusZ >= 0 ? w[minusZ] : 0.0));
        }
        return result;
    }

    /** Magnitude of the p+1 truncation term, using absolute moments.
     *
     * ``absoluteMoments`` must contain |M_a| (same layout).  The result is an
     * upper estimate of the neglected term at this target point; it is not a
     * rigorous bound because |sum| <= sum |.| is the only inequality used.
     */
    template<int Order>
    ALPAKA_FN_HOST_ACC inline Vec3d multipoleTruncation(
        Vec3d const& X,
        double const* absoluteMoments)
    {
        constexpr int count = momentCount(Order + 1);
        double const s0 = X.x * X.x + X.y * X.y + X.z * X.z;
        if(s0 <= 0.0)
            return Vec3d{};
        std::array<double, count> w{};
        detail::fillWeightCoefficients<Order + 1>(w, X, s0);
        Vec3d result{};
        for(int index = 0; index < count; ++index)
        {
            auto const m = multiIndices[index];
            if(m.degree != Order + 1)
                continue;
            double const weight = absoluteMoments[index];
            int const minusX = momentMinusOne[0][index];
            int const minusY = momentMinusOne[1][index];
            int const minusZ = momentMinusOne[2][index];
            result.x += weight * (X.x * w[index] + (minusX >= 0 ? w[minusX] : 0.0));
            result.y += weight * (X.y * w[index] + (minusY >= 0 ? w[minusY] : 0.0));
            result.z += weight * (X.z * w[index] + (minusZ >= 0 ? w[minusZ] : 0.0));
        }
        return result;
    }

    /** Lab-frame E and B (already including the node charge) of one far node. */
    template<int Order>
    ALPAKA_FN_HOST_ACC inline void multipoleFarField(
        Vec3d const& displacement, //!< target minus node centre, dx, lab frame
        Vec3d const& u, //!< node representative velocity
        double const* moments,
        Vec3d& E,
        Vec3d& B)
    {
        auto const frame = makeVelocityFrame(u);
        Vec3d const X = applyA(frame, displacement);
        Vec3d const transformed = multipoleFieldTransformed<Order>(X, moments);
        E = applyAInverse(frame, transformed);
        E = Vec3d{frame.gamma * E.x, frame.gamma * E.y, frame.gamma * E.z};
        double const inverseGamma = 1.0 / frame.gamma;
        Vec3d const beta{u.x * inverseGamma, u.y * inverseGamma, u.z * inverseGamma};
        B = Vec3d{
            beta.y * E.z - beta.z * E.y,
            beta.z * E.x - beta.x * E.z,
            beta.x * E.y - beta.y * E.x};
    }

} // namespace picongpu::particles::neuralBunch::farfield
