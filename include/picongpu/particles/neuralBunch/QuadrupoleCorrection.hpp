/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Quadrupole (order-2) shape correction of the analytic point field in the mid band.
 *
 * The field of one finite TSC source is the translation-invariant linear functional
 *
 *   E_dist(R) = int S(s) K(R - s * cells, u) ds^3,
 *
 * with the normalized point kernel K (the same one ``pointFieldDx`` evaluates). Expanding K in the source
 * displacement ``s * cells`` about the source centre gives an exact-in-form Taylor series whose coefficients are
 * the TSC shape moments. The dipole vanishes by symmetry, so the leading correction is the quadrupole
 *
 *   E_corr,c = (1/2) * m2 * sum_i cells_i^2 * d_i d_i E_c,
 *
 * where ``m2 = 1/4`` is the second moment of the 1-D TSC shape (in cell units). The magnetic correction follows
 * from B = beta x E, so it uses the same coefficients.
 *
 * Conventions: R is the target minus the source in dx units, ``u = p / (m c)``, ``cells = (1, a_y, a_z)``.
 * The correction is only valid outside the TSC support (distance > 1.5 cells); the near band keeps the exact
 * quadrature. The mid band (4 <= d <= 48 cells by default) is the intended caller, which is why the runtime
 * applies it only for pairs the band policy classifies as mid.
 */

#include "Features.hpp"

#include <alpaka/alpaka.hpp>

namespace picongpu::particles::neuralBunch
{
    /** Second moment of the 1-D TSC assignment shape, in cell units. */
    inline constexpr double kTscSecondMoment = 0.25;

    /** Quadrupole correction to the normalized point field, in dx units. */
    ALPAKA_FN_HOST_ACC inline PointFieldDx quadrupoleCorrectionDx(
        Vec3d const& R,
        Vec3d const& u,
        Vec3d const& cells)
    {
        double const uu2 = u.x * u.x + u.y * u.y + u.z * u.z;
        double const invG2 = 1.0 / (1.0 + uu2);
        double const gamma = alpaka::math::sqrt(1.0 + uu2);
        Vec3d const beta{u.x / gamma, u.y / gamma, u.z / gamma};

        double const r2 = R.x * R.x + R.y * R.y + R.z * R.z;
        double const rb = R.x * beta.x + R.y * beta.y + R.z * beta.z;
        double const D = r2 * invG2 + rb * rb;
        if(D <= 0.0)
            return {};
        double const h5 = 1.0 / (D * D * alpaka::math::sqrt(D));

        // The odd (dipole) moment vanishes, so only the three diagonal second derivatives survive. For component j
        // the diagonal derivative split into the j term (own axis, cell scale 1) and the two cross axes.
        double const Rv[3]{R.x, R.y, R.z};
        double const bv[3]{beta.x, beta.y, beta.z};
        double const scaleSq[3]{1.0, cells.y * cells.y, cells.z * cells.z};
        double const prefactor = -1.5 * kTscSecondMoment * h5 * invG2;

        double a[3];
        for(int i = 0; i < 3; ++i)
        {
            double const di = Rv[i] * invG2 + bv[i] * rb;
            a[i] = bv[i] * bv[i] + invG2 - 5.0 * di * di / D;
        }

        double corr[3];
        for(int j = 0; j < 3; ++j)
        {
            double sum = 0.0;
            for(int i = 0; i < 3; ++i)
                sum += (i == j) ? scaleSq[i] * (2.0 * Rv[j] * invG2 + Rv[j] * a[i] + 2.0 * bv[j] * rb)
                                : scaleSq[i] * Rv[j] * a[i];
            corr[j] = prefactor * sum;
        }

        PointFieldDx result;
        result.E = Vec3d{corr[0], corr[1], corr[2]};
        result.B = Vec3d{
            beta.y * result.E.z - beta.z * result.E.y,
            beta.z * result.E.x - beta.x * result.E.z,
            beta.x * result.E.y - beta.y * result.E.x};
        return result;
    }
} // namespace picongpu::particles::neuralBunch
