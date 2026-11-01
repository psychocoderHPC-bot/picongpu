/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Configurable near/mid/far band classification for a (source, tile) pair.
 *
 * The distance is the minimum over the 1536 staggered output points of |x_out - x_source| in dx units, exactly the
 * MODEL-FORMAT ``domain.distance_metric``.  Output positions include the Yee component offsets and the tile aspect
 * ``cells = (1, a_y, a_z)``, so anisotropic cells are handled without approximation.
 *
 * Defaults: ``d < 4`` -> near, ``4 <= d <= 48`` -> mid, ``d > 48`` -> far.  Boundaries are half-open at the
 * near/mid edge and inclusive at the far edge (a distance exactly 4.0 is mid, exactly 48.0 is mid).  Thresholds
 * are runtime-configurable and validated by the constructor.
 */

#include "PointField.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace picongpu::particles::neuralBunch
{
    enum class Band
    {
        near,
        mid,
        far
    };

    constexpr char const* bandName(Band band)
    {
        switch(band)
        {
        case Band::near:
            return "near";
        case Band::mid:
            return "mid";
        case Band::far:
            return "far";
        }
        return "unknown";
    }

    struct BandThresholds
    {
        double nearMaxCells = 4.0;
        double farMinCells = 48.0;
    };

    inline void validateBandThresholds(BandThresholds const& thresholds)
    {
        if(!std::isfinite(thresholds.nearMaxCells) || !std::isfinite(thresholds.farMinCells))
            throw std::invalid_argument{"band thresholds must be finite."};
        if(thresholds.nearMaxCells < 0.0)
            throw std::invalid_argument{"nearMaxCells must be non-negative."};
        if(!(thresholds.farMinCells > thresholds.nearMaxCells))
            throw std::invalid_argument{"farMinCells must be greater than nearMaxCells."};
    }

    inline Band classifyBand(double distanceCells, BandThresholds const& thresholds = {})
    {
        if(distanceCells < thresholds.nearMaxCells)
            return Band::near;
        if(distanceCells > thresholds.farMinCells)
            return Band::far;
        return Band::mid;
    }

    /** Minimum physical (dx-unit) distance from a source to any of the 1536 staggered outputs. */
    inline double minStaggeredDistance(Vec3d const& sourceR, Vec3d const& cells)
    {
        double best = std::numeric_limits<double>::infinity();
        for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
        {
            auto const position = staggeredPosition(slot, cells);
            auto const dx = position.x - sourceR.x;
            auto const dy = position.y - sourceR.y;
            auto const dz = position.z - sourceR.z;
            best = std::min(best, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        return best;
    }

    /** How the banded runtime may use a trained model instead of the exact reference.
     *
     * ``useModelForMid`` is tri-state: ``std::nullopt`` means "use the model when the artifact provides one",
     * ``true``/``false`` force the decision. ``useModelForNear`` defaults to false.
     */
    struct BandedModelUse
    {
        bool useModelForNear = false;
        std::optional<bool> useModelForMid{};
    };

    enum class BandRoute
    {
        model,
        reference,
        error
    };

    /** Effective model decision for one band. Far is always analytic/model data, never reference. */
    inline bool bandUsesModel(
        Band band,
        BandedModelUse const& use,
        bool hasNearModel,
        bool hasMidModel,
        bool hasPrimaryModel)
    {
        if(band == Band::far)
            return true;
        if(band == Band::near)
            return use.useModelForNear && (hasNearModel || hasPrimaryModel);
        // mid: nullopt = automatic (model iff configured), explicit value is honoured.
        bool const requested = use.useModelForMid.value_or(hasMidModel || hasPrimaryModel);
        return requested && (hasMidModel || hasPrimaryModel);
    }

    /** Route one classified pair. A requested model band without an artifact is a hard error. */
    inline BandRoute routeBandedBand(
        Band band,
        BandedModelUse const& use,
        bool hasNearModel,
        bool hasMidModel,
        bool hasPrimaryModel,
        bool enableReferenceFallback)
    {
        if(band == Band::far)
            return BandRoute::model;
        bool const requested
            = band == Band::near ? use.useModelForNear : use.useModelForMid.value_or(hasMidModel || hasPrimaryModel);
        if(requested)
            return bandUsesModel(band, use, hasNearModel, hasMidModel, hasPrimaryModel) ? BandRoute::model
                                                                                        : BandRoute::error;
        return enableReferenceFallback ? BandRoute::reference : BandRoute::error;
    }

    /** True when this band should emit the once-per-rank unvalidated-model warning. */
    inline bool needsUnvalidatedModelWarning(
        Band band,
        BandedModelUse const& use,
        bool hasNearModel,
        bool hasMidModel,
        bool hasPrimaryModel)
    {
        if(band == Band::far)
            return false;
        if(band == Band::near)
            return use.useModelForNear && (hasNearModel || hasPrimaryModel);
        return bandUsesModel(band, use, hasNearModel, hasMidModel, hasPrimaryModel);
    }

    class BandPolicy
    {
    public:
        explicit BandPolicy(BandThresholds thresholds = {}) : m_thresholds(thresholds)
        {
            validateBandThresholds(m_thresholds);
        }

        [[nodiscard]] Band classifyDistance(double distanceCells) const
        {
            return classifyBand(distanceCells, m_thresholds);
        }

        [[nodiscard]] Band classify(Vec3d const& sourceR, Vec3d const& cells) const
        {
            return classifyDistance(minStaggeredDistance(sourceR, cells));
        }

        [[nodiscard]] double distance(Vec3d const& sourceR, Vec3d const& cells) const
        {
            return minStaggeredDistance(sourceR, cells);
        }

        [[nodiscard]] BandThresholds const& thresholds() const
        {
            return m_thresholds;
        }

    private:
        BandThresholds m_thresholds;
    };
} // namespace picongpu::particles::neuralBunch
