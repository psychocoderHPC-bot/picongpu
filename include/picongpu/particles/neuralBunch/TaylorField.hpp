/* SPDX-FileCopyrightText: René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

/** Low-order polynomial (Taylor/multipole) far-field basis over the 1536 staggered outputs.
 *
 * Coordinates
 * -----------
 * ``x_n = x_out - origin`` with all lengths in dx units (dimensionless "normalized" tile coordinates).  Output
 * positions are the Yee-staggered positions from PointField.hpp, so ``dy = a_y`` and ``dz = a_z`` enter the
 * coordinates and therefore the evaluation matrix.  The origin is configurable:
 *   - ``cell_origin``: (0, 0, 0)
 *   - ``tile_centre``: (nx/2, ny/2*a_y, nz/2*a_z) = (4, 4*a_y, 2*a_z)
 *
 * Basis and index order
 * ---------------------
 * For order n the monomial set is all multi-indices (ax, ay, az) with ax+ay+az <= n, enumerated as
 *   degree d = 0..n, then ax = 0..d, then ay = 0..d-ax, az = d-ax-ay.
 * The number of monomials is M = C(n+3, 3).  With six components the coefficient count is
 *   K = 6*M = 6*C(n+3, 3).
 * Index order is component-major: k = component*M + monomial.  Output row ``slot`` of the evaluation matrix
 * A[1536,K] is non-zero only in the block of its own component and holds the monomial values at that output.
 *
 * A is generated analytically on the host in FP64.  ``taylorEvaluateHost`` evaluates A*c (+ optional per-component
 * scale).  The device path computes ``out[1,1536] = c[1,K] * A^T[K,1536]`` with one vendor GEMM.  The projection
 * routine solves the normal equations with a Cholesky factorisation (ridge ``1e-12`` by default) and is used to
 * produce training targets from FP64 reference tiles; ``taylorPseudoInverse`` returns the explicit K x 1536
 * pseudo-inverse.
 */

#include "PointField.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/nn/onHost/matrix/gemm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace picongpu::particles::neuralBunch
{
    inline constexpr std::size_t kTaylorComponents = 6;

    enum class TaylorOrigin
    {
        cellOrigin,
        tileCentre
    };

    inline std::optional<TaylorOrigin> taylorOriginFromString(std::string_view name)
    {
        if(name == "cell_origin")
            return TaylorOrigin::cellOrigin;
        if(name == "tile_centre")
            return TaylorOrigin::tileCentre;
        return std::nullopt;
    }

    constexpr std::string_view taylorOriginName(TaylorOrigin origin)
    {
        return origin == TaylorOrigin::cellOrigin ? "cell_origin" : "tile_centre";
    }

    struct MultiIndex
    {
        int ax{};
        int ay{};
        int az{};

        [[nodiscard]] constexpr int degree() const
        {
            return ax + ay + az;
        }
    };

    constexpr std::size_t binomial(std::size_t n, std::size_t k)
    {
        if(k > n)
            return 0u;
        k = std::min(k, n - k);
        std::size_t result = 1u;
        for(std::size_t i = 1u; i <= k; ++i)
            result = result * (n - k + i) / i;
        return result;
    }

    /** M = C(n+3,3): number of 3-D monomials with total degree <= n. */
    inline std::size_t taylorMonomialCount(int order)
    {
        if(order < 0 || order > 20)
            throw std::invalid_argument{"Taylor order must be in [0, 20]."};
        return binomial(static_cast<std::size_t>(order) + 3u, 3u);
    }

    /** K = 6*C(n+3,3). */
    inline std::size_t taylorBasisSize(int order)
    {
        return kTaylorComponents * taylorMonomialCount(order);
    }

    inline std::vector<MultiIndex> taylorMultiIndices(int order)
    {
        auto const count = taylorMonomialCount(order);
        std::vector<MultiIndex> indices;
        indices.reserve(count);
        for(int degree = 0; degree <= order; ++degree)
            for(int ax = 0; ax <= degree; ++ax)
                for(int ay = 0; ay <= degree - ax; ++ay)
                    indices.push_back(MultiIndex{ax, ay, degree - ax - ay});
        return indices;
    }

    inline Vec3d taylorOriginPoint(TaylorOrigin origin, Vec3d const& cells)
    {
        if(origin == TaylorOrigin::cellOrigin)
            return {0.0, 0.0, 0.0};
        return {
            0.5 * static_cast<double>(kTileNxPoint) * cells.x,
            0.5 * static_cast<double>(kTileNyPoint) * cells.y,
            0.5 * static_cast<double>(kTileNzPoint) * cells.z};
    }

    ALPAKA_FN_HOST_ACC inline double monomialValue(MultiIndex const& index, Vec3d const& coordinate)
    {
        double value = 1.0;
        for(int i = 0; i < index.ax; ++i)
            value *= coordinate.x;
        for(int i = 0; i < index.ay; ++i)
            value *= coordinate.y;
        for(int i = 0; i < index.az; ++i)
            value *= coordinate.z;
        return value;
    }

    ALPAKA_FN_HOST_ACC inline double monomialDerivative(MultiIndex const& index, Vec3d const& coordinate, int axis)
    {
        int exponent = index.ax;
        if(axis == 1)
            exponent = index.ay;
        else if(axis == 2)
            exponent = index.az;
        if(exponent == 0)
            return 0.0;
        double value = static_cast<double>(exponent);
        for(int i = 0; i < (axis == 0 ? index.ax - 1 : index.ax); ++i)
            value *= coordinate.x;
        for(int i = 0; i < (axis == 1 ? index.ay - 1 : index.ay); ++i)
            value *= coordinate.y;
        for(int i = 0; i < (axis == 2 ? index.az - 1 : index.az); ++i)
            value *= coordinate.z;
        return value;
    }

    inline Vec3d taylorCoordinates(std::size_t slot, Vec3d const& cells, TaylorOrigin origin)
    {
        auto const position = staggeredPosition(slot, cells);
        auto const originPoint = taylorOriginPoint(origin, cells);
        return {position.x - originPoint.x, position.y - originPoint.y, position.z - originPoint.z};
    }

    /** A[1536,K] row-major, row = output slot, column = component*M + monomial. */
    inline std::vector<double> taylorMatrixHost(int order, Vec3d const& cells, TaylorOrigin origin)
    {
        auto const monomials = taylorMultiIndices(order);
        auto const M = monomials.size();
        auto const K = kTaylorComponents * M;
        std::vector<double> matrix(kTileSlotCount * K, 0.0);
        for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
        {
            auto const coordinate = taylorCoordinates(slot, cells, origin);
            auto const component = slot % 6u;
            for(std::size_t m = 0u; m < M; ++m)
                matrix[slot * K + component * M + m] = monomialValue(monomials[m], coordinate);
        }
        return matrix;
    }

    /** A^T[K,1536] row-major, the layout used by the device GEMM. */
    inline std::vector<double> taylorMatrixTransposedHost(
        std::vector<double> const& matrix,
        std::size_t slots,
        std::size_t K)
    {
        if(matrix.size() != slots * K)
            throw std::invalid_argument{"taylorMatrixTransposedHost: matrix shape mismatch."};
        std::vector<double> transposed(K * slots, 0.0);
        for(std::size_t slot = 0u; slot < slots; ++slot)
            for(std::size_t k = 0u; k < K; ++k)
                transposed[k * slots + slot] = matrix[slot * K + k];
        return transposed;
    }

    inline double taylorValueAtComponent(
        int order,
        std::vector<double> const& coefficients,
        std::size_t component,
        Vec3d const& coordinate)
    {
        auto const monomials = taylorMultiIndices(order);
        auto const M = monomials.size();
        double value = 0.0;
        for(std::size_t m = 0u; m < M; ++m)
            value += coefficients[component * M + m] * monomialValue(monomials[m], coordinate);
        return value;
    }

    inline double taylorDerivativeAtComponent(
        int order,
        std::vector<double> const& coefficients,
        std::size_t component,
        Vec3d const& coordinate,
        int axis)
    {
        auto const monomials = taylorMultiIndices(order);
        auto const M = monomials.size();
        double value = 0.0;
        for(std::size_t m = 0u; m < M; ++m)
            value += coefficients[component * M + m] * monomialDerivative(monomials[m], coordinate, axis);
        return value;
    }

    /** Host evaluation out[slot] = sum_k A[slot,k]*coeff[k], with optional per-component scale (length 6). */
    inline std::vector<double> taylorEvaluateHost(
        std::vector<double> const& matrix,
        int order,
        std::vector<double> const& coefficients,
        std::vector<double> const& componentScale = {})
    {
        auto const K = taylorBasisSize(order);
        if(matrix.size() != kTileSlotCount * K || coefficients.size() != K)
            throw std::invalid_argument{"taylorEvaluateHost: matrix or coefficient shape mismatch."};
        if(!componentScale.empty() && componentScale.size() != kTaylorComponents)
            throw std::invalid_argument{"taylorEvaluateHost: componentScale must have six entries."};
        std::vector<double> out(kTileSlotCount, 0.0);
        for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
        {
            double value = 0.0;
            for(std::size_t k = 0u; k < K; ++k)
                value += matrix[slot * K + k] * coefficients[k];
            auto const component = slot % 6u;
            out[slot] = componentScale.empty() ? value : value * componentScale[component];
        }
        return out;
    }

    namespace detail
    {
        inline void choleskyFactor(std::vector<double>& matrix, std::size_t n)
        {
            for(std::size_t i = 0u; i < n; ++i)
            {
                for(std::size_t j = 0u; j <= i; ++j)
                {
                    double sum = matrix[i * n + j];
                    for(std::size_t k = 0u; k < j; ++k)
                        sum -= matrix[i * n + k] * matrix[j * n + k];
                    if(i == j)
                    {
                        if(!(sum > 0.0))
                            throw std::runtime_error{"Taylor projection: normal matrix is not positive definite."};
                        matrix[i * n + i] = std::sqrt(sum);
                    }
                    else
                    {
                        matrix[i * n + j] = sum / matrix[j * n + j];
                    }
                }
            }
        }

        inline void choleskySolve(std::vector<double> const& lower, std::size_t n, std::vector<double>& rhs)
        {
            for(std::size_t i = 0u; i < n; ++i)
            {
                double sum = rhs[i];
                for(std::size_t k = 0u; k < i; ++k)
                    sum -= lower[i * n + k] * rhs[k];
                rhs[i] = sum / lower[i * n + i];
            }
            for(std::size_t i = n; i-- > 0u;)
            {
                double sum = rhs[i];
                for(std::size_t k = i + 1u; k < n; ++k)
                    sum -= lower[k * n + i] * rhs[k];
                rhs[i] = sum / lower[i * n + i];
            }
        }

        inline std::vector<double> normalMatrix(
            std::vector<double> const& matrix,
            std::size_t rows,
            std::size_t K,
            double ridge)
        {
            std::vector<double> normal(K * K, 0.0);
            for(std::size_t row = 0u; row < rows; ++row)
                for(std::size_t a = 0u; a < K; ++a)
                {
                    double const value = matrix[row * K + a];
                    if(value == 0.0)
                        continue;
                    for(std::size_t b = 0u; b < K; ++b)
                        normal[a * K + b] += value * matrix[row * K + b];
                }
            for(std::size_t a = 0u; a < K; ++a)
                normal[a * K + a] += ridge;
            return normal;
        }
    } // namespace detail

    /** Least-squares projection coefficients = pinv(A) * referenceTile. */
    inline std::vector<double> taylorProjectHost(
        std::vector<double> const& matrix,
        int order,
        std::vector<double> const& referenceTile,
        double ridge = 1.0e-12)
    {
        auto const K = taylorBasisSize(order);
        if(matrix.size() != kTileSlotCount * K || referenceTile.size() != kTileSlotCount)
            throw std::invalid_argument{"taylorProjectHost: shape mismatch."};
        auto normal = detail::normalMatrix(matrix, kTileSlotCount, K, ridge);
        std::vector<double> rhs(K, 0.0);
        for(std::size_t row = 0u; row < kTileSlotCount; ++row)
            for(std::size_t k = 0u; k < K; ++k)
                rhs[k] += matrix[row * K + k] * referenceTile[row];
        detail::choleskyFactor(normal, K);
        detail::choleskySolve(normal, K, rhs);
        return rhs;
    }

    /** Explicit K x 1536 pseudo-inverse (row-major), pinv = (A^T A + ridge I)^-1 A^T. */
    inline std::vector<double> taylorPseudoInverse(
        std::vector<double> const& matrix,
        int order,
        double ridge = 1.0e-12)
    {
        auto const K = taylorBasisSize(order);
        if(matrix.size() != kTileSlotCount * K)
            throw std::invalid_argument{"taylorPseudoInverse: shape mismatch."};
        auto normal = detail::normalMatrix(matrix, kTileSlotCount, K, ridge);
        detail::choleskyFactor(normal, K);
        std::vector<double> pseudo(K * kTileSlotCount, 0.0);
        std::vector<double> column(K, 0.0);
        for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
        {
            for(std::size_t k = 0u; k < K; ++k)
                column[k] = matrix[slot * K + k];
            detail::choleskySolve(normal, K, column);
            for(std::size_t k = 0u; k < K; ++k)
                pseudo[k * kTileSlotCount + slot] = column[k];
        }
        return pseudo;
    }

    namespace detail
    {
        /** Native coefficient-major GEMV: out[slot] = sum_k matrix[k*1536+slot] * coeff[k].
         *
         * One work item per output slot. Consecutive slots read consecutive addresses in row k of the
         * coefficient-major matrix, so the global reads are coalesced. Coefficients are loaded into a small
         * register block and reused; the accumulation order is fixed (k ascending) for determinism.
         */
        template<typename T_Type>
        struct TaylorGemvKernel
        {
            std::uint32_t K{};

            ALPAKA_FN_ACC void operator()(auto const& acc, auto matrix, auto coefficients, auto out) const
            {
                for(auto index : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{kTileSlotCount}}))
                {
                    auto const slot = static_cast<std::size_t>(index[0]);
                    T_Type sum = T_Type{0};
                    std::size_t k = 0u;
                    constexpr std::size_t block = 8u;
                    for(; k + block <= static_cast<std::size_t>(K); k += block)
                    {
                        T_Type coefficient[block];
                        for(std::size_t j = 0u; j < block; ++j)
                            coefficient[j] = coefficients[alpaka::Vec<std::size_t, 1>{k + j}];
                        for(std::size_t j = 0u; j < block; ++j)
                            sum += matrix[alpaka::Vec<std::size_t, 2>{k + j, slot}] * coefficient[j];
                    }
                    for(; k < static_cast<std::size_t>(K); ++k)
                        sum += matrix[alpaka::Vec<std::size_t, 2>{k, slot}]
                               * coefficients[alpaka::Vec<std::size_t, 1>{k}];
                    out[alpaka::Vec<std::size_t, 1>{slot}] = sum;
                }
            }
        };

        template<typename T_Type>
        struct TaylorBatchedKernel
        {
            std::uint32_t K{};
            std::uint32_t S{};

            ALPAKA_FN_ACC void operator()(auto const& acc, auto matrix, auto coefficients, auto out) const
            {
                auto const total = static_cast<std::size_t>(S) * kTileSlotCount;
                for(auto index : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{total}}))
                {
                    auto const flat = static_cast<std::size_t>(index[0]);
                    auto const source = flat / kTileSlotCount;
                    auto const slot = flat % kTileSlotCount;
                    T_Type sum = T_Type{0};
                    for(std::size_t k = 0u; k < static_cast<std::size_t>(K); ++k)
                        sum += matrix[alpaka::Vec<std::size_t, 2>{k, slot}]
                               * coefficients[alpaka::Vec<std::size_t, 2>{source, k}];
                    out[alpaka::Vec<std::size_t, 2>{source, slot}] = sum;
                }
            }
        };
    } // namespace detail

    /** Native device GEMV: out[1536] = A^T[K,1536] * coeff[K], no vendor BLAS and no per-call handle.
     *
     * @p matrix is coefficient-major (row k holds all 1536 output coefficients); @p coefficients is a 1-D view of
     * length K; @p out is a 1-D view of length 1536. The caller keeps all views alive until the queue work completes.
     */
    template<
        typename T_Type = float,
        typename TQueue,
        typename TExec,
        typename TMatrix,
        typename TCoeff,
        typename TOut>
    void taylorGemv(TQueue& queue, TExec exec, TMatrix const& matrix, TCoeff const& coefficients, TOut& out)
    {
        auto const K = static_cast<std::uint32_t>(matrix.getExtents()[0]);
        if(static_cast<std::size_t>(matrix.getExtents()[1]) != kTileSlotCount)
            throw std::invalid_argument{"taylorGemv: matrix must be [K,1536]."};
        if(static_cast<std::size_t>(coefficients.getExtents()[0]) != static_cast<std::size_t>(K))
            throw std::invalid_argument{"taylorGemv: coefficient length does not match K."};
        using Vec = alpaka::Vec<std::uint32_t, 1>;
        auto const threads = Vec{128u};
        auto const frames = Vec{static_cast<std::uint32_t>((kTileSlotCount + 127u) / 128u)};
        queue.enqueue(
            alpaka::onHost::FrameSpec{frames, threads, exec},
            alpaka::KernelBundle{detail::TaylorGemvKernel<T_Type>{K}, matrix, coefficients, out});
    }

    /** Batched native GEMV: out[S,1536] = A^T[K,1536] * coeff[S,K] with one work item per (source, output). */
    template<
        typename T_Type = float,
        typename TQueue,
        typename TExec,
        typename TMatrix,
        typename TCoeff,
        typename TOut>
    void taylorBatchedNative(TQueue& queue, TExec exec, TMatrix const& matrix, TCoeff const& coefficients, TOut& out)
    {
        auto const K = static_cast<std::uint32_t>(matrix.getExtents()[0]);
        auto const S = static_cast<std::uint32_t>(coefficients.getExtents()[0]);
        if(static_cast<std::size_t>(matrix.getExtents()[1]) != kTileSlotCount)
            throw std::invalid_argument{"taylorBatchedNative: matrix must be [K,1536]."};
        if(static_cast<std::size_t>(coefficients.getExtents()[1]) != static_cast<std::size_t>(K))
            throw std::invalid_argument{"taylorBatchedNative: coefficient matrix must be [S,K]."};
        if(static_cast<std::size_t>(out.getExtents()[0]) != static_cast<std::size_t>(S)
           || static_cast<std::size_t>(out.getExtents()[1]) != kTileSlotCount)
            throw std::invalid_argument{"taylorBatchedNative: output must be [S,1536]."};
        if(K == 0u || S == 0u)
            return;

        using namespace alpaka;
        using Vec = alpaka::Vec<std::uint32_t, 1>;
        auto const total = static_cast<std::uint32_t>(static_cast<std::size_t>(S) * kTileSlotCount);
        auto const threads = Vec{128u};
        auto const frames = Vec{(total + 127u) / 128u};
        queue.enqueue(
            alpaka::onHost::FrameSpec{frames, threads, exec},
            alpaka::KernelBundle{detail::TaylorBatchedKernel<T_Type>{K, S}, matrix, coefficients, out});
    }

    /** Batched vendor-BLAS GEMV: one [S,K] x [K,1536] GEMM; amortises the vendor handle over S. */
    template<
        typename T_Type = float,
        typename TQueue,
        typename TExec,
        typename TMatrix,
        typename TCoeff,
        typename TOut>
    void taylorBatchedVendor(TQueue& queue, TExec exec, TMatrix const& matrix, TCoeff const& coefficients, TOut& out)
    {
        (void) exec;
        alpaka::nn::onHost::gemm<T_Type>(queue, coefficients, matrix, out);
    }

    enum class TaylorBatchBackend
    {
        automatic,
        native,
        vendor
    };

    /** Empirically chosen default: native kernel on CUDA (vendor handle cost dominates), vendor BLAS on CPU. */
    template<typename TDevice>
    constexpr TaylorBatchBackend defaultTaylorBatchBackend(TDevice const& device)
    {
        using Api = std::decay_t<decltype(alpaka::getApi(device))>;
        if constexpr(std::same_as<Api, alpaka::api::Cuda>)
            return TaylorBatchBackend::native;
        else
            return TaylorBatchBackend::vendor;
    }

    /** Host convenience for a coefficient batch: matrix is [1536,K], coefficients is [S,K]. */
    template<typename TQueue, typename TExec>
    std::vector<float> taylorEvaluateBatched(
        TQueue& queue,
        TExec exec,
        std::vector<double> const& matrix,
        int order,
        std::vector<float> const& coefficients,
        std::size_t batches,
        std::vector<double> const& componentScale = {},
        TaylorBatchBackend backend = TaylorBatchBackend::native)
    {
        auto const K = taylorBasisSize(order);
        if(matrix.size() != kTileSlotCount * K || coefficients.size() != batches * K)
            throw std::invalid_argument{"taylorEvaluateBatched: matrix or coefficient shape mismatch."};
        auto const transposed = taylorMatrixTransposedHost(matrix, kTileSlotCount, K);

        auto hostMatrix = alpaka::onHost::allocHost<float>(alpaka::Vec{K, kTileSlotCount});
        for(std::size_t k = 0u; k < K; ++k)
            for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
                hostMatrix[alpaka::Vec<std::size_t, 2>{k, slot}]
                    = static_cast<float>(transposed[k * kTileSlotCount + slot]);
        auto hostCoefficients = alpaka::onHost::allocHost<float>(alpaka::Vec{batches, K});
        for(std::size_t b = 0u; b < batches; ++b)
            for(std::size_t k = 0u; k < K; ++k)
                hostCoefficients[alpaka::Vec<std::size_t, 2>{b, k}] = coefficients[b * K + k];
        auto hostOut = alpaka::onHost::allocHost<float>(alpaka::Vec{batches, kTileSlotCount});

        auto deviceMatrix = alpaka::onHost::allocLike(queue.getDevice(), hostMatrix);
        auto deviceCoefficients = alpaka::onHost::allocLike(queue.getDevice(), hostCoefficients);
        auto deviceOut = alpaka::onHost::allocLike(queue.getDevice(), hostOut);
        alpaka::onHost::memcpy(queue, deviceMatrix, hostMatrix);
        alpaka::onHost::memcpy(queue, deviceCoefficients, hostCoefficients);
        hostMatrix.keepAlive(queue);
        hostCoefficients.keepAlive(queue);

        if(backend == TaylorBatchBackend::native)
            taylorBatchedNative<float>(queue, exec, deviceMatrix, deviceCoefficients, deviceOut);
        else
            taylorBatchedVendor<float>(queue, exec, deviceMatrix, deviceCoefficients, deviceOut);

        alpaka::onHost::memcpy(queue, hostOut, deviceOut);
        alpaka::onHost::wait(queue);

        std::vector<float> result(batches * kTileSlotCount);
        for(std::size_t b = 0u; b < batches; ++b)
            for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
            {
                double value = hostOut[alpaka::Vec<std::size_t, 2>{b, slot}];
                if(!componentScale.empty())
                    value *= componentScale[slot % 6u];
                result[b * kTileSlotCount + slot] = static_cast<float>(value);
            }
        return result;
    }

    /** Convenience device evaluation: uploads A^T and coefficients, applies optional per-component scale. */
    template<typename TQueue, typename TExec>
    std::vector<float> taylorEvaluateDevice(
        TQueue& queue,
        TExec exec,
        std::vector<double> const& matrix,
        int order,
        std::vector<double> const& coefficients,
        std::vector<double> const& componentScale = {})
    {
        auto const K = taylorBasisSize(order);
        if(matrix.size() != kTileSlotCount * K || coefficients.size() != K)
            throw std::invalid_argument{"taylorEvaluateDevice: shape mismatch."};
        auto const transposed = taylorMatrixTransposedHost(matrix, kTileSlotCount, K);
        auto hostMatrix = alpaka::onHost::allocHost<float>(alpaka::Vec{K, kTileSlotCount});
        for(std::size_t k = 0u; k < K; ++k)
            for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
                hostMatrix[alpaka::Vec<std::size_t, 2>{k, slot}]
                    = static_cast<float>(transposed[k * kTileSlotCount + slot]);
        auto deviceMatrix = alpaka::onHost::allocLike(queue.getDevice(), hostMatrix);
        alpaka::onHost::memcpy(queue, deviceMatrix, hostMatrix);
        hostMatrix.keepAlive(queue);

        auto hostCoefficients = alpaka::onHost::allocHost<float>(alpaka::Vec{K});
        for(std::size_t k = 0u; k < K; ++k)
            hostCoefficients[alpaka::Vec<std::size_t, 1>{k}] = static_cast<float>(coefficients[k]);
        auto deviceCoefficients = alpaka::onHost::allocLike(queue.getDevice(), hostCoefficients);
        alpaka::onHost::memcpy(queue, deviceCoefficients, hostCoefficients);
        hostCoefficients.keepAlive(queue);

        auto deviceOut = alpaka::onHost::alloc<float>(queue.getDevice(), alpaka::Vec{kTileSlotCount});
        taylorGemv<float>(queue, exec, deviceMatrix, deviceCoefficients, deviceOut);

        auto hostOut = alpaka::onHost::allocHost<float>(alpaka::Vec{kTileSlotCount});
        alpaka::onHost::memcpy(queue, hostOut, deviceOut);
        alpaka::onHost::wait(queue);
        std::vector<float> result(kTileSlotCount);
        for(std::size_t slot = 0u; slot < kTileSlotCount; ++slot)
        {
            double value = hostOut[alpaka::Vec<std::size_t, 1>{slot}];
            if(!componentScale.empty())
                value *= componentScale[slot % 6u];
            result[slot] = static_cast<float>(value);
        }
        return result;
    }
} // namespace picongpu::particles::neuralBunch
