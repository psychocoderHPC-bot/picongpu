/* Copyright 2016-2024 Rene Widera
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

#pragma once

#include "pmacc/types.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace pmacc
{
    namespace memory
    {
        /** static sized array
         *
         * mimic the most parts of the `std::array`
         */
        template<typename T_Type, size_t T_size>
        struct Array
        {
            /* Workaround for an icpx 2026.2 SYCL CPU (spir64_x86_64) defect: the fully inlined
             * shape-array interpolation returned NaN for finite field values, although the array
             * content and element addresses were verified correct. An explicit noinline call
             * boundary reproduces the fix. That boundary is the element lifetime here: elements are
             * constructed individually and accessed through typed pointers instead of being
             * reinterpreted out of an untyped byte buffer, so the shape values cannot be folded
             * into the interpolation loops.
             */

            using value_type = T_Type;
            using size_type = size_t;
            using reference = value_type&;
            using const_reference = value_type const&;
            using pointer = value_type*;
            using const_pointer = value_type const*;

            /** get number of elements */
            HDINLINE
            constexpr size_type size() const
            {
                return T_size;
            }

            /** get maximum number of elements */
            HDINLINE
            constexpr size_type max_size() const
            {
                return T_size;
            }

            /** get the direct access to the internal data
             *
             * @{
             */
            HDINLINE
            pointer data()
            {
                return reinterpret_cast<pointer>(m_data);
            }

            HDINLINE
            const_pointer data() const
            {
                return reinterpret_cast<const_pointer>(m_data);
            }

            /** @} */

            /** default constructor
             *
             * all members are uninitialized
             */
            Array() = default;

            /** copy constructor
             *
             * This has to be user-provided: a generic constructor below would
             * otherwise be a better match for a non-const lvalue and would
             * attempt to construct a single element from the array itself.
             */
            HDINLINE Array(Array const& other)
            {
                copyConstructFrom(other);
            }

            HDINLINE Array(Array&& other) noexcept(std::is_nothrow_move_constructible_v<T_Type>)
            {
                for(size_type i = 0; i < size(); ++i)
                    std::construct_at(elemPtr(i), std::move(other[i]));
            }

            HDINLINE Array& operator=(Array const& other)
            {
                if(this != &other)
                {
                    for(size_type i = 0; i < size(); ++i)
                        elemPtr(i)[0] = other[i];
                }
                return *this;
            }

            HDINLINE Array& operator=(Array&& other) noexcept(std::is_nothrow_move_assignable_v<T_Type>)
            {
                if(this != &other)
                {
                    for(size_type i = 0; i < size(); ++i)
                        elemPtr(i)[0] = std::move(other[i]);
                }
                return *this;
            }

            /** constructor
             *
             * initialize each member with the given value
             *
             * @param value element assigned to each member
             */
            template<typename... T_Args>
                requires(sizeof...(T_Args) > 0) && (!std::is_same_v<std::decay_t<T_Args>, Array<T_Type, T_size>> && ...)
                     && (std::is_constructible_v<T_Type, T_Args&&...>)
            HDINLINE Array(T_Args&&... args)
            {
                for(size_type i = 0; i < size(); ++i)
                    std::construct_at(elemPtr(i), T_Type{std::forward<T_Args>(args)...});
            }

            /** get N-th value
             *
             * @tparam T_Idx any type which can be implicit casted to an integral type
             * @param idx index within the array
             *
             * @{
             */
            template<typename T_Idx>
            HDINLINE const_reference operator[](T_Idx const idx) const
            {
                return reinterpret_cast<const_pointer>(m_data)[idx];
            }

            template<typename T_Idx>
            HDINLINE reference operator[](T_Idx const idx)
            {
                return elemPtr(idx)[0];
            }

            /** @} */

        private:
            /** Array does not call element destructors, so only types with a
             * trivial destructor are allowed.
             */
            static_assert(
                std::is_trivially_destructible_v<T_Type>,
                "pmacc::memory::Array requires a trivially destructible element type");

            /** data storage
             *
             * std::array is a so-called "aggregate" which does not default-initialize
             * its members. In order to allow arbitrary types to skip implementing
             * a default constructor, this member is not stored as
             * `value_type m_data[ T_size ]` but as a type-size aligned byte array.
             * Elements are created with placement construction.
             */
            alignas(T_Type) std::byte m_data[T_size * sizeof(T_Type)];

            /** get a pointer to the storage of the i-th element */
            HDINLINE pointer elemPtr(size_type const i)
            {
                return reinterpret_cast<pointer>(m_data) + i;
            }

            HDINLINE void copyConstructFrom(Array const& other)
            {
                for(size_type i = 0; i < size(); ++i)
                    std::construct_at(elemPtr(i), other[i]);
            }
        };

    } // namespace memory
} // namespace pmacc
