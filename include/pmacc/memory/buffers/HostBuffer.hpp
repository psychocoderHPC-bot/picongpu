/* Copyright 2013-2024 Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/alpakaHelper/Device.hpp"
#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/assert.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/eventSystem/eventSystem.hpp"
#include "pmacc/eventSystem/tasks/Factory.hpp"
#include "pmacc/memory/Array.hpp"
#include "pmacc/memory/boxes/DataBoxDim1Access.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/memory/buffers/Buffer.hpp"

namespace pmacc
{
    class EventTask;

    template<class TYPE, unsigned DIM>
    class DeviceBuffer;

    template<class TYPE, unsigned DIM>
    class Buffer;

    /** N-dimensional host buffer
     *
     * The memory is allocated as pinned host-mapped memory of the compute device so that
     * kernels can access it. Views onto the memory carry the host API which is required to
     * select the correct copy direction for host/device transfers.
     *
     * @tparam T_Type datatype for buffer data
     * @tparam T_dim dimension of the buffer
     */
    template<typename T_Type, uint32_t T_dim>
    class HostBuffer : public Buffer<T_Type, T_dim>
    {
        //! flat owning mapped allocation, the N-dimensional view below adds tight pitches
        using BufferType = ComputeMappedBuffer<T_Type, DIM1>;
        //! plain (non-pinned) host allocation used by code which must not allocate mapped memory
        using PlainBufferType = decltype(::alpaka::onHost::alloc<T_Type>(
            std::declval<HostDevice const&>(),
            std::declval<::alpaka::Vec<MemIdxType, DIM1>>()));
        using ViewType = ::alpaka::View<HostApi, T_Type, ::alpaka::Vec<MemIdxType, T_dim>>;

    public:
        using DataBoxType = typename Buffer<T_Type, T_dim>::DataBoxType;
        std::optional<BufferType> hostBuffer;
        std::optional<PlainBufferType> hostPlainBuffer;
        std::optional<ViewType> view;

        using BufferType1D = ::alpaka::View<HostApi, T_Type, ::alpaka::Vec<MemIdxType, DIM1>>;

        BufferType1D as1DBuffer()
        {
            auto numElements = this->size();
            eventSystem::startOperation(ITask::TASK_HOST);
            return ::alpaka::makeView(*view, ::alpaka::onHost::data(*view), MemSpace<DIM1>(numElements).toAlpakaMemVec());
        }

        ViewType getAlpakaView() const
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            return *view;
        }

        /** allocate data accessible from the host
         *
         * @param size extent for each dimension (in elements)
         */
        HostBuffer(MemSpace<T_dim> size) : HostBuffer(size, true)
        {
        }

        /** allocate data accessible from the host
         *
         * @param size extent for each dimension (in elements)
         * @param mapped allocate host-mapped (pinned) memory if true, else plain host memory
         */
        HostBuffer(MemSpace<T_dim> size, bool mapped) : Buffer<T_Type, T_dim>(size)
        {
            T_Type* ptr = nullptr;
            if(mapped)
            {
                hostBuffer.emplace(::alpaka::onHost::allocMapped<T_Type>(
                    manager::Device<ComputeDevice>::get().current(),
                    MemSpace<DIM1>(size.productOfComponents()).toAlpakaMemVec()));
                ptr = ::alpaka::onHost::data(*hostBuffer);
            }
            else
            {
                hostPlainBuffer.emplace(::alpaka::onHost::alloc<T_Type>(
                    manager::Device<HostDevice>::get().current(),
                    MemSpace<DIM1>(size.productOfComponents()).toAlpakaMemVec()));
                ptr = ::alpaka::onHost::data(*hostPlainBuffer);
            }
            auto const extents = size.toAlpakaMemVec();
            view.emplace(::alpaka::makeView(
                ::alpaka::onHost::DeviceSpec<HostApi, HostDeviceKind>{HostApi{}, HostDeviceKind{}},
                ptr,
                extents,
                ::alpaka::calculatePitchesFromExtents<T_Type>(extents)));
            reset(false);
        }

        /** create a shallow view into an existing buffer
         *
         * @param source buffer to create the view on
         * @param size extent for each dimension (in elements)
         * @param offset offset within the source (in elements)
         *
         * @attention offset + size must be less or equal to the size of the source buffer
         */
        HostBuffer(HostBuffer& source, MemSpace<T_dim> size, MemSpace<T_dim> offset = MemSpace<T_dim>())
            : Buffer<T_Type, T_dim>(size)
            , hostBuffer(source.hostBuffer)
            , hostPlainBuffer(source.hostPlainBuffer)
        {
            auto* shiftedPtr = &(*source.view)[offset.toAlpakaMemVec()];
            view.emplace(::alpaka::makeView(
                ::alpaka::onHost::DeviceSpec<HostApi, HostDeviceKind>{HostApi{}, HostDeviceKind{}},
                shiftedPtr,
                size.toAlpakaMemVec(),
                ::alpaka::onHost::getPitches(*source.view)));
            reset(true);
        }

        ~HostBuffer() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
        }

        /** Copies the data from the given DeviceBuffer to this HostBuffer.
         *
         * @param other DeviceBuffer to copy data from
         */
        void copyFrom(DeviceBuffer<T_Type, T_dim>& other)
        {
            Environment<>::get().Factory().createTaskCopy(other, *this);
        }

        T_Type* data() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            return ::alpaka::onHost::data(*view);
        }

        void reset(bool preserveData = true) override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            this->setSize(this->capacityND().productOfComponents());
            if(!preserveData)
            {
                /* if it is a pointer out of other memory we can not assume that
                 * that the physical memory is contiguous
                 */
                if(hostBuffer && ::alpaka::onHost::data(*hostBuffer) == ::alpaka::onHost::data(*view))
                    memset(
                        reinterpret_cast<void*>(::alpaka::onHost::data(*view)),
                        0,
                        this->capacityND().productOfComponents() * sizeof(T_Type));
                else
                {
                    // Using Array is a workaround for types without default constructor
                    memory::Array<uint8_t, sizeof(T_Type)> tmp(uint8_t{0});
                    // use first element to avoid issue because Array is aligned (sizeof can be larger than component
                    // type)
                    setValue(*reinterpret_cast<T_Type*>(tmp.data()));
                }
            }
        }

        void setValue(T_Type const& value) override
        {
            // getDataBox is notifying the event system, no need to do it manually
            auto memBox = this->getDataBox();
            // narrowing conversion, implicit assumption when using a DataSpace is that size fits in int.
            auto current_size = static_cast<int>(this->size());
            using D1Box = DataBoxDim1Access<DataBoxType>;
            D1Box d1Box(memBox, this->capacityND());
#if defined(_OPENMP)
#    pragma omp parallel for
#endif
            for(int i = 0; i < current_size; i++)
            {
                d1Box[i] = value;
            }
        }

        DataBoxType getDataBox() override
        {
            eventSystem::startOperation(ITask::TASK_HOST);
            auto pitchBytes = MemSpace<T_dim>(::alpaka::onHost::getPitches(*view));
            return DataBoxType(PitchedBox<T_Type, T_dim>(::alpaka::onHost::data(*view), pitchBytes));
        }

        typename Buffer<T_Type, T_dim>::CPtr getCPtrCurrentSize() final
        {
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            // size is notifying the event system, no need to do it manually
            size_t const size = this->size();
            return {::alpaka::onHost::data(*view), size};
        }

        typename Buffer<T_Type, T_dim>::CPtr getCPtrCapacity() final
        {
            PMACC_ASSERT_MSG(this->isContiguous(), "Memory must be contiguous!");
            size_t const size = this->capacityND().productOfComponents();
            eventSystem::startOperation(ITask::TASK_HOST);
            return {::alpaka::onHost::data(*view), size};
        }
    };

} // namespace pmacc
