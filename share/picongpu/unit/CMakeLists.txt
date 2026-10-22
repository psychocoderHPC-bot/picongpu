# Copyright 2023-2024 Rene Widera
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

cmake_minimum_required(VERSION 3.28.0)
project("UnitTest")

###############################################################################
# Language Flags
###############################################################################

# enforce C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

################################################################################
# CMake policies
#
# Set the Cmake policy version. This sets the behaviour of the following policies
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/latest/policy/CMP0074.html
# Search in <PACKAGENAME>_ROOT:
#   https://cmake.org/cmake/help/latest/policy/CMP0144.html
################################################################################
cmake_policy(VERSION 3.28.0)

################################################################################
# PMacc
################################################################################
find_package(PMacc REQUIRED CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../include/pmacc")

###############################################################################
# Compile definitions
###############################################################################

# The unit tests do not link against openPMD; make the feature flag explicit so
# `#if ENABLE_OPENPMD == 1` in shared headers does not trip -Wundef.
add_definitions(-DENABLE_OPENPMD=0)

###############################################################################
# Warnings
###############################################################################

# Apply the same warning policy as the PIConGPU application build so the tests
# can be compiled with alpaka_COMPILE_PEDANTIC=ON (-Wall -Wextra -Werror).
# clang applies warning options in order, so these suppressions must be appended
# after alpaka's -Wall/-Wextra; linking an INTERFACE target last does that.
add_library(pmacc_unit_warning_policy INTERFACE)
if(PIC_CI_COMPILE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        pmacc_unit_warning_policy
        INTERFACE
        -Wno-unknown-pragmas
        -Wno-unused-parameter
        -Wno-unused-local-typedefs
    )
endif()
if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
    # clang-only diagnostics for pre-existing upstream code that GCC does not emit
    target_compile_options(
        pmacc_unit_warning_policy
        INTERFACE
        -Wno-deprecated-copy
        -Wno-delete-non-abstract-non-virtual-dtor
    )
endif()
# nvcc needs the host-pass suppressions forwarded with -Xcompiler.
if(CMAKE_CUDA_COMPILER)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unused-parameter>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unknown-pragmas>
    )
endif()

###############################################################################
# Catch2
###############################################################################

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdParty/catch2 ${CMAKE_BINARY_DIR}/catch2)

################################################################################
# MPI
################################################################################

# MPI is provided by pmacc but to execute the binaries via root additional flags must be given to the execution command
option(USE_MPI_AS_ROOT_USER "add --allow-run-as-root mpiexec used by ctest" OFF)

if(USE_MPI_AS_ROOT_USER)
    set(MPI_RUNTIME_FLAGS "--allow-run-as-root")
endif()

###############################################################################
# Targets
###############################################################################

# CTest
enable_testing()

# Add subdirectories for tests
add_subdirectory(dimensional_tests)
add_subdirectory(non_dimensional_tests)
