# Copyright 2015-2024 Erik Zenker, Rene Widera, Axel Huebl, Jan Stephan
#
# This file is part of PMacc.
#
# PMacc is free software: you can redistribute it and/or modify
# it under the terms of either the GNU General Public License or
# the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PMacc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License and the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# and the GNU Lesser General Public License along with PMacc.
# If not, see <http://www.gnu.org/licenses/>.
#

# - Config file for the pmacc package and provide the target pmacc::pmacc

###############################################################################
# PMacc
###############################################################################
cmake_minimum_required(VERSION 3.28.0)

# set helper pathes to find libraries and packages
# Add specific hints
list(APPEND CMAKE_PREFIX_PATH "$ENV{MPI_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{BOOST_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{VT_ROOT}")
# Add from environment after specific env vars
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

# own modules for find_packages e.g. FindmallocMC
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PMacc_DIR}/../../thirdParty/cmake-modules/)
# own find module for the SYCL compiler target required by alpaka's oneAPI
# backend (IntelSYCL::SYCL_CXX)
list(APPEND CMAKE_MODULE_PATH "${PMacc_DIR}/../../cmake")

################################################################################
# Intel oneAPI/SYCL compiler setup
################################################################################

if(alpaka_DEP_ONEAPI)
    include(CheckCXXCompilerFlag)
    include(CheckCXXSourceCompiles)

    # Standalone DPC++ distributions may not ship clang-scan-deps. CMake then
    # adds a dependency scanner command to every compile once '-fsycl' is in
    # the flags, so even the following compiler probes fail with
    # "CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS-NOTFOUND: not found". C++20 modules
    # are not used by PMacc, therefore disable scanning for such toolchains.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
        set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
    endif()

    # Probe only compilation. Linking -fsycl requires the SYCL runtime in the
    # linker search path, which must not decide compiler support.
    set(_pmacc_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    check_cxx_compiler_flag("-fsycl" PMACC_CXX_HAS_FSYCL)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_pmacc_saved_try_compile_target_type}")

    if(NOT PMACC_CXX_HAS_FSYCL)
        message(
            FATAL_ERROR
            "alpaka_DEP_ONEAPI requires a SYCL capable compiler supporting '-fsycl' "
            "(CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER})."
        )
    endif()

    # alpaka derives ALPAKA_COMP_ICPX and ALPAKA_LANG_ONEAPI from
    # __INTEL_LLVM_COMPILER. icpx predefines it, clang based DPC++ builds do
    # not, although they are full SYCL compilers. Provide a non-zero version
    # only for such compilers; any non-zero value enables ALPAKA_LANG_ONEAPI.
    check_cxx_source_compiles(
        "#ifndef __INTEL_LLVM_COMPILER
#    error \"__INTEL_LLVM_COMPILER is not predefined\"
#endif
int main() { return 0; }"
        PMACC_CXX_DEFINES_INTEL_LLVM_COMPILER
    )
    if(NOT PMACC_CXX_DEFINES_INTEL_LLVM_COMPILER)
        # The value is only interpreted as a compiler version by alpaka. The
        # SYCL standard version is taken from SYCL_LANGUAGE_VERSION.
        add_compile_definitions(__INTEL_LLVM_COMPILER=20261014)
    endif()

    # clang 23's cpuid.h declares __cpuidex although '-fsycl' makes it a
    # builtin, so including <cpuid.h> (used by alpaka's host sysInfo) fails
    # with "static declaration follows non-static declaration". Renaming the
    # function via the preprocessor keeps both the header's guard and alpaka's
    # call consistent. Apply the workaround only for affected toolchains.
    set(_pmacc_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
    set(CMAKE_REQUIRED_FLAGS "-fsycl")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    check_cxx_source_compiles("#include <cpuid.h>\nint main() { return 0; }" PMACC_SYCL_CPUID_HEADER_COMPILES)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_pmacc_saved_try_compile_target_type}")
    set(CMAKE_REQUIRED_FLAGS "${_pmacc_saved_required_flags}")
    if(NOT PMACC_SYCL_CPUID_HEADER_COMPILES)
        add_compile_definitions(__cpuidex=w1_cpuidex)
    endif()

    # Select the oneAPI device kind at compile time. The default targets an
    # Intel GPU; Cpu runs the same SYCL executor on the oneAPI CPU device, which
    # is useful to validate the backend without accelerator hardware.
    set(
        PIC_ONEAPI_DEVICE_KIND
        "IntelGpu"
        CACHE STRING "oneAPI device kind used by PMacc and PIConGPU (IntelGpu or Cpu)"
    )
    set_property(CACHE PIC_ONEAPI_DEVICE_KIND PROPERTY STRINGS "IntelGpu" "Cpu")
    if(PIC_ONEAPI_DEVICE_KIND STREQUAL "Cpu")
        add_compile_definitions(PIC_ONEAPI_CPU_DEVICE=1)
    elseif(NOT PIC_ONEAPI_DEVICE_KIND STREQUAL "IntelGpu")
        message(FATAL_ERROR "PIC_ONEAPI_DEVICE_KIND must be 'IntelGpu' or 'Cpu'.")
    endif()
endif()

################################################################################
# alpaka path
################################################################################

# workaround for native CMake CUDA
# CMake is not forwarding CMAKE_CUDA_ARCHITECTURES to the CMake CUDA compiler check
# error: clang: error: cannot find libdevice for sm_20. Provide path to different CUDA installation via --cuda-path, or pass -nocudalib to build without linking with libdevice.
# The workaround is parsing CMAKE_CUDA_ARCHITECTURES and forward command line parameter directly to clang++.
if(alpaka_DEP_CUDA AND CMAKE_CUDA_COMPILER)
    string(REGEX MATCH "(.*clang.*)" IS_CLANGCUDA_COMPILER ${CMAKE_CUDA_COMPILER})
    if(IS_CLANGCUDA_COMPILER)
        foreach(_CUDA_ARCH_ELEM ${CMAKE_CUDA_ARCHITECTURES})
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cuda-gpu-arch=sm_${_CUDA_ARCH_ELEM}")
        endforeach()
    endif()
endif()

# workaround for a CMake bug which is not handled in alpaka 0.7.0
# https://github.com/alpaka-group/alpaka/pull/1423
if(alpaka_DEP_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    # Use user selected CMake CXX compiler as cuda host compiler to avoid fallback to the default system CXX host compiler.
    # CMAKE_CUDA_HOST_COMPILER is reset by check_language(CUDA) therefore definition passed by the user via -DCMAKE_CUDA_HOST_COMPILER are
    # ignored by CMake (looks like a CMake bug).
    # The if condition used here should work correct after the CMake bug is fixed, too.
    # Check the environment variable CUDAHOSTCXX to prefer the CUDA host compiler set by the user.
    if("$ENV{CUDAHOSTCXX}" STREQUAL "" AND NOT CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()
    enable_language(CUDA)
endif()

# set path to internal
set(PMACC_alpaka_PROVIDER "intern" CACHE STRING "Select which alpaka is used")
set_property(CACHE PMACC_alpaka_PROVIDER PROPERTY STRINGS "intern;extern")
mark_as_advanced(PMACC_alpaka_PROVIDER)

if(${PMACC_alpaka_PROVIDER} STREQUAL "intern")
    list(INSERT CMAKE_MODULE_PATH 0 "${PMacc_DIR}/../../thirdParty/alpaka")
endif()

# Set alpaka CXX standard because the default is currently C++14.
if(NOT DEFINED alpaka_CXX_STANDARD)
    set(alpaka_CXX_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "C++ standard version")
endif()

################################################################################
# setup alpaka
################################################################################

# the min and max. supported alpaka version
set(_PMACC_MIN_ALPAKA_VERSION 3.0.0)
set(_PMACC_MAX_ALPAKA_VERSION 3.0.0)

# do not search for alpaka if it already exists
# for example, a project that includes alpaka via add_subdirectory before including pmacc via add_subdirectory
if(NOT TARGET alpaka::alpaka)
    # the alpaka provider for the internal alpaka is only available,
    # if pmacc is used via add_subdirectory in another project
    # or examples are build

    set(PMACC_ALPAKA_PROVIDER "internal" CACHE STRING "Select which alpaka is used")
    set_property(CACHE PMACC_ALPAKA_PROVIDER PROPERTY STRINGS "internal;external")
    mark_as_advanced(PMACC_ALPAKA_PROVIDER)

    if(${PMACC_ALPAKA_PROVIDER} STREQUAL "internal")
        set(alpaka_EXAMPLES OFF)
        set(alpaka_TESTS OFF)
        set(BUILD_TESTING OFF)
        add_subdirectory(${PMacc_DIR}/../../thirdParty/alpaka ${CMAKE_BINARY_DIR}/alpaka EXCLUDE_FROM_ALL)
    else()
        find_package(alpaka ${_PMACC_MAX_ALPAKA_VERSION} HINTS $ENV{ALPAKA_ROOT})
        if(NOT TARGET alpaka::alpaka)
            message(
                STATUS
                "Could not find alpaka ${_PMACC_MAX_ALPAKA_VERSION}. Now searching for alpaka ${_PMACC_MIN_ALPAKA_VERSION}"
            )
            find_package(alpaka ${_PMACC_MIN_ALPAKA_VERSION} REQUIRED HINTS $ENV{ALPAKA_ROOT})
        endif()
        if(alpaka_VERSION VERSION_GREATER _PMACC_MAX_ALPAKA_VERSION)
            message(
                WARNING
                "Unsupported alpaka version ${alpaka_VERSION}. "
                "Supported versions [${_PMACC_MIN_ALPAKA_VERSION},${_PMACC_MAX_ALPAKA_VERSION}]."
            )
        endif()
    endif()

    if(NOT TARGET alpaka::alpaka)
        message(FATAL_ERROR "Required PMacc dependency alpaka could not be found!")
    endif()
endif()

################################################################################
# Host executor selection
################################################################################

# alpaka enables every host executor whose dependency is available and only
# disables one when the matching alpaka_EXEC_* build option is OFF. Its headers
# do not derive ALPAKA_DISABLE_EXEC_* from the available dependencies alone
# (in particular, disabling the OpenMP dependency with alpaka_DEP_OMP=OFF does
# not disable the CpuOmpBlocks executor), therefore a serial or TBB build would
# silently keep using the OpenMP blocks executor. Translate the requested host
# backend into explicit ALPAKA_DISABLE_EXEC_* definitions. This mapping is
# PMacc's own contract and is documented here rather than relying on alpaka's
# internal option handling.
#
# The definitions are added to the directory scope before the pmacc target is
# created: CMake snapshots the directory COMPILE_DEFINITIONS when a target is
# created, so adding them later would not reach the pmacc library sources. Not
# every target which compiles PMacc headers links pmacc or alpaka (for example
# PIConGPU's host-only library), so all translation units must see the same
# executor type from this directory scope.
if(alpaka_DEP_OMP)
    # OpenMP requested: keep CpuOmpBlocks as the preferred host executor.
elseif(alpaka_DEP_TBB)
    # TBB requested: the OpenMP blocks executor takes priority over TBB, so it
    # must be disabled explicitly for CpuTbbBlocks to be selected.
    add_compile_definitions(ALPAKA_DISABLE_EXEC_CpuOmpBlocks)
else()
    # Serial host backend: no block-parallel host executor is available.
    add_compile_definitions(ALPAKA_DISABLE_EXEC_CpuOmpBlocks ALPAKA_DISABLE_EXEC_CpuTbbBlocks)
endif()

################################################################################
# PMacc target
################################################################################

file(GLOB_RECURSE PMACC_SRC_FILES "${PMacc_DIR}/*.cpp")
# remove files located in the directory 'test'
string(REGEX REPLACE "${PMacc_DIR}/test/.*" "" PMACC_SRC_FILES "${PMACC_SRC_FILES}")

add_library(
        pmacc
        STATIC
        ${PMACC_SRC_FILES}
)

target_include_directories(pmacc PUBLIC $<BUILD_INTERFACE:${PMacc_DIR}/..> $<INSTALL_INTERFACE:${PMacc_DIR}/..>)

# Even if there are no sources CMAKE has to know the language.
set_target_properties(pmacc PROPERTIES LINKER_LANGUAGE CXX)

add_library(pmacc::pmacc ALIAS pmacc)
target_link_libraries(pmacc PUBLIC alpaka::alpaka)

if(alpaka_DEP_ONEAPI)
    # The SPIR64 device targets have no 128-bit long double, but PIConGPU's _X
    # and _COLL literal operators must take long double by language rule.
    # Compile the device pass with a 64-bit long double, the host ABI stays
    # unchanged.
    if(alpaka_ONEAPI_IntelGpu)
        target_compile_options(pmacc PUBLIC -Xsycl-target-frontend=spir64 -mlong-double-64)
    endif()
    if(alpaka_ONEAPI_Cpu)
        target_compile_options(pmacc PUBLIC -Xsycl-target-frontend=spir64_x86_64 -mlong-double-64)
    endif()
endif()

alpaka_finalize(pmacc)

###############################################################################
# Build Flags
###############################################################################

set(PMACC_BUILD_TYPE "Release;Debug")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type for the project" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${PMACC_BUILD_TYPE}")
unset(PMACC_BUILD_TYPE)

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

###############################################################################
# Language Flags
###############################################################################

# enforce C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

###############################################################################
# Definitely Unsupported Compilers
###############################################################################
# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
        message(FATAL_ERROR "GCC too old! Use GCC 7.0 or newer")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        message(WARNING "GCC < 9.0 is not a official supported, please try a newer!")
    endif()
    # Clang
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
        message(FATAL_ERROR "Apple Clang (Xcode) too old! Use Xcode 8.0 or newer")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        message(FATAL_ERROR "Clang too old! Use Clang 6.0 or newer")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
        message(WARNING "Clang < 10.0 is not a official supported, please try a newer!")
    endif()
endif()

###############################################################################
# CPU Architecture: available instruction sets for e.g. SIMD extensions
#
# Conveniently set the architecture for the CPU compiler via this option.
# For unsupported compilers, ignore this option and set CXXFLAGS.
###############################################################################

set(PMACC_CPU_ARCH $ENV{PMACC_CPU_ARCH} CACHE STRING "compiler dependent CPU architecture string")

# list of known compiler flags to set the CPU architecture
# GNU
if(CMAKE_COMPILER_IS_GNUCXX)
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "ppc64le")
        set(PMACC_CPU_ARCH_TEMPLATE "-mcpu={} -mtune={}")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
        set(PMACC_CPU_ARCH_TEMPLATE "-mcpu={}")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "riscv64")
        set(PMACC_CPU_ARCH_TEMPLATE "-march={}")
    else()
        set(PMACC_CPU_ARCH_TEMPLATE "-march={} -mtune={}")
    endif()
    # ICC
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    if("${PMACC_CPU_ARCH}" STREQUAL "native")
        set(PMACC_CPU_ARCH_TEMPLATE "-march={} -mtune={}")
    else()
        set(PMACC_CPU_ARCH_TEMPLATE "-x{}")
    endif()
    # Clang
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
        set(PMACC_CPU_ARCH_TEMPLATE "-mcpu={}")
    elseif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "riscv64")
        set(PMACC_CPU_ARCH_TEMPLATE "-march={}")
    else()
        set(PMACC_CPU_ARCH_TEMPLATE "-march={} -mtune={}")
    endif()
    # XL
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")
    set(PMACC_CPU_ARCH_TEMPLATE "-qarch={}")
    # PGI
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
    if(NOT "${PMACC_CPU_ARCH}" STREQUAL "native")
        set(PMACC_CPU_ARCH_TEMPLATE "-tp={}")
    endif()
endif()

# architecture is set and compiler is known
if(PMACC_CPU_ARCH AND PMACC_CPU_ARCH_TEMPLATE)
    string(REPLACE "{}" "${PMACC_CPU_ARCH}" PMACC_CPU_ARCH_STRING "${PMACC_CPU_ARCH_TEMPLATE}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PMACC_CPU_ARCH_STRING}")
endif()

################################################################################
# Find MPI
################################################################################

find_package(MPI REQUIRED)
target_link_libraries(pmacc PUBLIC MPI::MPI_CXX)

if(CMAKE_TRY_COMPILE_TARGET_TYPE STREQUAL "STATIC_LIBRARY" AND CMAKE_EXE_LINKER_FLAGS)
    # Workaround for linker issues when linking static MPI libraries.
    # Because of CMAKE_TRY_COMPILE_TARGET_TYPE CMake is providing the statics libraries before the object file from
    # `add_executable` therefore MPI symbols can not be resolved. Linking the linker flaks to the target again will
    # workaround the issue.
    target_link_libraries(pmacc PUBLIC ${CMAKE_EXE_LINKER_FLAGS})
endif()

################################################################################
# Find Threads
################################################################################

if(NOT THREADS_PREFER_PTHREAD_FLAG)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
endif()
find_package(Threads REQUIRED)
if(NOT APPLE)
    # librt: undefined reference to `clock_gettime'
    find_library(RT_LIBRARY rt)
    if(RT_LIBRARY)
        target_link_libraries(pmacc PUBLIC ${RT_LIBRARY})
    endif()
endif()

################################################################################
# Find Boost
################################################################################

find_package(Boost 1.74 REQUIRED CONFIG COMPONENTS program_options)
if(TARGET Boost::boost)
    target_link_libraries(pmacc PUBLIC Boost::boost)
    target_link_libraries(pmacc PUBLIC Boost::program_options)
else()
    target_include_directories(pmacc PUBLIC ${Boost_INCLUDE_DIRS})
    target_link_libraries(pmacc PUBLIC ${Boost_LIBRARIES})
endif()

# We do not use std::auto_ptr and keeping this enabled in Boost causes a
# warning with NVCC+GCC and is unnecessary time spend in compile time
# (note that std::auto_ptr is deprecated in C++11 and removed in C++17)
message(STATUS "Boost: deactivate std::auto_ptr")
target_compile_definitions(pmacc PUBLIC "-DBOOST_NO_AUTO_PTR")

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(STATUS "Boost: Disable variadic templates")
    message(STATUS "Boost: Do not use fenv.h from standard library")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_VARIADIC_TEMPLATES")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_FENV_H")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # suppress boost error
    # 'no member named "impl" in "boost::detail::thread_move_t<boost::detail::nullary_function<void ()> >"'
    # in 'boost/thread/detail/nullary_function.hpp'
    message(STATUS "Boost: Do not use C++11 smart pointers from standard library")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_SMART_PTR")
endif()

# Newer Boost releases: probably troublesome, warn at least
if(Boost_VERSION GREATER 107000)
    message(WARNING "Untested Boost release > 1.70.0 (Found ${Boost_VERSION})! " "Maybe use a newer PIConGPU?")
endif()

################################################################################
# Find OpenMP
################################################################################

if(
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
    AND (alpaka_DEP_HIP OR (alpaka_DEP_CUDA AND CMAKE_CUDA_COMPILER_ID MATCHES "Clang"))
)
    # For HIP the problem is that in alpaka '::isnan(), ::sinh(), ::isfinite(), ::isinf()' is not found.
    # The reason could be that if OpenMP is activated clang is using math C headers where all of these functions are macros.
    message(
        WARNING
        "OpenMP host side acceleration is disabled: CUDA/HIP compilation with clang is not supporting OpenMP."
    )
else()
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

################################################################################
# Find mallocMC
################################################################################

if(alpaka_DEP_CUDA OR alpaka_DEP_HIP OR alpaka_DEP_ONEAPI)
    include(FetchContent)

    FetchContent_Declare(
        mallocMC
        SOURCE_DIR
        "${PMacc_DIR}/../../thirdParty/mallocMC"
        EXCLUDE_FROM_ALL
        FIND_PACKAGE_ARGS 3.0.0 QUIET
    )
    FetchContent_MakeAvailable(mallocMC)

    target_link_libraries(pmacc PUBLIC mallocMC::mallocMC)
endif()

################################################################################
# Find hiprand
################################################################################

# ROCm ships the hiprand device headers inside an extra ``hiprand`` folder
# (``<rocm>/include/hiprand/hiprand_kernel.h``) while the source uses the
# unqualified ``<hiprand_kernel.h>`` include. Derive the ROCm include root from
# the HIP toolchain and expose both the root and its ``hiprand`` subfolder.
if(alpaka_DEP_HIP)
    set(_pmacc_hip_include_candidates "")
    if(TARGET hip::host)
        get_target_property(_pmacc_hip_imported_dirs hip::host INTERFACE_INCLUDE_DIRECTORIES)
        if(_pmacc_hip_imported_dirs)
            list(APPEND _pmacc_hip_include_candidates ${_pmacc_hip_imported_dirs})
        endif()
    endif()
    if(DEFINED ENV{ROCM_PATH})
        list(APPEND _pmacc_hip_include_candidates "$ENV{ROCM_PATH}/include")
    elseif(DEFINED ENV{HIP_PATH})
        list(APPEND _pmacc_hip_include_candidates "$ENV{HIP_PATH}/include")
    endif()
    if(CMAKE_HIP_COMPILER)
        # <rocm>/llvm/bin/clang++ -> <rocm>/include
        get_filename_component(_pmacc_hip_compiler_dir "${CMAKE_HIP_COMPILER}" DIRECTORY)
        get_filename_component(_pmacc_hip_compiler_dir "${_pmacc_hip_compiler_dir}" DIRECTORY)
        get_filename_component(_pmacc_hip_compiler_dir "${_pmacc_hip_compiler_dir}" DIRECTORY)
        list(APPEND _pmacc_hip_include_candidates "${_pmacc_hip_compiler_dir}/include")
    endif()
    find_path(
        PMACC_HIPRAND_INCLUDE_DIR
        NAMES hiprand/hiprand_kernel.h
        HINTS ${_pmacc_hip_include_candidates}
    )
    if(PMACC_HIPRAND_INCLUDE_DIR)
        target_include_directories(
            pmacc
            PUBLIC "${PMACC_HIPRAND_INCLUDE_DIR}" "${PMACC_HIPRAND_INCLUDE_DIR}/hiprand"
        )
    else()
        message(WARNING "hiprand device headers (hiprand/hiprand_kernel.h) were not found")
    endif()
endif()

################################################################################
# PMacc options
################################################################################

option(PMACC_ASYNC_QUEUES "Enable asynchronous alpaka queues" ON)
if(PMACC_ASYNC_QUEUES)
    target_compile_definitions(pmacc PUBLIC "PMACC_USE_ASYNC_QUEUES=1")
endif()
option(PMACC_BLOCKING_KERNEL "activate checks for every kernel call and synch after every kernel call" OFF)
if(PMACC_BLOCKING_KERNEL)
    target_compile_definitions(pmacc PUBLIC "-DPMACC_SYNC_KERNEL=1")
endif(PMACC_BLOCKING_KERNEL)

set(PMACC_VERBOSE "0" CACHE STRING "set verbose level for PMacc")
target_compile_definitions(pmacc PUBLIC "-DPMACC_VERBOSE_LVL=${PMACC_VERBOSE}")
