# - Find the IntelSYCL::SYCL_CXX compiler interface target
#
# Use this module by invoking find_package with the form:
#   find_package(IntelSYCL [REQUIRED] [QUIET])
#
# The official oneAPI installation ships IntelSYCLConfig.cmake. Standalone
# open-source DPC++ builds (and clang based SYCL compilers) do not, but alpaka's
# oneAPI backend requires the target IntelSYCL::SYCL_CXX. This module first
# searches for the official config package and otherwise provides an INTERFACE
# target that adds the -fsycl compile and link flags required to drive the
# compiler in SYCL mode.

include(FindPackageHandleStandardArgs)

if(NOT TARGET IntelSYCL::SYCL_CXX)
    find_package(IntelSYCL CONFIG QUIET)
endif()

if(NOT TARGET IntelSYCL::SYCL_CXX)
    include(CheckCXXCompilerFlag)
    # Only compilation is probed: linking -fsycl needs the SYCL runtime in the
    # linker search path, which must not decide whether the compiler can emit
    # SYCL code.
    set(_intel_sycl_saved_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    check_cxx_compiler_flag("-fsycl" PMACC_INTELSYCL_HAS_FSYCL)
    set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_intel_sycl_saved_try_compile_target_type}")

    if(PMACC_INTELSYCL_HAS_FSYCL)
        add_library(IntelSYCL::SYCL_CXX INTERFACE IMPORTED)
        set_target_properties(
            IntelSYCL::SYCL_CXX
            PROPERTIES
                INTERFACE_COMPILE_OPTIONS "-fsycl"
                INTERFACE_LINK_OPTIONS "-fsycl"
        )
    endif()
endif()

set(IntelSYCL_FOUND FALSE)
if(TARGET IntelSYCL::SYCL_CXX)
    set(IntelSYCL_FOUND TRUE)
endif()

find_package_handle_standard_args(IntelSYCL REQUIRED_VARS IntelSYCL_FOUND)
