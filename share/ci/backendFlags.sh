#!/bin/bash

###################################################
# translate PIConGPU backend names into CMake Flags
###################################################

get_backend_flags()
{
    backend_cfg=(${1//:/ })
    num_options="${#backend_cfg[@]}"
    if [ $num_options -gt 2 ] ; then
        echo "-b|--backend must be contain 'backend:arch' or 'backend'" >&2
        exit 1
    fi
    if [ "${backend_cfg[0]}" == "cuda" ] ; then
        result+=" -Dalpaka_DEP_CUDA=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DCMAKE_CUDA_ARCHITECTURES=${backend_cfg[1]}"
        else
            result+=" -DCMAKE_CUDA_ARCHITECTURES=60"
        fi
    elif [ "${backend_cfg[0]}" == "omp2b" ] ; then
        result+=" -Dalpaka_DEP_OMP=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=${backend_cfg[1]}"
        fi
    elif [ "${backend_cfg[0]}" == "serial" ] ; then
        result+=" -Dalpaka_DEP_OMP=OFF"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=${backend_cfg[1]}"
        fi
    elif [ "${backend_cfg[0]}" == "tbb" ] ; then
        result+=" -Dalpaka_DEP_TBB=ON -Dalpaka_DEP_OMP=OFF"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=${backend_cfg[1]}"
        fi
    elif [ "${backend_cfg[0]}" == "threads" ] ; then
        result+=" -Dalpaka_DEP_OMP=OFF"
        if [ $num_options -eq 2 ] ; then
            result+=" -DPMACC_CPU_ARCH=${backend_cfg[1]}"
        fi
    elif [ "${backend_cfg[0]}" == "hip" ] ; then
        result+=" -Dalpaka_DEP_HIP=ON"
        if [ $num_options -eq 2 ] ; then
            result+=" -DCMAKE_HIP_ARCHITECTURES=${backend_cfg[1]}"
        else
            # If no architecture is given build for Radeon VII or MI50/60.
            result+=" -DCMAKE_HIP_ARCHITECTURES=gfx906"
        fi
    else
        echo "unsupported backend given '$1'" >&2
        exit 1
    fi

    echo "$result"
    exit 0
}
