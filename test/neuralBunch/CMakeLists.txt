# Host-only tests for the neural bunch initializer.
add_executable(neuralBunchReferenceTest referenceKernelTest.cpp)
target_compile_features(neuralBunchReferenceTest PRIVATE cxx_std_20)
target_include_directories(neuralBunchReferenceTest PRIVATE "${PIConGPUapp_SOURCE_DIR}/..")
target_compile_definitions(neuralBunchReferenceTest PRIVATE NB_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata")
target_link_libraries(neuralBunchReferenceTest PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME neuralBunchReferenceTest COMMAND neuralBunchReferenceTest)

add_executable(neuralBunchBandedPolicyTest bandedPolicyTest.cpp)
target_compile_features(neuralBunchBandedPolicyTest PRIVATE cxx_std_20)
target_include_directories(neuralBunchBandedPolicyTest PRIVATE "${PIConGPUapp_SOURCE_DIR}/..")
target_link_libraries(
    neuralBunchBandedPolicyTest
    PRIVATE alpaka::nn alpakaVendor::alpakaVendor nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX
)
alpaka_finalize(neuralBunchBandedPolicyTest)
target_compile_definitions(
    neuralBunchBandedPolicyTest
    PRIVATE
        ALPAKA_CMAKE_TARGET_HEADERS_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_CUDA_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_HOST_FINALIZE_CALLED
)
add_test(NAME neuralBunchBandedPolicyTest COMMAND neuralBunchBandedPolicyTest)

add_executable(neuralBunchTileGeometryTest tileGeometryTest.cpp)
target_compile_features(neuralBunchTileGeometryTest PRIVATE cxx_std_20)
target_include_directories(neuralBunchTileGeometryTest PRIVATE "${PIConGPUapp_SOURCE_DIR}/..")
add_test(NAME neuralBunchTileGeometryTest COMMAND neuralBunchTileGeometryTest)

add_executable(neuralBunchProbeFeatureTest probeFeatureTest.cpp)
target_compile_features(neuralBunchProbeFeatureTest PRIVATE cxx_std_20)
target_include_directories(neuralBunchProbeFeatureTest PRIVATE "${PIConGPUapp_SOURCE_DIR}/..")
target_link_libraries(
    neuralBunchProbeFeatureTest
    PRIVATE alpaka::nn alpakaVendor::alpakaVendor nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX
)
alpaka_finalize(neuralBunchProbeFeatureTest)
# alpaka_finalize only records the alpaka::alpaka sentinel for interface-only
# targets; the headers/cuda/host sentinels are needed by alpaka's config check.
target_compile_definitions(
    neuralBunchProbeFeatureTest
    PRIVATE
        ALPAKA_CMAKE_TARGET_HEADERS_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_CUDA_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_HOST_FINALIZE_CALLED
)
add_test(NAME neuralBunchProbeFeatureTest COMMAND neuralBunchProbeFeatureTest)
# The benchmark binaries are CUDA-only; a CPU-only configuration must still configure and test the host tests.
if(alpaka_DEP_CUDA)
    enable_language(CUDA)
    add_executable(neuralBunchBandedBenchmark bandedBenchmark.cu)
    set_target_properties(neuralBunchBandedBenchmark PROPERTIES CUDA_ARCHITECTURES "70;75")
    target_compile_features(neuralBunchBandedBenchmark PRIVATE cxx_std_20)
    target_include_directories(neuralBunchBandedBenchmark PRIVATE "${PIConGPUapp_SOURCE_DIR}/..")
    target_link_libraries(
        neuralBunchBandedBenchmark
        PRIVATE alpaka::nn alpakaVendor::alpakaVendor nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX
    )
    alpaka_finalize(neuralBunchBandedBenchmark)
    target_compile_definitions(
        neuralBunchBandedBenchmark
        PRIVATE
            ALPAKA_CMAKE_TARGET_HEADERS_FINALIZE_CALLED
            ALPAKA_CMAKE_TARGET_CUDA_FINALIZE_CALLED
            ALPAKA_CMAKE_TARGET_HOST_FINALIZE_CALLED
    )
    add_executable(
        neuralBunchExactBenchmark
        exactBenchmark.cu
        "${PIConGPUapp_SOURCE_DIR}/particles/neuralBunch/ReferenceTile.cpp"
    )
    set_target_properties(neuralBunchExactBenchmark PROPERTIES CUDA_ARCHITECTURES "70;75")
    target_compile_features(neuralBunchExactBenchmark PRIVATE cxx_std_20)
    target_include_directories(neuralBunchExactBenchmark PRIVATE "${PIConGPUapp_SOURCE_DIR}/..")
    target_link_libraries(
        neuralBunchExactBenchmark
        PRIVATE alpaka::nn alpakaVendor::alpakaVendor nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX
    )
    alpaka_finalize(neuralBunchExactBenchmark)
    target_compile_definitions(
        neuralBunchExactBenchmark
        PRIVATE
            ALPAKA_CMAKE_TARGET_HEADERS_FINALIZE_CALLED
            ALPAKA_CMAKE_TARGET_CUDA_FINALIZE_CALLED
            ALPAKA_CMAKE_TARGET_HOST_FINALIZE_CALLED
    )
    # Device parity of the batched exact quadrature against the host FP64 reference plus the distance policy.
    add_test(NAME neuralBunchExactDeviceTest COMMAND neuralBunchExactBenchmark check cuda)
endif(alpaka_DEP_CUDA)

add_executable(neuralBunchDevicePipelineTest devicePipelineTest.cu)
set_target_properties(neuralBunchDevicePipelineTest PROPERTIES CUDA_ARCHITECTURES "70;75")
target_compile_features(neuralBunchDevicePipelineTest PRIVATE cxx_std_20)
target_include_directories(neuralBunchDevicePipelineTest PRIVATE "${PIConGPUapp_SOURCE_DIR}/..")
target_link_libraries(
    neuralBunchDevicePipelineTest
    PRIVATE alpaka::nn alpakaVendor::alpakaVendor nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX
)
alpaka_finalize(neuralBunchDevicePipelineTest)
target_compile_definitions(
    neuralBunchDevicePipelineTest
    PRIVATE
        ALPAKA_CMAKE_TARGET_HEADERS_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_CUDA_FINALIZE_CALLED
        ALPAKA_CMAKE_TARGET_HOST_FINALIZE_CALLED
)
add_test(NAME neuralBunchDevicePipelineTest COMMAND neuralBunchDevicePipelineTest cuda)
