# Host-only tests for the neural bunch initializer.
add_executable(neuralBunchReferenceTest referenceKernelTest.cpp)
target_compile_features(neuralBunchReferenceTest PRIVATE cxx_std_20)
target_include_directories(neuralBunchReferenceTest PRIVATE "${PIConGPUapp_SOURCE_DIR}/..")
target_compile_definitions(neuralBunchReferenceTest PRIVATE NB_TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata")
target_link_libraries(neuralBunchReferenceTest PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME neuralBunchReferenceTest COMMAND neuralBunchReferenceTest)

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
