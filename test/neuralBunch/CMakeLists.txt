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
