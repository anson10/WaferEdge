include(Catch)

# Synthetic wafer maps, shared by the tests and the benchmarks.
add_library(waferedge_synth STATIC support/synth.cpp)
target_include_directories(waferedge_synth PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(waferedge_synth PUBLIC waferedge PRIVATE waferedge_options)

add_executable(waferedge-tests
  test_backends.cpp
  test_classifier.cpp
  test_clusters.cpp
  test_cnn.cpp
  test_features.cpp
  test_geometry.cpp
  test_hough.cpp
  test_machine.cpp
  test_map_file.cpp
  test_randomness.cpp
  test_thread_pool.cpp)
target_link_libraries(waferedge-tests
  PRIVATE waferedge waferedge_synth waferedge_options Catch2::Catch2WithMain)
target_compile_definitions(waferedge-tests PRIVATE
  WAFEREDGE_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data"
  WAFEREDGE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
catch_discover_tests(waferedge-tests)

# GPU tests run locally only (CI has no GPU); `ctest -L gpu` selects them.
if(WAFEREDGE_CUDA)
  add_executable(waferedge-gpu-tests test_gpu.cpp test_gpu_gemm.cpp test_gpu_signatures.cpp
                                    gpu/iota_kernel.cu)
  target_link_libraries(waferedge-gpu-tests
    PRIVATE waferedge waferedge_synth waferedge_options CUDA::cudart Catch2::Catch2WithMain)
  target_compile_definitions(waferedge-gpu-tests PRIVATE
    WAFEREDGE_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/data")
  catch_discover_tests(waferedge-gpu-tests PROPERTIES LABELS gpu)
endif()
