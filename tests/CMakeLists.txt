include(Catch)

add_executable(waferedge-tests test_machine.cpp)
target_link_libraries(waferedge-tests PRIVATE waferedge waferedge_options Catch2::Catch2WithMain)
catch_discover_tests(waferedge-tests)

# GPU tests run locally only (CI has no GPU); `ctest -L gpu` selects them.
if(WAFEREDGE_CUDA)
  add_executable(waferedge-gpu-tests test_gpu.cpp gpu/iota_kernel.cu)
  target_link_libraries(waferedge-gpu-tests
    PRIVATE waferedge waferedge_options CUDA::cudart Catch2::Catch2WithMain)
  catch_discover_tests(waferedge-gpu-tests PROPERTIES LABELS gpu)
endif()
