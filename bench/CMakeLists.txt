# Synthetic maps come from the test support library.
if(WAFEREDGE_TESTS)
  add_executable(bench-features bench_features.cpp)
  target_link_libraries(bench-features PRIVATE waferedge waferedge_synth waferedge_options
                        benchmark::benchmark)
endif()

if(WAFEREDGE_CUDA)
  add_executable(bench-transfer bench_transfer.cpp)
  target_link_libraries(bench-transfer PRIVATE waferedge waferedge_options CUDA::cudart
                        benchmark::benchmark)
endif()
