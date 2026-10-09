if(WAFEREDGE_CUDA)
  add_executable(bench-transfer bench_transfer.cpp)
  target_link_libraries(bench-transfer PRIVATE waferedge waferedge_options CUDA::cudart
                        benchmark::benchmark)
endif()
