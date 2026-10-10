# Synthetic maps come from the test support library.
if(WAFEREDGE_TESTS)
  add_executable(bench-features bench_features.cpp)
  target_link_libraries(bench-features PRIVATE waferedge waferedge_synth waferedge_options
                        benchmark::benchmark)

  # Counts allocations per message: links the operator new replacement from tests/support.
  add_executable(bench-secs bench_secs.cpp)
  target_link_libraries(bench-secs PRIVATE waferedge waferedge_synth waferedge_alloc_counter
                        waferedge_options benchmark::benchmark)

  add_executable(bench-emulator bench_emulator.cpp)
  target_link_libraries(bench-emulator PRIVATE waferedge waferedge_synth waferedge_options)

  add_executable(bench-hsms bench_hsms.cpp)
  target_link_libraries(bench-hsms PRIVATE waferedge waferedge_synth waferedge_alloc_counter
                        waferedge_options benchmark::benchmark)
endif()

if(WAFEREDGE_CUDA)
  add_executable(bench-gpu bench_gpu.cpp)
  target_link_libraries(bench-gpu PRIVATE waferedge waferedge_synth waferedge_options
                        benchmark::benchmark)

  # cuBLAS is the GEMM ladder's yardstick: linked here only, never into the library.
  add_executable(bench-gemm bench_gemm.cpp)
  target_link_libraries(bench-gemm PRIVATE waferedge waferedge_options CUDA::cublas CUDA::cublasLt
                        benchmark::benchmark)

  add_executable(bench-cnn bench_cnn.cpp)
  target_link_libraries(bench-cnn PRIVATE waferedge waferedge_synth waferedge_options
                        benchmark::benchmark)
  target_compile_definitions(bench-cnn PRIVATE WAFEREDGE_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

  add_executable(bench-transfer bench_transfer.cpp)
  target_link_libraries(bench-transfer PRIVATE waferedge waferedge_options CUDA::cudart
                        benchmark::benchmark)
endif()
