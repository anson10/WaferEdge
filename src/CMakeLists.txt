add_library(waferedge
  backend.cpp
  classifier.cpp
  clusters.cpp
  cnn.cpp
  cnn_conformal.cpp
  cnn_reference.cpp
  features.cpp
  features_avx2.cpp
  gem/endpoint.cpp
  gem/equipment.cpp
  gem/host.cpp
  gem/messages.cpp
  geometry.cpp
  hough.cpp
  machine.cpp
  map_file.cpp
  randomness.cpp
  hsms/protocol.cpp
  hsms/session.cpp
  secs/encoder.cpp
  secs/item.cpp
  thread_pool.cpp
  wafer_map.cpp)
add_library(waferedge::waferedge ALIAS waferedge)
target_include_directories(waferedge PUBLIC ${PROJECT_SOURCE_DIR}/include)
# Public: hsms/session.hpp includes Asio.
target_link_libraries(waferedge PUBLIC asio::asio PRIVATE waferedge_options)
target_compile_definitions(waferedge
  PRIVATE
    WAFEREDGE_VERSION="${PROJECT_VERSION}"
    WAFEREDGE_BUILD_TYPE="$<IF:$<CONFIG:>,none,$<CONFIG>>"
  # Public: tools and tests that use the GPU path check it.
  PUBLIC
    WAFEREDGE_HAS_CUDA=$<BOOL:${WAFEREDGE_CUDA}>)

if(WAFEREDGE_CUDA)
  target_sources(waferedge PRIVATE
    gpu_info.cpp
    ${PROJECT_SOURCE_DIR}/cuda/cluster_kernel.cu
    ${PROJECT_SOURCE_DIR}/cuda/cnn_engine.cu
    ${PROJECT_SOURCE_DIR}/cuda/cnn_kernels.cu
    ${PROJECT_SOURCE_DIR}/cuda/feature_kernels.cu
    ${PROJECT_SOURCE_DIR}/cuda/gemm.cu
    ${PROJECT_SOURCE_DIR}/cuda/hough_kernel.cu
    ${PROJECT_SOURCE_DIR}/cuda/signature_engine.cu)
  target_link_libraries(waferedge PUBLIC CUDA::cudart)
endif()
