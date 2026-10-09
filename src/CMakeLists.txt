add_library(waferedge
  classifier.cpp
  clusters.cpp
  features.cpp
  geometry.cpp
  hough.cpp
  machine.cpp
  map_file.cpp
  randomness.cpp
  wafer_map.cpp)
add_library(waferedge::waferedge ALIAS waferedge)
target_include_directories(waferedge PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(waferedge PRIVATE waferedge_options)
target_compile_definitions(waferedge PRIVATE
  WAFEREDGE_VERSION="${PROJECT_VERSION}"
  WAFEREDGE_BUILD_TYPE="$<IF:$<CONFIG:>,none,$<CONFIG>>"
  WAFEREDGE_HAS_CUDA=$<BOOL:${WAFEREDGE_CUDA}>)

if(WAFEREDGE_CUDA)
  target_sources(waferedge PRIVATE gpu_info.cpp)
  target_link_libraries(waferedge PUBLIC CUDA::cudart)
endif()
