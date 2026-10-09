cmake_minimum_required(VERSION 3.25)
project(waferedge VERSION 0.1.0 DESCRIPTION "In-line wafer-map watchdog at the equipment edge"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(WAFEREDGE_CUDA "Build the CUDA backend (needs nvcc 12.x)" OFF)
option(WAFEREDGE_TESTS "Build the Catch2 tests" ON)
option(WAFEREDGE_BENCH "Build the Google Benchmark binaries" ON)
option(WAFEREDGE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
set(WAFEREDGE_SANITIZE "" CACHE STRING "Sanitizers: empty, 'address,undefined', 'thread' or 'fuzzer'")

if(WAFEREDGE_CUDA)
  enable_language(CUDA)
  # nvcc 12.4 accepts C++20 at most: .cu files are C++20, the host code C++23 (ADR-0001).
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  find_package(CUDAToolkit REQUIRED)
endif()

include(cmake/Options.cmake)
include(cmake/Dependencies.cmake)

add_subdirectory(src)
add_subdirectory(tools)

if(WAFEREDGE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(WAFEREDGE_BENCH)
  add_subdirectory(bench)
endif()
