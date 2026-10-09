# Compile settings shared by every WaferEdge target: link against waferedge_options.
add_library(waferedge_options INTERFACE)

target_compile_options(waferedge_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${WAFEREDGE_WARNINGS_AS_ERRORS}>>:-Werror>
  # nvcc forwards host warnings with -Xcompiler; -Wpedantic is left out because the CUDA
  # headers trip it.
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wshadow>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${WAFEREDGE_WARNINGS_AS_ERRORS}>>:-Werror=all-warnings>)

if(WAFEREDGE_SANITIZE)
  if(WAFEREDGE_SANITIZE STREQUAL "fuzzer")
    # Instrument everything for libFuzzer; the fuzz targets themselves link -fsanitize=fuzzer.
    set(_san "fuzzer-no-link,address,undefined")
  else()
    set(_san "${WAFEREDGE_SANITIZE}")
  endif()
  target_compile_options(waferedge_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=${_san} -fno-omit-frame-pointer -fno-sanitize-recover=all>)
  if(NOT WAFEREDGE_SANITIZE STREQUAL "fuzzer")
    target_link_options(waferedge_options INTERFACE -fsanitize=${_san})
  else()
    target_link_options(waferedge_options INTERFACE -fsanitize=address,undefined)
  endif()
endif()
