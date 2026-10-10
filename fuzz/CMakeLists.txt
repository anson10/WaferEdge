# The fuzz targets' bodies build in every configuration: the regression test replays the
# checked-in corpus through them (tests/test_fuzz_corpus.cpp). The libFuzzer binaries need
# Clang's -fsanitize=fuzzer: the fuzz preset only.
add_library(waferedge_fuzz_targets STATIC secs_item_target.cpp)
target_include_directories(waferedge_fuzz_targets PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(waferedge_fuzz_targets PUBLIC waferedge PRIVATE waferedge_options)

if(WAFEREDGE_SANITIZE STREQUAL "fuzzer")
  add_executable(fuzz-secs-item fuzz_secs_item.cpp)
  target_link_libraries(fuzz-secs-item PRIVATE waferedge_fuzz_targets waferedge_options)
  target_link_options(fuzz-secs-item PRIVATE -fsanitize=fuzzer)
  # Ubuntu's libFuzzer runtime (libclang-rt-18-dev) is built against libstdc++, while the
  # fuzz preset uses libc++ (ADR-0001): link libstdc++ for the runtime's own use. The two
  # don't meet (std:: vs std::__1), and both follow the Itanium C++ ABI. Named by file:
  # with -stdlib=libc++, Clang's driver rewrites a plain -lstdc++ into -lc++.
  target_link_libraries(fuzz-secs-item PRIVATE -l:libstdc++.so.6)
  if(WAFEREDGE_TESTS)
    # -runs=0: execute every corpus file once, no mutation.
    add_test(NAME fuzz-secs-item-corpus
             COMMAND fuzz-secs-item -runs=0 ${CMAKE_CURRENT_SOURCE_DIR}/corpus/secs_item)
  endif()
endif()
