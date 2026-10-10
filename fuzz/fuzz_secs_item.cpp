// libFuzzer entry point for the SECS-II item decoder (fuzz preset only):
//   cmake --preset fuzz && cmake --build --preset fuzz --target fuzz-secs-item
//   mkdir -p /tmp/secs-corpus
//   build/fuzz/fuzz/fuzz-secs-item -max_total_time=60 /tmp/secs-corpus fuzz/corpus/secs_item
// New inputs go to the first directory; fuzz/corpus/secs_item is only read. docs/secs.md.
#include "secs_item_target.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    return waferedge::fuzz::secs_item(data, size);
}
