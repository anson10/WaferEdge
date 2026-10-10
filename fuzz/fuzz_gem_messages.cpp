// libFuzzer entry point for the GEM layer (fuzz preset only):
//   build/fuzz/fuzz/fuzz-gem-messages -max_total_time=60 /tmp/gem-corpus fuzz/corpus/gem_messages
// docs/secs.md has the corpus and crash-file workflow.
#include "gem_messages_target.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    return waferedge::fuzz::gem_messages(data, size);
}
