// libFuzzer entry point for HSMS framing and the protocol engine (fuzz preset only):
//   build/fuzz/fuzz/fuzz-hsms-frames -max_total_time=60 /tmp/hsms-corpus fuzz/corpus/hsms_frames
// docs/secs.md has the corpus and crash-file workflow.
#include "hsms_frames_target.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    return waferedge::fuzz::hsms_frames(data, size);
}
