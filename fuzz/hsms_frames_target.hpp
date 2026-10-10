#pragma once

#include <cstddef>
#include <cstdint>

// The HSMS framing fuzz target's body, shared by fuzz-hsms-frames and the corpus replay test.
namespace waferedge::fuzz {

// Feeds `data` to a passive hsms::Protocol as a TCP byte stream: byte 0 picks the chunk
// size and whether a valid Select.req comes first; the rest is the stream. Polls after
// every chunk on a fake clock, answers primaries, and checks the protocol's own output
// parses as frames. Aborts on any inconsistency; returns 0 otherwise.
int hsms_frames(const std::uint8_t* data, std::size_t size);

} // namespace waferedge::fuzz
