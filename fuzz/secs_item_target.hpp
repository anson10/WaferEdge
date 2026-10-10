#pragma once

#include <cstddef>
#include <cstdint>

// The SECS-II item fuzz target's body, shared by the libFuzzer binary (fuzz-secs-item) and
// the regression test that replays the checked-in corpus in every build.
namespace waferedge::fuzz {

// Decodes `data` as a body and, if it is valid, walks it, re-encodes it and checks the
// copy decodes to the same tree. Aborts on any inconsistency; returns 0 otherwise.
int secs_item(const std::uint8_t* data, std::size_t size);

} // namespace waferedge::fuzz
