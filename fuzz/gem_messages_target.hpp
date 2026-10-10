#pragma once

#include <cstddef>
#include <cstdint>

// The GEM fuzz target's body, shared by fuzz-gem-messages and the corpus replay test.
namespace waferedge::fuzz {

// Splits `data` into records (flags, stream, function, 2-byte length, body) and feeds each
// to a communicating, on-line gem::Equipment or gem::Host as an HSMS event: a primary, a
// reply to one of their own transactions, a T3 timeout or a lost link. Everything they send
// back must decode as a SECS-II body. Aborts on any inconsistency; returns 0 otherwise.
int gem_messages(const std::uint8_t* data, std::size_t size);

} // namespace waferedge::fuzz
