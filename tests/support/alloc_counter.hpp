#pragma once

// Counts heap allocations made by the calling thread. Linking alloc_counter.cpp replaces the
// global operator new / delete for the whole program, so only the allocation test and the
// SECS benchmark link it (as an object library: the replacement must not be dropped).
#include <cstdint>

namespace waferedge::alloc {

// Allocations by this thread since it started.
[[nodiscard]] std::uint64_t count() noexcept;

} // namespace waferedge::alloc
