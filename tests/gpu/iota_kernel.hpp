#pragma once

// Shared by a C++23 test and a C++20 .cu file: keep it to C++20 and free of CUDA headers.
#include <span>
#include <string>

namespace waferedge::test {

// Writes out[i] = i on the device and copies it back. Returns an empty string on success,
// the CUDA error otherwise.
std::string iota_on_device(std::span<int> out);

} // namespace waferedge::test
