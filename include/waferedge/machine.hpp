#pragma once

#include <cstddef>
#include <expected>
#include <string>

// What a measured number was measured on. Every benchmark prints this next to its results,
// so a speed claim always names the build and the machine.
namespace waferedge {

struct BuildInfo {
    std::string version;    // project version, e.g. "0.1.0"
    std::string compiler;   // e.g. "gcc 13.1.0"
    std::string build_type; // CMAKE_BUILD_TYPE
    long cxx_standard;      // __cplusplus of the host code
    bool cuda_enabled;      // built with the CUDA backend
};

struct CpuInfo {
    std::string model; // "model name" from /proc/cpuinfo, empty if unknown
    unsigned hardware_threads;
    bool avx2;
    bool fma;
};

struct GpuInfo {
    std::string name;
    int compute_major;
    int compute_minor;
    std::size_t global_memory_bytes;
    int multiprocessors;
    int driver_version;  // CUDA driver API version, e.g. 12040 for 12.4
    int runtime_version; // CUDA runtime the binary was built against
};

[[nodiscard]] BuildInfo build_info();
[[nodiscard]] CpuInfo cpu_info();

// The error is a readable reason: built without CUDA, no device, or the CUDA error string.
[[nodiscard]] std::expected<GpuInfo, std::string> gpu_info(int device = 0);

// All of the above as a few "key: value" lines.
[[nodiscard]] std::string describe_machine();

} // namespace waferedge
