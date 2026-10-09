#include "waferedge/machine.hpp"

#include <format>
#include <fstream>
#include <thread>
#include <utility>

namespace waferedge {

BuildInfo build_info() {
#if defined(__clang__)
    std::string compiler =
        std::format("clang {}.{}.{}", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    std::string compiler =
        std::format("gcc {}.{}.{}", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    std::string compiler = "unknown";
#endif
    return BuildInfo{
        .version = WAFEREDGE_VERSION,
        .compiler = std::move(compiler),
        .build_type = WAFEREDGE_BUILD_TYPE,
        .cxx_standard = __cplusplus,
        .cuda_enabled = WAFEREDGE_HAS_CUDA != 0,
    };
}

CpuInfo cpu_info() {
    CpuInfo info{.model = {},
                 .hardware_threads = std::thread::hardware_concurrency(),
                 .avx2 = false,
                 .fma = false};
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    // The builtin returns int on GCC and bool on Clang; a condition reads right on both.
    if (__builtin_cpu_supports("avx2")) {
        info.avx2 = true;
    }
    if (__builtin_cpu_supports("fma")) {
        info.fma = true;
    }
#endif
    std::ifstream cpuinfo("/proc/cpuinfo");
    for (std::string line; std::getline(cpuinfo, line);) {
        if (line.starts_with("model name")) {
            if (auto colon = line.find(':'); colon != std::string::npos) {
                info.model = line.substr(line.find_first_not_of(' ', colon + 1));
            }
            break;
        }
    }
    return info;
}

#if !WAFEREDGE_HAS_CUDA
std::expected<GpuInfo, std::string> gpu_info(int /*device*/) {
    return std::unexpected("built without CUDA (configure with the cuda preset)");
}
#endif

std::string describe_machine() {
    const auto build = build_info();
    const auto cpu = cpu_info();
    std::string text =
        std::format("waferedge: {} ({}, {}, __cplusplus {}{})\n", build.version, build.build_type,
                    build.compiler, build.cxx_standard, build.cuda_enabled ? ", CUDA" : "");
    text += std::format("cpu: {} ({} threads, AVX2 {}, FMA {})\n",
                        cpu.model.empty() ? "unknown" : cpu.model, cpu.hardware_threads,
                        cpu.avx2 ? "yes" : "no", cpu.fma ? "yes" : "no");
    if (const auto gpu = gpu_info()) {
        text += std::format("gpu: {} (sm_{}{}, {} SMs, {} MiB, driver {}.{}, runtime {}.{})\n",
                            gpu->name, gpu->compute_major, gpu->compute_minor, gpu->multiprocessors,
                            gpu->global_memory_bytes >> 20, gpu->driver_version / 1000,
                            gpu->driver_version % 1000 / 10, gpu->runtime_version / 1000,
                            gpu->runtime_version % 1000 / 10);
    } else {
        text += std::format("gpu: none ({})\n", gpu.error());
    }
    return text;
}

} // namespace waferedge
