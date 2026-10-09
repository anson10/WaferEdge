// The CUDA runtime API is plain C, so this host code stays C++23 and needs no nvcc.
#include "waferedge/machine.hpp"

#include <cuda_runtime_api.h>

namespace waferedge {

std::expected<GpuInfo, std::string> gpu_info(int device) {
    int count = 0;
    if (const auto err = cudaGetDeviceCount(&count); err != cudaSuccess) {
        return std::unexpected(cudaGetErrorString(err));
    }
    if (device < 0 || device >= count) {
        return std::unexpected("no CUDA device " + std::to_string(device));
    }
    cudaDeviceProp prop{};
    if (const auto err = cudaGetDeviceProperties(&prop, device); err != cudaSuccess) {
        return std::unexpected(cudaGetErrorString(err));
    }
    GpuInfo info{.name = prop.name,
                 .compute_major = prop.major,
                 .compute_minor = prop.minor,
                 .global_memory_bytes = prop.totalGlobalMem,
                 .multiprocessors = prop.multiProcessorCount,
                 .driver_version = 0,
                 .runtime_version = 0};
    cudaDriverGetVersion(&info.driver_version);
    cudaRuntimeGetVersion(&info.runtime_version);
    return info;
}

} // namespace waferedge
