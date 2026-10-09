// Toolchain check: nvcc, the host compiler and sm_86 code agree with the host on a trivial
// kernel. Not a pattern for real kernels (those reuse device buffers, phase 2a).
#include "iota_kernel.hpp"

#include <cuda_runtime.h>

namespace {

__global__ void iota(int* out, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        out[i] = i;
    }
}

} // namespace

namespace waferedge::test {

std::string iota_on_device(std::span<int> out) {
    const int n = static_cast<int>(out.size());
    const std::size_t bytes = out.size_bytes();
    int* device = nullptr;
    if (auto err = cudaMalloc(&device, bytes); err != cudaSuccess) {
        return cudaGetErrorString(err);
    }
    constexpr int block = 256;
    iota<<<(n + block - 1) / block, block>>>(device, n);
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaMemcpy(out.data(), device, bytes, cudaMemcpyDeviceToHost);
    }
    cudaFree(device);
    return err == cudaSuccess ? std::string{} : std::string{cudaGetErrorString(err)};
}

} // namespace waferedge::test
