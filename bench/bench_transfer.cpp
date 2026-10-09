// Host-to-device copy bandwidth, pageable vs pinned host memory. The sizes run from one 40x40
// map of int16 bins (3.2 KB) to a 64 MiB batch: the baseline for phase 2a's buffer design.
//
//   build/cuda/bench/bench-transfer --benchmark_min_time=0.5s
#include "waferedge/machine.hpp"

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {

void check(cudaError_t err, benchmark::State& state) {
    if (err != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(err));
    }
}

struct DeviceBuffer {
    void* ptr = nullptr;
    explicit DeviceBuffer(std::size_t bytes) { cudaMalloc(&ptr, bytes); }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

void copy_loop(benchmark::State& state, const void* host, std::size_t bytes) {
    DeviceBuffer device(bytes);
    if (device.ptr == nullptr) {
        state.SkipWithError("cudaMalloc failed");
        return;
    }
    // cudaMemcpy returns when the copy is done, so wall time per iteration is the copy time.
    for (auto _ : state) {
        check(cudaMemcpy(device.ptr, host, bytes, cudaMemcpyHostToDevice), state);
    }
    state.SetBytesProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(bytes));
}

void BM_H2D_pageable(benchmark::State& state) {
    const auto bytes = static_cast<std::size_t>(state.range(0));
    std::vector<std::byte> host(bytes, std::byte{1});
    copy_loop(state, host.data(), bytes);
}

void BM_H2D_pinned(benchmark::State& state) {
    const auto bytes = static_cast<std::size_t>(state.range(0));
    void* host = nullptr;
    check(cudaMallocHost(&host, bytes), state);
    if (host == nullptr) {
        return;
    }
    std::memset(host, 1, bytes);
    copy_loop(state, host, bytes);
    cudaFreeHost(host);
}

constexpr std::int64_t one_map = 40 * 40 * 2;
BENCHMARK(BM_H2D_pageable)->Arg(one_map)->RangeMultiplier(16)->Range(4 << 10, 64 << 20);
BENCHMARK(BM_H2D_pinned)->Arg(one_map)->RangeMultiplier(16)->Range(4 << 10, 64 << 20);

} // namespace

int main(int argc, char** argv) {
    benchmark::AddCustomContext("waferedge", waferedge::describe_machine());
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
