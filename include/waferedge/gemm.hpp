#pragma once

// The GEMM ladder (phase 2b, docs/inference.md): C = A * B on the GPU, one rung at a time,
// each checked against a CPU reference and measured against cuBLAS (benchmark only).
//
// Row-major throughout: A is M x K, B is K x N, C is M x N. Pointers are device pointers.
// C++20 and free of CUDA headers (ADR-0001): fp16 data travels as its 16-bit pattern.
#include <cstdint>
#include <span>
#include <string_view>

namespace waferedge::gpu {

enum class GemmKernel : std::uint8_t {
    naive,            // 1: a thread per element of C, reading A and B from global memory
    tiled,            // 2: 32 x 32 tiles of A and B staged in shared memory
    register_blocked, // 3: 128 x 128 block tiles, an 8 x 8 patch of C per thread in registers
    vectorized,       // 4: rung 3 with float4 loads and A transposed in shared memory
};

[[nodiscard]] std::string_view gemm_name(GemmKernel k) noexcept;

// fp32: C = A * B for any M, N, K >= 1. The vectorized rung needs K and N to be multiples of
// 4 (16-byte rows); for other sizes it runs the register-blocked rung. Asynchronous on
// `stream` (a cudaStream_t, or nullptr for the default stream); returns a cudaError_t as int.
int gemm(GemmKernel kernel, int m, int n, int k, const float* a, const float* b, float* c,
         void* stream = nullptr) noexcept;

// 5: tensor cores (WMMA): fp16 A and B, fp32 accumulation and C. M, N and K must be multiples
// of 16 (an engine pads to that); returns cudaErrorInvalidValue otherwise.
int gemm_tensor_core(int m, int n, int k, const std::uint16_t* a, const std::uint16_t* b, float* c,
                     void* stream = nullptr) noexcept;

// Round-to-nearest-even fp32 -> fp16 and back, on the host (same rounding as the device).
void to_half(std::span<const float> in, std::span<std::uint16_t> out) noexcept;
void from_half(std::span<const std::uint16_t> in, std::span<float> out) noexcept;

} // namespace waferedge::gpu
