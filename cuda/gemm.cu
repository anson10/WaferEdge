// The GEMM ladder: C = A * B, row-major, one rung per kernel (docs/inference.md).
// C++20 (nvcc 12.4's limit, ADR-0001).
#include "waferedge/gemm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <mma.h>

namespace waferedge::gpu {

namespace {

// ---- Rung 1: naive -------------------------------------------------------------------------
// One thread per element of C. threadIdx.x runs along a row of C, so the 32 threads of a warp
// read 32 consecutive elements of B (one coalesced transaction) and all read the same
// element of A (one broadcast). Every multiply-add still fetches two floats through the
// memory system: ~0.25 FLOP per byte, far from what the arithmetic units could use.
__global__ void gemm_naive(int m, int n, int k, const float* a, const float* b, float* c) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0F;
    for (int i = 0; i < k; ++i) {
        acc += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = acc;
}

// ---- Rung 2: shared-memory tiling ----------------------------------------------------------
// A 32 x 32 block computes a 32 x 32 tile of C. It walks along K in steps of 32: each step,
// every thread loads one element of A's tile and one of B's into shared memory, the block
// syncs, and each thread does 32 multiply-adds from shared memory. Each value loaded from
// global memory is used by 32 threads: 32x fewer global reads than rung 1. Out-of-range
// elements (edges of matrices that aren't multiples of 32) load as 0.
constexpr int kTile = 32;

__global__ void gemm_tiled(int m, int n, int k, const float* a, const float* b, float* c) {
    __shared__ float as[kTile][kTile];
    __shared__ float bs[kTile][kTile];
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int row = static_cast<int>(blockIdx.y) * kTile + ty;
    const int col = static_cast<int>(blockIdx.x) * kTile + tx;
    float acc = 0.0F;
    for (int t = 0; t < k; t += kTile) {
        as[ty][tx] = (row < m && t + tx < k) ? a[row * k + t + tx] : 0.0F;
        bs[ty][tx] = (t + ty < k && col < n) ? b[(t + ty) * n + col] : 0.0F;
        __syncthreads(); // both tiles complete
        // as[ty][i]: the same address across the warp (a broadcast); bs[i][tx]: 32
        // consecutive floats, one per bank: no bank conflicts.
        for (int i = 0; i < kTile; ++i) {
            acc += as[ty][i] * bs[i][tx];
        }
        __syncthreads(); // everyone is done with these tiles before the next load
    }
    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

// ---- Rung 3: register blocking -------------------------------------------------------------
// A block of 256 threads computes a 128 x 128 tile of C; each thread an 8 x 8 patch of it,
// held in 64 registers. Per step along K, a thread loads 8 values of A and 8 of B from shared
// memory into registers and does all 64 multiply-adds between them: 0.25 shared loads per
// multiply-add instead of rung 2's 2. Tiles: A 128 x 8 and B 8 x 128 in shared memory.
constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 8;
constexpr int kTM = 8;
constexpr int kTN = 8;
constexpr int kThreads = (kBM / kTM) * (kBN / kTN); // 256

__global__ void __launch_bounds__(kThreads)
    gemm_register_blocked(int m, int n, int k, const float* a, const float* b, float* c) {
    __shared__ float as[kBM][kBK];
    __shared__ float bs[kBK][kBN];
    const int t = static_cast<int>(threadIdx.x);
    const int block_row = static_cast<int>(blockIdx.y) * kBM;
    const int block_col = static_cast<int>(blockIdx.x) * kBN;
    const int thread_row = (t / (kBN / kTN)) * kTM; // this thread's patch inside the tile
    const int thread_col = (t % (kBN / kTN)) * kTN;

    float acc[kTM][kTN] = {};
    float reg_a[kTM];
    float reg_b[kTN];
    for (int step = 0; step < k; step += kBK) {
        // 1024 values per tile, 4 per thread; out-of-range ones are 0.
        for (int i = t; i < kBM * kBK; i += kThreads) {
            const int r = i / kBK;
            const int cc = i % kBK;
            const int gr = block_row + r;
            const int gc = step + cc;
            as[r][cc] = (gr < m && gc < k) ? a[gr * k + gc] : 0.0F;
        }
        for (int i = t; i < kBK * kBN; i += kThreads) {
            const int r = i / kBN;
            const int cc = i % kBN;
            const int gr = step + r;
            const int gc = block_col + cc;
            bs[r][cc] = (gr < k && gc < n) ? b[gr * n + gc] : 0.0F;
        }
        __syncthreads();
        for (int d = 0; d < kBK; ++d) {
            for (int i = 0; i < kTM; ++i) {
                reg_a[i] = as[thread_row + i][d];
            }
            for (int j = 0; j < kTN; ++j) {
                reg_b[j] = bs[d][thread_col + j];
            }
            for (int i = 0; i < kTM; ++i) { // 64 multiply-adds from 16 loads
                for (int j = 0; j < kTN; ++j) {
                    acc[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < kTM; ++i) {
        const int gr = block_row + thread_row + i;
        for (int j = 0; j < kTN; ++j) {
            const int gc = block_col + thread_col + j;
            if (gr < m && gc < n) {
                c[gr * n + gc] = acc[i][j];
            }
        }
    }
}

// ---- Rung 4: vectorised -------------------------------------------------------------------
// Rung 3 with two changes. Tiles are loaded with float4 (16 bytes per instruction; one load
// of A and one of B per thread per step), which needs K and N to be multiples of 4. And A is
// stored transposed (as[k][m]), so a thread's 8 values of A for one step are contiguous: two
// float4 reads from shared memory instead of eight strided ones; B's 8 values likewise.
__global__ void __launch_bounds__(kThreads)
    gemm_vectorized(int m, int n, int k, const float* a, const float* b, float* c) {
    __shared__ alignas(16) float as[kBK][kBM]; // transposed: as[k][m]
    __shared__ alignas(16) float bs[kBK][kBN];
    const int t = static_cast<int>(threadIdx.x);
    const int block_row = static_cast<int>(blockIdx.y) * kBM;
    const int block_col = static_cast<int>(blockIdx.x) * kBN;
    const int thread_row = (t / (kBN / kTN)) * kTM;
    const int thread_col = (t % (kBN / kTN)) * kTN;
    // This thread's float4 of each tile: A rows of 8 floats are 2 float4s, B rows 32.
    const int a_row = t / (kBK / 4);
    const int a_col = (t % (kBK / 4)) * 4;
    const int b_row = t / (kBN / 4);
    const int b_col = (t % (kBN / 4)) * 4;

    float acc[kTM][kTN] = {};
    float reg_a[kTM];
    float reg_b[kTN];
    const float4 zero = make_float4(0, 0, 0, 0);
    for (int step = 0; step < k; step += kBK) {
        const int ar = block_row + a_row;
        const int ac = step + a_col; // K % 4 == 0: a float4 is wholly inside or outside
        const float4 va =
            (ar < m && ac < k) ? *reinterpret_cast<const float4*>(&a[ar * k + ac]) : zero;
        as[a_col + 0][a_row] = va.x;
        as[a_col + 1][a_row] = va.y;
        as[a_col + 2][a_row] = va.z;
        as[a_col + 3][a_row] = va.w;
        const int br = step + b_row;
        const int bc = block_col + b_col;
        *reinterpret_cast<float4*>(&bs[b_row][b_col]) =
            (br < k && bc < n) ? *reinterpret_cast<const float4*>(&b[br * n + bc]) : zero;
        __syncthreads();
        for (int d = 0; d < kBK; ++d) {
            const float4 a0 = *reinterpret_cast<const float4*>(&as[d][thread_row]);
            const float4 a1 = *reinterpret_cast<const float4*>(&as[d][thread_row + 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&bs[d][thread_col]);
            const float4 b1 = *reinterpret_cast<const float4*>(&bs[d][thread_col + 4]);
            reg_a[0] = a0.x, reg_a[1] = a0.y, reg_a[2] = a0.z, reg_a[3] = a0.w;
            reg_a[4] = a1.x, reg_a[5] = a1.y, reg_a[6] = a1.z, reg_a[7] = a1.w;
            reg_b[0] = b0.x, reg_b[1] = b0.y, reg_b[2] = b0.z, reg_b[3] = b0.w;
            reg_b[4] = b1.x, reg_b[5] = b1.y, reg_b[6] = b1.z, reg_b[7] = b1.w;
            for (int i = 0; i < kTM; ++i) {
                for (int j = 0; j < kTN; ++j) {
                    acc[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < kTM; ++i) {
        const int gr = block_row + thread_row + i;
        if (gr >= m) {
            break;
        }
        for (int j = 0; j < kTN; j += 4) {
            const int gc = block_col + thread_col + j;
            if (gc < n) { // N % 4 == 0: the float4 is wholly inside
                *reinterpret_cast<float4*>(&c[gr * n + gc]) =
                    make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
            }
        }
    }
}

// ---- Rung 5: tensor cores (WMMA) -----------------------------------------------------------
// fp16 A and B, fp32 accumulation. A warp computes a 16 x 16 x 16 product with one mma_sync
// on the tensor cores. A block of 4 warps (2 x 2) owns a 64 x 64 tile of C, each warp a
// 32 x 32 quarter (2 x 2 fragments). Per step of 32 along K, A (64 x 32) and B (32 x 64) are
// staged in shared memory as fp16; rows are padded by 8 halves so the warps' fragment loads
// don't all start on the same bank. M, N, K multiples of 16: a fragment is wholly inside C or
// wholly outside, and staging loads of 8 halves (16 bytes) are wholly inside or outside.
constexpr int kWmma = 16;
constexpr int kTcBM = 64;
constexpr int kTcBN = 64;
constexpr int kTcBK = 32;
constexpr int kTcPad = 8;
constexpr int kTcThreads = 128;

__global__ void __launch_bounds__(kTcThreads)
    gemm_wmma(int m, int n, int k, const __half* a, const __half* b, float* c) {
    namespace wmma = nvcuda::wmma;
    __shared__ alignas(32) __half as[kTcBM][kTcBK + kTcPad];
    __shared__ alignas(32) __half bs[kTcBK][kTcBN + kTcPad];
    const int t = static_cast<int>(threadIdx.x);
    const int warp = t / 32;
    const int warp_row = (warp / 2) * 32; // this warp's 32 x 32 quarter of the tile
    const int warp_col = (warp % 2) * 32;
    const int block_row = static_cast<int>(blockIdx.y) * kTcBM;
    const int block_col = static_cast<int>(blockIdx.x) * kTcBN;

    wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float> acc[2][2];
    for (auto& row : acc) {
        for (auto& f : row) {
            wmma::fill_fragment(f, 0.0F);
        }
    }
    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (int step = 0; step < k; step += kTcBK) {
        // A tile: 64 rows x 32 halves = 256 chunks of 8 halves (16 bytes), 2 per thread.
        for (int i = t; i < kTcBM * kTcBK / 8; i += kTcThreads) {
            const int r = i / (kTcBK / 8);
            const int cc = (i % (kTcBK / 8)) * 8;
            const int gr = block_row + r;
            const int gc = step + cc;
            *reinterpret_cast<uint4*>(&as[r][cc]) =
                (gr < m && gc < k) ? *reinterpret_cast<const uint4*>(&a[gr * k + gc]) : zero;
        }
        // B tile: 32 rows x 64 halves, likewise.
        for (int i = t; i < kTcBK * kTcBN / 8; i += kTcThreads) {
            const int r = i / (kTcBN / 8);
            const int cc = (i % (kTcBN / 8)) * 8;
            const int gr = step + r;
            const int gc = block_col + cc;
            *reinterpret_cast<uint4*>(&bs[r][cc]) =
                (gr < k && gc < n) ? *reinterpret_cast<const uint4*>(&b[gr * n + gc]) : zero;
        }
        __syncthreads();
        for (int kk = 0; kk < kTcBK; kk += kWmma) {
            wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, __half, wmma::row_major> fb[2];
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(fa[i], &as[warp_row + i * kWmma][kk], kTcBK + kTcPad);
                wmma::load_matrix_sync(fb[i], &bs[kk][warp_col + i * kWmma], kTcBN + kTcPad);
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]); // 4,096 multiply-adds
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            const int gr = block_row + warp_row + i * kWmma;
            const int gc = block_col + warp_col + j * kWmma;
            if (gr < m && gc < n) {
                wmma::store_matrix_sync(&c[gr * n + gc], acc[i][j], n, wmma::mem_row_major);
            }
        }
    }
}

unsigned blocks(int size, int per_block) {
    return static_cast<unsigned>((size + per_block - 1) / per_block);
}

} // namespace

std::string_view gemm_name(GemmKernel k) noexcept {
    switch (k) {
    case GemmKernel::naive:
        return "naive";
    case GemmKernel::tiled:
        return "tiled";
    case GemmKernel::register_blocked:
        return "register_blocked";
    case GemmKernel::vectorized:
        return "vectorized";
    }
    return "?";
}

int gemm(GemmKernel kernel, int m, int n, int k, const float* a, const float* b, float* c,
         void* stream) noexcept {
    if (m < 1 || n < 1 || k < 1) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    auto* s = static_cast<cudaStream_t>(stream);
    switch (kernel) {
    case GemmKernel::naive: {
        const dim3 threads(32, 8);
        gemm_naive<<<dim3(blocks(n, 32), blocks(m, 8)), threads, 0, s>>>(m, n, k, a, b, c);
        break;
    }
    case GemmKernel::tiled:
        gemm_tiled<<<dim3(blocks(n, kTile), blocks(m, kTile)), dim3(kTile, kTile), 0, s>>>(m, n, k,
                                                                                           a, b, c);
        break;
    case GemmKernel::vectorized:
        if (k % 4 == 0 && n % 4 == 0) {
            gemm_vectorized<<<dim3(blocks(n, kBN), blocks(m, kBM)), kThreads, 0, s>>>(m, n, k, a, b,
                                                                                      c);
            break;
        }
        [[fallthrough]]; // rows not 16-byte aligned: rung 3
    case GemmKernel::register_blocked:
        gemm_register_blocked<<<dim3(blocks(n, kBN), blocks(m, kBM)), kThreads, 0, s>>>(m, n, k, a,
                                                                                        b, c);
        break;
    }
    return static_cast<int>(cudaGetLastError());
}

int gemm_tensor_core(int m, int n, int k, const std::uint16_t* a, const std::uint16_t* b, float* c,
                     void* stream) noexcept {
    if (m < 1 || n < 1 || k < 1 || m % kWmma != 0 || n % kWmma != 0 || k % kWmma != 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    gemm_wmma<<<dim3(blocks(n, kTcBN), blocks(m, kTcBM)), kTcThreads, 0,
                static_cast<cudaStream_t>(stream)>>>(m, n, k, reinterpret_cast<const __half*>(a),
                                                     reinterpret_cast<const __half*>(b), c);
    return static_cast<int>(cudaGetLastError());
}

void to_half(std::span<const float> in, std::span<std::uint16_t> out) noexcept {
    for (std::size_t i = 0; i < in.size(); ++i) {
        const __half h = __float2half_rn(in[i]);
        out[i] = __half_as_ushort(h);
    }
}

void from_half(std::span<const std::uint16_t> in, std::span<float> out) noexcept {
    for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = __half2float(__ushort_as_half(in[i]));
    }
}

} // namespace waferedge::gpu
