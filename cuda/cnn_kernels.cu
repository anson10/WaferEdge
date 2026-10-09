// FabEye's CNN on the GPU: preprocessing, implicit-GEMM convolutions (fp32 and tensor-core
// fp16), max pooling, and the head (docs/inference.md). Activations are [channel][image][y][x].
// C++20 (nvcc 12.4's limit, ADR-0001).
#include "cnn_kernels.cuh"

#include <algorithm>
#include <cstdint>
#include <mma.h>

namespace waferedge::gpu::cnn_detail {

namespace {

constexpr int kSide = 64;
constexpr int kChannelsIn = 3;

// ---- preprocessing ---------------------------------------------------------------------------

// cv2.resize INTER_NEAREST's source index: floor(i * (1 / (64 / n))) in double precision,
// the same IEEE-rounded operations as the CPU's nearest_source, so the same pixel.
__device__ int nearest_source(int i, int n) {
    const double inv_scale = static_cast<double>(kSide) / n;
    const double step = 1.0 / inv_scale;
    return min(static_cast<int>(floor(i * step)), n - 1);
}

template <typename T>
__global__ void preprocess_kernel(const detail::MapDesc* descs, const std::uint8_t* bins,
                                  int images, T* out, T one) {
    const detail::MapDesc d = descs[blockIdx.x];
    const std::uint8_t* map = bins + d.bins;
    const int image = static_cast<int>(blockIdx.x);
    const std::size_t plane = static_cast<std::size_t>(images) * kSide * kSide; // one channel
    for (int i = static_cast<int>(threadIdx.x); i < kSide * kSide;
         i += static_cast<int>(blockDim.x)) {
        const int y = i / kSide;
        const int x = i % kSide;
        const std::uint8_t bin =
            map[nearest_source(y, d.rows) * d.cols + nearest_source(x, d.cols)];
        const int channel = bin == 0 ? 0 : (bin == 1 ? 1 : 2); // fail bins >= 2 are channel 2
        const std::size_t at =
            static_cast<std::size_t>(image) * kSide * kSide + static_cast<std::size_t>(i);
        for (int c = 0; c < kChannelsIn; ++c) {
            out[c * plane + at] = c == channel ? one : static_cast<T>(0.0F);
        }
    }
}

// ---- the implicit-GEMM gather ------------------------------------------------------------------

// One output pixel of a layer, p = image * side^2 + y * side + x, decomposed once per thread.
struct Pixel {
    int image;
    int y;
    int x;
};

__device__ Pixel decompose(int p, int side) {
    const int plane = side * side;
    return {p / plane, (p % plane) / side, p % side};
}

// Element (k, p) of the patch matrix P: input channel k / 9 at offset (ky, kx) = (k % 9) / 3,
// (k % 9) % 3 around pixel p, or 0 in the zero padding. Never stored: read from the
// activations while a tile is staged.
template <typename T>
__device__ T patch(const T* in, int k, Pixel px, int images, int side) {
    const int channel = k / 9;
    const int tap = k % 9;
    const int yy = px.y + tap / 3 - 1;
    const int xx = px.x + tap % 3 - 1;
    if (yy < 0 || yy >= side || xx < 0 || xx >= side) {
        return static_cast<T>(0.0F);
    }
    return in[((static_cast<std::size_t>(channel) * images + px.image) * side + yy) * side + xx];
}

// ---- fp32 convolution: register-blocked implicit GEMM -------------------------------------------
// A 256-thread block computes a 64 (out channels) x 64 (pixels) tile of the output; each thread
// a 4 x 4 patch in registers. Per step of 8 along K: the weights' 64 x 8 tile (stored
// transposed) and the patches' 8 x 64 tile in shared memory. Smaller tiles than the GEMM
// ladder's 128 x 128: the layers have 32 to 256 output channels.
constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 8;
constexpr int kTM = 4;
constexpr int kTN = 4;
constexpr int kThreads = (kBM / kTM) * (kBN / kTN); // 256

template <bool kFused>
__global__ void __launch_bounds__(kThreads)
    conv_fp32(ConvShape s, const float* weight, const float* bias, const float* in, float* out) {
    __shared__ alignas(16) float ws[kBK][kBM]; // transposed: ws[k][m]
    __shared__ alignas(16) float ps[kBK][kBN];
    const int m_total = s.out_channels;
    const int n_total = s.images * s.side * s.side;
    const int k_total = s.in_channels * 9;
    const int t = static_cast<int>(threadIdx.x);
    const int m0 = static_cast<int>(blockIdx.y) * kBM;
    const int n0 = static_cast<int>(blockIdx.x) * kBN;
    const int tm = (t / (kBN / kTN)) * kTM;
    const int tn = (t % (kBN / kTN)) * kTN;

    // Each thread stages the same 2 columns of the patch tile at every step: decompose their
    // pixels once.
    Pixel px[2];
    bool valid[2];
    for (int j = 0; j < 2; ++j) {
        const int p = n0 + (t + j * kThreads) % kBN;
        valid[j] = p < n_total;
        px[j] = decompose(valid[j] ? p : 0, s.side);
    }

    float acc[kTM][kTN] = {};
    for (int k0 = 0; k0 < k_total; k0 += kBK) {
        for (int j = 0; j < 2; ++j) {
            const int i = t + j * kThreads;
            const int r = i / kBK; // weight tile: 64 rows x 8
            const int kk = i % kBK;
            const int m = m0 + r;
            const int k = k0 + kk;
            ws[kk][r] = (m < m_total && k < k_total) ? weight[m * k_total + k] : 0.0F;
            const int pk = i / kBN; // patch tile: 8 rows x 64
            const int k_p = k0 + pk;
            ps[pk][i % kBN] =
                (valid[j] && k_p < k_total) ? patch(in, k_p, px[j], s.images, s.side) : 0.0F;
        }
        __syncthreads();
        for (int kk = 0; kk < kBK; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(&ws[kk][tm]);
            const float4 b = *reinterpret_cast<const float4*>(&ps[kk][tn]);
            const float av[kTM] = {a.x, a.y, a.z, a.w};
            const float bv[kTN] = {b.x, b.y, b.z, b.w};
            for (int i = 0; i < kTM; ++i) {
                for (int j = 0; j < kTN; ++j) {
                    acc[i][j] += av[i] * bv[j];
                }
            }
        }
        __syncthreads();
    }
    // Epilogue. Output element (m, p) is out[m * N + p]: channel m, pixel p, already the
    // [channel][image][y][x] layout the next layer reads. Fused: + bias, ReLU, here, while the
    // values are still in registers.
    for (int i = 0; i < kTM; ++i) {
        const int m = m0 + tm + i;
        if (m >= m_total) {
            break;
        }
        for (int j = 0; j < kTN; ++j) {
            const int p = n0 + tn + j;
            if (p < n_total) {
                const float v = acc[i][j];
                out[static_cast<std::size_t>(m) * n_total + p] =
                    kFused ? fmaxf(v + bias[m], 0.0F) : v;
            }
        }
    }
}

// ---- fp16 convolution: tensor-core implicit GEMM
// ------------------------------------------------- The GEMM ladder's WMMA rung with the patch
// gather: 4 warps own a 64 x 64 tile, each warp a 32 x 32 quarter (2 x 2 fragments of 16 x 16 x
// 16); per step of 32 along K the weights' tile is copied 16 bytes at a time (rows padded to a
// multiple of 32) and the patches' tile is gathered element by element. The epilogue goes through
// shared memory: fragments' element order is opaque, so they are stored to a float tile first, then
// bias + ReLU are applied and the result is written as fp16.
constexpr int kWmma = 16;
constexpr int kTcBM = 64;
constexpr int kTcBN = 64;
constexpr int kTcBK = 32;
constexpr int kTcPad = 8;
constexpr int kTcThreads = 128;

template <bool kFused>
__global__ void __launch_bounds__(kTcThreads)
    conv_fp16(ConvShape s, int padded_k, const __half* weight, const float* bias, const __half* in,
              __half* out) {
    namespace wmma = nvcuda::wmma;
    __shared__ alignas(32) __half ws[kTcBM][kTcBK + kTcPad];
    __shared__ alignas(32) __half ps[kTcBK][kTcBN + kTcPad];
    __shared__ alignas(32) float cs[kTcBM][kTcBN + 4];
    const int m_total = s.out_channels;
    const int n_total = s.images * s.side * s.side;
    const int k_total = s.in_channels * 9;
    const int t = static_cast<int>(threadIdx.x);
    const int warp = t / 32;
    const int warp_m = (warp / 2) * 32;
    const int warp_n = (warp % 2) * 32;
    const int m0 = static_cast<int>(blockIdx.y) * kTcBM;
    const int n0 = static_cast<int>(blockIdx.x) * kTcBN;

    // The patch tile is 32 x 64: thread t always gathers column t % 64, rows t / 64 + 2 j.
    const int my_col = t % kTcBN;
    const int p = n0 + my_col;
    const bool valid = p < n_total;
    const Pixel px = decompose(valid ? p : 0, s.side);

    wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, float> acc[2][2];
    for (auto& row : acc) {
        for (auto& f : row) {
            wmma::fill_fragment(f, 0.0F);
        }
    }
    const uint4 zero = make_uint4(0, 0, 0, 0);
    for (int k0 = 0; k0 < padded_k; k0 += kTcBK) {
        for (int i = t; i < kTcBM * kTcBK / 8; i += kTcThreads) { // weights: 16-byte chunks
            const int r = i / (kTcBK / 8);
            const int kk = (i % (kTcBK / 8)) * 8;
            const int m = m0 + r;
            *reinterpret_cast<uint4*>(&ws[r][kk]) =
                m < m_total ? *reinterpret_cast<const uint4*>(&weight[m * padded_k + k0 + kk])
                            : zero;
        }
        for (int kk = t / kTcBN; kk < kTcBK; kk += kTcThreads / kTcBN) { // patches: gathered
            const int k = k0 + kk;
            ps[kk][my_col] =
                (valid && k < k_total) ? patch(in, k, px, s.images, s.side) : __float2half(0.0F);
        }
        __syncthreads();
        for (int kk = 0; kk < kTcBK; kk += kWmma) {
            wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, __half, wmma::row_major> fb[2];
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(fa[i], &ws[warp_m + i * kWmma][kk], kTcBK + kTcPad);
                wmma::load_matrix_sync(fb[i], &ps[kk][warp_n + i * kWmma], kTcBN + kTcPad);
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&cs[warp_m + i * kWmma][warp_n + j * kWmma], acc[i][j],
                                    kTcBN + 4, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int i = t; i < kTcBM * kTcBN; i += kTcThreads) {
        const int r = i / kTcBN;
        const int cc = i % kTcBN;
        const int m = m0 + r;
        const int pp = n0 + cc;
        if (m < m_total && pp < n_total) {
            const float v = cs[r][cc];
            out[static_cast<std::size_t>(m) * n_total + pp] =
                __float2half(kFused ? fmaxf(v + bias[m], 0.0F) : v);
        }
    }
}

// ---- the rest
// ------------------------------------------------------------------------------------

template <typename T>
__global__ void bias_relu_kernel(ConvShape s, const float* bias, T* x) {
    const std::size_t pixels = static_cast<std::size_t>(s.images) * s.side * s.side;
    const std::size_t total = pixels * s.out_channels;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        x[i] = static_cast<T>(fmaxf(static_cast<float>(x[i]) + bias[i / pixels], 0.0F));
    }
}

template <typename T>
__global__ void maxpool_kernel(int channels, int images, int side, const T* in, T* out) {
    const int half = side / 2;
    const std::size_t total = static_cast<std::size_t>(channels) * images * half * half;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t plane =
            i / (static_cast<std::size_t>(half) * half); // channel * images + image
        const int y = static_cast<int>((i / half) % half);
        const int x = static_cast<int>(i % half);
        const T* src = in + (plane * side + 2 * y) * side + 2 * x;
        const float m =
            fmaxf(fmaxf(static_cast<float>(src[0]), static_cast<float>(src[1])),
                  fmaxf(static_cast<float>(src[side]), static_cast<float>(src[side + 1])));
        out[i] = static_cast<T>(m);
    }
}

// One block per image: 256 threads average one channel's 4 x 4 each, then 9 threads compute
// the logits.
template <typename T>
__global__ void head_kernel(int images, const T* in, float scale, const float* fc_weight,
                            const float* fc_bias, float* logits) {
    constexpr int kFeatures = 256;
    __shared__ float pooled[kFeatures];
    const int image = static_cast<int>(blockIdx.x);
    const int c = static_cast<int>(threadIdx.x);
    const T* src = in + (static_cast<std::size_t>(c) * images + image) * 16;
    float sum = 0.0F;
    for (int i = 0; i < 16; ++i) {
        sum += static_cast<float>(src[i]);
    }
    pooled[c] = sum * scale / 16.0F; // scale: int8 back to float (1 otherwise)
    __syncthreads();
    if (c < 9) {
        float z = fc_bias[c];
        for (int f = 0; f < kFeatures; ++f) {
            z += fc_weight[c * kFeatures + f] * pooled[f];
        }
        logits[image * 9 + c] = z;
    }
}

// ---- int8 convolution: tensor-core implicit GEMM ------------------------------------------------
// Signed 8-bit weights and activations, int32 accumulation: the same structure as conv_fp16
// with two differences forced by int8 fragments. Their loads need 32-byte aligned addresses,
// and a 16-element step along K inside a 32-wide tile lands on a 16-byte boundary: so K steps
// by 16 (one fragment deep), and the patch tile is stored transposed ([pixel][k], read as a
// column-major fragment) so every fragment starts on an aligned row. The epilogue turns the
// exact int32 sum back into a float (scale[c] = weight scale * input scale), adds the bias,
// applies ReLU and requantises to 0..127 for the next layer.
constexpr int kI8BK = 16;
constexpr int kI8Pad = 16; // rows of 32 bytes

__global__ void __launch_bounds__(kTcThreads)
    conv_int8(ConvShape s, int padded_k, const signed char* weight, const float* scale,
              const float* bias, float out_inv, const signed char* in, signed char* out) {
    namespace wmma = nvcuda::wmma;
    __shared__ alignas(32) signed char ws[kTcBM][kI8BK + kI8Pad];
    __shared__ alignas(32) signed char ps[kTcBN][kI8BK + kI8Pad]; // transposed: ps[pixel][k]
    __shared__ alignas(32) int cs[kTcBM][kTcBN + 4];
    const int m_total = s.out_channels;
    const int n_total = s.images * s.side * s.side;
    const int k_total = s.in_channels * 9;
    const int t = static_cast<int>(threadIdx.x);
    const int warp = t / 32;
    const int warp_m = (warp / 2) * 32;
    const int warp_n = (warp % 2) * 32;
    const int m0 = static_cast<int>(blockIdx.y) * kTcBM;
    const int n0 = static_cast<int>(blockIdx.x) * kTcBN;
    const int my_col = t % kTcBN; // this thread gathers pixel column my_col, k rows t/64 + 2 j
    const int p = n0 + my_col;
    const bool valid = p < n_total;
    const Pixel px = decompose(valid ? p : 0, s.side);

    wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, int> acc[2][2];
    for (auto& row : acc) {
        for (auto& f : row) {
            wmma::fill_fragment(f, 0);
        }
    }
    for (int k0 = 0; k0 < padded_k; k0 += kI8BK) {
        if (t < kTcBM) { // weights: 64 rows of 16 bytes, one uint4 each
            const int m = m0 + t;
            *reinterpret_cast<uint4*>(&ws[t][0]) =
                m < m_total ? *reinterpret_cast<const uint4*>(&weight[m * padded_k + k0])
                            : make_uint4(0, 0, 0, 0);
        }
        for (int kk = t / kTcBN; kk < kI8BK; kk += kTcThreads / kTcBN) {
            const int k = k0 + kk;
            ps[my_col][kk] = (valid && k < k_total) ? patch(in, k, px, s.images, s.side)
                                                    : static_cast<signed char>(0);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, signed char, wmma::row_major> fa[2];
        wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, signed char, wmma::col_major> fb[2];
        for (int i = 0; i < 2; ++i) {
            wmma::load_matrix_sync(fa[i], &ws[warp_m + i * kWmma][0], kI8BK + kI8Pad);
            wmma::load_matrix_sync(fb[i], &ps[warp_n + i * kWmma][0], kI8BK + kI8Pad);
        }
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&cs[warp_m + i * kWmma][warp_n + j * kWmma], acc[i][j],
                                    kTcBN + 4, wmma::mem_row_major);
        }
    }
    __syncthreads();
    for (int i = t; i < kTcBM * kTcBN; i += kTcThreads) {
        const int r = i / kTcBN;
        const int cc = i % kTcBN;
        const int m = m0 + r;
        const int pp = n0 + cc;
        if (m < m_total && pp < n_total) {
            const float y = fmaxf(static_cast<float>(cs[r][cc]) * scale[m] + bias[m], 0.0F);
            out[static_cast<std::size_t>(m) * n_total + pp] =
                static_cast<signed char>(fminf(rintf(y * out_inv), 127.0F));
        }
    }
}

// Calibration: the largest value of x. Non-negative floats order like their bit patterns, so
// an integer atomicMax on the bits is a float max.
__global__ void max_kernel(const float* x, std::size_t n, unsigned* max_bits) {
    float m = 0.0F;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        m = fmaxf(m, x[i]);
    }
    for (int delta = 16; delta > 0; delta /= 2) {
        m = fmaxf(m, __shfl_down_sync(0xFFFFFFFFU, m, delta));
    }
    if (threadIdx.x % 32 == 0) {
        atomicMax(max_bits, __float_as_uint(m));
    }
}

unsigned blocks(int size, int per_block) {
    return static_cast<unsigned>((size + per_block - 1) / per_block);
}

unsigned grid_for(std::size_t total) {
    return static_cast<unsigned>(std::min<std::size_t>((total + 255) / 256, 65535));
}

} // namespace

cudaError_t launch_preprocess(unsigned images, const detail::MapDesc* descs,
                              const std::uint8_t* bins, float* out, cudaStream_t s) {
    preprocess_kernel<<<images, 256, 0, s>>>(descs, bins, static_cast<int>(images), out, 1.0F);
    return cudaGetLastError();
}

cudaError_t launch_preprocess(unsigned images, const detail::MapDesc* descs,
                              const std::uint8_t* bins, __half* out, cudaStream_t s) {
    preprocess_kernel<<<images, 256, 0, s>>>(descs, bins, static_cast<int>(images), out,
                                             __float2half(1.0F));
    return cudaGetLastError();
}

cudaError_t launch_preprocess(unsigned images, const detail::MapDesc* descs,
                              const std::uint8_t* bins, signed char* out, cudaStream_t s) {
    preprocess_kernel<<<images, 256, 0, s>>>(descs, bins, static_cast<int>(images), out,
                                             static_cast<signed char>(127));
    return cudaGetLastError();
}

cudaError_t launch_conv_int8(const ConvShape& s, int padded_k, const signed char* weight,
                             const float* scale, const float* bias, float out_inv,
                             const signed char* in, signed char* out, cudaStream_t stream) {
    const dim3 grid(blocks(s.images * s.side * s.side, kTcBN), blocks(s.out_channels, kTcBM));
    conv_int8<<<grid, kTcThreads, 0, stream>>>(s, padded_k, weight, scale, bias, out_inv, in, out);
    return cudaGetLastError();
}

cudaError_t launch_max(const float* x, std::size_t n, unsigned* max_bits, cudaStream_t stream) {
    max_kernel<<<grid_for(n), 256, 0, stream>>>(x, n, max_bits);
    return cudaGetLastError();
}

cudaError_t launch_conv_fp32(const ConvShape& s, const float* weight, const float* bias,
                             const float* in, float* out, bool fused, cudaStream_t stream) {
    const dim3 grid(blocks(s.images * s.side * s.side, kBN), blocks(s.out_channels, kBM));
    if (fused) {
        conv_fp32<true><<<grid, kThreads, 0, stream>>>(s, weight, bias, in, out);
    } else {
        conv_fp32<false><<<grid, kThreads, 0, stream>>>(s, weight, bias, in, out);
    }
    return cudaGetLastError();
}

cudaError_t launch_conv_fp16(const ConvShape& s, int padded_k, const __half* weight,
                             const float* bias, const __half* in, __half* out, bool fused,
                             cudaStream_t stream) {
    const dim3 grid(blocks(s.images * s.side * s.side, kTcBN), blocks(s.out_channels, kTcBM));
    if (fused) {
        conv_fp16<true><<<grid, kTcThreads, 0, stream>>>(s, padded_k, weight, bias, in, out);
    } else {
        conv_fp16<false><<<grid, kTcThreads, 0, stream>>>(s, padded_k, weight, bias, in, out);
    }
    return cudaGetLastError();
}

cudaError_t launch_bias_relu(const ConvShape& s, const float* bias, float* x, cudaStream_t stream) {
    bias_relu_kernel<<<grid_for(static_cast<std::size_t>(s.out_channels) * s.images * s.side *
                                s.side),
                       256, 0, stream>>>(s, bias, x);
    return cudaGetLastError();
}

cudaError_t launch_bias_relu(const ConvShape& s, const float* bias, __half* x,
                             cudaStream_t stream) {
    bias_relu_kernel<<<grid_for(static_cast<std::size_t>(s.out_channels) * s.images * s.side *
                                s.side),
                       256, 0, stream>>>(s, bias, x);
    return cudaGetLastError();
}

cudaError_t launch_maxpool(int channels, int images, int side, const float* in, float* out,
                           cudaStream_t stream) {
    maxpool_kernel<<<grid_for(static_cast<std::size_t>(channels) * images * (side / 2) *
                              (side / 2)),
                     256, 0, stream>>>(channels, images, side, in, out);
    return cudaGetLastError();
}

cudaError_t launch_maxpool(int channels, int images, int side, const __half* in, __half* out,
                           cudaStream_t stream) {
    maxpool_kernel<<<grid_for(static_cast<std::size_t>(channels) * images * (side / 2) *
                              (side / 2)),
                     256, 0, stream>>>(channels, images, side, in, out);
    return cudaGetLastError();
}

cudaError_t launch_head(int images, const float* in, const float* fc_weight, const float* fc_bias,
                        float* logits, cudaStream_t stream) {
    head_kernel<<<static_cast<unsigned>(images), 256, 0, stream>>>(images, in, 1.0F, fc_weight,
                                                                   fc_bias, logits);
    return cudaGetLastError();
}

cudaError_t launch_head(int images, const __half* in, const float* fc_weight, const float* fc_bias,
                        float* logits, cudaStream_t stream) {
    head_kernel<<<static_cast<unsigned>(images), 256, 0, stream>>>(images, in, 1.0F, fc_weight,
                                                                   fc_bias, logits);
    return cudaGetLastError();
}

cudaError_t launch_head(int images, const signed char* in, float scale, const float* fc_weight,
                        const float* fc_bias, float* logits, cudaStream_t stream) {
    head_kernel<<<static_cast<unsigned>(images), 256, 0, stream>>>(images, in, scale, fc_weight,
                                                                   fc_bias, logits);
    return cudaGetLastError();
}

cudaError_t launch_maxpool(int channels, int images, int side, const signed char* in,
                           signed char* out, cudaStream_t stream) {
    maxpool_kernel<<<grid_for(static_cast<std::size_t>(channels) * images * (side / 2) *
                              (side / 2)),
                     256, 0, stream>>>(channels, images, side, in, out);
    return cudaGetLastError();
}

} // namespace waferedge::gpu::cnn_detail
