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
// K is ordered tap-major (the engine reorders the weights): k = tap * in_channels + channel,
// tap = ky * 3 + kx. Element (k, p) of the patch matrix P is input channel `channel` at
// (y + ky - 1, x + kx - 1) around pixel p, or 0 in the zero padding; it is never stored, only
// read while a tile is staged.
//
// Why tap-major: every layer but the first has a multiple of 32 input channels, so a K step
// (8, 16 or 32 deep) lies inside one tap. The tap, its bounds check and its address offset are
// then the same for the whole step, and each element is one load at a fixed stride (one
// channel plane) from the last. Channel-major order (PyTorch's) changes tap every element:
// two divisions, four compares and a 64-bit address per element.

// One output pixel, decomposed once per thread: its offset in channel 0 and which of its 9
// taps fall inside the image (bit tap; all 0 for a pixel past the end of the batch).
struct Pixel {
    int base;
    unsigned taps;
};

__device__ Pixel pixel_of(int p, bool valid, int side) {
    const int plane = side * side;
    const int image = p / plane;
    const int y = (p % plane) / side;
    const int x = p % side;
    unsigned taps = 0;
    for (int tap = 0; tap < 9; ++tap) {
        const int yy = y + tap / 3 - 1;
        const int xx = x + tap % 3 - 1;
        if (valid && yy >= 0 && yy < side && xx >= 0 && xx < side) {
            taps |= 1U << tap;
        }
    }
    return {image * plane + y * side + x, taps};
}

// Offset of a tap's input pixel relative to the output pixel, within one channel plane.
__device__ int tap_delta(int tap, int side) {
    return (tap / 3 - 1) * side + tap % 3 - 1;
}

// A convolution's grid is 1-D: block b computes output-channel tile b % m_tiles of pixel tile
// b / m_tiles. Blocks run roughly in index order, so the blocks that read the same input
// pixels (all channel tiles of a pixel tile) run together and share it through L2; with the
// pixel tile as the fast index, each channel tile re-read the whole input from DRAM (4x for
// the 256-channel layer, whose 4 MB input at batch 256 doesn't fit the 2 MB L2).
struct Tile {
    int m0;
    int n0;
};

__device__ Tile tile_of(int out_channels, int bm, int bn) {
    const int m_tiles = (out_channels + bm - 1) / bm;
    const int b = static_cast<int>(blockIdx.x);
    return {(b % m_tiles) * bm, (b / m_tiles) * bn};
}

// Gathers rows kk0 + j * step (j < rows) of a K step starting at k0, for one pixel: store(j,
// value), j a compile-time index once unrolled (so callers can pack values in registers).
// `channel_stride` is images * side^2 (activations are [channel][image][y][x], so one channel
// plane). Uniform case (in_channels a multiple of the step): one tap for all rows. Otherwise (the
// first layer, 3 channels; K padded past 27): per element, a tap of 9 or more has no bit in `taps`
// and reads as 0.
template <typename T, int kRows, int kStep, typename Store>
__device__ void gather(const T* in, int k0, int kk0, int in_channels, bool uniform, Pixel px,
                       int channel_stride, int side, Store store) {
    if (uniform) {
        const int tap = k0 / in_channels;
        const bool inside = (px.taps >> tap) & 1U;
        const int off =
            (k0 - tap * in_channels + kk0) * channel_stride + px.base + tap_delta(tap, side);
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
            store(j, inside ? in[off + j * kStep * channel_stride] : static_cast<T>(0.0F));
        }
    } else {
#pragma unroll
        for (int j = 0; j < kRows; ++j) {
            const int k = k0 + kk0 + j * kStep;
            const int tap = k / in_channels;
            const bool inside = (px.taps >> tap) & 1U; // tap <= 31 / 3: in range of the shift
            store(
                j,
                inside
                    ? in[(k - tap * in_channels) * channel_stride + px.base + tap_delta(tap, side)]
                    : static_cast<T>(0.0F));
        }
    }
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
    const auto [m0, n0] = tile_of(s.out_channels, kBM, kBN);
    const int tm = (t / (kBN / kTN)) * kTM;
    const int tn = (t % (kBN / kTN)) * kTN;

    // Each thread stages the same column of the patch tile (256 threads, 64 columns) at
    // every step, rows t / 64 and t / 64 + 4: decompose its pixel once.
    const int my_col = t % kBN;
    const Pixel px = pixel_of(n0 + my_col, n0 + my_col < n_total, s.side);
    const int channel_stride = n_total;
    const bool uniform = s.in_channels % kBK == 0;

    float acc[kTM][kTN] = {};
    for (int k0 = 0; k0 < k_total; k0 += kBK) {
        for (int j = 0; j < 2; ++j) {
            const int i = t + j * kThreads;
            const int r = i / kBK; // weight tile: 64 rows x 8
            const int kk = i % kBK;
            const int m = m0 + r;
            const int k = k0 + kk;
            ws[kk][r] = (m < m_total && k < k_total) ? weight[m * k_total + k] : 0.0F;
        }
        gather<float, 2, kThreads / kBN>(
            in, k0, t / kBN, s.in_channels, uniform, px, channel_stride, s.side,
            [&](int j, float v) { ps[t / kBN + j * (kThreads / kBN)][my_col] = v; });
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

// ---- fp16 convolution: tensor-core implicit GEMM ------------------------------------------------
// The GEMM ladder's WMMA rung with the patch gather: 4 warps own a 64 x 64 tile, each warp a
// 32 x 32 quarter (2 x 2 fragments of 16 x 16 x 16). Per step of 32 along K, the weights' tile
// is copied 16 bytes at a time (rows padded to a multiple of 32) and the patches' tile is
// gathered. The patch tile is stored transposed, ps[pixel][k] (read as a column-major
// fragment): thread t gathers 16 consecutive k of pixel t % 64, packs them and writes two
// 16-byte stores; 80-byte rows put 8 lanes' stores in 8 different groups of banks. (Row-major
// ps[k][pixel] took one 2-byte store per element.) The epilogue goes fragment by fragment
// through a per-warp scratch: fragments' element order is opaque, so each is stored as floats,
// then bias + ReLU are applied and the result is written as fp16.
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
    __shared__ alignas(32) __half ps[kTcBN][kTcBK + kTcPad]; // transposed: ps[pixel][k]
    // The epilogue's per-warp scratch reuses ws (free after the K loop's last barrier).
    static_assert(sizeof(ws) >= sizeof(float) * (kTcThreads / 32) * kWmma * (kWmma + 4));
    auto& cs = *reinterpret_cast<float(*)[kTcThreads / 32][kWmma][kWmma + 4]>(&ws[0][0]);
    const int m_total = s.out_channels;
    const int n_total = s.images * s.side * s.side;
    const int t = static_cast<int>(threadIdx.x);
    const int warp = t / 32;
    const int lane = t % 32;
    const int warp_m = (warp / 2) * 32;
    const int warp_n = (warp % 2) * 32;
    const auto [m0, n0] = tile_of(s.out_channels, kTcBM, kTcBN);
    const bool warp_has_rows = m0 + warp_m < m_total;

    // Thread t gathers k [16 * (t / 64), + 16) of pixel n0 + t % 64 at every step.
    const int my_col = t % kTcBN;
    const int half = t / kTcBN;
    const Pixel px = pixel_of(n0 + my_col, n0 + my_col < n_total, s.side);
    const bool uniform = s.in_channels % kTcBK == 0;

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
        unsigned packed[8] = {};
        gather<__half, 16, 1>(
            in, k0, half * 16, s.in_channels, uniform, px, n_total, s.side, [&](int j, __half v) {
                packed[j / 2] |= static_cast<unsigned>(__half_as_ushort(v)) << (16 * (j % 2));
            });
        auto* dst = reinterpret_cast<uint4*>(&ps[my_col][half * 16]);
        dst[0] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
        dst[1] = make_uint4(packed[4], packed[5], packed[6], packed[7]);
        __syncthreads();
        // Layers with 32 output channels fill half the 64-row tile: warps whose rows are all
        // padding skip the multiply (warp-uniform, no divergence) but still stage and sync.
        for (int kk = 0; kk < kTcBK && warp_has_rows; kk += kWmma) {
            wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, __half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, __half, wmma::col_major> fb[2];
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(fa[i], &ws[warp_m + i * kWmma][kk], kTcBK + kTcPad);
                wmma::load_matrix_sync(fb[i], &ps[warp_n + i * kWmma][kk], kTcBK + kTcPad);
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    // Epilogue, one 16 x 16 fragment at a time through the warp's own scratch, 8 values per
    // lane. A whole-tile float buffer (17 KB) capped the SM at 3 blocks.
    // Unrolled: acc[i][j] must have compile-time indices, or acc lives in local memory
    // (a 128-byte stack frame, 9x the DRAM writes when measured).
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&cs[warp][0][0], acc[i][j], kWmma + 4, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < kWmma * kWmma; e += 32) {
                const int m = m0 + warp_m + i * kWmma + e / kWmma;
                const int pp = n0 + warp_n + j * kWmma + e % kWmma;
                if (m < m_total && pp < n_total) {
                    const float v = cs[warp][e / kWmma][e % kWmma];
                    out[static_cast<std::size_t>(m) * n_total + pp] =
                        __float2half(kFused ? fmaxf(v + bias[m], 0.0F) : v);
                }
            }
            __syncwarp(); // the scratch is reused by the next fragment
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
// Signed 8-bit weights and activations, int32 accumulation: conv_fp16's structure, with two
// constraints of int8 fragments shaping the shared tiles. Fragment loads need 32-byte aligned
// addresses, and the second 16 of a 32-wide row start 16 bytes in: so a 32-deep K step is
// staged as two 16-deep halves, each its own array. The patch tile is stored transposed,
// ps[half][pixel][k] (read as a column-major fragment), so each thread gathers 16 consecutive
// k of its pixel, packs them into a uint4 and writes one 16-byte store; rows are 48 bytes
// apart so that 8 lanes' 16-byte stores fall in 8 different groups of banks (with 32-byte
// rows, byte stores into this layout were 8-way bank conflicts: 72M conflicts per conv2).
// The epilogue turns the exact int32 sum back into a float (scale[c] = weight scale * input
// scale), adds the bias, applies ReLU and requantises to 0..127 for the next layer.
constexpr int kI8BK = 32;
constexpr int kI8Half = 16;
constexpr int kI8Ld = 48; // bytes per staged row: 16 used, a multiple of 16, conflict-free

__global__ void __launch_bounds__(kTcThreads)
    conv_int8(ConvShape s, int padded_k, const signed char* weight, const float* scale,
              const float* bias, float out_inv, const signed char* in, signed char* out) {
    namespace wmma = nvcuda::wmma;
    __shared__ alignas(32) signed char ws[2][kTcBM][kI8Ld];
    __shared__ alignas(32) signed char ps[2][kTcBN][kI8Ld]; // transposed: ps[half][pixel][k]
    // The epilogue's per-warp scratch reuses ws (free after the K loop's last barrier):
    // 5 KB less shared memory, so registers, not shared memory, limit the blocks per SM.
    static_assert(sizeof(ws) >= sizeof(int) * (kTcThreads / 32) * kWmma * (kWmma + 4));
    auto& cs = *reinterpret_cast<int(*)[kTcThreads / 32][kWmma][kWmma + 4]>(&ws[0][0][0]);
    const int m_total = s.out_channels;
    const int n_total = s.images * s.side * s.side;
    const int t = static_cast<int>(threadIdx.x);
    const int warp = t / 32;
    const int lane = t % 32;
    const int warp_m = (warp / 2) * 32;
    const int warp_n = (warp % 2) * 32;
    const auto [m0, n0] = tile_of(s.out_channels, kTcBM, kTcBN);
    const bool warp_has_rows = m0 + warp_m < m_total;
    // Thread t stages row t % 64 of half t / 64: 16 weights of channel m0 + t % 64, and the
    // 16 patch values of pixel n0 + t % 64.
    const int row = t % kTcBM;
    const int half = t / kTcBM;
    const Pixel px = pixel_of(n0 + row, n0 + row < n_total, s.side);
    const bool uniform = s.in_channels % kI8BK == 0;

    wmma::fragment<wmma::accumulator, kWmma, kWmma, kWmma, int> acc[2][2];
    for (auto& r : acc) {
        for (auto& f : r) {
            wmma::fill_fragment(f, 0);
        }
    }
    for (int k0 = 0; k0 < padded_k; k0 += kI8BK) {
        const int m = m0 + row;
        *reinterpret_cast<uint4*>(&ws[half][row][0]) =
            m < m_total
                ? *reinterpret_cast<const uint4*>(&weight[m * padded_k + k0 + half * kI8Half])
                : make_uint4(0, 0, 0, 0);
        unsigned packed[4] = {};
        gather<signed char, kI8Half, 1>(in, k0, half * kI8Half, s.in_channels, uniform, px, n_total,
                                        s.side, [&](int j, signed char v) {
                                            packed[j / 4] |=
                                                static_cast<unsigned>(static_cast<unsigned char>(v))
                                                << (8 * (j % 4));
                                        });
        *reinterpret_cast<uint4*>(&ps[half][row][0]) =
            make_uint4(packed[0], packed[1], packed[2], packed[3]);
        __syncthreads();
#pragma unroll
        for (int h = 0; h < 2 && warp_has_rows; ++h) { // padding-only warps skip, as in fp16
            wmma::fragment<wmma::matrix_a, kWmma, kWmma, kWmma, signed char, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, kWmma, kWmma, kWmma, signed char, wmma::col_major> fb[2];
            for (int i = 0; i < 2; ++i) {
                wmma::load_matrix_sync(fa[i], &ws[h][warp_m + i * kWmma][0], kI8Ld);
                wmma::load_matrix_sync(fb[i], &ps[h][warp_n + i * kWmma][0], kI8Ld);
            }
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    // Epilogue per fragment through the warp's scratch, as in conv_fp16.
    // Unrolled: acc[i][j] must have compile-time indices, or acc lives in local memory
    // (a 128-byte stack frame, 9x the DRAM writes when measured).
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(&cs[warp][0][0], acc[i][j], kWmma + 4, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < kWmma * kWmma; e += 32) {
                const int m = m0 + warp_m + i * kWmma + e / kWmma;
                const int pp = n0 + warp_n + j * kWmma + e % kWmma;
                if (m < m_total && pp < n_total) {
                    const float y = fmaxf(
                        static_cast<float>(cs[warp][e / kWmma][e % kWmma]) * scale[m] + bias[m],
                        0.0F);
                    out[static_cast<std::size_t>(m) * n_total + pp] =
                        static_cast<signed char>(fminf(rintf(y * out_inv), 127.0F));
                }
            }
            __syncwarp();
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
    const unsigned grid = blocks(s.images * s.side * s.side, kTcBN) * blocks(s.out_channels, kTcBM);
    conv_int8<<<grid, kTcThreads, 0, stream>>>(s, padded_k, weight, scale, bias, out_inv, in, out);
    return cudaGetLastError();
}

cudaError_t launch_max(const float* x, std::size_t n, unsigned* max_bits, cudaStream_t stream) {
    max_kernel<<<grid_for(n), 256, 0, stream>>>(x, n, max_bits);
    return cudaGetLastError();
}

cudaError_t launch_conv_fp32(const ConvShape& s, const float* weight, const float* bias,
                             const float* in, float* out, bool fused, cudaStream_t stream) {
    const unsigned grid = blocks(s.images * s.side * s.side, kBN) * blocks(s.out_channels, kBM);
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
    const unsigned grid = blocks(s.images * s.side * s.side, kTcBN) * blocks(s.out_channels, kTcBM);
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
