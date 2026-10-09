#pragma once

// The CNN's kernels, launched by the CNN engine (cuda/cnn_engine.cu). Activations are
// [channel][image][y][x] ("CNHW"); a convolution is the GEMM out = W * P with
//   M = out channels, N = images * H * W, K = in channels * 9 (in, ky, kx),
// where P (the patches) is never stored: tiles of it are gathered while they are staged.
#include "kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace waferedge::gpu::cnn_detail {

// Bins -> one-hot input, cv2-exact nearest resize to 64 x 64. out: 3 x images x 64 x 64.
cudaError_t launch_preprocess(unsigned images, const detail::MapDesc* descs,
                              const std::uint8_t* bins, float* out, cudaStream_t s);
cudaError_t launch_preprocess(unsigned images, const detail::MapDesc* descs,
                              const std::uint8_t* bins, __half* out, cudaStream_t s);

struct ConvShape {
    int in_channels;
    int out_channels;
    int images;
    int side; // H = W
};

// fp32 implicit-GEMM 3x3 convolution, padding 1. weight: out x (in * 9). fused: out =
// ReLU(conv + bias); otherwise out = conv (bias_relu does the rest).
cudaError_t launch_conv_fp32(const ConvShape& s, const float* weight, const float* bias,
                             const float* in, float* out, bool fused, cudaStream_t stream);

// Tensor-core implicit GEMM: weight fp16, out x padded_k (padded_k = in * 9 rounded up to a
// multiple of 32, zero-filled); bias fp32; fp32 accumulation.
cudaError_t launch_conv_fp16(const ConvShape& s, int padded_k, const __half* weight,
                             const float* bias, const __half* in, __half* out, bool fused,
                             cudaStream_t stream);

// The unfused path's second kernel: x = ReLU(x + bias[channel]).
cudaError_t launch_bias_relu(const ConvShape& s, const float* bias, float* x, cudaStream_t stream);
cudaError_t launch_bias_relu(const ConvShape& s, const float* bias, __half* x, cudaStream_t stream);

// 2 x 2 max pooling, stride 2: channels x images x side x side -> ... x side/2 x side/2.
cudaError_t launch_maxpool(int channels, int images, int side, const float* in, float* out,
                           cudaStream_t stream);
cudaError_t launch_maxpool(int channels, int images, int side, const __half* in, __half* out,
                           cudaStream_t stream);

// Global average pool over 4 x 4, then the linear layer: in 256 x images x 4 x 4 -> logits
// images x 9.
cudaError_t launch_head(int images, const float* in, const float* fc_weight, const float* fc_bias,
                        float* logits, cudaStream_t stream);
cudaError_t launch_head(int images, const __half* in, const float* fc_weight, const float* fc_bias,
                        float* logits, cudaStream_t stream);

} // namespace waferedge::gpu::cnn_detail
