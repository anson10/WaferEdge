// The reference forward pass: plain loops, in an order the compiler can vectorise. Every
// faster backend (the CUDA GEMM ladder, fused kernels, int8) is checked against this, and this
// is checked against ONNX Runtime's logits (docs/inference.md).
#include "waferedge/cnn.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>

namespace waferedge::cnn {

void conv3x3_relu(std::span<const float> in, int in_channels, int h, int w,
                  std::span<const float> weight, std::span<const float> bias, int out_channels,
                  std::span<float> out) noexcept {
    // Signed offsets throughout (std::ptrdiff_t): a kernel tap reads one row or column
    // outside the map before the padding check, and pointer arithmetic is signed anyway.
    const std::ptrdiff_t plane = std::ptrdiff_t{h} * w;
    assert(std::ssize(in) >= in_channels * plane && std::ssize(out) >= out_channels * plane);
    for (std::ptrdiff_t co = 0; co < out_channels; ++co) {
        float* o = out.data() + co * plane;
        std::fill(o, o + plane, bias[static_cast<std::size_t>(co)]);
        for (std::ptrdiff_t ci = 0; ci < in_channels; ++ci) {
            const float* src = in.data() + ci * plane;
            const float* k = weight.data() + (co * in_channels + ci) * 9;
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    const float wk = k[ky * 3 + kx];
                    // Output (y, x) reads input (y + ky - 1, x + kx - 1); outside the map is the
                    // zero padding, so only the overlapping rows and columns contribute.
                    const int dy = ky - 1;
                    const int dx = kx - 1;
                    const int y0 = std::max(0, -dy);
                    const int y1 = std::min(h, h - dy);
                    const int x0 = std::max(0, -dx);
                    const int x1 = std::min(w, w - dx);
                    for (int y = y0; y < y1; ++y) {
                        float* orow = o + std::ptrdiff_t{y} * w;
                        const float* irow = src + std::ptrdiff_t{y + dy} * w + dx;
                        for (int x = x0; x < x1; ++x) { // the vectorised loop
                            orow[x] += wk * irow[x];
                        }
                    }
                }
            }
        }
        for (std::ptrdiff_t i = 0; i < plane; ++i) {
            o[i] = std::max(o[i], 0.0F);
        }
    }
}

void maxpool2(std::span<const float> in, int channels, int h, int w,
              std::span<float> out) noexcept {
    const int oh = h / 2;
    const int ow = w / 2;
    for (std::ptrdiff_t c = 0; c < channels; ++c) {
        const float* src = in.data() + c * h * w;
        float* dst = out.data() + c * oh * ow;
        for (int y = 0; y < oh; ++y) {
            for (int x = 0; x < ow; ++x) {
                const float* p = src + (std::ptrdiff_t{2} * y) * w + std::ptrdiff_t{2} * x;
                dst[y * ow + x] = std::max({p[0], p[1], p[w], p[w + 1]});
            }
        }
    }
}

void ReferenceForward::run(const Model& model, std::span<const float> input,
                           std::span<float> logits) {
    assert(input.size() == kInputSize && logits.size() >= kClasses);
    const std::size_t largest = std::size_t{32} * kSide * kSide; // 32 x 64 x 64 after conv 1
    a_.resize(largest);
    b_.resize(largest);
    std::copy(input.begin(), input.end(), a_.begin());

    // a_ -> b_ (conv) -> a_ (conv) -> b_ (pool) -> a_ ... ping-pong between two buffers.
    int side = kSide;
    for (int block = 0; block < 4; ++block) {
        const Conv& c1 = model.conv[std::size_t{2} * static_cast<std::size_t>(block)];
        const Conv& c2 = model.conv[std::size_t{2} * static_cast<std::size_t>(block) + 1];
        conv3x3_relu(a_, c1.in, side, side, c1.weight, c1.bias, c1.out, b_);
        conv3x3_relu(b_, c2.in, side, side, c2.weight, c2.bias, c2.out, a_);
        maxpool2(a_, c2.out, side, side, b_);
        side /= 2;
        std::swap(a_, b_);
    }
    // a_ holds 256 x 4 x 4: global average pool, then the linear layer.
    constexpr int kFeatures = 256;
    const int plane = side * side;
    std::array<float, kFeatures> pooled{};
    for (int c = 0; c < kFeatures; ++c) {
        float sum = 0;
        for (int i = 0; i < plane; ++i) {
            sum += a_[static_cast<std::size_t>(c) * static_cast<std::size_t>(plane) +
                      static_cast<std::size_t>(i)];
        }
        pooled[static_cast<std::size_t>(c)] = sum / static_cast<float>(plane);
    }
    for (int k = 0; k < kClasses; ++k) {
        float z = model.fc_bias[static_cast<std::size_t>(k)];
        for (int c = 0; c < kFeatures; ++c) {
            z += model.fc_weight[static_cast<std::size_t>(k) * kFeatures +
                                 static_cast<std::size_t>(c)] *
                 pooled[static_cast<std::size_t>(c)];
        }
        logits[static_cast<std::size_t>(k)] = z;
    }
}

} // namespace waferedge::cnn
