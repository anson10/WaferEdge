// FabEye's CNN on the GPU, end to end: bins up, logits down (docs/inference.md).
// C++20 (nvcc 12.4's limit, ADR-0001).
#include "cnn_kernels.cuh"
#include "waferedge/gpu_cnn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace waferedge::gpu {

namespace {

using cnn_detail::ConvShape;
using detail::MapDesc;

// Batches larger than this run in chunks: activations for 1,024 maps are 512 MB per buffer
// in fp32 (32 channels x 64 x 64 x 4 bytes x 1,024).
constexpr std::size_t kChunk = 1024;

std::size_t align16(std::size_t n) {
    return (n + 15) & ~std::size_t{15};
}

int padded_k(int in_channels) {
    return (in_channels * 9 + 31) / 32 * 32; // K = in * 9, rounded up for 32-deep WMMA steps
}

// PyTorch orders a convolution's weights [out][in][tap]; the kernels want K tap-major,
// [out][tap][in], so that one K step (a multiple of 8, 16 or 32) covers a single tap of
// consecutive channels and the gather checks its bounds once per step (cnn_kernels.cu).
std::vector<float> tap_major(const cnn::Conv& c) {
    std::vector<float> w(c.weight.size());
    const auto in = static_cast<std::size_t>(c.in);
    for (std::size_t o = 0; o < static_cast<std::size_t>(c.out); ++o) {
        for (std::size_t i = 0; i < in; ++i) {
            for (std::size_t tap = 0; tap < 9; ++tap) {
                w[(o * 9 + tap) * in + i] = c.weight[(o * in + i) * 9 + tap];
            }
        }
    }
    return w;
}

} // namespace

struct CnnEngine::Impl {
    CnnPrecision precision;
    bool fused = true;
    bool timing = false;
    std::string error;
    CnnTimings timings;
    cudaStream_t stream = nullptr;
    std::vector<cudaEvent_t> events;

    // Weights on the device, K tap-major: fp32 [out][9 * in], fp16 / int8 [out][padded k];
    // biases fp32.
    std::array<void*, cnn::kConvLayers> weight{};
    std::array<float*, cnn::kConvLayers> bias{};
    std::array<cnn::Conv, cnn::kConvLayers> shape{}; // in / out channels only
    float* fc_weight = nullptr;
    float* fc_bias = nullptr;

    // Per-batch buffers, grown as needed.
    std::uint8_t* host_in = nullptr;
    std::size_t host_in_cap = 0;
    std::uint8_t* dev_in = nullptr;
    std::size_t dev_in_cap = 0;
    void* act[2] = {nullptr, nullptr};
    std::size_t act_cap = 0; // bytes per buffer
    float* dev_logits = nullptr;
    float* host_logits = nullptr;
    std::size_t logits_cap = 0; // maps

    bool ok(cudaError_t e, const char* what) {
        if (e == cudaSuccess) {
            return true;
        }
        if (error.empty()) {
            error = std::string(what) + ": " + cudaGetErrorString(e);
        }
        return false;
    }

    template <typename T>
    bool upload(T*& dev, const T* host, std::size_t count) {
        return ok(cudaMalloc(reinterpret_cast<void**>(&dev), count * sizeof(T)),
                  "cudaMalloc weights") &&
               ok(cudaMemcpy(dev, host, count * sizeof(T), cudaMemcpyHostToDevice),
                  "upload weights");
    }

    bool grow(void*& p, std::size_t& cap, std::size_t need, bool pinned) {
        if (need <= cap) {
            return true;
        }
        pinned ? cudaFreeHost(p) : cudaFree(p);
        p = nullptr;
        cap = 0;
        const std::size_t want = need + need / 2;
        if (!ok(pinned ? cudaMallocHost(&p, want) : cudaMalloc(&p, want), "allocate")) {
            return false;
        }
        cap = want;
        return true;
    }

    // int8: per layer, scale[c] = weight scale of channel c * input activation scale, and
    // 1 / output activation scale; the last layer's output scale for the head.
    std::array<float*, cnn::kConvLayers> int8_scale{};
    std::array<float, cnn::kConvLayers> int8_out_inv{};
    float int8_head_scale = 1.0F;
    // Calibration (fp32): running maxima of each conv's output, as float bit patterns.
    unsigned* dev_max = nullptr;
    bool collect_max = false;

    std::size_t elem() const {
        return precision == CnnPrecision::fp32 ? 4 : (precision == CnnPrecision::fp16 ? 2 : 1);
    }

    // Runs one chunk of maps (<= kChunk), already packed in host_in.
    bool forward(unsigned images, std::size_t in_bytes, std::size_t desc_bytes, float* logits_out);
};

CnnEngine::CnnEngine(const cnn::Model& model, CnnPrecision precision,
                     const ActivationMax* activation_max)
    : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.precision = precision;
    if (precision == CnnPrecision::int8 && activation_max == nullptr) {
        m.error = "int8 needs calibrated activation maxima";
        return;
    }
    m.ok(cudaStreamCreate(&m.stream), "cudaStreamCreate");
    // One event per mark in forward(): before the upload, after it, after preprocessing, after
    // each of the 8 convolutions and 4 pools, after the head and after the download.
    m.events.resize(3 + cnn::kConvLayers + 4 + 2);
    for (auto& e : m.events) {
        m.ok(cudaEventCreate(&e), "cudaEventCreate");
    }
    for (std::size_t l = 0; l < cnn::kConvLayers; ++l) {
        const auto& c = model.conv[l];
        m.shape[l].in = c.in;
        m.shape[l].out = c.out;
        m.upload(m.bias[l], c.bias.data(), c.bias.size());
        const std::vector<float> weight = tap_major(c);
        if (precision == CnnPrecision::fp32) {
            float* w = nullptr;
            m.upload(w, weight.data(), weight.size());
            m.weight[l] = w;
        } else if (precision == CnnPrecision::int8) {
            // Symmetric per output channel: w = w_int * s_w[c], s_w[c] = max |w[c, :]| / 127.
            // Activations per layer: x = x_int * s_a, s_a = calibrated max / 127; the input's
            // one-hot is 0 / 127 (s_a = 1 / 127).
            const int k = c.in * 9;
            const int kp = padded_k(c.in);
            const float s_in = l == 0 ? 1.0F / 127.0F : (*activation_max)[l - 1] / 127.0F;
            const float s_out = (*activation_max)[l] / 127.0F;
            std::vector<signed char> w(static_cast<std::size_t>(c.out) * kp, 0);
            std::vector<float> scale(static_cast<std::size_t>(c.out));
            for (int o = 0; o < c.out; ++o) {
                float top = 0.0F;
                for (int i = 0; i < k; ++i) {
                    top = std::max(top, std::abs(weight[static_cast<std::size_t>(o) * k + i]));
                }
                const float s_w = top > 0.0F ? top / 127.0F : 1.0F;
                for (int i = 0; i < k; ++i) {
                    const float q =
                        std::nearbyint(weight[static_cast<std::size_t>(o) * k + i] / s_w);
                    w[static_cast<std::size_t>(o) * kp + i] =
                        static_cast<signed char>(std::clamp(q, -127.0F, 127.0F));
                }
                scale[static_cast<std::size_t>(o)] = s_w * s_in;
            }
            signed char* dw = nullptr;
            m.upload(dw, w.data(), w.size());
            m.weight[l] = dw;
            m.upload(m.int8_scale[l], scale.data(), scale.size());
            m.int8_out_inv[l] = s_out > 0.0F ? 1.0F / s_out : 0.0F;
            m.int8_head_scale = s_out;
        } else {
            // fp16, each row padded with zeros from in * 9 to a multiple of 32.
            const int k = c.in * 9;
            const int kp = padded_k(c.in);
            std::vector<__half> w(static_cast<std::size_t>(c.out) * kp, __float2half(0.0F));
            for (int o = 0; o < c.out; ++o) {
                for (int i = 0; i < k; ++i) {
                    w[static_cast<std::size_t>(o) * kp + i] =
                        __float2half(weight[static_cast<std::size_t>(o) * k + i]);
                }
            }
            __half* dw = nullptr;
            m.upload(dw, w.data(), w.size());
            m.weight[l] = dw;
        }
    }
    m.upload(m.fc_weight, model.fc_weight.data(), model.fc_weight.size());
    m.upload(m.fc_bias, model.fc_bias.data(), model.fc_bias.size());
}

CnnEngine::~CnnEngine() {
    if (!impl_) {
        return;
    }
    Impl& m = *impl_;
    for (std::size_t l = 0; l < cnn::kConvLayers; ++l) {
        cudaFree(m.weight[l]);
        cudaFree(m.bias[l]);
        cudaFree(m.int8_scale[l]);
    }
    cudaFree(m.dev_max);
    cudaFree(m.fc_weight);
    cudaFree(m.fc_bias);
    cudaFreeHost(m.host_in);
    cudaFree(m.dev_in);
    cudaFree(m.act[0]);
    cudaFree(m.act[1]);
    cudaFree(m.dev_logits);
    cudaFreeHost(m.host_logits);
    for (auto& e : m.events) {
        cudaEventDestroy(e);
    }
    cudaStreamDestroy(m.stream);
}

CnnEngine::CnnEngine(CnnEngine&&) noexcept = default;
CnnEngine& CnnEngine::operator=(CnnEngine&&) noexcept = default;

bool CnnEngine::activation_max(std::span<const WaferMapView> maps, ActivationMax& out) {
    Impl& m = *impl_;
    if (m.precision != CnnPrecision::fp32) {
        m.error = "activation_max needs the fp32 engine";
        return false;
    }
    if (m.dev_max == nullptr &&
        !m.ok(cudaMalloc(reinterpret_cast<void**>(&m.dev_max), cnn::kConvLayers * sizeof(unsigned)),
              "cudaMalloc")) {
        return false;
    }
    if (!m.ok(cudaMemset(m.dev_max, 0, cnn::kConvLayers * sizeof(unsigned)), "cudaMemset")) {
        return false;
    }
    std::vector<float> logits(maps.size() * cnn::kClasses);
    m.collect_max = true;
    const bool done = run(maps, logits);
    m.collect_max = false;
    std::array<unsigned, cnn::kConvLayers> bits{};
    if (!done || !m.ok(cudaMemcpy(bits.data(), m.dev_max, sizeof bits, cudaMemcpyDeviceToHost),
                       "download maxima")) {
        return false;
    }
    for (std::size_t l = 0; l < cnn::kConvLayers; ++l) {
        std::memcpy(&out[l], &bits[l], sizeof(float));
    }
    return true;
}

const std::string& CnnEngine::error() const noexcept {
    return impl_->error;
}
CnnPrecision CnnEngine::precision() const noexcept {
    return impl_->precision;
}
void CnnEngine::set_fused(bool on) noexcept {
    impl_->fused = on;
}
void CnnEngine::set_timing(bool on) noexcept {
    impl_->timing = on;
}
CnnTimings CnnEngine::last_timings() const noexcept {
    return impl_->timings;
}

bool CnnEngine::Impl::forward(unsigned images, std::size_t in_bytes, std::size_t desc_bytes,
                              float* logits_out) {
    std::size_t ev = 0;
    const auto mark = [&] {
        if (timing) {
            cudaEventRecord(events[ev], stream);
        }
        ++ev;
    };
    mark();
    if (!ok(cudaMemcpyAsync(dev_in, host_in, in_bytes, cudaMemcpyHostToDevice, stream), "upload")) {
        return false;
    }
    mark();
    const auto* descs = reinterpret_cast<const MapDesc*>(dev_in);
    const std::uint8_t* bins = dev_in + desc_bytes;
    const int n = static_cast<int>(images);
    cudaError_t e = cudaSuccess;
    switch (precision) {
    case CnnPrecision::fp32:
        e = cnn_detail::launch_preprocess(images, descs, bins, static_cast<float*>(act[0]), stream);
        break;
    case CnnPrecision::fp16:
        e = cnn_detail::launch_preprocess(images, descs, bins, static_cast<__half*>(act[0]),
                                          stream);
        break;
    case CnnPrecision::int8:
        e = cnn_detail::launch_preprocess(images, descs, bins, static_cast<signed char*>(act[0]),
                                          stream);
        break;
    }
    if (!ok(e, "preprocess")) {
        return false;
    }
    mark();
    int cur = 0; // act[cur] holds the current activations
    int side = cnn::kSide;
    for (int layer = 0; layer < cnn::kConvLayers; ++layer) {
        const auto l = static_cast<std::size_t>(layer);
        const ConvShape s{shape[l].in, shape[l].out, n, side};
        switch (precision) {
        case CnnPrecision::fp32:
            e = cnn_detail::launch_conv_fp32(s, static_cast<const float*>(weight[l]), bias[l],
                                             static_cast<const float*>(act[cur]),
                                             static_cast<float*>(act[1 - cur]), fused, stream);
            if (e == cudaSuccess && !fused) {
                e = cnn_detail::launch_bias_relu(s, bias[l], static_cast<float*>(act[1 - cur]),
                                                 stream);
            }
            if (e == cudaSuccess && collect_max) { // calibration: this layer's largest output
                e = cnn_detail::launch_max(static_cast<const float*>(act[1 - cur]),
                                           static_cast<std::size_t>(s.out_channels) * n * side *
                                               side,
                                           dev_max + l, stream);
            }
            break;
        case CnnPrecision::fp16:
            e = cnn_detail::launch_conv_fp16(s, padded_k(shape[l].in),
                                             static_cast<const __half*>(weight[l]), bias[l],
                                             static_cast<const __half*>(act[cur]),
                                             static_cast<__half*>(act[1 - cur]), fused, stream);
            if (e == cudaSuccess && !fused) {
                e = cnn_detail::launch_bias_relu(s, bias[l], static_cast<__half*>(act[1 - cur]),
                                                 stream);
            }
            break;
        case CnnPrecision::int8:
            e = cnn_detail::launch_conv_int8(
                s, padded_k(shape[l].in), static_cast<const signed char*>(weight[l]), int8_scale[l],
                bias[l], int8_out_inv[l], static_cast<const signed char*>(act[cur]),
                static_cast<signed char*>(act[1 - cur]), stream);
            break;
        }
        if (!ok(e, "conv")) {
            return false;
        }
        mark();
        cur = 1 - cur;
        if (layer % 2 == 1) { // end of a block: 2 x 2 max pooling
            switch (precision) {
            case CnnPrecision::fp32:
                e = cnn_detail::launch_maxpool(s.out_channels, n, side,
                                               static_cast<const float*>(act[cur]),
                                               static_cast<float*>(act[1 - cur]), stream);
                break;
            case CnnPrecision::fp16:
                e = cnn_detail::launch_maxpool(s.out_channels, n, side,
                                               static_cast<const __half*>(act[cur]),
                                               static_cast<__half*>(act[1 - cur]), stream);
                break;
            case CnnPrecision::int8:
                e = cnn_detail::launch_maxpool(s.out_channels, n, side,
                                               static_cast<const signed char*>(act[cur]),
                                               static_cast<signed char*>(act[1 - cur]), stream);
                break;
            }
            if (!ok(e, "maxpool")) {
                return false;
            }
            mark();
            cur = 1 - cur;
            side /= 2;
        }
    }
    switch (precision) {
    case CnnPrecision::fp32:
        e = cnn_detail::launch_head(n, static_cast<const float*>(act[cur]), fc_weight, fc_bias,
                                    dev_logits, stream);
        break;
    case CnnPrecision::fp16:
        e = cnn_detail::launch_head(n, static_cast<const __half*>(act[cur]), fc_weight, fc_bias,
                                    dev_logits, stream);
        break;
    case CnnPrecision::int8:
        e = cnn_detail::launch_head(n, static_cast<const signed char*>(act[cur]), int8_head_scale,
                                    fc_weight, fc_bias, dev_logits, stream);
        break;
    }
    if (!ok(e, "head")) {
        return false;
    }
    mark();
    if (!ok(cudaMemcpyAsync(host_logits, dev_logits, images * 9 * sizeof(float),
                            cudaMemcpyDeviceToHost, stream),
            "download")) {
        return false;
    }
    mark();
    if (!ok(cudaStreamSynchronize(stream), "CNN kernels")) {
        return false;
    }
    std::memcpy(logits_out, host_logits, images * 9 * sizeof(float));
    if (timing) {
        const auto between = [&](std::size_t a) {
            float ms = 0;
            cudaEventElapsedTime(&ms, events[a], events[a + 1]);
            return ms;
        };
        // Event order: upload, preprocess, then conv, conv, pool per block, head, download.
        CnnTimings t;
        t.upload_ms = between(0);
        t.preprocess_ms = between(1);
        std::size_t at = 2;
        for (int b = 0; b < 4; ++b) {
            t.conv_ms[static_cast<std::size_t>(2 * b)] = between(at++);
            t.conv_ms[static_cast<std::size_t>(2 * b + 1)] = between(at++);
            t.pool_ms[static_cast<std::size_t>(b)] = between(at++);
        }
        t.head_ms = between(at++);
        t.download_ms = between(at);
        timings = t; // the last chunk's
    }
    return true;
}

bool CnnEngine::run(std::span<const WaferMapView> maps, std::span<float> logits) {
    Impl& m = *impl_;
    if (!m.error.empty()) {
        return false;
    }
    for (std::size_t first = 0; first < maps.size(); first += kChunk) {
        const auto chunk = maps.subspan(first, std::min(kChunk, maps.size() - first));
        const auto images = static_cast<unsigned>(chunk.size());
        const std::size_t desc_bytes = align16(chunk.size() * sizeof(MapDesc));
        std::size_t bin_bytes = 0;
        for (const auto& map : chunk) {
            bin_bytes = align16(bin_bytes) + map.bins().size();
        }
        const std::size_t in_bytes = desc_bytes + align16(bin_bytes);
        // The largest activation: 32 channels x 64 x 64 per image (block 1).
        const std::size_t act_bytes = std::size_t{32} * cnn::kSide * cnn::kSide * images * m.elem();
        void* host_in = m.host_in;
        void* dev_in = m.dev_in;
        std::size_t act_cap = m.act_cap;
        if (!m.grow(host_in, m.host_in_cap, in_bytes, true) ||
            !m.grow(dev_in, m.dev_in_cap, in_bytes, false)) {
            return false;
        }
        m.host_in = static_cast<std::uint8_t*>(host_in);
        m.dev_in = static_cast<std::uint8_t*>(dev_in);
        if (act_bytes > act_cap) {
            std::size_t cap1 = act_cap;
            if (!m.grow(m.act[0], act_cap, act_bytes, false) ||
                !m.grow(m.act[1], cap1, act_bytes, false)) {
                return false;
            }
            m.act_cap = act_cap;
        }
        if (images > m.logits_cap) {
            cudaFree(m.dev_logits);
            cudaFreeHost(m.host_logits);
            if (!m.ok(
                    cudaMalloc(reinterpret_cast<void**>(&m.dev_logits), images * 9 * sizeof(float)),
                    "logits") ||
                !m.ok(cudaMallocHost(reinterpret_cast<void**>(&m.host_logits),
                                     images * 9 * sizeof(float)),
                      "logits")) {
                return false;
            }
            m.logits_cap = images;
        }
        auto* descs = reinterpret_cast<MapDesc*>(m.host_in);
        std::uint8_t* bins = m.host_in + desc_bytes;
        std::size_t at = 0;
        for (std::size_t i = 0; i < chunk.size(); ++i) {
            at = align16(at);
            const auto b = chunk[i].bins();
            descs[i] = MapDesc{static_cast<std::uint32_t>(at), static_cast<std::uint32_t>(b.size()),
                               0, static_cast<std::uint16_t>(chunk[i].rows()),
                               static_cast<std::uint16_t>(chunk[i].cols())};
            std::memcpy(bins + at, b.data(), b.size());
            at += b.size();
        }
        if (!m.forward(images, in_bytes, desc_bytes, logits.data() + first * 9)) {
            return false;
        }
    }
    return true;
}

} // namespace waferedge::gpu
