#pragma once

// FabEye's CNN on the GPU (phase 2b, docs/inference.md): every convolution as an implicit
// GEMM (patches computed while a tile is staged, never stored), bias + ReLU fused into the
// GEMM's epilogue, preprocessing on the GPU from the raw bins (cv2-exact nearest resize and
// one-hot), activations channel-major with the batch inside ([channel][image][y][x]) so a
// GEMM's output is the next layer's input as it is.
//
// C++20 and free of CUDA headers (ADR-0001).
#include "waferedge/cnn_model.hpp"
#include "waferedge/wafer_map.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace waferedge::gpu {

enum class CnnPrecision : std::uint8_t {
    fp32, // fp32 everywhere: equal to the CPU reference within float rounding
    fp16, // tensor cores (WMMA): fp16 weights and activations, fp32 accumulation
};

// Milliseconds per stage of the last run(), when set_timing(true) (CUDA events between the
// kernels; off by default, they cost ~15 us each on WSL2).
struct CnnTimings {
    float upload_ms = 0;
    float preprocess_ms = 0;
    std::array<float, cnn::kConvLayers> conv_ms{}; // including a separate bias + ReLU when unfused
    std::array<float, 4> pool_ms{};
    float head_ms = 0; // average pool + linear
    float download_ms = 0;
};

class CnnEngine {
public:
    CnnEngine(const cnn::Model& model, CnnPrecision precision);
    ~CnnEngine();
    CnnEngine(const CnnEngine&) = delete;
    CnnEngine& operator=(const CnnEngine&) = delete;
    CnnEngine(CnnEngine&&) noexcept;
    CnnEngine& operator=(CnnEngine&&) noexcept;

    // Logits of every map into logits[i * 9 .. i * 9 + 9). Buffers grow to the largest batch
    // and are reused. Returns false on a CUDA error; error() says which.
    bool run(std::span<const WaferMapView> maps, std::span<float> logits);

    [[nodiscard]] const std::string& error() const noexcept;
    [[nodiscard]] CnnPrecision precision() const noexcept;
    // Bias + ReLU in the GEMM's epilogue (default) or as a separate kernel per conv: the
    // before / after of fusion.
    void set_fused(bool on) noexcept;
    void set_timing(bool on) noexcept;
    [[nodiscard]] CnnTimings last_timings() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace waferedge::gpu
