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
    int8, // tensor cores (WMMA): int8 weights (per-channel scales) and activations (per-layer
          // scales from calibration), int32 accumulation; the head runs in float
};

// The largest output of each convolution (after ReLU) on calibration maps: what int8 needs
// to choose each layer's activation scale (max / 127). From CnnEngine::activation_max on the
// fp32 engine, on validation maps, never test.
using ActivationMax = std::array<float, cnn::kConvLayers>;

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
    // int8 needs the calibration maxima; the other precisions ignore them.
    CnnEngine(const cnn::Model& model, CnnPrecision precision,
              const ActivationMax* activation_max = nullptr);
    ~CnnEngine();
    CnnEngine(const CnnEngine&) = delete;
    CnnEngine& operator=(const CnnEngine&) = delete;
    CnnEngine(CnnEngine&&) noexcept;
    CnnEngine& operator=(CnnEngine&&) noexcept;

    // Logits of every map into logits[i * 9 .. i * 9 + 9). Buffers grow to the largest batch
    // and are reused. Returns false on a CUDA error; error() says which.
    bool run(std::span<const WaferMapView> maps, std::span<float> logits);

    // fp32 engine only: runs the maps and records each convolution's largest output.
    bool activation_max(std::span<const WaferMapView> maps, ActivationMax& out);

    [[nodiscard]] const std::string& error() const noexcept;
    [[nodiscard]] CnnPrecision precision() const noexcept;
    // Bias + ReLU in the GEMM's epilogue (default) or as a separate kernel per conv: the
    // before / after of fusion (fp32 and fp16; int8 is always fused, its epilogue requantises).
    void set_fused(bool on) noexcept;
    void set_timing(bool on) noexcept;
    [[nodiscard]] CnnTimings last_timings() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace waferedge::gpu
