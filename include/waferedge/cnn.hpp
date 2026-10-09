#pragma once

// FabEye's CNN in WaferEdge (phase 2b): the model file, FabEye's preprocessing, and a plain
// CPU forward pass that is the reference for every faster backend (docs/inference.md).
//
// Architecture (FabEye's WaferCNN, BatchNorm folded into the convolutions at export):
//   input 3 x 64 x 64 (one-hot: off wafer, good, fail)
//   4 blocks of [conv 3x3 pad 1 -> ReLU -> conv 3x3 pad 1 -> ReLU -> maxpool 2x2]
//   channels 3 -> 32 -> 64 -> 128 -> 256, spatial 64 -> 32 -> 16 -> 8 -> 4
//   global average pool over 4 x 4 -> 256 -> linear -> 9 logits (Pattern order)
// About 211M multiply-adds per map.
#include "waferedge/wafer_map.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace waferedge::cnn {

inline constexpr int kSide = 64;
inline constexpr int kInputChannels = 3;
inline constexpr std::size_t kInputSize = std::size_t{kInputChannels} * kSide * kSide;
inline constexpr int kClasses = 9;
inline constexpr int kConvLayers = 8;
// (in, out) channels of the 8 convolutions.
inline constexpr std::array<std::array<int, 2>, kConvLayers> kConvShapes = {
    {{3, 32}, {32, 32}, {32, 64}, {64, 64}, {64, 128}, {128, 128}, {128, 256}, {256, 256}}};

struct Conv {
    int in = 0;
    int out = 0;
    std::vector<float> weight; // [out][in][3][3]
    std::vector<float> bias;   // [out]
};

struct Model {
    std::array<Conv, kConvLayers> conv;
    std::vector<float> fc_weight;               // [kClasses][256]
    std::vector<float> fc_bias;                 // [kClasses]
    std::array<std::uint8_t, 32> onnx_sha256{}; // the FabEye ONNX model these weights came from
};

// Reads a .wcnn file (tools/export_cnn.py): checks the magic, the version, the checksum and
// that every tensor has the shape the architecture above needs.
[[nodiscard]] std::expected<Model, std::string> load_model(const std::filesystem::path& path);
[[nodiscard]] std::expected<Model, std::string> parse_model(std::span<const std::uint8_t> image);

// cv2.resize(..., INTER_NEAREST) source index for output position i of 64, from a source of
// length n: floor(i * (1 / (64 / n))) in double precision, clamped to n - 1. The plain integer
// i * n / 64 differs from OpenCV at some lengths (186, 198, 210, ...), where 1 / (64 / n)
// rounds just below n / 64.
[[nodiscard]] int nearest_source(int i, int n) noexcept;

// FabEye's preprocessing: bins >= 2 become fail, the grid is resized to 64 x 64 with OpenCV's
// nearest rule, and one-hot encoded into out[3][64][64] (channel 0 off wafer, 1 good, 2 fail).
// Precondition: out.size() == kInputSize.
void preprocess(WaferMapView map, std::span<float> out) noexcept;

// The plain forward pass, one map at a time, readable on purpose: direct convolutions with
// the loops ordered so the innermost runs along a row (the compiler vectorises it). Owns its
// activation buffers; one per thread.
class ReferenceForward {
public:
    // Logits of one preprocessed map (input: kInputSize floats) into logits (kClasses).
    void run(const Model& model, std::span<const float> input, std::span<float> logits);

private:
    std::vector<float> a_;
    std::vector<float> b_;
};

// The building blocks, exposed for tests. Activations are [channels][height][width].
// out = ReLU(conv3x3(in, padding 1) + bias), out_channels x h x w.
void conv3x3_relu(std::span<const float> in, int in_channels, int h, int w,
                  std::span<const float> weight, std::span<const float> bias, int out_channels,
                  std::span<float> out) noexcept;
// 2 x 2 max pooling, stride 2: channels x h x w -> channels x h/2 x w/2.
void maxpool2(std::span<const float> in, int channels, int h, int w, std::span<float> out) noexcept;

// Probabilities from logits (numerically stable: the largest logit is subtracted first).
[[nodiscard]] std::array<float, kClasses> softmax(std::span<const float> logits) noexcept;
[[nodiscard]] Pattern predicted(std::span<const float> logits) noexcept;

} // namespace waferedge::cnn
