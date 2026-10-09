#pragma once

// FabEye's CNN: the architecture's constants and the weights in memory. C++20 and free of
// C++23 library types, so CUDA code can include it (ADR-0001); loading lives in cnn.hpp.
#include <array>
#include <cstddef>
#include <cstdint>
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

} // namespace waferedge::cnn
