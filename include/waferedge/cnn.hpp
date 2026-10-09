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
#include "waferedge/cnn_model.hpp"
#include "waferedge/wafer_map.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace waferedge::cnn {

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

// FabEye's conformal calibration (tools/export_cnn.py -> data/fabeye_conformal.txt): per-class
// thresholds q for alpha 0.1 and 0.05, and the confidence above which a wafer is accepted
// without review. FabEye's own results on unseen lots come along, to compare against.
struct Conformal {
    std::array<std::uint8_t, 32> model_sha256{};
    std::array<float, kClasses> thresholds_10{}; // 90% target coverage
    std::array<float, kClasses> thresholds_05{}; // 95%
    float accept_confidence = 0;
    double reported_coverage_10 = 0;
    double reported_worst_class_10 = 0;
    double reported_coverage_05 = 0;
    double reported_worst_class_05 = 0;
    double reported_accept_rate = 0;
    double reported_accept_error = 0;
};

[[nodiscard]] std::expected<Conformal, std::string>
load_conformal(const std::filesystem::path& path);

// The prediction set as a bitmask (bit k: class k), FabEye's rule: class k is in the set when
// 1 - p_k <= q_k.
[[nodiscard]] std::uint16_t prediction_set(std::span<const float> probabilities,
                                           const std::array<float, kClasses>& thresholds) noexcept;

struct ConformalMetrics {
    double coverage = 0;             // true class in the set
    double worst_class_coverage = 0; // the lowest coverage of any true class present
    double mean_set_size = 0;
    double accept_rate = 0;          // top probability >= the accept confidence
    double error_among_accepted = 0; // top class wrong, among the accepted
};

// Logits of n maps (n * 9) against their true classes (maps with unknown truth are skipped).
[[nodiscard]] ConformalMetrics evaluate_conformal(std::span<const float> logits,
                                                  std::span<const Pattern> truth,
                                                  const std::array<float, kClasses>& thresholds,
                                                  float accept_confidence);

// Probabilities from logits (numerically stable: the largest logit is subtracted first).
[[nodiscard]] std::array<float, kClasses> softmax(std::span<const float> logits) noexcept;
[[nodiscard]] Pattern predicted(std::span<const float> logits) noexcept;

} // namespace waferedge::cnn
