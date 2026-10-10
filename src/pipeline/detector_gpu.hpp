#pragma once

// The GPU detectors, built with the cuda preset only (detector_gpu.cpp); the factory in
// detector.cpp reaches them through these two functions.
#include "waferedge/pipeline/detector.hpp"

#include <memory>

namespace waferedge::pipeline {

#if WAFEREDGE_HAS_CUDA
// Signals from the GPU signature engine (features, Hough line, clusters in one upload), the
// join-count z on the CPU, then the same rules: equal to RulesDetector map for map.
[[nodiscard]] std::unique_ptr<Detector> make_rules_gpu(const RuleClassifier& rules);
// FabEye's CNN on the GPU engine; `activation_max` only for int8 (copied).
[[nodiscard]] std::unique_ptr<Detector> make_cnn_gpu(const cnn::Model& model,
                                                     gpu::CnnPrecision precision,
                                                     const gpu::ActivationMax* activation_max);
#endif

} // namespace waferedge::pipeline
