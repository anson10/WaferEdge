// The GPU detectors (cuda preset only): one engine call per batch.
#include "detector_gpu.hpp"

#include "waferedge/gpu_signatures.hpp"
#include "waferedge/randomness.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace waferedge::pipeline {

namespace {

class RulesGpuDetector final : public Detector {
public:
    explicit RulesGpuDetector(const RuleClassifier& rules) : rules_(rules) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "rules-gpu"; }

    bool detect(std::span<const WaferMapView> maps, std::span<Detection> out) override {
        const std::size_t n = maps.size();
        for (std::size_t i = 0; i < n; ++i) {
            geometries_[i] = &geometry_.get(maps[i].rows(), maps[i].cols());
        }
        const auto features = std::span(features_).first(n);
        const auto lines = std::span(lines_).first(n);
        const auto clusters = std::span(clusters_).first(n);
        if (!engine_.run(maps, std::span(geometries_).first(n), features, lines, clusters)) {
            return false;
        }
        for (std::size_t i = 0; i < n; ++i) {
            // The join count stays on the CPU: one pass over the map, no GPU kernel for it.
            const double z = join_count_z(join_count(maps[i]));
            const Cluster* largest = clusters[i].clusters > 0 ? &clusters[i].largest : nullptr;
            const Decision d = rules_.classify(
                SignalExtractor::compose(maps[i], features[i], z, largest, lines[i]));
            out[i] = {.pattern = d.pattern, .rule = d.rule, .second = Pattern::unknown};
        }
        return true;
    }

private:
    const RuleClassifier& rules_;
    gpu::SignatureEngine engine_;
    GeometryCache geometry_; // references stay valid: the engine caches tables by Geometry id
    std::array<const Geometry*, kMaxBatch> geometries_{};
    std::array<Features, kMaxBatch> features_{};
    std::array<HoughLine, kMaxBatch> lines_{};
    std::array<ClusterSummary, kMaxBatch> clusters_{};
};

class CnnGpuDetector final : public Detector {
public:
    CnnGpuDetector(const cnn::Model& model, gpu::CnnPrecision precision,
                   const gpu::ActivationMax* activation_max)
        : max_(activation_max != nullptr ? std::optional(*activation_max) : std::nullopt),
          engine_(model, precision, max_ ? &*max_ : nullptr),
          name_(precision == gpu::CnnPrecision::int8   ? "cnn-int8"
                : precision == gpu::CnnPrecision::fp16 ? "cnn-fp16"
                                                       : "cnn-fp32"),
          logits_(kMaxBatch * cnn::kClasses) {}
    [[nodiscard]] std::string_view name() const noexcept override { return name_; }

    bool detect(std::span<const WaferMapView> maps, std::span<Detection> out) override {
        const auto logits = std::span(logits_).first(maps.size() * cnn::kClasses);
        if (!engine_.run(maps, logits)) {
            return false;
        }
        for (std::size_t i = 0; i < maps.size(); ++i) {
            out[i] = {.pattern = cnn::predicted(logits.subspan(i * cnn::kClasses, cnn::kClasses)),
                      .rule = -1,
                      .second = Pattern::unknown};
        }
        return true;
    }

private:
    std::optional<gpu::ActivationMax> max_; // before engine_: it reads the copy
    gpu::CnnEngine engine_;
    std::string name_;
    std::vector<float> logits_;
};

} // namespace

std::unique_ptr<Detector> make_rules_gpu(const RuleClassifier& rules) {
    return std::make_unique<RulesGpuDetector>(rules);
}

std::unique_ptr<Detector> make_cnn_gpu(const cnn::Model& model, gpu::CnnPrecision precision,
                                       const gpu::ActivationMax* activation_max) {
    return std::make_unique<CnnGpuDetector>(model, precision, activation_max);
}

} // namespace waferedge::pipeline
