#pragma once

#include "waferedge/classifier.hpp"
#include "waferedge/cnn.hpp"
#include "waferedge/gpu_cnn.hpp"
#include "waferedge/wafer_map.hpp"

#include <array>
#include <cstddef>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// What an analytics worker runs on a batch of wafer maps (ADR-0016). One interface for every
// detector the experiments compare:
//
//   rules       signals on the CPU (AVX2 features), then the rule table      per map
//   cnn-cpu     FabEye's CNN, the plain CPU reference forward pass           per map
//   rules-gpu   signals from the GPU signature engine, the same rules        one launch per batch
//   cnn-fp32 | cnn-fp16 | cnn-int8   the CNN on WaferEdge's GPU engine       one launch per batch
//   A+B         both: A decides, B's pattern is recorded next to it (agreement)
//
// A detector is used by one worker thread; each worker builds its own from a factory.
namespace waferedge::pipeline {

// The largest batch a worker hands a detector: the ring's capacity.
inline constexpr std::size_t kMaxBatch = 256;

struct Detection {
    Pattern pattern = Pattern::none;
    int rule = -1;                     // the rule that fired (rule detectors), else -1
    Pattern second = Pattern::unknown; // a paired detector's pattern; unknown if none
};

class Detector {
public:
    Detector() = default;
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) = delete;
    Detector& operator=(Detector&&) = delete;
    virtual ~Detector() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    // Classifies maps[i] into out[i]; out.size() == maps.size() <= kMaxBatch. False on a
    // failure (a CUDA error); out is unspecified then.
    virtual bool detect(std::span<const WaferMapView> maps, std::span<Detection> out) = 0;
};

using DetectorFactory = std::function<std::unique_ptr<Detector>()>;

class RulesDetector final : public Detector {
public:
    explicit RulesDetector(const RuleClassifier& rules) : rules_(rules) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "rules"; }
    bool detect(std::span<const WaferMapView> maps, std::span<Detection> out) override;

private:
    const RuleClassifier& rules_;
    SignalExtractor extractor_;
};

class CnnCpuDetector final : public Detector {
public:
    explicit CnnCpuDetector(const cnn::Model& model) : model_(model), input_(cnn::kInputSize) {}
    [[nodiscard]] std::string_view name() const noexcept override { return "cnn-cpu"; }
    bool detect(std::span<const WaferMapView> maps, std::span<Detection> out) override;

private:
    const cnn::Model& model_;
    cnn::ReferenceForward forward_;
    std::vector<float> input_;
    std::array<float, cnn::kClasses> logits_{};
};

// A decides; B runs on the same batch and its pattern goes into Detection::second.
class PairDetector final : public Detector {
public:
    PairDetector(std::unique_ptr<Detector> a, std::unique_ptr<Detector> b);
    [[nodiscard]] std::string_view name() const noexcept override { return name_; }
    bool detect(std::span<const WaferMapView> maps, std::span<Detection> out) override;

private:
    std::unique_ptr<Detector> a_;
    std::unique_ptr<Detector> b_;
    std::string name_;
    std::array<Detection, kMaxBatch> scratch_{};
};

// What the factory may need; the referenced objects must outlive every detector it makes.
struct DetectorResources {
    const RuleClassifier* rules = nullptr;
    const cnn::Model* model = nullptr;          // the cnn-* detectors
    std::optional<gpu::ActivationMax> int8_max; // cnn-int8 (waferedge-cnn calibrate)
};

// A factory for "rules", "cnn-cpu", "rules-gpu", "cnn-fp32", "cnn-fp16", "cnn-int8", or two of
// them joined by '+'. An error names what is missing (a model, the calibration, a CUDA build).
[[nodiscard]] std::expected<DetectorFactory, std::string>
detector_factory(std::string_view spec, const DetectorResources& resources);

} // namespace waferedge::pipeline
