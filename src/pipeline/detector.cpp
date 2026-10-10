#include "waferedge/pipeline/detector.hpp"

#include "detector_gpu.hpp"

#include <format>

namespace waferedge::pipeline {

bool RulesDetector::detect(std::span<const WaferMapView> maps, std::span<Detection> out) {
    for (std::size_t i = 0; i < maps.size(); ++i) {
        const Decision d = rules_.classify(extractor_.extract(maps[i]));
        out[i] = {.pattern = d.pattern, .rule = d.rule, .second = Pattern::unknown};
    }
    return true;
}

bool CnnCpuDetector::detect(std::span<const WaferMapView> maps, std::span<Detection> out) {
    for (std::size_t i = 0; i < maps.size(); ++i) {
        cnn::preprocess(maps[i], input_);
        forward_.run(model_, input_, logits_);
        out[i] = {.pattern = cnn::predicted(logits_), .rule = -1, .second = Pattern::unknown};
    }
    return true;
}

PairDetector::PairDetector(std::unique_ptr<Detector> a, std::unique_ptr<Detector> b)
    : a_(std::move(a)), b_(std::move(b)), name_(std::format("{}+{}", a_->name(), b_->name())) {}

bool PairDetector::detect(std::span<const WaferMapView> maps, std::span<Detection> out) {
    if (!a_->detect(maps, out)) {
        return false;
    }
    const auto second = std::span(scratch_).first(maps.size());
    if (!b_->detect(maps, second)) {
        return false;
    }
    for (std::size_t i = 0; i < maps.size(); ++i) {
        out[i].second = second[i].pattern;
    }
    return true;
}

namespace {

std::expected<DetectorFactory, std::string> single(std::string_view name,
                                                   const DetectorResources& r) {
    const auto need_rules = [&]() -> std::expected<void, std::string> {
        if (r.rules == nullptr) {
            return std::unexpected(std::format("{} needs the rules", name));
        }
        return {};
    };
    const auto need_model = [&]() -> std::expected<void, std::string> {
        if (r.model == nullptr) {
            return std::unexpected(
                std::format("{} needs FabEye's model (tools/export_cnn.py --out data)", name));
        }
        return {};
    };
    if (name == "rules") {
        if (auto ok = need_rules(); !ok) {
            return std::unexpected(ok.error());
        }
        const RuleClassifier* rules = r.rules;
        return DetectorFactory([rules] { return std::make_unique<RulesDetector>(*rules); });
    }
    if (name == "cnn-cpu") {
        if (auto ok = need_model(); !ok) {
            return std::unexpected(ok.error());
        }
        const cnn::Model* model = r.model;
        return DetectorFactory([model] { return std::make_unique<CnnCpuDetector>(*model); });
    }
    const bool gpu_name =
        name == "rules-gpu" || name == "cnn-fp32" || name == "cnn-fp16" || name == "cnn-int8";
    if (!gpu_name) {
        return std::unexpected(std::format("unknown detector '{}'", name));
    }
#if WAFEREDGE_HAS_CUDA
    if (name == "rules-gpu") {
        if (auto ok = need_rules(); !ok) {
            return std::unexpected(ok.error());
        }
        const RuleClassifier* rules = r.rules;
        return DetectorFactory([rules] { return make_rules_gpu(*rules); });
    }
    if (auto ok = need_model(); !ok) {
        return std::unexpected(ok.error());
    }
    const cnn::Model* model = r.model;
    if (name == "cnn-int8") {
        if (!r.int8_max) {
            return std::unexpected(
                "cnn-int8 needs the activation scales (waferedge-cnn calibrate, --scales)");
        }
        const gpu::ActivationMax max = *r.int8_max;
        return DetectorFactory(
            [model, max] { return make_cnn_gpu(*model, gpu::CnnPrecision::int8, &max); });
    }
    const auto precision = name == "cnn-fp16" ? gpu::CnnPrecision::fp16 : gpu::CnnPrecision::fp32;
    return DetectorFactory([model, precision] { return make_cnn_gpu(*model, precision, nullptr); });
#else
    return std::unexpected(std::format("{} needs the cuda build", name));
#endif
}

} // namespace

std::expected<DetectorFactory, std::string> detector_factory(std::string_view spec,
                                                             const DetectorResources& resources) {
    const auto plus = spec.find('+');
    if (plus == std::string_view::npos) {
        return single(spec, resources);
    }
    auto a = single(spec.substr(0, plus), resources);
    if (!a) {
        return a;
    }
    auto b = single(spec.substr(plus + 1), resources);
    if (!b) {
        return b;
    }
    return DetectorFactory([fa = std::move(*a), fb = std::move(*b)] {
        return std::make_unique<PairDetector>(fa(), fb());
    });
}

} // namespace waferedge::pipeline
