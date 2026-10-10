// Detectors and the batching worker: the factory's names and errors, rules == signals by hand,
// the CPU CNN on a model with a known answer, a pair recording agreement, batches formed by
// size and by deadline, and a failing detector. GPU detectors are tested locally
// (test_gpu_detector.cpp, label gpu).
#include "support/synth.hpp"
#include "waferedge/clusters.hpp"
#include "waferedge/hough.hpp"
#include "waferedge/pipeline/analyzer.hpp"
#include "waferedge/pipeline/detector.hpp"
#include "waferedge/randomness.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace waferedge;
using namespace waferedge::pipeline;
using namespace std::chrono_literals;

namespace {

const RuleClassifier& rules() {
    static const RuleClassifier c = *RuleClassifier::load(WAFEREDGE_SOURCE_DIR "/config/rules.txt");
    return c;
}

WaferMap blob() {
    auto map = synth::disc(40, 40);
    synth::paint(map, [](int r, int c) { return synth::rho2(r, c, 40, 40) < 0.03; });
    return map;
}

// FabEye's architecture with zero weights: every activation is 0, so the logits are the
// final bias and the CNN always answers `answer`.
cnn::Model constant_model(Pattern answer) {
    cnn::Model m;
    for (int l = 0; l < cnn::kConvLayers; ++l) {
        auto& c = m.conv[static_cast<std::size_t>(l)];
        c.in = cnn::kConvShapes[static_cast<std::size_t>(l)][0];
        c.out = cnn::kConvShapes[static_cast<std::size_t>(l)][1];
        c.weight.assign(static_cast<std::size_t>(c.out) * static_cast<std::size_t>(c.in) * 9, 0.F);
        c.bias.assign(static_cast<std::size_t>(c.out), 0.F);
    }
    m.fc_weight.assign(static_cast<std::size_t>(cnn::kClasses) * 256, 0.F);
    m.fc_bias.assign(cnn::kClasses, 0.F);
    m.fc_bias[static_cast<std::size_t>(answer)] = 1.F;
    return m;
}

// Records the size of every batch it sees; answers center for all, or fails on request.
class CountingDetector final : public Detector {
public:
    std::vector<std::size_t>* sizes = nullptr;
    bool fail = false;
    [[nodiscard]] std::string_view name() const noexcept override { return "counting"; }
    bool detect(std::span<const WaferMapView> maps, std::span<Detection> out) override {
        sizes->push_back(maps.size());
        for (auto& d : out) {
            d = {.pattern = Pattern::center, .rule = -1, .second = Pattern::unknown};
        }
        return !fail;
    }
};

void submit(Analyzer& a, std::uint64_t seq, const WaferMap& map) {
    WaferSlot* slot = a.begin_submit();
    REQUIRE(slot != nullptr);
    slot->seq = seq;
    slot->lot = LotId::from("LOT").value_or(LotId{});
    slot->wafer = seq;
    slot->received = Clock::now();
    slot->rows = map.rows();
    slot->cols = map.cols();
    std::memcpy(slot->bins.data(), map.view().bins().data(), map.view().bins().size());
    a.commit_submit();
}

std::vector<Verdict> take(Analyzer& a) {
    std::vector<Verdict> out;
    while (const Verdict* v = a.front_verdict()) {
        out.push_back(*v);
        a.pop_verdict();
    }
    return out;
}

} // namespace

TEST_CASE("signals: compose over the CPU detectors equals extract", "[detector]") {
    SignalExtractor extractor;
    ClusterFinder clusters;
    HoughTransform hough;
    GeometryCache geometry;
    for (std::uint32_t seed = 1; seed <= 40; ++seed) {
        const WaferMap map =
            synth::random_map(30 + static_cast<int>(seed % 11), 34, seed * 23 % 400, seed);
        const auto f = compute_features(map, geometry.get(map.rows(), map.cols()));
        clusters.run(map);
        const Signals composed = SignalExtractor::compose(
            map, f, join_count_z(join_count(map)), f.fails > 0 ? clusters.largest() : nullptr,
            f.fails > 0 ? hough.run(map) : HoughLine{});
        REQUIRE(composed == extractor.extract(map));
    }
}

TEST_CASE("detector factory: names, pairs and what is missing", "[detector]") {
    DetectorResources with_rules;
    with_rules.rules = &rules();
    auto f = detector_factory("rules", with_rules);
    REQUIRE(f.has_value());
    CHECK((*f)()->name() == "rules");

    auto pair = detector_factory("rules+rules", with_rules);
    REQUIRE(pair.has_value());
    CHECK((*pair)()->name() == "rules+rules");

    CHECK_FALSE(detector_factory("nonsense", with_rules).has_value());
    const auto no_model = detector_factory("cnn-cpu", with_rules);
    REQUIRE_FALSE(no_model.has_value());
    CHECK(no_model.error().find("model") != std::string::npos);
    CHECK_FALSE(detector_factory("rules", DetectorResources{}).has_value());
#if !WAFEREDGE_HAS_CUDA
    const auto gpu = detector_factory("cnn-fp16", with_rules);
    REQUIRE_FALSE(gpu.has_value());
    CHECK(gpu.error().find("cuda build") != std::string::npos);
#endif
}

TEST_CASE("detectors: rules, the CPU CNN, and a pair", "[detector]") {
    const WaferMap center = blob();
    const WaferMap clean = synth::disc(40, 40);
    const std::vector<WaferMapView> maps{center.view(), clean.view()};
    std::vector<Detection> out(2);

    RulesDetector r(rules());
    REQUIRE(r.detect(maps, out));
    CHECK(out[0].pattern == Pattern::center);
    CHECK(out[0].rule >= 0);
    CHECK(out[1].pattern == Pattern::none);

    const cnn::Model model = constant_model(Pattern::scratch);
    CnnCpuDetector c(model);
    REQUIRE(c.detect(maps, out));
    CHECK(out[0].pattern == Pattern::scratch);
    CHECK(out[1].pattern == Pattern::scratch);

    PairDetector both(std::make_unique<RulesDetector>(rules()),
                      std::make_unique<CnnCpuDetector>(model));
    CHECK(both.name() == "rules+cnn-cpu");
    REQUIRE(both.detect(maps, out));
    CHECK(out[0].pattern == Pattern::center); // the first decides
    CHECK(out[0].second == Pattern::scratch); // the second is recorded
    CHECK(out[1].second == Pattern::scratch);
}

TEST_CASE("batching: full batches, then the rest when flushed", "[detector]") {
    std::vector<std::size_t> sizes;
    auto d = std::make_unique<CountingDetector>();
    d->sizes = &sizes;
    Analyzer a(std::move(d), {.spin = 0us, .cpu = -1, .batch = 4, .deadline = 0us});
    const WaferMap map = synth::disc(30, 30);
    for (std::uint64_t i = 0; i < 10; ++i) {
        submit(a, i, map);
    }
    CHECK(a.poll() == 10);
    CHECK(sizes == std::vector<std::size_t>{4, 4, 2});
    const auto v = take(a);
    REQUIRE(v.size() == 10);
    for (std::uint64_t i = 0; i < 10; ++i) {
        CHECK(v[i].seq == i);
        CHECK(v[i].pattern == Pattern::center);
    }
    CHECK(a.batches() == 3);
}

TEST_CASE("batching on the worker thread: a partial batch waits for its deadline", "[detector]") {
    std::vector<std::size_t> sizes; // written by the worker, read after stop()
    auto d = std::make_unique<CountingDetector>();
    d->sizes = &sizes;
    Analyzer a(std::move(d), {.spin = 20us, .cpu = -1, .batch = 8, .deadline = 20ms});
    a.start();
    const WaferMap map = synth::disc(30, 30);
    for (std::uint64_t i = 0; i < 3; ++i) {
        submit(a, i, map);
    }
    // Three wafers: not a full batch, so nothing until the oldest has waited 20 ms.
    std::vector<Verdict> got;
    const auto until = Clock::now() + 5s;
    while (got.size() < 3 && Clock::now() < until) {
        for (const auto& v : take(a)) {
            got.push_back(v);
        }
        std::this_thread::sleep_for(1ms);
    }
    REQUIRE(got.size() == 3);
    // The deadline counts from the oldest wafer; the other two arrived just after it and go
    // in the same batch.
    CHECK(got[0].analysed - got[0].received >= 20ms);
    CHECK(got[1].analysed == got[0].analysed);
    CHECK(got[2].analysed == got[0].analysed);
    // Eight at once: a full batch goes without waiting.
    for (std::uint64_t i = 3; i < 11; ++i) {
        submit(a, i, map);
    }
    got.clear();
    while (got.size() < 8 && Clock::now() < until) {
        for (const auto& v : take(a)) {
            got.push_back(v);
        }
    }
    REQUIRE(got.size() == 8);
    a.stop();
    CHECK(sizes == std::vector<std::size_t>{3, 8});
    CHECK(got.front().analysed - got.front().received < 20ms);
}

TEST_CASE("a failing detector: verdicts say unknown, the failure is counted", "[detector]") {
    std::vector<std::size_t> sizes;
    auto d = std::make_unique<CountingDetector>();
    d->sizes = &sizes;
    d->fail = true;
    Analyzer a(std::move(d), {.spin = 0us, .cpu = -1, .batch = 2, .deadline = 0us});
    const WaferMap map = synth::disc(30, 30);
    submit(a, 0, map);
    submit(a, 1, map);
    CHECK(a.poll() == 2);
    for (const auto& v : take(a)) {
        CHECK(v.pattern == Pattern::unknown);
    }
    CHECK(a.failures() == 1);
}
