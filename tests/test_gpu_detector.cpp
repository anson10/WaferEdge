// The GPU detectors against the CPU ones on the same maps (label gpu: runs locally with
// `ctest --preset cuda -L gpu`, not in CI). rules-gpu must agree with rules map for map; the
// fp32 CNN with the CPU reference except where two logits are within float rounding.
#include "support/synth.hpp"
#include "waferedge/backend.hpp"
#include "waferedge/map_file.hpp"
#include "waferedge/pipeline/detector.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace waferedge;
using namespace waferedge::pipeline;

namespace {

void require_gpu() {
    if (!backend::Cuda::available()) {
        SKIP("no CUDA device with compute capability 8.6 or newer");
    }
}

std::vector<Detection> run(Detector& d, const std::vector<WaferMapView>& maps) {
    std::vector<Detection> out(maps.size());
    REQUIRE(d.detect(maps, out));
    return out;
}

} // namespace

TEST_CASE("rules-gpu equals rules on mixed shapes and the WaferLens fixture", "[detector]") {
    require_gpu();
    const auto rules = RuleClassifier::load(WAFEREDGE_SOURCE_DIR "/config/rules.txt");
    REQUIRE(rules.has_value());
    std::vector<WaferMap> owned;
    std::uint32_t seed = 1;
    for (const auto& [rows, cols] :
         std::vector<std::pair<int, int>>{{24, 24}, {25, 27}, {30, 34}, {40, 40}, {64, 64}}) {
        for (const unsigned per_mille : {0U, 20U, 150U, 500U, 950U}) {
            owned.push_back(synth::random_map(rows, cols, per_mille, seed++));
        }
    }
    auto blob = synth::disc(40, 40);
    synth::paint(blob, [](int r, int c) { return synth::rho2(r, c, 40, 40) < 0.03; });
    owned.push_back(blob);
    std::vector<WaferMapView> maps(owned.begin(), owned.end());
    const auto fixture = MapSet::load(WAFEREDGE_TEST_DATA "/waferlens_sample.wmap");
    REQUIRE(fixture.has_value());
    for (const auto& r : fixture->records()) {
        maps.push_back(r.map);
    }
    REQUIRE(maps.size() <= kMaxBatch);

    DetectorResources res;
    res.rules = &*rules;
    auto gpu = (*detector_factory("rules-gpu", res))();
    RulesDetector cpu(*rules);
    const auto a = run(cpu, maps);
    const auto b = run(*gpu, maps);
    for (std::size_t i = 0; i < maps.size(); ++i) {
        INFO("map " << i);
        CHECK(a[i].pattern == b[i].pattern);
        CHECK(a[i].rule == b[i].rule);
    }
    // A second batch of another size reuses the engine's buffers.
    const std::vector<WaferMapView> few(maps.begin(), maps.begin() + 3);
    const auto c = run(*gpu, few);
    CHECK(c[0].pattern == a[0].pattern);
}

TEST_CASE("cnn-fp32 agrees with cnn-cpu on random maps", "[detector]") {
    require_gpu();
    // A random model with FabEye's shapes (as test_gpu_cnn.cpp): the plumbing is what is
    // tested here, the engine's numerics are tested there.
    std::mt19937 rng(5);
    const auto uniform = [&](float scale) {
        return (static_cast<float>(rng() % 20001) / 10000.0F - 1.0F) * scale;
    };
    cnn::Model model;
    for (std::size_t l = 0; l < cnn::kConvLayers; ++l) {
        auto& c = model.conv[l];
        c.in = cnn::kConvShapes[l][0];
        c.out = cnn::kConvShapes[l][1];
        const float scale = std::sqrt(6.0F / static_cast<float>(c.in * 9));
        c.weight.resize(static_cast<std::size_t>(c.out) * static_cast<std::size_t>(c.in) * 9);
        for (auto& w : c.weight) {
            w = uniform(scale);
        }
        c.bias.resize(static_cast<std::size_t>(c.out));
        for (auto& b : c.bias) {
            b = uniform(0.1F);
        }
    }
    model.fc_weight.resize(cnn::kClasses * 256);
    for (auto& w : model.fc_weight) {
        w = uniform(0.2F);
    }
    model.fc_bias.resize(cnn::kClasses);
    for (auto& b : model.fc_bias) {
        b = uniform(0.1F);
    }

    std::vector<WaferMap> owned;
    for (std::uint32_t s = 1; s <= 48; ++s) {
        owned.push_back(synth::random_map(30 + static_cast<int>(s % 11), 32, s * 17 % 600, s));
    }
    const std::vector<WaferMapView> maps(owned.begin(), owned.end());
    DetectorResources res;
    res.model = &model;
    auto gpu = (*detector_factory("cnn-fp32", res))();
    CHECK(gpu->name() == "cnn-fp32");
    CnnCpuDetector cpu(model);
    const auto a = run(cpu, maps);
    const auto b = run(*gpu, maps);
    std::size_t same = 0;
    for (std::size_t i = 0; i < maps.size(); ++i) {
        same += a[i].pattern == b[i].pattern ? 1U : 0U;
    }
    CHECK(same + 1 >= maps.size()); // at most one near-tie may flip
}
