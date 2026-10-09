// The GPU CNN engine against the CPU reference forward pass. A random model (no FabEye data
// needed) for the fp32 equality, the preprocessing and the batch handling; labelled gpu, runs
// locally (ctest --preset cuda), not in CI.
#include "support/synth.hpp"
#include "waferedge/backend.hpp"
#include "waferedge/cnn.hpp"
#include "waferedge/gpu_cnn.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace waferedge;
using gpu::CnnEngine;
using gpu::CnnPrecision;

namespace {

void require_gpu() {
    if (!backend::Cuda::available()) {
        SKIP("no CUDA device with compute capability 8.6 or newer");
    }
}

// FabEye's shapes with random weights, scaled so activations stay O(1) through 8 layers.
cnn::Model random_model(std::uint32_t seed) {
    std::mt19937 rng(seed);
    auto uniform = [&](float scale) {
        return (static_cast<float>(rng() % 20001) / 10000.0F - 1.0F) * scale;
    };
    cnn::Model m;
    for (std::size_t l = 0; l < cnn::kConvLayers; ++l) {
        auto& c = m.conv[l];
        c.in = cnn::kConvShapes[l][0];
        c.out = cnn::kConvShapes[l][1];
        const float scale = std::sqrt(6.0F / static_cast<float>(c.in * 9)); // He-style
        c.weight.resize(static_cast<std::size_t>(c.out) * static_cast<std::size_t>(c.in) * 9);
        for (auto& w : c.weight) {
            w = uniform(scale);
        }
        c.bias.resize(static_cast<std::size_t>(c.out));
        for (auto& b : c.bias) {
            b = uniform(0.1F);
        }
    }
    m.fc_weight.resize(cnn::kClasses * 256);
    for (auto& w : m.fc_weight) {
        w = uniform(0.2F);
    }
    m.fc_bias.resize(cnn::kClasses);
    for (auto& b : m.fc_bias) {
        b = uniform(0.1F);
    }
    return m;
}

std::vector<float> cpu_logits(const cnn::Model& model, const std::vector<WaferMap>& maps) {
    cnn::ReferenceForward forward;
    std::vector<float> x(cnn::kInputSize);
    std::vector<float> logits(maps.size() * cnn::kClasses);
    for (std::size_t i = 0; i < maps.size(); ++i) {
        cnn::preprocess(maps[i], x);
        forward.run(model, x, std::span(logits).subspan(i * cnn::kClasses, cnn::kClasses));
    }
    return logits;
}

std::vector<float> gpu_logits(CnnEngine& engine, const std::vector<WaferMap>& maps) {
    std::vector<WaferMapView> views(maps.begin(), maps.end());
    std::vector<float> logits(maps.size() * cnn::kClasses);
    REQUIRE(engine.run(views, logits));
    return logits;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, static_cast<double>(std::abs(a[i] - b[i])));
    }
    return d;
}

// Shapes from 1 x 1 to beyond WM-811K's largest, including lengths where OpenCV's nearest
// rule and the plain integer formula pick different pixels (186, 198, 210).
std::vector<WaferMap> mixed_maps() {
    std::vector<WaferMap> maps;
    std::uint32_t seed = 40;
    for (const auto& [rows, cols] :
         {std::pair{1, 1}, std::pair{3, 5}, std::pair{24, 24}, std::pair{25, 27}, std::pair{40, 40},
          std::pair{64, 64}, std::pair{65, 63}, std::pair{186, 198}, std::pair{210, 246},
          std::pair{212, 204}}) {
        for (const unsigned per_mille : {50U, 400U}) {
            maps.push_back(synth::random_map(rows, cols, per_mille, seed++));
        }
    }
    return maps;
}

} // namespace

TEST_CASE("GPU fp32 logits equal the CPU reference, preprocessing included") {
    require_gpu();
    const auto model = random_model(1);
    const auto maps = mixed_maps();
    CnnEngine engine(model, CnnPrecision::fp32);
    const auto want = cpu_logits(model, maps);
    const auto got = gpu_logits(engine, maps);
    CHECK(max_abs_diff(got, want) < 1e-4); // float rounding, different summation order
    for (std::size_t i = 0; i < maps.size(); ++i) {
        INFO("map " << i << " (" << maps[i].rows() << "x" << maps[i].cols() << ")");
        CHECK(cnn::predicted(std::span(got).subspan(i * 9, 9)) ==
              cnn::predicted(std::span(want).subspan(i * 9, 9)));
    }
}

TEST_CASE("GPU fp16 logits stay close to the CPU reference") {
    require_gpu();
    const auto model = random_model(2);
    const auto maps = mixed_maps();
    CnnEngine engine(model, CnnPrecision::fp16);
    const auto want = cpu_logits(model, maps);
    const auto got = gpu_logits(engine, maps);
    // fp16 activations carry ~3 significant digits; the logits here are O(1).
    for (std::size_t i = 0; i < want.size(); ++i) {
        CHECK(std::abs(got[i] - want[i]) <= 0.03 * (1.0 + std::abs(want[i])));
    }
}

TEST_CASE("bias + ReLU fused into the GEMM equals the separate kernel") {
    require_gpu();
    const auto model = random_model(3);
    const auto maps = mixed_maps();
    CnnEngine fp32(model, CnnPrecision::fp32);
    const auto fused = gpu_logits(fp32, maps);
    fp32.set_fused(false);
    // Same float operations in the same order, either way: bit for bit.
    CHECK(gpu_logits(fp32, maps) == fused);
    CnnEngine fp16(model, CnnPrecision::fp16);
    const auto fused16 = gpu_logits(fp16, maps);
    fp16.set_fused(false);
    // Unfused fp16 rounds the convolution to fp16 once more before the bias: close, not equal.
    CHECK(max_abs_diff(gpu_logits(fp16, maps), fused16) < 0.05);
}

TEST_CASE("batches: one map, more than a chunk, and growing buffers give the same logits") {
    require_gpu();
    const auto model = random_model(4);
    CnnEngine engine(model, CnnPrecision::fp32);
    std::vector<WaferMap> maps;
    for (std::uint32_t i = 0; i < 1100; ++i) { // > the engine's 1,024-map chunk
        maps.push_back(synth::random_map(8 + static_cast<int>(i % 5), 9, 150, i));
    }
    const auto all = gpu_logits(engine, maps);
    for (const std::size_t i :
         {std::size_t{0}, std::size_t{1023}, std::size_t{1024}, std::size_t{1099}}) {
        const auto one = gpu_logits(engine, {maps[i]});
        INFO("map " << i);
        CHECK(std::vector<float>(all.begin() + static_cast<std::ptrdiff_t>(i * 9),
                                 all.begin() + static_cast<std::ptrdiff_t>(i * 9 + 9)) == one);
    }
}
