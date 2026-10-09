// FabEye's CNN on the GPU engine, by batch size: maps/s end to end (bins up, preprocessing,
// 8 convolutions, pools, head, logits down), fp32 and fp16, bias + ReLU fused or not.
//
//   build/cuda/bench/bench-cnn --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
//
// Needs data/fabeye_cnn.wcnn (tools/export_cnn.py) and, for int8, data/fabeye_int8_scales.txt
// (waferedge-cnn calibrate). 40 x 40 random maps (10% fails): the network's cost doesn't
// depend on the map, every map is resized to 64 x 64.
#include "support/synth.hpp"
#include "waferedge/backend.hpp"
#include "waferedge/cnn.hpp"
#include "waferedge/gpu_cnn.hpp"
#include "waferedge/machine.hpp"

#include <benchmark/benchmark.h>

#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace waferedge;

const std::optional<cnn::Model>& model() {
    static const std::optional<cnn::Model> m = [] {
        auto loaded = cnn::load_model(WAFEREDGE_SOURCE_DIR "/data/fabeye_cnn.wcnn");
        return loaded ? std::optional<cnn::Model>(std::move(*loaded)) : std::nullopt;
    }();
    return m;
}

// int8 activation maxima from data/fabeye_int8_scales.txt ("layer max" lines).
std::optional<gpu::ActivationMax> scales() {
    std::ifstream in(WAFEREDGE_SOURCE_DIR "/data/fabeye_int8_scales.txt");
    gpu::ActivationMax m{};
    std::size_t found = 0;
    for (std::string line; std::getline(in, line);) {
        std::istringstream f(line);
        std::size_t layer = 0;
        float v = 0;
        if (line[0] != '#' && f >> layer >> v && layer >= 1 && layer <= m.size()) {
            m[layer - 1] = v;
            ++found;
        }
    }
    return found == m.size() ? std::optional(m) : std::nullopt;
}

// The benchmark's precision argument: 32, 16 or 8 bits.
gpu::CnnPrecision precision_of(std::int64_t bits) {
    switch (bits) {
    case 16:
        return gpu::CnnPrecision::fp16;
    case 8:
        return gpu::CnnPrecision::int8;
    default:
        return gpu::CnnPrecision::fp32;
    }
}

void BM_cnn_gpu(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto precision = precision_of(state.range(1));
    const auto int8_scales = scales();
    if (precision == gpu::CnnPrecision::int8 && !int8_scales) {
        state.SkipWithError("needs data/fabeye_int8_scales.txt");
        return;
    }
    if (!backend::Cuda::available() || !model()) {
        state.SkipWithError("needs a CUDA device and data/fabeye_cnn.wcnn");
        return;
    }
    std::vector<WaferMap> maps;
    for (std::size_t i = 0; i < n; ++i) {
        maps.push_back(synth::random_map(40, 40, 100, static_cast<std::uint32_t>(i)));
    }
    const std::vector<WaferMapView> views(maps.begin(), maps.end());
    std::vector<float> logits(n * cnn::kClasses);
    gpu::CnnEngine engine(*model(), precision, int8_scales ? &*int8_scales : nullptr);
    engine.set_fused(state.range(2) != 0);
    engine.set_graphs(state.range(3) != 0);
    if (!engine.run(views, logits)) { // warm-up: buffers (and the graph's recording)
        state.SkipWithError(engine.error().c_str());
        return;
    }
    for (auto _ : state) {
        engine.run(views, logits);
        benchmark::DoNotOptimize(logits.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

void batches(benchmark::internal::Benchmark* b) {
    for (const std::int64_t precision : {32, 16, 8}) {
        for (const std::int64_t n : {1, 4, 16, 64, 256, 1024, 4096}) {
            b->Args({n, precision, 1, 0});
            if (n <= 256) {
                b->Args({n, precision, 1, 1}); // CUDA Graphs: where launches are a visible share
            }
        }
        if (precision != 8) {
            b->Args({256, precision, 0, 0}); // unfused, for the fusion comparison
        }
    }
}

BENCHMARK(BM_cnn_gpu)->Apply(batches)->ArgNames({"batch", "fp", "fused", "graph"})->UseRealTime();

} // namespace

int main(int argc, char** argv) {
    benchmark::AddCustomContext("machine", describe_machine());
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
