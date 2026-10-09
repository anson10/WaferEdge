// FabEye's CNN on the GPU engine, by batch size: maps/s end to end (bins up, preprocessing,
// 8 convolutions, pools, head, logits down), fp32 and fp16, bias + ReLU fused or not.
//
//   build/cuda/bench/bench-cnn --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
//
// Needs data/fabeye_cnn.wcnn (tools/export_cnn.py). 40 x 40 random maps (10% fails): the
// network's cost doesn't depend on the map, every map is resized to 64 x 64.
#include "support/synth.hpp"
#include "waferedge/backend.hpp"
#include "waferedge/cnn.hpp"
#include "waferedge/gpu_cnn.hpp"
#include "waferedge/machine.hpp"

#include <benchmark/benchmark.h>

#include <optional>
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

void BM_cnn_gpu(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto precision = state.range(1) == 16 ? gpu::CnnPrecision::fp16 : gpu::CnnPrecision::fp32;
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
    gpu::CnnEngine engine(*model(), precision);
    engine.set_fused(state.range(2) != 0);
    if (!engine.run(views, logits)) { // warm-up: buffers
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
    for (const std::int64_t precision : {32, 16}) {
        for (const std::int64_t n : {1, 4, 16, 64, 256, 1024, 4096}) {
            b->Args({n, precision, 1});
        }
        b->Args({256, precision, 0}); // unfused, for the fusion comparison
    }
}

BENCHMARK(BM_cnn_gpu)->Apply(batches)->ArgNames({"batch", "fp", "fused"})->UseRealTime();

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
