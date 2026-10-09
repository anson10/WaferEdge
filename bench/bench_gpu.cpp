// CPU vs GPU features by batch size: where does the GPU start to pay off?
//
//   build/cuda/bench/bench-gpu --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
//
// 40x40 random maps (10% fails, WaferLens's largest product). Each iteration processes one
// batch end to end: the GPU number includes packing into pinned memory, the upload, the
// kernel, the download and the sync (wall time, UseRealTime); the counters split out the GPU
// side from CUDA events. The CPU number is AVX2 on one thread, map by map.
#include "support/synth.hpp"
#include "waferedge/backend.hpp"
#include "waferedge/gpu_features.hpp"
#include "waferedge/machine.hpp"

#include <benchmark/benchmark.h>

#include <vector>

namespace {

using namespace waferedge;

constexpr int kSide = 40;
constexpr std::size_t kMaxBatch = 65536;

const std::vector<WaferMap>& maps() {
    static const std::vector<WaferMap> all = [] {
        std::vector<WaferMap> v;
        v.reserve(kMaxBatch);
        for (std::size_t i = 0; i < kMaxBatch; ++i) {
            v.push_back(synth::random_map(kSide, kSide, 100, static_cast<std::uint32_t>(i)));
        }
        return v;
    }();
    return all;
}

void BM_cpu_avx2(benchmark::State& state) {
    if (!backend::Avx2::available()) {
        state.SkipWithError("no AVX2");
        return;
    }
    const auto n = static_cast<std::size_t>(state.range(0));
    const Geometry geometry(kSide, kSide);
    const auto& all = maps();
    for (auto _ : state) {
        for (std::size_t i = 0; i < n; ++i) {
            benchmark::DoNotOptimize(backend::Avx2::features(all[i], geometry));
        }
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// timed = false: the end-to-end number, nothing extra on the hot path. timed = true: the same
// with CUDA events, to split the time into upload / kernel / download (costs ~60 us a batch
// on WSL2, so its wall time is not the headline).
void run_gpu(benchmark::State& state, gpu::FeatureKernel kernel, bool timed) {
    if (!backend::Cuda::available()) {
        state.SkipWithError("no CUDA device");
        return;
    }
    const auto n = static_cast<std::size_t>(state.range(0));
    const Geometry geometry(kSide, kSide);
    const auto& all = maps();
    const std::vector<WaferMapView> views(all.begin(),
                                          all.begin() + static_cast<std::ptrdiff_t>(n));
    const std::vector<const Geometry*> geometries(n, &geometry);
    std::vector<Features> out(n);
    gpu::FeatureEngine engine(kernel);
    engine.set_timing(timed);
    if (!engine.run(views, geometries, out)) { // warm-up: buffers, geometry upload
        state.SkipWithError(engine.error().c_str());
        return;
    }
    double pack = 0;
    double upload = 0;
    double compute = 0;
    double download = 0;
    for (auto _ : state) {
        if (!engine.run(views, geometries, out)) {
            state.SkipWithError(engine.error().c_str());
            return;
        }
        const auto t = engine.last_timings();
        pack += t.pack_ms;
        upload += t.upload_ms;
        compute += t.kernel_ms;
        download += t.download_ms;
    }
    const auto iterations = static_cast<double>(state.iterations());
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
    if (timed) {
        state.counters["pack_us"] = pack * 1e3 / iterations;
        state.counters["upload_us"] = upload * 1e3 / iterations;
        state.counters["kernel_us"] = compute * 1e3 / iterations;
        state.counters["download_us"] = download * 1e3 / iterations;
    }
}

void BM_gpu_shared_atomics(benchmark::State& state) {
    run_gpu(state, gpu::FeatureKernel::shared_atomics, false);
}

void BM_gpu_warp_aggregated(benchmark::State& state) {
    run_gpu(state, gpu::FeatureKernel::warp_aggregated, false);
}

void BM_gpu_shared_atomics_timed(benchmark::State& state) {
    run_gpu(state, gpu::FeatureKernel::shared_atomics, true);
}

void BM_gpu_warp_aggregated_timed(benchmark::State& state) {
    run_gpu(state, gpu::FeatureKernel::warp_aggregated, true);
}

void batches(benchmark::internal::Benchmark* b) {
    for (std::int64_t n = 1; n <= static_cast<std::int64_t>(kMaxBatch); n *= 4) {
        b->Arg(n);
    }
    b->Arg(static_cast<std::int64_t>(kMaxBatch));
}

BENCHMARK(BM_cpu_avx2)->Apply(batches)->UseRealTime();
BENCHMARK(BM_gpu_shared_atomics)->Apply(batches)->UseRealTime();
BENCHMARK(BM_gpu_warp_aggregated)->Apply(batches)->UseRealTime();
BENCHMARK(BM_gpu_shared_atomics_timed)->Apply(batches)->UseRealTime();
BENCHMARK(BM_gpu_warp_aggregated_timed)->Apply(batches)->UseRealTime();

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
