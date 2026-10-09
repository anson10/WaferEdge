// CPU vs GPU by batch size, for each signature and all three together: where does the GPU pay
// off, against one CPU core and against the whole CPU (6 cores, 12 threads)?
//
//   build/cuda/bench/bench-gpu --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
//
// 40x40 random maps (10% fails, WaferLens's largest product). Each iteration processes one
// batch end to end: the GPU number includes packing into pinned memory, the upload, the
// kernel, the download and the sync (wall time, UseRealTime); the counters split out the GPU
// side from CUDA events. The CPU number is AVX2 on one thread, map by map.
#include "support/synth.hpp"
#include "waferedge/backend.hpp"
#include "waferedge/clusters.hpp"
#include "waferedge/gpu_signatures.hpp"
#include "waferedge/hough.hpp"
#include "waferedge/machine.hpp"
#include "waferedge/thread_pool.hpp"

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
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
    gpu::SignatureEngine engine(kernel);
    engine.set_timing(timed);
    if (!engine.run(views, geometries, out, {})) { // warm-up: buffers, geometry upload
        state.SkipWithError(engine.error().c_str());
        return;
    }
    double pack = 0;
    double upload = 0;
    double compute = 0;
    double download = 0;
    for (auto _ : state) {
        if (!engine.run(views, geometries, out, {})) {
            state.SkipWithError(engine.error().c_str());
            return;
        }
        const auto t = engine.last_timings();
        pack += t.pack_ms;
        upload += t.upload_ms;
        compute += t.features_ms;
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

void BM_cpu_hough(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto& all = maps();
    HoughTransform hough;
    for (auto _ : state) {
        for (std::size_t i = 0; i < n; ++i) {
            benchmark::DoNotOptimize(hough.run(all[i]));
        }
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

void BM_gpu_hough(benchmark::State& state) {
    if (!backend::Cuda::available()) {
        state.SkipWithError("no CUDA device");
        return;
    }
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto& all = maps();
    const std::vector<WaferMapView> views(all.begin(),
                                          all.begin() + static_cast<std::ptrdiff_t>(n));
    std::vector<HoughLine> lines(n);
    gpu::SignatureEngine engine;
    if (!engine.run(views, {}, {}, lines)) { // warm-up: buffers, cos / sin tables
        state.SkipWithError(engine.error().c_str());
        return;
    }
    for (auto _ : state) {
        if (!engine.run(views, {}, {}, lines)) {
            state.SkipWithError(engine.error().c_str());
            return;
        }
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

void BM_cpu_clusters(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto& all = maps();
    ClusterFinder finder;
    for (auto _ : state) {
        for (std::size_t i = 0; i < n; ++i) {
            finder.run(all[i]);
            benchmark::DoNotOptimize(finder.summary());
        }
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

// GPU signatures by batch: clusters only, or features + Hough + clusters on one upload.
void run_gpu_signatures(benchmark::State& state, bool all_three) {
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
    std::vector<Features> features(all_three ? n : 0);
    std::vector<HoughLine> lines(all_three ? n : 0);
    std::vector<ClusterSummary> clusters(n);
    gpu::SignatureEngine engine;
    if (!engine.run(views, geometries, features, lines, clusters)) { // warm-up
        state.SkipWithError(engine.error().c_str());
        return;
    }
    for (auto _ : state) {
        if (!engine.run(views, geometries, features, lines, clusters)) {
            state.SkipWithError(engine.error().c_str());
            return;
        }
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

void BM_gpu_clusters(benchmark::State& state) {
    run_gpu_signatures(state, false);
}

// The pipeline's case: every signature from one upload.
void BM_gpu_all_signatures(benchmark::State& state) {
    run_gpu_signatures(state, true);
}

// The whole CPU: a thread pool, ~4 chunks per thread (at most 64 maps a chunk) so small
// batches still spread, per-thread Hough and cluster scratch. all_three = false: AVX2 features
// only; true: features + Hough + clusters, the GPU's all-signatures workload.
void run_cpu_threads(benchmark::State& state, bool all_three) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto threads = static_cast<unsigned>(state.range(1));
    const auto spin = std::chrono::microseconds{state.range(2)};
    const Geometry geometry(kSide, kSide);
    const auto& all = maps();
    ThreadPool pool(threads, spin);
    struct Scratch {
        HoughTransform hough;
        ClusterFinder clusters;
    };
    std::vector<Scratch> scratch(pool.size());
    std::vector<Features> features(n);
    std::vector<HoughLine> lines(n);
    std::vector<ClusterSummary> clusters(n);
    const auto features_fn = best_features();
    const std::size_t grain = std::clamp<std::size_t>(n / (4 * pool.size()), 1, 64);
    for (auto _ : state) {
        pool.for_each(n, grain, [&](unsigned worker, std::size_t begin, std::size_t end) noexcept {
            auto& s = scratch[worker];
            for (std::size_t i = begin; i < end; ++i) {
                features[i] = features_fn(all[i], geometry);
                if (all_three) {
                    lines[i] = s.hough.run(all[i]);
                    s.clusters.run(all[i]);
                    clusters[i] = s.clusters.summary();
                }
            }
        });
        benchmark::DoNotOptimize(features.data());
        benchmark::DoNotOptimize(lines.data());
        benchmark::DoNotOptimize(clusters.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

void BM_cpu_features_threads(benchmark::State& state) {
    run_cpu_threads(state, false);
}

void BM_cpu_all_signatures_threads(benchmark::State& state) {
    run_cpu_threads(state, true);
}

// spin_us 0: workers sleep between batches; 200: they spin up to 200 us first (back-to-back
// batches in a benchmark loop then never sleep).
void cpu_batches(benchmark::internal::Benchmark* b, std::int64_t max_batch) {
    for (const std::int64_t spin : {0, 200}) {
        for (const std::int64_t threads : {1, 6, 12}) {
            for (std::int64_t n = 1; n <= max_batch; n *= 4) {
                if (threads > 1 || spin == 0) {
                    b->Args({n, threads, spin});
                }
            }
        }
    }
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
// The CPU Hough is ~50 us a map: stop its sweep at 4096 maps (~0.2 s a batch).
BENCHMARK(BM_cpu_hough)->RangeMultiplier(4)->Range(1, 4096)->UseRealTime();
BENCHMARK(BM_gpu_hough)->Apply(batches)->UseRealTime();
BENCHMARK(BM_cpu_clusters)->RangeMultiplier(4)->Range(1, 16384)->UseRealTime();
BENCHMARK(BM_gpu_clusters)->Apply(batches)->UseRealTime();
BENCHMARK(BM_gpu_all_signatures)->Apply(batches)->UseRealTime();
BENCHMARK(BM_cpu_features_threads)->Apply([](auto* b) {
    cpu_batches(b, 65536);
}) -> ArgNames({"batch", "threads", "spin_us"}) -> UseRealTime();
BENCHMARK(BM_cpu_all_signatures_threads)->Apply([](auto* b) {
    cpu_batches(b, 16384);
}) -> ArgNames({"batch", "threads", "spin_us"}) -> UseRealTime();
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
