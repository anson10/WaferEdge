// Scalar signature stages, maps/s by map size: the references the AVX2 and CUDA backends
// are measured against.
//
//   build/release/bench/bench-features
#include "support/synth.hpp"
#include "waferedge/clusters.hpp"
#include "waferedge/features.hpp"
#include "waferedge/hough.hpp"
#include "waferedge/machine.hpp"
#include "waferedge/randomness.hpp"

#include <benchmark/benchmark.h>

#include <vector>

namespace {

using namespace waferedge;

// A batch of 256 random maps (10% fails) of one size, so the loop isn't one cached map.
std::vector<WaferMap> batch(int n) {
    std::vector<WaferMap> maps;
    for (std::uint32_t seed = 0; seed < 256; ++seed) {
        maps.push_back(synth::random_map(n, n, 100, seed));
    }
    return maps;
}

template <typename Stage>
void run_batch(benchmark::State& state, Stage&& stage) {
    const auto maps = batch(static_cast<int>(state.range(0)));
    for (auto _ : state) {
        for (const auto& map : maps) {
            benchmark::DoNotOptimize(stage(map));
        }
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(maps.size()));
}

void BM_features_scalar(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    const Geometry geometry(n, n);
    run_batch(state, [&](const WaferMap& m) { return compute_features(m, geometry); });
}

void BM_clusters_scalar(benchmark::State& state) {
    ClusterFinder finder;
    run_batch(state, [&](const WaferMap& m) {
        finder.run(m);
        return finder.clusters().size();
    });
}

void BM_hough_scalar(benchmark::State& state) {
    HoughTransform hough;
    run_batch(state, [&](const WaferMap& m) { return hough.run(m); });
}

void BM_join_count_scalar(benchmark::State& state) {
    run_batch(state, [](const WaferMap& m) { return join_count(m); });
}

BENCHMARK(BM_features_scalar)->Arg(24)->Arg(30)->Arg(40)->Arg(64);
BENCHMARK(BM_clusters_scalar)->Arg(24)->Arg(30)->Arg(40)->Arg(64);
BENCHMARK(BM_hough_scalar)->Arg(24)->Arg(30)->Arg(40)->Arg(64);
BENCHMARK(BM_join_count_scalar)->Arg(24)->Arg(30)->Arg(40)->Arg(64);

} // namespace

int main(int argc, char** argv) {
    benchmark::AddCustomContext("waferedge", describe_machine());
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
