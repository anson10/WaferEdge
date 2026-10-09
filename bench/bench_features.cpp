// Scalar feature extraction, maps/s by map size: the reference the AVX2 and CUDA backends
// are measured against.
//
//   build/release/bench/bench-features
#include "support/synth.hpp"
#include "waferedge/features.hpp"
#include "waferedge/machine.hpp"

#include <benchmark/benchmark.h>

#include <vector>

namespace {

using namespace waferedge;

// A batch of 256 random maps (10% fails) of one size, so the loop isn't one cached map.
void BM_features_scalar(benchmark::State& state) {
    const int n = static_cast<int>(state.range(0));
    std::vector<WaferMap> maps;
    for (std::uint32_t seed = 0; seed < 256; ++seed) {
        maps.push_back(synth::random_map(n, n, 100, seed));
    }
    const Geometry geometry(n, n);
    for (auto _ : state) {
        for (const auto& map : maps) {
            benchmark::DoNotOptimize(compute_features(map, geometry));
        }
    }
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(maps.size()));
}
BENCHMARK(BM_features_scalar)->Arg(24)->Arg(30)->Arg(40)->Arg(64);

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
