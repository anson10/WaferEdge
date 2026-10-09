// The GPU feature backend against the scalar reference, with ==. Labelled gpu: runs locally
// (ctest --preset cuda), not in CI.
#include "support/synth.hpp"
#include "waferedge/backend.hpp"
#include "waferedge/gpu_features.hpp"
#include "waferedge/map_file.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <deque>
#include <random>
#include <vector>

using namespace waferedge;
using gpu::FeatureEngine;
using gpu::FeatureKernel;

namespace {

void require_gpu() {
    if (!backend::Cuda::available()) {
        SKIP("no CUDA device with compute capability 8.6 or newer");
    }
}

// A batch: maps, a Geometry per map (shared per shape, as the pipeline will do), the scalar
// answers, and a run of the engine compared map by map.
struct Batch {
    std::vector<WaferMap> maps;
    std::deque<Geometry> geometries; // deque: references stay valid as it grows
    std::vector<const Geometry*> per_map;

    void add(WaferMap map) {
        const Geometry* g = nullptr;
        for (const auto& existing : geometries) {
            if (existing.rows() == map.rows() && existing.cols() == map.cols()) {
                g = &existing;
            }
        }
        if (g == nullptr) {
            g = &geometries.emplace_back(map.rows(), map.cols());
        }
        per_map.push_back(g);
        maps.push_back(std::move(map));
    }

    void check(FeatureEngine& engine) const {
        std::vector<WaferMapView> views(maps.begin(), maps.end());
        std::vector<Features> out(maps.size());
        REQUIRE(engine.run(views, per_map, out));
        std::size_t differ = 0;
        for (std::size_t i = 0; i < maps.size(); ++i) {
            if (out[i] != backend::Scalar::features(maps[i], *per_map[i])) {
                ++differ;
                UNSCOPED_INFO("map " << i << " (" << maps[i].rows() << "x" << maps[i].cols()
                                     << ") differs");
            }
        }
        CHECK(differ == 0);
    }
};

} // namespace

TEST_CASE("both GPU kernels match scalar on a batch of mixed shapes and densities") {
    require_gpu();
    const auto kernel = GENERATE(FeatureKernel::shared_atomics, FeatureKernel::warp_aggregated);
    Batch batch;
    const std::vector<std::pair<int, int>> shapes = {{1, 1},   {1, 31},  {1, 33},  {3, 11},
                                                     {24, 24}, {25, 27}, {29, 26}, {30, 34},
                                                     {40, 40}, {64, 64}};
    std::uint32_t seed = 1;
    for (const auto& [rows, cols] : shapes) {
        for (const unsigned per_mille : {0U, 30U, 300U, 900U, 1000U}) {
            batch.add(synth::random_map(rows, cols, per_mille, seed++));
        }
    }
    FeatureEngine engine(kernel);
    batch.check(engine);
}

TEST_CASE("the GPU backend, one map per call, matches scalar") {
    require_gpu();
    for (const auto& [rows, cols] :
         {std::pair{1, 1}, std::pair{24, 24}, std::pair{25, 27}, std::pair{64, 64}}) {
        const auto map = synth::random_map(rows, cols, 200, 9);
        const Geometry g(rows, cols);
        CHECK(backend::Cuda::features(map, g) == backend::Scalar::features(map, g));
    }
}

TEST_CASE("the GPU counts a whole map into one bucket, and every byte value as a bin") {
    require_gpu();
    const auto kernel = GENERATE(FeatureKernel::shared_atomics, FeatureKernel::warp_aggregated);
    FeatureEngine engine(kernel);
    // Every die in bucket 0: maximum contention on one shared counter.
    const int n = 600 * 32 + 5;
    const auto zeros = std::vector<std::uint8_t>(static_cast<std::size_t>(n), 0);
    const auto one_bucket = Geometry::from_tables(1, n, zeros, zeros, zeros);
    WaferMap fail_all(1, n, 2);
    Features f;
    const WaferMapView view = fail_all;
    const Geometry* g = &one_bucket;
    REQUIRE(engine.run({&view, 1}, {&g, 1}, {&f, 1}));
    CHECK(f.zone_fails[0] == static_cast<std::uint32_t>(n));
    CHECK(f == backend::Scalar::features(fail_all, one_bucket));

    Batch bytes;
    WaferMap all_values(16, 16);
    for (int i = 0; i < 256; ++i) {
        all_values.set(i / 16, i % 16, static_cast<std::uint8_t>(i));
    }
    bytes.add(std::move(all_values));
    bytes.add(synth::random_map(300, 300, 400, 5)); // a large WM-811K-sized map
    bytes.check(engine);
}

TEST_CASE("the GPU matches scalar on the real WaferLens fixture, as one batch") {
    require_gpu();
    auto set = MapSet::load(WAFEREDGE_TEST_DATA "/waferlens_sample.wmap");
    REQUIRE(set.has_value());
    GeometryCache cache;
    std::vector<WaferMapView> views;
    std::vector<const Geometry*> geometries;
    for (const auto& r : set->records()) {
        views.push_back(r.map);
        geometries.push_back(&cache.get(r.map.rows(), r.map.cols()));
    }
    std::vector<Features> out(views.size());
    FeatureEngine engine;
    REQUIRE(engine.run(views, geometries, out));
    for (std::size_t i = 0; i < views.size(); ++i) {
        CHECK(out[i] == backend::Scalar::features(views[i], *geometries[i]));
    }
}

TEST_CASE("buffers grow and geometry stays cached across runs") {
    require_gpu();
    FeatureEngine engine;
    engine.set_timing(true);
    SECTION("an empty batch is fine") {
        CHECK(engine.run({}, {}, {}));
    }
    SECTION("small, then large, then small batches give the same answers") {
        for (const std::size_t size : {std::size_t{1}, std::size_t{5000}, std::size_t{3}}) {
            Batch batch;
            for (std::size_t i = 0; i < size; ++i) {
                batch.add(synth::random_map(40, 40, 100, static_cast<std::uint32_t>(i)));
            }
            batch.check(engine);
            batch.check(engine); // second run: geometry from the cache, buffers reused
            const auto t = engine.last_timings();
            CHECK(t.upload_ms >= 0);
            CHECK(t.kernel_ms > 0);
            CHECK(t.download_ms >= 0);
        }
    }
}
