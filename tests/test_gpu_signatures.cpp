// The GPU signatures (features, Hough line) against the CPU references, with ==. Labelled gpu: runs
// locally (ctest --preset cuda), not in CI.
#include "support/synth.hpp"
#include "waferedge/backend.hpp"
#include "waferedge/gpu_signatures.hpp"
#include "waferedge/hough.hpp"
#include "waferedge/map_file.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <deque>
#include <random>
#include <vector>

using namespace waferedge;
using gpu::FeatureKernel;
using gpu::SignatureEngine;

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

    void check(SignatureEngine& engine) const {
        std::vector<WaferMapView> views(maps.begin(), maps.end());
        std::vector<Features> out(maps.size());
        REQUIRE(engine.run(views, per_map, out, {}));
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
    SignatureEngine engine(kernel);
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
    SignatureEngine engine(kernel);
    // Every die in bucket 0: maximum contention on one shared counter.
    const int n = 600 * 32 + 5;
    const auto zeros = std::vector<std::uint8_t>(static_cast<std::size_t>(n), 0);
    const auto one_bucket = Geometry::from_tables(1, n, zeros, zeros, zeros);
    WaferMap fail_all(1, n, 2);
    Features f;
    const WaferMapView view = fail_all;
    const Geometry* g = &one_bucket;
    REQUIRE(engine.run({&view, 1}, {&g, 1}, {&f, 1}, {}));
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
    SignatureEngine engine;
    REQUIRE(engine.run(views, geometries, out, {}));
    for (std::size_t i = 0; i < views.size(); ++i) {
        CHECK(out[i] == backend::Scalar::features(views[i], *geometries[i]));
    }
}

TEST_CASE("buffers grow and geometry stays cached across runs") {
    require_gpu();
    SignatureEngine engine;
    engine.set_timing(true);
    SECTION("an empty batch is fine") {
        CHECK(engine.run({}, {}, {}, {}));
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
            CHECK(t.features_ms > 0);
            CHECK(t.download_ms >= 0);
        }
    }
}

namespace {

// Hough lines of a batch on the GPU, compared map by map with the CPU's HoughTransform.
void check_lines(SignatureEngine& engine, const std::vector<WaferMap>& maps) {
    std::vector<WaferMapView> views(maps.begin(), maps.end());
    std::vector<HoughLine> lines(maps.size());
    REQUIRE(engine.run(views, {}, {}, lines));
    HoughTransform cpu;
    for (std::size_t i = 0; i < maps.size(); ++i) {
        const auto expected = cpu.run(maps[i]);
        INFO("map " << i << " (" << maps[i].rows() << "x" << maps[i].cols() << "): gpu angle "
                    << lines[i].angle << " rho " << lines[i].rho << " votes " << lines[i].votes
                    << ", cpu angle " << expected.angle << " rho " << expected.rho << " votes "
                    << expected.votes);
        CHECK(lines[i] == expected);
    }
}

WaferMap with_scratch(int rows, int cols, unsigned per_mille, std::uint32_t seed) {
    auto map = synth::random_map(rows, cols, per_mille, seed);
    synth::paint(map, [&](int r, int c) {
        return c == cols / 4 + r / 2 && r > rows / 6 && r < rows - rows / 6;
    });
    return map;
}

} // namespace

TEST_CASE("the GPU Hough line equals the CPU's on mixed shapes, densities and scratches") {
    require_gpu();
    std::vector<WaferMap> maps;
    std::uint32_t seed = 100;
    for (const auto& [rows, cols] :
         {std::pair{3, 11}, std::pair{24, 24}, std::pair{25, 27}, std::pair{29, 26},
          std::pair{30, 34}, std::pair{40, 40}, std::pair{64, 64}}) {
        for (const unsigned per_mille : {0U, 20U, 150U, 600U, 1000U}) {
            maps.push_back(synth::random_map(rows, cols, per_mille, seed++));
            maps.push_back(with_scratch(rows, cols, per_mille / 4, seed++));
        }
    }
    SignatureEngine engine;
    check_lines(engine, maps);
}

TEST_CASE("the GPU Hough line equals the CPU's on maps too big for one chunk of angles") {
    require_gpu();
    // rows + cols > 64: the vote table doesn't fit 180 angles; WM-811K's largest is 212 x 204.
    const std::vector<WaferMap> maps = {
        with_scratch(212, 204, 30, 1), synth::random_map(212, 204, 300, 2),
        with_scratch(300, 300, 10, 3), synth::random_map(97, 131, 100, 4)};
    SignatureEngine engine;
    check_lines(engine, maps);
}

TEST_CASE("ties go the CPU's way: lowest angle, then lowest offset") {
    require_gpu();
    std::vector<WaferMap> maps;
    // No fails: no line at all.
    maps.push_back(synth::disc(30, 30));
    // One fail die: every angle gets one vote, the line is angle 0.
    auto one = synth::disc(30, 30);
    one.set(10, 12, 2);
    maps.push_back(std::move(one));
    // Two equally long parallel scratches: two bins tie at the same angle.
    auto two = synth::disc(40, 40);
    synth::paint(two, [](int r, int c) { return (r == 12 || r == 27) && c >= 10 && c < 30; });
    maps.push_back(std::move(two));
    // A short scratch: a plateau of equal-vote angles around its normal.
    auto plateau = synth::disc(40, 40);
    synth::paint(plateau, [](int r, int c) { return c == 20 && r >= 17 && r < 23; });
    maps.push_back(std::move(plateau));
    SignatureEngine engine;
    check_lines(engine, maps);
}

TEST_CASE("features and Hough from one upload equal each computed alone") {
    require_gpu();
    Batch batch;
    for (std::uint32_t i = 0; i < 64; ++i) {
        batch.add(with_scratch(40, 40, 80, i));
    }
    std::vector<WaferMapView> views(batch.maps.begin(), batch.maps.end());
    std::vector<Features> both_f(views.size());
    std::vector<HoughLine> both_l(views.size());
    std::vector<Features> alone_f(views.size());
    std::vector<HoughLine> alone_l(views.size());
    SignatureEngine engine;
    REQUIRE(engine.run(views, batch.per_map, both_f, both_l));
    REQUIRE(engine.run(views, batch.per_map, alone_f, {}));
    REQUIRE(engine.run(views, {}, {}, alone_l));
    CHECK(both_f == alone_f);
    CHECK(both_l == alone_l);
}

namespace {

// Cluster summaries of a batch on the GPU, compared map by map with the CPU's ClusterFinder.
void check_clusters(SignatureEngine& engine, const std::vector<WaferMap>& maps) {
    std::vector<WaferMapView> views(maps.begin(), maps.end());
    std::vector<ClusterSummary> gpu(maps.size());
    REQUIRE(engine.run(views, {}, {}, {}, gpu));
    ClusterFinder cpu;
    for (std::size_t i = 0; i < maps.size(); ++i) {
        cpu.run(maps[i]);
        const auto expected = cpu.summary();
        INFO("map " << i << " (" << maps[i].rows() << "x" << maps[i].cols() << "): gpu "
                    << gpu[i].clusters << " clusters, largest " << gpu[i].largest.size << " at "
                    << gpu[i].largest.first << "; cpu " << expected.clusters << ", "
                    << expected.largest.size << " at " << expected.largest.first);
        CHECK(gpu[i] == expected);
    }
}

// One cluster that winds through the whole map: rows filled left to right, joined at
// alternating ends. The longest parent chains, and every row merging into the rest late.
WaferMap snake(int n) {
    WaferMap map(n, n, 1);
    for (int r = 0; r < n; r += 2) {
        for (int c = 0; c < n; ++c) {
            map.set(r, c, 2);
        }
        if (r + 1 < n) {
            map.set(r + 1, (r / 2) % 2 == 0 ? n - 1 : 0, 2);
        }
    }
    return map;
}

} // namespace

TEST_CASE("GPU clusters equal the CPU's on mixed shapes and densities") {
    require_gpu();
    std::vector<WaferMap> maps;
    std::uint32_t seed = 500;
    for (const auto& [rows, cols] :
         {std::pair{1, 1}, std::pair{1, 33}, std::pair{3, 11}, std::pair{24, 24}, std::pair{25, 27},
          std::pair{40, 40}, std::pair{64, 64}}) {
        for (const unsigned per_mille : {0U, 30U, 300U, 550U, 700U, 1000U}) {
            maps.push_back(synth::random_map(rows, cols, per_mille, seed++));
        }
    }
    SignatureEngine engine;
    check_clusters(engine, maps);
}

TEST_CASE("GPU clusters survive union-find's hard cases") {
    require_gpu();
    std::vector<WaferMap> maps;
    maps.push_back(snake(40)); // one winding cluster
    maps.push_back(snake(64));
    WaferMap all_fail(40, 40, 2); // one cluster of every die
    maps.push_back(all_fail);
    WaferMap checker(40, 40, 1); // only diagonal neighbours: one big 8-connected cluster
    synth::paint(checker, [](int r, int c) { return (r + c) % 2 == 0; });
    maps.push_back(checker);
    WaferMap dots(40, 40, 1); // isolated dies: 400 single-die clusters, all tied for largest
    synth::paint(dots, [](int r, int c) { return r % 2 == 0 && c % 2 == 0; });
    maps.push_back(dots);
    WaferMap two(30, 30, 1); // two equal blobs: the tie goes to the one whose first die comes first
    synth::paint(two, [](int r, int c) {
        return (r >= 5 && r < 9 && c >= 20 && c < 24) || (r >= 15 && r < 19 && c >= 3 && c < 7);
    });
    maps.push_back(two);
    SignatureEngine engine;
    check_clusters(engine, maps);
}

TEST_CASE("GPU clusters of maps too big for shared memory use the global scratch") {
    require_gpu();
    // 65 x 65 = 4225 positions, just over the 4096 kept in shared memory; WM-811K's largest.
    const std::vector<WaferMap> maps = {synth::random_map(65, 65, 400, 1), snake(65),
                                        synth::random_map(24, 24, 300, 2),
                                        synth::random_map(212, 204, 450, 3), snake(150)};
    SignatureEngine engine;
    check_clusters(engine, maps);
}
