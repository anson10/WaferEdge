#include "support/synth.hpp"
#include "waferedge/features.hpp"
#include "waferedge/map_file.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <numeric>
#include <tuple>

using namespace waferedge;
using Catch::Approx;

namespace {

Features features_of(const WaferMap& map) {
    return compute_features(map, Geometry(map.rows(), map.cols()));
}

template <typename A>
std::uint32_t sum(const A& a) {
    return std::accumulate(a.begin(), a.end(), 0U);
}

} // namespace

TEST_CASE("a map without dies has all-zero features") {
    const auto f = features_of(WaferMap(10, 10, 0));
    CHECK(f == Features{});
    CHECK(yield(f) == 0.0);
    CHECK(edge_ratio(f) == 0.0);
    CHECK(max_sector_ratio(f) == 0.0);
}

TEST_CASE("a single passing die") {
    const auto f = features_of(WaferMap(1, 1, 1));
    CHECK(f.dies == 1);
    CHECK(f.fails == 0);
    CHECK(f.zone_dies[0] == 1);
    CHECK(f.sector_dies[0] == 1);
    CHECK(yield(f) == 1.0);
    CHECK(center_ratio(f) == 0.0); // no fail field to compare against
}

TEST_CASE("a fully passing and a fully failing disc") {
    auto map = synth::disc(30, 30);
    const auto pass = features_of(map);
    CHECK(pass.fails == 0);
    CHECK(yield(pass) == 1.0);

    synth::paint(map, [](int, int) { return true; }, 3);
    const auto fail = features_of(map);
    CHECK(fail.dies == pass.dies);
    CHECK(fail.fails == fail.dies);
    CHECK(yield(fail) == 0.0);
    for (int z = 0; z < kZones; ++z) {
        CHECK(zone_ratio(fail, z) == 1.0);
    }
    CHECK(max_sector_ratio(fail) == 1.0);
}

TEST_CASE("counts partition the dies, for random maps of any shape") {
    const std::tuple<int, int> shape = GENERATE(table<int, int>(
        {{24, 24}, {25, 27}, {30, 30}, {40, 40}, {52, 47}, {64, 64}, {1, 9}, {2, 2}}));
    const auto [rows, cols] = shape;
    const auto seed = GENERATE(1U, 2U, 3U);
    const auto map = synth::random_map(rows, cols, 150, seed);
    const auto f = features_of(map);
    INFO(rows << "x" << cols << " seed " << seed);
    CHECK(sum(f.zone_dies) == f.dies);
    CHECK(sum(f.sector_dies) == f.dies);
    CHECK(sum(f.zone_fails) == f.fails);
    CHECK(sum(f.sector_fails) == f.fails);
    CHECK(sum(f.ring_dies) == f.dies);
    CHECK(sum(f.ring_fails) == f.fails);
    for (std::size_t z = 0; z < kZones; ++z) {
        CHECK(f.zone_fails[z] <= f.zone_dies[z]);
    }
    std::uint32_t dies = 0;
    std::uint32_t fails = 0;
    const WaferMapView view = map;
    for (auto b : view.bins()) {
        dies += on_wafer(b) ? 1U : 0U;
        fails += is_fail(b) ? 1U : 0U;
    }
    CHECK(f.dies == dies);
    CHECK(f.fails == fails);
}

TEST_CASE("rotating a map keeps zone counts and moves sector counts by two") {
    const int n = GENERATE(24, 25, 40);
    const auto map = synth::random_map(n, n, 200, 7);
    const auto f = features_of(map);
    const auto g = features_of(synth::rotate90(map));
    CHECK(g.zone_dies == f.zone_dies);
    CHECK(g.zone_fails == f.zone_fails);
    CHECK(g.ring_dies == f.ring_dies);
    CHECK(g.ring_fails == f.ring_fails);
    for (std::size_t s = 0; s < kSectors; ++s) {
        // The centre die of an odd grid stays in sector 0, so compare fails only off-centre.
        if (n % 2 == 0) {
            CHECK(g.sector_dies[(s + 2) % kSectors] == f.sector_dies[s]);
            CHECK(g.sector_fails[(s + 2) % kSectors] == f.sector_fails[s]);
        }
    }
    CHECK(g.dies == f.dies);
    CHECK(g.fails == f.fails);
}

TEST_CASE("pattern shapes move the ratios the right way") {
    const int n = 30;
    SECTION("center blob") {
        auto map = synth::disc(n, n);
        synth::paint(map, [&](int r, int c) { return synth::rho2(r, c, n, n) < 0.15; });
        const auto f = features_of(map);
        CHECK(center_ratio(f) > 3.0);
        CHECK(edge_ratio(f) == 0.0);
    }
    SECTION("edge ring") {
        auto map = synth::disc(n, n);
        synth::paint(map, [&](int r, int c) { return zone_of(r, c, n, n) == kZones - 1; });
        const auto f = features_of(map);
        CHECK(edge_ratio(f) == Approx(static_cast<double>(f.dies) / f.zone_dies[kZones - 1]));
        CHECK(center_ratio(f) == 0.0);
    }
    SECTION("edge-loc: one side of the edge") {
        auto map = synth::disc(n, n);
        synth::paint(map, [&](int r, int c) {
            return zone_of(r, c, n, n) == kZones - 1 && sector_of(r, c, n, n) == 0;
        });
        const auto f = features_of(map);
        CHECK(max_sector_ratio(f) > 5.0);
        CHECK(f.sector_fails[0] == f.fails);
    }
}

TEST_CASE("features of real WaferLens maps match counts from NumPy") {
    auto set = MapSet::load(WAFEREDGE_TEST_DATA "/waferlens_sample.wmap");
    REQUIRE(set.has_value());
    GeometryCache cache;
    const auto& first = set->records()[0]; // wafer 14, "none", 30x30
    const auto f = compute_features(first.map, cache.get(first.map.rows(), first.map.cols()));
    CHECK(f.dies == 656);
    CHECK(f.fails == 60);

    // An edge-ring wafer (10570, 24x24): 132 fails, concentrated at the edge.
    const auto& ring = set->records()[12];
    REQUIRE(ring.wafer_id == 10570);
    const auto g = compute_features(ring.map, cache.get(ring.map.rows(), ring.map.cols()));
    CHECK(g.dies == 432);
    CHECK(g.fails == 132);
    CHECK(edge_ratio(g) > 2.0);
}
