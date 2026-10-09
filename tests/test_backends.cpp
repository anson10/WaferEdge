// Every feature backend against the scalar reference, with ==. One template, so a new
// backend (CUDA in phase 2a) is one more type in the list.
#include "support/synth.hpp"
#include "waferedge/backend.hpp"
#include "waferedge/map_file.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

using namespace waferedge;

namespace {

template <typename B>
void require_available() {
    if (!B::available()) {
        SKIP(B::name << " is not supported by this CPU");
    }
}

template <typename B>
void check_equal(const WaferMap& map) {
    const Geometry g(map.rows(), map.cols());
    INFO(B::name << " on " << map.rows() << "x" << map.cols());
    CHECK(B::features(map, g) == backend::Scalar::features(map, g));
}

} // namespace

TEMPLATE_TEST_CASE("backends match scalar on random maps of every size class", "[backend]",
                   backend::Scalar, backend::Avx2) {
    require_available<TestType>();
    // Multiples of 32 dies, and sizes that leave 1..31 dies for the tail; single rows and
    // columns; WM-811K's odd shapes.
    const std::vector<std::pair<int, int>> shapes = {
        {1, 1},   {1, 31},  {1, 32},  {1, 33},  {31, 1},  {2, 16},  {3, 11},  {8, 8},   {24, 24},
        {25, 27}, {26, 26}, {29, 26}, {30, 30}, {30, 34}, {40, 40}, {52, 47}, {64, 64}, {71, 3}};
    for (const auto& [rows, cols] : shapes) {
        for (const unsigned per_mille : {0U, 30U, 300U, 900U, 1000U}) {
            check_equal<TestType>(synth::random_map(rows, cols, per_mille,
                                                    static_cast<std::uint32_t>(rows * 131 + cols)));
        }
    }
}

TEMPLATE_TEST_CASE("byte counters are flushed before they wrap", "[backend]", backend::Scalar,
                   backend::Avx2) {
    require_available<TestType>();
    // A counter lane gains 1 per 32-die block only if that lane's die is in the bucket. On a
    // real wafer the tables change from block to block, so no lane gets near 255; to reach
    // and pass the limit, every die must be in the same bucket: explicit tables, all 0.
    // 255 blocks fill each lane to exactly 255, the most a byte holds; 256 would wrap to 0.
    for (const int blocks : {255, 256, 257, 511, 600}) {
        const int n = blocks * 32 + 5; // plus a tail
        const auto zeros = std::vector<std::uint8_t>(static_cast<std::size_t>(n), 0);
        const auto g = Geometry::from_tables(1, n, zeros, zeros, zeros);
        WaferMap fail_all(1, n, 2);
        INFO(blocks << " blocks");
        const auto f = TestType::features(fail_all, g);
        CHECK(f.zone_dies[0] == static_cast<std::uint32_t>(n));
        CHECK(f.zone_fails[0] == static_cast<std::uint32_t>(n));
        CHECK(f.ring_fails[0] == static_cast<std::uint32_t>(n));
        CHECK(f == backend::Scalar::features(fail_all, g));
    }
    // And on real shapes, large maps that need many rounds.
    check_equal<TestType>(synth::random_map(300, 300, 400, 5));
    check_equal<TestType>(WaferMap(1, 8161, 2));
}

TEMPLATE_TEST_CASE("backends match scalar on bins the generator doesn't make", "[backend]",
                   backend::Scalar, backend::Avx2) {
    require_available<TestType>();
    // Every byte value, including 128..255 (negative as signed bytes: a signed compare
    // for "bin >= 2" would get these wrong).
    WaferMap map(16, 16);
    for (int i = 0; i < 256; ++i) {
        map.set(i / 16, i % 16, static_cast<std::uint8_t>(i));
    }
    check_equal<TestType>(map);
    std::mt19937 rng(3);
    WaferMap noise(37, 41);
    for (auto& b : noise.bins()) {
        b = static_cast<std::uint8_t>(rng());
    }
    check_equal<TestType>(noise);
}

TEMPLATE_TEST_CASE("backends match scalar on the real WaferLens fixture", "[backend]",
                   backend::Scalar, backend::Avx2) {
    require_available<TestType>();
    auto set = MapSet::load(WAFEREDGE_TEST_DATA "/waferlens_sample.wmap");
    REQUIRE(set.has_value());
    GeometryCache cache;
    for (const auto& r : set->records()) {
        const auto& g = cache.get(r.map.rows(), r.map.cols());
        CHECK(TestType::features(r.map, g) == backend::Scalar::features(r.map, g));
    }
}

TEST_CASE("the run-time choice is a backend this CPU supports") {
    const auto name = best_features_name();
    CHECK((name == backend::Scalar::name || name == backend::Avx2::name));
    if (backend::Avx2::available()) {
        CHECK(name == backend::Avx2::name);
    }
    const auto map = synth::random_map(30, 30, 100, 1);
    const Geometry g(30, 30);
    CHECK(best_features()(map, g) == compute_features(map, g));
}
