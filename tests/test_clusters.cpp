#include "support/synth.hpp"
#include "waferedge/clusters.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <deque>
#include <string_view>
#include <tuple>
#include <vector>

using namespace waferedge;
using Catch::Approx;

namespace {

// '.' off wafer, 'o' pass, 'x' fail.
WaferMap ascii(std::initializer_list<std::string_view> lines) {
    const int rows = static_cast<int>(lines.size());
    const int cols = static_cast<int>(lines.begin()->size());
    WaferMap map(rows, cols);
    int r = 0;
    for (auto line : lines) {
        for (int c = 0; c < cols; ++c) {
            const char ch = line[static_cast<std::size_t>(c)];
            map.set(r, c, ch == '.' ? 0 : ch == 'o' ? 1 : 2);
        }
        ++r;
    }
    return map;
}

// An independent labelling: breadth-first flood fill from each unlabelled fail die in
// row-major order, so labels come out in the same canonical order as union-find's.
std::vector<std::int32_t> flood_fill(const WaferMap& map) {
    const int rows = map.rows();
    const int cols = map.cols();
    std::vector<std::int32_t> label(static_cast<std::size_t>(rows * cols), -1);
    std::int32_t next = 0;
    for (int start = 0; start < rows * cols; ++start) {
        if (!is_fail(map.at(start / cols, start % cols)) ||
            label[static_cast<std::size_t>(start)] != -1) {
            continue;
        }
        std::deque<int> queue{start};
        label[static_cast<std::size_t>(start)] = next;
        while (!queue.empty()) {
            const int i = queue.front();
            queue.pop_front();
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    const int r = i / cols + dr;
                    const int c = i % cols + dc;
                    const auto j = static_cast<std::size_t>(r * cols + c);
                    if (r >= 0 && r < rows && c >= 0 && c < cols && is_fail(map.at(r, c)) &&
                        label[j] == -1) {
                        label[j] = next;
                        queue.push_back(r * cols + c);
                    }
                }
            }
        }
        ++next;
    }
    return label;
}

} // namespace

TEST_CASE("no fails, no clusters") {
    ClusterFinder finder;
    finder.run(synth::disc(20, 20));
    CHECK(finder.clusters().empty());
    CHECK(finder.largest() == nullptr);
    CHECK(std::ranges::all_of(finder.labels(), [](auto l) { return l == -1; }));
}

TEST_CASE("diagonal neighbours touch, a one-die gap separates") {
    ClusterFinder finder;
    finder.run(ascii({"xoo", "oxo", "oox"}));
    CHECK(finder.clusters().size() == 1);
    finder.run(ascii({"xox", "ooo", "xox"}));
    CHECK(finder.clusters().size() == 4);
}

TEST_CASE("a U shape is one cluster, rooted at its first die") {
    // The two arms only meet in the bottom row, after both have been labelled separately in
    // the scan: the second pass must still give them one label.
    ClusterFinder finder;
    finder.run(ascii({"xooox", "xooox", "xxxxx"}));
    REQUIRE(finder.clusters().size() == 1);
    const auto& u = finder.clusters()[0];
    CHECK(u.size == 9);
    CHECK(u.first == 0);
    CHECK(u.min_r == 0);
    CHECK(u.max_r == 2);
    CHECK(u.min_c == 0);
    CHECK(u.max_c == 4);
}

TEST_CASE("union-find agrees with a flood fill, label for label, on random maps") {
    const std::tuple<int, int> shape = GENERATE(table<int, int>({{24, 24}, {25, 27}, {40, 40}}));
    const auto per_mille = GENERATE(50U, 300U, 600U);
    const auto seed = GENERATE(1U, 2U);
    const auto [rows, cols] = shape;
    const auto map = synth::random_map(rows, cols, per_mille, seed);
    ClusterFinder finder;
    finder.run(map);
    const auto expected = flood_fill(map);
    CHECK(std::ranges::equal(finder.labels(), expected));

    std::uint32_t total = 0;
    for (std::size_t k = 0; k < finder.clusters().size(); ++k) {
        const auto& c = finder.clusters()[k];
        total += c.size;
        CHECK(finder.labels()[static_cast<std::size_t>(c.first)] == static_cast<std::int32_t>(k));
        if (k > 0) {
            CHECK(c.first > finder.clusters()[k - 1].first);
        }
    }
    CHECK(total == static_cast<std::uint32_t>(std::ranges::count_if(
                       map.view().bins(), [](auto b) { return is_fail(b); })));
}

TEST_CASE("cluster sizes don't change when the map is rotated") {
    const auto map = synth::random_map(30, 30, 250, 9);
    auto sizes = [](const WaferMap& m) {
        ClusterFinder finder;
        finder.run(m);
        std::vector<std::uint32_t> s;
        for (const auto& c : finder.clusters()) {
            s.push_back(c.size);
        }
        std::ranges::sort(s);
        return s;
    };
    CHECK(sizes(synth::rotate90(map)) == sizes(map));
}

TEST_CASE("shape: a single die, a line, a square") {
    ClusterFinder finder;
    SECTION("single die: elongation 1") {
        finder.run(ascii({"ooo", "oxo", "ooo"}));
        const auto s = shape(finder.clusters()[0]);
        CHECK(s.row == 1.0);
        CHECK(s.col == 1.0);
        CHECK(s.elongation == Approx(1.0));
    }
    SECTION("horizontal line of 10: elongation 10, angle 0") {
        WaferMap map(3, 12, 1);
        for (int c = 1; c <= 10; ++c) {
            map.set(1, c, 2);
        }
        finder.run(map);
        const auto s = shape(finder.clusters()[0]);
        CHECK(s.elongation == Approx(10.0));
        CHECK(s.angle_deg == Approx(0.0).margin(1e-9));
        CHECK(s.col == Approx(5.5));
    }
    SECTION("vertical line: angle 90") {
        WaferMap map(12, 3, 1);
        for (int r = 1; r <= 10; ++r) {
            map.set(r, 1, 2);
        }
        finder.run(map);
        CHECK(shape(finder.clusters()[0]).angle_deg == Approx(90.0));
    }
    SECTION("rising diagonal: angle 45, long and thin") {
        WaferMap map(12, 12, 1);
        for (int i = 0; i < 10; ++i) {
            map.set(10 - i, 1 + i, 2); // row decreases as column increases: up and right
        }
        finder.run(map);
        const auto s = shape(finder.clusters()[0]);
        CHECK(s.angle_deg == Approx(45.0));
        CHECK(s.elongation > 10.0); // diagonal dies are sqrt(2) apart
    }
    SECTION("a 5x5 block is round") {
        WaferMap map(7, 7, 1);
        synth::paint(map, [](int r, int c) { return r >= 1 && r <= 5 && c >= 1 && c <= 5; });
        finder.run(map);
        CHECK(shape(finder.clusters()[0]).elongation == Approx(1.0));
    }
}

TEST_CASE("largest() picks the biggest cluster, the first one on ties") {
    ClusterFinder finder;
    finder.run(ascii({"xxoxx", "ooooo", "xxxoo"}));
    REQUIRE(finder.clusters().size() == 3);
    CHECK(finder.largest() == &finder.clusters()[2]);
    finder.run(ascii({"xxoxx", "ooooo", "ooooo"}));
    CHECK(finder.largest() == &finder.clusters()[0]);
}
