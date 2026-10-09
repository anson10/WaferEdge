#include "support/synth.hpp"
#include "waferedge/hough.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ranges>

using namespace waferedge;

namespace {

// A short scratch votes equally for a small fan of angles around its true normal (a
// plateau); the peak is the plateau's first angle, so tests allow a couple of degrees.
int angle_distance(int a, int b) {
    const int d = std::abs(a - b) % 180;
    return std::min(d, 180 - d);
}

} // namespace

TEST_CASE("fixed-point tables hit the exact values at 0, 45, 90 and 135 degrees") {
    const auto& c = hough_cos();
    const auto& s = hough_sin();
    constexpr std::int32_t one = 1 << kHoughFracBits;
    CHECK(c[0] == one);
    CHECK(s[0] == 0);
    CHECK(c[90] == 0);
    CHECK(s[90] == one);
    CHECK(c[45] == s[45]);
    CHECK(c[135] == -s[135]);
    for (std::size_t k = 1; k < 90; ++k) {
        CHECK(c[k] == s[90 - k]); // cos(k) = sin(90 - k), exact after rounding too
    }
}

TEST_CASE("a map without fails has no line") {
    HoughTransform hough;
    CHECK(hough.run(synth::disc(30, 30)) == HoughLine{});
}

TEST_CASE("a horizontal scratch: normal near 90 degrees, every die votes") {
    const int n = 30;
    auto map = synth::disc(n, n);
    synth::paint(map, [](int r, int c) { return r == 10 && c >= 5 && c < 25; });
    HoughTransform hough;
    const auto line = hough.run(map);
    CHECK(angle_distance(line.angle, 90) <= 2);
    CHECK(line.votes == 20);
    // Row 10 of 30 sits at y = 29 - 20 = 9 half-dies: floor(9 / 2) = 4 dies above centre.
    CHECK(line.rho == 4);
    // line_dies is every on-wafer die in the peak's bin: the whole row, plus at a plateau
    // angle off 90 maybe a die of the next row at the far end.
    std::uint32_t in_bin = 0;
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            in_bin += on_wafer(map.at(r, c)) &&
                              hough_rho(2 * c - (n - 1), (n - 1) - 2 * r, line.angle) == line.rho
                          ? 1U
                          : 0U;
        }
    }
    CHECK(line.line_dies == in_bin);
    const auto row_dies = std::ranges::count_if(std::views::iota(0, n),
                                                [&](int c) { return on_wafer(map.at(10, c)); });
    CHECK(line.line_dies >= static_cast<std::uint32_t>(row_dies));
}

TEST_CASE("a vertical scratch: normal near 0 degrees") {
    auto map = synth::disc(40, 40);
    synth::paint(map, [](int r, int c) { return c == 30 && r >= 8 && r < 32; });
    HoughTransform hough;
    const auto line = hough.run(map);
    CHECK(angle_distance(line.angle, 0) <= 2);
    CHECK(line.votes == 24);
}

TEST_CASE("a diagonal scratch peaks at its true normal") {
    // Dies (r, c) = (i, i) run down and to the right: direction -45, normal 45. Kept inside
    // the disc (die (5, 5) of a 40x40 grid is already off the wafer).
    auto map = synth::disc(40, 40);
    synth::paint(map, [](int r, int c) { return r == c && r >= 7 && r < 33; });
    HoughTransform hough;
    const auto line = hough.run(map);
    CHECK(angle_distance(line.angle, 45) <= 2);
    CHECK(line.votes == 26);
}

TEST_CASE("votes are bounded by fails and by the dies on the line") {
    const auto per_mille = GENERATE(20U, 200U, 700U);
    const auto map = synth::random_map(30, 30, per_mille, 4);
    HoughTransform hough;
    const auto line = hough.run(map);
    const auto fails = static_cast<std::uint32_t>(
        std::ranges::count_if(map.view().bins(), [](auto b) { return is_fail(b); }));
    CHECK(line.votes <= fails);
    CHECK(line.votes <= line.line_dies);
    // Each fail votes exactly once per angle.
    const auto acc = hough.accumulator();
    const auto bins = static_cast<std::size_t>(hough.rho_bins());
    for (std::size_t k = 0; k < kHoughAngles; ++k) {
        CHECK(std::accumulate(acc.begin() + static_cast<std::ptrdiff_t>(k * bins),
                              acc.begin() + static_cast<std::ptrdiff_t>((k + 1) * bins),
                              0U) == fails);
    }
}

TEST_CASE("a scratch stands out from a sparse random field") {
    auto noise = synth::random_map(40, 40, 30, 11);
    auto scratch = noise;
    synth::paint(scratch, [](int r, int c) { return c == 12 + r / 3 && r >= 4 && r < 36; });
    HoughTransform hough;
    const auto a = hough.run(noise);
    const auto b = hough.run(scratch);
    const double density_a = static_cast<double>(a.votes) / a.line_dies;
    const double density_b = static_cast<double>(b.votes) / b.line_dies;
    CHECK(b.votes >= 25);
    CHECK(density_b > 3 * density_a);
}
