#include "support/synth.hpp"
#include "waferedge/randomness.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <bit>
#include <cmath>
#include <utility>
#include <vector>

using namespace waferedge;
using Catch::Approx;

namespace {

// Exact mean and variance of BB over every way to place `fails` fails on the map's on-wafer
// dies: the null distribution itself, enumerated, against the closed form.
std::pair<double, double> enumerate_null(const WaferMap& base, int fails) {
    std::vector<int> dies;
    for (int i = 0; i < base.rows() * base.cols(); ++i) {
        if (on_wafer(base.view().bins()[static_cast<std::size_t>(i)])) {
            dies.push_back(i);
        }
    }
    const int n = static_cast<int>(dies.size());
    double sum = 0;
    double sum_sq = 0;
    double count = 0;
    for (std::uint32_t mask = 0; mask < (1U << n); ++mask) {
        if (std::popcount(mask) != fails) {
            continue;
        }
        WaferMap m = base;
        for (int k = 0; k < n; ++k) {
            if ((mask >> k) & 1U) {
                m.set(dies[static_cast<std::size_t>(k)] / base.cols(),
                      dies[static_cast<std::size_t>(k)] % base.cols(), 2);
            }
        }
        const double bb = join_count(m).fail_joins;
        sum += bb;
        sum_sq += bb * bb;
        count += 1;
    }
    const double mean = sum / count;
    return {mean, sum_sq / count - mean * mean};
}

} // namespace

TEST_CASE("the closed-form mean and variance match exact enumeration") {
    // A 4x4 grid and a small disc with off-wafer corners (degrees vary), several fail counts.
    const auto grid = WaferMap(4, 4, 1);
    const auto disc = synth::disc(5, 5);
    const auto fails = GENERATE(2, 3, 4, 7);
    for (const auto* base : {&grid, &disc}) {
        auto j = join_count(*base);
        j.fails = static_cast<std::uint32_t>(fails);
        const auto [mean, var] = enumerate_null(*base, fails);
        INFO("dies " << j.dies << " fails " << fails);
        CHECK(expected_fail_joins(j) == Approx(mean).epsilon(1e-12));
        CHECK(fail_joins_variance(j) == Approx(var).epsilon(1e-9));
    }
}

TEST_CASE("joins and join pairs of a full 3x3 grid") {
    const auto j = join_count(WaferMap(3, 3, 1));
    CHECK(j.dies == 9);
    CHECK(j.joins == 12);
    // Degrees: 4 corners 2, 4 edges 3, centre 4 -> 4*1 + 4*3 + 6 = 22 pairs.
    CHECK(j.join_pairs == 22);
    CHECK(j.fail_joins == 0);
}

TEST_CASE("off-wafer dies are not part of the join graph") {
    WaferMap map(3, 3, 1);
    map.set(1, 1, 0);
    const auto j = join_count(map);
    CHECK(j.dies == 8);
    CHECK(j.joins == 8); // the ring around the hole
}

TEST_CASE("clustered fields score high, a checkerboard low, random near zero") {
    const int n = 40;
    SECTION("a solid blob") {
        auto map = synth::disc(n, n);
        synth::paint(map, [](int r, int c) { return r >= 15 && r < 25 && c >= 15 && c < 25; });
        CHECK(join_count_z(join_count(map)) > 10.0);
    }
    SECTION("a checkerboard: fails never touch") {
        auto map = synth::disc(n, n);
        synth::paint(map, [](int r, int c) { return (r + c) % 2 == 0; });
        const auto j = join_count(map);
        CHECK(j.fail_joins == 0);
        CHECK(join_count_z(j) < -10.0);
    }
    SECTION("random fields: z over many seeds averages about 0") {
        double sum = 0;
        double sum_sq = 0;
        const int runs = 200;
        for (int seed = 0; seed < runs; ++seed) {
            const double z = join_count_z(
                join_count(synth::random_map(n, n, 150, static_cast<std::uint32_t>(seed))));
            sum += z;
            sum_sq += z * z;
        }
        // sprinkle draws fails independently (binomial), not a fixed count; the test
        // conditions on the drawn count, so z is still standardised: mean ~0, sd ~1.
        CHECK(std::abs(sum / runs) < 0.25);
        CHECK(std::sqrt(sum_sq / runs) == Approx(1.0).margin(0.2));
    }
}

TEST_CASE("the test is undefined, and reports 0, without enough fails or variance") {
    CHECK(join_count_z(join_count(synth::disc(10, 10))) == 0.0);
    auto one = synth::disc(10, 10);
    one.set(5, 5, 2);
    CHECK(join_count_z(join_count(one)) == 0.0);
    auto all = synth::disc(10, 10);
    synth::paint(all, [](int, int) { return true; });
    CHECK(join_count_z(join_count(all)) == 0.0);
}
