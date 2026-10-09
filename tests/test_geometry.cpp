#include "support/synth.hpp"
#include "waferedge/geometry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>

using namespace waferedge;

TEST_CASE("zones and sectors of a 3x3 grid, by hand") {
    // Doubled coordinates: centre (0,0), corners (+-2,+-2), edge midpoints (0,+-2) / (+-2,0).
    // rho^2 = (X^2 + Y^2) / 9, zone = floor(5 rho^2).
    CHECK(zone_of(1, 1, 3, 3) == 0); // centre
    CHECK(zone_of(0, 1, 3, 3) == 2); // 5 * 4/9 = 2.2
    CHECK(zone_of(0, 0, 3, 3) == 4); // 5 * 8/9 = 4.4

    CHECK(sector_of(1, 1, 3, 3) == 0); // centre, by convention
    CHECK(sector_of(1, 2, 3, 3) == 0); // right: 0 degrees
    CHECK(sector_of(0, 2, 3, 3) == 1); // top right: exactly 45 degrees opens sector 1
    CHECK(sector_of(0, 1, 3, 3) == 2); // up: 90
    CHECK(sector_of(0, 0, 3, 3) == 3); // top left: 135
    CHECK(sector_of(1, 0, 3, 3) == 4); // left: 180
    CHECK(sector_of(2, 0, 3, 3) == 5); // bottom left: 225
    CHECK(sector_of(2, 1, 3, 3) == 6); // down: 270
    CHECK(sector_of(2, 2, 3, 3) == 7); // bottom right: 315
}

TEST_CASE("equal-width rings, by hand") {
    // 21x21: doubled coordinates step by 2, rho = sqrt(x^2 + y^2) / 21.
    CHECK(ring_of(10, 10, 21, 21) == 0);        // centre
    CHECK(ring_of(10, 11, 21, 21) == 0);        // rho = 2/21 = 0.095
    CHECK(ring_of(10, 12, 21, 21) == 1);        // 4/21 = 0.19
    CHECK(ring_of(10, 13, 21, 21) == 2);        // 6/21 = 0.29
    CHECK(ring_of(10, 20, 21, 21) == 9);        // 20/21 = 0.95
    CHECK(ring_of(0, 0, 21, 21) == kRings - 1); // corner, rho > 1: last ring
    // 10x10: the die at doubled (1, 1) has rho = sqrt(2)/10 = 0.141.
    CHECK(ring_of(4, 5, 10, 10) == 1);
}

TEST_CASE("an even grid has no centre die; the four middle dies take one octant each") {
    // 4x4: the middle dies sit at doubled (+-1, +-1), exactly on the diagonals.
    CHECK(sector_of(1, 2, 4, 4) == 1); // (1, 1): 45 degrees
    CHECK(sector_of(1, 1, 4, 4) == 3); // (-1, 1): 135
    CHECK(sector_of(2, 1, 4, 4) == 5); // (-1, -1): 225
    CHECK(sector_of(2, 2, 4, 4) == 7); // (1, -1): 315
    CHECK(zone_of(1, 1, 4, 4) == 0);
}

TEST_CASE("non-square grids are normalised per axis (dies aren't square)") {
    // 25 rows x 27 cols: the top and left extremes are both near rho = 1, so both edge zone.
    CHECK(zone_of(0, 13, 25, 27) == kZones - 1);
    CHECK(zone_of(12, 0, 25, 27) == kZones - 1);
    CHECK(zone_of(12, 13, 25, 27) == 0);
    CHECK(sector_of(0, 13, 25, 27) == 2); // straight up
    CHECK(sector_of(12, 0, 25, 27) == 4); // straight left
}

TEST_CASE("rotating a square grid by 90 degrees keeps zones and moves sectors by two") {
    const int n = GENERATE(3, 4, 7, 24, 25, 30, 40, 64);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            // synth::rotate90 moves die (r, c) to (n-1-c, r).
            const int rr = n - 1 - c;
            const int rc = r;
            INFO("n=" << n << " die (" << r << "," << c << ")");
            CHECK(zone_of(rr, rc, n, n) == zone_of(r, c, n, n));
            CHECK(ring_of(rr, rc, n, n) == ring_of(r, c, n, n));
            if (!(2 * r == n - 1 && 2 * c == n - 1)) { // the centre die has no direction
                CHECK(sector_of(rr, rc, n, n) == (sector_of(r, c, n, n) + 2) % kSectors);
            }
        }
    }
}

TEST_CASE("zones are equal-area: on a large disc each holds about a fifth of the dies") {
    const auto disc = synth::disc(200, 200);
    const Geometry g(200, 200);
    std::array<int, kZones> dies{};
    int total = 0;
    for (std::size_t i = 0; i < disc.view().bins().size(); ++i) {
        if (on_wafer(disc.view().bins()[i])) {
            ++dies[g.zones()[i]];
            ++total;
        }
    }
    for (int z = 0; z < kZones; ++z) {
        INFO("zone " << z);
        CHECK(dies[static_cast<std::size_t>(z)] * kZones > total * 97 / 100);
        CHECK(dies[static_cast<std::size_t>(z)] * kZones < total * 103 / 100);
    }
}

TEST_CASE("a Geometry table matches zone_of and sector_of") {
    const Geometry g(24, 30);
    REQUIRE(g.zones().size() == 24U * 30U);
    for (int r = 0; r < 24; ++r) {
        for (int c = 0; c < 30; ++c) {
            const auto i = static_cast<std::size_t>(r * 30 + c);
            CHECK(g.zones()[i] == zone_of(r, c, 24, 30));
            CHECK(g.sectors()[i] == sector_of(r, c, 24, 30));
            CHECK(g.rings()[i] == ring_of(r, c, 24, 30));
        }
    }
}

TEST_CASE("the geometry cache builds each shape once") {
    GeometryCache cache;
    const Geometry& a = cache.get(30, 30);
    const Geometry& b = cache.get(24, 24);
    CHECK(&cache.get(30, 30) == &a);
    CHECK(&cache.get(24, 24) == &b);
    CHECK(cache.size() == 2);
    CHECK(a.rows() == 30);
    CHECK(b.cols() == 24);
}
