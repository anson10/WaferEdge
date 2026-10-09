#include "support/synth.hpp"
#include "waferedge/map_file.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <ranges>

using namespace waferedge;

namespace {

const std::filesystem::path kSample = WAFEREDGE_TEST_DATA "/waferlens_sample.wmap";

std::vector<std::uint8_t> sample_image() {
    const std::vector maps = {synth::random_map(24, 24, 100, 1), synth::random_map(25, 27, 300, 2)};
    const std::array records = {
        WaferRecord{1, 10, 1'700'000'000'000'000, Pattern::scratch, Split::train, maps[0]},
        WaferRecord{2, 11, 0, Pattern::unknown, Split::unsplit, maps[1]},
    };
    return serialize(records);
}

} // namespace

TEST_CASE("serialize then parse gives back the records and the maps") {
    const std::vector maps = {synth::random_map(24, 24, 100, 1), synth::random_map(25, 27, 300, 2),
                              WaferMap(1, 1, 1)};
    const std::array records = {
        WaferRecord{1, 10, 1'700'000'000'000'000, Pattern::scratch, Split::train, maps[0]},
        WaferRecord{-2, 11, 0, Pattern::unknown, Split::unsplit, maps[1]},
        WaferRecord{3, 12, 5, Pattern::none, Split::test, maps[2]},
    };
    auto set = MapSet::parse(serialize(records));
    REQUIRE(set.has_value());
    REQUIRE(set->size() == records.size());
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& a = records[i];
        const auto& b = set->records()[i];
        CHECK(b.wafer_id == a.wafer_id);
        CHECK(b.lot_id == a.lot_id);
        CHECK(b.tested_at_us == a.tested_at_us);
        CHECK(b.truth == a.truth);
        CHECK(b.split == a.split);
        CHECK(b.map.rows() == a.map.rows());
        CHECK(b.map.cols() == a.map.cols());
        CHECK(std::ranges::equal(b.map.bins(), a.map.bins()));
    }
}

TEST_CASE("views stay valid when a MapSet is moved") {
    auto set = MapSet::parse(sample_image());
    REQUIRE(set.has_value());
    const auto* before = set->records()[1].map.bins().data();
    MapSet moved = std::move(*set);
    CHECK(moved.records()[1].map.bins().data() == before);
    CHECK(moved.records()[1].map.rows() == 25);
}

TEST_CASE("the WaferLens fixture written by tools/export_maps.py loads") {
    auto set = MapSet::load(kSample);
    REQUIRE(set.has_value());
    REQUIRE(set->size() == 24);
    const auto& first = set->records()[0];
    CHECK(first.wafer_id == 14);
    CHECK(first.lot_id == 1);
    CHECK(first.tested_at_us == 1'768'034'963'913'034);
    CHECK(first.truth == Pattern::none);
    CHECK(first.split == Split::unsplit);
    CHECK(first.map.rows() == 30);
    CHECK(first.map.cols() == 30);
    // Three wafers of each WaferLens pattern (it has no near_full).
    for (auto p : {Pattern::none, Pattern::center, Pattern::donut, Pattern::edge_loc,
                   Pattern::edge_ring, Pattern::loc, Pattern::random, Pattern::scratch}) {
        INFO(pattern_name(p));
        CHECK(std::ranges::count(set->records(), p, &WaferRecord::truth) == 3);
    }
    // Maps are in tested_at order.
    CHECK(std::ranges::is_sorted(set->records(), {}, &WaferRecord::tested_at_us));
}

TEST_CASE("broken files are rejected with a reason, never read out of bounds") {
    const auto good = sample_image();
    auto reject = [](std::vector<std::uint8_t> image) {
        auto set = MapSet::parse(std::move(image));
        REQUIRE_FALSE(set.has_value());
        CHECK_FALSE(set.error().empty());
        return set.error();
    };

    SECTION("empty and short") {
        reject({});
        reject(std::vector<std::uint8_t>(good.begin(), good.begin() + 31));
    }
    SECTION("bad magic") {
        auto image = good;
        image[0] = 'X';
        CHECK(reject(image).find("magic") != std::string::npos);
    }
    SECTION("unknown version") {
        auto image = good;
        image[8] = 2;
        CHECK(reject(image).find("version") != std::string::npos);
    }
    SECTION("truncated bin section") {
        reject(std::vector<std::uint8_t>(good.begin(), good.end() - 1));
    }
    SECTION("a record count that would overflow") {
        auto image = good;
        const std::uint64_t huge = ~std::uint64_t{0} / 16;
        std::memcpy(image.data() + 16, &huge, sizeof huge);
        reject(image);
    }
    SECTION("a map offset past the bins") {
        auto image = good;
        const std::uint64_t offset = 1'000'000;
        std::memcpy(image.data() + 32 + 32 + 16, &offset, sizeof offset); // record 1
        CHECK(reject(image).find("record 1") != std::string::npos);
    }
    SECTION("a truth code that isn't a pattern") {
        auto image = good;
        image[32 + 28] = 42;
        reject(image);
    }
}

TEST_CASE("loading a missing file is an error") {
    auto set = MapSet::load("/nonexistent/maps.wmap");
    REQUIRE_FALSE(set.has_value());
    CHECK(set.error().find("cannot open") != std::string::npos);
}
