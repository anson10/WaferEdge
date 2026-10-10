// The latency histogram: bucket arithmetic at every power of two, percentiles against the
// exact ones from sorted samples, merging.
#include "waferedge/pipeline/histogram.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using waferedge::pipeline::LatencyHistogram;
using H = LatencyHistogram;

TEST_CASE("histogram: buckets are contiguous and each holds values within 1/128", "[histogram]") {
    CHECK(H::bucket_count() == 7424);
    // Below 256 every value has its own bucket.
    for (std::uint64_t v = 0; v < 256; ++v) {
        REQUIRE(H::index(v) == v);
        REQUIRE(H::upper(H::index(v)) == v);
    }
    // Around every power of two: a value's bucket's upper edge is >= the value, within
    // 1/128 of it, and consecutive values never skip a bucket.
    for (unsigned bit = 8; bit < 64; ++bit) {
        const std::uint64_t p = std::uint64_t{1} << bit;
        for (const std::uint64_t v : {p - 2, p - 1, p, p + 1, p + p / 3, p + p / 2}) {
            const auto i = H::index(v);
            REQUIRE(i < H::bucket_count());
            REQUIRE(H::upper(i) >= v);
            REQUIRE(static_cast<double>(H::upper(i) - v) <= static_cast<double>(v) / 128.0);
            REQUIRE(H::index(v + 1) - i <= 1);
        }
    }
    CHECK(H::index(~std::uint64_t{0}) == H::bucket_count() - 1);
    CHECK(H::upper(H::bucket_count() - 1) == ~std::uint64_t{0});
}

TEST_CASE("histogram: percentiles match sorted samples to within one bucket", "[histogram]") {
    std::mt19937_64 rng(7);
    // Log-uniform from ~100 ns to ~100 ms, the range the pipeline lives in.
    std::vector<std::uint64_t> values(100'000);
    for (auto& v : values) {
        const double e = 2.0 + 6.0 * static_cast<double>(rng() >> 11) / 9007199254740992.0;
        v = static_cast<std::uint64_t>(std::pow(10.0, e));
    }
    H h;
    for (const auto v : values) {
        h.record(v);
    }
    std::ranges::sort(values);
    CHECK(h.count() == values.size());
    CHECK(h.min() == values.front());
    CHECK(h.max() == values.back());
    for (const double p : {0.01, 0.25, 0.5, 0.9, 0.99, 0.999, 0.9999}) {
        // The exact percentile: the value at rank ceil(p * n).
        const auto rank =
            static_cast<std::size_t>(std::ceil(p * static_cast<double>(values.size())));
        const std::uint64_t exact = values[rank - 1];
        const std::uint64_t got = h.percentile(p);
        INFO("p = " << p << ", exact " << exact << ", histogram " << got);
        CHECK(got >= exact); // never under-reports
        CHECK(static_cast<double>(got - exact) <= static_cast<double>(exact) / 128.0);
    }
    CHECK(h.percentile(1.0) == values.back());
}

TEST_CASE("histogram: small exact cases, empty, and merging", "[histogram]") {
    const H empty;
    CHECK(empty.percentile(0.5) == 0);
    CHECK(empty.min() == 0);
    CHECK(empty.max() == 0);

    H a;
    H b;
    for (std::uint64_t v = 1; v <= 100; ++v) {
        (v % 2 == 0 ? a : b).record(v);
    }
    a.merge(b);
    CHECK(a.count() == 100);
    CHECK(a.percentile(0.5) == 50); // exact below 256
    CHECK(a.percentile(0.99) == 99);
    CHECK(a.percentile(0.001) == 1);
    CHECK(a.min() == 1);
    CHECK(a.max() == 100);
    // A single large value: the percentile is capped at the exact maximum.
    H one;
    one.record(1'000'001);
    CHECK(one.percentile(0.5) == 1'000'001);
}
