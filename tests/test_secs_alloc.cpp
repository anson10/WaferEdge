// The codec's hot path allocates nothing: decoding a body, reading every value, and encoding
// into a buffer that has already grown. Its own binary, because counting allocations means
// replacing the global operator new (tests/support/alloc_counter.cpp).
#include "support/alloc_counter.hpp"
#include "support/synth.hpp"
#include "waferedge/gem/messages.hpp"
#include "waferedge/secs/encoder.hpp"
#include "waferedge/secs/item.hpp"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

using namespace waferedge;

TEST_CASE("the allocation counter sees allocations") {
    const auto before = alloc::count();
    auto p = std::make_unique<int>(1);
    CHECK(alloc::count() == before + 1);
}

TEST_CASE("decode, walk and encode allocate nothing once the buffer has grown") {
    const WaferMap map = synth::random_map(40, 40, 100, 1);
    std::vector<std::uint8_t> buffer;
    {
        secs::Encoder warm(buffer);
        gem::encode_wafer_report(warm, 1, "LOT-0042", 17, map);
    }
    const std::vector<std::uint8_t> received(buffer.begin(), buffer.end());

    // No Catch2 macros inside the counted region: they may allocate themselves.
    const auto before = alloc::count();
    std::uint64_t sum = 0;
    bool ok = true;
    for (std::uint32_t i = 0; i < 100; ++i) {
        auto report = gem::decode_wafer_report(received);
        ok = ok && report.has_value();
        if (!report) {
            break;
        }
        for (const auto bin : report->map.bins()) {
            sum += bin;
        }
        secs::Encoder e(buffer);
        gem::encode_wafer_report(e, i, report->lot, static_cast<std::uint32_t>(report->wafer),
                                 report->map);
        ok = ok && e.finish().has_value();
    }
    const auto allocations = alloc::count() - before;
    CHECK(ok);
    CHECK(allocations == 0);
    std::uint64_t expected = 0;
    const WaferMapView view = map;
    for (const auto bin : view.bins()) {
        expected += bin;
    }
    CHECK(sum == 100 * expected);
}
