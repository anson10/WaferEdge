// The tool emulator's replay on a fake clock: the constant-rate schedule, catching up when
// late, holds that skip a lot's wafers, releases, pauses.
#include "support/synth.hpp"
#include "waferedge/emulator/replay.hpp"

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace waferedge;
using namespace waferedge::emulator;
using namespace std::chrono_literals;

namespace {

const TimePoint t0 = TimePoint{} + 1h;

// Six wafers: lot 1 (wafers 10, 11, 12), lot 2 (20, 21), lot 1 again (13); listed out of
// tested_at order on purpose.
struct Fixture {
    WaferMap map = synth::random_map(4, 4, 100, 1);
    std::vector<WaferRecord> records;
    Fixture() {
        records.reserve(6); // also keeps GCC 13 -O3 from a false -Wstringop-overflow on regrowth
        const auto add = [&](std::int32_t wafer, std::int32_t lot, std::int64_t t) {
            records.push_back({wafer, lot, t, Pattern::unknown, Split::unsplit, map});
        };
        add(12, 1, 3);
        add(10, 1, 1);
        add(20, 2, 4);
        add(11, 1, 2);
        add(13, 1, 6);
        add(21, 2, 5);
    }
};

std::vector<std::int32_t> drain(Replay& r, TimePoint now) {
    std::vector<std::int32_t> sent;
    while (auto due = r.take(now)) {
        sent.push_back(due->record->wafer_id);
    }
    return sent;
}

} // namespace

TEST_CASE("wafers come in tested_at order, one per slot", "[emulator]") {
    Fixture f;
    Replay r(f.records, {.rate = 10.0});
    CHECK_FALSE(r.next_due().has_value()); // not started
    r.start(t0);
    CHECK(r.next_due() == t0);
    auto first = r.take(t0);
    REQUIRE(first.has_value());
    CHECK(first->record->wafer_id == 10);
    CHECK(first->lot == "LOT-000001");
    CHECK(first->scheduled == t0);
    CHECK(r.next_due() == t0 + 100ms);
    CHECK_FALSE(r.take(t0 + 100ms - 1ns).has_value());
    CHECK(r.take(t0 + 100ms)->record->wafer_id == 11);
}

TEST_CASE("a late sender catches up at the scheduled times", "[emulator]") {
    Fixture f;
    Replay r(f.records, {.rate = 10.0});
    r.start(t0);
    // 250 ms late: slots 0, 1, 2 are all due; their scheduled times stay on the grid.
    std::vector<TimePoint> scheduled;
    while (auto due = r.take(t0 + 250ms)) {
        scheduled.push_back(due->scheduled);
    }
    CHECK(scheduled == std::vector{t0, t0 + 100ms, t0 + 200ms});
    CHECK(r.next_due() == t0 + 300ms);
}

TEST_CASE("HOLD skips the lot's remaining wafers and counts them", "[emulator]") {
    Fixture f;
    Replay r(f.records, {.rate = 10.0});
    r.start(t0);
    CHECK(drain(r, t0) == std::vector<std::int32_t>{10});
    CHECK(r.hold("LOT-000001", t0 + 50ms) == LotResult::done);
    CHECK(r.hold("LOT-000001", t0 + 60ms) == LotResult::already);
    CHECK(r.hold("LOT-999999", t0) == LotResult::unknown);
    CHECK(r.held("LOT-000001"));
    CHECK(r.hold_time("LOT-000001") == t0 + 50ms);

    // Lot 1's 11 and 12 are skipped: lot 2's wafers take their slots, then 13 is skipped too.
    CHECK(drain(r, t0 + 1s) == std::vector<std::int32_t>{20, 21});
    CHECK(r.finished());
    CHECK(r.stats().sent == 3);
    CHECK(r.stats().withheld == 3);
    CHECK(r.withheld(1)); // wafer 11, in tested_at order
    CHECK_FALSE(r.scheduled(1).has_value());
    CHECK(r.scheduled(3) == t0 + 100ms); // wafer 20 took the second slot
}

TEST_CASE("RELEASE lets the lot's later wafers through", "[emulator]") {
    Fixture f;
    Replay r(f.records, {.rate = 10.0});
    r.start(t0);
    REQUIRE(r.hold("LOT-000001", t0) == LotResult::done);
    CHECK(drain(r, t0) == std::vector<std::int32_t>{20}); // 10, 11, 12 withheld
    CHECK(r.release("LOT-000001") == LotResult::done);
    CHECK(r.release("LOT-000001") == LotResult::already);
    CHECK(drain(r, t0 + 1s) == std::vector<std::int32_t>{21, 13});
    CHECK(r.stats().withheld == 3);
    CHECK(r.stats().releases == 1);
}

TEST_CASE("pause stops the schedule; start re-bases it", "[emulator]") {
    Fixture f;
    Replay r(f.records, {.rate = 10.0});
    r.start(t0);
    (void)r.take(t0);
    r.pause();
    CHECK_FALSE(r.take(t0 + 1h).has_value());
    CHECK_FALSE(r.next_due().has_value());
    r.start(t0 + 10s); // e.g. back on-line
    CHECK(r.next_due() == t0 + 10s);
    CHECK(r.take(t0 + 10s)->record->wafer_id == 11);
}

TEST_CASE("limit, and the real sample in tested_at order", "[emulator]") {
    Fixture f;
    Replay limited(f.records, {.rate = 10.0, .limit = 2});
    limited.start(t0);
    CHECK(drain(limited, t0 + 1h) == std::vector<std::int32_t>{10, 11});

    auto set = MapSet::load(WAFEREDGE_TEST_DATA "/waferlens_sample.wmap");
    REQUIRE(set.has_value());
    Replay r(set->records(), {.rate = 0}); // rate 0: as fast as possible
    r.start(t0);
    std::int64_t last = 0;
    std::size_t n = 0;
    while (auto due = r.take(t0)) {
        CHECK(due->record->tested_at_us >= last);
        last = due->record->tested_at_us;
        ++n;
    }
    CHECK(n == set->size());
    CHECK(r.lot_of(0) == "LOT-000001");
}
