// The closed-loop replay: the excursion and alarm files, a hand-built excursion whose hold
// and withheld wafers are known, the latency against a lot's wafer gap, and the replay
// against a live run of the tool emulator and the edge host over TCP.
#include "support/synth.hpp"
#include "waferedge/emulator/tool.hpp"
#include "waferedge/pipeline/closed_loop.hpp"
#include "waferedge/pipeline/edge_host.hpp"

#include <asio/co_spawn.hpp>
#include <asio/io_context.hpp>
#include <asio/steady_timer.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <vector>

using namespace waferedge;
using namespace waferedge::pipeline;
using namespace std::chrono_literals;

namespace {

constexpr std::int64_t kMinute = 60'000'000;

std::filesystem::path write_temp(const std::string& name, const std::string& text) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream(path) << text;
    return path;
}

// Three lots of 10 wafers, sorted one per minute lot after lot. Lot 2's wafers 3..9 show
// excursion 7 (center); the "detector" sees them all except wafer 4.
struct Scenario {
    WaferMap map = synth::disc(10, 10);
    std::vector<WaferRecord> records;
    std::vector<Pattern> verdicts;
    ExcursionTruth truth;

    Scenario() {
        truth.excursions.push_back(
            {.id = 7, .pattern = Pattern::center, .start_us = 0, .end_us = 100 * kMinute});
        std::int32_t id = 0;
        for (std::int32_t lot = 1; lot <= 3; ++lot) {
            for (int w = 0; w < 10; ++w) {
                const bool shows = lot == 2 && w >= 3;
                records.push_back({id, lot, 1000 * kMinute + id * kMinute,
                                   shows ? Pattern::center : Pattern::none, Split::unsplit,
                                   map.view()});
                verdicts.push_back(shows && w != 4 ? Pattern::center : Pattern::none);
                if (shows) {
                    truth.excursion_of[id] = 7;
                }
                ++id;
            }
        }
    }
};

} // namespace

TEST_CASE("closed loop: the excursion and alarm files", "[closedloop]") {
    const auto good = write_temp("we_exc_good.csv", "# comment\n"
                                                    "excursion,7,center,100,200\n"
                                                    "excursion,9,scratch,300,400\n"
                                                    "wafer,11,7\n"
                                                    "wafer,12,9\n");
    const auto truth = load_excursions(good);
    REQUIRE(truth.has_value());
    REQUIRE(truth->excursions.size() == 2);
    CHECK(truth->excursions[1].pattern == Pattern::scratch);
    CHECK(truth->excursions[0].start_us == 100);
    CHECK(truth->excursion_of.at(12) == 9);

    CHECK_FALSE(load_excursions(write_temp("we_exc_bad.csv", "excursion,7,blob,1,2\n")));
    CHECK_FALSE(load_excursions(write_temp("we_exc_short.csv", "wafer,1\n")));
    CHECK_FALSE(load_excursions(write_temp("we_exc_empty.csv", "# nothing\n")));
    CHECK_FALSE(load_excursions("/nonexistent/excursions.csv"));

    const auto alarms =
        load_waferlens_alarms(WAFEREDGE_SOURCE_DIR "/config/waferlens_pattern_alarms.csv");
    REQUIRE(alarms.has_value());
    CHECK(alarms->size() == 13);
    std::size_t edge_loc = 0;
    for (const auto& a : *alarms) {
        edge_loc += a.pattern == Pattern::edge_loc ? a.pattern_wafers : 0;
    }
    CHECK(edge_loc == 634); // WaferLens demo: 634 edge-loc wafers
}

TEST_CASE("closed loop: a known excursion, its hold and the wafers it withholds", "[closedloop]") {
    const Scenario s;
    ClosedLoopConfig config;
    config.decision = {.k = 3, .window = 5, .lots = 8};
    config.latency_us = 5'000; // well under the one-minute gap
    const auto r = replay_closed_loop(s.records, s.verdicts, s.truth, config);

    // Lot 2: wafers 3, 5, 6 seen as center -> the third (wafer 6, id 16) completes 3 of 5.
    REQUIRE(r.excursions.size() == 1);
    const auto& e = r.excursions[0];
    CHECK(e.wafers == 7);
    CHECK(e.first_sorted_us == 1000 * kMinute + 13 * kMinute); // an optional: empty fails
    CHECK(e.hold_us == 1000 * kMinute + 16 * kMinute + 5'000);
    CHECK(e.lots_held == 1);
    CHECK(e.sorted_before_hold == 4); // wafers 3, 4, 5, 6
    CHECK(e.sorted == 4);             // 7, 8, 9 withheld
    CHECK(r.holds == 1);
    CHECK(r.false_holds == 0);
    CHECK(r.wafers_withheld == 3);
    CHECK(r.wafers_sorted == 27);
    CHECK(r.lots == 3);
    CHECK(r.min_lot_gap_us == kMinute);
}

TEST_CASE("closed loop: a hold slower than the gap lets the next wafer through", "[closedloop]") {
    const Scenario s;
    ClosedLoopConfig config;
    config.decision = {.k = 3, .window = 5, .lots = 8};
    config.latency_us = 90'000'000; // 1.5 minutes: wafer 7 is sorted before the hold lands
    const auto r = replay_closed_loop(s.records, s.verdicts, s.truth, config);
    CHECK(r.excursions[0].sorted == 5);
    CHECK(r.excursions[0].sorted_before_hold == 5);
    CHECK(r.wafers_withheld == 2);
}

TEST_CASE("closed loop: holds without an excursion are false holds", "[closedloop]") {
    Scenario s;
    // The detector sees a pattern on lot 3's first three wafers, which show none.
    for (std::size_t i = 20; i < 23; ++i) {
        s.verdicts[i] = Pattern::scratch;
    }
    ClosedLoopConfig config;
    config.decision = {.k = 3, .window = 5, .lots = 8};
    const auto r = replay_closed_loop(s.records, s.verdicts, s.truth, config);
    CHECK(r.holds == 2);
    CHECK(r.false_holds == 1);
    CHECK(r.wafers_withheld == 3 + 7);
}

TEST_CASE("closed loop: the replay withholds what the live loop withholds", "[closedloop][net]") {
    // The WaferLens sample, k = 1: the live run (emulator + edge host over TCP, one wafer per
    // 20 ms, holds landing in ~1 ms) against the replay of the same verdicts.
    auto set = MapSet::load(WAFEREDGE_TEST_DATA "/waferlens_sample.wmap");
    REQUIRE(set.has_value());
    const auto classifier = RuleClassifier::load(WAFEREDGE_SOURCE_DIR "/config/rules.txt");
    REQUIRE(classifier.has_value());
    const DecisionConfig decision{.k = 1, .window = 1, .lots = 64};

    // Replay: every patterned wafer as one excursion per pattern.
    ExcursionTruth truth;
    for (const auto& r : set->records()) {
        if (r.truth != Pattern::none && r.truth != Pattern::unknown) {
            const int id = static_cast<int>(r.truth);
            if (std::ranges::none_of(truth.excursions,
                                     [&](const Excursion& e) { return e.id == id; })) {
                truth.excursions.push_back(
                    {.id = id, .pattern = r.truth, .start_us = 0, .end_us = 0});
            }
            truth.excursion_of[r.wafer_id] = id;
        }
    }
    RulesDetector rules(*classifier);
    const auto verdicts = classify_all(set->records(), rules);
    REQUIRE(verdicts.has_value());
    ClosedLoopConfig config;
    config.decision = decision;
    config.latency_us = 0;
    const auto replayed = replay_closed_loop(set->records(), *verdicts, truth, config);

    // Live.
    asio::io_context io;
    asio::ip::tcp::acceptor acceptor(io, {asio::ip::address_v4::loopback(), 0});
    emulator::ToolConfig tool_config;
    tool_config.replay.rate = 50;
    tool_config.linger = 300ms;
    emulator::Tool tool(io.get_executor(), set->records(), tool_config);
    EdgeHostConfig edge_config;
    edge_config.edge.decision = decision;
    EdgeHost edge(io.get_executor(), *classifier, edge_config);
    const auto rethrow = [](const std::exception_ptr& e) {
        if (e) {
            std::rethrow_exception(e);
        }
    };
    asio::co_spawn(io, tool.run(acceptor), rethrow);
    asio::co_spawn(io, edge.run(acceptor.local_endpoint()), rethrow);
    io.run_for(30s);
    REQUIRE(io.stopped());

    CHECK(tool.replay().stats().withheld == replayed.wafers_withheld);
    CHECK(tool.replay().stats().sent == replayed.wafers_sorted);
    CHECK(edge.core().stats().holds == replayed.holds);
    CHECK(replayed.holds > 0);
}
