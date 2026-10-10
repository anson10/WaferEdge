// The closed loop over TCP on 127.0.0.1: the tool emulator replays the WaferLens sample, the
// edge host (two analytics threads) classifies every wafer and holds each lot on its first
// wafer with a pattern (k = 1). Checked against the classifier run directly on the file.
#include "waferedge/emulator/tool.hpp"
#include "waferedge/pipeline/edge_host.hpp"

#include <asio/co_spawn.hpp>
#include <asio/io_context.hpp>
#include <asio/steady_timer.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <exception>
#include <functional>
#include <set>
#include <string>

using namespace waferedge;
using namespace std::chrono_literals;

TEST_CASE("edge host: the closed loop against the tool emulator", "[edge][net]") {
    auto set = MapSet::load(WAFEREDGE_TEST_DATA "/waferlens_sample.wmap");
    REQUIRE(set.has_value());
    const auto classifier = RuleClassifier::load(WAFEREDGE_SOURCE_DIR "/config/rules.txt");
    REQUIRE(classifier.has_value());

    // What k = 1 must hold: every lot with a wafer the classifier calls anything but none.
    // (A lot's first such wafer is always sent: wafers are withheld only after a hold.)
    std::set<std::string> expected;
    SignalExtractor extractor;
    for (const auto& r : set->records()) {
        if (classifier->classify(extractor.extract(r.map)).pattern != Pattern::none) {
            expected.insert(emulator::lot_name(r.lot_id));
        }
    }
    REQUIRE(!expected.empty());

    asio::io_context io;
    asio::ip::tcp::acceptor acceptor(io, {asio::ip::address_v4::loopback(), 0});
    emulator::ToolConfig tool_config;
    tool_config.replay.rate = 2000;
    // Not when every S6F12 is in: the host acknowledges a report before analysing it, so the
    // last lot's HOLD would find the tool gone. The watcher below stops it instead.
    tool_config.stop_when_done = false;
    emulator::Tool tool(io.get_executor(), set->records(), tool_config);

    pipeline::EdgeHostConfig config;
    config.workers = 2;
    config.edge.decision = {.k = 1, .window = 1, .lots = 64};
    pipeline::EdgeHost edge(io.get_executor(), *classifier, config);

    const auto rethrow = [](const std::exception_ptr& e) {
        if (e) {
            std::rethrow_exception(e);
        }
    };
    // Stops the tool once it has replayed everything, the edge host has every verdict and
    // every HOLD is answered; the edge host then stops because the tool left.
    const auto settled = [&] {
        return tool.replay().finished() && edge.core().idle() &&
               edge.core().stats().verdicts == tool.replay().stats().sent &&
               std::ranges::none_of(edge.core().holds(), [](const auto& h) {
                   return h.state == pipeline::HoldRecord::State::waiting ||
                          h.state == pipeline::HoldRecord::State::sent;
               });
    };
    asio::steady_timer watch(io);
    std::function<void()> check = [&] {
        if (settled()) {
            tool.stop();
            return;
        }
        watch.expires_after(1ms);
        watch.async_wait([&](const asio::error_code&) { check(); });
    };
    check();
    asio::co_spawn(io, tool.run(acceptor), rethrow);
    asio::co_spawn(io, edge.run(acceptor.local_endpoint()), rethrow);
    io.run_for(20s);
    REQUIRE(io.stopped()); // both finished: the tool when done, the edge host when it left

    const auto& replay = tool.replay().stats();
    const auto& s = edge.core().stats();
    CHECK(s.received == replay.sent);
    CHECK(s.submitted == s.received);
    CHECK(s.verdicts == s.submitted);
    CHECK(s.dropped == 0);
    // Both workers took wafers (round robin).
    CHECK(edge.analyzers()[0]->analysed() > 0);
    CHECK(edge.analyzers()[1]->analysed() > 0);

    std::set<std::string> held;
    for (const auto& h : edge.core().holds()) {
        CHECK(h.state == pipeline::HoldRecord::State::acked);
        CHECK(h.hcack == 0);
        CHECK(h.received <= h.analysed);
        CHECK(h.analysed <= h.decided);
        CHECK(h.decided <= h.sent);
        CHECK(h.sent <= h.acked);
        held.emplace(h.lot.view());
    }
    CHECK(held == expected);
    CHECK(replay.holds == expected.size());
}
