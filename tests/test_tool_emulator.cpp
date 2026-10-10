// The tool emulator over TCP on 127.0.0.1 against a gem::Host: it replays the WaferLens
// sample, the host holds a lot after its first wafer, and no later wafer of that lot arrives.
#include "waferedge/emulator/tool.hpp"
#include "waferedge/gem/host.hpp"

#include <asio/co_spawn.hpp>
#include <asio/io_context.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <exception>
#include <optional>
#include <string>
#include <vector>

using namespace waferedge;
using namespace std::chrono_literals;

TEST_CASE("the emulator replays the sample and honours a HOLD", "[emulator][net]") {
    auto set = MapSet::load(WAFEREDGE_TEST_DATA "/waferlens_sample.wmap");
    REQUIRE(set.has_value());
    asio::io_context io;
    asio::ip::tcp::acceptor acceptor(io, {asio::ip::address_v4::loopback(), 0});

    emulator::ToolConfig config;
    config.replay.rate = 2000; // 24 wafers in ~12 ms
    std::vector<std::string> tool_log;
    emulator::Tool tool(io.get_executor(), set->records(), config,
                        [&](std::string_view line) { tool_log.emplace_back(line); });

    // The host: holds LOT-000281 as soon as its first wafer arrives.
    const std::string target = "LOT-000281";
    gem::Host host;
    std::vector<std::string> received; // "lot/wafer", in arrival order
    bool hold_sent = false;
    bool hold_acked = false;
    std::vector<std::string> after_hold;
    std::optional<hsms::Session> session;
    std::optional<gem::SessionLink> link;
    hsms::Config active;
    active.role = hsms::Role::active;
    session.emplace(io.get_executor(), active, [&](hsms::Session& s, const hsms::Event& e) {
        host.on_hsms(e, *link, hsms::Clock::now());
        while (auto g = host.poll()) {
            if (const auto* w = std::get_if<gem::WaferReported>(&*g)) {
                const std::string lot(w->report.lot);
                received.push_back(lot + "/" + std::to_string(w->report.wafer));
                if (hold_acked) {
                    after_hold.push_back(lot);
                }
                if (lot == target && !hold_sent) {
                    hold_sent = true;
                    REQUIRE(host.command(*link, {gem::LotAction::hold, target}).has_value());
                }
            } else if (const auto* r = std::get_if<gem::ReplyReceived>(&*g);
                       r != nullptr && r->function == 42) {
                CHECK(static_cast<gem::Hcack>(r->code) == gem::Hcack::done);
                hold_acked = true;
            } else if (std::holds_alternative<gem::NotCommunicating>(*g)) {
                s.stop(); // the tool separated: it is done
            }
        }
    });
    link.emplace(*session);

    const auto rethrow = [](const std::exception_ptr& e) {
        if (e) {
            std::rethrow_exception(e);
        }
    };
    asio::co_spawn(io, tool.run(acceptor), rethrow);
    asio::co_spawn(io, session->run_active(acceptor.local_endpoint()), rethrow);
    io.run_for(20s);

    CHECK(io.stopped());
    const auto& stats = tool.replay().stats();
    CHECK(stats.holds == 1);
    CHECK(stats.sent + stats.withheld == set->size());
    CHECK(received.size() == stats.sent);
    CHECK(tool.acknowledged() == stats.sent);
    CHECK(hold_acked);
    // Nothing of the held lot after the HCACK: the tool replied after its last report of it.
    CHECK(std::ranges::count(after_hold, target) == 0);
    CHECK(tool.replay().hold_time(target).has_value());
    CHECK(std::ranges::find(tool_log, "HOLD LOT-000281: HCACK 0") != tool_log.end());
}
