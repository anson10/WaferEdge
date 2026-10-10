// The whole conversation over TCP on 127.0.0.1: the equipment on a passive hsms::Session,
// the host on an active one; communication, on-line, a wafer report, a HOLD and its answer.
#include "support/synth.hpp"
#include "waferedge/gem/equipment.hpp"
#include "waferedge/gem/host.hpp"

#include <asio/co_spawn.hpp>
#include <asio/io_context.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <exception>
#include <string>
#include <vector>

using namespace waferedge;
using namespace waferedge::gem;
using namespace std::chrono_literals;

TEST_CASE("GEM over TCP: report a wafer, hold its lot", "[gem][net]") {
    asio::io_context io;
    asio::ip::tcp::acceptor acceptor(io, {asio::ip::address_v4::loopback(), 0});
    const WaferMap map = synth::random_map(40, 40, 100, 11);
    Equipment equipment(EquipmentConfig{});
    Host host;
    std::vector<std::string> log;
    std::vector<std::uint8_t> received_map;

    std::optional<hsms::Session> tool_session;
    std::optional<hsms::Session> host_session;
    std::optional<SessionLink> tool_link;
    std::optional<SessionLink> host_link;

    tool_session.emplace(
        io.get_executor(), hsms::Config{}, [&](hsms::Session&, const hsms::Event& e) {
            equipment.on_hsms(e, *tool_link, hsms::Clock::now());
            while (auto g = equipment.poll()) {
                if (std::holds_alternative<Communicating>(*g)) {
                    equipment.go_online(*tool_link);
                } else if (const auto* c = std::get_if<ControlStateChanged>(&*g);
                           c != nullptr && c->state == ControlState::online_remote) {
                    log.emplace_back("tool on-line");
                    REQUIRE(equipment.report_wafer(*tool_link, "LOT-0042", 17, map).has_value());
                } else if (const auto* cmd = std::get_if<LotCommandReceived>(&*g)) {
                    log.emplace_back("tool holds " + std::string(cmd->command.lot));
                    REQUIRE(equipment.answer(*tool_link, cmd->primary, Hcack::done).has_value());
                }
            }
        });
    hsms::Config active;
    active.role = hsms::Role::active;
    host_session.emplace(io.get_executor(), active, [&](hsms::Session&, const hsms::Event& e) {
        host.on_hsms(e, *host_link, hsms::Clock::now());
        while (auto g = host.poll()) {
            if (const auto* w = std::get_if<WaferReported>(&*g)) {
                log.emplace_back("host got wafer " + std::to_string(w->report.wafer) + " of " +
                                 std::string(w->report.lot));
                received_map.assign(w->report.map.bins().begin(), w->report.map.bins().end());
                REQUIRE(host.command(*host_link, {LotAction::hold, w->report.lot}).has_value());
            } else if (const auto* r = std::get_if<ReplyReceived>(&*g);
                       r != nullptr && r->function == 42) {
                log.emplace_back("host: HCACK " + std::to_string(r->code));
                host_session->stop();
                tool_session->stop();
            }
        }
    });
    tool_link.emplace(*tool_session);
    host_link.emplace(*host_session);

    const auto rethrow = [](const std::exception_ptr& e) {
        if (e) {
            std::rethrow_exception(e);
        }
    };
    asio::co_spawn(io, tool_session->run_passive(acceptor), rethrow);
    asio::co_spawn(io, host_session->run_active(acceptor.local_endpoint()), rethrow);
    io.run_for(10s);

    CHECK(io.stopped());
    CHECK(log == std::vector<std::string>{"tool on-line", "host got wafer 17 of LOT-0042",
                                          "tool holds LOT-0042", "host: HCACK 0"});
    CHECK(std::ranges::equal(received_map, map.view().bins()));
}
