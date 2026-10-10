// The GEM hot loop allocates nothing once warm: a wafer report and its S6F12, a HOLD and its
// S2F42, through Equipment, Host and two hsms::Protocols.
#include "support/alloc_counter.hpp"
#include "support/synth.hpp"
#include "waferedge/gem/equipment.hpp"
#include "waferedge/gem/host.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <vector>

using namespace waferedge;
using namespace waferedge::gem;
using namespace std::chrono_literals;

TEST_CASE("GEM: report, ack, hold, answer allocate nothing once warm", "[gem]") {
    hsms::TimePoint now = hsms::TimePoint{} + 1h;
    hsms::Config tool_config;
    hsms::Config host_config;
    host_config.role = hsms::Role::active;
    hsms::Protocol tool_hsms(tool_config);
    hsms::Protocol host_hsms(host_config);
    ProtocolLink tool_link(tool_hsms, now);
    ProtocolLink host_link(host_hsms, now);
    Equipment equipment(EquipmentConfig{});
    Host host;
    std::vector<std::uint8_t> to_tool;
    std::vector<std::uint8_t> to_host;
    std::optional<hsms::Header> command; // the HOLD to answer
    bool ok = true;

    const auto pass = [&](hsms::Protocol& from, hsms::Protocol& to, std::vector<std::uint8_t>& wire,
                          auto&& on_event) {
        from.take_output(wire);
        auto buffer = to.receive_buffer(wire.size());
        std::memcpy(buffer.data(), wire.data(), wire.size());
        to.on_received(wire.size(), now);
        while (auto e = to.poll(now)) {
            on_event(*e);
        }
    };
    const auto pump = [&] {
        for (int i = 0; i < 4; ++i) {
            pass(host_hsms, tool_hsms, to_tool, [&](const hsms::Event& e) {
                equipment.on_hsms(e, tool_link, now);
                while (auto g = equipment.poll()) {
                    if (const auto* c = std::get_if<LotCommandReceived>(&*g)) {
                        command = c->primary;
                    }
                }
            });
            pass(tool_hsms, host_hsms, to_host, [&](const hsms::Event& e) {
                host.on_hsms(e, host_link, now);
                while (auto g = host.poll()) {
                    ok = ok && !std::holds_alternative<TransactionFailed>(*g);
                }
            });
        }
    };
    host_hsms.on_connected(now);
    tool_hsms.on_connected(now);
    pump();
    equipment.go_online(tool_link);
    pump();
    REQUIRE(equipment.control_state() == ControlState::online_remote);

    const WaferMap map = synth::random_map(40, 40, 100, 1);
    const auto round = [&] {
        ok = ok && equipment.report_wafer(tool_link, "LOT-0042", 17, map).has_value();
        ok = ok && host.command(host_link, {LotAction::hold, "LOT-0042"}).has_value();
        pump();
        ok = ok && command.has_value() &&
             equipment.answer(tool_link, *command, Hcack::done).has_value();
        command.reset();
        pump();
    };
    for (int i = 0; i < 3; ++i) {
        round(); // warm-up: every buffer reaches its size
    }
    const auto before = alloc::count();
    for (int i = 0; i < 100; ++i) {
        round();
    }
    const auto allocations = alloc::count() - before;
    CHECK(ok);
    CHECK(allocations == 0);
}
