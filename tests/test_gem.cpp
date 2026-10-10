// GEM Equipment and Host on two hsms::Protocols wired back to back, on a fake clock: the
// communication and control state machines, the wafer report, the lot hold, and every way
// the equipment refuses a message.
#include "support/synth.hpp"
#include "waferedge/gem/equipment.hpp"
#include "waferedge/gem/host.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace waferedge;
using namespace waferedge::gem;
using namespace std::chrono_literals;
using hsms::TimePoint;

namespace {

hsms::Config hsms_config(hsms::Role role) {
    hsms::Config c;
    c.role = role;
    c.session_id = 1;
    return c;
}

// What one side saw: its GEM events (with views copied out) and the data messages that
// reached it, as "S1F13", "S9F7", ...
struct Seen {
    std::vector<Event> events;
    std::vector<std::string> messages;
    std::vector<std::string> lots;               // WaferReported / LotCommandReceived, copied
    std::vector<std::vector<std::uint8_t>> maps; // WaferReported, copied

    template <typename E>
    [[nodiscard]] std::vector<E> all() const {
        std::vector<E> out;
        for (const auto& e : events) {
            if (const auto* x = std::get_if<E>(&e)) {
                out.push_back(*x);
            }
        }
        return out;
    }
    [[nodiscard]] bool got(std::string_view message) const {
        return std::ranges::find(messages, message) != messages.end();
    }
};

struct Fab {
    TimePoint now = TimePoint{} + 1h;
    hsms::Protocol tool_hsms{hsms_config(hsms::Role::passive)};
    hsms::Protocol host_hsms{hsms_config(hsms::Role::active)};
    ProtocolLink tool_link{tool_hsms, now};
    ProtocolLink host_link{host_hsms, now};
    Equipment equipment;
    Host host;
    Seen tool;
    Seen at_host;

    explicit Fab(EquipmentConfig eq = {}, HostConfig h = {}) : equipment(std::move(eq)), host(h) {}

    // Passes bytes both ways until both sides are quiet, feeding every HSMS event to GEM.
    void pump() {
        for (int round = 0; round < 50; ++round) {
            const bool a = pass(
                host_hsms, tool_hsms,
                [&](const hsms::Event& e) {
                    equipment.on_hsms(e, tool_link, now);
                    collect(equipment, tool);
                },
                tool);
            const bool b = pass(
                tool_hsms, host_hsms,
                [&](const hsms::Event& e) {
                    host.on_hsms(e, host_link, now);
                    collect(host, at_host);
                },
                at_host);
            if (!a && !b) {
                return;
            }
        }
        FAIL("the two sides never went quiet");
    }

    // Only timers: polls both protocols at `now` and ticks both endpoints.
    void advance(hsms::Duration d) {
        now += d;
        drain(
            tool_hsms,
            [&](const hsms::Event& e) {
                equipment.on_hsms(e, tool_link, now);
                collect(equipment, tool);
            },
            tool);
        drain(
            host_hsms,
            [&](const hsms::Event& e) {
                host.on_hsms(e, host_link, now);
                collect(host, at_host);
            },
            at_host);
        equipment.tick(tool_link, now);
        host.tick(host_link, now);
        collect(equipment, tool);
        collect(host, at_host);
    }

    void connect() {
        host_hsms.on_connected(now);
        tool_hsms.on_connected(now);
        pump();
    }

    void online() {
        connect();
        equipment.go_online(tool_link);
        pump();
        REQUIRE(equipment.control_state() == ControlState::online_remote);
    }

    template <typename Handle>
    bool pass(hsms::Protocol& from, hsms::Protocol& to, Handle&& handle, Seen& seen) {
        std::vector<std::uint8_t> wire;
        from.take_output(wire);
        if (!wire.empty()) {
            auto buffer = to.receive_buffer(wire.size());
            std::memcpy(buffer.data(), wire.data(), wire.size());
            to.on_received(wire.size(), now);
        }
        drain(to, handle, seen);
        return !wire.empty();
    }

    template <typename Handle>
    void drain(hsms::Protocol& p, Handle&& handle, Seen& seen) {
        while (auto e = p.poll(now)) {
            if (const auto* m = std::get_if<hsms::DataMessage>(&*e)) {
                seen.messages.push_back("S" + std::to_string(m->header.stream()) + "F" +
                                        std::to_string(m->header.function()));
            }
            handle(*e);
        }
    }

    static void collect(Endpoint& endpoint, Seen& seen) {
        while (auto e = endpoint.poll()) {
            if (const auto* w = std::get_if<WaferReported>(&*e)) {
                seen.lots.emplace_back(w->report.lot);
                seen.maps.emplace_back(w->report.map.bins().begin(), w->report.map.bins().end());
            } else if (const auto* c = std::get_if<LotCommandReceived>(&*e)) {
                seen.lots.emplace_back(c->command.lot);
            }
            seen.events.push_back(*e);
        }
    }
};

std::vector<ControlState> control_changes(const Seen& s) {
    std::vector<ControlState> out;
    for (const auto& c : s.all<ControlStateChanged>()) {
        out.push_back(c.state);
    }
    return out;
}

} // namespace

TEST_CASE("both sides establish communication when the link is selected", "[gem]") {
    Fab fab;
    CHECK(fab.equipment.comm_state() == CommState::not_communicating);
    fab.connect();
    CHECK(fab.equipment.communicating());
    CHECK(fab.host.communicating());
    CHECK(fab.tool.all<Communicating>().size() == 1);
    CHECK(fab.at_host.all<Communicating>().size() == 1);
    CHECK(fab.tool.got("S1F13")); // both sent one; each answered the other's
    CHECK(fab.at_host.got("S1F13"));
    CHECK(fab.equipment.control_state() == ControlState::equipment_offline);
}

TEST_CASE("the operator puts the equipment on-line: S1F1, S1F2", "[gem]") {
    Fab fab;
    fab.connect();
    fab.equipment.go_online(fab.tool_link);
    fab.pump();
    CHECK(fab.at_host.got("S1F1"));
    CHECK(control_changes(fab.tool) ==
          std::vector{ControlState::attempt_online, ControlState::online_remote});

    fab.equipment.set_remote(false);
    CHECK(fab.equipment.control_state() == ControlState::online_local);
    fab.equipment.go_offline();
    CHECK(fab.equipment.control_state() == ControlState::equipment_offline);
}

TEST_CASE("a wafer report reaches the host as a view, and is acknowledged", "[gem]") {
    Fab fab;
    fab.online();
    const WaferMap map = synth::random_map(40, 40, 100, 7);
    auto system = fab.equipment.report_wafer(fab.tool_link, "LOT-0042", 17, map);
    REQUIRE(system.has_value());
    fab.pump();

    const auto reports = fab.at_host.all<WaferReported>();
    REQUIRE(reports.size() == 1);
    CHECK(reports[0].report.wafer == 17);
    CHECK(fab.at_host.lots == std::vector<std::string>{"LOT-0042"});
    REQUIRE(fab.at_host.maps.size() == 1);
    CHECK(std::ranges::equal(fab.at_host.maps[0], map.view().bins()));

    const auto acks = fab.tool.all<ReplyReceived>();
    REQUIRE(acks.size() == 1);
    CHECK(acks[0].system == *system);
    CHECK(acks[0].stream == 6);
    CHECK(acks[0].function == 12);
    CHECK(acks[0].code == 0);
}

TEST_CASE("HOLD: the host commands, the equipment's app answers", "[gem]") {
    Fab fab;
    fab.online();
    auto system = fab.host.command(fab.host_link, {LotAction::hold, "LOT-0042"});
    REQUIRE(system.has_value());
    fab.pump();
    const auto commands = fab.tool.all<LotCommandReceived>();
    REQUIRE(commands.size() == 1);
    CHECK(commands[0].command.action == LotAction::hold);
    CHECK(fab.tool.lots == std::vector<std::string>{"LOT-0042"});

    REQUIRE(fab.equipment.answer(fab.tool_link, commands[0].primary, Hcack::done).has_value());
    fab.pump();
    const auto acks = fab.at_host.all<ReplyReceived>();
    REQUIRE(acks.size() == 1);
    CHECK(acks[0].system == *system);
    CHECK(acks[0].function == 42);
    CHECK(static_cast<Hcack>(acks[0].code) == Hcack::done);
}

TEST_CASE("HCACK without the app: local mode, unknown command, bad LOTID", "[gem]") {
    Fab fab;
    fab.online();
    const auto hcack_for = [&](auto build) {
        std::vector<std::uint8_t> body;
        secs::Encoder e(body);
        build(e);
        REQUIRE(fab.host_link.send(2, 41, true, *e.finish()).has_value());
        fab.pump();
        const auto acks = fab.at_host.all<ReplyReceived>();
        REQUIRE_FALSE(acks.empty());
        CHECK(fab.tool.all<LotCommandReceived>().empty());
        return static_cast<Hcack>(acks.back().code);
    };
    CHECK(hcack_for([](secs::Encoder& e) { e.list(2).ascii("START").list(0); }) ==
          Hcack::invalid_command);
    CHECK(hcack_for([](secs::Encoder& e) { e.list(2).ascii("HOLD").list(0); }) ==
          Hcack::parameter_invalid);
    CHECK(hcack_for([](secs::Encoder& e) {
              e.list(2).ascii("HOLD").list(1).list(2).ascii("LOTID").u4(42);
          }) == Hcack::parameter_invalid);
    fab.equipment.set_remote(false);
    CHECK(hcack_for([](secs::Encoder& e) { encode_lot_command(e, {LotAction::hold, "L"}); }) ==
          Hcack::cannot_do_now);
}

TEST_CASE("S9: unknown stream, unknown function, illegal data", "[gem]") {
    Fab fab;
    fab.online();
    REQUIRE(fab.host_link.send(7, 1, true, {}).has_value());  // no stream 7 here
    REQUIRE(fab.host_link.send(2, 13, true, {}).has_value()); // stream 2, no F13
    std::vector<std::uint8_t> body;
    secs::Encoder e(body);
    e.u4(1); // an S2F41 that isn't a list
    auto bad = fab.host_link.send(2, 41, true, *e.finish());
    REQUIRE(bad.has_value());
    fab.pump();
    const auto errors = fab.at_host.all<ErrorReported>();
    REQUIRE(errors.size() == 3);
    CHECK(errors[0].function == 3);
    CHECK(errors[0].about.stream() == 7);
    CHECK(errors[1].function == 5);
    CHECK(errors[1].about.function() == 13);
    CHECK(errors[2].function == 7);
    CHECK(errors[2].about.system_bytes == *bad);
    CHECK(fab.tool.all<MalformedMessage>().size() == 1);
}

TEST_CASE("off-line: the equipment aborts what it won't serve", "[gem]") {
    Fab fab;
    fab.connect(); // communicating, but EQUIPMENT OFF-LINE
    auto hold = fab.host.command(fab.host_link, {LotAction::hold, "L"});
    auto ping = fab.host.are_you_there(fab.host_link);
    REQUIRE(hold.has_value());
    REQUIRE(ping.has_value());
    fab.pump();
    CHECK(fab.at_host.got("S2F0"));
    CHECK(fab.at_host.got("S1F0"));
    CHECK(fab.at_host.all<TransactionFailed>().size() == 2);
    CHECK(fab.at_host.all<TransactionFailed>()[0].why == TransactionFailed::Why::aborted);
    CHECK(fab.equipment.report_wafer(fab.tool_link, "L", 1, WaferMap(2, 2, 1)).error() ==
          GemError::offline);
}

TEST_CASE("the host asks: S1F17 on-line, S1F15 off-line", "[gem]") {
    Fab fab;
    fab.connect();
    // EQUIPMENT OFF-LINE: the operator's choice wins.
    REQUIRE(fab.host.request_online(fab.host_link).has_value());
    fab.pump();
    CHECK(static_cast<Onlack>(fab.at_host.all<ReplyReceived>().back().code) == Onlack::refused);

    fab.equipment.go_online(fab.tool_link);
    fab.pump();
    REQUIRE(fab.host.request_offline(fab.host_link).has_value());
    fab.pump();
    CHECK(fab.equipment.control_state() == ControlState::host_offline);
    REQUIRE(fab.host.request_online(fab.host_link).has_value());
    fab.pump();
    CHECK(static_cast<Onlack>(fab.at_host.all<ReplyReceived>().back().code) == Onlack::accepted);
    CHECK(fab.equipment.control_state() == ControlState::online_remote);
    REQUIRE(fab.host.request_online(fab.host_link).has_value());
    fab.pump();
    CHECK(static_cast<Onlack>(fab.at_host.all<ReplyReceived>().back().code) ==
          Onlack::already_online);
}

TEST_CASE("WAIT DELAY: a denied S1F13 is retried after the delay", "[gem]") {
    EquipmentConfig eq;
    eq.establish_delay = 10s;
    HostConfig host;
    host.comm_enabled = false; // denies every S1F13
    Fab fab(eq, host);
    fab.connect();
    CHECK(fab.equipment.comm_state() == CommState::wait_delay);
    CHECK(fab.equipment.next_deadline() == fab.now + 10s);
    CHECK(fab.host.comm_state() == CommState::disabled);

    fab.advance(10s - 1ns);
    CHECK_FALSE(fab.tool_hsms.has_output());
    fab.advance(1ns);
    CHECK(fab.equipment.comm_state() == CommState::wait_cra);
    fab.pump(); // denied again
    CHECK(fab.equipment.comm_state() == CommState::wait_delay);
    CHECK(std::ranges::count(fab.at_host.messages, "S1F13") == 2);

    // Meanwhile anything else from the host is aborted: not communicating.
    REQUIRE(fab.host_link.send(1, 1, true, {}).has_value());
    fab.pump();
    CHECK(fab.at_host.got("S1F0"));
}

TEST_CASE("WAIT CRA: no S1F14 within T3 also waits the delay", "[gem]") {
    HostConfig host;
    host.comm_enabled = false;
    Fab fab({}, host);
    fab.host_hsms.on_connected(fab.now);
    fab.tool_hsms.on_connected(fab.now);
    // Select, and the equipment's S1F13 reaches the host but its answer is lost.
    fab.pump();
    CHECK(fab.equipment.comm_state() == CommState::wait_delay); // denied: the quick path
    fab.advance(10s);                                           // resend, then lose it
    CHECK(fab.equipment.comm_state() == CommState::wait_cra);
    std::vector<std::uint8_t> lost;
    fab.tool_hsms.take_output(lost);
    fab.advance(45s); // T3
    CHECK(fab.equipment.comm_state() == CommState::wait_delay);
    CHECK_FALSE(fab.tool_hsms.has_output()); // no S9F9 for S1F13: not communicating
}

TEST_CASE("T3 on a report: TransactionFailed, S9F9 to the host", "[gem]") {
    Fab fab;
    fab.online();
    auto system = fab.equipment.report_wafer(fab.tool_link, "L", 1, WaferMap(2, 2, 1));
    REQUIRE(system.has_value());
    std::vector<std::uint8_t> held; // the report is stuck in the network
    fab.tool_hsms.take_output(held);
    fab.advance(45s);
    const auto failed = fab.tool.all<TransactionFailed>();
    REQUIRE(failed.size() == 1);
    CHECK(failed[0].system == *system);
    CHECK(failed[0].why == TransactionFailed::Why::timeout);
    fab.pump();
    const auto errors = fab.at_host.all<ErrorReported>();
    REQUIRE(errors.size() == 1);
    CHECK(errors[0].function == 9);
    CHECK(errors[0].about.system_bytes == *system);
}

TEST_CASE("an alarm is reported and acknowledged", "[gem]") {
    Fab fab;
    fab.online();
    REQUIRE(fab.equipment.report_alarm(fab.tool_link, {true, 2, 7, "EDGE-RING"}).has_value());
    fab.pump();
    const auto alarms = fab.at_host.all<AlarmReported>();
    REQUIRE(alarms.size() == 1);
    CHECK(alarms[0].alarm.set);
    CHECK(alarms[0].alarm.id == 7);
    CHECK(fab.tool.all<ReplyReceived>().back().function == 2);
}

TEST_CASE("losing the link ends communication and a pending on-line attempt", "[gem]") {
    Fab fab;
    fab.connect();
    fab.equipment.go_online(fab.tool_link); // S1F1 sent, never answered
    fab.tool_hsms.on_disconnected(fab.now);
    fab.host_hsms.on_disconnected(fab.now);
    fab.advance(0s);
    CHECK(fab.tool.all<NotCommunicating>().size() == 1);
    CHECK(fab.at_host.all<NotCommunicating>().size() == 1);
    CHECK(fab.equipment.control_state() == ControlState::host_offline);
    CHECK(fab.host.command(fab.host_link, {LotAction::hold, "L"}).error() ==
          GemError::not_communicating);

    fab.connect(); // a new connection establishes communication again
    CHECK(fab.equipment.communicating());
    CHECK(fab.tool.all<Communicating>().size() == 2);
}

TEST_CASE("a full HSMS transaction window is busy, not a dead link", "[gem]") {
    Fab fab;
    fab.online();
    const WaferMap map(2, 2, 1);
    for (std::size_t i = 0; i < hsms::kMaxOpenTransactions; ++i) {
        REQUIRE(fab.equipment.report_wafer(fab.tool_link, "L", 1, map).has_value());
    }
    // 64 reports await S6F12: the next must wait for one, and the caller can tell.
    CHECK(fab.equipment.report_wafer(fab.tool_link, "L", 1, map).error() == GemError::busy);
    fab.pump(); // the host acknowledges them all
    CHECK(fab.equipment.report_wafer(fab.tool_link, "L", 1, map).has_value());
}
