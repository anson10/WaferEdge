// The HSMS protocol engine on a fake clock: two Protocols wired back to back (what one
// writes, the other receives), time passed in by hand. Every timer is tested at its deadline
// and one tick before it, without sleeping.
#include "waferedge/hsms/protocol.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

using namespace waferedge::hsms;
using namespace std::chrono_literals;
using Bytes = std::vector<std::uint8_t>;

namespace {

const TimePoint t0 = TimePoint{} + 1h; // the fake clock's start
constexpr Duration tick = 1ns;

Config config(Role role) {
    Config c;
    c.role = role;
    c.session_id = 7;
    c.t3 = 45s;
    c.t5 = 10s;
    c.t6 = 5s;
    c.t7 = 10s;
    c.t8 = 5s;
    c.max_message = 1024;
    return c;
}

std::vector<Event> drain(Protocol& p, TimePoint now) {
    std::vector<Event> events;
    while (auto e = p.poll(now)) {
        events.push_back(*e);
    }
    return events;
}

void feed(Protocol& p, std::span<const std::uint8_t> bytes, TimePoint now) {
    auto buffer = p.receive_buffer(bytes.size());
    REQUIRE(buffer.size() >= bytes.size());
    if (!bytes.empty()) {
        std::memcpy(buffer.data(), bytes.data(), bytes.size());
    }
    p.on_received(bytes.size(), now);
}

Bytes take(Protocol& p) {
    Bytes out;
    p.take_output(out);
    return out;
}

// What `from` has written, received by `to`; then `to`'s events.
std::vector<Event> deliver(Protocol& from, Protocol& to, TimePoint now) {
    feed(to, take(from), now);
    return drain(to, now);
}

template <typename E>
bool holds(const std::vector<Event>& events, std::size_t i = 0) {
    return events.size() > i && std::holds_alternative<E>(events[i]);
}

CloseReason closed_reason(const std::vector<Event>& events) {
    REQUIRE(events.size() == 1);
    REQUIRE(std::holds_alternative<Closed>(events[0]));
    return std::get<Closed>(events[0]).reason;
}

Bytes frame(const Header& h, const Bytes& body = {}) {
    Bytes b(kLengthSize + kHeaderSize);
    const auto length = static_cast<std::uint32_t>(kHeaderSize + body.size());
    b[0] = static_cast<std::uint8_t>(length >> 24U);
    b[1] = static_cast<std::uint8_t>(length >> 16U);
    b[2] = static_cast<std::uint8_t>(length >> 8U);
    b[3] = static_cast<std::uint8_t>(length);
    write_header(h, b.data() + kLengthSize);
    b.insert(b.end(), body.begin(), body.end()); // appended: GCC 13 -O3 misreads a copy
    return b;                                    // to a fixed offset as out of bounds
}

Header only_header(const Bytes& wire) {
    REQUIRE(wire.size() == kLengthSize + kHeaderSize);
    return read_header(wire.data() + kLengthSize);
}

// Both sides connected and selected at t0.
struct Link {
    Protocol host{config(Role::active)};
    Protocol tool{config(Role::passive)};
    Link() {
        host.on_connected(t0);
        tool.on_connected(t0);
        REQUIRE(holds<Selected>(deliver(host, tool, t0)));
        REQUIRE(holds<Selected>(deliver(tool, host, t0)));
    }
};

} // namespace

TEST_CASE("Select: the active side asks, the passive side answers", "[hsms]") {
    Protocol host(config(Role::active));
    Protocol tool(config(Role::passive));
    CHECK(host.state() == State::not_connected);
    host.on_connected(t0);
    tool.on_connected(t0);
    CHECK(host.state() == State::not_selected);
    CHECK_FALSE(tool.has_output()); // the passive side waits

    const Bytes select = take(host);
    // length 10, session 0xFFFF, PType 0, SType 1 (Select.req), system bytes 1
    CHECK(select == Bytes{0, 0, 0, 10, 0xFF, 0xFF, 0, 0, 0, 1, 0, 0, 0, 1});
    feed(tool, select, t0);
    CHECK(holds<Selected>(drain(tool, t0)));
    CHECK(tool.state() == State::selected);

    const Bytes rsp = take(tool);
    CHECK(rsp == Bytes{0, 0, 0, 10, 0xFF, 0xFF, 0, 0, 0, 2, 0, 0, 0, 1}); // status 0
    feed(host, rsp, t0);
    CHECK(holds<Selected>(drain(host, t0)));
    CHECK(host.state() == State::selected);
    CHECK_FALSE(host.next_deadline().has_value()); // T6 and T7 stopped, no linktest
}

TEST_CASE("a data transaction: primary with W-bit, reply with the same system bytes", "[hsms]") {
    Link l;
    const Bytes body = {0x41, 0x02, 'O', 'K'};
    auto system = l.host.send(1, 1, true, body, t0);
    REQUIRE(system.has_value());
    CHECK(l.host.next_deadline() == t0 + 45s); // T3

    auto at_tool = deliver(l.host, l.tool, t0);
    REQUIRE(holds<DataMessage>(at_tool));
    const auto request = std::get<DataMessage>(at_tool[0]);
    CHECK(request.kind == DataMessage::Kind::primary);
    CHECK(request.header.session_id == 7);
    CHECK(request.header.stream() == 1);
    CHECK(request.header.function() == 1);
    CHECK(request.header.reply_expected());
    CHECK(request.header.system_bytes == *system);
    CHECK(Bytes(request.body.begin(), request.body.end()) == body);

    REQUIRE(l.tool.reply(request.header, Bytes{0x01, 0x00}).has_value());
    auto at_host = deliver(l.tool, l.host, t0 + 1s);
    REQUIRE(holds<DataMessage>(at_host));
    const auto reply = std::get<DataMessage>(at_host[0]);
    CHECK(reply.kind == DataMessage::Kind::reply);
    CHECK(reply.header.function() == 2);
    CHECK_FALSE(reply.header.reply_expected());
    CHECK(reply.header.system_bytes == *system);
    CHECK_FALSE(l.host.next_deadline().has_value()); // the transaction is closed
    CHECK(drain(l.host, t0 + 1h).empty());           // and T3 never fires
}

TEST_CASE("T3: a primary without a reply times out; a late reply is unmatched", "[hsms]") {
    Link l;
    auto system = l.host.send(6, 11, true, {}, t0);
    REQUIRE(system.has_value());
    auto request = std::get<DataMessage>(deliver(l.host, l.tool, t0)[0]);

    CHECK(drain(l.host, t0 + 45s - tick).empty());
    auto events = drain(l.host, t0 + 45s);
    REQUIRE(holds<ReplyTimeout>(events));
    CHECK(std::get<ReplyTimeout>(events[0]).request.system_bytes == *system);
    CHECK(l.host.state() == State::selected); // T3 ends the transaction, not the connection
    CHECK(l.host.stats().reply_timeouts == 1);

    REQUIRE(l.tool.reply(request.header, {}).has_value());
    auto late = deliver(l.tool, l.host, t0 + 46s);
    REQUIRE(holds<DataMessage>(late));
    CHECK(std::get<DataMessage>(late[0]).kind == DataMessage::Kind::unmatched_reply);
}

TEST_CASE("a primary without the W-bit opens no transaction", "[hsms]") {
    Link l;
    REQUIRE(l.host.send(5, 1, false, {}, t0).has_value());
    CHECK_FALSE(l.host.next_deadline().has_value());
}

TEST_CASE("T6: no Select.rsp closes the connection; T5 gates the next attempt", "[hsms]") {
    Protocol host(config(Role::active));
    host.on_connected(t0);
    (void)take(host);
    CHECK(host.next_deadline() == t0 + 5s);
    CHECK(drain(host, t0 + 5s - tick).empty());
    CHECK(closed_reason(drain(host, t0 + 5s)) == CloseReason::t6_control);
    CHECK(host.state() == State::not_connected);
    CHECK(host.wants_close());
    CHECK(host.connect_allowed_at() == t0 + 5s + 10s);
    CHECK(drain(host, t0 + 1h).empty()); // Closed is reported once

    host.on_connect_failed(t0 + 20s);
    CHECK(host.connect_allowed_at() == t0 + 30s);
}

TEST_CASE("T7: connected but never selected", "[hsms]") {
    Protocol tool(config(Role::passive));
    tool.on_connected(t0);
    CHECK(tool.next_deadline() == t0 + 10s);
    CHECK(drain(tool, t0 + 10s - tick).empty());
    CHECK(closed_reason(drain(tool, t0 + 10s)) == CloseReason::t7_not_selected);
}

TEST_CASE("T8: a message stalled half-way; each new byte restarts the clock", "[hsms]") {
    Link l;
    REQUIRE(l.host.send(1, 1, false, Bytes(20, 0xAB), t0).has_value());
    const Bytes wire = take(l.host);

    feed(l.tool, std::span(wire).first(7), t0);
    CHECK(drain(l.tool, t0 + 5s - tick).empty());
    CHECK(l.tool.next_deadline() == t0 + 5s);
    feed(l.tool, std::span(wire).subspan(7, 10), t0 + 4s); // more bytes: T8 restarts
    CHECK(drain(l.tool, t0 + 8s).empty());
    CHECK(closed_reason(drain(l.tool, t0 + 9s)) == CloseReason::t8_intercharacter);
}

TEST_CASE("Linktest: periodic while selected, answered, and T6 if unanswered", "[hsms]") {
    Config host_config = config(Role::active);
    host_config.linktest = 30s;
    Protocol host(host_config);
    Protocol tool(config(Role::passive));
    host.on_connected(t0);
    tool.on_connected(t0);
    (void)deliver(host, tool, t0);
    (void)deliver(tool, host, t0);
    CHECK(host.next_deadline() == t0 + 30s);

    CHECK(drain(host, t0 + 30s).empty());
    const Bytes req = take(host);
    CHECK(static_cast<SType>(only_header(req).stype) == SType::linktest_req);
    feed(tool, req, t0 + 30s);
    CHECK(drain(tool, t0 + 30s).empty()); // answered silently
    const Bytes rsp = take(tool);
    CHECK(static_cast<SType>(only_header(rsp).stype) == SType::linktest_rsp);
    CHECK(only_header(rsp).system_bytes == only_header(req).system_bytes);
    feed(host, rsp, t0 + 31s);
    CHECK(drain(host, t0 + 31s).empty());
    CHECK(host.next_deadline() == t0 + 61s); // the next period starts at the response

    CHECK(drain(host, t0 + 61s).empty()); // sends the next one; the tool stays silent
    CHECK(host.has_output());
    CHECK(closed_reason(drain(host, t0 + 66s)) == CloseReason::t6_control);
}

TEST_CASE("data before Select is rejected: entity not selected", "[hsms]") {
    Protocol tool(config(Role::passive));
    tool.on_connected(t0);
    feed(tool, frame(data_header(7, 1, 1, true, 42)), t0);
    CHECK(drain(tool, t0).empty());
    const Header r = only_header(take(tool));
    CHECK(static_cast<SType>(r.stype) == SType::reject_req);
    CHECK(r.byte2 == 0); // the rejected message's SType
    CHECK(static_cast<RejectReason>(r.byte3) == RejectReason::entity_not_selected);
    CHECK(r.system_bytes == 42);
    CHECK(r.session_id == 7);
}

TEST_CASE("unknown SType and PType are rejected, Rejects never are", "[hsms]") {
    Link l;
    Header odd = control_header(SType::linktest_req, 9);
    odd.stype = 8;
    feed(l.tool, frame(odd), t0);
    CHECK(drain(l.tool, t0).empty());
    Header r = only_header(take(l.tool));
    CHECK(static_cast<RejectReason>(r.byte3) == RejectReason::stype_not_supported);
    CHECK(r.byte2 == 8);

    Header ptype = data_header(7, 1, 1, false, 10);
    ptype.ptype = 3;
    feed(l.tool, frame(ptype), t0);
    CHECK(drain(l.tool, t0).empty());
    r = only_header(take(l.tool));
    CHECK(static_cast<RejectReason>(r.byte3) == RejectReason::ptype_not_supported);
    CHECK(r.byte2 == 3);

    // A Reject with a bad PType is dropped, not rejected back.
    Header bad_reject = control_header(SType::reject_req, 11);
    bad_reject.ptype = 1;
    feed(l.tool, frame(bad_reject), t0);
    CHECK(drain(l.tool, t0).empty());
    CHECK_FALSE(l.tool.has_output());
}

TEST_CASE("responses nobody asked for are rejected: transaction not open", "[hsms]") {
    Link l;
    for (auto s : {SType::select_rsp, SType::deselect_rsp, SType::linktest_rsp}) {
        INFO(stype_name(s));
        feed(l.tool, frame(control_header(s, 99)), t0);
        CHECK(drain(l.tool, t0).empty());
        const Header r = only_header(take(l.tool));
        CHECK(static_cast<RejectReason>(r.byte3) == RejectReason::transaction_not_open);
        CHECK(r.byte2 == static_cast<std::uint8_t>(s));
    }
    CHECK(l.tool.stats().rejects_sent == 3);
}

TEST_CASE("Select refused, by status or by Reject", "[hsms]") {
    for (const bool by_reject : {false, true}) {
        INFO("by reject " << by_reject);
        Protocol host(config(Role::active));
        host.on_connected(t0);
        const Header req = only_header(take(host));
        const Header answer =
            by_reject ? control_header(SType::reject_req, req.system_bytes,
                                       static_cast<std::uint8_t>(SType::select_req),
                                       static_cast<std::uint8_t>(RejectReason::stype_not_supported))
                      : control_header(SType::select_rsp, req.system_bytes, 0,
                                       static_cast<std::uint8_t>(SelectStatus::not_ready));
        feed(host, frame(answer), t0);
        CHECK(closed_reason(drain(host, t0)) == CloseReason::select_refused);
    }
}

TEST_CASE("a second Select.req gets 'already active'", "[hsms]") {
    Link l;
    feed(l.tool, frame(control_header(SType::select_req, 50)), t0);
    CHECK(drain(l.tool, t0).empty());
    const Header r = only_header(take(l.tool));
    CHECK(static_cast<SelectStatus>(r.byte3) == SelectStatus::already_active);
    CHECK(l.tool.state() == State::selected);
}

TEST_CASE("simultaneous Select: two active sides each select once", "[hsms]") {
    Protocol a(config(Role::active));
    Protocol b(config(Role::active));
    a.on_connected(t0);
    b.on_connected(t0);
    const Bytes a_req = take(a);
    const Bytes b_req = take(b);
    feed(a, b_req, t0);
    feed(b, a_req, t0);
    CHECK(holds<Selected>(drain(a, t0)));
    CHECK(holds<Selected>(drain(b, t0)));
    CHECK(deliver(a, b, t0).empty()); // the responses to each side's own Select
    CHECK(deliver(b, a, t0).empty());
    CHECK(a.state() == State::selected);
    CHECK(b.state() == State::selected);
    CHECK_FALSE(a.next_deadline().has_value());
}

TEST_CASE("Deselect returns both sides to NOT SELECTED", "[hsms]") {
    Link l;
    REQUIRE(l.host.send(1, 1, true, {}, t0).has_value()); // dropped by the deselect
    (void)take(l.host);
    REQUIRE(l.host.deselect(t0).has_value());
    CHECK(l.host.deselect(t0).error() == SendError::control_pending);
    CHECK(holds<Deselected>(deliver(l.host, l.tool, t0)));
    CHECK(holds<Deselected>(deliver(l.tool, l.host, t0)));
    CHECK(l.host.state() == State::not_selected);
    CHECK(l.tool.state() == State::not_selected);
    CHECK(l.host.send(1, 1, false, {}, t0).error() == SendError::not_selected);
    CHECK(l.host.next_deadline() == t0 + 10s); // T7 again; the open transaction is gone
}

TEST_CASE("Separate: sent, then closed on both sides", "[hsms]") {
    Link l;
    l.host.separate(t0);
    CHECK(l.host.wants_close());
    CHECK(closed_reason(drain(l.host, t0)) == CloseReason::local);
    const Bytes sep = take(l.host);
    CHECK(static_cast<SType>(only_header(sep).stype) == SType::separate_req);
    feed(l.tool, sep, t0);
    CHECK(closed_reason(drain(l.tool, t0)) == CloseReason::peer_separate);
    CHECK(l.tool.wants_close());
}

TEST_CASE("the transport dropping is reported once", "[hsms]") {
    Link l;
    l.tool.on_disconnected(t0);
    CHECK(closed_reason(drain(l.tool, t0)) == CloseReason::transport);
    l.tool.on_disconnected(t0);
    CHECK(drain(l.tool, t0).empty());
}

TEST_CASE("bad lengths close the connection before the body arrives", "[hsms]") {
    SECTION("below the header size") {
        Link l;
        feed(l.tool, Bytes{0, 0, 0, 9}, t0);
        CHECK(closed_reason(drain(l.tool, t0)) == CloseReason::bad_length);
    }
    SECTION("above max_message, from the length field alone") {
        Link l;
        feed(l.tool, Bytes{0, 0, 4, 1}, t0); // 1025 > 1024
        CHECK(closed_reason(drain(l.tool, t0)) == CloseReason::too_long);
    }
    SECTION("the longest message passes") {
        Link l;
        REQUIRE(l.host.send(1, 1, false, Bytes(1024 - kHeaderSize, 1), t0).has_value());
        CHECK(l.host.send(1, 1, false, Bytes(1024 - kHeaderSize + 1, 1), t0).error() ==
              SendError::too_long);
        CHECK(holds<DataMessage>(deliver(l.host, l.tool, t0)));
    }
}

TEST_CASE("send errors", "[hsms]") {
    Protocol host(config(Role::active));
    CHECK(host.send(1, 1, false, {}, t0).error() == SendError::not_connected);
    CHECK(host.linktest(t0).error() == SendError::not_connected);
    host.on_connected(t0);
    CHECK(host.send(1, 1, false, {}, t0).error() == SendError::not_selected);
    CHECK(host.reply(data_header(0, 1, 1, true, 1), {}).error() == SendError::not_selected);

    Link l;
    for (std::size_t i = 0; i < kMaxOpenTransactions; ++i) {
        REQUIRE(l.host.send(1, 1, true, {}, t0).has_value());
    }
    CHECK(l.host.send(1, 1, true, {}, t0).error() == SendError::too_many_open);
    CHECK(l.host.send(1, 1, false, {}, t0).has_value()); // no slot needed
}

TEST_CASE("a reconnect starts clean", "[hsms]") {
    Link l;
    REQUIRE(l.host.send(1, 1, true, {}, t0).has_value());
    l.host.on_disconnected(t0);
    (void)drain(l.host, t0);
    CHECK_FALSE(l.host.has_output());
    l.host.on_connected(t0 + 20s);
    CHECK(l.host.state() == State::not_selected);
    CHECK(l.host.stats().connections == 2);
    CHECK(l.host.next_deadline() == t0 + 25s); // T6 of the new Select; the old T3 is gone
}

TEST_CASE("framing: any split of the stream gives the same messages", "[hsms]") {
    Link l;
    std::vector<Bytes> bodies;
    for (std::uint8_t i = 0; i < 40; ++i) {
        bodies.emplace_back(static_cast<std::size_t>(i) * 7, i);
        REQUIRE(l.host.send(6, 11, false, bodies.back(), t0).has_value());
    }
    const Bytes wire = take(l.host);
    std::mt19937 rng(1);
    for (int round = 0; round < 50; ++round) {
        Protocol tool(config(Role::passive));
        tool.on_connected(t0);
        feed(tool, frame(control_header(SType::select_req, 1)), t0);
        REQUIRE(holds<Selected>(drain(tool, t0)));
        std::vector<Bytes> got;
        std::size_t pos = 0;
        while (pos < wire.size()) {
            const std::size_t n = std::min<std::size_t>(wire.size() - pos, 1 + rng() % 300);
            feed(tool, std::span(wire).subspan(pos, n), t0);
            pos += n;
            for (const auto& e : drain(tool, t0)) {
                const auto& m = std::get<DataMessage>(e);
                got.emplace_back(m.body.begin(), m.body.end());
            }
        }
        CHECK(got == bodies);
        CHECK(tool.state() == State::selected);
    }
}
