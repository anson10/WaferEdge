// hsms::Session over real TCP on 127.0.0.1: the coroutine driver wired to the protocol
// engine. The protocol rules are tested on a fake clock (test_hsms_protocol.cpp); these
// tests check the wiring with short timers and a time limit on every run.
#include "waferedge/hsms/session.hpp"
#include "waferedge/secs/encoder.hpp"
#include "waferedge/secs/item.hpp"

#include <asio/as_tuple.hpp>
#include <asio/co_spawn.hpp>
#include <asio/io_context.hpp>
#include <asio/use_awaitable.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <exception>
#include <optional>
#include <string>
#include <vector>

using namespace waferedge;
using namespace waferedge::hsms;
using namespace std::chrono_literals;

namespace {

Config fast(Role role) {
    Config c;
    c.role = role;
    c.session_id = 1;
    c.t3 = 2s;
    c.t5 = 50ms;
    c.t6 = 1s;
    c.t7 = 1s;
    c.t8 = 1s;
    return c;
}

void rethrow(const std::exception_ptr& e) {
    if (e) {
        std::rethrow_exception(e);
    }
}

asio::ip::tcp::acceptor loopback_acceptor(asio::io_context& io) {
    return {io, asio::ip::tcp::endpoint(asio::ip::address_v4::loopback(), 0)};
}

// The S1F2 body an emulator answers "are you there" with: <L [2] <A MDLN> <A SOFTREV>>.
std::vector<std::uint8_t> s1f2_body() {
    std::vector<std::uint8_t> body;
    secs::Encoder e(body);
    e.list(2).ascii("WAFEREDGE-EMU").ascii("1.0.0");
    return body;
}

} // namespace

TEST_CASE("select, a transaction, and separate over TCP", "[hsms][net]") {
    asio::io_context io;
    auto acceptor = loopback_acceptor(io);
    const auto reply_body = s1f2_body();
    std::vector<std::string> host_log;
    std::vector<std::string> tool_log;

    Session tool(io.get_executor(), fast(Role::passive), [&](Session& s, const Event& e) {
        if (std::holds_alternative<Selected>(e)) {
            tool_log.emplace_back("selected");
        } else if (const auto* m = std::get_if<DataMessage>(&e)) {
            tool_log.emplace_back("S1F1");
            REQUIRE(m->header.stream() == 1);
            REQUIRE(m->header.function() == 1);
            REQUIRE(s.reply(m->header, reply_body).has_value());
        } else if (const auto* c = std::get_if<Closed>(&e)) {
            tool_log.emplace_back(close_reason_name(c->reason));
            s.stop();
        }
    });
    Session host(io.get_executor(), fast(Role::active), [&](Session& s, const Event& e) {
        if (std::holds_alternative<Selected>(e)) {
            host_log.emplace_back("selected");
            REQUIRE(s.send(1, 1, true, {}).has_value());
        } else if (const auto* m = std::get_if<DataMessage>(&e)) {
            REQUIRE(m->kind == DataMessage::Kind::reply);
            auto item = secs::decode(m->body);
            REQUIRE(item.has_value());
            host_log.push_back(secs::to_sml(*item->list()->at(0)));
            s.stop(); // Separate.req, then no reconnect
        } else if (const auto* c = std::get_if<Closed>(&e)) {
            host_log.emplace_back(close_reason_name(c->reason));
        }
    });

    asio::co_spawn(io, tool.run_passive(acceptor), rethrow);
    asio::co_spawn(io, host.run_active(acceptor.local_endpoint()), rethrow);
    io.run_for(10s);

    CHECK(io.stopped()); // both finished on their own
    CHECK(host_log ==
          std::vector<std::string>{"selected", "<A \"WAFEREDGE-EMU\">", "closed locally"});
    CHECK(tool_log == std::vector<std::string>{"selected", "S1F1", "peer sent Separate.req"});
    CHECK(host.state() == State::not_connected);
}

TEST_CASE("the active side reconnects T5 after the passive side separates", "[hsms][net]") {
    asio::io_context io;
    auto acceptor = loopback_acceptor(io);
    int tool_selects = 0;
    int host_selects = 0;
    TimePoint closed_at{};
    TimePoint reselected_at{};

    Session tool(io.get_executor(), fast(Role::passive), [&](Session& s, const Event& e) {
        if (std::holds_alternative<Selected>(e) && ++tool_selects == 1) {
            s.separate(); // drop the first connection
        }
    });
    Session host(io.get_executor(), fast(Role::active), [&](Session& s, const Event& e) {
        if (std::holds_alternative<Closed>(e) && closed_at == TimePoint{}) {
            closed_at = Clock::now(); // the first close; stop() closes again at the end
        } else if (std::holds_alternative<Selected>(e) && ++host_selects == 2) {
            reselected_at = Clock::now();
            s.stop();
            tool.stop();
        }
    });

    asio::co_spawn(io, tool.run_passive(acceptor), rethrow);
    asio::co_spawn(io, host.run_active(acceptor.local_endpoint()), rethrow);
    io.run_for(10s);

    CHECK(io.stopped());
    CHECK(tool_selects == 2);
    CHECK(host_selects == 2);
    CHECK(host.protocol().stats().connections == 2);
    CHECK(reselected_at - closed_at >= 50ms); // T5
}

TEST_CASE("a connection that never selects is closed after T7", "[hsms][net]") {
    asio::io_context io;
    auto acceptor = loopback_acceptor(io);
    Config config = fast(Role::passive);
    config.t7 = 100ms;
    std::optional<CloseReason> reason;
    Session tool(io.get_executor(), config, [&](Session& s, const Event& e) {
        if (const auto* c = std::get_if<Closed>(&e)) {
            reason = c->reason;
            s.stop();
        }
    });
    asio::co_spawn(io, tool.run_passive(acceptor), rethrow);

    // A raw client: connects and says nothing; the tool must hang up on it.
    bool eof = false;
    auto client = [&]() -> asio::awaitable<void> {
        asio::ip::tcp::socket socket(io);
        co_await socket.async_connect(acceptor.local_endpoint(), asio::use_awaitable);
        std::array<std::uint8_t, 16> buffer{};
        auto [ec, n] = co_await socket.async_read_some(asio::buffer(buffer),
                                                       asio::as_tuple(asio::use_awaitable));
        eof = ec == asio::error::eof && n == 0;
    };
    const auto start = Clock::now();
    asio::co_spawn(io, client(), rethrow);
    io.run_for(10s);

    CHECK(io.stopped());
    REQUIRE(reason.has_value());
    CHECK(*reason == CloseReason::t7_not_selected);
    CHECK(eof);
    CHECK(Clock::now() - start >= 100ms);
}

TEST_CASE("periodic linktests keep a quiet connection selected", "[hsms][net]") {
    asio::io_context io;
    auto acceptor = loopback_acceptor(io);
    Config host_config = fast(Role::active);
    host_config.linktest = 20ms;
    bool closed = false;
    Session tool(io.get_executor(), fast(Role::passive), [&](Session&, const Event& e) {
        closed = closed || std::holds_alternative<Closed>(e);
    });
    Session host(io.get_executor(), host_config, [](Session&, const Event&) {});

    asio::co_spawn(io, tool.run_passive(acceptor), rethrow);
    asio::co_spawn(io, host.run_active(acceptor.local_endpoint()), rethrow);
    asio::steady_timer done(io, 300ms);
    done.async_wait([&](asio::error_code) {
        CHECK(host.state() == State::selected);
        CHECK(host.protocol().stats().linktests_sent >= 5);
        CHECK_FALSE(closed);
        host.stop();
        tool.stop();
    });
    io.run_for(10s);
    CHECK(io.stopped());
}

TEST_CASE("a thousand transactions back to back, every reply matched", "[hsms][net]") {
    asio::io_context io;
    auto acceptor = loopback_acceptor(io);
    constexpr int kMessages = 1000;
    const std::vector<std::uint8_t> map(1600, 1); // a 40x40 map's worth of body
    int replies = 0;
    int timeouts = 0;

    Session tool(io.get_executor(), fast(Role::passive), [&](Session& s, const Event& e) {
        if (const auto* m = std::get_if<DataMessage>(&e)) {
            REQUIRE(m->body.size() == map.size());
            REQUIRE(s.reply(m->header, {}).has_value());
        }
    });
    Session host(io.get_executor(), fast(Role::active), [&](Session& s, const Event& e) {
        if (std::holds_alternative<Selected>(e)) {
            REQUIRE(s.send(6, 11, true, map).has_value());
        } else if (const auto* m = std::get_if<DataMessage>(&e)) {
            REQUIRE(m->kind == DataMessage::Kind::reply);
            if (++replies < kMessages) {
                REQUIRE(s.send(6, 11, true, map).has_value());
            } else {
                s.stop();
                tool.stop();
            }
        } else if (std::holds_alternative<ReplyTimeout>(e)) {
            ++timeouts;
        }
    });

    asio::co_spawn(io, tool.run_passive(acceptor), rethrow);
    asio::co_spawn(io, host.run_active(acceptor.local_endpoint()), rethrow);
    io.run_for(30s);

    CHECK(io.stopped());
    CHECK(replies == kMessages);
    CHECK(timeouts == 0);
    CHECK(tool.protocol().stats().data_received == kMessages);
}
