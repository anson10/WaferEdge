// HSMS steady state allocates nothing: the protocol engine once its buffers have grown, and
// the Asio session (coroutine frames recycled by Asio's per-thread cache) per transaction.
#include "support/alloc_counter.hpp"
#include "waferedge/hsms/protocol.hpp"
#include "waferedge/hsms/session.hpp"

#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/io_context.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <vector>

using namespace waferedge;
using namespace waferedge::hsms;
using namespace std::chrono_literals;

namespace {

void pass(Protocol& from, Protocol& to, std::vector<std::uint8_t>& wire, TimePoint now) {
    from.take_output(wire);
    auto buffer = to.receive_buffer(wire.size());
    std::memcpy(buffer.data(), wire.data(), wire.size());
    to.on_received(wire.size(), now);
}

} // namespace

TEST_CASE("the protocol engine: a transaction allocates nothing once warm", "[hsms]") {
    Config host_config;
    host_config.role = Role::active;
    Protocol host(host_config);
    Protocol tool(Config{});
    // One wire per direction, as each Session has its own write buffer: take_output swaps
    // vectors, so a shared one would carry a small buffer into the big direction.
    std::vector<std::uint8_t> to_tool;
    std::vector<std::uint8_t> to_host;
    const std::vector<std::uint8_t> map(1600, 1);
    const TimePoint t0 = TimePoint{} + 1h;
    host.on_connected(t0);
    tool.on_connected(t0);
    pass(host, tool, to_tool, t0);
    (void)tool.poll(t0);
    pass(tool, host, to_host, t0);
    (void)host.poll(t0);

    bool ok = true;
    const auto round_trip = [&] {
        ok = ok && host.send(6, 11, true, map, t0).has_value();
        pass(host, tool, to_tool, t0);
        while (auto e = tool.poll(t0)) {
            if (const auto* m = std::get_if<DataMessage>(&*e)) {
                ok = ok && tool.reply(m->header, {}).has_value();
            }
        }
        pass(tool, host, to_host, t0);
        while (auto e = host.poll(t0)) {
            ok = ok && std::holds_alternative<DataMessage>(*e);
        }
    };
    for (int i = 0; i < 3; ++i) {
        round_trip(); // warm-up: both buffers of each swap pair grow to the message size
    }
    const auto before = alloc::count();
    for (int i = 0; i < 100; ++i) {
        round_trip();
    }
    const auto allocations = alloc::count() - before;
    CHECK(ok);
    CHECK(allocations == 0);
}

TEST_CASE("the TCP session: transactions after warm-up allocate nothing", "[hsms][net]") {
    asio::io_context io(1);
    asio::ip::tcp::acceptor acceptor(io, {asio::ip::address_v4::loopback(), 0});
    const std::vector<std::uint8_t> map(1600, 1);
    int replies = 0;
    Session tool(io.get_executor(), Config{}, [&](Session& s, const Event& e) {
        if (const auto* m = std::get_if<DataMessage>(&e)) {
            (void)s.reply(m->header, {});
        }
    });
    Config host_config;
    host_config.role = Role::active;
    Session host(io.get_executor(), host_config, [&](Session&, const Event& e) {
        if (std::holds_alternative<DataMessage>(e)) {
            ++replies;
        }
    });
    asio::co_spawn(io, tool.run_passive(acceptor), asio::detached);
    asio::co_spawn(io, host.run_active(acceptor.local_endpoint()), asio::detached);
    while (host.state() != State::selected) {
        io.run_one();
    }
    const auto transaction = [&] {
        const int before = replies;
        (void)host.send(6, 11, true, map);
        while (replies == before) {
            io.run_one();
        }
    };
    for (int i = 0; i < 10; ++i) {
        transaction(); // warm-up: buffers and Asio's recycled frames
    }
    const auto before = alloc::count();
    for (int i = 0; i < 100; ++i) {
        transaction();
    }
    const auto allocations = alloc::count() - before;
    CHECK(allocations == 0);

    host.stop();
    tool.stop();
    io.run_for(5s);
}
