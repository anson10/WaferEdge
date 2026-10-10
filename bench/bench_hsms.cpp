// HSMS: transactions per second and allocations per transaction, for the protocol engine
// alone (in memory) and for two Sessions over TCP on 127.0.0.1 (one thread, one
// io_context). The TCP case also reports round-trip percentiles from a 1 us histogram.
//
//   build/release/bench/bench-hsms
//
// A transaction is a primary with the W-bit (a body of range(0) bytes: 0, a 40x40 map,
// a 200x200 map) and its empty reply. Closed loop: the next primary goes out when the reply
// is in, so the TCP numbers are round-trip service times on an idle link, not latency under
// load (no coordinated-omission correction applies; phase 4 measures under load).
#include "support/alloc_counter.hpp"
#include "waferedge/hsms/protocol.hpp"
#include "waferedge/hsms/session.hpp"

#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/io_context.hpp>
#include <benchmark/benchmark.h>

#include <array>
#include <cstring>
#include <vector>

namespace {

using namespace waferedge;
using namespace waferedge::hsms;
using namespace std::chrono_literals;

void pass(Protocol& from, Protocol& to, std::vector<std::uint8_t>& wire, TimePoint now) {
    from.take_output(wire);
    auto buffer = to.receive_buffer(wire.size());
    std::memcpy(buffer.data(), wire.data(), wire.size());
    to.on_received(wire.size(), now);
}

void report(benchmark::State& state, std::uint64_t allocations, std::size_t body) {
    const auto n = static_cast<std::int64_t>(state.iterations());
    state.SetItemsProcessed(n);
    state.counters["allocs/txn"] =
        benchmark::Counter(static_cast<double>(allocations), benchmark::Counter::kAvgIterations);
    state.counters["body"] = static_cast<double>(body);
}

void BM_protocol_transaction(benchmark::State& state) {
    const std::vector<std::uint8_t> body(static_cast<std::size_t>(state.range(0)), 1);
    Config host_config;
    host_config.role = Role::active;
    Protocol host(host_config);
    Protocol tool(Config{});
    std::vector<std::uint8_t> to_tool;
    std::vector<std::uint8_t> to_host;
    const TimePoint t = TimePoint{} + 1h;
    host.on_connected(t);
    tool.on_connected(t);
    pass(host, tool, to_tool, t);
    (void)tool.poll(t);
    pass(tool, host, to_host, t);
    (void)host.poll(t);
    const auto transaction = [&] {
        (void)host.send(6, 11, true, body, t);
        pass(host, tool, to_tool, t);
        while (auto e = tool.poll(t)) {
            (void)tool.reply(std::get<DataMessage>(*e).header, {});
        }
        pass(tool, host, to_host, t);
        while (auto e = host.poll(t)) {
            benchmark::DoNotOptimize(e);
        }
    };
    for (int i = 0; i < 3; ++i) {
        transaction();
    }
    const auto before = alloc::count();
    for (auto _ : state) {
        transaction();
    }
    report(state, alloc::count() - before, body.size());
}

void BM_tcp_transaction(benchmark::State& state) {
    const std::vector<std::uint8_t> body(static_cast<std::size_t>(state.range(0)), 1);
    asio::io_context io(1);
    asio::ip::tcp::acceptor acceptor(io, {asio::ip::address_v4::loopback(), 0});
    int replies = 0;
    Session tool(io.get_executor(), Config{}, [](Session& s, const Event& e) {
        if (const auto* m = std::get_if<DataMessage>(&e)) {
            (void)s.reply(m->header, {});
        }
    });
    Config host_config;
    host_config.role = Role::active;
    Session host(io.get_executor(), host_config, [&](Session&, const Event& e) {
        replies += std::holds_alternative<DataMessage>(e) ? 1 : 0;
    });
    asio::co_spawn(io, tool.run_passive(acceptor), asio::detached);
    asio::co_spawn(io, host.run_active(acceptor.local_endpoint()), asio::detached);
    while (host.state() != State::selected) {
        io.run_one();
    }
    const auto transaction = [&] {
        const int before = replies;
        (void)host.send(6, 11, true, body);
        while (replies == before) {
            io.run_one();
        }
    };
    for (int i = 0; i < 100; ++i) {
        transaction();
    }

    // 1 us buckets up to 4 ms; the last bucket collects anything slower.
    std::array<std::uint64_t, 4096> histogram{};
    const auto before = alloc::count();
    for (auto _ : state) {
        const auto start = Clock::now();
        transaction();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start);
        ++histogram[std::min<std::size_t>(static_cast<std::size_t>(us.count()),
                                          histogram.size() - 1)];
    }
    report(state, alloc::count() - before, body.size());
    const auto percentile = [&](double p) {
        const auto target = static_cast<std::uint64_t>(p * static_cast<double>(state.iterations()));
        std::uint64_t seen = 0;
        for (std::size_t i = 0; i < histogram.size(); ++i) {
            seen += histogram[i];
            if (seen > target) {
                return static_cast<double>(i);
            }
        }
        return static_cast<double>(histogram.size() - 1);
    };
    state.counters["p50_us"] = percentile(0.50);
    state.counters["p99_us"] = percentile(0.99);
    state.counters["p99.9_us"] = percentile(0.999);

    host.stop();
    tool.stop();
    io.run_for(1s);
}

BENCHMARK(BM_protocol_transaction)->Arg(0)->Arg(1600)->Arg(40000);
BENCHMARK(BM_tcp_transaction)->Arg(0)->Arg(1600)->Arg(40000)->UseRealTime();

} // namespace

BENCHMARK_MAIN();
