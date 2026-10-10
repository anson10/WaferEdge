// HSMS: transactions per second and allocations per transaction, for the protocol engine
// alone (in memory) and for two Sessions over TCP on 127.0.0.1 (one thread, one
// io_context). The TCP case also reports round-trip percentiles from a 1 us histogram.
// BM_gem_cycle runs the GEM loop in memory: a 40x40 wafer report and its S6F12, an S2F41
// HOLD and its S2F42, through gem::Equipment, gem::Host and two engines.
//
//   build/release/bench/bench-hsms
//
// A transaction is a primary with the W-bit (a body of range(0) bytes: 0, a 40x40 map,
// a 200x200 map) and its empty reply. Closed loop: the next primary goes out when the reply
// is in, so the TCP numbers are round-trip service times on an idle link, not latency under
// load (no coordinated-omission correction applies; phase 4 measures under load).
#include "support/alloc_counter.hpp"
#include "support/synth.hpp"
#include "waferedge/gem/equipment.hpp"
#include "waferedge/gem/host.hpp"
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

void BM_gem_cycle(benchmark::State& state) {
    TimePoint now = TimePoint{} + 1h;
    Config host_config;
    host_config.role = Role::active;
    Protocol tool_hsms(Config{});
    Protocol host_hsms(host_config);
    gem::ProtocolLink tool_link(tool_hsms, now);
    gem::ProtocolLink host_link(host_hsms, now);
    gem::Equipment equipment(gem::EquipmentConfig{});
    gem::Host host;
    std::vector<std::uint8_t> to_tool;
    std::vector<std::uint8_t> to_host;
    std::optional<Header> command;
    const auto pump = [&] {
        for (int i = 0; i < 3; ++i) {
            pass(host_hsms, tool_hsms, to_tool, now);
            while (auto e = tool_hsms.poll(now)) {
                equipment.on_hsms(*e, tool_link, now);
                while (auto g = equipment.poll()) {
                    if (const auto* c = std::get_if<gem::LotCommandReceived>(&*g)) {
                        command = c->primary;
                    }
                }
            }
            pass(tool_hsms, host_hsms, to_host, now);
            while (auto e = host_hsms.poll(now)) {
                host.on_hsms(*e, host_link, now);
                while (auto g = host.poll()) {
                    benchmark::DoNotOptimize(g);
                }
            }
        }
    };
    host_hsms.on_connected(now);
    tool_hsms.on_connected(now);
    pump();
    equipment.go_online(tool_link);
    pump();
    const WaferMap map = synth::random_map(40, 40, 100, 1);
    const auto cycle = [&] {
        (void)equipment.report_wafer(tool_link, "LOT-0042", 17, map);
        (void)host.command(host_link, {gem::LotAction::hold, "LOT-0042"});
        pump();
        if (command) {
            (void)equipment.answer(tool_link, *command, gem::Hcack::done);
            command.reset();
        }
        pump();
    };
    for (int i = 0; i < 3; ++i) {
        cycle();
    }
    const auto before = alloc::count();
    for (auto _ : state) {
        cycle();
    }
    report(state, alloc::count() - before, map.view().bins().size());
}

BENCHMARK(BM_protocol_transaction)->Arg(0)->Arg(1600)->Arg(40000);
BENCHMARK(BM_tcp_transaction)->Arg(0)->Arg(1600)->Arg(40000)->UseRealTime();
BENCHMARK(BM_gem_cycle);

} // namespace

BENCHMARK_MAIN();
