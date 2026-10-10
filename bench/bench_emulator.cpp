// The tool emulator as a load generator: at each rate, how many wafers per second it really
// sends, and how late each goes out against its scheduled time (actual - scheduled, from a
// 1 us histogram). That lateness is a floor under every latency phase 4 measures with it.
//
//   build/release/bench/bench-emulator
//
// The emulator and a gem::Host run in one process on two threads with their own io_context,
// over TCP on 127.0.0.1; every wafer is a 40x40 map (the host decodes and acknowledges it).
#include "support/synth.hpp"
#include "waferedge/emulator/tool.hpp"
#include "waferedge/gem/host.hpp"

#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/io_context.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <format>
#include <optional>
#include <thread>
#include <vector>

using namespace waferedge;
using namespace std::chrono_literals;

namespace {

struct Result {
    double rate = 0;
    std::size_t sent = 0;
    double achieved = 0;                        // wafers/s between the first and the last send
    double p50 = 0, p99 = 0, p999 = 0, max = 0; // lateness, us
    std::size_t window_full = 0;                // reports that waited for a free slot
};

Result run(double rate, std::size_t n) {
    const WaferMap map = synth::random_map(40, 40, 100, 1);
    std::vector<WaferRecord> records;
    records.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        records.push_back({static_cast<std::int32_t>(i), static_cast<std::int32_t>(i / 25),
                           static_cast<std::int64_t>(i), Pattern::unknown, Split::unsplit, map});
    }

    asio::io_context tool_io(1);
    asio::ip::tcp::acceptor acceptor(tool_io, {asio::ip::address_v4::loopback(), 0});
    emulator::ToolConfig config;
    config.replay.rate = rate;
    emulator::Tool tool(tool_io.get_executor(), records, config);
    asio::co_spawn(tool_io, tool.run(acceptor), asio::detached);

    asio::io_context host_io(1);
    gem::Host host;
    std::optional<gem::SessionLink> link;
    hsms::Config active;
    active.role = hsms::Role::active;
    hsms::Session session(host_io.get_executor(), active,
                          [&](hsms::Session& s, const hsms::Event& e) {
                              host.on_hsms(e, *link, hsms::Clock::now());
                              while (auto g = host.poll()) {
                                  if (std::holds_alternative<gem::NotCommunicating>(*g)) {
                                      s.stop();
                                  }
                              }
                          });
    link.emplace(session);
    asio::co_spawn(host_io, session.run_active(acceptor.local_endpoint()), asio::detached);

    std::thread host_thread([&] { host_io.run(); });
    tool_io.run();
    host_thread.join();

    const auto& replay = tool.replay();
    std::array<std::uint64_t, 100'000> histogram{}; // 1 us buckets up to 100 ms
    std::optional<hsms::TimePoint> first;
    hsms::TimePoint last{};
    Result r{rate, replay.stats().sent};
    r.window_full = tool.window_full();
    for (std::size_t i = 0; i < replay.size(); ++i) {
        const auto& at = replay.sent_at(i);
        if (!at) {
            continue;
        }
        if (!first) {
            first = *at;
        }
        last = *at;
        const auto late =
            std::chrono::duration_cast<std::chrono::microseconds>(*at - *replay.scheduled(i));
        const auto us = static_cast<std::size_t>(std::max<std::int64_t>(0, late.count()));
        ++histogram[std::min(us, histogram.size() - 1)];
        r.max = std::max(r.max, static_cast<double>(late.count()));
    }
    const auto percentile = [&](double p) {
        const auto target = static_cast<std::uint64_t>(p * static_cast<double>(r.sent));
        std::uint64_t seen = 0;
        for (std::size_t i = 0; i < histogram.size(); ++i) {
            seen += histogram[i];
            if (seen > target) {
                return static_cast<double>(i);
            }
        }
        return static_cast<double>(histogram.size());
    };
    r.p50 = percentile(0.5);
    r.p99 = percentile(0.99);
    r.p999 = percentile(0.999);
    if (first && last > *first) {
        r.achieved =
            static_cast<double>(r.sent - 1) / std::chrono::duration<double>(last - *first).count();
    }
    return r;
}

} // namespace

int main() {
    std::puts("| Target wafers/s | Sent | Achieved wafers/s | Late p50 | p99 | p99.9 | max | "
              "Window full |");
    std::puts("|---|---|---|---|---|---|---|---|");
    for (const double rate : {100.0, 1'000.0, 10'000.0, 50'000.0, 200'000.0}) {
        const auto n = static_cast<std::size_t>(std::min(rate * 3, 300'000.0)); // ~3 s each
        const Result r = run(rate, n);
        std::puts(std::format("| {:.0f} | {} | {:.0f} | {:.0f} us | {:.0f} us | {:.0f} us | "
                              "{:.0f} us | {} |",
                              r.rate, r.sent, r.achieved, r.p50, r.p99, r.p999, r.max,
                              r.window_full)
                      .c_str());
        std::fflush(stdout);
    }
}
