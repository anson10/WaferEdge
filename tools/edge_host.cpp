// waferedge-edge: the edge host. Connects to a tool (active HSMS, GEM host), classifies every
// wafer report on analytics threads, and holds a lot (S2F41) once k of its last W wafers show
// the same signature (docs/pipeline.md, ADR-0014).
//
//   build/release/tools/tool-emulator data/waferlens_demo.wmap --port 5000 --rate 50 &
//   build/release/tools/waferedge-edge --port 5000 --k 3 --window 5
//
// Runs until the tool leaves (or Ctrl-C), then prints what it received, what the classifier
// found, and each stage's median time for the holds.
#include "waferedge/pipeline/edge_host.hpp"

#include <asio/co_spawn.hpp>
#include <asio/io_context.hpp>
#include <asio/signal_set.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <format>
#include <string>
#include <string_view>
#include <vector>

using namespace waferedge;
using namespace waferedge::pipeline;

namespace {

template <typename T>
bool parse(std::string_view text, T& out) {
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc{} && end == text.data() + text.size();
}

// "2,4,6" -> {2, 4, 6}.
bool parse_list(std::string_view text, std::vector<int>& out) {
    out.clear();
    while (!text.empty()) {
        const auto comma = text.find(',');
        int v = 0;
        if (!parse(text.substr(0, comma), v)) {
            return false;
        }
        out.push_back(v);
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
    }
    return !out.empty();
}

// std::println is GCC 14 (libc++ 18 has it, so this can't share its name).
template <typename... Args>
void say(std::FILE* to, std::format_string<Args...> fmt, Args&&... args) {
    std::fputs(std::format(fmt, std::forward<Args>(args)...).c_str(), to);
    std::fputc('\n', to);
}

int usage() {
    std::fputs(
        "usage: waferedge-edge [--host ADDR] [--port N] [--rules FILE] [--k N] [--window N]\n"
        "                      [--workers N] [--spin-us N] [--cpus NET,W1,W2,...]\n"
        "                      [--keep-running]\n"
        "  --host, --port  the tool to connect to (default 127.0.0.1:5000)\n"
        "  --rules         classifier thresholds (default config/rules.txt)\n"
        "  --k, --window   hold after k wafers of one lot with the same signature among its\n"
        "                  last `window` (default 3 of 5)\n"
        "  --workers       analytics threads (default 1)\n"
        "  --spin-us       a worker's busy-wait before it sleeps (default 50)\n"
        "  --cpus          pin the network thread, then each worker, to these CPUs\n"
        "  --keep-running  reconnect when the tool leaves, until Ctrl-C\n",
        stderr);
    return 2;
}

double seconds_since(TimePoint t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

// Median of a stage over the holds that reached it, in microseconds.
template <typename Stage>
std::string median_us(std::span<const HoldRecord> holds, Stage stage) {
    std::vector<double> v;
    for (const auto& h : holds) {
        if (h.state == HoldRecord::State::acked) {
            v.push_back(std::chrono::duration<double, std::micro>(stage(h)).count());
        }
    }
    if (v.empty()) {
        return "-";
    }
    std::ranges::nth_element(v, v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2));
    return std::format("{:.0f} µs", v[v.size() / 2]);
}

} // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    unsigned short port = 5000;
    std::string rules_path = "config/rules.txt";
    std::vector<int> cpus;
    EdgeHostConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const bool has_value = i + 1 < argc;
        const std::string_view value = has_value ? argv[i + 1] : "";
        unsigned spin = 0;
        if (arg == "--keep-running") {
            config.stop_when_tool_leaves = false;
            continue;
        }
        if (!has_value) {
            return usage();
        }
        ++i;
        bool ok = true;
        if (arg == "--host") {
            host = value;
        } else if (arg == "--rules") {
            rules_path = value;
        } else if (arg == "--port") {
            ok = parse(value, port);
        } else if (arg == "--k") {
            ok = parse(value, config.edge.decision.k);
        } else if (arg == "--window") {
            ok = parse(value, config.edge.decision.window);
        } else if (arg == "--workers") {
            ok = parse(value, config.workers) && config.workers >= 1;
        } else if (arg == "--spin-us") {
            ok = parse(value, spin);
            config.analyzer.spin = std::chrono::microseconds(spin);
        } else if (arg == "--cpus") {
            ok = parse_list(value, cpus);
        } else {
            ok = false;
        }
        if (!ok) {
            return usage();
        }
    }
    if (!DecisionRule::valid(config.edge.decision)) {
        say(stderr, "waferedge-edge: need 1 <= k <= window <= {}", DecisionRule::kMaxWindow);
        return 2;
    }
    if (cpus.size() > 1) {
        config.worker_cpus.assign(cpus.begin() + 1, cpus.end());
    }
    const auto classifier = RuleClassifier::load(rules_path);
    if (!classifier) {
        say(stderr, "waferedge-edge: {}", classifier.error());
        return 1;
    }
    asio::error_code ec;
    const auto address = asio::ip::make_address(host, ec);
    if (ec) {
        say(stderr, "waferedge-edge: bad address {}", host);
        return 2;
    }
    if (!cpus.empty() && !pin_current_thread(cpus.front())) {
        say(stderr, "waferedge-edge: could not pin the network thread to CPU {}", cpus.front());
    }
    say(stdout, "waferedge-edge: tool {}:{}, hold after {} of {} wafers, {} worker(s)", host, port,
        config.edge.decision.k, config.edge.decision.window, config.workers);

    asio::io_context io(1);
    const auto start = Clock::now();
    EdgeHost edge(io.get_executor(), *classifier, config, [&](std::string_view line) {
        say(stdout, "[{:9.3f} s] {}", seconds_since(start), line);
    });
    asio::signal_set signals(io, SIGINT, SIGTERM);
    signals.async_wait([&](const asio::error_code& e, int) {
        if (!e) {
            edge.stop();
        }
    });
    asio::co_spawn(io, edge.run({address, port}), [&](const std::exception_ptr& e) {
        signals.cancel();
        if (e) {
            std::rethrow_exception(e);
        }
    });
    io.run();

    const auto& s = edge.core().stats();
    say(stdout,
        "received {} wafers: {} analysed, {} dropped (rings full), {} oversized, {} bad lot "
        "ids; {:.1f} s",
        s.received, s.verdicts, s.dropped, s.oversized, s.bad_lot, seconds_since(start));
    std::string found;
    for (std::size_t p = 0; p < kPatternCount; ++p) {
        if (s.by_pattern[p] > 0) {
            found += std::format(" {} {}", pattern_name(static_cast<Pattern>(p)), s.by_pattern[p]);
        }
    }
    say(stdout, "classified:{}", found);
    const auto holds = edge.core().holds();
    const auto acked = std::ranges::count_if(
        holds, [](const HoldRecord& h) { return h.state == HoldRecord::State::acked; });
    say(stdout, "holds: {} decided, {} answered by the tool", s.holds, acked);
    // Median per stage of the answered holds; phase 4's latency measurement replaces this
    // with full histograms from the tool's schedule.
    say(stdout, "  received -> analysed {}, -> decided {}, -> S2F41 sent {}, -> S2F42 {}",
        median_us(holds, [](const HoldRecord& h) { return h.analysed - h.received; }),
        median_us(holds, [](const HoldRecord& h) { return h.decided - h.analysed; }),
        median_us(holds, [](const HoldRecord& h) { return h.sent - h.decided; }),
        median_us(holds, [](const HoldRecord& h) { return h.acked - h.sent; }));
    return 0;
}
