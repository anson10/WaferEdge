// The closed loop's latency, sort result to decision and to HOLD, by load (ADR-0015).
//
//   build/release/bench/bench-latency [--maps FILE.wmap] [--rates 100,1000,10000]
//                                     [--seconds S] [--workers N] [--spin-us N]
//                                     [--k K] [--window W] [--cpus TOOL,NET,W1,...]
//                                     [--detector SPEC] [--batch B] [--deadline-us D]
//                                     [--model FILE.wcnn] [--scales FILE.txt]
//
// --detector picks what the workers run (ADR-0016): rules (default), cnn-cpu, rules-gpu,
// cnn-fp32 / cnn-fp16 / cnn-int8, or two joined by '+' (the first decides, the agreement is
// reported). Batches start at B wafers or when the oldest has waited D µs. The CNN reads
// FabEye's model from data/fabeye_cnn.wcnn (tools/export_cnn.py), int8 also its scales.
//
// The tool emulator and the edge host run in one process: the emulator on this thread, the
// edge host's network thread on another, its analytics workers on their own, all on one
// steady clock, talking HSMS over TCP on 127.0.0.1. For every wafer:
//
//   scheduled  the emulator's open-loop slot (start + k / rate): the time the tool sorted it
//   sent       when the emulator really sent it (lateness = sent - scheduled)
//   received   the edge host decoded the S6F11
//   analysed   a worker classified the map
//   decided    the network thread took the verdict: a HOLD, if due, goes out now
//
// Latency is measured from `scheduled`, not `sent`, so a stall anywhere (emulator, HSMS
// window, rings, workers) shows in the wafers queued behind it: no coordinated omission.
// For each hold, sort -> HOLD is the triggering wafer's scheduled time to the tool receiving
// the S2F41. Percentiles come from LatencyHistogram (within 1/128, never under-reported).
//
// Without --maps the wafers are synthetic 40 x 40 maps: lots of 25, every tenth lot with a
// centre blob from its 6th wafer on (an excursion the rule holds), the rest sparse noise.
//
// Accuracy, from the same run: each verdict against its map's truth (wafers with unknown
// truth skipped), and the holds against the lots: a lot is an excursion if any of its
// replayed wafers has a pattern for truth. Precision: held lots that are excursions; recall:
// excursion lots that were held.
#include "support/synth.hpp"
#include "waferedge/emulator/tool.hpp"
#include "waferedge/pipeline/edge_host.hpp"
#include "waferedge/pipeline/histogram.hpp"

#include <asio/co_spawn.hpp>
#include <asio/io_context.hpp>

#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace waferedge;
using namespace waferedge::pipeline;
using namespace std::chrono_literals;

namespace {

template <typename T>
bool parse(std::string_view text, T& out) {
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc{} && end == text.data() + text.size();
}

// libc++ 18 has no floating-point from_chars.
bool parse(std::string_view text, double& out) {
    const std::string copy(text); // strtod needs a terminated string
    char* end = nullptr;
    out = std::strtod(copy.c_str(), &end);
    return !copy.empty() && end == copy.c_str() + copy.size();
}

template <typename T>
bool parse_list(std::string_view text, std::vector<T>& out) {
    out.clear();
    while (!text.empty()) {
        const auto comma = text.find(',');
        T v{};
        if (!parse(text.substr(0, comma), v)) {
            return false;
        }
        out.push_back(v);
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
    }
    return !out.empty();
}

int usage() {
    std::fputs("usage: bench-latency [--maps FILE.wmap] [--rates R1,R2,...] [--seconds S]\n"
               "                     [--workers N] [--spin-us N] [--k K] [--window W]\n"
               "                     [--cpus TOOL,NET,W1,...]\n",
               stderr);
    return 2;
}

std::uint64_t ns(Clock::duration d) {
    return d.count() <= 0 ? 0
                          : static_cast<std::uint64_t>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
}

std::string us(std::uint64_t v) {
    const double x = static_cast<double>(v) / 1e3;
    if (x >= 1'000'000) {
        return std::format("{:.2f} s", x / 1e6);
    }
    if (x >= 1'000) {
        return std::format("{:.1f} ms", x / 1e3);
    }
    if (x >= 100) {
        return std::format("{:.0f} µs", x);
    }
    return std::format("{:.1f} µs", x);
}

// Synthetic wafers: a few distinct maps, shared by many records (views).
struct Synthetic {
    std::vector<WaferMap> noise;
    std::vector<WaferMap> blobs;
    std::vector<WaferRecord> records;

    explicit Synthetic(std::size_t n) {
        for (std::uint32_t s = 0; s < 32; ++s) {
            noise.push_back(synth::random_map(40, 40, 15, s + 1));
            auto b = synth::random_map(40, 40, 15, s + 101);
            const double r2 = 0.02 + 0.002 * s;
            synth::paint(b, [&](int r, int c) { return synth::rho2(r, c, 40, 40) < r2; });
            blobs.push_back(std::move(b));
        }
        records.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t lot = i / 25;
            const bool excursion = lot % 10 == 3 && i % 25 >= 5;
            const auto& maps = excursion ? blobs : noise;
            records.push_back({static_cast<std::int32_t>(i), static_cast<std::int32_t>(lot),
                               static_cast<std::int64_t>(i),
                               excursion ? Pattern::center : Pattern::none, Split::unsplit,
                               maps[i % maps.size()].view()});
        }
    }
};

struct Sample {
    std::uint64_t wafer = 0;
    Pattern pattern = Pattern::none;
    Pattern second = Pattern::unknown;
    TimePoint received, analysed, decided;
};

struct Row {
    double rate = 0;
    double achieved = 0;
    std::size_t sent = 0, decided = 0, dropped = 0, window_full = 0;
    LatencyHistogram late, to_host, analyse, decide, total, hold;
    std::size_t holds = 0, holds_answered = 0;
    // Accuracy.
    std::size_t scored = 0, correct = 0;     // verdicts with a known truth, and right ones
    std::size_t paired = 0, agree = 0;       // verdicts with a second detector's pattern
    std::size_t lots_bad = 0, held_bad = 0;  // excursion lots, and those held
    std::size_t held_lots = 0;               // lots held (recorded)
    std::uint64_t batches = 0, failures = 0; // detector calls, and failed ones
};

struct Options {
    std::vector<double> rates{100, 1'000, 5'000, 10'000, 20'000};
    double seconds = 3;
    EdgeHostConfig edge;
    std::string detector = "rules";
    std::vector<int> cpus; // tool, network, workers...
};

Row run(std::span<const WaferRecord> records, double rate, std::size_t n, const Options& o,
        const RuleClassifier& classifier) {
    Row row;
    row.rate = rate;

    asio::io_context tool_io(1);
    asio::ip::tcp::acceptor acceptor(tool_io, {asio::ip::address_v4::loopback(), 0});
    emulator::ToolConfig tool_config;
    tool_config.replay.rate = rate;
    tool_config.replay.limit = n;
    tool_config.linger = 300ms; // the last lots' HOLDs still land
    emulator::Tool tool(tool_io.get_executor(), records, tool_config);

    asio::io_context edge_io(1);
    EdgeHostConfig edge_config = o.edge;
    if (o.cpus.size() > 2) {
        edge_config.worker_cpus.assign(o.cpus.begin() + 2, o.cpus.end());
    }
    EdgeHost edge(edge_io.get_executor(), classifier, edge_config);
    std::vector<Sample> samples;
    samples.reserve(n);
    edge.set_observer([&](const Verdict& v, TimePoint decided) {
        if (samples.size() < samples.capacity()) {
            samples.push_back({v.wafer, v.pattern, v.second, v.received, v.analysed, decided});
        }
    });

    const auto rethrow = [](const std::exception_ptr& e) {
        if (e) {
            std::rethrow_exception(e);
        }
    };
    asio::co_spawn(tool_io, tool.run(acceptor), rethrow);
    asio::co_spawn(edge_io, edge.run(acceptor.local_endpoint()), rethrow);
    std::thread edge_thread([&] {
        if (o.cpus.size() > 1) {
            (void)pin_current_thread(o.cpus[1]);
        }
        edge_io.run();
    });
    if (!o.cpus.empty()) {
        (void)pin_current_thread(o.cpus[0]);
    }
    tool_io.run();
    edge_thread.join();

    // Join the edge host's samples with the emulator's schedule by wafer id.
    const auto& replay = tool.replay();
    std::unordered_map<std::uint64_t, std::size_t> index;
    index.reserve(replay.size());
    for (std::size_t i = 0; i < replay.size(); ++i) {
        index.emplace(static_cast<std::uint64_t>(replay.record(i).wafer_id), i);
    }
    std::optional<TimePoint> first;
    TimePoint last{};
    for (std::size_t i = 0; i < replay.size(); ++i) {
        const auto& sent = replay.sent_at(i);
        if (!sent) {
            continue;
        }
        first = first ? std::min(*first, *sent) : *sent;
        last = std::max(last, *sent);
        row.late.record(ns(*sent - *replay.scheduled(i)));
    }
    for (const Sample& s : samples) {
        const auto it = index.find(s.wafer);
        if (it == index.end() || !replay.scheduled(it->second)) {
            continue;
        }
        const TimePoint scheduled = *replay.scheduled(it->second);
        const Pattern truth = replay.record(it->second).truth;
        if (truth != Pattern::unknown) {
            ++row.scored;
            row.correct += s.pattern == truth ? 1U : 0U;
        }
        if (s.second != Pattern::unknown) {
            ++row.paired;
            row.agree += s.pattern == s.second ? 1U : 0U;
        }
        row.to_host.record(ns(s.received - scheduled));
        row.analyse.record(ns(s.analysed - s.received));
        row.decide.record(ns(s.decided - s.analysed));
        row.total.record(ns(s.decided - scheduled));
    }
    for (const HoldRecord& h : edge.core().holds()) {
        ++row.holds;
        const auto at_tool = replay.hold_time(h.lot.view());
        const auto it = index.find(h.wafer);
        if (h.state != HoldRecord::State::acked || !at_tool || it == index.end() ||
            !replay.scheduled(it->second)) {
            continue;
        }
        ++row.holds_answered;
        row.hold.record(ns(*at_tool - *replay.scheduled(it->second)));
    }
    // Lots: an excursion if any replayed wafer has a pattern for truth.
    std::unordered_set<std::string_view> bad;
    for (std::size_t i = 0; i < std::min(n, replay.size()); ++i) {
        const Pattern truth = replay.record(i).truth;
        if (truth != Pattern::none && truth != Pattern::unknown) {
            bad.insert(replay.lot_of(i));
        }
    }
    std::unordered_set<std::string> held;
    for (const HoldRecord& h : edge.core().holds()) {
        held.emplace(h.lot.view());
    }
    row.lots_bad = bad.size();
    row.held_lots = held.size();
    for (const auto& lot : held) {
        row.held_bad += bad.contains(lot) ? 1U : 0U;
    }
    for (const auto& a : edge.analyzers()) {
        row.batches += a->batches();
        row.failures += a->failures();
    }
    row.sent = replay.stats().sent;
    row.decided = row.total.count();
    row.dropped = edge.core().stats().dropped;
    row.window_full = tool.window_full();
    if (first && last > *first) {
        row.achieved = static_cast<double>(row.sent - 1) /
                       std::chrono::duration<double>(last - *first).count();
    }
    return row;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    std::string maps_path;
    std::string model_path = WAFEREDGE_SOURCE_DIR "/data/fabeye_cnn.wcnn";
    std::string scales_path = WAFEREDGE_SOURCE_DIR "/data/fabeye_int8_scales.txt";
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (i + 1 >= argc) {
            return usage();
        }
        const std::string_view value = argv[++i];
        unsigned spin = 0;
        bool ok = true;
        if (arg == "--maps") {
            maps_path = value;
        } else if (arg == "--rates") {
            ok = parse_list(value, o.rates);
        } else if (arg == "--seconds") {
            ok = parse(value, o.seconds) && o.seconds > 0;
        } else if (arg == "--workers") {
            ok = parse(value, o.edge.workers) && o.edge.workers >= 1;
        } else if (arg == "--spin-us") {
            ok = parse(value, spin);
            o.edge.analyzer.spin = std::chrono::microseconds(spin);
        } else if (arg == "--k") {
            ok = parse(value, o.edge.edge.decision.k);
        } else if (arg == "--window") {
            ok = parse(value, o.edge.edge.decision.window);
        } else if (arg == "--cpus") {
            ok = parse_list(value, o.cpus);
        } else if (arg == "--detector") {
            o.detector = value;
        } else if (arg == "--batch") {
            ok = parse(value, o.edge.analyzer.batch) && o.edge.analyzer.batch >= 1 &&
                 o.edge.analyzer.batch <= kMaxBatch;
        } else if (arg == "--deadline-us") {
            unsigned d = 0;
            ok = parse(value, d);
            o.edge.analyzer.deadline = std::chrono::microseconds(d);
        } else if (arg == "--model") {
            model_path = value;
        } else if (arg == "--scales") {
            scales_path = value;
        } else {
            ok = false;
        }
        if (!ok) {
            return usage();
        }
    }
    if (!DecisionRule::valid(o.edge.edge.decision)) {
        return usage();
    }
    const auto classifier = RuleClassifier::load(WAFEREDGE_SOURCE_DIR "/config/rules.txt");
    if (!classifier) {
        std::fputs(std::format("bench-latency: {}\n", classifier.error()).c_str(), stderr);
        return 1;
    }
    // The CNN's model and int8 scales, only if the detector needs them.
    std::optional<cnn::Model> model;
    DetectorResources resources;
    resources.rules = &*classifier;
    if (o.detector.find("cnn") != std::string::npos) {
        auto loaded = cnn::load_model(model_path);
        if (!loaded) {
            std::fputs(std::format("bench-latency: {}\n", loaded.error()).c_str(), stderr);
            return 1;
        }
        model.emplace(std::move(*loaded));
        resources.model = &*model;
        if (o.detector.find("cnn-int8") != std::string::npos) {
            resources.int8_max = gpu::read_activation_max(scales_path);
        }
    }
    auto factory = detector_factory(o.detector, resources);
    if (!factory) {
        std::fputs(std::format("bench-latency: {}\n", factory.error()).c_str(), stderr);
        return 1;
    }
    o.edge.detector = *factory;

    // Each rate runs for `seconds`, and at least 2,000 wafers (for a p99.9 worth reading).
    const auto wafers = [&](double rate) {
        return static_cast<std::size_t>(std::max(rate * o.seconds, 2'000.0));
    };
    std::size_t most = 0;
    for (const double r : o.rates) {
        most = std::max(most, wafers(r));
    }
    std::optional<MapSet> file;
    std::optional<Synthetic> synthetic;
    std::span<const WaferRecord> records;
    if (!maps_path.empty()) {
        auto loaded = MapSet::load(maps_path);
        if (!loaded) {
            std::fputs(std::format("bench-latency: {}\n", loaded.error()).c_str(), stderr);
            return 1;
        }
        file.emplace(std::move(*loaded));
        records = file->records();
    } else {
        synthetic.emplace(most);
        records = synthetic->records;
    }

    std::puts(std::format("{} wafers ({}); hold after {} of {}; detector {}, batch {}, deadline "
                          "{} µs; {} worker(s), spin {} µs{}",
                          maps_path.empty() ? "synthetic 40x40" : maps_path, records.size(),
                          o.edge.edge.decision.k, o.edge.edge.decision.window, o.detector,
                          o.edge.analyzer.batch, o.edge.analyzer.deadline.count(), o.edge.workers,
                          o.edge.analyzer.spin.count(), o.cpus.empty() ? "" : ", pinned")
                  .c_str());
    std::puts(
        "\nSort result -> decision (every wafer) and sort -> HOLD at the tool (each hold):\n");
    std::puts("| Wafers/s | Achieved | Wafers | Dropped | Decision p50 | p99 | p99.9 | max | "
              "Holds | HOLD p50 | HOLD max |");
    std::puts("|---|---|---|---|---|---|---|---|---|---|---|");
    std::vector<Row> rows;
    for (const double rate : o.rates) {
        const Row r = run(records, rate, std::min(wafers(rate), records.size()), o, *classifier);
        std::puts(std::format("| {:.0f} | {:.0f} | {} | {} | {} | {} | {} | {} | {}/{} | {} | {} |",
                              r.rate, r.achieved, r.sent, r.dropped, us(r.total.percentile(0.5)),
                              us(r.total.percentile(0.99)), us(r.total.percentile(0.999)),
                              us(r.total.max()), r.holds_answered, r.holds,
                              r.hold.count() == 0 ? "-" : us(r.hold.percentile(0.5)),
                              r.hold.count() == 0 ? "-" : us(r.hold.max()))
                      .c_str());
        std::fflush(stdout);
        rows.push_back(r);
    }
    std::puts("\nWhere the time goes, p50 / p99 per stage:\n");
    std::puts("| Wafers/s | Emulator late | Scheduled -> received | Received -> analysed | "
              "Analysed -> decided | Window full |");
    std::puts("|---|---|---|---|---|---|");
    for (const Row& r : rows) {
        const auto pair = [](const LatencyHistogram& h) {
            return std::format("{} / {}", us(h.percentile(0.5)), us(h.percentile(0.99)));
        };
        std::puts(std::format("| {:.0f} | {} | {} | {} | {} | {} |", r.rate, pair(r.late),
                              pair(r.to_host), pair(r.analyse), pair(r.decide), r.window_full)
                      .c_str());
    }
    std::puts(
        "\nAccuracy of the same runs (verdicts against truth; holds against excursion lots):\n");
    std::puts("| Wafers/s | Wafers right | Held lots | Precision | Recall | Agreement | Batches | "
              "Failed |");
    std::puts("|---|---|---|---|---|---|---|---|");
    const auto pct = [](std::size_t a, std::size_t b) {
        return b == 0 ? std::string("-")
                      : std::format("{:.1f}%",
                                    100.0 * static_cast<double>(a) / static_cast<double>(b));
    };
    for (const Row& r : rows) {
        std::puts(std::format("| {:.0f} | {} | {} | {} | {} | {} | {} | {} |", r.rate,
                              pct(r.correct, r.scored), r.held_lots, pct(r.held_bad, r.held_lots),
                              pct(r.held_bad, r.lots_bad), pct(r.agree, r.paired), r.batches,
                              r.failures)
                      .c_str());
    }
}
