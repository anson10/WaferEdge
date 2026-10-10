// waferedge-closedloop: what WaferEdge's holds would have done to WaferLens's spatial
// excursions, next to WaferLens's batch pattern alarm and to no hold (ADR-0017).
//
//   python3 tools/export_maps.py waferlens ~/waferLens/data/demo data/waferlens_demo.wmap
//       --excursions data/waferlens_excursions.csv
//   build/release/tools/waferedge-closedloop data/waferlens_demo.wmap
//       --excursions data/waferlens_excursions.csv [--k 3 --window 5] [--sweep]
//
// Every wafer is classified (--detector, default rules), then replayed in sort-time order
// through the edge host's decision rule; a hold lands --latency-us after its triggering wafer
// was sorted, and the held lot's later wafers are withheld. Per excursion, in hours after it
// began: its first patterned wafer at sort (the floor for any sort-based monitor), WaferEdge's
// first hold, WaferLens's alarm; and how many of its patterned wafers were sorted before each.
#include "waferedge/pipeline/closed_loop.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <optional>
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

// std::println is GCC 14 (libc++ 18 has it, so this can't share its name).
template <typename... Args>
void say(std::format_string<Args...> fmt, Args&&... args) {
    std::fputs(std::format(fmt, std::forward<Args>(args)...).c_str(), stdout);
    std::fputc('\n', stdout);
}

int usage() {
    std::fputs(
        "usage: waferedge-closedloop <maps.wmap> --excursions FILE [--alarms FILE]\n"
        "                            [--detector SPEC] [--k K] [--window W] [--latency-us N]\n"
        "                            [--model FILE] [--scales FILE] [--sweep]\n"
        "  --excursions  tools/export_maps.py waferlens ... --excursions FILE\n"
        "  --alarms      WaferLens's published alarm times (default\n"
        "                config/waferlens_pattern_alarms.csv)\n"
        "  --detector    as waferedge-edge (default rules)\n"
        "  --k, --window hold after k of a lot's last `window` wafers (default 3 of 5)\n"
        "  --latency-us  sort -> HOLD at the tool (default 5000, above bench-latency's p99.9)\n"
        "  --sweep       also a table over several k / window\n",
        stderr);
    return 2;
}

constexpr double kHour = 3600e6; // us

std::string hours(std::optional<double> h) {
    return h ? std::format("{:.1f}", *h) : std::string("-");
}

std::optional<double> median(std::vector<double> v) {
    if (v.empty()) {
        return std::nullopt;
    }
    std::ranges::sort(v);
    const std::size_t n = v.size();
    return n % 2 == 1 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

struct Summary {
    std::size_t held = 0;
    std::optional<double> inline_h;         // median first sorted -> hold, held excursions
    std::optional<double> start_h;          // median start -> hold
    std::size_t before_hold = 0, total = 0; // patterned wafers sorted before the hold / all
    std::size_t holds = 0, false_holds = 0;
};

Summary summarise(const ClosedLoopResult& r, const ExcursionTruth& truth) {
    Summary s;
    std::vector<double> inline_h;
    std::vector<double> start_h;
    for (std::size_t i = 0; i < r.excursions.size(); ++i) {
        const auto& e = r.excursions[i];
        s.before_hold += e.sorted_before_hold;
        s.total += e.wafers;
        if (e.hold_us && e.first_sorted_us) {
            ++s.held;
            inline_h.push_back(static_cast<double>(*e.hold_us - *e.first_sorted_us) / kHour);
            start_h.push_back(static_cast<double>(*e.hold_us - truth.excursions[i].start_us) /
                              kHour);
        }
    }
    s.inline_h = median(inline_h);
    s.start_h = median(start_h);
    s.holds = r.holds;
    s.false_holds = r.false_holds;
    return s;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return usage();
    }
    const std::string maps_path = argv[1];
    std::string excursions_path;
    std::string alarms_path = "config/waferlens_pattern_alarms.csv";
    std::string detector = "rules";
    std::string model_path = "data/fabeye_cnn.wcnn";
    std::string scales_path = "data/fabeye_int8_scales.txt";
    ClosedLoopConfig config;
    bool sweep = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--sweep") {
            sweep = true;
            continue;
        }
        if (i + 1 >= argc) {
            return usage();
        }
        const std::string_view value = argv[++i];
        bool ok = true;
        if (arg == "--excursions") {
            excursions_path = value;
        } else if (arg == "--alarms") {
            alarms_path = value;
        } else if (arg == "--detector") {
            detector = value;
        } else if (arg == "--model") {
            model_path = value;
        } else if (arg == "--scales") {
            scales_path = value;
        } else if (arg == "--k") {
            ok = parse(value, config.decision.k);
        } else if (arg == "--window") {
            ok = parse(value, config.decision.window);
        } else if (arg == "--latency-us") {
            ok = parse(value, config.latency_us) && config.latency_us >= 0;
        } else {
            ok = false;
        }
        if (!ok) {
            return usage();
        }
    }
    if (excursions_path.empty() || !DecisionRule::valid(config.decision)) {
        return usage();
    }
    const auto fail = [](const std::string& what) {
        std::fputs(std::format("waferedge-closedloop: {}\n", what).c_str(), stderr);
        return 1;
    };
    auto set = MapSet::load(maps_path);
    if (!set) {
        return fail(set.error());
    }
    const auto truth = load_excursions(excursions_path);
    if (!truth) {
        return fail(truth.error());
    }
    const auto alarms = load_waferlens_alarms(alarms_path);
    if (!alarms) {
        return fail(alarms.error());
    }
    const auto rules = RuleClassifier::load("config/rules.txt");
    if (!rules) {
        return fail(rules.error());
    }
    std::optional<cnn::Model> model;
    DetectorResources resources;
    resources.rules = &*rules;
    if (detector.find("cnn") != std::string::npos) {
        auto loaded = cnn::load_model(model_path);
        if (!loaded) {
            return fail(loaded.error());
        }
        model.emplace(std::move(*loaded));
        resources.model = &*model;
        if (detector.find("cnn-int8") != std::string::npos) {
            resources.int8_max = gpu::read_activation_max(scales_path);
        }
    }
    const auto factory = detector_factory(detector, resources);
    if (!factory) {
        return fail(factory.error());
    }
    const auto verdicts = classify_all(set->records(), *(*factory)());
    if (!verdicts) {
        return fail(verdicts.error());
    }
    const auto r = replay_closed_loop(set->records(), *verdicts, *truth, config);

    say("{}: {} wafers in {} lots, {} spatial excursions; detector {}, hold after {} of {}, "
        "hold lands {} ms after its wafer's sort",
        maps_path, set->size(), r.lots, truth->excursions.size(), detector, config.decision.k,
        config.decision.window, static_cast<double>(config.latency_us) / 1e3);
    say("");
    say("Hours after each excursion began, and its patterned wafers sorted before each:");
    say("");
    say("| Id | Pattern | Patterned wafers | First sorted (h, WaferLens) | WaferEdge hold (h) | "
        "WaferLens alarm (h) | Sorted before hold | before alarm | no hold | Lots held |");
    say("|---|---|---|---|---|---|---|---|---|---|");
    std::vector<double> alarm_inline;
    std::vector<double> alarm_start;
    std::size_t before_alarm_total = 0;
    for (std::size_t i = 0; i < r.excursions.size(); ++i) {
        const auto& e = r.excursions[i];
        const Excursion& x = truth->excursions[i];
        const auto rel = [&](std::optional<std::int64_t> t) -> std::optional<double> {
            if (!t) {
                return std::nullopt;
            }
            return static_cast<double>(*t - x.start_us) / kHour;
        };
        const auto a = std::ranges::find(*alarms, e.id, &WaferLensAlarm::id);
        const bool has_alarm = a != alarms->end();
        // The excursion's wafers sorted before WaferLens's alarm (which holds nothing back).
        std::size_t before_alarm = 0;
        if (has_alarm) {
            const auto alarm_us = x.start_us + static_cast<std::int64_t>(a->alarm_h * kHour);
            for (const auto& rec : set->records()) {
                const auto ex = truth->excursion_of.find(rec.wafer_id);
                before_alarm += ex != truth->excursion_of.end() && ex->second == e.id &&
                                        rec.tested_at_us < alarm_us
                                    ? 1U
                                    : 0U;
            }
            before_alarm_total += before_alarm;
            alarm_start.push_back(a->alarm_h);
            alarm_inline.push_back(a->alarm_h - a->first_sorted_h);
        }
        say("| {} | {} | {} | {} ({}) | {} | {} | {} | {} | {} | {} |", e.id,
            pattern_name(e.pattern), e.wafers, hours(rel(e.first_sorted_us)),
            has_alarm ? hours(a->first_sorted_h) : "-", hours(rel(e.hold_us)),
            has_alarm ? hours(a->alarm_h) : "-", e.sorted_before_hold,
            has_alarm ? std::to_string(before_alarm) : "-", e.wafers, e.lots_held);
    }
    const Summary s = summarise(r, *truth);
    say("");
    say("Excursions held: {} of {}. Median hours after the excursion began: WaferEdge's hold {}, "
        "WaferLens's alarm {}.",
        s.held, r.excursions.size(), hours(s.start_h), hours(median(alarm_start)));
    say("After the first patterned wafer reached sort (the part a monitor at sort controls): "
        "WaferEdge {} h, WaferLens {} h (medians).",
        hours(s.inline_h), hours(median(alarm_inline)));
    say("Patterned wafers sorted: {} before WaferEdge's holds, {} before WaferLens's alarms, {} "
        "with no hold.",
        s.before_hold, before_alarm_total, s.total);
    say("Holds: {} ({} false: a lot whose triggering wafer shows no excursion); wafers withheld "
        "by holds: {} of {}.",
        r.holds, r.false_holds, r.wafers_withheld, set->size());
    if (r.min_lot_gap_us) {
        say("Smallest gap between a lot's sorted wafers: {:.1f} s; the replay is the live loop "
            "while the hold's latency ({} ms) stays below it.",
            static_cast<double>(*r.min_lot_gap_us) / 1e6,
            static_cast<double>(config.latency_us) / 1e3);
    }
    say("Lots forgotten by the decision table: {}.", r.evictions);

    if (sweep) {
        say("");
        say("The hold rule, swept (same verdicts):");
        say("");
        say("| k of window | Excursions held | In-line h (median) | Patterned sorted before "
            "hold | Holds | False holds |");
        say("|---|---|---|---|---|---|");
        for (const auto& [k, w] : std::vector<std::pair<int, int>>{
                 {1, 1}, {2, 2}, {2, 3}, {2, 5}, {3, 5}, {3, 8}, {4, 8}}) {
            ClosedLoopConfig c = config;
            c.decision.k = k;
            c.decision.window = w;
            const auto rr = replay_closed_loop(set->records(), *verdicts, *truth, c);
            const Summary ss = summarise(rr, *truth);
            say("| {} of {} | {} of {} | {} | {} of {} | {} | {} |", k, w, ss.held,
                rr.excursions.size(), hours(ss.inline_h), ss.before_hold, ss.total, ss.holds,
                ss.false_holds);
        }
    }
    return 0;
}
