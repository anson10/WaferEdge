// The rule classifier from the command line.
//
//   waferedge-classify fit     <file.wmap> <rules-out> [--none-weight W]
//                                                          fit thresholds on the train split
//   waferedge-classify eval    <file.wmap> <rules> [train|val|test|all] [--csv <out.csv>]
//   waferedge-classify explain <file.wmap> <rules> <wafer_id>
//
// eval prints per-class precision / recall / F1, macro-F1 over the classes present in the
// truth, the confusion matrix and how often each rule fired; --csv writes one row per map
// (ids, truth, prediction, rule, every signal) for tools/evaluate.py.
#include "waferedge/classifier.hpp"
#include "waferedge/machine.hpp"
#include "waferedge/map_file.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace waferedge;

namespace {

void print(const std::string& s) {
    std::fputs(s.c_str(), stdout);
}

int fail(const std::string& message) {
    std::fputs(std::format("error: {}\n", message).c_str(), stderr);
    return 1;
}

constexpr std::string_view kUsage =
    "usage:\n"
    "  waferedge-classify fit     <file.wmap> <rules-out> [--none-weight W]\n"
    "  waferedge-classify eval    <file.wmap> <rules> [train|val|test|all] [--csv <out.csv>]\n"
    "  waferedge-classify explain <file.wmap> <rules> <wafer_id>\n";

std::optional<Split> parse_split(std::string_view s) {
    for (auto split : {Split::train, Split::val, Split::test, Split::unsplit}) {
        if (s == split_name(split)) {
            return split;
        }
    }
    return std::nullopt;
}

struct Selection {
    std::vector<const WaferRecord*> records;
    std::vector<Signals> signals;
    std::vector<Pattern> truth;
    double seconds = 0; // time to extract the signals
};

template <typename Keep>
Selection select(const MapSet& set, Keep keep) {
    Selection sel;
    SignalExtractor extractor;
    const auto start = std::chrono::steady_clock::now();
    for (const auto& r : set.records()) {
        if (keep(r)) {
            sel.records.push_back(&r);
            sel.signals.push_back(extractor.extract(r.map));
            sel.truth.push_back(r.truth);
        }
    }
    sel.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return sel;
}

std::string report(const Metrics& m) {
    std::string text = std::format("{:<10} {:>9} {:>7} {:>7} {:>7}\n", "class", "precision",
                                   "recall", "f1", "support");
    for (std::size_t c = 0; c < kPatternCount; ++c) {
        if (m.support[c] == 0 && m.precision[c] == 0) {
            continue;
        }
        text += std::format("{:<10} {:>9.3f} {:>7.3f} {:>7.3f} {:>7}\n",
                            pattern_name(static_cast<Pattern>(c)), m.precision[c], m.recall[c],
                            m.f1[c], m.support[c]);
    }
    text += std::format("macro-F1 {:.3f} (over classes in the truth), accuracy {:.3f}\n",
                        m.macro_f1, m.accuracy);
    text += "confusion (rows truth, columns predicted):\n           ";
    for (std::size_t c = 0; c < kPatternCount; ++c) {
        text += std::format("{:>7.6}", pattern_name(static_cast<Pattern>(c)));
    }
    text += "\n";
    for (std::size_t t = 0; t < kPatternCount; ++t) {
        if (m.support[t] == 0) {
            continue;
        }
        text += std::format("{:<10} ", pattern_name(static_cast<Pattern>(t)));
        for (std::size_t c = 0; c < kPatternCount; ++c) {
            text += std::format("{:>7}", m.confusion[t][c]);
        }
        text += "\n";
    }
    return text;
}

int cmd_fit(const MapSet& set, const std::string& out_path, double none_weight) {
    const auto train = select(set, [](const WaferRecord& r) { return r.split == Split::train; });
    if (train.records.empty()) {
        return fail("no maps in the train split");
    }
    print(std::format("fitting on {} train maps (signals in {:.1f} s)\n", train.records.size(),
                      train.seconds));
    FitOptions options;
    options.class_weight[static_cast<std::size_t>(Pattern::none)] = none_weight;
    print(std::format("'none' maps weighted x{}\n", none_weight));
    std::string log;
    const auto fitted = fit(RuleClassifier::defaults(), train.signals, train.truth, options, &log);
    print(log);
    std::ofstream out(out_path);
    out << std::format(
               "# Fitted by: waferedge-classify fit <wm811k_lot.wmap> {} --none-weight {}\n",
               out_path, none_weight)
        << "# on WM-811K's train split only (FabEye's lot-disjoint split; its train 'none' is\n"
        << "# capped at 10,000 maps, weighted back to the uncapped count, see ADR-0005).\n"
        << fitted.to_text();
    if (!out) {
        return fail(std::format("cannot write {}", out_path));
    }
    print(std::format("wrote {}\n", out_path));
    return 0;
}

int cmd_eval(const MapSet& set, const RuleClassifier& rules, std::optional<Split> split,
             const std::string& csv_path) {
    const auto sel = select(set, [&](const WaferRecord& r) { return !split || r.split == *split; });
    std::vector<Pattern> pred;
    std::vector<int> fired(rules.rules().size() + 1, 0);
    for (const auto& s : sel.signals) {
        const auto d = rules.classify(s);
        pred.push_back(d.pattern);
        ++fired[d.rule < 0 ? 0 : static_cast<std::size_t>(d.rule) + 1];
    }
    print(std::format("{} maps{} | signals + rules: {:.0f} maps/s\n", sel.records.size(),
                      split ? std::format(" ({})", split_name(*split)) : "",
                      static_cast<double>(sel.records.size()) / sel.seconds));
    print(report(evaluate(sel.truth, pred)));
    print("rules fired:");
    for (std::size_t i = 0; i < rules.rules().size(); ++i) {
        print(std::format(" {} {},", rules.rules()[i].name, fired[i + 1]));
    }
    print(std::format(" none of them {}\n", fired[0]));

    if (!csv_path.empty()) {
        std::ofstream csv(csv_path);
        csv << "wafer_id,lot_id,split,truth,pred,rule";
        for (std::size_t k = 0; k < kSignalCount; ++k) {
            csv << ',' << signal_name(static_cast<Signal>(k));
        }
        csv << '\n';
        for (std::size_t i = 0; i < sel.records.size(); ++i) {
            const auto& r = *sel.records[i];
            const auto d = rules.classify(sel.signals[i]);
            csv << std::format("{},{},{},{},{},{}", r.wafer_id, r.lot_id, split_name(r.split),
                               pattern_name(r.truth), pattern_name(d.pattern),
                               d.rule < 0 ? "-"
                                          : rules.rules()[static_cast<std::size_t>(d.rule)].name);
            for (const double v : sel.signals[i]) {
                csv << std::format(",{:.6g}", v);
            }
            csv << '\n';
        }
        if (!csv) {
            return fail(std::format("cannot write {}", csv_path));
        }
        print(std::format("wrote {}\n", csv_path));
    }
    print(describe_machine());
    return 0;
}

int cmd_explain(const MapSet& set, const RuleClassifier& rules, std::int32_t wafer_id) {
    SignalExtractor extractor;
    for (const auto& r : set.records()) {
        if (r.wafer_id == wafer_id) {
            const auto s = extractor.extract(r.map);
            print(std::format("wafer {} (lot {}, {}x{}, truth {})\n", r.wafer_id, r.lot_id,
                              r.map.rows(), r.map.cols(), pattern_name(r.truth)));
            for (std::size_t k = 0; k < kSignalCount; ++k) {
                print(std::format("  {:<14} {:.4g}\n", signal_name(static_cast<Signal>(k)), s[k]));
            }
            print(rules.explain(s) + "\n");
            return 0;
        }
    }
    return fail(std::format("no wafer {} in the file", wafer_id));
}

int run(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() < 3) {
        std::fputs(kUsage.data(), stderr);
        return 2;
    }
    auto set = MapSet::load(args[1]);
    if (!set) {
        return fail(set.error());
    }
    if (args[0] == "fit" &&
        (args.size() == 3 || (args.size() == 5 && args[3] == "--none-weight"))) {
        return cmd_fit(*set, args[2], args.size() == 5 ? std::stod(args[4]) : 1.0);
    }
    auto rules = RuleClassifier::load(args[2]);
    if (!rules) {
        return fail(rules.error());
    }
    if (args[0] == "eval") {
        std::optional<Split> split;
        std::string csv;
        for (std::size_t i = 3; i < args.size(); ++i) {
            if (args[i] == "--csv" && i + 1 < args.size()) {
                csv = args[++i];
            } else if (args[i] == "all") {
                split.reset();
            } else if (auto s = parse_split(args[i])) {
                split = s;
            } else {
                std::fputs(kUsage.data(), stderr);
                return 2;
            }
        }
        return cmd_eval(*set, *rules, split, csv);
    }
    if (args[0] == "explain" && args.size() == 4) {
        return cmd_explain(*set, *rules, static_cast<std::int32_t>(std::stol(args[3])));
    }
    std::fputs(kUsage.data(), stderr);
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        return fail(e.what());
    }
}
