#include "waferedge/classifier.hpp"

#include "waferedge/features.hpp"
#include "waferedge/randomness.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <fstream>
#include <numbers>
#include <sstream>

namespace waferedge {

std::string_view signal_name(Signal s) noexcept {
    switch (s) {
    case Signal::density:
        return "density";
    case Signal::center:
        return "center";
    case Signal::edge:
        return "edge";
    case Signal::sector:
        return "sector";
    case Signal::inner:
        return "inner";
    case Signal::ring:
        return "ring";
    case Signal::cluster_share:
        return "cluster_share";
    case Signal::cluster_size:
        return "cluster_size";
    case Signal::cluster_rho:
        return "cluster_rho";
    case Signal::elongation:
        return "elongation";
    case Signal::line:
        return "line";
    case Signal::line_length:
        return "line_length";
    case Signal::z:
        return "z";
    }
    return "?";
}

namespace {

double safe_div(double num, double den) noexcept {
    return den == 0.0 ? 0.0 : num / den;
}

void set(Signals& s, Signal which, double value) noexcept {
    s[static_cast<std::size_t>(which)] = value;
}

} // namespace

Signals SignalExtractor::extract(WaferMapView map) {
    Signals s{};
    const auto f = compute_features(map, geometry_.get(map.rows(), map.cols()));
    const double density = fail_density(f);
    set(s, Signal::density, density);
    set(s, Signal::z, join_count_z(join_count(map)));
    if (f.fails == 0) {
        return s; // every other signal compares fails against something: all 0
    }
    set(s, Signal::center, center_ratio(f));
    set(s, Signal::edge, edge_ratio(f));
    set(s, Signal::sector, max_sector_ratio(f));

    const double inner =
        safe_div(f.ring_fails[0] + f.ring_fails[1], f.ring_dies[0] + f.ring_dies[1]);
    set(s, Signal::inner, inner / density);
    double ring = 0;
    for (int k = 2; k <= 6; ++k) {
        ring = std::max(ring, ring_density(f, k));
    }
    set(s, Signal::ring, ring / density);

    clusters_.run(map);
    if (const Cluster* big = clusters_.largest()) {
        const auto sh = shape(*big);
        set(s, Signal::cluster_share, static_cast<double>(big->size) / f.fails);
        set(s, Signal::cluster_size, static_cast<double>(big->size) / f.dies);
        const double x = (2 * sh.col - (map.cols() - 1)) / map.cols();
        const double y = ((map.rows() - 1) - 2 * sh.row) / map.rows();
        set(s, Signal::cluster_rho, std::sqrt(x * x + y * y));
        set(s, Signal::elongation, sh.elongation);
    }

    const auto line = hough_.run(map);
    set(s, Signal::line, safe_div(safe_div(line.votes, line.line_dies), density));
    // Diameter of a disc with the wafer's die count: line_length 1 is a scratch across.
    const double diameter = 2.0 * std::sqrt(f.dies / std::numbers::pi);
    set(s, Signal::line_length, safe_div(line.votes, diameter));
    return s;
}

RuleClassifier RuleClassifier::defaults() {
    using enum Signal;
    constexpr bool ge = true;
    constexpr bool le = false;
    RuleClassifier c;
    // Order matters: the first rule that holds wins. Dense fields first (every other signal
    // is meaningless when almost every die fails), then the edge patterns (an edge ring would
    // otherwise pass as a line or a blob), then lines, then the radial and local blobs.
    c.rules_ = {
        {"near_full", Pattern::near_full, {{density, ge, 0.7}}},
        {"random", Pattern::random, {{density, ge, 0.35}, {z, le, 5.0}}},
        {"edge_ring", Pattern::edge_ring, {{edge, ge, 2.5}, {sector, le, 1.6}, {z, ge, 4.0}}},
        {"edge_loc", Pattern::edge_loc, {{edge, ge, 1.8}, {sector, ge, 1.7}, {z, ge, 3.0}}},
        {"scratch_cluster",
         Pattern::scratch,
         {{elongation, ge, 4.0}, {line_length, ge, 0.4}, {cluster_share, ge, 0.15}}},
        {"scratch_line",
         Pattern::scratch,
         {{line, ge, 6.0}, {line_length, ge, 0.5}, {edge, le, 2.0}}},
        {"donut", Pattern::donut, {{ring, ge, 1.8}, {inner, le, 1.2}, {z, ge, 6.0}}},
        {"center", Pattern::center, {{inner, ge, 2.0}, {cluster_rho, le, 0.3}, {z, ge, 3.0}}},
        {"loc", Pattern::loc, {{cluster_size, ge, 0.02}, {cluster_rho, le, 0.85}, {z, ge, 4.0}}},
    };
    return c;
}

Decision RuleClassifier::classify(const Signals& s) const noexcept {
    for (std::size_t i = 0; i < rules_.size(); ++i) {
        const auto& rule = rules_[i];
        const bool holds = std::ranges::all_of(rule.conditions, [&](const Condition& c) {
            const double v = get(s, c.signal);
            return c.at_least ? v >= c.threshold : v <= c.threshold;
        });
        if (holds) {
            return {rule.pattern, static_cast<int>(i)};
        }
    }
    return {};
}

std::string RuleClassifier::explain(const Signals& s) const {
    const auto d = classify(s);
    if (d.rule < 0) {
        return "no rule held -> none";
    }
    const auto& rule = rules_[static_cast<std::size_t>(d.rule)];
    std::string text = std::format("{} -> {}:", rule.name, pattern_name(rule.pattern));
    for (const auto& c : rule.conditions) {
        text += std::format(" {} {:.3g} {} {:.3g},", signal_name(c.signal), get(s, c.signal),
                            c.at_least ? ">=" : "<=", c.threshold);
    }
    text.pop_back();
    return text;
}

std::string RuleClassifier::to_text() const {
    std::string text = "# rule signal op threshold (first rule that holds wins; order is fixed)\n";
    for (const auto& rule : rules_) {
        for (const auto& c : rule.conditions) {
            // {} on a double is the shortest text that reads back as the same double.
            text += std::format("{} {} {} {}\n", rule.name, signal_name(c.signal),
                                c.at_least ? ">=" : "<=", c.threshold);
        }
    }
    return text;
}

std::expected<RuleClassifier, std::string> RuleClassifier::parse(std::string_view text) {
    auto c = defaults();
    std::vector<std::vector<bool>> seen;
    seen.reserve(c.rules_.size());
    for (const auto& rule : c.rules_) {
        seen.emplace_back(rule.conditions.size(), false);
    }
    std::istringstream in{std::string(text)};
    int line_no = 0;
    for (std::string line; std::getline(in, line);) {
        ++line_no;
        if (line.empty() || line.starts_with('#')) {
            continue;
        }
        std::istringstream fields(line);
        std::string rule_name;
        std::string signal;
        std::string op;
        std::string value;
        if (!(fields >> rule_name >> signal >> op >> value) || (op != ">=" && op != "<=")) {
            return std::unexpected(
                std::format("line {}: expected 'rule signal >=|<= value'", line_no));
        }
        char* end = nullptr;
        const double threshold = std::strtod(value.c_str(), &end);
        if (end != value.c_str() + value.size() || !std::isfinite(threshold)) {
            return std::unexpected(std::format("line {}: bad number '{}'", line_no, value));
        }
        bool matched = false;
        for (std::size_t r = 0; r < c.rules_.size() && !matched; ++r) {
            auto& rule = c.rules_[r];
            for (std::size_t k = 0; k < rule.conditions.size(); ++k) {
                auto& cond = rule.conditions[k];
                if (rule.name == rule_name && signal_name(cond.signal) == signal &&
                    cond.at_least == (op == ">=")) {
                    if (seen[r][k]) {
                        return std::unexpected(std::format("line {}: given twice", line_no));
                    }
                    cond.threshold = threshold;
                    seen[r][k] = true;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            return std::unexpected(std::format("line {}: no such rule condition '{} {} {}'",
                                               line_no, rule_name, signal, op));
        }
    }
    for (std::size_t r = 0; r < c.rules_.size(); ++r) {
        for (std::size_t k = 0; k < seen[r].size(); ++k) {
            if (!seen[r][k]) {
                const auto& cond = c.rules_[r].conditions[k];
                return std::unexpected(std::format("missing threshold for {} {}", c.rules_[r].name,
                                                   signal_name(cond.signal)));
            }
        }
    }
    return c;
}

std::expected<RuleClassifier, std::string> RuleClassifier::load(const std::string& path) {
    const std::ifstream in(path);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path));
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    auto c = parse(buffer.str());
    if (!c) {
        return std::unexpected(std::format("{}: {}", path, c.error()));
    }
    return c;
}

Metrics evaluate(std::span<const Pattern> truth, std::span<const Pattern> pred) {
    Metrics m;
    std::uint32_t total = 0;
    std::uint32_t correct = 0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (truth[i] == Pattern::unknown) {
            continue;
        }
        const auto t = static_cast<std::size_t>(truth[i]);
        const auto p = static_cast<std::size_t>(pred[i]);
        ++m.confusion[t][p];
        ++m.support[t];
        ++total;
        correct += t == p ? 1U : 0U;
    }
    double f1_sum = 0;
    int present = 0;
    for (std::size_t c = 0; c < kPatternCount; ++c) {
        std::uint32_t predicted = 0;
        for (std::size_t t = 0; t < kPatternCount; ++t) {
            predicted += m.confusion[t][c];
        }
        const double hit = m.confusion[c][c];
        m.precision[c] = safe_div(hit, predicted);
        m.recall[c] = safe_div(hit, m.support[c]);
        m.f1[c] = safe_div(2 * m.precision[c] * m.recall[c], m.precision[c] + m.recall[c]);
        if (m.support[c] > 0) {
            f1_sum += m.f1[c];
            ++present;
        }
    }
    m.accuracy = safe_div(correct, total);
    m.macro_f1 = safe_div(f1_sum, present);
    return m;
}

double weighted_macro_f1(std::span<const Pattern> truth, std::span<const Pattern> pred,
                         const std::array<double, kPatternCount>& class_weight) {
    std::array<std::array<double, kPatternCount>, kPatternCount> confusion{};
    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (truth[i] != Pattern::unknown) {
            const auto t = static_cast<std::size_t>(truth[i]);
            confusion[t][static_cast<std::size_t>(pred[i])] += class_weight[t];
        }
    }
    double f1_sum = 0;
    int present = 0;
    for (std::size_t c = 0; c < kPatternCount; ++c) {
        double predicted = 0;
        double support = 0;
        for (std::size_t k = 0; k < kPatternCount; ++k) {
            predicted += confusion[k][c];
            support += confusion[c][k];
        }
        if (support > 0) {
            const double p = safe_div(confusion[c][c], predicted);
            const double r = safe_div(confusion[c][c], support);
            f1_sum += safe_div(2 * p * r, p + r);
            ++present;
        }
    }
    return safe_div(f1_sum, present);
}

namespace {

double score(const RuleClassifier& c, std::span<const Signals> signals,
             std::span<const Pattern> truth, const FitOptions& options,
             std::vector<Pattern>& pred) {
    pred.resize(signals.size());
    for (std::size_t i = 0; i < signals.size(); ++i) {
        pred[i] = c.classify(signals[i]).pattern;
    }
    return weighted_macro_f1(truth, pred, options.class_weight);
}

} // namespace

RuleClassifier fit(RuleClassifier start, std::span<const Signals> signals,
                   std::span<const Pattern> truth, const FitOptions& options, std::string* log) {
    // Candidate thresholds per signal: quantiles of its values over the training maps.
    std::array<std::vector<double>, kSignalCount> candidates;
    for (std::size_t k = 0; k < kSignalCount; ++k) {
        std::vector<double> values;
        values.reserve(signals.size());
        for (const auto& s : signals) {
            values.push_back(s[k]);
        }
        std::ranges::sort(values);
        for (int q = 0; q < options.candidates && !values.empty(); ++q) {
            // Integer arithmetic: a floating-point q / (C-1) * (n-1) can land just below an
            // integer and truncate to the value before it.
            const auto at = static_cast<std::size_t>(q) * (values.size() - 1) /
                            static_cast<std::size_t>(options.candidates - 1);
            candidates[k].push_back(values[at]);
        }
        const auto [first, last] = std::ranges::unique(candidates[k]);
        candidates[k].erase(first, last);
    }

    std::vector<Pattern> pred;
    double best = score(start, signals, truth, options, pred);
    if (log != nullptr) {
        *log += std::format("start: weighted macro-F1 {:.4f}\n", best);
    }
    for (int sweep = 1; sweep <= options.sweeps; ++sweep) {
        bool improved = false;
        for (auto& rule : start.rules()) {
            for (auto& cond : rule.conditions) {
                const double keep = cond.threshold;
                double best_value = keep;
                for (const double v : candidates[static_cast<std::size_t>(cond.signal)]) {
                    cond.threshold = v;
                    const double s = score(start, signals, truth, options, pred);
                    if (s > best + 1e-12) { // strictly better, so ties keep the current value
                        best = s;
                        best_value = v;
                        improved = true;
                    }
                }
                cond.threshold = best_value;
            }
        }
        if (log != nullptr) {
            *log += std::format("sweep {}: weighted macro-F1 {:.4f}\n", sweep, best);
        }
        if (!improved) {
            break;
        }
    }
    return start;
}

} // namespace waferedge
