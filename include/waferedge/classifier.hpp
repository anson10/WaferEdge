#pragma once

#include "waferedge/clusters.hpp"
#include "waferedge/geometry.hpp"
#include "waferedge/hough.hpp"
#include "waferedge/wafer_map.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// The rule-based pattern classifier: per-map signals from the detectors, then an ordered list
// of rules, the first that holds names the pattern.
//
// The rules' structure (order, which signals, which direction) is designed by hand and stays
// fixed; only the thresholds are fitted, on a training split. Rules are data, so the same
// table drives classification, the explanation of a decision, fitting, and the thresholds
// file (config/rules.txt). ADR-0005 has the why.
namespace waferedge {

enum class Signal : std::uint8_t {
    density,       // fail density over the wafer
    center,        // zone 0 fail density / wafer's (equal-area zones)
    edge,          // last zone's fail density / wafer's
    sector,        // densest octant's fail density / wafer's
    inner,         // rings 0-1 (rho < 0.2) fail density / wafer's
    ring,          // densest of rings 2-6 (0.2 <= rho < 0.7) / wafer's
    cluster_share, // largest cluster's share of the fails
    cluster_size,  // largest cluster's dies / wafer dies
    cluster_rho,   // normalised radius of the largest cluster's centroid
    elongation,    // largest cluster's length / width
    line,          // Hough line's fail density / wafer's
    line_length,   // Hough line's fail votes / wafer diameter in dies
    z,             // join-count z
};
inline constexpr std::size_t kSignalCount = 13;
using Signals = std::array<double, kSignalCount>;

[[nodiscard]] std::string_view signal_name(Signal s) noexcept;
[[nodiscard]] constexpr double get(const Signals& s, Signal which) noexcept {
    return s[static_cast<std::size_t>(which)];
}

// Computes every signal for one map. Owns the detectors' buffers; one per thread.
class SignalExtractor {
public:
    [[nodiscard]] Signals extract(WaferMapView map);

private:
    GeometryCache geometry_;
    ClusterFinder clusters_;
    HoughTransform hough_;
};

struct Condition {
    Signal signal;
    bool at_least; // signal >= threshold; otherwise signal <= threshold
    double threshold;
};

struct Rule {
    std::string_view name;
    Pattern pattern;
    std::vector<Condition> conditions; // all must hold
};

struct Decision {
    Pattern pattern = Pattern::none;
    int rule = -1; // index of the rule that fired; -1: none fired, the map is "none"
};

class RuleClassifier {
public:
    // The rule structure with hand-set starting thresholds (read off docs/signatures.md).
    [[nodiscard]] static RuleClassifier defaults();
    // Thresholds from a rules file (as written by to_text) on top of the fixed structure.
    // Every condition must be given exactly once.
    [[nodiscard]] static std::expected<RuleClassifier, std::string> parse(std::string_view text);
    [[nodiscard]] static std::expected<RuleClassifier, std::string> load(const std::string& path);

    [[nodiscard]] Decision classify(const Signals& s) const noexcept;
    // One line: the rule that fired and each of its conditions with value and threshold.
    [[nodiscard]] std::string explain(const Signals& s) const;
    [[nodiscard]] std::string to_text() const;

    [[nodiscard]] std::span<const Rule> rules() const noexcept { return rules_; }
    [[nodiscard]] std::span<Rule> rules() noexcept { return rules_; }

private:
    std::vector<Rule> rules_;
};

// Per-class and overall scores of predictions against truth.
struct Metrics {
    std::array<std::array<std::uint32_t, kPatternCount>, kPatternCount>
        confusion{}; // [truth][pred]
    std::array<double, kPatternCount> precision{};
    std::array<double, kPatternCount> recall{};
    std::array<double, kPatternCount> f1{};
    std::array<std::uint32_t, kPatternCount> support{}; // maps per true class
    double accuracy = 0;
    double macro_f1 = 0; // mean F1 over the classes present in the truth
};

// Maps with truth Pattern::unknown are skipped.
[[nodiscard]] Metrics evaluate(std::span<const Pattern> truth, std::span<const Pattern> pred);

// Macro-F1 with every map weighted by its true class's weight.
[[nodiscard]] double weighted_macro_f1(std::span<const Pattern> truth,
                                       std::span<const Pattern> pred,
                                       const std::array<double, kPatternCount>& class_weight);

struct FitOptions {
    int sweeps = 8;      // passes over all thresholds (stops early when nothing improves)
    int candidates = 64; // quantiles of each signal tried per threshold
    // Each training map counts as this many maps of its true class when scoring a candidate:
    // undoes a cap on a class in the training data (WM-811K's train 'none', ADR-0005).
    std::array<double, kPatternCount> class_weight = {1, 1, 1, 1, 1, 1, 1, 1, 1};
};

// Coordinate ascent on (weighted) macro-F1: for each threshold in turn, try candidate values
// (quantiles of that signal over the given maps) and keep the best; repeat until a sweep changes
// nothing. `log` gets one line per sweep.
[[nodiscard]] RuleClassifier fit(RuleClassifier start, std::span<const Signals> signals,
                                 std::span<const Pattern> truth, const FitOptions& options,
                                 std::string* log = nullptr);

} // namespace waferedge
