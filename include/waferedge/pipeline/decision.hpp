#pragma once

#include "waferedge/pipeline/slots.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

// When to hold a lot: once k of its last W analysed wafers show the same signature (anything
// but `none`). One wafer with a pattern is often a classifier false positive or a one-off
// defect; the same pattern again within a few wafers is an excursion (ADR-0014).
//
// The state is a fixed table of the most recently seen lots, scanned linearly: a tool works
// through one or a few lots at a time, so the lot looked up is almost always among the first
// entries, and a 64-entry scan of 33-byte names costs less than hashing would. When the table
// is full the least recently seen lot is forgotten; if it reappears it starts afresh (and a
// second HOLD of an already-held lot is answered HCACK 5, "already in condition").
//
// A second, optional rule looks across lots (ADR-0018): hold a lot whose wafer shows a
// pattern once `tool_k` wafers with that pattern, from any lots, arrived within
// `tool_window_us`. An excursion upstream spreads its wafers thinly over many lots, so a
// per-lot window fills slowly; the tool sees all of them. Either rule can hold a lot.
namespace waferedge::pipeline {

struct DecisionConfig {
    // Hold after k wafers with the same pattern among the lot's last `window` verdicts;
    // 1 <= k <= window <= kMaxWindow.
    int k = 3;
    int window = 5;
    std::size_t lots = 64; // lots remembered
    // Across lots: hold once tool_k wafers with the same pattern arrived within
    // tool_window_us (0: off); 1 <= tool_k <= kMaxWindow.
    int tool_k = 0;
    std::int64_t tool_window_us = 0;
};

class DecisionRule {
public:
    static constexpr int kMaxWindow = 32;

    // Precondition: the config is valid (valid() is true).
    explicit DecisionRule(DecisionConfig config);
    [[nodiscard]] static bool valid(const DecisionConfig& c) noexcept {
        return c.k >= 1 && c.k <= c.window && c.window <= kMaxWindow && c.lots >= 1 &&
               (c.tool_k == 0 || (c.tool_k >= 1 && c.tool_k <= kMaxWindow && c.tool_window_us > 0));
    }

    // Records one verdict for `lot`, for a wafer that arrived at `at_us` (any steady time
    // base; only differences matter, and only to the cross-lot rule). Returns the pattern if
    // this verdict makes the lot due for a hold (once per lot: later verdicts of a held lot
    // return nullopt). Times should not go backwards by more than the window's resolution
    // needs: the cross-lot rule keeps the last tool_k arrivals per pattern in arrival order.
    [[nodiscard]] std::optional<Pattern> observe(const LotId& lot, Pattern pattern,
                                                 std::int64_t at_us = 0) noexcept;

    [[nodiscard]] bool held(const LotId& lot) const noexcept;
    [[nodiscard]] const DecisionConfig& config() const noexcept { return config_; }
    // Lots forgotten to make room.
    [[nodiscard]] std::uint64_t evictions() const noexcept { return evictions_; }
    // Holds the cross-lot rule made that the per-lot rule wouldn't have (yet).
    [[nodiscard]] std::uint64_t tool_holds() const noexcept { return tool_holds_; }

private:
    struct Lot {
        LotId id;
        std::uint64_t last_seen = 0;              // 0: empty entry
        std::array<Pattern, kMaxWindow> recent{}; // a ring of the last `window` patterns
        std::uint8_t next = 0;
        std::uint8_t filled = 0;
        bool held = false;
    };
    // The last tool_k arrival times of one pattern, a ring.
    struct Arrivals {
        std::array<std::int64_t, kMaxWindow> at{};
        std::uint8_t next = 0;
        std::uint8_t filled = 0;
    };
    Lot& find_or_evict(const LotId& lot) noexcept;
    bool tool_due(Pattern pattern, std::int64_t at_us) noexcept;

    DecisionConfig config_;
    std::vector<Lot> lots_; // sized once
    std::uint64_t clock_ = 0;
    std::uint64_t evictions_ = 0;
    std::uint64_t tool_holds_ = 0;
    std::array<Arrivals, kPatternCount> arrivals_{};
};

} // namespace waferedge::pipeline
