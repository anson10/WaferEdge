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
namespace waferedge::pipeline {

struct DecisionConfig {
    // Hold after k wafers with the same pattern among the lot's last `window` verdicts;
    // 1 <= k <= window <= kMaxWindow.
    int k = 3;
    int window = 5;
    std::size_t lots = 64; // lots remembered
};

class DecisionRule {
public:
    static constexpr int kMaxWindow = 32;

    // Precondition: the config is valid (valid() is true).
    explicit DecisionRule(DecisionConfig config);
    [[nodiscard]] static bool valid(const DecisionConfig& c) noexcept {
        return c.k >= 1 && c.k <= c.window && c.window <= kMaxWindow && c.lots >= 1;
    }

    // Records one verdict for `lot`. Returns the pattern if this verdict makes the lot due for
    // a hold (once per lot: later verdicts of a held lot return nullopt).
    [[nodiscard]] std::optional<Pattern> observe(const LotId& lot, Pattern pattern) noexcept;

    [[nodiscard]] bool held(const LotId& lot) const noexcept;
    [[nodiscard]] const DecisionConfig& config() const noexcept { return config_; }
    // Lots forgotten to make room.
    [[nodiscard]] std::uint64_t evictions() const noexcept { return evictions_; }

private:
    struct Lot {
        LotId id;
        std::uint64_t last_seen = 0;              // 0: empty entry
        std::array<Pattern, kMaxWindow> recent{}; // a ring of the last `window` patterns
        std::uint8_t next = 0;
        std::uint8_t filled = 0;
        bool held = false;
    };
    Lot& find_or_evict(const LotId& lot) noexcept;

    DecisionConfig config_;
    std::vector<Lot> lots_; // sized once
    std::uint64_t clock_ = 0;
    std::uint64_t evictions_ = 0;
};

} // namespace waferedge::pipeline
