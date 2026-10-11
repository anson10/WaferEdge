#include "waferedge/pipeline/decision.hpp"

#include <algorithm>

namespace waferedge::pipeline {

DecisionRule::DecisionRule(DecisionConfig config) : config_(config), lots_(config.lots) {}

DecisionRule::Lot& DecisionRule::find_or_evict(const LotId& lot) noexcept {
    Lot* oldest = &lots_.front();
    for (Lot& l : lots_) {
        if (l.last_seen != 0 && l.id == lot) {
            return l;
        }
        if (l.last_seen < oldest->last_seen) {
            oldest = &l; // an empty entry (0) wins over any used one
        }
    }
    if (oldest->last_seen != 0) {
        ++evictions_;
    }
    *oldest = Lot{.id = lot};
    return *oldest;
}

bool DecisionRule::tool_due(Pattern pattern, std::int64_t at_us) noexcept {
    const auto k = static_cast<std::uint8_t>(config_.tool_k);
    Arrivals& a = arrivals_[static_cast<std::size_t>(pattern) % kPatternCount];
    a.at[a.next] = at_us;
    a.next = static_cast<std::uint8_t>((a.next + 1) % k);
    a.filled = std::min<std::uint8_t>(static_cast<std::uint8_t>(a.filled + 1), k);
    // Full ring: `next` now points at the oldest of the last k arrivals.
    return a.filled == k && at_us - a.at[a.next] <= config_.tool_window_us;
}

std::optional<Pattern> DecisionRule::observe(const LotId& lot, Pattern pattern,
                                             std::int64_t at_us) noexcept {
    const bool patterned = pattern != Pattern::none && pattern != Pattern::unknown;
    // Counted for every lot, held or not: a held lot's wafers still say the excursion is on.
    const bool tool = patterned && config_.tool_k > 0 && tool_due(pattern, at_us);

    Lot& l = find_or_evict(lot);
    l.last_seen = ++clock_;
    if (l.held) {
        return std::nullopt;
    }
    const auto window = static_cast<std::uint8_t>(config_.window);
    l.recent[l.next] = pattern;
    l.next = static_cast<std::uint8_t>((l.next + 1) % window);
    l.filled = std::min<std::uint8_t>(static_cast<std::uint8_t>(l.filled + 1), window);
    if (!patterned) {
        return std::nullopt;
    }
    const bool in_lot =
        std::count(l.recent.begin(), l.recent.begin() + l.filled, pattern) >= config_.k;
    if (!in_lot && !tool) {
        return std::nullopt;
    }
    tool_holds_ += in_lot ? 0U : 1U;
    l.held = true;
    return pattern;
}

bool DecisionRule::held(const LotId& lot) const noexcept {
    return std::ranges::any_of(
        lots_, [&](const Lot& l) { return l.last_seen != 0 && l.id == lot && l.held; });
}

} // namespace waferedge::pipeline
