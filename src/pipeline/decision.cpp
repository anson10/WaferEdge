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

std::optional<Pattern> DecisionRule::observe(const LotId& lot, Pattern pattern) noexcept {
    Lot& l = find_or_evict(lot);
    l.last_seen = ++clock_;
    if (l.held) {
        return std::nullopt;
    }
    const auto window = static_cast<std::uint8_t>(config_.window);
    l.recent[l.next] = pattern;
    l.next = static_cast<std::uint8_t>((l.next + 1) % window);
    l.filled = std::min<std::uint8_t>(static_cast<std::uint8_t>(l.filled + 1), window);
    if (pattern == Pattern::none || pattern == Pattern::unknown) {
        return std::nullopt;
    }
    const auto same = std::count(l.recent.begin(), l.recent.begin() + l.filled, pattern);
    if (same < config_.k) {
        return std::nullopt;
    }
    l.held = true;
    return pattern;
}

bool DecisionRule::held(const LotId& lot) const noexcept {
    return std::ranges::any_of(
        lots_, [&](const Lot& l) { return l.last_seen != 0 && l.id == lot && l.held; });
}

} // namespace waferedge::pipeline
