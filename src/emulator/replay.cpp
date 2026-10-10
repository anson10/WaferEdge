#include "waferedge/emulator/replay.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <numeric>

namespace waferedge::emulator {

std::string lot_name(std::int32_t lot_id) {
    return std::format("LOT-{:06}", lot_id);
}

Replay::Replay(std::span<const WaferRecord> records, ReplayConfig config)
    : records_(records),
      period_(config.rate > 0 ? std::chrono::duration_cast<Duration>(
                                    std::chrono::duration<double>(1.0 / config.rate))
                              : Duration::zero()) {
    order_.resize(records.size());
    std::iota(order_.begin(), order_.end(), std::size_t{0});
    std::ranges::stable_sort(order_, {}, [&](std::size_t i) { return records[i].tested_at_us; });
    if (config.limit > 0 && config.limit < order_.size()) {
        order_.resize(config.limit);
    }
    // Dense lot indices and their names, built once: the hot path only compares strings.
    std::unordered_map<std::int32_t, std::size_t> by_id;
    lot_.reserve(order_.size());
    for (const std::size_t i : order_) {
        const auto id = records[i].lot_id;
        auto [it, inserted] = by_id.try_emplace(id, lot_names_.size());
        if (inserted) {
            lot_names_.push_back(lot_name(id));
            lots_by_name_.emplace(lot_names_.back(), it->second);
        }
        lot_.push_back(it->second);
    }
    held_.assign(lot_names_.size(), false);
    hold_at_.assign(lot_names_.size(), std::nullopt);
    scheduled_.assign(order_.size(), std::nullopt);
    sent_at_.assign(order_.size(), std::nullopt);
    withheld_.assign(order_.size(), false);
}

void Replay::start(TimePoint now) noexcept {
    base_ = now;
    slot_ = 0;
    running_ = true;
}

std::optional<TimePoint> Replay::next_due() const noexcept {
    if (!running_ || finished()) {
        return std::nullopt;
    }
    return base_ + period_ * static_cast<Duration::rep>(slot_);
}

std::optional<Replay::Due> Replay::take(TimePoint now) noexcept {
    const auto due = next_due();
    if (!due || now < *due) {
        return std::nullopt;
    }
    while (!finished() && held_[lot_[position_]]) {
        withheld_[position_] = true; // the tool skips the held lot's wafer; no slot used
        ++stats_.withheld;
        ++position_;
    }
    if (finished()) {
        return std::nullopt;
    }
    const std::size_t index = position_++;
    ++slot_;
    ++stats_.sent;
    scheduled_[index] = *due;
    return Due{index, &record(index), lot_of(index), *due};
}

std::optional<std::size_t> Replay::lot_index(std::string_view lot) const {
    const auto it = lots_by_name_.find(lot); // heterogeneous: no std::string built
    if (it == lots_by_name_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::string_view Replay::lot_of(std::size_t index) const noexcept {
    return lot_names_[lot_[index]];
}

LotResult Replay::hold(std::string_view lot, TimePoint now) {
    const auto i = lot_index(lot);
    if (!i) {
        return LotResult::unknown;
    }
    if (held_[*i]) {
        return LotResult::already;
    }
    held_[*i] = true;
    if (!hold_at_[*i]) {
        hold_at_[*i] = now;
    }
    ++stats_.holds;
    return LotResult::done;
}

LotResult Replay::release(std::string_view lot) {
    const auto i = lot_index(lot);
    if (!i) {
        return LotResult::unknown;
    }
    if (!held_[*i]) {
        return LotResult::already;
    }
    held_[*i] = false;
    ++stats_.releases;
    return LotResult::done;
}

bool Replay::held(std::string_view lot) const {
    const auto i = lot_index(lot);
    return i && held_[*i];
}

std::optional<TimePoint> Replay::hold_time(std::string_view lot) const {
    const auto i = lot_index(lot);
    if (!i) {
        return std::nullopt;
    }
    return hold_at_[*i];
}

} // namespace waferedge::emulator
