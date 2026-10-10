#pragma once

#include "waferedge/hsms/protocol.hpp"
#include "waferedge/map_file.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// The tool emulator's replay: which wafer is sorted when, and which lots are on hold. No I/O
// and no clock of its own, like hsms::Protocol; emulator::Tool drives it over TCP and tests
// drive it on a fake clock.
//
// Pacing is an open-loop, constant-rate schedule: the k-th wafer sent after start() is due at
// start + k / rate, whether or not the host kept up. A sender that falls behind catches up
// instead of shifting the schedule, so a slow host shows as late wafers rather than as fewer
// wafers (coordinated omission, docs/pipeline.md). A held lot's wafers are skipped and
// counted as withheld, the wafers a real tool would not have processed; they take no slot,
// as the tool would sort the next lot's wafer in that time.
namespace waferedge::emulator {

using hsms::Duration;
using hsms::TimePoint;

struct ReplayConfig {
    double rate = 10.0;    // wafers per second
    std::size_t limit = 0; // replay only the first `limit` wafers (0: all)
};

// The lot name the emulator reports for a .wmap lot id: "LOT-000281".
[[nodiscard]] std::string lot_name(std::int32_t lot_id);

enum class LotResult : std::uint8_t { done, already, unknown };

class Replay {
public:
    struct Due {
        std::size_t index = 0; // position in tested_at order, also the log's index
        const WaferRecord* record = nullptr;
        std::string_view lot;
        TimePoint scheduled; // the schedule's send time (may be in the past if behind)
    };
    struct Stats {
        std::size_t sent = 0;     // wafers handed out by take()
        std::size_t withheld = 0; // wafers skipped because their lot was held
        std::size_t holds = 0;    // HOLD commands that took effect
        std::size_t releases = 0;
    };

    // Orders the records by tested_at (stable); they must outlive the Replay.
    Replay(std::span<const WaferRecord> records, ReplayConfig config);

    // Starts (or restarts after pause) the schedule: the next wafer is due at `now`.
    void start(TimePoint now) noexcept;
    void pause() noexcept { running_ = false; }
    [[nodiscard]] bool running() const noexcept { return running_; }
    [[nodiscard]] bool finished() const noexcept { return position_ == order_.size(); }
    // When the next wafer is due, while running and not finished.
    [[nodiscard]] std::optional<TimePoint> next_due() const noexcept;

    // The next wafer whose slot has come, skipping held lots; nullopt if none is due yet.
    std::optional<Due> take(TimePoint now) noexcept;

    LotResult hold(std::string_view lot, TimePoint now);
    LotResult release(std::string_view lot);
    [[nodiscard]] bool held(std::string_view lot) const;

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    [[nodiscard]] std::size_t size() const noexcept { return order_.size(); }
    [[nodiscard]] const WaferRecord& record(std::size_t index) const noexcept {
        return records_[order_[index]];
    }
    [[nodiscard]] std::string_view lot_of(std::size_t index) const noexcept;
    // Per wafer (by index): its scheduled send time; nullopt if not (yet) sent or withheld.
    [[nodiscard]] const std::optional<TimePoint>& scheduled(std::size_t index) const noexcept {
        return scheduled_[index];
    }
    [[nodiscard]] bool withheld(std::size_t index) const noexcept { return withheld_[index]; }
    // When the wafer actually went out (the sender records it); its lateness is this minus
    // scheduled(): the load generator's own error, a floor under any latency it measures.
    void mark_sent(std::size_t index, TimePoint at) noexcept { sent_at_[index] = at; }
    [[nodiscard]] const std::optional<TimePoint>& sent_at(std::size_t index) const noexcept {
        return sent_at_[index];
    }
    // When the first HOLD of `lot` arrived, if it was ever held.
    [[nodiscard]] std::optional<TimePoint> hold_time(std::string_view lot) const;

private:
    struct Hash {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const noexcept {
            return std::hash<std::string_view>{}(s);
        }
    };
    [[nodiscard]] std::optional<std::size_t> lot_index(std::string_view lot) const;

    std::span<const WaferRecord> records_;
    Duration period_;
    std::vector<std::size_t> order_;     // record indices in tested_at order
    std::vector<std::size_t> lot_;       // per position: dense lot index
    std::vector<std::string> lot_names_; // per dense lot index
    std::unordered_map<std::string, std::size_t, Hash, std::equal_to<>> lots_by_name_;
    std::vector<bool> held_;                          // per lot
    std::vector<std::optional<TimePoint>> hold_at_;   // per lot
    std::vector<std::optional<TimePoint>> scheduled_; // per position
    std::vector<std::optional<TimePoint>> sent_at_;   // per position
    std::vector<bool> withheld_;                      // per position
    std::size_t position_ = 0;
    std::size_t slot_ = 0; // slots used since start()
    TimePoint base_;
    bool running_ = false;
    Stats stats_;
};

} // namespace waferedge::emulator
