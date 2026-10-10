#pragma once

#include "waferedge/gem/host.hpp"
#include "waferedge/pipeline/analyzer.hpp"
#include "waferedge/pipeline/decision.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

// The edge host's network-thread logic, without I/O (ADR-0014): GEM host events in, wafers
// out to the analyzers, verdicts back, the decision rule, S2F41 HOLD out. EdgeHost drives it
// from an hsms::Session; the tests drive it from an hsms::Protocol on a fake clock.
//
//   on_hsms: S6F11 wafer report -> copy the map into the next analyzer's free slot
//            (round robin; every ring full: dropped, counted)
//   drain:   each verdict -> DecisionRule -> HOLD the lot (window full: retried on a reply)
//
// Nothing here allocates per wafer: slots, the decision table and the hold records are sized
// at construction.
namespace waferedge::pipeline {

using Logger = std::function<void(std::string_view)>;

struct EdgeConfig {
    gem::HostConfig gem{};
    DecisionConfig decision{};
    std::size_t max_holds = 4096; // hold records kept; later holds are sent but not recorded
};

// One lot hold, from the verdict that triggered it to the tool's answer.
struct HoldRecord {
    enum class State : std::uint8_t {
        waiting, // not sent yet: HSMS's window was full or GEM not communicating
        sent,    // S2F41 sent, awaiting S2F42
        acked,   // S2F42 received (hcack says what the tool did)
        failed,  // the link refused it, or the transaction failed (T3, abort)
    };
    LotId lot;
    Pattern pattern = Pattern::none;
    std::uint64_t wafer = 0; // the wafer whose verdict completed k of W
    TimePoint received;      // that wafer's arrival at the edge host
    TimePoint analysed;      // its verdict
    TimePoint decided;       // the network thread took the verdict and decided
    TimePoint sent;          // S2F41 handed to HSMS
    TimePoint acked;         // S2F42 received
    std::uint32_t system = 0;
    State state = State::waiting;
    std::uint8_t hcack = 0;
};

struct EdgeStats {
    std::uint64_t received = 0;  // wafer reports decoded
    std::uint64_t submitted = 0; // handed to an analyzer
    std::uint64_t dropped = 0;   // every analyzer's ring full
    std::uint64_t oversized = 0; // more than kMaxBins dies
    std::uint64_t bad_lot = 0;   // LOTID longer than LotId::kMax
    std::uint64_t verdicts = 0;  // taken back from the analyzers
    std::array<std::uint64_t, kPatternCount> by_pattern{};
    std::uint64_t holds = 0;            // decided (recorded or not)
    std::uint64_t holds_unrecorded = 0; // past max_holds
};

class EdgeCore {
public:
    // The analyzers must outlive the core; the core is their only producer / consumer.
    EdgeCore(std::span<Analyzer* const> analyzers, EdgeConfig config, Logger log = {});

    // `now` drives GEM's timers (a fake clock in tests). The stamps in slots and hold records
    // are read from the steady clock when each step happens, so they stay ordered even when
    // a worker pushes a verdict while drain() runs.

    // An HSMS event for the GEM host; wafer reports are submitted to the analyzers.
    void on_hsms(const hsms::Event& event, gem::Link& link, TimePoint now);
    // GEM's timers (WAIT DELAY).
    void tick(gem::Link& link, TimePoint now);
    [[nodiscard]] std::optional<TimePoint> next_deadline() const noexcept {
        return host_.next_deadline();
    }
    // Takes every verdict waiting in the analyzers, decides, sends holds. Returns how many.
    std::size_t drain(gem::Link& link);

    // Every submitted wafer's verdict has been taken.
    [[nodiscard]] bool idle() const noexcept { return stats_.verdicts == stats_.submitted; }
    // GEM communication was established and then lost (the tool separated or went away).
    [[nodiscard]] bool tool_left() const noexcept { return tool_left_; }

    [[nodiscard]] const gem::Host& host() const noexcept { return host_; }
    [[nodiscard]] const EdgeStats& stats() const noexcept { return stats_; }
    [[nodiscard]] std::span<const HoldRecord> holds() const noexcept { return holds_; }
    [[nodiscard]] const DecisionRule& decision() const noexcept { return decision_; }

private:
    void on_gem(const gem::Event& event, gem::Link& link);
    void submit(const gem::WaferReport& report);
    void decide(const Verdict& verdict, gem::Link& link);
    void send(HoldRecord& hold, gem::Link& link);
    void send_waiting(gem::Link& link);
    HoldRecord* find_sent(std::uint32_t system) noexcept;
    void log(std::string_view line) const;

    std::vector<Analyzer*> analyzers_;
    std::size_t next_ = 0; // round robin
    EdgeConfig config_;
    gem::Host host_;
    DecisionRule decision_;
    std::vector<HoldRecord> holds_; // capacity max_holds, never reallocated
    EdgeStats stats_;
    bool communicated_ = false;
    bool tool_left_ = false;
    Logger log_;
};

} // namespace waferedge::pipeline
