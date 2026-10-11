#pragma once

#include "waferedge/map_file.hpp"
#include "waferedge/pipeline/decision.hpp"
#include "waferedge/pipeline/detector.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

// The closed-loop result (ADR-0017): what WaferEdge's holds would have done to WaferLens's
// spatial excursions, replayed in sort-time order with the live loop's decision rule.
//
// The demo set spans weeks of sort time, so it is replayed, not run live: each wafer's
// verdict goes through the same DecisionRule in `tested_at` order, and a hold takes effect
// `latency` after its triggering wafer was sorted (the live loop's measured sort -> HOLD).
// Wafers of a held lot sorted after that are withheld. This is the live loop as long as the
// latency is shorter than the gap between a lot's wafers; the replay reports that gap.
namespace waferedge::pipeline {

// One spatial-pattern excursion from WaferLens's ground truth.
struct Excursion {
    int id = 0;
    Pattern pattern = Pattern::unknown;
    std::int64_t start_us = 0; // when the excursion began (the fab's clock, us since the epoch)
    std::int64_t end_us = 0;
};

struct ExcursionTruth {
    std::vector<Excursion> excursions;
    std::unordered_map<std::int32_t, int> excursion_of; // wafer id -> excursion id
};

// The CSV tools/export_maps.py --excursions writes:
//   excursion,<id>,<pattern>,<start_us>,<end_us>  and  wafer,<wafer_id>,<excursion_id>
[[nodiscard]] std::expected<ExcursionTruth, std::string>
load_excursions(const std::filesystem::path& path);

// WaferLens's published batch alarm per excursion (config/waferlens_pattern_alarms.csv).
struct WaferLensAlarm {
    int id = 0;
    Pattern pattern = Pattern::unknown;
    std::size_t pattern_wafers = 0;
    double first_sorted_h = 0; // hours after the excursion's start
    double alarm_h = 0;
};
[[nodiscard]] std::expected<std::vector<WaferLensAlarm>, std::string>
load_waferlens_alarms(const std::filesystem::path& path);

struct ClosedLoopConfig {
    DecisionConfig decision{};
    std::int64_t latency_us = 5'000; // sort -> HOLD at the tool (bench-latency's p99.9 or so)
};

struct ExcursionOutcome {
    int id = 0;
    Pattern pattern = Pattern::unknown;
    std::size_t wafers = 0; // wafers showing its pattern, in the file
    std::optional<std::int64_t> first_sorted_us;
    std::optional<std::int64_t> hold_us; // its first hold: trigger wafer sorted + latency
    std::size_t lots_held = 0;           // holds triggered by its wafers
    std::size_t sorted_before_hold = 0;  // its wafers sorted before the hold (all if none)
    std::size_t sorted = 0;              // its wafers sorted at all (the rest were withheld)
};

struct ClosedLoopResult {
    std::vector<ExcursionOutcome> excursions; // in the order of ExcursionTruth::excursions
    std::size_t wafers_sorted = 0;
    std::size_t wafers_withheld = 0;
    std::size_t holds = 0;
    std::size_t false_holds = 0; // triggered by a wafer that shows no excursion
    std::size_t lots = 0;
    std::uint64_t evictions = 0;  // lots the decision table forgot
    std::uint64_t tool_holds = 0; // holds only the cross-lot rule made
    // The smallest gap between two consecutive sorted wafers of one lot: the replay equals
    // the live loop while the latency stays below it.
    std::optional<std::int64_t> min_lot_gap_us;
};

// Every record's verdict from `detector`, in record order (batches of kMaxBatch).
[[nodiscard]] std::expected<std::vector<Pattern>, std::string>
classify_all(std::span<const WaferRecord> records, Detector& detector);

// Replays the records in tested_at order (stable) with their verdicts (one per record).
[[nodiscard]] ClosedLoopResult replay_closed_loop(std::span<const WaferRecord> records,
                                                  std::span<const Pattern> verdicts,
                                                  const ExcursionTruth& truth,
                                                  const ClosedLoopConfig& config);

} // namespace waferedge::pipeline
