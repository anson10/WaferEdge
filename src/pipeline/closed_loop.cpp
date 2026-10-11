#include "waferedge/pipeline/closed_loop.hpp"

#include "waferedge/emulator/replay.hpp"

#include <algorithm>
#include <charconv>
#include <format>
#include <fstream>
#include <numeric>
#include <string_view>

namespace waferedge::pipeline {

namespace {

// The comma-separated fields of one line.
std::vector<std::string_view> split(std::string_view line) {
    std::vector<std::string_view> out;
    while (true) {
        const auto comma = line.find(',');
        out.push_back(line.substr(0, comma));
        if (comma == std::string_view::npos) {
            return out;
        }
        line.remove_prefix(comma + 1);
    }
}

template <typename T>
bool number(std::string_view text, T& out) {
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc{} && end == text.data() + text.size();
}

// libc++ 18 has no floating-point from_chars.
bool number(std::string_view text, double& out) {
    const std::string copy(text);
    char* end = nullptr;
    out = std::strtod(copy.c_str(), &end);
    return !copy.empty() && end == copy.c_str() + copy.size();
}

} // namespace

std::expected<ExcursionTruth, std::string> load_excursions(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path.string()));
    }
    ExcursionTruth truth;
    std::size_t line_no = 0;
    for (std::string line; std::getline(in, line);) {
        ++line_no;
        if (line.empty() || line.starts_with('#')) {
            continue;
        }
        const auto f = split(line);
        const auto bad = [&] {
            return std::unexpected(
                std::format("{}:{}: bad line '{}'", path.string(), line_no, line));
        };
        if (f[0] == "excursion" && f.size() == 5) {
            Excursion e;
            const auto pattern = pattern_from_name(f[2]);
            if (!number(f[1], e.id) || !pattern || !number(f[3], e.start_us) ||
                !number(f[4], e.end_us)) {
                return bad();
            }
            e.pattern = *pattern;
            truth.excursions.push_back(e);
        } else if (f[0] == "wafer" && f.size() == 3) {
            std::int32_t wafer = 0;
            int excursion = 0;
            if (!number(f[1], wafer) || !number(f[2], excursion)) {
                return bad();
            }
            truth.excursion_of[wafer] = excursion;
        } else {
            return bad();
        }
    }
    if (truth.excursions.empty()) {
        return std::unexpected(std::format("{}: no excursions", path.string()));
    }
    return truth;
}

std::expected<std::vector<WaferLensAlarm>, std::string>
load_waferlens_alarms(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path.string()));
    }
    std::vector<WaferLensAlarm> out;
    bool header = true;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line.starts_with('#')) {
            continue;
        }
        if (header) { // excursion_id,pattern,pattern_wafers,first_sorted_h,alarm_h
            header = false;
            continue;
        }
        const auto f = split(line);
        WaferLensAlarm a;
        const auto pattern = f.size() == 5 ? pattern_from_name(f[1]) : std::nullopt;
        if (!pattern || !number(f[0], a.id) || !number(f[2], a.pattern_wafers) ||
            !number(f[3], a.first_sorted_h) || !number(f[4], a.alarm_h)) {
            return std::unexpected(std::format("{}: bad line '{}'", path.string(), line));
        }
        a.pattern = *pattern;
        out.push_back(a);
    }
    return out;
}

std::expected<std::vector<Pattern>, std::string> classify_all(std::span<const WaferRecord> records,
                                                              Detector& detector) {
    std::vector<Pattern> out(records.size(), Pattern::unknown);
    std::vector<WaferMapView> maps(kMaxBatch);
    std::vector<Detection> results(kMaxBatch);
    for (std::size_t at = 0; at < records.size(); at += kMaxBatch) {
        const std::size_t n = std::min(kMaxBatch, records.size() - at);
        for (std::size_t i = 0; i < n; ++i) {
            maps[i] = records[at + i].map;
        }
        if (!detector.detect(std::span(maps).first(n), std::span(results).first(n))) {
            return std::unexpected(
                std::format("{} failed on wafers {}..{}", detector.name(), at, at + n - 1));
        }
        for (std::size_t i = 0; i < n; ++i) {
            out[at + i] = results[i].pattern;
        }
    }
    return out;
}

ClosedLoopResult replay_closed_loop(std::span<const WaferRecord> records,
                                    std::span<const Pattern> verdicts, const ExcursionTruth& truth,
                                    const ClosedLoopConfig& config) {
    ClosedLoopResult result;
    std::unordered_map<int, std::size_t> slot; // excursion id -> index in result.excursions
    for (const Excursion& e : truth.excursions) {
        slot[e.id] = result.excursions.size();
        ExcursionOutcome o;
        o.id = e.id;
        o.pattern = e.pattern;
        result.excursions.push_back(o);
    }
    const auto outcome = [&](std::int32_t wafer) -> ExcursionOutcome* {
        const auto it = truth.excursion_of.find(wafer);
        if (it == truth.excursion_of.end()) {
            return nullptr;
        }
        const auto s = slot.find(it->second);
        return s == slot.end() ? nullptr : &result.excursions[s->second];
    };

    // Sort-time order, as the tool emulator replays (stable: equal times keep file order).
    std::vector<std::size_t> order(records.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::stable_sort(order, {}, [&](std::size_t i) { return records[i].tested_at_us; });

    DecisionRule rule(config.decision);
    std::unordered_map<std::int32_t, std::int64_t> held_at;   // lot -> when its hold landed
    std::unordered_map<std::int32_t, std::int64_t> last_sort; // lot -> its last sorted wafer
    std::vector<std::pair<std::size_t, std::int64_t>> sorted; // (record, tested_at)
    sorted.reserve(records.size());

    for (const std::size_t i : order) {
        const WaferRecord& r = records[i];
        ExcursionOutcome* ex = outcome(r.wafer_id);
        if (ex != nullptr) {
            ++ex->wafers;
            ex->first_sorted_us =
                std::min(ex->first_sorted_us.value_or(r.tested_at_us), r.tested_at_us);
        }
        if (const auto h = held_at.find(r.lot_id);
            h != held_at.end() && r.tested_at_us > h->second) {
            ++result.wafers_withheld; // the tool doesn't sort a held lot's later wafers
            continue;
        }
        ++result.wafers_sorted;
        sorted.emplace_back(i, r.tested_at_us);
        if (ex != nullptr) {
            ++ex->sorted;
        }
        if (const auto prev = last_sort.find(r.lot_id); prev != last_sort.end()) {
            const std::int64_t gap = r.tested_at_us - prev->second;
            result.min_lot_gap_us = std::min(result.min_lot_gap_us.value_or(gap), gap);
        }
        last_sort[r.lot_id] = r.tested_at_us;

        const auto lot = LotId::from(emulator::lot_name(r.lot_id));
        if (!lot || !rule.observe(*lot, verdicts[i], r.tested_at_us)) {
            continue;
        }
        const std::int64_t lands = r.tested_at_us + config.latency_us;
        held_at[r.lot_id] = lands;
        ++result.holds;
        if (ex == nullptr) {
            ++result.false_holds;
            continue;
        }
        ++ex->lots_held;
        if (!ex->hold_us || lands < *ex->hold_us) {
            ex->hold_us = lands;
        }
    }

    // Each excursion's wafers sorted before its first hold (all of them if never held).
    for (const auto& [i, at] : sorted) {
        ExcursionOutcome* ex = outcome(records[i].wafer_id);
        if (ex != nullptr && (!ex->hold_us || at < *ex->hold_us)) {
            ++ex->sorted_before_hold;
        }
    }
    result.lots = last_sort.size();
    result.evictions = rule.evictions();
    result.tool_holds = rule.tool_holds();
    return result;
}

} // namespace waferedge::pipeline
