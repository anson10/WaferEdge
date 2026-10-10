#pragma once

#include "waferedge/hsms/protocol.hpp"
#include "waferedge/wafer_map.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

// What the edge host's threads hand each other through SpscRings (ADR-0014): fixed-size
// values, so a ring is allocated once and a wafer costs one copy of its map, from the HSMS
// receive buffer into its slot.
namespace waferedge::pipeline {

using hsms::Clock;
using hsms::TimePoint;

// A lot name in place: GEM's LOTID is an <A> item of any length, but tools use short ids.
// Longer ones are refused (counted by the edge host), not truncated: two lots must never
// share a name.
class LotId {
public:
    static constexpr std::size_t kMax = 32;

    LotId() = default;
    [[nodiscard]] static std::optional<LotId> from(std::string_view name) noexcept {
        if (name.size() > kMax) {
            return std::nullopt;
        }
        LotId id;
        std::ranges::copy(name, id.chars_.begin());
        id.size_ = static_cast<std::uint8_t>(name.size());
        return id;
    }
    [[nodiscard]] std::string_view view() const noexcept { return {chars_.data(), size_}; }
    bool operator==(const LotId& other) const noexcept { return view() == other.view(); }

private:
    std::array<char, kMax> chars_{};
    std::uint8_t size_ = 0;
};

// The largest map a slot holds: 64 x 64 dies (WaferLens maps are at most 40 x 40). The edge
// host counts larger ones as oversized and doesn't analyse them.
inline constexpr std::size_t kMaxBins = std::size_t{64} * 64;

// Network thread -> analytics thread: one wafer report.
struct WaferSlot {
    std::uint64_t seq = 0; // arrival number at the edge host
    LotId lot;
    std::uint64_t wafer = 0;
    TimePoint received; // when the network thread decoded the S6F11
    int rows = 0;
    int cols = 0;
    std::array<std::uint8_t, kMaxBins> bins{};

    [[nodiscard]] WaferMapView map() const noexcept {
        return {
            std::span(bins).first(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)),
            rows, cols};
    }
};

// Analytics thread -> network thread: what the detector found.
struct Verdict {
    std::uint64_t seq = 0;
    LotId lot;
    std::uint64_t wafer = 0;
    Pattern pattern = Pattern::none;
    int rule = -1; // the classifier rule that fired
    TimePoint received;
    TimePoint analysed; // when the classifier finished
};

} // namespace waferedge::pipeline
