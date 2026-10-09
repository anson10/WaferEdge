#pragma once

#include "waferedge/wafer_map.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

// Straight-line (scratch) search with a Hough transform over fail dies.
//
// A line is (angle, rho): its normal points at `angle` degrees counter-clockwise from +column
// (0..179) and it passes `rho` die widths from the wafer centre along that normal. Every fail
// die votes once per angle, for the rho bin its centre falls in; a scratch is a peak.
//
// All integer: cos and sin are fixed-point (Q14) table entries, die centres are the doubled
// coordinates of geometry.hpp, so rho_raw = X cos + Y sin is exact and its bin is a floor
// shift. Every backend that uses the same tables finds the same votes.
namespace waferedge {

inline constexpr int kHoughAngles = 180; // 1-degree steps
inline constexpr int kHoughFracBits = 14;

// round(cos(k degrees) * 2^14), round(sin(...)), k = 0..179.
[[nodiscard]] const std::array<std::int32_t, kHoughAngles>& hough_cos() noexcept;
[[nodiscard]] const std::array<std::int32_t, kHoughAngles>& hough_sin() noexcept;

struct HoughLine {
    int angle = 0;               // degrees, 0..179
    int rho = 0;                 // signed, in die widths
    std::uint32_t votes = 0;     // fail dies on the line
    std::uint32_t line_dies = 0; // on-wafer dies on the line (the most votes it could get)

    bool operator==(const HoughLine&) const = default;
};

// rho bin of the die at doubled coordinates (x, y) for angle k.
[[nodiscard]] int hough_rho(int x, int y, int angle) noexcept;

class HoughTransform {
public:
    // The strongest line: most votes; ties go to the lowest angle, then the lowest rho. A
    // map without fails gives votes = 0. Buffers are reused across calls.
    HoughLine run(WaferMapView map);

    // The last run's votes, [angle][rho + rho_offset()], rho_bins() per angle.
    [[nodiscard]] std::span<const std::uint32_t> accumulator() const noexcept { return votes_; }
    [[nodiscard]] int rho_offset() const noexcept { return offset_; }
    [[nodiscard]] int rho_bins() const noexcept { return 2 * offset_ + 1; }

private:
    std::vector<std::uint32_t> votes_;
    int offset_ = 0;
};

} // namespace waferedge
