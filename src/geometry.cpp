#include "waferedge/geometry.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <utility>

namespace waferedge {

namespace {

struct Doubled {
    std::int64_t x;
    std::int64_t y;
};

Doubled doubled_centre(int row, int col, int rows, int cols) noexcept {
    return {2 * std::int64_t{col} - (cols - 1), std::int64_t{rows - 1} - 2 * std::int64_t{row}};
}

// rho^2 = x^2/cols^2 + y^2/rows^2 = (x^2 rows^2 + y^2 cols^2) / (rows^2 cols^2) = num / den
struct Rho2 {
    std::int64_t num;
    std::int64_t den;
};

Rho2 rho2(int row, int col, int rows, int cols) noexcept {
    const auto [x, y] = doubled_centre(row, col, rows, cols);
    const std::int64_t r2 = std::int64_t{rows} * rows;
    const std::int64_t c2 = std::int64_t{cols} * cols;
    return {x * x * r2 + y * y * c2, r2 * c2};
}

std::uint64_t next_geometry_id() noexcept {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

int zone_of(int row, int col, int rows, int cols) noexcept {
    const auto [num, den] = rho2(row, col, rows, cols);
    return static_cast<int>(std::min<std::int64_t>(kZones - 1, kZones * num / den));
}

int ring_of(int row, int col, int rows, int cols) noexcept {
    const auto [num, den] = rho2(row, col, rows, cols);
    // The largest k with k / R <= rho, i.e. k^2 den <= R^2 num. num <= 2 den, so a short
    // count up is exact and cheap (it runs once per grid position per shape).
    int k = 0;
    while (k + 1 < kRings &&
           std::int64_t{k + 1} * (k + 1) * den <= std::int64_t{kRings} * kRings * num) {
        ++k;
    }
    return k;
}

int sector_of(int row, int col, int rows, int cols) noexcept {
    const auto [x, y] = doubled_centre(row, col, rows, cols);
    // Normalise both axes by cross-multiplying: compare x/cols with y/rows as x*rows, y*cols.
    const std::int64_t u = x * rows;
    const std::int64_t v = y * cols;
    if (u == 0 && v == 0) {
        return 0; // the centre die of an odd-sized grid
    }
    // Quadrant q covers [90q, 90q + 90) degrees. Rotate the point into quadrant 0, where
    // a > 0 and b >= 0, then the octant is whether the angle reached 45 degrees (b >= a).
    int quadrant = 0;
    std::int64_t a = u;
    std::int64_t b = v;
    if (u > 0 && v >= 0) {
        quadrant = 0;
    } else if (u <= 0 && v > 0) {
        quadrant = 1, a = v, b = -u;
    } else if (u < 0 && v <= 0) {
        quadrant = 2, a = -u, b = -v;
    } else {
        quadrant = 3, a = -v, b = u;
    }
    return 2 * quadrant + (b >= a ? 1 : 0);
}

Geometry::Geometry() : id_(next_geometry_id()) {}

Geometry::Geometry(int rows, int cols)
    : id_(next_geometry_id()), rows_(rows), cols_(cols),
      zones_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)),
      sectors_(zones_.size()), rings_(zones_.size()) {
    std::size_t i = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c, ++i) {
            zones_[i] = static_cast<std::uint8_t>(zone_of(r, c, rows, cols));
            sectors_[i] = static_cast<std::uint8_t>(sector_of(r, c, rows, cols));
            rings_[i] = static_cast<std::uint8_t>(ring_of(r, c, rows, cols));
        }
    }
}

Geometry Geometry::from_tables(int rows, int cols, std::vector<std::uint8_t> zones,
                               std::vector<std::uint8_t> sectors, std::vector<std::uint8_t> rings) {
    const auto n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    assert(zones.size() == n && sectors.size() == n && rings.size() == n);
    assert(std::ranges::all_of(zones, [](auto z) { return z < kZones; }));
    assert(std::ranges::all_of(sectors, [](auto s) { return s < kSectors; }));
    assert(std::ranges::all_of(rings, [](auto k) { return k < kRings; }));
    (void)n;
    Geometry g;
    g.rows_ = rows;
    g.cols_ = cols;
    g.zones_ = std::move(zones);
    g.sectors_ = std::move(sectors);
    g.rings_ = std::move(rings);
    return g;
}

const Geometry& GeometryCache::get(int rows, int cols) {
    const auto key = std::pair{rows, cols};
    if (auto it = shapes_.find(key); it != shapes_.end()) {
        return it->second;
    }
    return shapes_.try_emplace(key, rows, cols).first->second;
}

} // namespace waferedge
