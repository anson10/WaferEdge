#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <utility>
#include <vector>

// Where each die sits on the wafer: its radial zone and angular sector.
//
// Both depend only on the grid shape, so they are computed once per shape and every map of
// that shape reuses them; per map, features are then integer counts over these tables (the
// same counts on the CPU, with AVX2 and on the GPU).
//
// The arithmetic is exact. Die (r, c) has its centre at doubled, integer coordinates
//   X = 2c - (cols - 1),  Y = (rows - 1) - 2r        (Y points up)
// and the wafer is the ellipse inscribed in the grid (WM-811K dies are not square, so a
// 25 x 27 grid is a circle in millimetres). The normalised radius is
//   rho^2 = (X / cols)^2 + (Y / rows)^2,   0 at the centre, 1 at the grid's inscribed edge.
// Zones are equal-area rings: zone k holds k/K <= rho^2 < (k+1)/K, last zone open-ended.
// Rings are a finer, equal-width radial profile: ring k holds k/R <= rho < (k+1)/R, last ring
// open-ended (compared as k^2 / R^2 <= rho^2, still exact). Zones weigh area evenly (each
// holds a fifth of the dies); rings resolve small structures near the centre, where an
// equal-area zone 0 is already 45% of the radius.
// Sectors are 45-degree octants counter-clockwise from +X, in the same normalised frame,
// half-open so every die has exactly one sector; rotating a square map by 90 degrees moves
// every die exactly two sectors on.
namespace waferedge {

inline constexpr int kZones = 5;   // centre, three middle rings, edge (equal area)
inline constexpr int kSectors = 8; // 45-degree octants
inline constexpr int kRings = 10;  // equal-width radial rings, 0.1 of the radius each

// Exact zone and sector of a die, from the formulas above. Exposed for tests.
[[nodiscard]] int zone_of(int row, int col, int rows, int cols) noexcept;
[[nodiscard]] int sector_of(int row, int col, int rows, int cols) noexcept;
[[nodiscard]] int ring_of(int row, int col, int rows, int cols) noexcept;

class Geometry {
public:
    Geometry(int rows, int cols);
    // Move-only: every Geometry has a unique id (the GPU caches its tables by id), and a copy
    // would carry the same id with tables that may later differ.
    Geometry(const Geometry&) = delete;
    Geometry& operator=(const Geometry&) = delete;
    Geometry(Geometry&&) noexcept = default;
    Geometry& operator=(Geometry&&) noexcept = default;
    ~Geometry() = default;
    // Explicit tables instead of the formulas: for tests that need a layout no wafer shape
    // produces (every die in one bucket), and for custom zone layouts. Preconditions: each
    // table has rows * cols entries, zones < kZones, sectors < kSectors, rings < kRings.
    [[nodiscard]] static Geometry from_tables(int rows, int cols, std::vector<std::uint8_t> zones,
                                              std::vector<std::uint8_t> sectors,
                                              std::vector<std::uint8_t> rings);

    [[nodiscard]] int rows() const noexcept { return rows_; }
    [[nodiscard]] int cols() const noexcept { return cols_; }
    // Unique per constructed Geometry in this process (never reused), kept across moves.
    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
    // Row-major, one entry per grid position (off-wafer positions too).
    [[nodiscard]] std::span<const std::uint8_t> zones() const noexcept { return zones_; }
    [[nodiscard]] std::span<const std::uint8_t> sectors() const noexcept { return sectors_; }
    [[nodiscard]] std::span<const std::uint8_t> rings() const noexcept { return rings_; }

private:
    Geometry();
    std::uint64_t id_;
    int rows_ = 0;
    int cols_ = 0;
    std::vector<std::uint8_t> zones_;
    std::vector<std::uint8_t> sectors_;
    std::vector<std::uint8_t> rings_;
};

// Geometry per grid shape, built on first use. References stay valid for the cache's life.
// Not thread-safe: one cache per thread, or fill it before the threads start.
class GeometryCache {
public:
    [[nodiscard]] const Geometry& get(int rows, int cols);
    [[nodiscard]] std::size_t size() const noexcept { return shapes_.size(); }

private:
    std::map<std::pair<int, int>, Geometry> shapes_;
};

} // namespace waferedge
