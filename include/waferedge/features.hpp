#pragma once

#include "waferedge/geometry.hpp"
#include "waferedge/wafer_map.hpp"

#include <array>
#include <cstdint>

// Per-map spatial features. The stored values are integer counts, so every backend (scalar,
// AVX2, CUDA) must produce exactly the same struct; the densities and ratios below are derived
// from the counts in one place.
namespace waferedge {

struct Features {
    std::uint32_t dies = 0;  // on-wafer dies
    std::uint32_t fails = 0; // dies with a fail bin
    std::array<std::uint32_t, kZones> zone_dies{};
    std::array<std::uint32_t, kZones> zone_fails{};
    std::array<std::uint32_t, kSectors> sector_dies{};
    std::array<std::uint32_t, kSectors> sector_fails{};

    bool operator==(const Features&) const = default;
};

// The scalar reference. Precondition: map and geometry have the same shape.
[[nodiscard]] Features compute_features(WaferMapView map, const Geometry& geometry) noexcept;

// Derived values. All are 0 for a map without dies (or a zone without dies); the ratios are 0
// for a map without fails, since there is no fail field to compare against.
[[nodiscard]] double yield(const Features& f) noexcept;
[[nodiscard]] double fail_density(const Features& f) noexcept;
[[nodiscard]] double zone_density(const Features& f, int zone) noexcept;
[[nodiscard]] double sector_density(const Features& f, int sector) noexcept;
// Fail density of a zone relative to the whole wafer: 1 for an even spread, > 1 where fails
// concentrate. center_ratio uses zone 0, edge_ratio the last zone.
[[nodiscard]] double zone_ratio(const Features& f, int zone) noexcept;
[[nodiscard]] double center_ratio(const Features& f) noexcept;
[[nodiscard]] double edge_ratio(const Features& f) noexcept;
// The densest sector relative to the whole wafer: high for one-sided (edge-loc, loc) fields.
[[nodiscard]] double max_sector_ratio(const Features& f) noexcept;

} // namespace waferedge
