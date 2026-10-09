#include "waferedge/features.hpp"

#include <algorithm>
#include <cassert>

namespace waferedge {

Features compute_features(WaferMapView map, const Geometry& geometry) noexcept {
    assert(map.rows() == geometry.rows() && map.cols() == geometry.cols());
    const auto bins = map.bins();
    const auto zones = geometry.zones();
    const auto sectors = geometry.sectors();
    const auto rings = geometry.rings();
    Features f;
    for (std::size_t i = 0; i < bins.size(); ++i) {
        const std::uint8_t bin = bins[i];
        if (!on_wafer(bin)) {
            continue;
        }
        const std::uint32_t fail = is_fail(bin) ? 1U : 0U;
        ++f.zone_dies[zones[i]];
        ++f.sector_dies[sectors[i]];
        ++f.ring_dies[rings[i]];
        f.zone_fails[zones[i]] += fail;
        f.sector_fails[sectors[i]] += fail;
        f.ring_fails[rings[i]] += fail;
    }
    for (int z = 0; z < kZones; ++z) {
        f.dies += f.zone_dies[static_cast<std::size_t>(z)];
        f.fails += f.zone_fails[static_cast<std::size_t>(z)];
    }
    return f;
}

namespace {

double ratio(std::uint32_t num, std::uint32_t den) noexcept {
    return den == 0 ? 0.0 : static_cast<double>(num) / static_cast<double>(den);
}

} // namespace

double yield(const Features& f) noexcept {
    return ratio(f.dies - f.fails, f.dies);
}

double fail_density(const Features& f) noexcept {
    return ratio(f.fails, f.dies);
}

double zone_density(const Features& f, int zone) noexcept {
    const auto z = static_cast<std::size_t>(zone);
    return ratio(f.zone_fails[z], f.zone_dies[z]);
}

double sector_density(const Features& f, int sector) noexcept {
    const auto s = static_cast<std::size_t>(sector);
    return ratio(f.sector_fails[s], f.sector_dies[s]);
}

double ring_density(const Features& f, int ring) noexcept {
    const auto k = static_cast<std::size_t>(ring);
    return ratio(f.ring_fails[k], f.ring_dies[k]);
}

double zone_ratio(const Features& f, int zone) noexcept {
    const double overall = fail_density(f);
    return overall == 0.0 ? 0.0 : zone_density(f, zone) / overall;
}

double center_ratio(const Features& f) noexcept {
    return zone_ratio(f, 0);
}

double edge_ratio(const Features& f) noexcept {
    return zone_ratio(f, kZones - 1);
}

double max_sector_ratio(const Features& f) noexcept {
    const double overall = fail_density(f);
    if (overall == 0.0) {
        return 0.0;
    }
    double densest = 0.0;
    for (int s = 0; s < kSectors; ++s) {
        densest = std::max(densest, sector_density(f, s));
    }
    return densest / overall;
}

} // namespace waferedge
