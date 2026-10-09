#include "waferedge/features.hpp"

#include "features_detail.hpp"

#include <algorithm>
#include <cassert>

namespace waferedge {

Features compute_features(WaferMapView map, const Geometry& geometry) noexcept {
    assert(map.rows() == geometry.rows() && map.cols() == geometry.cols());
    Features f;
    detail::count_dies(f, map.bins().data(), geometry.zones().data(), geometry.sectors().data(),
                       geometry.rings().data(), 0, map.bins().size());
    detail::finish_totals(f);
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
