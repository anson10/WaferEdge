#pragma once

// Shared by the feature backends: how one die is counted, and the totals. The scalar backend
// runs count_dies over the whole map; the AVX2 backend over the tail its 32-die blocks leave.
#include "waferedge/features.hpp"

#include <cstddef>
#include <cstdint>

namespace waferedge::detail {

inline void count_dies(Features& f, const std::uint8_t* bins, const std::uint8_t* zones,
                       const std::uint8_t* sectors, const std::uint8_t* rings, std::size_t begin,
                       std::size_t end) noexcept {
    for (std::size_t i = begin; i < end; ++i) {
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
}

// dies and fails are the zone totals (every on-wafer die is in exactly one zone).
inline void finish_totals(Features& f) noexcept {
    f.dies = 0;
    f.fails = 0;
    for (std::size_t z = 0; z < kZones; ++z) {
        f.dies += f.zone_dies[z];
        f.fails += f.zone_fails[z];
    }
}

} // namespace waferedge::detail
