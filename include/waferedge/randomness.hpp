#pragma once

#include "waferedge/wafer_map.hpp"

#include <cstdint>

// Is the fail field random or clustered? The join-count test.
//
// A join is a pair of on-wafer dies that are 4-neighbours. BB counts joins whose two dies
// both fail. Null hypothesis: the F fails are F dies drawn at random, without replacement,
// from the N on-wafer dies (the wafer's own yield, no spatial structure). Then, with
//   p2 = F(F-1) / N(N-1),  p3 = p2 (F-2)/(N-2),  p4 = p3 (F-3)/(N-3),
//   J = number of joins,   K = number of pairs of joins sharing a die = sum_i d_i(d_i-1)/2,
// (d_i the number of on-wafer neighbours of die i):
//   E[BB]   = J p2
//   Var[BB] = J p2 + 2K p3 + (J(J-1) - 2K) p4 - (J p2)^2
// The variance sums E[X_j X_k] over ordered pairs of joins: the pair itself (J terms, two
// dies must fail), pairs sharing a die (2K, three dies) and disjoint pairs (the rest, four).
// z = (BB - E) / sqrt(Var): about 0 for a random field, large and positive when fails touch
// each other more than chance allows (clusters, rings, scratches), negative when they avoid
// each other.
namespace waferedge {

struct JoinCount {
    std::uint32_t dies = 0;       // N
    std::uint32_t fails = 0;      // F
    std::uint32_t joins = 0;      // J
    std::uint64_t join_pairs = 0; // K
    std::uint32_t fail_joins = 0; // BB

    bool operator==(const JoinCount&) const = default;
};

[[nodiscard]] JoinCount join_count(WaferMapView map) noexcept;

[[nodiscard]] double expected_fail_joins(const JoinCount& j) noexcept;
[[nodiscard]] double fail_joins_variance(const JoinCount& j) noexcept;
// 0 when the test is undefined (fewer than 2 fails, or no variance: every die fails, ...).
[[nodiscard]] double join_count_z(const JoinCount& j) noexcept;

} // namespace waferedge
