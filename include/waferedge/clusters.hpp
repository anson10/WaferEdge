#pragma once

#include "waferedge/wafer_map.hpp"

#include <cstdint>
#include <span>
#include <vector>

// Connected clusters of fail dies (8-connected: diagonal neighbours touch, so a diagonal
// scratch is one cluster).
//
// Union-find over the grid, two passes. Unions always hang the larger root under the smaller
// one, so every cluster's root is its first die in row-major order; clusters are numbered in
// that order. The numbering is therefore a property of the map, not of the merge order, and
// another labelling (the GPU's) can be compared with this one cluster by cluster.
namespace waferedge {

// Integer moments of one cluster: position statistics are exact, shape is derived from them.
struct Cluster {
    std::uint32_t size = 0;
    std::int32_t first = 0; // row-major index of its first die (the union-find root)
    std::int64_t sum_r = 0;
    std::int64_t sum_c = 0;
    std::int64_t sum_rr = 0;
    std::int64_t sum_cc = 0;
    std::int64_t sum_rc = 0;
    std::uint16_t min_r = 0;
    std::uint16_t max_r = 0;
    std::uint16_t min_c = 0;
    std::uint16_t max_c = 0;

    bool operator==(const Cluster&) const = default;
};

struct ClusterShape {
    double row = 0; // centroid
    double col = 0;
    double major_var = 0;  // variance along the main axis, dies treated as unit squares
    double minor_var = 0;  // ... and across it (>= 1/12)
    double elongation = 1; // sqrt(major / minor): length over width, 1 for a blob, L for a line
    double angle_deg = 0;  // main axis, degrees counter-clockwise from +column, in [0, 180)
};

[[nodiscard]] ClusterShape shape(const Cluster& c) noexcept;

class ClusterFinder {
public:
    // Labels the map's fail dies. Buffers are reused across calls: after the first few maps
    // of the largest shape, no allocation. Results stay valid until the next call.
    void run(WaferMapView map);

    // In order of each cluster's first die.
    [[nodiscard]] std::span<const Cluster> clusters() const noexcept { return clusters_; }
    // Per grid position: the index of the die's cluster, or -1 for dies that don't fail.
    [[nodiscard]] std::span<const std::int32_t> labels() const noexcept { return labels_; }
    // The largest cluster (the first one on ties), or nullptr if no die fails.
    [[nodiscard]] const Cluster* largest() const noexcept;

private:
    std::vector<std::int32_t> parent_;
    std::vector<std::int32_t> labels_;
    std::vector<Cluster> clusters_;
};

} // namespace waferedge
