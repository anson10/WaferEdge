#include "waferedge/clusters.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace waferedge {

namespace {

// Path halving: every visited node skips to its grandparent, which flattens the tree as a
// side effect of the search.
std::int32_t find(std::int32_t* parent, std::int32_t i) noexcept {
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

void unite(std::int32_t* parent, std::int32_t a, std::int32_t b) noexcept {
    a = find(parent, a);
    b = find(parent, b);
    if (a != b) {
        parent[std::max(a, b)] = std::min(a, b); // the root is always the earliest die
    }
}

} // namespace

void ClusterFinder::run(WaferMapView map) {
    const int rows = map.rows();
    const int cols = map.cols();
    const auto n = map.bins().size();
    parent_.resize(n);
    labels_.assign(n, -1);
    clusters_.clear();
    std::int32_t* parent = parent_.data();

    // Pass 1: link each fail die to the fail neighbours already visited (W, NW, N, NE).
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!is_fail(map.at(r, c))) {
                continue;
            }
            const std::int32_t i = r * cols + c;
            parent[i] = i;
            if (c > 0 && is_fail(map.at(r, c - 1))) {
                unite(parent, i, i - 1);
            }
            if (r > 0) {
                for (int dc = -1; dc <= 1; ++dc) {
                    const int cc = c + dc;
                    if (cc >= 0 && cc < cols && is_fail(map.at(r - 1, cc))) {
                        unite(parent, i, i - cols + dc);
                    }
                }
            }
        }
    }

    // Pass 2: number the roots in scan order (a root precedes its other dies) and add up
    // the moments.
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!is_fail(map.at(r, c))) {
                continue;
            }
            const std::int32_t i = r * cols + c;
            const std::int32_t root = find(parent, i);
            std::int32_t label = 0;
            if (root == i) {
                label = static_cast<std::int32_t>(clusters_.size());
                clusters_.push_back(Cluster{.first = i,
                                            .min_r = static_cast<std::uint16_t>(r),
                                            .max_r = static_cast<std::uint16_t>(r),
                                            .min_c = static_cast<std::uint16_t>(c),
                                            .max_c = static_cast<std::uint16_t>(c)});
            } else {
                label = labels_[static_cast<std::size_t>(root)];
            }
            labels_[static_cast<std::size_t>(i)] = label;
            Cluster& k = clusters_[static_cast<std::size_t>(label)];
            ++k.size;
            k.sum_r += r;
            k.sum_c += c;
            k.sum_rr += std::int64_t{r} * r;
            k.sum_cc += std::int64_t{c} * c;
            k.sum_rc += std::int64_t{r} * c;
            k.min_r = std::min(k.min_r, static_cast<std::uint16_t>(r));
            k.max_r = std::max(k.max_r, static_cast<std::uint16_t>(r));
            k.min_c = std::min(k.min_c, static_cast<std::uint16_t>(c));
            k.max_c = std::max(k.max_c, static_cast<std::uint16_t>(c));
        }
    }
}

const Cluster* ClusterFinder::largest() const noexcept {
    const Cluster* best = nullptr;
    for (const auto& c : clusters_) {
        if (best == nullptr || c.size > best->size) {
            best = &c;
        }
    }
    return best;
}

ClusterShape shape(const Cluster& c) noexcept {
    if (c.size == 0) {
        return {};
    }
    const auto n = static_cast<double>(c.size);
    ClusterShape s;
    s.row = static_cast<double>(c.sum_r) / n;
    s.col = static_cast<double>(c.sum_c) / n;
    // Covariance of die centres, plus 1/12 on each axis: a die is a unit square, not a
    // point, so a single die has variance 1/12 and a line of L dies L^2/12 along its length.
    constexpr double die = 1.0 / 12.0;
    const double vrr = static_cast<double>(c.sum_rr) / n - s.row * s.row + die;
    const double vcc = static_cast<double>(c.sum_cc) / n - s.col * s.col + die;
    const double vrc = static_cast<double>(c.sum_rc) / n - s.row * s.col;
    // Eigenvalues of [[vcc, -vrc], [-vrc, vrr]] (x = column, y = row flipped to point up).
    const double mean = 0.5 * (vrr + vcc);
    const double half_gap = std::sqrt(0.25 * (vcc - vrr) * (vcc - vrr) + vrc * vrc);
    s.major_var = mean + half_gap;
    s.minor_var = std::max(mean - half_gap, die);
    s.elongation = std::sqrt(s.major_var / s.minor_var);
    double angle = 0.5 * std::atan2(-2.0 * vrc, vcc - vrr) * 180.0 / std::numbers::pi;
    if (angle < 0) {
        angle += 180.0;
    }
    s.angle_deg = angle >= 180.0 ? 0.0 : angle;
    return s;
}

} // namespace waferedge
