#include "waferedge/randomness.hpp"

#include <cmath>

namespace waferedge {

JoinCount join_count(WaferMapView map) noexcept {
    JoinCount j;
    const int rows = map.rows();
    const int cols = map.cols();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const std::uint8_t bin = map.at(r, c);
            if (!on_wafer(bin)) {
                continue;
            }
            ++j.dies;
            j.fails += is_fail(bin) ? 1U : 0U;
            // Degree for K; joins counted once, towards the east and the south.
            std::uint32_t degree = 0;
            const auto neighbour = [&](int rr, int cc) {
                return rr >= 0 && rr < rows && cc >= 0 && cc < cols && on_wafer(map.at(rr, cc));
            };
            degree += neighbour(r - 1, c) ? 1U : 0U;
            degree += neighbour(r, c - 1) ? 1U : 0U;
            const auto join = [&](int rr, int cc) {
                if (neighbour(rr, cc)) {
                    ++degree;
                    ++j.joins;
                    if (is_fail(bin) && is_fail(map.at(rr, cc))) {
                        ++j.fail_joins;
                    }
                }
            };
            join(r + 1, c);
            join(r, c + 1);
            j.join_pairs += std::uint64_t{degree} * (degree - 1) / 2;
        }
    }
    return j;
}

namespace {

// p_k = F(F-1)...(F-k+1) / N(N-1)...(N-k+1): the chance that k given dies all fail.
double draw_probability(double n, double f, int k) noexcept {
    double p = 1.0;
    for (int i = 0; i < k; ++i) {
        if (n - i <= 0) {
            return 0.0;
        }
        p *= (f - i) / (n - i);
    }
    return p < 0 ? 0.0 : p;
}

} // namespace

double expected_fail_joins(const JoinCount& j) noexcept {
    return j.joins * draw_probability(j.dies, j.fails, 2);
}

double fail_joins_variance(const JoinCount& j) noexcept {
    const double n = j.dies;
    const double f = j.fails;
    const double jn = j.joins;
    const auto k = static_cast<double>(j.join_pairs);
    const double p2 = draw_probability(n, f, 2);
    const double p3 = draw_probability(n, f, 3);
    const double p4 = draw_probability(n, f, 4);
    const double mean = jn * p2;
    return jn * p2 + 2.0 * k * p3 + (jn * (jn - 1.0) - 2.0 * k) * p4 - mean * mean;
}

double join_count_z(const JoinCount& j) noexcept {
    if (j.fails < 2) {
        return 0.0;
    }
    const double var = fail_joins_variance(j);
    // Rounding can leave a tiny positive variance where the true value is 0 (all dies fail).
    if (!(var > 1e-9)) {
        return 0.0;
    }
    return (j.fail_joins - expected_fail_joins(j)) / std::sqrt(var);
}

} // namespace waferedge
