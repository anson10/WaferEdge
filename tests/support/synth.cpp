#include "synth.hpp"

#include <cassert>

namespace waferedge::synth {

double rho2(int row, int col, int rows, int cols) noexcept {
    const double x = (2.0 * col - (cols - 1)) / cols;
    const double y = ((rows - 1) - 2.0 * row) / rows;
    return x * x + y * y;
}

WaferMap disc(int rows, int cols) {
    WaferMap map(rows, cols, 0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (rho2(r, c, rows, cols) <= 1.0) {
                map.set(r, c, 1);
            }
        }
    }
    return map;
}

void sprinkle(WaferMap& map, unsigned per_mille, std::mt19937& rng) {
    for (auto& bin : map.bins()) {
        if (on_wafer(bin) && rng() % 1000 < per_mille) {
            bin = static_cast<std::uint8_t>(2 + rng() % 4);
        }
    }
}

WaferMap random_map(int rows, int cols, unsigned per_mille, std::uint32_t seed) {
    std::mt19937 rng(seed);
    WaferMap map = disc(rows, cols);
    sprinkle(map, per_mille, rng);
    return map;
}

WaferMap rotate90(const WaferMap& map) {
    assert(map.rows() == map.cols());
    const int n = map.rows();
    WaferMap out(n, n, 0);
    // Die (r, c) at doubled coordinates (X, Y) moves to (-Y, X): row n-1-c, column r.
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            out.set(n - 1 - c, r, map.at(r, c));
        }
    }
    return out;
}

} // namespace waferedge::synth
