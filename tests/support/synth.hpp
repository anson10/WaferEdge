#pragma once

// Synthetic wafer maps for tests and benchmarks: a disc of passing dies, then fails painted
// where a predicate holds or at random. Deterministic for a given seed on every platform
// (std::mt19937's output is specified; the standard distributions are not, so none are used).
#include "waferedge/wafer_map.hpp"

#include <cstdint>
#include <random>

namespace waferedge::synth {

// rows x cols grid; on-wafer (bin 1) inside the inscribed ellipse, the same rho <= 1 as the
// geometry's zones, off-wafer (0) outside.
[[nodiscard]] WaferMap disc(int rows, int cols);

// Normalised squared radius of a die centre, in [0, 2]: the geometry's rho^2 as a double.
[[nodiscard]] double rho2(int row, int col, int rows, int cols) noexcept;

// Sets every on-wafer die where pred(row, col) holds to `bin`.
template <typename Pred>
void paint(WaferMap& map, Pred pred, std::uint8_t bin = 2) {
    for (int r = 0; r < map.rows(); ++r) {
        for (int c = 0; c < map.cols(); ++c) {
            if (on_wafer(map.at(r, c)) && pred(r, c)) {
                map.set(r, c, bin);
            }
        }
    }
}

// Fails each on-wafer die with probability per_mille / 1000, using a bin in 2..5.
void sprinkle(WaferMap& map, unsigned per_mille, std::mt19937& rng);

// A disc with a random fail field: the generic input for property tests and benchmarks.
[[nodiscard]] WaferMap random_map(int rows, int cols, unsigned per_mille, std::uint32_t seed);

// Rotates a square map 90 degrees counter-clockwise (precondition: rows == cols).
[[nodiscard]] WaferMap rotate90(const WaferMap& map);

} // namespace waferedge::synth
