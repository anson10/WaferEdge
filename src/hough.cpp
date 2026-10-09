#include "waferedge/hough.hpp"

#include <cmath>
#include <numbers>

namespace waferedge {

namespace {

struct Tables {
    std::array<std::int32_t, kHoughAngles> cos{};
    std::array<std::int32_t, kHoughAngles> sin{};

    Tables() noexcept {
        constexpr double one = 1 << kHoughFracBits;
        for (int k = 0; k < kHoughAngles; ++k) {
            const double rad = k * std::numbers::pi / 180.0;
            const auto i = static_cast<std::size_t>(k);
            cos[i] = static_cast<std::int32_t>(std::lround(std::cos(rad) * one));
            sin[i] = static_cast<std::int32_t>(std::lround(std::sin(rad) * one));
        }
    }
};

const Tables& tables() noexcept {
    static const Tables t;
    return t;
}

} // namespace

const std::array<std::int32_t, kHoughAngles>& hough_cos() noexcept {
    return tables().cos;
}
const std::array<std::int32_t, kHoughAngles>& hough_sin() noexcept {
    return tables().sin;
}

int hough_rho(int x, int y, int angle) noexcept {
    const auto k = static_cast<std::size_t>(angle);
    const std::int32_t raw = x * tables().cos[k] + y * tables().sin[k];
    // raw is in half-die units times 2^14; one die width is 2^(14+1). >> on a negative int
    // is an arithmetic (floor) shift since C++20.
    return raw >> (kHoughFracBits + 1);
}

HoughLine HoughTransform::run(WaferMapView map) {
    const int rows = map.rows();
    const int cols = map.cols();
    // |rho| <= (|x| + |y|) / 2 + 1 die widths, with |x| <= cols - 1 and |y| <= rows - 1.
    offset_ = (rows + cols) / 2 + 1;
    const int bins = rho_bins();
    votes_.assign(static_cast<std::size_t>(kHoughAngles) * static_cast<std::size_t>(bins), 0);
    const auto& t = tables();

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!is_fail(map.at(r, c))) {
                continue;
            }
            const int x = 2 * c - (cols - 1);
            const int y = (rows - 1) - 2 * r;
            std::uint32_t* row = votes_.data();
            for (std::size_t k = 0; k < kHoughAngles; ++k, row += bins) {
                const std::int32_t raw = x * t.cos[k] + y * t.sin[k];
                ++row[(raw >> (kHoughFracBits + 1)) + offset_];
            }
        }
    }

    HoughLine best;
    for (int k = 0; k < kHoughAngles; ++k) {
        for (int b = 0; b < bins; ++b) {
            const auto v = votes_[static_cast<std::size_t>(k) * static_cast<std::size_t>(bins) +
                                  static_cast<std::size_t>(b)];
            if (v > best.votes) {
                best = HoughLine{.angle = k, .rho = b - offset_, .votes = v, .line_dies = 0};
            }
        }
    }
    if (best.votes == 0) {
        return HoughLine{};
    }
    // How many dies the peak line could have collected: on-wafer dies in the same bin.
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (on_wafer(map.at(r, c)) &&
                hough_rho(2 * c - (cols - 1), (rows - 1) - 2 * r, best.angle) == best.rho) {
                ++best.line_dies;
            }
        }
    }
    return best;
}

} // namespace waferedge
