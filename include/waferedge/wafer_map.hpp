#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

// The wafer map: a dense row-major grid of bin codes, one byte per die.
//   0 = off wafer, 1 = pass, >= 2 = a fail bin.
// WaferLens maps carry fail bins 2..5; WM-811K maps only 2.
namespace waferedge {

[[nodiscard]] constexpr bool on_wafer(std::uint8_t bin) noexcept {
    return bin != 0;
}
[[nodiscard]] constexpr bool is_fail(std::uint8_t bin) noexcept {
    return bin >= 2;
}

// Spatial signatures, in FabEye's class order (the CNN's logits use the same indices).
enum class Pattern : std::uint8_t {
    none = 0,
    center = 1,
    donut = 2,
    edge_loc = 3,
    edge_ring = 4,
    loc = 5,
    near_full = 6,
    random = 7,
    scratch = 8,
    unknown = 255, // no label
};
inline constexpr std::size_t kPatternCount = 9;

[[nodiscard]] std::string_view pattern_name(Pattern p) noexcept;

// Which part of a dataset a map belongs to. Thresholds are fitted on train only.
enum class Split : std::uint8_t { train = 0, val = 1, test = 2, unsplit = 3 };

[[nodiscard]] std::string_view split_name(Split s) noexcept;

// Non-owning view of one map. Cheap to copy; the bins live elsewhere (a MapSet's buffer, a
// receive buffer, a WaferMap).
class WaferMapView {
public:
    constexpr WaferMapView() noexcept = default;
    // Precondition: bins.size() == rows * cols.
    constexpr WaferMapView(std::span<const std::uint8_t> bins, int rows, int cols) noexcept
        : bins_(bins), rows_(rows), cols_(cols) {}

    [[nodiscard]] constexpr int rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr int cols() const noexcept { return cols_; }
    [[nodiscard]] constexpr std::span<const std::uint8_t> bins() const noexcept { return bins_; }
    [[nodiscard]] constexpr std::uint8_t at(int row, int col) const noexcept {
        return bins_[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols_) +
                     static_cast<std::size_t>(col)];
    }

private:
    std::span<const std::uint8_t> bins_;
    int rows_ = 0;
    int cols_ = 0;
};

// An owning map, for tests, synthetic data and maps built at run time. The pipeline uses views.
class WaferMap {
public:
    WaferMap() = default;
    WaferMap(int rows, int cols, std::uint8_t fill = 0);

    [[nodiscard]] int rows() const noexcept { return rows_; }
    [[nodiscard]] int cols() const noexcept { return cols_; }
    [[nodiscard]] std::uint8_t at(int row, int col) const noexcept { return view().at(row, col); }
    void set(int row, int col, std::uint8_t bin) noexcept {
        bins_[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols_) +
              static_cast<std::size_t>(col)] = bin;
    }
    [[nodiscard]] std::span<std::uint8_t> bins() noexcept { return bins_; }
    [[nodiscard]] WaferMapView view() const noexcept { return {bins_, rows_, cols_}; }
    operator WaferMapView() const noexcept { return view(); } // NOLINT(google-explicit-constructor)

private:
    std::vector<std::uint8_t> bins_;
    int rows_ = 0;
    int cols_ = 0;
};

} // namespace waferedge
