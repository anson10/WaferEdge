#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

// A latency histogram in the style of HdrHistogram: log-linear buckets, so every value from
// 1 ns to centuries is kept to within 1 / 128 of itself in a fixed 58 KB, recording is O(1)
// and never allocates, and percentiles are read from the counts (ADR-0015).
//
//   values 0 .. 255          one bucket each (exact)
//   values 256 .. 511        128 buckets of width 2
//   values 512 .. 1023       128 buckets of width 4      ... and so on: in every power-of-two
//                                                        range, 128 equal sub-buckets
//
// A bucket's width is at most 1/128 of the values it holds, so a percentile read from the
// histogram is at most ~0.8% above the exact one. Percentiles report the bucket's upper edge:
// never below the true value. The maximum and minimum are kept exactly.
namespace waferedge::pipeline {

class LatencyHistogram {
public:
    static constexpr unsigned kSubBits = 7; // 2^7 = 128 sub-buckets per power of two
    static constexpr std::uint64_t kSub = std::uint64_t{1} << kSubBits;

    LatencyHistogram() : counts_(bucket_count(), 0) {}

    void record(std::uint64_t value) noexcept {
        ++counts_[index(value)];
        ++count_;
        min_ = std::min(min_, value);
        max_ = std::max(max_, value);
    }
    void merge(const LatencyHistogram& other) noexcept {
        for (std::size_t i = 0; i < counts_.size(); ++i) {
            counts_[i] += other.counts_[i];
        }
        count_ += other.count_;
        min_ = std::min(min_, other.min_);
        max_ = std::max(max_, other.max_);
    }

    [[nodiscard]] std::uint64_t count() const noexcept { return count_; }
    [[nodiscard]] std::uint64_t min() const noexcept { return count_ == 0 ? 0 : min_; }
    [[nodiscard]] std::uint64_t max() const noexcept { return max_; }

    // The smallest recorded-bucket upper edge with at least a fraction p (0..1] of the values
    // at or below it; p = 1 gives the exact maximum. 0 if empty.
    [[nodiscard]] std::uint64_t percentile(double p) const noexcept {
        if (count_ == 0) {
            return 0;
        }
        if (p >= 1.0) {
            return max_;
        }
        // The rank of the value wanted: ceil(p * count), at least 1.
        auto rank = static_cast<std::uint64_t>(p * static_cast<double>(count_));
        if (static_cast<double>(rank) < p * static_cast<double>(count_) || rank == 0) {
            ++rank;
        }
        std::uint64_t seen = 0;
        for (std::size_t i = 0; i < counts_.size(); ++i) {
            seen += counts_[i];
            if (seen >= rank) {
                return std::min(upper(i), max_); // never past the largest value seen
            }
        }
        return max_;
    }

    // Bucket arithmetic, public for the tests.
    [[nodiscard]] static constexpr std::size_t index(std::uint64_t v) noexcept {
        const auto bits = static_cast<unsigned>(std::bit_width(v));
        if (bits <= kSubBits + 1) {
            return static_cast<std::size_t>(v); // below 256: exact
        }
        // v has `bits` significant bits: keep the top kSubBits + 1 (the leading 1 and the
        // next 7), drop the rest. Each extra bit of magnitude adds kSub buckets.
        const unsigned shift = bits - (kSubBits + 1);
        return static_cast<std::size_t>(shift * kSub + (v >> shift));
    }
    // The largest value that lands in bucket i.
    [[nodiscard]] static constexpr std::uint64_t upper(std::size_t i) noexcept {
        if (i < 2 * kSub) {
            return i;
        }
        const auto shift = static_cast<unsigned>(i / kSub - 1);
        const std::uint64_t top = i - shift * kSub; // in [kSub, 2 kSub)
        return ((top + 1) << shift) - 1;
    }
    [[nodiscard]] static constexpr std::size_t bucket_count() noexcept {
        return index(~std::uint64_t{0}) + 1;
    }

private:
    std::vector<std::uint64_t> counts_; // sized once
    std::uint64_t count_ = 0;
    std::uint64_t min_ = ~std::uint64_t{0};
    std::uint64_t max_ = 0;
};

} // namespace waferedge::pipeline
