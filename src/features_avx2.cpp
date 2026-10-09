// AVX2 feature backend: the scalar histogram turned around.
//
// Scalar: for each die, increment the counter of its zone, sector and ring (a scatter, with
// load-modify-store chains when neighbouring dies hit the same counter).
// AVX2: for each bucket, ask 32 dies at once "are you in this bucket?" (one compare), and add
// the answers into a vector of 32 byte counters. No memory writes in the loop, no conflicts.
//
// Only the functions marked [[gnu::target("avx2")]] use AVX2; the file is compiled for the
// baseline CPU, so inline functions from shared headers never pick up AVX2 instructions
// (the linker keeps one copy of each, and it must run everywhere). docs/avx2.md walks through
// the code and the measurements.
#include "features_detail.hpp"
#include "waferedge/backend.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace waferedge {

#if defined(__x86_64__)

namespace {

constexpr std::size_t kBlock = 32;     // dies per 256-bit vector (32 x 8 bits)
constexpr std::size_t kMaxSteps = 255; // a byte counter gains at most 1 per step
// Buckets per pass. Measured on the Ryzen 5 7535HS (docs/avx2.md): 3 is fastest. At 4 or more
// the compiler spills counters to the stack (a store/reload chain every block); at 2 the
// extra passes over the data cost more than they save.
constexpr std::size_t kPerPass = 3;

// Adds up the 32 byte lanes of a counter vector. vpsadbw against zero sums each group of 8
// bytes into a 64-bit lane (4 lanes), then the 4 lanes are added.
[[gnu::target("avx2")]] std::uint64_t sum_bytes(__m256i counts) noexcept {
    const __m256i sums = _mm256_sad_epu8(counts, _mm256_setzero_si256());
    const __m128i both =
        _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    return static_cast<std::uint64_t>(_mm_cvtsi128_si64(both)) +
           static_cast<std::uint64_t>(_mm_extract_epi64(both, 1));
}

// One pass over the full 32-die blocks for K buckets (first .. first+K-1) of one table.
// The 2K byte counters must stay in registers for the whole inner loop: there are 16 ymm
// registers, and the loop also needs the bins, the table, two masks, the bucket keys and
// temporaries. Too many buckets per pass and counters spill to the stack.
template <std::size_t K>
[[gnu::target("avx2")]] void count_pass(const std::uint8_t* bins, const std::uint8_t* table,
                                        std::size_t blocks, std::size_t first, std::uint32_t* dies,
                                        std::uint32_t* fails) noexcept {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i two = _mm256_set1_epi8(2);
    std::array<std::uint64_t, K> total_dies{};
    std::array<std::uint64_t, K> total_fails{};

    std::size_t block = 0;
    while (block < blocks) {
        // Byte counters overflow after 255 increments: count at most 255 blocks, then flush.
        const std::size_t end = std::min(blocks, block + kMaxSteps);
        // C arrays on purpose: std::array<__m256i, K> drops the vector type's attributes
        // (GCC: "ignoring attributes on template argument"), its 32-byte alignment included.
        __m256i die_count[K];  // NOLINT(*-avoid-c-arrays)
        __m256i fail_count[K]; // NOLINT(*-avoid-c-arrays)
        for (std::size_t k = 0; k < K; ++k) {
            die_count[k] = zero;
            fail_count[k] = zero;
        }
        for (; block < end; ++block) {
            // Unaligned loads: maps sit back to back in the MapSet buffer, so a block
            // starts anywhere. On this CPU an unaligned load costs the same as an aligned one
            // unless it crosses a cache line.
            const __m256i b =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bins + block * kBlock));
            const __m256i t =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(table + block * kBlock));
            // Lane masks are 0xFF (true) or 0x00 (false).
            const __m256i off_wafer = _mm256_cmpeq_epi8(b, zero);
            // AVX2 has no unsigned >= for bytes: b >= 2 exactly when max(b, 2) == b.
            const __m256i fail = _mm256_cmpeq_epi8(_mm256_max_epu8(b, two), b);
            for (std::size_t k = 0; k < K; ++k) {
                const __m256i in_bucket =
                    _mm256_cmpeq_epi8(t, _mm256_set1_epi8(static_cast<char>(first + k)));
                // 0xFF is -1 as a signed byte: subtracting the mask adds 1 where it is set.
                // andnot(a, b) = ~a & b: in the bucket and not off the wafer.
                die_count[k] =
                    _mm256_sub_epi8(die_count[k], _mm256_andnot_si256(off_wafer, in_bucket));
                // A fail die is on the wafer by definition (bin >= 2), so no off-wafer test.
                fail_count[k] = _mm256_sub_epi8(fail_count[k], _mm256_and_si256(in_bucket, fail));
            }
        }
        for (std::size_t k = 0; k < K; ++k) {
            total_dies[k] += sum_bytes(die_count[k]);
            total_fails[k] += sum_bytes(fail_count[k]);
        }
    }
    for (std::size_t k = 0; k < K; ++k) {
        dies[first + k] += static_cast<std::uint32_t>(total_dies[k]);
        fails[first + k] += static_cast<std::uint32_t>(total_fails[k]);
    }
}

// All Count buckets of one table, kPerPass at a time: zones 3 + 2, sectors 3 + 3 + 2,
// rings 3 + 3 + 3 + 1. The split happens at compile time, so every pass has a fixed K.
template <std::size_t Count, std::size_t First = 0>
[[gnu::target("avx2")]] void count_table(const std::uint8_t* bins, const std::uint8_t* table,
                                         std::size_t blocks, std::uint32_t* dies,
                                         std::uint32_t* fails) noexcept {
    if constexpr (First < Count) {
        constexpr std::size_t k = std::min(kPerPass, Count - First);
        count_pass<k>(bins, table, blocks, First, dies, fails);
        count_table<Count, First + k>(bins, table, blocks, dies, fails);
    }
}

[[gnu::target("avx2")]] Features features_avx2(WaferMapView map,
                                               const Geometry& geometry) noexcept {
    assert(map.rows() == geometry.rows() && map.cols() == geometry.cols());
    const std::uint8_t* bins = map.bins().data();
    const std::uint8_t* zones = geometry.zones().data();
    const std::uint8_t* sectors = geometry.sectors().data();
    const std::uint8_t* rings = geometry.rings().data();
    const std::size_t n = map.bins().size();
    const std::size_t blocks = n / kBlock;

    Features f;
    count_table<kZones>(bins, zones, blocks, f.zone_dies.data(), f.zone_fails.data());
    count_table<kSectors>(bins, sectors, blocks, f.sector_dies.data(), f.sector_fails.data());
    count_table<kRings>(bins, rings, blocks, f.ring_dies.data(), f.ring_fails.data());
    // The last n % 32 dies: a full 32-byte load would read past the map (and past the end
    // of the MapSet buffer for the last map), so they go one at a time.
    detail::count_dies(f, bins, zones, sectors, rings, blocks * kBlock, n);
    detail::finish_totals(f);
    return f;
}

} // namespace

bool backend::Avx2::available() noexcept {
    __builtin_cpu_init();
    // The builtin returns int on GCC and bool on Clang; a condition reads right on both.
    bool has_avx2 = false;
    if (__builtin_cpu_supports("avx2")) {
        has_avx2 = true;
    }
    return has_avx2;
}

Features backend::Avx2::features(WaferMapView map, const Geometry& geometry) noexcept {
    return features_avx2(map, geometry);
}

#else // not x86: the backend exists but is never available

bool backend::Avx2::available() noexcept {
    return false;
}

Features backend::Avx2::features(WaferMapView map, const Geometry& geometry) noexcept {
    return compute_features(map, geometry);
}

#endif

} // namespace waferedge
