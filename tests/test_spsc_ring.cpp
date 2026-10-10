// The SPSC ring: single-thread semantics, then two-thread stress tests with a checker on
// every message. The whole file runs under ThreadSanitizer in the tsan preset (and in CI):
// a missing acquire / release on the counters shows up there as a data race on a slot.
#include "waferedge/pipeline/spsc_ring.hpp"

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <thread>

using namespace waferedge::pipeline;

namespace {

// A payload whose every field depends on its sequence number: a torn or stale read can't
// pass the check.
struct Small {
    std::uint64_t seq = 0;
    std::uint64_t check = 0;
};
constexpr std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33U;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33U;
    return x;
}

// A wafer-map-sized payload: every byte must match.
struct Big {
    std::uint64_t seq = 0;
    std::array<std::uint8_t, 2040> bytes{};
};

// Runs a producer and a consumer through the ring for `n` messages; returns the number of
// messages that failed the check (0 expected) and whether the order held.
template <typename Ring, typename Fill, typename Check>
std::uint64_t stress(Ring& ring, std::uint64_t n, Fill fill, Check check) {
    std::uint64_t bad = 0;
    std::thread consumer([&] {
        std::uint64_t expected = 0;
        while (expected < n) {
            auto* slot = ring.front();
            if (slot == nullptr) {
                std::this_thread::yield(); // keeps the test fair on few cores and under TSan
                continue;
            }
            if (!check(*slot, expected)) {
                ++bad;
            }
            ring.pop();
            ++expected;
        }
    });
    for (std::uint64_t i = 0; i < n;) {
        auto* slot = ring.begin_push();
        if (slot == nullptr) {
            std::this_thread::yield();
            continue;
        }
        fill(*slot, i);
        ring.commit_push();
        ++i;
    }
    consumer.join();
    return bad;
}

} // namespace

TEST_CASE("an empty ring has nothing; a full one takes nothing", "[ring]") {
    SpscRing<int, 4> ring;
    CHECK(ring.front() == nullptr);
    CHECK(ring.empty_approx());
    for (int i = 0; i < 4; ++i) {
        CHECK(ring.try_push(i));
    }
    CHECK(ring.size_approx() == 4);
    CHECK_FALSE(ring.try_push(99));
    CHECK(ring.begin_push() == nullptr);
    int out = -1;
    for (int i = 0; i < 4; ++i) {
        REQUIRE(ring.try_pop(out));
        CHECK(out == i); // first in, first out
    }
    CHECK_FALSE(ring.try_pop(out));
}

TEST_CASE("the counters wrap around the slots many times", "[ring]") {
    SpscRing<std::uint64_t, 8> ring;
    std::uint64_t next_in = 0;
    std::uint64_t next_out = 0;
    // Fill to varying depths, then drain partly: every slot index is reused many times.
    for (int round = 0; round < 1000; ++round) {
        const auto fill = static_cast<std::uint64_t>(1 + round % 8);
        for (std::uint64_t k = 0; k < fill && ring.try_push(next_in); ++k) {
            ++next_in;
        }
        const auto drain = static_cast<std::uint64_t>(1 + (round * 7) % 8);
        std::uint64_t out = 0;
        for (std::uint64_t k = 0; k < drain && ring.try_pop(out); ++k) {
            REQUIRE(out == next_out);
            ++next_out;
        }
    }
    CHECK(next_in - next_out == ring.size_approx());
    CHECK(next_in > 1000);
}

TEST_CASE("slots are filled and read in place", "[ring]") {
    SpscRing<Big, 2> ring;
    Big* slot = ring.begin_push();
    REQUIRE(slot != nullptr);
    slot->seq = 7;
    slot->bytes.fill(7);
    CHECK(ring.front() == nullptr); // not published yet
    ring.commit_push();
    const Big* read = ring.front();
    REQUIRE(read == slot); // the same memory: no copy
    CHECK(read->bytes[2039] == 7);
    ring.pop();
    CHECK(ring.front() == nullptr);
}

TEST_CASE("layout: counters and slots on their own cache lines", "[ring]") {
    using Ring = SpscRing<Small, 16>;
    STATIC_CHECK(alignof(Ring) >= kCacheLine);
    // head, consumer cache, tail, producer cache: four lines before the slots.
    STATIC_CHECK(sizeof(Ring) >= 4 * kCacheLine + sizeof(Small) * 16);
}

TEST_CASE("two threads, a tiny ring: every message arrives once, in order, intact", "[ring]") {
    auto ring = std::make_unique<SpscRing<Small, 16>>();
    constexpr std::uint64_t n = 1'000'000;
    const auto bad = stress(
        *ring, n,
        [](Small& s, std::uint64_t i) {
            s.seq = i;
            s.check = mix(i);
        },
        [](const Small& s, std::uint64_t expected) {
            return s.seq == expected && s.check == mix(expected);
        });
    CHECK(bad == 0);
    CHECK(ring->empty_approx());
}

TEST_CASE("two threads, big slots: no torn reads", "[ring]") {
    auto ring = std::make_unique<SpscRing<Big, 8>>();
    constexpr std::uint64_t n = 100'000;
    const auto bad = stress(
        *ring, n,
        [](Big& b, std::uint64_t i) {
            b.seq = i;
            b.bytes.fill(static_cast<std::uint8_t>(i * 31));
        },
        [](const Big& b, std::uint64_t expected) {
            const auto v = static_cast<std::uint8_t>(expected * 31);
            if (b.seq != expected) {
                return false;
            }
            for (const auto x : b.bytes) {
                if (x != v) {
                    return false;
                }
            }
            return true;
        });
    CHECK(bad == 0);
}

TEST_CASE("the uncached variant is just as correct", "[ring]") {
    auto ring = std::make_unique<SpscRing<Small, 2, false>>(); // capacity 2: always contended
    constexpr std::uint64_t n = 200'000;
    const auto bad = stress(
        *ring, n, [](Small& s, std::uint64_t i) { s = {i, mix(i)}; },
        [](const Small& s, std::uint64_t expected) {
            return s.seq == expected && s.check == mix(expected);
        });
    CHECK(bad == 0);
}
