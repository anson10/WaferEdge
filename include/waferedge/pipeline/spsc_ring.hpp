#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <type_traits>
#include <utility>

// A bounded, lock-free single-producer / single-consumer ring (ADR-0013): how the edge
// host's threads hand wafers to each other without locks, allocation or extra copies.
//
//   head_ (written by the consumer only)       tail_ (written by the producer only)
//      |                                            |
//      v                                            v
//   [ slot | slot | slot | ...               ... | slot ]     N slots, N a power of two
//
//   empty: head == tail      full: tail - head == N      slot of counter c: c & (N - 1)
//
// The counters only grow (64-bit: they don't wrap in practice), so full and empty never look
// alike. Each counter has one writer, so plain atomic loads and stores suffice, no CAS:
//   producer: fill slot[tail], then tail.store(release)   -> the consumer's tail.load(acquire)
//             sees the finished slot, never a half-written one;
//   consumer: read slot[head], then head.store(release)   -> the producer's head.load(acquire)
//             sees the slot is free before overwriting it.
// head_ and tail_ live on separate cache lines (no false sharing), and each side keeps a
// private copy of the other's counter, reloading it only when the ring looks full / empty:
// most operations then touch no shared cache line at all (Rigtorp's SPSCQueue).
//
// Exactly one producer thread and one consumer thread; either may be any thread.
namespace waferedge::pipeline {

// The x86-64 (and most ARM) cache line. std::hardware_destructive_interference_size would
// say the same, but GCC warns that its value may change between compiler versions (an ABI
// hazard for a header), so the constant is spelled out.
inline constexpr std::size_t kCacheLine = 64;

template <typename T, std::size_t N, bool CacheIndices = true>
class SpscRing {
    static_assert(N >= 2 && std::has_single_bit(N), "the capacity must be a power of two");
    static_assert(std::is_default_constructible_v<T>, "slots are constructed up front");

public:
    using value_type = T;
    static constexpr std::size_t capacity = N;

    SpscRing() = default;
    SpscRing(const SpscRing&) = delete;
    SpscRing& operator=(const SpscRing&) = delete;
    SpscRing(SpscRing&&) = delete;
    SpscRing& operator=(SpscRing&&) = delete;
    ~SpscRing() = default;

    // --- Producer -----------------------------------------------------------------------
    // The next free slot to fill in place, or nullptr if the ring is full. Nothing is
    // published until commit_push().
    [[nodiscard]] T* begin_push() noexcept {
        const std::size_t tail = tail_.value.load(std::memory_order_relaxed); // our own counter
        // Looks full with the cached head (or caching is off, the benchmark's comparison):
        // read the consumer's counter, the one access that touches its cache line.
        if (!CacheIndices || tail - producer_.head_cache == N) {
            producer_.head_cache = head_.value.load(std::memory_order_acquire);
            if (tail - producer_.head_cache == N) {
                return nullptr;
            }
        }
        return &slots_[tail & (N - 1)];
    }
    // Publishes the slot from begin_push().
    void commit_push() noexcept {
        tail_.value.store(tail_.value.load(std::memory_order_relaxed) + 1,
                          std::memory_order_release);
    }
    template <typename U>
    [[nodiscard]] bool try_push(U&& value) noexcept(std::is_nothrow_assignable_v<T&, U&&>) {
        T* slot = begin_push();
        if (slot == nullptr) {
            return false;
        }
        *slot = std::forward<U>(value);
        commit_push();
        return true;
    }

    // --- Consumer -----------------------------------------------------------------------
    // The oldest published slot to read in place, or nullptr if the ring is empty. It stays
    // valid until pop().
    [[nodiscard]] T* front() noexcept {
        const std::size_t head = head_.value.load(std::memory_order_relaxed); // our own counter
        if (!CacheIndices || consumer_.tail_cache == head) {
            consumer_.tail_cache = tail_.value.load(std::memory_order_acquire);
            if (consumer_.tail_cache == head) {
                return nullptr;
            }
        }
        return &slots_[head & (N - 1)];
    }
    // Frees the slot from front() for the producer.
    void pop() noexcept {
        head_.value.store(head_.value.load(std::memory_order_relaxed) + 1,
                          std::memory_order_release);
    }
    [[nodiscard]] bool try_pop(T& out) noexcept(std::is_nothrow_move_assignable_v<T>) {
        T* slot = front();
        if (slot == nullptr) {
            return false;
        }
        out = std::move(*slot);
        pop();
        return true;
    }

    // --- Either side (a snapshot: the other side may change it right after) ------------
    [[nodiscard]] std::size_t size_approx() const noexcept {
        const std::size_t head = head_.value.load(std::memory_order_acquire);
        const std::size_t tail = tail_.value.load(std::memory_order_acquire);
        return tail - head;
    }
    [[nodiscard]] bool empty_approx() const noexcept { return size_approx() == 0; }

private:
    // One counter per cache line: the producer writes tail_, the consumer head_, and neither
    // write invalidates the line the other side is reading most of the time.
    struct alignas(kCacheLine) Counter {
        std::atomic<std::size_t> value{0};
    };
    // Each side's private copy of the other's counter, on its own line too.
    struct alignas(kCacheLine) ProducerState {
        std::size_t head_cache = 0;
    };
    struct alignas(kCacheLine) ConsumerState {
        std::size_t tail_cache = 0;
    };

    Counter head_;
    ConsumerState consumer_;
    Counter tail_;
    ProducerState producer_;
    // Slots after the counters, starting on a fresh line.
    alignas(kCacheLine) std::array<T, N> slots_{};
};

} // namespace waferedge::pipeline
