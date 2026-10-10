#pragma once

#include <atomic>
#include <cstdint>

// How a consumer thread sleeps on an empty SpscRing and gets woken when the producer pushes,
// without a mutex and without a system call per push (ADR-0014).
//
//   consumer                                     producer
//   key = prepare()      (announce: sleeping)    push into the ring (release)
//   if ring not empty: cancel(), go on           ring()  (if a sleeper is announced:
//   else wait(key)       (futex sleep)                    bump the epoch, futex wake)
//
// The lost wake-up to avoid: the consumer finds the ring empty, the producer pushes and sees
// nobody sleeping, then the consumer sleeps with a message waiting. Both sides therefore touch
// `sleeping_` with an exchange, a read-modify-write: RMWs on one atomic are totally ordered
// and each reads the latest value, so either the producer's exchange comes second and sees
// the announcement (and wakes the consumer), or the consumer's comes second and synchronizes
// with the producer's release, so its recheck of the ring sees the push. The epoch makes the
// sleep itself safe: wait(key) returns at once if the epoch moved since prepare().
//
// One waiter (the ring's consumer); any number of threads may ring.
namespace waferedge::pipeline {

class Doorbell {
public:
    // Consumer: announces it is about to sleep; returns the key for wait(). Recheck the
    // condition after this, before wait().
    [[nodiscard]] std::uint32_t prepare() noexcept {
        const std::uint32_t key = epoch_.load(std::memory_order_acquire);
        sleeping_.exchange(true, std::memory_order_acq_rel);
        return key;
    }
    // Consumer: the recheck found work; withdraw the announcement (saves the producer a wake).
    void cancel() noexcept { sleeping_.exchange(false, std::memory_order_acq_rel); }
    // Consumer: sleeps until ring() (or a spurious wake-up); recheck the condition after.
    void wait(std::uint32_t key) noexcept {
        epoch_.wait(key, std::memory_order_acquire);
        sleeping_.store(false, std::memory_order_relaxed);
    }

    // Producer, after publishing: wakes the consumer if it announced it sleeps. One RMW per
    // call (an uncontended exchange: tens of ns); the futex wake only when someone sleeps.
    void ring() noexcept {
        if (sleeping_.exchange(false, std::memory_order_acq_rel)) {
            wake();
        }
    }
    // Wakes the consumer whatever it announced (shutdown).
    void wake() noexcept {
        epoch_.fetch_add(1, std::memory_order_release);
        epoch_.notify_one();
    }

private:
    std::atomic<std::uint32_t> epoch_{0};
    std::atomic<bool> sleeping_{false};
};

} // namespace waferedge::pipeline
