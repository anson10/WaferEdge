#pragma once

// A fixed pool of threads for data-parallel batches: the CPU side of the CPU-vs-GPU
// comparison (docs/gpu.md), and the CPU backend of the pipeline later.
//
// The threads start once. A job hands out chunks of `grain` items from an atomic counter, so
// threads that finish early take more work: maps don't all cost the same (the Hough
// transform scales with the fail dies). The calling thread works too.
//
// Between jobs a worker first spins for up to `spin` (polling an atomic, with the CPU's pause
// hint) and only then sleeps on a condition variable. Waking a sleeping thread took
// ~100-200 us per job on WSL2 (docs/gpu.md), far more than the work of a small batch; a
// spinning worker starts in well under a microsecond, at the price of burning its core
// while it waits. spin = 0 sleeps at once.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace waferedge {

class ThreadPool {
public:
    // `threads` in total, including the caller of for_each (so threads - 1 workers); 0 means
    // std::thread::hardware_concurrency().
    explicit ThreadPool(unsigned threads = 0,
                        std::chrono::microseconds spin = std::chrono::microseconds{0});
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    [[nodiscard]] unsigned size() const noexcept { return size_; }

    // Calls fn(worker, begin, end) on chunks [begin, end) of at most `grain` items that
    // together cover [0, n) exactly once; worker is in [0, size()) and is the same for every
    // chunk a thread runs, so fn can index per-thread scratch with it. Returns when every chunk
    // is done. fn must not throw (it runs on worker threads), and for_each must not be called
    // from two threads at once.
    template <typename Fn>
    void for_each(std::size_t n, std::size_t grain, const Fn& fn) {
        static_assert(std::is_nothrow_invocable_v<const Fn&, unsigned, std::size_t, std::size_t>,
                      "for_each's function must be noexcept");
        run(n, grain == 0 ? 1 : grain, &fn,
            [](const void* f, unsigned worker, std::size_t begin, std::size_t end) noexcept {
                (*static_cast<const Fn*>(f))(worker, begin, end);
            });
    }

private:
    using Trampoline = void (*)(const void*, unsigned, std::size_t, std::size_t) noexcept;

    void run(std::size_t n, std::size_t grain, const void* fn, Trampoline call);
    void work(unsigned worker) noexcept; // take chunks of the current job until none are left
    void worker_loop(unsigned worker) noexcept;
    // Spins until pred() holds or `spin` has passed; true if pred() held.
    template <typename Pred>
    bool spin_until(Pred pred) const noexcept;

    unsigned size_;
    std::chrono::microseconds spin_;
    std::vector<std::thread> threads_;

    // The current job: written by run() before it publishes a new generation (a release
    // store), read by workers after they see that generation (an acquire load). That pair
    // orders the writes before the reads, for spinning and sleeping workers alike.
    const void* fn_ = nullptr;
    Trampoline call_ = nullptr;
    std::size_t n_ = 0;
    std::size_t grain_ = 1;
    std::atomic<std::size_t> next_{0}; // the next unclaimed item

    std::atomic<std::uint64_t> generation_{0}; // bumped for each job
    std::atomic<unsigned> busy_{0};            // workers still in the current job
    std::atomic<bool> stop_{false};
    // For workers (and the caller) that stopped spinning and sleep. Changes that a sleeper
    // waits for are made while holding the mutex, so a wake-up is never lost.
    std::mutex mutex_;
    std::condition_variable start_;
    std::condition_variable done_;
};

} // namespace waferedge
