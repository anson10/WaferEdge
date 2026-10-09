#include "waferedge/thread_pool.hpp"

#include <algorithm>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace waferedge {

namespace {

// Tells the CPU this is a spin-wait: it slows the loop down a little and leaves the core's
// resources to the sibling hardware thread instead of hammering the memory system.
void cpu_relax() noexcept {
#if defined(__x86_64__)
    _mm_pause();
#endif
}

} // namespace

ThreadPool::ThreadPool(unsigned threads, std::chrono::microseconds spin)
    : size_(threads != 0 ? threads : std::max(1U, std::thread::hardware_concurrency())),
      spin_(spin) {
    threads_.reserve(size_ - 1);
    for (unsigned w = 1; w < size_; ++w) {
        threads_.emplace_back([this, w] { worker_loop(w); });
    }
}

ThreadPool::~ThreadPool() {
    {
        const std::lock_guard lock(mutex_);
        stop_.store(true, std::memory_order_release);
    }
    start_.notify_all();
    for (auto& t : threads_) {
        t.join();
    }
}

template <typename Pred>
bool ThreadPool::spin_until(Pred pred) const noexcept {
    if (spin_.count() == 0) {
        return pred();
    }
    const auto deadline = std::chrono::steady_clock::now() + spin_;
    while (true) {
        // Check the clock only every 64 polls: reading it costs more than a poll.
        for (int i = 0; i < 64; ++i) {
            if (pred()) {
                return true;
            }
            cpu_relax();
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return pred();
        }
    }
}

void ThreadPool::work(unsigned worker) noexcept {
    while (true) {
        // relaxed is enough for the counter: it only decides who takes which chunk; the job's
        // data was published to this thread by the acquire load of the generation.
        const std::size_t begin = next_.fetch_add(grain_, std::memory_order_relaxed);
        if (begin >= n_) {
            return;
        }
        call_(fn_, worker, begin, std::min(n_, begin + grain_));
    }
}

void ThreadPool::worker_loop(unsigned worker) noexcept {
    std::uint64_t seen = 0;
    const auto new_job = [&] {
        return stop_.load(std::memory_order_acquire) ||
               generation_.load(std::memory_order_acquire) != seen;
    };
    while (true) {
        if (!spin_until(new_job)) {
            std::unique_lock lock(mutex_);
            start_.wait(lock, new_job);
        }
        if (stop_.load(std::memory_order_acquire)) {
            return;
        }
        seen = generation_.load(std::memory_order_acquire);
        work(worker);
        // The last worker out wakes the caller if it went to sleep. acq_rel: this worker's
        // results happen-before the caller's acquire load that sees busy_ == 0.
        if (busy_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            const std::lock_guard lock(mutex_);
            done_.notify_one();
        }
    }
}

void ThreadPool::run(std::size_t n, std::size_t grain, const void* fn, Trampoline call) {
    if (n == 0) {
        return;
    }
    if (size_ == 1 || n <= grain) {
        // Not worth waking anyone: the caller runs it alone, still in chunks of <= grain.
        for (std::size_t begin = 0; begin < n; begin += grain) {
            call(fn, 0, begin, std::min(n, begin + grain));
        }
        return;
    }
    {
        // Under the mutex so that a worker deciding to sleep can't miss this job.
        const std::lock_guard lock(mutex_);
        fn_ = fn;
        call_ = call;
        n_ = n;
        grain_ = grain;
        next_.store(0, std::memory_order_relaxed);
        busy_.store(size_ - 1, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release); // publishes everything above
    }
    start_.notify_all();
    work(0); // the caller is worker 0
    const auto all_done = [&] { return busy_.load(std::memory_order_acquire) == 0; };
    if (!spin_until(all_done)) {
        std::unique_lock lock(mutex_);
        done_.wait(lock, all_done);
    }
}

} // namespace waferedge
