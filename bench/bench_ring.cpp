// The SPSC ring against a mutex + condition-variable queue, and against itself without cached
// counters: throughput (messages/s, one producer and one consumer thread) and round-trip
// latency (ping-pong between two queues, percentiles from a 10 ns histogram).
//
//   build/release/bench/bench-ring [--cpus A B]
//
// The producer is pinned to CPU A and the consumer to CPU B (default 0 and 1) when the OS
// allows it. Which pair matters: two hyperthreads of one core share its L1 and L2 caches,
// two cores only the L3. On the laptop CPUs 0 and 1 are siblings
// (/sys/devices/system/cpu/cpu0/topology/thread_siblings_list), 0 and 2 separate cores.
#include "waferedge/pipeline/spsc_ring.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <format>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

using namespace waferedge::pipeline;
using Clock = std::chrono::steady_clock;

namespace {

bool pin(std::size_t cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

// The baseline: a bounded queue guarded by a mutex, blocking with condition variables,
// as a straightforward implementation would be written.
template <typename T, std::size_t N>
class MutexQueue {
public:
    using value_type = T;
    void push(const T& v) {
        std::unique_lock lock(mutex_);
        not_full_.wait(lock, [&] { return count_ < N; });
        slots_[(head_ + count_) % N] = v;
        ++count_;
        lock.unlock();
        not_empty_.notify_one();
    }
    void pop(T& out) {
        std::unique_lock lock(mutex_);
        not_empty_.wait(lock, [&] { return count_ > 0; });
        out = slots_[head_];
        head_ = (head_ + 1) % N;
        --count_;
        lock.unlock();
        not_full_.notify_one();
    }

private:
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::array<T, N> slots_{};
    std::size_t head_ = 0;
    std::size_t count_ = 0;
};

// Ring adaptors with the same blocking-by-spinning interface.
template <typename Ring>
void ring_push(Ring& r, const typename Ring::value_type& v) {
    while (!r.try_push(v)) {
    }
}
template <typename Ring>
void ring_pop(Ring& r, typename Ring::value_type& out) {
    while (!r.try_pop(out)) {
    }
}
template <typename T, std::size_t N>
void ring_push(MutexQueue<T, N>& q, const T& v) {
    q.push(v);
}
template <typename T, std::size_t N>
void ring_pop(MutexQueue<T, N>& q, T& out) {
    q.pop(out);
}

template <std::size_t Bytes>
struct Payload {
    std::uint64_t seq = 0;
    std::array<std::uint8_t, Bytes - 8> rest{};
};

bool pinned = true;
std::size_t cpu_a = 0; // producer, ping side
std::size_t cpu_b = 1; // consumer, echo side

template <typename Queue>
double throughput(std::uint64_t n) {
    using T = typename Queue::value_type;
    auto q = std::make_unique<Queue>();
    std::uint64_t sum = 0;
    const auto start = Clock::now();
    std::thread consumer([&] {
        pinned = pin(cpu_b) && pinned;
        T v;
        for (std::uint64_t i = 0; i < n; ++i) {
            ring_pop(*q, v);
            sum += v.seq;
        }
    });
    pinned = pin(cpu_a) && pinned;
    T v;
    for (std::uint64_t i = 0; i < n; ++i) {
        v.seq = i;
        ring_push(*q, v);
    }
    consumer.join();
    const double s = std::chrono::duration<double>(Clock::now() - start).count();
    if (sum != n * (n - 1) / 2) {
        std::puts("checksum mismatch");
    }
    return static_cast<double>(n) / s;
}

struct Latency {
    double p50 = 0, p99 = 0, p999 = 0;
};

// Ping-pong: thread A pushes into q1, B pops and pushes back into q2, A pops. One sample is
// one round trip (two hand-offs).
template <typename Queue>
Latency round_trip(std::uint64_t n) {
    using T = typename Queue::value_type;
    auto q1 = std::make_unique<Queue>();
    auto q2 = std::make_unique<Queue>();
    std::vector<std::uint64_t> histogram(100'000); // 10 ns buckets up to 1 ms
    std::thread echo([&] {
        pin(cpu_b);
        T v;
        for (std::uint64_t i = 0; i < n; ++i) {
            ring_pop(*q1, v);
            ring_push(*q2, v);
        }
    });
    pin(cpu_a);
    T v;
    for (std::uint64_t i = 0; i < n; ++i) {
        v.seq = i;
        const auto t0 = Clock::now();
        ring_push(*q1, v);
        ring_pop(*q2, v);
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0);
        ++histogram[std::min<std::size_t>(static_cast<std::size_t>(ns.count()) / 10,
                                          histogram.size() - 1)];
    }
    echo.join();
    const auto percentile = [&](double p) {
        const auto target = static_cast<std::uint64_t>(p * static_cast<double>(n));
        std::uint64_t seen = 0;
        for (std::size_t i = 0; i < histogram.size(); ++i) {
            seen += histogram[i];
            if (seen > target) {
                return static_cast<double>(i) * 10;
            }
        }
        return 1e6;
    };
    return {percentile(0.5), percentile(0.99), percentile(0.999)};
}

// Three runs each: on a shared machine one run can mislead by 3x. Median, and the range.
struct Runs {
    double median = 0, low = 0, high = 0;
};
template <typename Queue>
Runs repeat(std::uint64_t n) {
    std::array<double, 3> r{};
    for (auto& x : r) {
        x = throughput<Queue>(n);
    }
    std::ranges::sort(r);
    return {r[1], r[0], r[2]};
}
std::string show(const Runs& r) {
    return std::format("{:.1f} M/s ({:.1f}-{:.1f})", r.median / 1e6, r.low / 1e6, r.high / 1e6);
}

template <typename Ring, typename Uncached, typename Mutex>
void row(const char* payload, std::uint64_t n) {
    const Runs ring = repeat<Ring>(n);
    const Runs uncached = repeat<Uncached>(n);
    const Runs mutex = repeat<Mutex>(n / 10);
    std::puts(std::format("| {} | {} | {} | {} | {:.0f}x |", payload, show(ring), show(uncached),
                          show(mutex), ring.median / mutex.median)
                  .c_str());
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string_view(argv[1]) == "--cpus") {
        cpu_a = std::stoul(argv[2]);
        cpu_b = std::stoul(argv[3]);
    } else if (argc != 1) {
        std::fputs("usage: bench-ring [--cpus A B]\n", stderr);
        return 2;
    }
    constexpr std::size_t N = 1024;
    std::puts("Throughput, one producer and one consumer (median of 3 runs, range):\n");
    std::puts(
        "| Payload | SPSC ring | ring, no cached counters | mutex + condvar | ring / mutex |");
    std::puts("|---|---|---|---|---|");
    row<SpscRing<Payload<16>, N>, SpscRing<Payload<16>, N, false>, MutexQueue<Payload<16>, N>>(
        "16 B", 50'000'000);
    row<SpscRing<Payload<1600>, N>, SpscRing<Payload<1600>, N, false>,
        MutexQueue<Payload<1600>, N>>("1,600 B (a 40x40 map)", 5'000'000);

    std::puts("\nRound trip (ping-pong through two queues), 16 B:\n");
    std::puts("| Queue | p50 | p99 | p99.9 |");
    std::puts("|---|---|---|---|");
    const auto print = [](const char* name, Latency l) {
        std::puts(
            std::format("| {} | {:.0f} ns | {:.0f} ns | {:.0f} ns |", name, l.p50, l.p99, l.p999)
                .c_str());
    };
    print("SPSC ring", round_trip<SpscRing<Payload<16>, N>>(1'000'000));
    print("ring, no cached counters", round_trip<SpscRing<Payload<16>, N, false>>(1'000'000));
    print("mutex + condvar", round_trip<MutexQueue<Payload<16>, N>>(200'000));
    std::puts(pinned ? std::format("\n(threads pinned to CPUs {} and {})", cpu_a, cpu_b).c_str()
                     : "\n(threads not pinned)");
}
