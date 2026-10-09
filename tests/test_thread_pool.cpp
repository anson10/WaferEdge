// The thread pool: every item exactly once, worker ids in range and stable per thread, and
// thousands of jobs back to back (run under the tsan preset to check the synchronisation).
#include "waferedge/thread_pool.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

using namespace waferedge;

using std::chrono::microseconds;

TEST_CASE("every item is processed exactly once, for any size and grain") {
    const unsigned threads = GENERATE(1U, 2U, 4U, 12U);
    const auto spin = GENERATE(microseconds{0}, microseconds{50});
    ThreadPool pool(threads, spin);
    REQUIRE(pool.size() == threads);
    for (const std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{11},
                                std::size_t{1000}, std::size_t{100003}}) {
        for (const std::size_t grain : {std::size_t{1}, std::size_t{16}, std::size_t{5000}}) {
            std::vector<std::atomic<int>> hits(n);
            std::atomic<bool> bad_worker{false};
            pool.for_each(n, grain,
                          [&](unsigned worker, std::size_t begin, std::size_t end) noexcept {
                              if (worker >= threads || end - begin > grain || begin >= end) {
                                  bad_worker = true;
                              }
                              for (std::size_t i = begin; i < end; ++i) {
                                  hits[i].fetch_add(1, std::memory_order_relaxed);
                              }
                          });
            INFO(threads << " threads, n " << n << ", grain " << grain);
            CHECK_FALSE(bad_worker.load());
            std::size_t wrong = 0;
            for (auto& h : hits) {
                wrong += h.load() == 1 ? 0U : 1U;
            }
            CHECK(wrong == 0);
        }
    }
}

TEST_CASE("a worker id belongs to one thread, so per-worker scratch is never shared") {
    ThreadPool pool(8);
    std::mutex m;
    std::map<unsigned, std::set<std::thread::id>> owners;
    for (int job = 0; job < 50; ++job) {
        pool.for_each(10000, 7, [&](unsigned worker, std::size_t, std::size_t) noexcept {
            const std::lock_guard lock(m);
            owners[worker].insert(std::this_thread::get_id());
        });
    }
    for (const auto& [worker, threads] : owners) {
        INFO("worker " << worker);
        CHECK(threads.size() == 1);
    }
}

TEST_CASE("thousands of jobs back to back, small and large") {
    // Spinning workers catch most jobs without sleeping, sleeping ones none: both hand-offs.
    const auto spin = GENERATE(microseconds{0}, microseconds{20}, microseconds{1000});
    ThreadPool pool(12, spin);
    std::vector<long long> per_worker(pool.size());
    long long expected = 0;
    for (int job = 0; job < 3000; ++job) {
        const std::size_t n = static_cast<std::size_t>(job % 97) * 13 + 1;
        expected += static_cast<long long>(n * (n - 1) / 2);
        pool.for_each(n, 8, [&](unsigned worker, std::size_t begin, std::size_t end) noexcept {
            for (std::size_t i = begin; i < end; ++i) {
                per_worker[worker] += static_cast<long long>(i); // per-worker: no race
            }
        });
    }
    long long total = 0;
    for (const auto v : per_worker) {
        total += v;
    }
    CHECK(total == expected);
}

TEST_CASE("a pool shuts down cleanly, used or not, spinning or not") {
    for (int i = 0; i < 20; ++i) {
        const ThreadPool idle(6, microseconds{i % 2 == 0 ? 0 : 100});
        ThreadPool used(6, microseconds{i % 2 == 0 ? 100 : 0});
        used.for_each(100, 1, [](unsigned, std::size_t, std::size_t) noexcept {});
    }
    SUCCEED();
}
