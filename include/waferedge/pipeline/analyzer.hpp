#pragma once

#include "waferedge/classifier.hpp"
#include "waferedge/pipeline/detector.hpp"
#include "waferedge/pipeline/doorbell.hpp"
#include "waferedge/pipeline/slots.hpp"
#include "waferedge/pipeline/spsc_ring.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>

// One analytics worker of the edge host (ADR-0014): wafers in through one SPSC ring, verdicts
// out through another, a thread in between running a Detector on batches (ADR-0016).
//
//   network thread --begin_submit / commit_submit--> [wafers] --> worker thread
//                                                                   detector on a batch
//   network thread <--front_verdict / pop_verdict--- [verdicts] <--+  then on_verdict()
//
// The network thread is the only producer of `wafers` and the only consumer of `verdicts`;
// the worker the other way round. A batch starts when `batch` wafers wait, or when the oldest
// has waited `deadline` since it arrived: batch 1 (the default) analyses each wafer at once,
// a GPU detector wants more. The batch is read in place (SpscRing::peek) and freed at once
// (pop(n)). While a partial batch waits for its deadline the worker spins; with nothing to do
// it spins for `spin`, then sleeps on a Doorbell that commit_submit rings. After pushing a
// batch's verdicts it calls on_verdict, which must be cheap and thread-safe (the edge host
// writes an eventfd).
namespace waferedge::pipeline {

struct AnalyzerConfig {
    std::chrono::microseconds spin{50};    // busy-wait before sleeping; 0: sleep at once
    int cpu = -1;                          // pin the worker thread to this CPU; -1: don't
    std::size_t batch = 1;                 // wafers per detector call, 1..kMaxBatch
    std::chrono::microseconds deadline{0}; // a partial batch starts when its oldest waited this
};

class Analyzer {
public:
    static constexpr std::size_t kSlots = kMaxBatch; // per ring: 1 MB of wafer slots

    // The worker runs `detector`.
    Analyzer(std::unique_ptr<Detector> detector, AnalyzerConfig config,
             std::function<void()> on_verdict = {});
    // The rule classifier (RulesDetector); `classifier` must outlive the analyzer.
    Analyzer(const RuleClassifier& classifier, AnalyzerConfig config,
             std::function<void()> on_verdict = {});
    Analyzer(const Analyzer&) = delete;
    Analyzer& operator=(const Analyzer&) = delete;
    Analyzer(Analyzer&&) = delete;
    Analyzer& operator=(Analyzer&&) = delete;
    ~Analyzer(); // stops the thread

    // Starts the worker thread. Without start(), poll() does the work on the caller's thread.
    void start();
    // Ends the thread after the wafer in hand (queued wafers stay unanalysed). Idempotent.
    void stop();

    // --- Network thread -------------------------------------------------------------------
    // A free wafer slot to fill, or nullptr if the worker is kSlots wafers behind.
    [[nodiscard]] WaferSlot* begin_submit() noexcept { return wafers_->begin_push(); }
    void commit_submit() noexcept {
        wafers_->commit_push();
        bell_.ring();
    }
    [[nodiscard]] const Verdict* front_verdict() noexcept { return verdicts_->front(); }
    void pop_verdict() noexcept { verdicts_->pop(); }

    // --- Without a thread (tests): analyses every submitted wafer on the caller's thread,
    // partial batches too (no deadline).
    std::size_t poll() noexcept;

    // Wafers analysed, batches run and failed detector calls (read from any thread).
    [[nodiscard]] std::uint64_t analysed() const noexcept {
        return analysed_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t batches() const noexcept {
        return batches_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t failures() const noexcept {
        return failures_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] const Detector& detector() const noexcept { return *detector_; }

private:
    void run() noexcept;
    // Runs every batch that is ready (full, past its deadline, or any with `flush`) while there
    // is room for its verdicts; returns how many wafers it analysed.
    std::size_t drain(bool flush) noexcept;
    [[nodiscard]] bool has_work() noexcept;

    std::unique_ptr<Detector> detector_;
    AnalyzerConfig config_;
    std::function<void()> on_verdict_;
    // The batch being analysed: views of its slots, and the detector's answers. Fixed: no
    // allocation per batch.
    std::array<WaferMapView, kMaxBatch> views_{};
    std::array<Detection, kMaxBatch> results_{};
    std::unique_ptr<SpscRing<WaferSlot, kSlots>> wafers_;
    std::unique_ptr<SpscRing<Verdict, kSlots>> verdicts_;
    Doorbell bell_;
    std::atomic<bool> stop_{false};
    std::atomic<std::uint64_t> analysed_{0};
    std::atomic<std::uint64_t> batches_{0};
    std::atomic<std::uint64_t> failures_{0};
    std::thread thread_;
};

// Pins the calling thread to one CPU. False if the OS refused (or isn't Linux).
bool pin_current_thread(int cpu) noexcept;

} // namespace waferedge::pipeline
