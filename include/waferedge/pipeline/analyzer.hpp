#pragma once

#include "waferedge/classifier.hpp"
#include "waferedge/pipeline/doorbell.hpp"
#include "waferedge/pipeline/slots.hpp"
#include "waferedge/pipeline/spsc_ring.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <thread>

// One analytics worker of the edge host (ADR-0014): wafers in through one SPSC ring, verdicts
// out through another, a thread in between running the rule classifier.
//
//   network thread --begin_submit / commit_submit--> [wafers] --> worker thread
//                                                                   extract signals, classify
//   network thread <--front_verdict / pop_verdict--- [verdicts] <--+  then on_verdict()
//
// The network thread is the only producer of `wafers` and the only consumer of `verdicts`;
// the worker the other way round. When `wafers` is empty the worker spins for `spin`, then
// sleeps on a Doorbell that commit_submit rings. After pushing a verdict it calls on_verdict,
// which must be cheap and thread-safe (the edge host writes an eventfd).
namespace waferedge::pipeline {

struct AnalyzerConfig {
    std::chrono::microseconds spin{50}; // busy-wait before sleeping; 0: sleep at once
    int cpu = -1;                       // pin the worker thread to this CPU; -1: don't
};

class Analyzer {
public:
    static constexpr std::size_t kSlots = 256; // per ring: 1 MB of wafer slots

    // `classifier` must outlive the analyzer; it is shared read-only between workers.
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

    // --- Without a thread (tests): analyses every submitted wafer on the caller's thread.
    std::size_t poll() noexcept;

    // Classifies one map: what the worker does per wafer.
    [[nodiscard]] Verdict analyse(const WaferSlot& wafer) noexcept;

    // Wafers analysed (read from any thread).
    [[nodiscard]] std::uint64_t analysed() const noexcept {
        return analysed_.load(std::memory_order_relaxed);
    }

private:
    void run() noexcept;
    // Analyses queued wafers while there is room for their verdicts; returns how many.
    std::size_t drain() noexcept;
    [[nodiscard]] bool has_work() noexcept;

    const RuleClassifier& classifier_;
    AnalyzerConfig config_;
    std::function<void()> on_verdict_;
    SignalExtractor extractor_;
    std::unique_ptr<SpscRing<WaferSlot, kSlots>> wafers_;
    std::unique_ptr<SpscRing<Verdict, kSlots>> verdicts_;
    Doorbell bell_;
    std::atomic<bool> stop_{false};
    std::atomic<std::uint64_t> analysed_{0};
    std::thread thread_;
};

// Pins the calling thread to one CPU. False if the OS refused (or isn't Linux).
bool pin_current_thread(int cpu) noexcept;

} // namespace waferedge::pipeline
