#include "waferedge/pipeline/analyzer.hpp"

#include <algorithm>

#if defined(__x86_64__)
#include <immintrin.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace waferedge::pipeline {

namespace {

void cpu_relax() noexcept {
#if defined(__x86_64__)
    _mm_pause();
#endif
}

} // namespace

bool pin_current_thread(int cpu) noexcept {
#if defined(__linux__)
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        return false;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<std::size_t>(cpu), &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

Analyzer::Analyzer(std::unique_ptr<Detector> detector, AnalyzerConfig config,
                   std::function<void()> on_verdict)
    : detector_(std::move(detector)), config_(config), on_verdict_(std::move(on_verdict)),
      wafers_(std::make_unique<SpscRing<WaferSlot, kSlots>>()),
      verdicts_(std::make_unique<SpscRing<Verdict, kSlots>>()) {
    config_.batch = std::clamp<std::size_t>(config_.batch, 1, kMaxBatch);
}

Analyzer::Analyzer(const RuleClassifier& classifier, AnalyzerConfig config,
                   std::function<void()> on_verdict)
    : Analyzer(std::make_unique<RulesDetector>(classifier), config, std::move(on_verdict)) {}

Analyzer::~Analyzer() {
    stop();
}

void Analyzer::start() {
    if (!thread_.joinable()) {
        stop_.store(false, std::memory_order_relaxed);
        thread_ = std::thread([this] { run(); });
    }
}

void Analyzer::stop() {
    if (thread_.joinable()) {
        stop_.store(true, std::memory_order_release);
        bell_.wake();
        thread_.join();
    }
}

bool Analyzer::has_work() noexcept {
    return wafers_->front() != nullptr;
}

// NOLINTNEXTLINE(bugprone-exception-escape): std::function::operator() only throws when empty
std::size_t Analyzer::drain(bool flush) noexcept {
    std::size_t done = 0;
    for (;;) {
        std::size_t n = 0;
        while (n < config_.batch && wafers_->peek(n) != nullptr) {
            ++n;
        }
        if (n == 0) {
            return done;
        }
        // A partial batch waits for more wafers until its oldest has waited `deadline`.
        if (n < config_.batch && !flush &&
            Clock::now() - wafers_->peek(0)->received < config_.deadline) {
            return done;
        }
        // Room for n verdicts? The network thread only frees slots meanwhile, so this is a
        // lower bound.
        n = std::min(n, kSlots - verdicts_->size_approx());
        if (n == 0) {
            return done; // the network thread hasn't taken the last kSlots verdicts yet
        }
        for (std::size_t i = 0; i < n; ++i) {
            views_[i] = wafers_->peek(i)->map(); // read in place, freed by pop(n) below
        }
        const bool ok = detector_->detect(std::span(views_).first(n), std::span(results_).first(n));
        const TimePoint analysed = Clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            const WaferSlot& wafer = *wafers_->peek(i);
            Verdict* verdict = verdicts_->begin_push(); // room checked above
            *verdict = {.seq = wafer.seq,
                        .lot = wafer.lot,
                        .wafer = wafer.wafer,
                        .pattern = ok ? results_[i].pattern : Pattern::unknown,
                        .rule = ok ? results_[i].rule : -1,
                        .second = ok ? results_[i].second : Pattern::unknown,
                        .received = wafer.received,
                        .analysed = analysed};
            verdicts_->commit_push();
        }
        wafers_->pop(n);
        analysed_.fetch_add(n, std::memory_order_relaxed);
        batches_.fetch_add(1, std::memory_order_relaxed);
        if (!ok) {
            failures_.fetch_add(1, std::memory_order_relaxed);
        }
        done += n;
        if (on_verdict_) {
            on_verdict_();
        }
    }
}

std::size_t Analyzer::poll() noexcept {
    return drain(true);
}

void Analyzer::run() noexcept {
    if (config_.cpu >= 0) {
        (void)pin_current_thread(config_.cpu);
    }
    while (!stop_.load(std::memory_order_acquire)) {
        if (drain(false) > 0) {
            continue;
        }
        if (has_work()) {
            // A partial batch before its deadline, or the verdict ring is full: keep checking
            // (deadlines are milliseconds, and a futex wait has no timeout).
            cpu_relax();
            continue;
        }
        // Empty: spin a while (a wafer arriving now starts in well under a microsecond)...
        if (config_.spin.count() > 0) {
            const auto until = Clock::now() + config_.spin;
            while (!has_work() && !stop_.load(std::memory_order_relaxed) && Clock::now() < until) {
                cpu_relax();
            }
            if (has_work()) {
                continue;
            }
        }
        // ...then sleep until commit_submit() or stop() rings.
        const std::uint32_t key = bell_.prepare();
        if (has_work() || stop_.load(std::memory_order_acquire)) {
            bell_.cancel();
            continue;
        }
        bell_.wait(key);
    }
}

} // namespace waferedge::pipeline
