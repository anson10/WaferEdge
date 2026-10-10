#include "waferedge/pipeline/analyzer.hpp"

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

Analyzer::Analyzer(const RuleClassifier& classifier, AnalyzerConfig config,
                   std::function<void()> on_verdict)
    : classifier_(classifier), config_(config), on_verdict_(std::move(on_verdict)),
      wafers_(std::make_unique<SpscRing<WaferSlot, kSlots>>()),
      verdicts_(std::make_unique<SpscRing<Verdict, kSlots>>()) {}

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

Verdict Analyzer::analyse(const WaferSlot& wafer) noexcept {
    const Decision d = classifier_.classify(extractor_.extract(wafer.map()));
    return {.seq = wafer.seq,
            .lot = wafer.lot,
            .wafer = wafer.wafer,
            .pattern = d.pattern,
            .rule = d.rule,
            .received = wafer.received,
            .analysed = Clock::now()};
}

bool Analyzer::has_work() noexcept {
    return wafers_->front() != nullptr;
}

// NOLINTNEXTLINE(bugprone-exception-escape): std::function::operator() only throws when empty
std::size_t Analyzer::drain() noexcept {
    std::size_t n = 0;
    while (const WaferSlot* wafer = wafers_->front()) {
        Verdict* verdict = verdicts_->begin_push();
        if (verdict == nullptr) {
            break; // the network thread hasn't taken the last kSlots verdicts yet
        }
        *verdict = analyse(*wafer); // the slot is read in place, then freed
        wafers_->pop();
        verdicts_->commit_push();
        analysed_.fetch_add(1, std::memory_order_relaxed);
        ++n;
        if (on_verdict_) {
            on_verdict_();
        }
    }
    return n;
}

std::size_t Analyzer::poll() noexcept {
    return drain();
}

void Analyzer::run() noexcept {
    if (config_.cpu >= 0) {
        (void)pin_current_thread(config_.cpu);
    }
    while (!stop_.load(std::memory_order_acquire)) {
        if (drain() > 0) {
            continue;
        }
        if (has_work()) {
            std::this_thread::yield(); // wafers waiting, but the verdict ring is full
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
