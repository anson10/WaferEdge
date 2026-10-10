#include "waferedge/pipeline/edge_host.hpp"

#include <asio/as_tuple.hpp>
#include <asio/bind_cancellation_slot.hpp>
#include <asio/buffer.hpp>
#include <asio/experimental/awaitable_operators.hpp>
#include <asio/use_awaitable.hpp>

#include <algorithm>
#include <cerrno>
#include <sys/eventfd.h>
#include <system_error>
#include <unistd.h>

namespace waferedge::pipeline {

namespace {

// As in hsms::Session: errors as values, no cancellation slot (ADR-0010).
auto token() {
    return asio::bind_cancellation_slot(asio::cancellation_slot(),
                                        asio::as_tuple(asio::use_awaitable));
}

hsms::Config active(hsms::Config c) {
    c.role = hsms::Role::active;
    return c;
}

int make_eventfd() {
    const int fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category(), "eventfd");
    }
    return fd;
}

// Adds 1 to the eventfd's counter, making it readable. The write can only fail if the counter
// would pass 2^64 - 2 (EAGAIN), and a readable eventfd is all the reader needs anyway.
void bump(int fd) noexcept {
    const std::uint64_t one = 1;
    if (::write(fd, &one, sizeof one) < 0) {
        return;
    }
}

std::vector<Analyzer*> raw(const std::vector<std::unique_ptr<Analyzer>>& analyzers) {
    std::vector<Analyzer*> out;
    out.reserve(analyzers.size());
    for (const auto& a : analyzers) {
        out.push_back(a.get());
    }
    return out;
}

} // namespace

EdgeHost::EdgeHost(const asio::any_io_executor& executor, const RuleClassifier& classifier,
                   EdgeHostConfig config, Logger log)
    : config_(std::move(config)), event_fd_(make_eventfd()), events_(executor, event_fd_),
      analyzers_([&] {
          std::vector<std::unique_ptr<Analyzer>> workers;
          workers.reserve(static_cast<std::size_t>(std::max(config_.workers, 1)));
          for (int i = 0; i < std::max(config_.workers, 1); ++i) {
              AnalyzerConfig a = config_.analyzer;
              const auto at = static_cast<std::size_t>(i);
              a.cpu = at < config_.worker_cpus.size() ? config_.worker_cpus[at] : -1;
              workers.push_back(std::make_unique<Analyzer>(classifier, a, [this] { signal(); }));
          }
          return workers;
      }()),
      core_(raw(analyzers_), config_.edge, std::move(log)),
      session_(executor, active(config_.hsms),
               [this](hsms::Session&, const hsms::Event& e) { on_hsms(e); }),
      link_(session_), wake_(executor) {}

EdgeHost::~EdgeHost() {
    stop_workers();
}

void EdgeHost::stop_workers() {
    for (auto& a : analyzers_) {
        a->stop();
    }
}

// Worker threads: one eventfd write per burst of verdicts.
void EdgeHost::signal() noexcept {
    if (!pending_.exchange(true, std::memory_order_acq_rel)) {
        bump(event_fd_);
    }
}

void EdgeHost::on_hsms(const hsms::Event& event) {
    const auto deadline = core_.next_deadline();
    core_.on_hsms(event, link_, hsms::Clock::now());
    if (config_.stop_when_tool_leaves && core_.tool_left() && !stopping_) {
        stop();
    } else if (core_.next_deadline() != deadline) {
        wake_.cancel(); // GEM's timer changed: timers() re-arms
    }
}

asio::awaitable<void> EdgeHost::timers() {
    while (!stopping_) {
        const auto deadline = core_.next_deadline();
        wake_.expires_at(deadline ? *deadline : asio::steady_timer::time_point::max());
        co_await wake_.async_wait(token());
        if (!stopping_) {
            core_.tick(link_, hsms::Clock::now());
        }
    }
}

asio::awaitable<void> EdgeHost::verdicts() {
    std::uint64_t count = 0;
    for (;;) {
        const auto [ec, n] =
            co_await events_.async_read_some(asio::buffer(&count, sizeof count), token());
        if (ec) {
            break;
        }
        // Clear the flag first, then drain: a verdict pushed after the drain looked finds the
        // flag clear and writes the eventfd again; one pushed before it is seen by the drain
        // (the exchange reads the worker's exchange, which follows its push).
        pending_.exchange(false, std::memory_order_acq_rel);
        core_.drain(link_);
        if (stopping_ && core_.idle()) {
            break;
        }
    }
}

asio::awaitable<void> EdgeHost::run(asio::ip::tcp::endpoint tool) {
    using namespace asio::experimental::awaitable_operators;
    for (auto& a : analyzers_) {
        a->start();
    }
    co_await (session_.run_active(tool) && timers() && verdicts());
    stop_workers();
}

void EdgeHost::stop() {
    stopping_ = true;
    session_.stop();
    wake_.cancel();
    bump(event_fd_); // wakes verdicts() even if every verdict is in already
}

} // namespace waferedge::pipeline
