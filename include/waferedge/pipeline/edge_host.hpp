#pragma once

#include "waferedge/gem/link.hpp"
#include "waferedge/hsms/session.hpp"
#include "waferedge/pipeline/edge_core.hpp"

#include <asio/awaitable.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/posix/stream_descriptor.hpp>
#include <asio/steady_timer.hpp>

#include <atomic>
#include <memory>
#include <vector>

// The edge host (ADR-0014): active HSMS host, GEM host, analytics workers, decision rule.
//
//   network thread (the executor's)                        worker threads
//     hsms::Session --events--> EdgeCore::on_hsms --wafer slots--> Analyzer: classify
//     eventfd readable --> EdgeCore::drain <--verdicts-- Analyzer: on_verdict writes eventfd
//       --> decision rule --> S2F41 HOLD
//
// Three coroutines on the network thread: the session, a timer for GEM's WAIT DELAY, and a
// reader of the eventfd the workers signal. A burst of verdicts costs one eventfd write: a
// worker writes only when the `pending` flag was clear, and the network thread clears it
// (an exchange, so it synchronizes with the worker's push) before draining every ring.
//
// Linux only (eventfd). Use from the executor's thread.
namespace waferedge::pipeline {

struct EdgeHostConfig {
    hsms::Config hsms{}; // role is forced to active
    EdgeConfig edge{};
    int workers = 1;              // analytics threads
    AnalyzerConfig analyzer{};    // spin; cpu is ignored here (see worker_cpus)
    std::vector<int> worker_cpus; // worker i is pinned to worker_cpus[i] if given
    bool stop_when_tool_leaves = true;
};

class EdgeHost {
public:
    EdgeHost(const asio::any_io_executor& executor, const RuleClassifier& classifier,
             EdgeHostConfig config, Logger log = {});
    EdgeHost(const EdgeHost&) = delete;
    EdgeHost& operator=(const EdgeHost&) = delete;
    EdgeHost(EdgeHost&&) = delete;
    EdgeHost& operator=(EdgeHost&&) = delete;
    ~EdgeHost();

    // Connects to the tool (reconnecting after T5) and runs until stop(), or until the tool
    // leaves once communication was established (stop_when_tool_leaves). Every submitted
    // wafer's verdict is taken before it returns; then the workers stop.
    asio::awaitable<void> run(asio::ip::tcp::endpoint tool);
    void stop();

    [[nodiscard]] const EdgeCore& core() const noexcept { return core_; }
    [[nodiscard]] std::span<const std::unique_ptr<Analyzer>> analyzers() const noexcept {
        return analyzers_;
    }

private:
    void on_hsms(const hsms::Event& event);
    void signal() noexcept; // any thread
    asio::awaitable<void> timers();
    asio::awaitable<void> verdicts();
    void stop_workers();

    EdgeHostConfig config_;
    // Declared before the analyzers: destroyed after their threads have stopped.
    int event_fd_ = -1;
    asio::posix::stream_descriptor events_;
    std::atomic<bool> pending_{false};
    std::vector<std::unique_ptr<Analyzer>> analyzers_;
    EdgeCore core_;
    hsms::Session session_;
    gem::SessionLink link_;
    asio::steady_timer wake_;
    bool stopping_ = false;
};

} // namespace waferedge::pipeline
