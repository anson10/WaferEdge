#include "waferedge/emulator/tool.hpp"

#include <asio/as_tuple.hpp>
#include <asio/bind_cancellation_slot.hpp>
#include <asio/experimental/awaitable_operators.hpp>
#include <asio/use_awaitable.hpp>

#include <format>

namespace waferedge::emulator {

namespace {

// As in hsms::Session: errors as values, no cancellation slot (ADR-0010).
auto token() {
    return asio::bind_cancellation_slot(asio::cancellation_slot(),
                                        asio::as_tuple(asio::use_awaitable));
}

gem::Hcack hcack(LotResult r) {
    switch (r) {
    case LotResult::done:
        return gem::Hcack::done;
    case LotResult::already:
        return gem::Hcack::already_in_condition;
    case LotResult::unknown:
        return gem::Hcack::no_such_object;
    }
    return gem::Hcack::no_such_object;
}

hsms::Config passive(hsms::Config c) {
    c.role = hsms::Role::passive;
    return c;
}

} // namespace

Tool::Tool(const asio::any_io_executor& executor, std::span<const WaferRecord> records,
           ToolConfig config, Logger log)
    : config_(std::move(config)), replay_(records, config_.replay), equipment_(config_.gem),
      session_(executor, passive(config_.hsms),
               [this](hsms::Session&, const hsms::Event& e) { on_hsms(e); }),
      link_(session_), wake_(executor), log_(std::move(log)) {}

void Tool::log(std::string_view line) const {
    if (log_) {
        log_(line);
    }
}

void Tool::on_hsms(const hsms::Event& event) {
    equipment_.on_hsms(event, link_, hsms::Clock::now());
    while (auto g = equipment_.poll()) {
        on_gem(*g);
    }
    wake_.cancel(); // the pacer recomputes its next wake-up
}

void Tool::on_gem(const gem::Event& event) {
    const TimePoint now = hsms::Clock::now();
    if (std::holds_alternative<gem::Communicating>(event)) {
        log("communicating");
        if (config_.auto_online) {
            equipment_.go_online(link_);
        }
    } else if (std::holds_alternative<gem::NotCommunicating>(event)) {
        log("not communicating");
        replay_.pause();
    } else if (const auto* c = std::get_if<gem::ControlStateChanged>(&event)) {
        log(std::format("control: {}", gem::control_state_name(c->state)));
        if (equipment_.online()) {
            if (!replay_.running()) {
                replay_.start(now);
            }
        } else {
            replay_.pause();
        }
    } else if (const auto* cmd = std::get_if<gem::LotCommandReceived>(&event)) {
        const bool hold = cmd->command.action == gem::LotAction::hold;
        const LotResult result =
            hold ? replay_.hold(cmd->command.lot, now) : replay_.release(cmd->command.lot);
        (void)equipment_.answer(link_, cmd->primary, hcack(result));
        log(std::format("{} {}: HCACK {}", hold ? "HOLD" : "RELEASE", cmd->command.lot,
                        static_cast<int>(hcack(result))));
    } else if (const auto* r = std::get_if<gem::ReplyReceived>(&event)) {
        if (r->stream == 6 && r->function == 12) {
            ++acknowledged_;
        }
    } else if (std::holds_alternative<gem::TransactionFailed>(event)) {
        ++failed_; // settled, but never acknowledged
        log("a transaction failed (T3, abort or reject)");
    } else if (std::holds_alternative<gem::MalformedMessage>(event)) {
        log("malformed message from the host (S9F7 sent)");
    }
    if (config_.stop_when_done && replay_.finished() && !blocked_ &&
        acknowledged_ + failed_ >= replay_.stats().sent && !stopped_) {
        log("every wafer reported and answered");
        stop();
    }
}

void Tool::send_due() {
    if (!equipment_.communicating() || !equipment_.online()) {
        return;
    }
    const TimePoint now = hsms::Clock::now();
    window_blocked_ = false;
    for (;;) {
        std::optional<Replay::Due> due = blocked_ ? blocked_ : replay_.take(now);
        if (!due) {
            return;
        }
        auto sent = equipment_.report_wafer(
            link_, due->lot, static_cast<std::uint32_t>(due->record->wafer_id), due->record->map);
        if (!sent) {
            blocked_ = due;
            if (sent.error() == gem::GemError::busy) {
                // Backpressure: the host has kMaxOpenTransactions reports to answer. Retry
                // when a reply frees a slot; the wafer goes out late, and its lateness shows.
                ++window_full_;
                window_blocked_ = true;
                return;
            }
            log("report not sent: link down; schedule paused");
            replay_.pause();
            return;
        }
        blocked_.reset();
        replay_.mark_sent(due->index, hsms::Clock::now());
    }
}

asio::awaitable<void> Tool::pacer() {
    while (!stopped_) {
        // The earliest of: the next wafer's slot, GEM's WAIT DELAY end. Events wake it too.
        // While the window is full only a reply (an event) can help: no timer.
        std::optional<TimePoint> next = window_blocked_ ? std::nullopt : replay_.next_due();
        if (const auto gem_deadline = equipment_.next_deadline();
            gem_deadline && (!next || *gem_deadline < *next)) {
            next = gem_deadline;
        }
        wake_.expires_at(next ? *next : asio::steady_timer::time_point::max());
        co_await wake_.async_wait(token());
        if (stopped_) {
            break;
        }
        equipment_.tick(link_, hsms::Clock::now());
        while (auto g = equipment_.poll()) {
            on_gem(*g);
        }
        send_due();
    }
}

asio::awaitable<void> Tool::run(asio::ip::tcp::acceptor& acceptor) {
    using namespace asio::experimental::awaitable_operators;
    co_await (session_.run_passive(acceptor) && pacer());
}

void Tool::stop() {
    stopped_ = true;
    session_.stop(); // Separate.req if connected, and no new connections
    wake_.cancel();
}

} // namespace waferedge::emulator
