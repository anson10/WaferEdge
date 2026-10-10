#include "waferedge/hsms/session.hpp"

#include <asio/as_tuple.hpp>
#include <asio/bind_cancellation_slot.hpp>
#include <asio/buffer.hpp>
#include <asio/experimental/awaitable_operators.hpp>
#include <asio/this_coro.hpp>
#include <asio/use_awaitable.hpp>
#include <asio/write.hpp>

namespace waferedge::hsms {

namespace {

// Every operation completes with its error as a value (as_tuple: a closed socket is an
// ordinary end of a connection) and with no cancellation slot: Asio would otherwise keep a
// cancellation handler per operation, and when consecutive operations need handlers of
// different sizes (a socket write, then a timer wait: 40 and 32 bytes with libc++) it frees
// and reallocates that memory every time. The connection ends explicitly instead (finish()).
auto token() {
    return asio::bind_cancellation_slot(asio::cancellation_slot(),
                                        asio::as_tuple(asio::use_awaitable));
}

TimePoint now() {
    return Clock::now();
}

} // namespace

Session::Session(const asio::any_io_executor& executor, Config config, Handler handler)
    : protocol_(config), handler_(std::move(handler)), wake_writer_(executor), deadline_(executor),
      reconnect_(executor) {}

void Session::drain() {
    const TimePoint t = now();
    while (auto event = protocol_.poll(t)) {
        handler_(*this, *event);
    }
    wake_writer_.cancel();
    deadline_.cancel();
}

void Session::finish() {
    // Closing the socket aborts the pending read and write; the timers are woken. Each
    // coroutine sees closing_ when it resumes and returns.
    closing_ = true;
    asio::error_code ignored;
    if (socket_ != nullptr) {
        socket_->shutdown(asio::ip::tcp::socket::shutdown_both, ignored);
        socket_->close(ignored);
    }
    wake_writer_.cancel();
    deadline_.cancel();
}

asio::awaitable<void> Session::reader() {
    while (!closing_) {
        const auto buffer = protocol_.receive_buffer();
        auto [ec, n] =
            co_await socket_->async_read_some(asio::buffer(buffer.data(), buffer.size()), token());
        if (closing_) {
            break;
        }
        if (ec) {
            protocol_.on_disconnected(now()); // EOF or reset: the peer is gone
            drain();
            finish();
            break;
        }
        protocol_.on_received(n, now());
        drain();
    }
}

asio::awaitable<void> Session::writer() {
    while (!closing_) {
        if (protocol_.has_output()) {
            // Swap, then write from our buffer: the handler may append to the protocol's
            // buffer while this write is in flight.
            protocol_.take_output(write_buffer_);
            auto [ec, n] =
                co_await asio::async_write(*socket_, asio::buffer(write_buffer_), token());
            if (closing_) {
                break;
            }
            if (ec) {
                protocol_.on_disconnected(now());
                drain();
                finish();
                break;
            }
            continue;
        }
        if (protocol_.wants_close()) {
            finish(); // everything is written, a Separate.req last
            break;
        }
        wake_writer_.expires_at(asio::steady_timer::time_point::max());
        co_await wake_writer_.async_wait(token()); // drain() and send() cancel it to wake us
    }
}

asio::awaitable<void> Session::deadlines() {
    while (!closing_) {
        const auto next = protocol_.next_deadline();
        deadline_.expires_at(next ? *next : asio::steady_timer::time_point::max());
        co_await deadline_.async_wait(token());
        if (closing_) {
            break;
        }
        // Expired, or cancelled because the deadlines changed: poll either way.
        drain();
    }
}

// The socket is owned by run_active / run_passive, which co_await this call.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-reference-coroutine-parameters)
asio::awaitable<void> Session::serve(asio::ip::tcp::socket& socket) {
    using namespace asio::experimental::awaitable_operators;
    asio::error_code ignored;
    socket.set_option(asio::ip::tcp::no_delay(true), ignored); // small messages, now
    socket_ = &socket;
    closing_ = false;
    protocol_.on_connected(now());
    drain();
    co_await (reader() && writer() && deadlines()); // all three, until finish()
    socket_ = nullptr;
    protocol_.on_disconnected(now()); // no-op if the protocol closed it
    drain();
}

asio::awaitable<void> Session::run_active(asio::ip::tcp::endpoint peer) {
    const auto executor = co_await asio::this_coro::executor;
    while (!stopped_) {
        if (protocol_.connect_allowed_at() > now()) { // T5
            reconnect_.expires_at(protocol_.connect_allowed_at());
            co_await reconnect_.async_wait(token()); // stop() cancels it
            if (stopped_) {
                break;
            }
        }
        asio::ip::tcp::socket socket(executor);
        auto [ec] = co_await socket.async_connect(peer, token());
        if (stopped_) {
            break;
        }
        if (ec) {
            protocol_.on_connect_failed(now());
            continue;
        }
        co_await serve(socket);
    }
}

// The caller owns the acceptor and keeps it open until this returns.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-reference-coroutine-parameters)
asio::awaitable<void> Session::run_passive(asio::ip::tcp::acceptor& acceptor) {
    acceptor_ = &acceptor;
    while (!stopped_) {
        auto [ec, socket] = co_await acceptor.async_accept(token()); // stop() cancels it
        if (stopped_) {
            break;
        }
        if (ec) {
            continue;
        }
        co_await serve(socket);
    }
    acceptor_ = nullptr;
}

std::expected<std::uint32_t, SendError> Session::send(std::uint8_t stream, std::uint8_t function,
                                                      bool reply_expected,
                                                      std::span<const std::uint8_t> body) {
    auto system = protocol_.send(stream, function, reply_expected, body, now());
    wake_writer_.cancel();
    deadline_.cancel(); // a new T3 may be the earliest deadline
    return system;
}

std::expected<void, SendError> Session::reply(const Header& primary,
                                              std::span<const std::uint8_t> body) {
    auto result = protocol_.reply(primary, body);
    wake_writer_.cancel();
    return result;
}

std::expected<void, SendError> Session::abort(const Header& primary) {
    auto result = protocol_.abort(primary);
    wake_writer_.cancel();
    return result;
}

void Session::separate() {
    protocol_.separate(now());
    drain();
}

void Session::stop() {
    stopped_ = true;
    reconnect_.cancel();
    if (acceptor_ != nullptr) {
        asio::error_code ignored;
        acceptor_->cancel(ignored);
    }
    if (protocol_.state() != State::not_connected) {
        separate(); // the writer sends Separate.req, then finishes the connection
    }
}

} // namespace waferedge::hsms
