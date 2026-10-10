#pragma once

#include "waferedge/hsms/protocol.hpp"

#include <asio/any_io_executor.hpp>
#include <asio/awaitable.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/steady_timer.hpp>

#include <expected>
#include <functional>
#include <span>
#include <vector>

// One HSMS connection over TCP, driven by Asio C++20 coroutines (ADR-0010). All protocol
// rules live in hsms::Protocol; this class only moves bytes between it and the socket and
// sleeps until its next deadline:
//
//   reader     socket -> Protocol::receive_buffer / on_received, then events to the handler
//   writer     Protocol::take_output -> socket; closes when the protocol wants it closed
//   deadline   sleeps until Protocol::next_deadline, then polls (T3, T6, T7, T8, linktest)
//
// The three run as one `&&` group per connection. When the connection ends (EOF, an error,
// the protocol closing it), finish() closes the socket and wakes the timers, and all three
// return. Single-threaded: use a Session only from its executor's thread (post to it from
// elsewhere).
namespace waferedge::hsms {

class Session {
public:
    // Called for every event, on the executor's thread. It may call send(), reply(),
    // separate() and stop(); a DataMessage's body is valid until the handler returns.
    using Handler = std::function<void(Session&, const Event&)>;

    Session(const asio::any_io_executor& executor, Config config, Handler handler);
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) = delete;
    Session& operator=(Session&&) = delete;
    ~Session() = default;

    // Active role: connects to `peer`, selects, serves the connection; after a close or a
    // failed attempt waits T5 and connects again, until stop().
    asio::awaitable<void> run_active(asio::ip::tcp::endpoint peer);
    // Passive role: accepts connections on `acceptor`, one at a time, until stop(). The
    // acceptor must stay open until this returns.
    asio::awaitable<void> run_passive(asio::ip::tcp::acceptor& acceptor);

    std::expected<std::uint32_t, SendError> send(std::uint8_t stream, std::uint8_t function,
                                                 bool reply_expected,
                                                 std::span<const std::uint8_t> body);
    std::expected<void, SendError> reply(const Header& primary, std::span<const std::uint8_t> body);
    std::expected<void, SendError> abort(const Header& primary);
    // Sends Separate.req and closes the connection; the active side reconnects after T5.
    void separate();
    // Separates if connected, and ends run_active / run_passive.
    void stop();

    [[nodiscard]] State state() const noexcept { return protocol_.state(); }
    [[nodiscard]] const Protocol& protocol() const noexcept { return protocol_; }

private:
    asio::awaitable<void> serve(asio::ip::tcp::socket& socket);
    asio::awaitable<void> reader();
    asio::awaitable<void> writer();
    asio::awaitable<void> deadlines();
    // Ends the current connection: closes the socket and wakes every coroutine.
    void finish();
    // Hands every pending event to the handler, then wakes the writer and the deadline
    // coroutine (new output, new deadlines).
    void drain();

    Protocol protocol_;
    Handler handler_;
    asio::steady_timer wake_writer_;
    asio::steady_timer deadline_;
    asio::steady_timer reconnect_;
    asio::ip::tcp::acceptor* acceptor_ = nullptr;
    asio::ip::tcp::socket* socket_ = nullptr; // the connection being served
    bool closing_ = false;
    std::vector<std::uint8_t> write_buffer_; // swapped with the protocol's output
    bool stopped_ = false;
};

} // namespace waferedge::hsms
