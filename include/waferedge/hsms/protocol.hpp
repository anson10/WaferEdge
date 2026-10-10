#pragma once

#include "waferedge/hsms/header.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

// The HSMS protocol as a state machine without I/O (ADR-0010). It never touches a socket or
// reads a clock: the caller hands it received bytes and the current time, and takes back the
// bytes to send, events, and the next time it must be polled. hsms::Session drives it over
// TCP with Asio; tests drive it with a fake clock.
//
//   NOT CONNECTED --on_connected--> NOT SELECTED --Select--> SELECTED
//        ^                              |   ^                   |
//        +---- close (Separate, timeout, transport) --+  +-- Deselect
//
// Single session (HSMS-SS): one connection, one device id.
namespace waferedge::hsms {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration = Clock::duration;

enum class Role : std::uint8_t {
    active,  // connects and sends Select.req: the host (edge host)
    passive, // listens and answers Select.req: the equipment (tool emulator)
};
enum class State : std::uint8_t { not_connected, not_selected, selected };

[[nodiscard]] std::string_view state_name(State s) noexcept;

struct Config {
    Role role = Role::passive;
    std::uint16_t session_id = 0; // device id of the data messages this side sends
    // E37's timers; the defaults are the usual factory settings.
    Duration t3 = std::chrono::seconds(45); // reply timeout of a data transaction
    Duration t5 = std::chrono::seconds(10); // active: wait between connection attempts
    Duration t6 = std::chrono::seconds(5);  // control transaction timeout
    Duration t7 = std::chrono::seconds(10); // connected but not selected
    Duration t8 = std::chrono::seconds(5);  // gap between bytes inside one message
    Duration linktest = Duration::zero();   // Linktest.req period while selected; 0: none
    // Longest message accepted or sent, header included: a peer announcing more is
    // disconnected before anything is allocated for it.
    std::size_t max_message = kHeaderSize + std::size_t{16} * 1024 * 1024;
};

enum class CloseReason : std::uint8_t {
    local,             // separate() or close() from this side
    peer_separate,     // the peer sent Separate.req
    transport,         // the TCP connection ended or failed (on_disconnected)
    t6_control,        // no Select / Deselect / Linktest response within T6
    t7_not_selected,   // connected for T7 without being selected
    t8_intercharacter, // a message stalled half-received for T8
    select_refused,    // Select.rsp with a non-zero status, or a Reject of our Select.req
    bad_length,        // a length field below the 10-byte header
    too_long,          // a length field above Config::max_message
};
[[nodiscard]] std::string_view close_reason_name(CloseReason r) noexcept;

// What poll() reports. Spans point into the protocol's receive buffer and stay valid until
// the next receive_buffer() call.
struct Selected {};
struct Deselected {};
struct DataMessage {
    enum class Kind : std::uint8_t {
        primary,         // odd function: a request or an unsolicited report
        reply,           // even function closing one of our open transactions
        unmatched_reply, // even function with no open transaction (late, after T3)
    };
    Header header;
    std::span<const std::uint8_t> body; // the SECS-II item: secs::decode(body)
    Kind kind = Kind::primary;
};
struct ReplyTimeout {
    Header request; // T3 expired for this primary; the transaction is closed
};
struct Rejected {
    Header reject; // the peer's Reject.req: byte2 the rejected SType, byte3 the reason
    [[nodiscard]] RejectReason reason() const noexcept {
        return static_cast<RejectReason>(reject.byte3);
    }
};
struct Closed {
    CloseReason reason;
};
using Event = std::variant<Selected, Deselected, DataMessage, ReplyTimeout, Rejected, Closed>;

enum class SendError : std::uint8_t {
    not_selected,    // data messages need a selected connection
    too_many_open,   // kMaxOpenTransactions replies already awaited
    too_long,        // above Config::max_message
    control_pending, // one control transaction at a time
    not_connected,
};
[[nodiscard]] std::string_view send_error_name(SendError e) noexcept;

struct Stats {
    std::uint64_t data_received = 0;
    std::uint64_t data_sent = 0;
    std::uint64_t linktests_sent = 0;
    std::uint64_t rejects_sent = 0;
    std::uint64_t rejects_received = 0;
    std::uint64_t reply_timeouts = 0;
    std::uint64_t connections = 0;
};

// Primaries awaiting a reply at once; a fixed table, so sending never allocates for it.
inline constexpr std::size_t kMaxOpenTransactions = 64;

class Protocol {
public:
    explicit Protocol(Config config);

    [[nodiscard]] State state() const noexcept { return state_; }
    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

    // --- Transport ---------------------------------------------------------------------
    // A TCP connection is up: the active side sends Select.req (T6), both arm T7.
    void on_connected(TimePoint now);
    // The TCP connection ended without a Separate: reports Closed{transport} once.
    void on_disconnected(TimePoint now);
    // Active side: a connection attempt failed; the next waits T5 like after a close.
    void on_connect_failed(TimePoint now);
    // Earliest time the active side may connect again (T5 after the last close or failure).
    [[nodiscard]] TimePoint connect_allowed_at() const noexcept { return reconnect_at_; }

    // Free space to read at least `min_bytes` into. May move unparsed bytes to the front of
    // the buffer, which ends the life of spans handed out by earlier events.
    [[nodiscard]] std::span<std::uint8_t> receive_buffer(std::size_t min_bytes = 4096);
    // `n` bytes were read into the span from receive_buffer().
    void on_received(std::size_t n, TimePoint now);

    // Bytes waiting to be written to the socket.
    [[nodiscard]] bool has_output() const noexcept { return !out_.empty(); }
    // Hands them over by swapping vectors: `buffer` (whose contents are dropped) becomes the
    // protocol's next output buffer. No copy and, once both have grown, no allocation; and a
    // write in flight from `buffer` is safe while send() appends to the protocol's buffer.
    void take_output(std::vector<std::uint8_t>& buffer) noexcept;
    // Close the socket once output() is written (Separate.req is sent first).
    [[nodiscard]] bool wants_close() const noexcept { return wants_close_; }

    // --- Time and events ---------------------------------------------------------------
    // The next event, if any, after checking the timers against `now`. Call until empty
    // after on_received, on timer wake-ups, and after send().
    [[nodiscard]] std::optional<Event> poll(TimePoint now);
    // When poll() must next be called even if no bytes arrive (nullopt: no timer running).
    [[nodiscard]] std::optional<TimePoint> next_deadline() const noexcept;

    // --- Messages ----------------------------------------------------------------------
    // A primary data message (body: an encoded SECS-II item, or empty). With reply_expected
    // the transaction stays open until the reply arrives or T3 expires. Returns the system
    // bytes. The body is copied into the output buffer.
    std::expected<std::uint32_t, SendError> send(std::uint8_t stream, std::uint8_t function,
                                                 bool reply_expected,
                                                 std::span<const std::uint8_t> body, TimePoint now);
    // The reply to `primary`: function + 1, the same system bytes.
    std::expected<void, SendError> reply(const Header& primary, std::span<const std::uint8_t> body);
    // SxF0, "abort transaction": the answer to a primary this side won't serve (GEM: not
    // communicating, off-line). Header only, the same stream and system bytes.
    std::expected<void, SendError> abort(const Header& primary);

    // Control transactions.
    std::expected<void, SendError> linktest(TimePoint now);
    std::expected<void, SendError> deselect(TimePoint now);
    // Sends Separate.req and closes: poll() reports Closed{local}.
    void separate(TimePoint now);
    // Closes without a Separate (shutdown, or the app gave up on the peer).
    void close(TimePoint now);

private:
    struct Transaction {
        Header request;
        TimePoint deadline;
        bool open = false;
    };
    enum class Control : std::uint8_t { none, select, deselect, linktest };

    std::optional<Event> check_timers(TimePoint now);
    std::optional<Event> take_closed() noexcept;
    // Parses one complete frame into `event` (if it makes one); false if none is complete.
    bool next_frame(TimePoint now, std::optional<Event>& event);
    std::optional<Event> handle(const Header& h, std::span<const std::uint8_t> body, TimePoint now);
    std::optional<Event> handle_control(const Header& h, TimePoint now);
    void close_now(CloseReason reason, TimePoint now);
    [[nodiscard]] bool partial_frame() const noexcept;
    void become_selected(TimePoint now);
    void become_not_selected(TimePoint now);
    void start_control(Control kind, SType stype, TimePoint now);
    void write_frame(const Header& h, std::span<const std::uint8_t> body);
    void reject(const Header& h, RejectReason reason);
    void reset_connection();

    Config config_;
    State state_ = State::not_connected;
    Stats stats_;

    // Receive buffer: [begin, end) holds bytes not yet parsed.
    std::vector<std::uint8_t> in_;
    std::size_t in_begin_ = 0;
    std::size_t in_end_ = 0;
    TimePoint last_rx_;

    std::vector<std::uint8_t> out_;
    bool wants_close_ = false;

    std::uint32_t next_system_ = 1;
    std::array<Transaction, kMaxOpenTransactions> open_{};

    Control control_ = Control::none;
    std::uint32_t control_system_ = 0;
    TimePoint control_deadline_;
    std::optional<TimePoint> t7_deadline_;
    std::optional<TimePoint> linktest_at_;
    TimePoint reconnect_at_;
    std::optional<Closed> closed_; // reported by the next poll()
};

} // namespace waferedge::hsms
