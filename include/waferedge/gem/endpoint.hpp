#pragma once

#include "waferedge/gem/link.hpp"
#include "waferedge/gem/messages.hpp"
#include "waferedge/hsms/protocol.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

// What GEM's Equipment and Host share: the communication state machine (E30 4.2), the queue
// of GEM events, and the body buffer. Both are sans-I/O like hsms::Protocol: feed them HSMS
// events (on_hsms) and the time (tick), take GEM events (poll), and they send through a
// Link. ADR-0011.
//
//   DISABLED --enable--> NOT COMMUNICATING --link selected--> WAIT CRA --S1F14 ok--> COMMUNICATING
//                              ^                  |    ^                              |
//                              |      T3 / denied v    | delay over                   |
//                              |               WAIT DELAY                             |
//                              +--------------- HSMS link lost ------------------------+
//
// Either side may send S1F13; receiving one, answered with COMMACK 0, also establishes
// communication. Until then every other primary is answered with SxF0 (abort).
namespace waferedge::gem {

enum class CommState : std::uint8_t {
    disabled,
    not_communicating,
    wait_cra,
    wait_delay,
    communicating
};

// E30 4.3. OFF-LINE: equipment off-line (the operator's choice), attempt on-line (S1F1 sent),
// host off-line (the host's choice, or the attempt failed). ON-LINE: local or remote.
enum class ControlState : std::uint8_t {
    equipment_offline,
    attempt_online,
    host_offline,
    online_local,
    online_remote,
};

[[nodiscard]] std::string_view comm_state_name(CommState s) noexcept;
[[nodiscard]] std::string_view control_state_name(ControlState s) noexcept;

// --- Events ----------------------------------------------------------------------------------
// Views (strings, the wafer map) point into the HSMS receive buffer: valid until the next
// read, so poll() right after on_hsms().
struct Communicating {};
struct NotCommunicating {};
struct ControlStateChanged {
    ControlState state;
};
// Equipment: S2F41 HOLD / RELEASE from the host, valid and accepted for processing. The app
// must answer it with Equipment::answer (the host's T3 runs meanwhile).
struct LotCommandReceived {
    LotCommand command;
    hsms::Header primary;
};
// Host: an S6F11 wafer report (already acknowledged with S6F12).
struct WaferReported {
    WaferReport report;
};
// Host: an S6F11 for another event (acknowledged).
struct EventReported {
    std::uint64_t ceid = 0;
    std::uint64_t data_id = 0;
};
// Host: an S5F1 alarm (acknowledged).
struct AlarmReported {
    Alarm alarm;
};
// A reply to one of our primaries: S1F2 (code 0), S1F16 OFLACK, S1F18 ONLACK, S2F42 HCACK,
// S5F2 ACKC5, S6F12 ACKC6.
struct ReplyReceived {
    std::uint32_t system = 0;
    std::uint8_t stream = 0;
    std::uint8_t function = 0;
    std::uint8_t code = 0;
};
// One of our primaries got no usable reply.
struct TransactionFailed {
    enum class Why : std::uint8_t { timeout, aborted, rejected, malformed_reply };
    std::uint32_t system = 0;
    Why why = Why::timeout;
};
// Host: an S9Fx from the equipment about one of our messages.
struct ErrorReported {
    std::uint8_t function = 0; // 1, 3, 5, 7 or 9
    hsms::Header about;        // the MHEAD it carried
};
// A primary whose body didn't have the expected shape (the equipment also sent S9F7; the
// host answered an S6F11 with ACKC6 = 1).
struct MalformedMessage {
    hsms::Header header;
};

using Event = std::variant<Communicating, NotCommunicating, ControlStateChanged, LotCommandReceived,
                           WaferReported, EventReported, AlarmReported, ReplyReceived,
                           TransactionFailed, ErrorReported, MalformedMessage>;

// What the app's own requests can fail with.
enum class GemError : std::uint8_t {
    not_communicating,
    offline,
    busy, // HSMS has kMaxOpenTransactions primaries awaiting replies: retry after one arrives
    link, // the HSMS link refused it (not selected, too long)
};
[[nodiscard]] constexpr GemError gem_error(SendError e) noexcept {
    return e == SendError::too_many_open ? GemError::busy : GemError::link;
}
template <typename T>
using GemResult = std::expected<T, GemError>;

class Endpoint {
public:
    Endpoint(const Endpoint&) = delete;
    Endpoint& operator=(const Endpoint&) = delete;
    Endpoint(Endpoint&&) = delete;
    Endpoint& operator=(Endpoint&&) = delete;
    virtual ~Endpoint() = default;

    [[nodiscard]] CommState comm_state() const noexcept { return comm_; }
    [[nodiscard]] bool communicating() const noexcept { return comm_ == CommState::communicating; }

    // The next GEM event, if any.
    [[nodiscard]] std::optional<Event> poll() noexcept;
    // When tick() must run (the end of WAIT DELAY), if a timer is running.
    [[nodiscard]] std::optional<hsms::TimePoint> next_deadline() const noexcept;
    // Retries S1F13 when WAIT DELAY is over.
    void tick(Link& link, hsms::TimePoint now);

protected:
    Endpoint(hsms::Duration establish_delay, bool enabled);

    // The communication part of an HSMS event. Returns true if it consumed the event; a
    // derived class handles the rest.
    bool communication(const hsms::Event& event, Link& link, hsms::TimePoint now);
    // Our S1F13 body (equipment: MDLN and SOFTREV; host: empty list).
    [[nodiscard]] virtual const Identity* identity() const noexcept = 0;

    void push(const Event& event) noexcept;
    // An encoder over the endpoint's reused body buffer; finish it before the next one.
    [[nodiscard]] secs::Encoder encoder() noexcept { return secs::Encoder(buffer_); }
    // The encoded body (the encoder can't fail on the fixed shapes GEM writes).
    [[nodiscard]] static std::span<const std::uint8_t> body(const secs::Encoder& e) noexcept {
        return *e.finish();
    }

private:
    void send_s1f13(Link& link, hsms::TimePoint now);
    void wait_delay(hsms::TimePoint now);

    hsms::Duration delay_;
    bool enabled_;
    CommState comm_;
    std::optional<std::uint32_t> s1f13_system_;
    hsms::TimePoint retry_at_;
    std::vector<std::uint8_t> buffer_;
    // A fixed ring: one HSMS event makes at most two GEM events, and poll() drains them.
    std::array<Event, 8> queue_{};
    std::size_t head_ = 0;
    std::size_t count_ = 0;
};

} // namespace waferedge::gem
