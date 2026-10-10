#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

// HSMS (SEMI E37) message framing: on the TCP stream every message is
//
//   4 bytes   length, big-endian: the bytes that follow (10 + body)
//   10 bytes  header
//   n bytes   body: one SECS-II item (secs::decode), or nothing
//
// Header bytes:
//   0-1  session id (the device id of data messages; 0xFFFF for most control messages)
//   2    data: W-bit (bit 7, "reply expected") and stream; control: depends on the SType
//   3    data: function; control: a status or reason code
//   4    PType: 0, SECS-II
//   5    SType: 0 for a data message, otherwise which control message
//   6-9  system bytes: the transaction id; a reply carries its request's
// docs/secs.md has the layouts used.
namespace waferedge::hsms {

inline constexpr std::size_t kLengthSize = 4;
inline constexpr std::size_t kHeaderSize = 10;
// HSMS-SS uses this session id for Select, Deselect, Linktest and Separate.
inline constexpr std::uint16_t kControlSession = 0xFFFF;

enum class SType : std::uint8_t {
    data = 0,
    select_req = 1,
    select_rsp = 2,
    deselect_req = 3,
    deselect_rsp = 4,
    linktest_req = 5,
    linktest_rsp = 6,
    reject_req = 7,
    separate_req = 9,
};

[[nodiscard]] constexpr bool known_stype(std::uint8_t s) noexcept {
    return s <= 7 || s == 9;
}
[[nodiscard]] std::string_view stype_name(SType s) noexcept;

// Byte 3 of Select.rsp.
enum class SelectStatus : std::uint8_t {
    ok = 0,
    already_active = 1,
    not_ready = 2,
    connections_exhausted = 3,
};
// Byte 3 of Deselect.rsp.
enum class DeselectStatus : std::uint8_t { ok = 0, not_established = 1, busy = 2 };
// Byte 3 of Reject.req; byte 2 carries the rejected message's SType (its PType for
// ptype_not_supported).
enum class RejectReason : std::uint8_t {
    stype_not_supported = 1,
    ptype_not_supported = 2,
    transaction_not_open = 3,
    entity_not_selected = 4,
};

struct Header {
    std::uint16_t session_id = 0;
    std::uint8_t byte2 = 0;
    std::uint8_t byte3 = 0;
    std::uint8_t ptype = 0;
    std::uint8_t stype = 0; // raw: a peer may send one we don't know
    std::uint32_t system_bytes = 0;

    // Data messages.
    [[nodiscard]] constexpr std::uint8_t stream() const noexcept { return byte2 & 0x7FU; }
    [[nodiscard]] constexpr std::uint8_t function() const noexcept { return byte3; }
    [[nodiscard]] constexpr bool reply_expected() const noexcept { return (byte2 & 0x80U) != 0; }
    [[nodiscard]] constexpr bool is_data() const noexcept { return stype == 0; }
    // An even function answers the odd one before it (S6F12 answers S6F11); F0 aborts a
    // transaction and also closes it.
    [[nodiscard]] constexpr bool is_reply() const noexcept { return is_data() && byte3 % 2 == 0; }

    friend constexpr bool operator==(const Header&, const Header&) = default;
};

[[nodiscard]] constexpr Header data_header(std::uint16_t session, std::uint8_t stream,
                                           std::uint8_t function, bool reply_expected,
                                           std::uint32_t system_bytes) noexcept {
    return {session,  static_cast<std::uint8_t>((reply_expected ? 0x80U : 0U) | (stream & 0x7FU)),
            function, 0,
            0,        system_bytes};
}
[[nodiscard]] constexpr Header control_header(SType s, std::uint32_t system_bytes,
                                              std::uint8_t byte2 = 0, std::uint8_t byte3 = 0,
                                              std::uint16_t session = kControlSession) noexcept {
    return {session, byte2, byte3, 0, static_cast<std::uint8_t>(s), system_bytes};
}

constexpr void write_header(const Header& h, std::uint8_t* out) noexcept {
    out[0] = static_cast<std::uint8_t>(h.session_id >> 8U);
    out[1] = static_cast<std::uint8_t>(h.session_id);
    out[2] = h.byte2;
    out[3] = h.byte3;
    out[4] = h.ptype;
    out[5] = h.stype;
    out[6] = static_cast<std::uint8_t>(h.system_bytes >> 24U);
    out[7] = static_cast<std::uint8_t>(h.system_bytes >> 16U);
    out[8] = static_cast<std::uint8_t>(h.system_bytes >> 8U);
    out[9] = static_cast<std::uint8_t>(h.system_bytes);
}

[[nodiscard]] constexpr Header read_header(const std::uint8_t* in) noexcept {
    return {static_cast<std::uint16_t>((in[0] << 8U) | in[1]),
            in[2],
            in[3],
            in[4],
            in[5],
            (std::uint32_t{in[6]} << 24U) | (std::uint32_t{in[7]} << 16U) |
                (std::uint32_t{in[8]} << 8U) | std::uint32_t{in[9]}};
}

[[nodiscard]] constexpr std::uint32_t read_length(const std::uint8_t* in) noexcept {
    return (std::uint32_t{in[0]} << 24U) | (std::uint32_t{in[1]} << 16U) |
           (std::uint32_t{in[2]} << 8U) | std::uint32_t{in[3]};
}

static_assert([] {
    std::array<std::uint8_t, kHeaderSize> b{};
    const Header h = data_header(0x0102, 6, 11, true, 0xA0B0C0D0);
    write_header(h, b.data());
    return read_header(b.data()) == h && b[2] == 0x86 && b[3] == 11 && b[6] == 0xA0;
}());

} // namespace waferedge::hsms
