#pragma once

#include "waferedge/secs/encoder.hpp"
#include "waferedge/secs/item.hpp"
#include "waferedge/wafer_map.hpp"

#include <cstdint>
#include <span>
#include <string_view>

// The GEM (SEMI E30) messages WaferEdge uses, as typed encoders and zero-copy decoders over
// the SECS-II codec. Decoders take the message body (an ItemView, or the raw bytes) and
// return views into it: strings and the wafer map stay in the receive buffer. A body that
// doesn't have the expected shape is an Error{wrong_format} (the equipment answers S9F7).
// docs/secs.md lists every layout.
namespace waferedge::gem {

using secs::Result;

// --- Identifiers ----------------------------------------------------------------------------
// The tool emulator's predefined data collection (ADR-0011): one event, one report.
inline constexpr std::uint32_t kCeidWaferSorted = 100;
inline constexpr std::uint32_t kRptidWafer = 10;
// S2F41 remote commands and their one parameter.
inline constexpr std::string_view kRcmdHold = "HOLD";
inline constexpr std::string_view kRcmdRelease = "RELEASE";
inline constexpr std::string_view kCpLotId = "LOTID";

// S1F14 COMMACK.
enum class Commack : std::uint8_t { accepted = 0, denied = 1 };
// S1F16 OFLACK / S1F18 ONLACK.
enum class Onlack : std::uint8_t { accepted = 0, refused = 1, already_online = 2 };
// S2F42 HCACK.
enum class Hcack : std::uint8_t {
    done = 0,
    invalid_command = 1,
    cannot_do_now = 2,
    parameter_invalid = 3,
    will_finish_later = 4,
    already_in_condition = 5,
    no_such_object = 6,
};
// S2F42 CPACK, per parameter.
enum class Cpack : std::uint8_t { no_such_name = 1, illegal_value = 2, illegal_format = 3 };

// --- S1F1 / S1F2, S1F13 / S1F14 --------------------------------------------------------------
// The equipment's identity: S1F2 and S1F13 from the equipment carry <L [2] <A MDLN> <A SOFTREV>>,
// the host's carry <L [0]>.
struct Identity {
    std::string_view model;    // MDLN
    std::string_view software; // SOFTREV
};
void encode_identity(secs::Encoder& e, const Identity* identity);    // nullptr: the host's <L [0]>
[[nodiscard]] Result<Identity> decode_identity(secs::ItemView body); // empty for <L [0]>

// S1F14: <L [2] <B COMMACK> <L MDLN SOFTREV or empty>>.
void encode_s1f14(secs::Encoder& e, Commack ack, const Identity* identity);
struct EstablishAck {
    Commack ack = Commack::accepted;
    Identity identity;
};
[[nodiscard]] Result<EstablishAck> decode_s1f14(secs::ItemView body);

// One-byte acknowledgements: S1F16 OFLACK, S1F18 ONLACK, S5F2 ACKC5, S6F12 ACKC6: <B code>.
void encode_ack(secs::Encoder& e, std::uint8_t code);
[[nodiscard]] Result<std::uint8_t> decode_ack(secs::ItemView body);

// --- S6F11 wafer report -----------------------------------------------------------------------
//   <L [3] <U4 DATAID> <U4 CEID 100>
//     <L [1] <L [2] <U4 RPTID 10>
//       <L [5] <A LOTID> <U4 WAFERID> <U2 ROWS> <U2 COLS> <U1 BINS...>>>>>
struct WaferReport {
    std::uint64_t data_id = 0;
    std::string_view lot;
    std::uint64_t wafer = 0;
    WaferMapView map; // the U1 item's bytes in the receive buffer
};
void encode_wafer_report(secs::Encoder& e, std::uint32_t data_id, std::string_view lot,
                         std::uint32_t wafer, WaferMapView map);
// Any S6F11: DATAID, CEID and the report list, before looking inside.
struct EventReport {
    std::uint64_t data_id = 0;
    std::uint64_t ceid = 0;
    secs::ListView reports;
};
[[nodiscard]] Result<EventReport> decode_event_report(secs::ItemView body);
// An S6F11 with CEID 100 and report 10. ID items may be any U1-U8 width (GEM lets the tool
// choose); the bins must be U1 with rows x cols values.
[[nodiscard]] Result<WaferReport> decode_wafer_report(secs::ItemView body);
[[nodiscard]] Result<WaferReport> decode_wafer_report(std::span<const std::uint8_t> body);

// --- S5F1 alarm -------------------------------------------------------------------------------
//   <L [3] <B ALCD> <U4 ALID> <A ALTX>>; ALCD bit 7: set (1) or cleared (0), bits 0-6: category.
struct Alarm {
    bool set = false;
    std::uint8_t category = 0;
    std::uint64_t id = 0;
    std::string_view text;
};
void encode_alarm(secs::Encoder& e, const Alarm& alarm);
[[nodiscard]] Result<Alarm> decode_alarm(secs::ItemView body);

// --- S2F41 / S2F42 host command ------------------------------------------------------------
//   S2F41: <L [2] <A RCMD> <L [n] <L [2] <A CPNAME> CPVAL>...>>
//   S2F42: <L [2] <B HCACK> <L [n] <L [2] <A CPNAME> <B CPACK>>...>>
// WaferEdge sends HOLD and RELEASE with one parameter, LOTID, as <A>.
enum class LotAction : std::uint8_t { hold, release };
struct LotCommand {
    LotAction action = LotAction::hold;
    std::string_view lot;
};
void encode_lot_command(secs::Encoder& e, const LotCommand& command);
struct HostCommand {
    std::string_view rcmd;
    secs::ListView parameters; // each <L [2] <A CPNAME> CPVAL>
};
[[nodiscard]] Result<HostCommand> decode_host_command(secs::ItemView body);

// The value of parameter `name`, or out_of_range if absent / wrong_format if malformed.
[[nodiscard]] Result<secs::ItemView> find_parameter(const HostCommand& command,
                                                    std::string_view name);

// S2F42 with no per-parameter errors, or one parameter's error.
struct ParameterError {
    std::string_view name;
    Cpack ack;
};
void encode_host_command_ack(secs::Encoder& e, Hcack ack,
                             std::span<const ParameterError> errors = {});
struct HostCommandAck {
    Hcack ack = Hcack::done;
    secs::ListView errors; // each <L [2] <A CPNAME> <B CPACK>>
};
[[nodiscard]] Result<HostCommandAck> decode_host_command_ack(secs::ItemView body);

// --- S9 errors: the equipment's report of a message it couldn't take --------------------
// S9F1 unrecognized device id, S9F3 unrecognized stream, S9F5 unrecognized function,
// S9F7 illegal data, S9F9 transaction timer (T3) timeout. Body: <B MHEAD>, the 10-byte
// HSMS header of the message at fault.
void encode_mhead(secs::Encoder& e, std::span<const std::uint8_t, 10> header);

} // namespace waferedge::gem
