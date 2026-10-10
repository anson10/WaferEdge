#pragma once

#include "waferedge/gem/endpoint.hpp"

#include <string>

// The equipment side of the GEM subset: what the tool emulator runs (phase 4).
//
// Control state (E30 4.3), changed by the operator (go_online, go_offline, set_remote) and
// by the host (S1F17 request on-line, S1F15 request off-line):
//
//   EQUIPMENT OFF-LINE --go_online--> ATTEMPT ON-LINE --S1F2--> ON-LINE (LOCAL | REMOTE)
//                                          | S1F0, T3                   |
//                                          v                            | S1F15
//                                     HOST OFF-LINE <-------------------+
//                                          | S1F17
//                                          +--------------------------> ON-LINE
//
// Off-line, the equipment answers S1F1 and every primary but S1F13 / S1F17 with SxF0, and
// sends nothing but S1F13 and S1F1. On-line it serves S1F1, S1F15, S2F41 HOLD / RELEASE
// (executed only in REMOTE; LOCAL answers HCACK 2), and sends S6F11 wafer reports and S5F1
// alarms. Unknown streams and functions get S9F3 / S9F5, malformed bodies S9F7, and a T3
// timeout on its own primary S9F9.
namespace waferedge::gem {

struct EquipmentConfig {
    std::string model = "WAFEREDGE-EMU"; // MDLN
    std::string software = "1.0.0";      // SOFTREV
    hsms::Duration establish_delay = std::chrono::seconds(10);
    bool comm_enabled = true;
    ControlState initial = ControlState::equipment_offline;
    bool remote = true; // the LOCAL / REMOTE switch, used on entering ON-LINE
};

class Equipment final : public Endpoint {
public:
    explicit Equipment(EquipmentConfig config);

    [[nodiscard]] ControlState control_state() const noexcept { return control_; }
    [[nodiscard]] bool online() const noexcept {
        return control_ == ControlState::online_local || control_ == ControlState::online_remote;
    }

    // Everything the HSMS layer reports, in order. Then poll() the GEM events.
    void on_hsms(const hsms::Event& event, Link& link, hsms::TimePoint now);

    // --- The operator ------------------------------------------------------------------
    // Off-line → ATTEMPT ON-LINE (sends S1F1); ON-LINE when the host answers S1F2.
    void go_online(Link& link);
    void go_offline();
    void set_remote(bool remote);

    // --- Reports -----------------------------------------------------------------------
    // S6F11 for CEID 100 (needs COMMUNICATING and ON-LINE). Returns the system bytes; the
    // host's S6F12 comes back as ReplyReceived.
    GemResult<std::uint32_t> report_wafer(Link& link, std::string_view lot, std::uint32_t wafer,
                                          WaferMapView map);
    GemResult<std::uint32_t> report_alarm(Link& link, const Alarm& alarm);

    // The S2F42 for a LotCommandReceived: done, already_in_condition, no_such_object, ...
    GemResult<void> answer(Link& link, const hsms::Header& primary, Hcack ack);

private:
    [[nodiscard]] const Identity* identity() const noexcept override { return &identity_; }
    void handle_primary(const hsms::DataMessage& m, Link& link);
    void handle_reply(const hsms::DataMessage& m);
    void host_command(const hsms::DataMessage& m, Link& link);
    void error(Link& link, std::uint8_t function, const hsms::Header& about);
    void set_control(ControlState state);
    void enter_online();
    void ack(Link& link, const hsms::Header& primary, std::uint8_t code);

    EquipmentConfig config_;
    Identity identity_;
    ControlState control_;
    std::optional<std::uint32_t> s1f1_system_; // ATTEMPT ON-LINE's S1F1
    std::uint32_t next_data_id_ = 1;
};

} // namespace waferedge::gem
