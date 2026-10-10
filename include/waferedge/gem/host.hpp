#pragma once

#include "waferedge/gem/endpoint.hpp"

// The host side of the GEM subset: what the edge host runs (phase 4). It establishes
// communication, acknowledges the equipment's reports (S6F11 → S6F12, S5F1 → S5F2) and turns
// them into events, answers S1F1, and sends lot holds and releases (S2F41), S1F1, S1F15 and
// S1F17. An S9Fx from the equipment becomes ErrorReported; other primaries get SxF0.
namespace waferedge::gem {

struct HostConfig {
    hsms::Duration establish_delay = std::chrono::seconds(10);
    bool comm_enabled = true;
};

class Host final : public Endpoint {
public:
    explicit Host(HostConfig config = {});

    void on_hsms(const hsms::Event& event, Link& link, hsms::TimePoint now);

    // S2F41 HOLD / RELEASE for `lot`; the S2F42 comes back as ReplyReceived (code: HCACK).
    GemResult<std::uint32_t> command(Link& link, const LotCommand& command);
    // S1F1 are you there, S1F17 request on-line, S1F15 request off-line.
    GemResult<std::uint32_t> are_you_there(Link& link);
    GemResult<std::uint32_t> request_online(Link& link);
    GemResult<std::uint32_t> request_offline(Link& link);

private:
    [[nodiscard]] const Identity* identity() const noexcept override { return nullptr; }
    void handle_primary(const hsms::DataMessage& m, Link& link);
    void handle_reply(const hsms::DataMessage& m);
    GemResult<std::uint32_t> send_empty(Link& link, std::uint8_t stream, std::uint8_t function);
};

} // namespace waferedge::gem
