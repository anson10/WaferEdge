#include "gem_messages_target.hpp"

#include "waferedge/gem/equipment.hpp"
#include "waferedge/gem/host.hpp"

#include <cstdlib>
#include <vector>

namespace waferedge::fuzz {

namespace {

void check(bool ok) {
    if (!ok) {
        std::abort();
    }
}

// Accepts everything, checks every body is one valid SECS-II item (or empty), and remembers
// the system bytes it handed out so replies can be aimed at them.
class CheckingLink final : public gem::Link {
public:
    std::vector<std::uint32_t> sent;

    std::expected<std::uint32_t, gem::SendError> send(std::uint8_t, std::uint8_t, bool,
                                                      std::span<const std::uint8_t> body) override {
        check_body(body);
        sent.push_back(next_);
        return next_++;
    }
    std::expected<void, gem::SendError> reply(const hsms::Header&,
                                              std::span<const std::uint8_t> body) override {
        check_body(body);
        return {};
    }
    std::expected<void, gem::SendError> abort(const hsms::Header&) override { return {}; }

private:
    static void check_body(std::span<const std::uint8_t> body) {
        check(body.empty() || secs::decode(body).has_value());
    }
    std::uint32_t next_ = 1000;
};

template <typename Endpoint>
void drain(Endpoint& endpoint, gem::Equipment* equipment, CheckingLink& link) {
    while (auto e = endpoint.poll()) {
        if (const auto* c = std::get_if<gem::LotCommandReceived>(&*e)) {
            check(equipment != nullptr &&
                  equipment->control_state() == gem::ControlState::online_remote);
            (void)equipment->answer(link, c->primary, gem::Hcack::done);
        } else if (const auto* w = std::get_if<gem::WaferReported>(&*e)) {
            check(static_cast<std::size_t>(w->report.map.rows()) *
                      static_cast<std::size_t>(w->report.map.cols()) ==
                  w->report.map.bins().size());
        }
    }
}

// Selected, then an S1F14 accepting the endpoint's own S1F13: COMMUNICATING.
template <typename Endpoint>
void establish(Endpoint& endpoint, CheckingLink& link, hsms::TimePoint now) {
    endpoint.on_hsms(hsms::Selected{}, link, now);
    static constexpr std::array<std::uint8_t, 7> accepted = {0x01, 0x02, 0x21, 0x01,
                                                             0x00, 0x01, 0x00};
    endpoint.on_hsms(hsms::DataMessage{hsms::data_header(1, 1, 14, false, link.sent.back()),
                                       accepted, hsms::DataMessage::Kind::reply},
                     link, now);
    check(endpoint.communicating());
}

} // namespace

int gem_messages(const std::uint8_t* data, std::size_t size) {
    using namespace std::chrono_literals;
    hsms::TimePoint now = hsms::TimePoint{} + 1h;
    CheckingLink tool_link;
    CheckingLink host_link;
    gem::EquipmentConfig config;
    config.initial = gem::ControlState::online_remote;
    gem::Equipment equipment(config);
    gem::Host host;
    establish(equipment, tool_link, now);
    establish(host, host_link, now);
    drain(equipment, &equipment, tool_link);
    drain(host, nullptr, host_link);

    std::size_t pos = 0;
    while (size - pos >= 5) {
        const std::uint8_t flags = data[pos];
        const auto stream = static_cast<std::uint8_t>(data[pos + 1] & 0x7FU);
        const std::uint8_t function = data[pos + 2];
        const std::size_t length = std::min<std::size_t>(
            (std::size_t{data[pos + 3]} << 8U) | data[pos + 4], size - pos - 5);
        const std::span<const std::uint8_t> body(data + pos + 5, length);
        pos += 5 + length;

        const bool to_tool = (flags & 1U) != 0;
        CheckingLink& link = to_tool ? tool_link : host_link;
        // A reply aims at a transaction the endpoint really opened, when there is one.
        const bool reply = (flags & 2U) != 0 && !link.sent.empty();
        const std::uint32_t system = reply ? link.sent[(flags >> 4U) % link.sent.size()]
                                           : 0xF000U + static_cast<std::uint32_t>(pos & 0xFFFU);
        const hsms::Header h = hsms::data_header(1, stream, function, (flags & 4U) != 0, system);
        hsms::Event event = hsms::DataMessage{
            h, body, reply ? hsms::DataMessage::Kind::reply : hsms::DataMessage::Kind::primary};
        if ((flags & 0xC8U) == 0x08U) {
            event = hsms::ReplyTimeout{h};
        } else if ((flags & 0xC8U) == 0xC8U) {
            event = hsms::Closed{hsms::CloseReason::transport};
        }
        now += 1s;
        if (to_tool) {
            equipment.on_hsms(event, tool_link, now);
            equipment.tick(tool_link, now);
            drain(equipment, &equipment, tool_link);
        } else {
            host.on_hsms(event, host_link, now);
            host.tick(host_link, now);
            drain(host, nullptr, host_link);
        }
        if (std::holds_alternative<hsms::Closed>(event)) {
            if (to_tool) {
                check(!equipment.communicating());
                establish(equipment, tool_link, now);
            } else {
                check(!host.communicating());
                establish(host, host_link, now);
            }
        }
    }
    return 0;
}

} // namespace waferedge::fuzz
