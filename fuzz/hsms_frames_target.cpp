#include "hsms_frames_target.hpp"

#include "waferedge/hsms/protocol.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <span>
#include <variant>
#include <vector>

namespace waferedge::fuzz {

namespace {

void check(bool ok) {
    if (!ok) {
        std::abort();
    }
}

// Everything the protocol writes must be whole frames with valid lengths and PType 0.
void check_output(hsms::Protocol& p, std::vector<std::uint8_t>& out) {
    p.take_output(out);
    std::size_t pos = 0;
    while (pos < out.size()) {
        check(out.size() - pos >= hsms::kLengthSize + hsms::kHeaderSize);
        const std::uint32_t length = hsms::read_length(out.data() + pos);
        check(length >= hsms::kHeaderSize && length <= p.config().max_message);
        check(out.size() - pos >= hsms::kLengthSize + length);
        const auto h = hsms::read_header(out.data() + pos + hsms::kLengthSize);
        check(h.ptype == 0 && hsms::known_stype(h.stype));
        pos += hsms::kLengthSize + length;
    }
}

void feed(hsms::Protocol& p, std::span<const std::uint8_t> bytes, hsms::TimePoint now) {
    auto buffer = p.receive_buffer(bytes.size());
    check(buffer.size() >= bytes.size());
    std::memcpy(buffer.data(), bytes.data(), bytes.size());
    p.on_received(bytes.size(), now);
}

void drain(hsms::Protocol& p, hsms::TimePoint now) {
    while (auto e = p.poll(now)) {
        if (const auto* m = std::get_if<hsms::DataMessage>(&*e)) {
            check(p.state() == hsms::State::selected); // data only arrives when selected
            check(m->body.size() + hsms::kHeaderSize <= p.config().max_message);
            if (m->kind == hsms::DataMessage::Kind::primary && m->header.reply_expected()) {
                check(p.reply(m->header, m->body).has_value()); // echo it back
            }
        } else if (std::holds_alternative<hsms::Closed>(*e)) {
            check(p.state() == hsms::State::not_connected && p.wants_close());
        } else if (std::holds_alternative<hsms::Selected>(*e)) {
            check(p.state() == hsms::State::selected);
        }
    }
}

} // namespace

int hsms_frames(const std::uint8_t* data, std::size_t size) {
    using namespace std::chrono_literals;
    if (size == 0) {
        return 0;
    }
    hsms::Config config;
    config.role = hsms::Role::passive;
    config.max_message = 4096;
    hsms::Protocol p(config);
    hsms::TimePoint now = hsms::TimePoint{} + 1h;
    p.on_connected(now);

    const std::size_t chunk = 1 + (data[0] >> 1U); // 1..128 bytes per read
    if ((data[0] & 1U) != 0) {
        std::array<std::uint8_t, hsms::kLengthSize + hsms::kHeaderSize> select{0, 0, 0, 10};
        hsms::write_header(hsms::control_header(hsms::SType::select_req, 1),
                           select.data() + hsms::kLengthSize);
        feed(p, select, now);
        drain(p, now);
    }
    std::vector<std::uint8_t> out;
    const std::span<const std::uint8_t> stream(data + 1, size - 1);
    for (std::size_t pos = 0; pos < stream.size() && p.state() != hsms::State::not_connected;
         pos += chunk) {
        feed(p, stream.subspan(pos, std::min(chunk, stream.size() - pos)), now);
        drain(p, now);
        check_output(p, out);
        now += 100ms;
    }
    // Let every timer run out: the protocol must end up closed or idle, never stuck.
    now += 1h;
    drain(p, now);
    check_output(p, out);
    check(p.state() != hsms::State::not_selected); // T7 closes an unselected connection
    return 0;
}

} // namespace waferedge::fuzz
