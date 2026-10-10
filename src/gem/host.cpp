#include "waferedge/gem/host.hpp"

namespace waferedge::gem {

Host::Host(HostConfig config) : Endpoint(config.establish_delay, config.comm_enabled) {}

void Host::on_hsms(const hsms::Event& event, Link& link, hsms::TimePoint now) {
    if (communication(event, link, now)) {
        return;
    }
    if (const auto* timeout = std::get_if<hsms::ReplyTimeout>(&event)) {
        push(TransactionFailed{timeout->request.system_bytes, TransactionFailed::Why::timeout});
        return;
    }
    if (const auto* rejected = std::get_if<hsms::Rejected>(&event)) {
        push(TransactionFailed{rejected->reject.system_bytes, TransactionFailed::Why::rejected});
        return;
    }
    if (const auto* m = std::get_if<hsms::DataMessage>(&event)) {
        if (m->kind == hsms::DataMessage::Kind::primary) {
            handle_primary(*m, link);
        } else if (m->kind == hsms::DataMessage::Kind::reply) {
            handle_reply(*m);
        }
    }
}

void Host::handle_primary(const hsms::DataMessage& m, Link& link) {
    const hsms::Header& h = m.header;
    const std::uint8_t s = h.stream();
    const std::uint8_t f = h.function();
    auto item = secs::decode(m.body);
    const auto ack = [&](std::uint8_t code) {
        auto e = encoder();
        encode_ack(e, code);
        (void)link.reply(h, body(e));
    };

    if (s == 6 && f == 11) {
        auto event = item.and_then([](secs::ItemView v) { return decode_event_report(v); });
        if (!event) {
            ack(1); // ACKC6 != 0: not accepted
            push(MalformedMessage{h});
            return;
        }
        if (event->ceid == kCeidWaferSorted) {
            auto report = decode_wafer_report(*item);
            if (!report) {
                ack(1);
                push(MalformedMessage{h});
                return;
            }
            ack(0);
            push(WaferReported{*report});
            return;
        }
        ack(0);
        push(EventReported{event->ceid, event->data_id});
        return;
    }
    if (s == 5 && f == 1) {
        auto alarm = item.and_then([](secs::ItemView v) { return decode_alarm(v); });
        if (!alarm) {
            ack(1);
            push(MalformedMessage{h});
            return;
        }
        ack(0);
        push(AlarmReported{*alarm});
        return;
    }
    if (s == 1 && f == 1) { // the equipment attempting on-line: answer S1F2 <L [0]>
        auto e = encoder();
        encode_identity(e, nullptr);
        (void)link.reply(h, body(e));
        return;
    }
    if (s == 9) { // the equipment reporting one of our messages: <B MHEAD>
        auto mhead = item.and_then([](secs::ItemView v) { return v.as<secs::Format::binary>(); });
        if (mhead && mhead->size() == hsms::kHeaderSize) {
            push(ErrorReported{f, hsms::read_header(mhead->bytes().data())});
        } else {
            push(MalformedMessage{h});
        }
        return;
    }
    if (h.reply_expected()) {
        (void)link.abort(h);
    }
}

void Host::handle_reply(const hsms::DataMessage& m) {
    const hsms::Header& h = m.header;
    if (h.function() == 0) {
        push(TransactionFailed{h.system_bytes, TransactionFailed::Why::aborted});
        return;
    }
    auto item = secs::decode(m.body);
    if (h.stream() == 1 && h.function() == 2) {
        push(ReplyReceived{h.system_bytes, 1, 2, 0});
        return;
    }
    if (h.stream() == 2 && h.function() == 42) {
        auto ack = item.and_then([](secs::ItemView v) { return decode_host_command_ack(v); });
        if (!ack) {
            push(TransactionFailed{h.system_bytes, TransactionFailed::Why::malformed_reply});
            return;
        }
        push(ReplyReceived{h.system_bytes, 2, 42, static_cast<std::uint8_t>(ack->ack)});
        return;
    }
    auto code = item.and_then([](secs::ItemView v) { return decode_ack(v); }); // S1F16, S1F18
    if (!code) {
        push(TransactionFailed{h.system_bytes, TransactionFailed::Why::malformed_reply});
        return;
    }
    push(ReplyReceived{h.system_bytes, h.stream(), h.function(), *code});
}

GemResult<std::uint32_t> Host::command(Link& link, const LotCommand& command) {
    if (!communicating()) {
        return std::unexpected(GemError::not_communicating);
    }
    auto e = encoder();
    encode_lot_command(e, command);
    auto system = link.send(2, 41, true, body(e));
    if (!system) {
        return std::unexpected(GemError::link);
    }
    return *system;
}

GemResult<std::uint32_t> Host::send_empty(Link& link, std::uint8_t stream, std::uint8_t function) {
    if (!communicating()) {
        return std::unexpected(GemError::not_communicating);
    }
    auto system = link.send(stream, function, true, {});
    if (!system) {
        return std::unexpected(GemError::link);
    }
    return *system;
}

GemResult<std::uint32_t> Host::are_you_there(Link& link) {
    return send_empty(link, 1, 1);
}
GemResult<std::uint32_t> Host::request_online(Link& link) {
    return send_empty(link, 1, 17);
}
GemResult<std::uint32_t> Host::request_offline(Link& link) {
    return send_empty(link, 1, 15);
}

} // namespace waferedge::gem
