#include "waferedge/gem/equipment.hpp"

namespace waferedge::gem {

Equipment::Equipment(EquipmentConfig config)
    : Endpoint(config.establish_delay, config.comm_enabled), config_(std::move(config)),
      identity_{config_.model, config_.software}, control_(config_.initial) {}

void Equipment::set_control(ControlState state) {
    if (state != control_) {
        control_ = state;
        push(ControlStateChanged{state});
    }
}

void Equipment::enter_online() {
    s1f1_system_.reset();
    set_control(config_.remote ? ControlState::online_remote : ControlState::online_local);
}

void Equipment::go_online(Link& link) {
    if (online() || control_ == ControlState::attempt_online) {
        return;
    }
    if (!communicating()) {
        set_control(ControlState::host_offline); // nobody to ask
        return;
    }
    set_control(ControlState::attempt_online);
    auto system = link.send(1, 1, true, {});
    if (!system) {
        set_control(ControlState::host_offline);
        return;
    }
    s1f1_system_ = *system;
}

void Equipment::go_offline() {
    s1f1_system_.reset();
    set_control(ControlState::equipment_offline);
}

void Equipment::set_remote(bool remote) {
    config_.remote = remote;
    if (online()) {
        set_control(remote ? ControlState::online_remote : ControlState::online_local);
    }
}

void Equipment::on_hsms(const hsms::Event& event, Link& link, hsms::TimePoint now) {
    if (communication(event, link, now)) {
        return;
    }
    if (std::holds_alternative<hsms::Closed>(event) ||
        std::holds_alternative<hsms::Deselected>(event)) {
        if (control_ == ControlState::attempt_online) {
            s1f1_system_.reset();
            set_control(ControlState::host_offline);
        }
        return;
    }
    if (const auto* timeout = std::get_if<hsms::ReplyTimeout>(&event)) {
        const auto system = timeout->request.system_bytes;
        if (system == s1f1_system_) {
            s1f1_system_.reset();
            set_control(ControlState::host_offline);
        } else {
            push(TransactionFailed{system, TransactionFailed::Why::timeout});
        }
        // E30: the equipment reports its own T3 expiry to the host with S9F9.
        std::array<std::uint8_t, hsms::kHeaderSize> mhead{};
        hsms::write_header(timeout->request, mhead.data());
        auto e = encoder();
        encode_mhead(e, mhead);
        (void)link.send(9, 9, false, body(e));
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

void Equipment::handle_reply(const hsms::DataMessage& m) {
    const hsms::Header& h = m.header;
    if (h.system_bytes == s1f1_system_) {
        s1f1_system_.reset();
        if (h.function() == 2) {
            enter_online();
        } else {
            set_control(ControlState::host_offline); // S1F0: the host refuses
        }
        return;
    }
    if (h.function() == 0) {
        push(TransactionFailed{h.system_bytes, TransactionFailed::Why::aborted});
        return;
    }
    // S5F2 ACKC5, S6F12 ACKC6.
    auto code = secs::decode(m.body).and_then([](secs::ItemView v) { return decode_ack(v); });
    if (!code) {
        push(TransactionFailed{h.system_bytes, TransactionFailed::Why::malformed_reply});
        return;
    }
    push(ReplyReceived{h.system_bytes, h.stream(), h.function(), *code});
}

void Equipment::handle_primary(const hsms::DataMessage& m, Link& link) {
    const hsms::Header& h = m.header;
    const std::uint8_t s = h.stream();
    const std::uint8_t f = h.function();
    if (s == 1 && f == 17) { // request on-line
        Onlack result = Onlack::accepted;
        if (online()) {
            result = Onlack::already_online;
        } else if (control_ == ControlState::equipment_offline) {
            result = Onlack::refused; // the operator keeps it off-line
        } else {
            enter_online();
        }
        ack(link, h, static_cast<std::uint8_t>(result));
        return;
    }
    if (!online()) {
        if (h.reply_expected()) {
            (void)link.abort(h); // off-line: S1F1 too gets S1F0
        }
        return;
    }
    if (s == 1 && f == 1) {
        auto e = encoder();
        encode_identity(e, &identity_);
        (void)link.reply(h, body(e));
        return;
    }
    if (s == 1 && f == 15) { // request off-line
        ack(link, h, 0);
        set_control(ControlState::host_offline);
        return;
    }
    if (s == 2 && f == 41) {
        host_command(m, link);
        return;
    }
    error(link, (s == 1 || s == 2) ? 5 : 3, h); // unknown function / stream
}

void Equipment::host_command(const hsms::DataMessage& m, Link& link) {
    const hsms::Header& h = m.header;
    auto command =
        secs::decode(m.body).and_then([](secs::ItemView v) { return decode_host_command(v); });
    if (!command) {
        error(link, 7, h); // illegal data
        push(MalformedMessage{h});
        return;
    }
    const auto reply = [&](Hcack hcack, std::span<const ParameterError> errors = {}) {
        auto e = encoder();
        encode_host_command_ack(e, hcack, errors);
        (void)link.reply(h, body(e));
    };
    const bool hold = command->rcmd == kRcmdHold;
    if (!hold && command->rcmd != kRcmdRelease) {
        reply(Hcack::invalid_command);
        return;
    }
    auto lot_item = find_parameter(*command, kCpLotId);
    if (!lot_item) {
        reply(Hcack::parameter_invalid); // LOTID missing
        return;
    }
    auto lot = lot_item->text();
    if (!lot) {
        const ParameterError bad{kCpLotId, Cpack::illegal_format};
        reply(Hcack::parameter_invalid, std::span(&bad, 1));
        return;
    }
    if (control_ != ControlState::online_remote) {
        reply(Hcack::cannot_do_now); // ON-LINE LOCAL: the operator is in charge
        return;
    }
    push(LotCommandReceived{{hold ? LotAction::hold : LotAction::release, *lot}, h});
}

GemResult<void> Equipment::answer(Link& link, const hsms::Header& primary, Hcack ack) {
    auto e = encoder();
    encode_host_command_ack(e, ack);
    if (!link.reply(primary, body(e))) {
        return std::unexpected(GemError::link);
    }
    return {};
}

void Equipment::ack(Link& link, const hsms::Header& primary, std::uint8_t code) {
    auto e = encoder();
    encode_ack(e, code);
    (void)link.reply(primary, body(e));
}

void Equipment::error(Link& link, std::uint8_t function, const hsms::Header& about) {
    std::array<std::uint8_t, hsms::kHeaderSize> mhead{};
    hsms::write_header(about, mhead.data());
    auto e = encoder();
    encode_mhead(e, mhead);
    (void)link.send(9, function, false, body(e));
}

GemResult<std::uint32_t> Equipment::report_wafer(Link& link, std::string_view lot,
                                                 std::uint32_t wafer, WaferMapView map) {
    if (!communicating()) {
        return std::unexpected(GemError::not_communicating);
    }
    if (!online()) {
        return std::unexpected(GemError::offline);
    }
    auto e = encoder();
    encode_wafer_report(e, next_data_id_++, lot, wafer, map);
    auto system = link.send(6, 11, true, body(e));
    if (!system) {
        return std::unexpected(GemError::link);
    }
    return *system;
}

GemResult<std::uint32_t> Equipment::report_alarm(Link& link, const Alarm& alarm) {
    if (!communicating()) {
        return std::unexpected(GemError::not_communicating);
    }
    if (!online()) {
        return std::unexpected(GemError::offline);
    }
    auto e = encoder();
    encode_alarm(e, alarm);
    auto system = link.send(5, 1, true, body(e));
    if (!system) {
        return std::unexpected(GemError::link);
    }
    return *system;
}

} // namespace waferedge::gem
