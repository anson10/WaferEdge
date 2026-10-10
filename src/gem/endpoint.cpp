#include "waferedge/gem/endpoint.hpp"

namespace waferedge::gem {

std::string_view comm_state_name(CommState s) noexcept {
    switch (s) {
    case CommState::disabled:
        return "DISABLED";
    case CommState::not_communicating:
        return "NOT COMMUNICATING";
    case CommState::wait_cra:
        return "WAIT CRA";
    case CommState::wait_delay:
        return "WAIT DELAY";
    case CommState::communicating:
        return "COMMUNICATING";
    }
    return "?";
}

std::string_view control_state_name(ControlState s) noexcept {
    switch (s) {
    case ControlState::equipment_offline:
        return "EQUIPMENT OFF-LINE";
    case ControlState::attempt_online:
        return "ATTEMPT ON-LINE";
    case ControlState::host_offline:
        return "HOST OFF-LINE";
    case ControlState::online_local:
        return "ON-LINE LOCAL";
    case ControlState::online_remote:
        return "ON-LINE REMOTE";
    }
    return "?";
}

Endpoint::Endpoint(hsms::Duration establish_delay, bool enabled)
    : delay_(establish_delay), enabled_(enabled),
      comm_(enabled ? CommState::not_communicating : CommState::disabled) {}

void Endpoint::push(const Event& event) noexcept {
    // Full only if the app stops polling; the oldest event is then overwritten.
    queue_[(head_ + count_) % queue_.size()] = event;
    if (count_ < queue_.size()) {
        ++count_;
    } else {
        head_ = (head_ + 1) % queue_.size();
    }
}

std::optional<Event> Endpoint::poll() noexcept {
    if (count_ == 0) {
        return std::nullopt;
    }
    Event e = queue_[head_];
    head_ = (head_ + 1) % queue_.size();
    --count_;
    return e;
}

std::optional<hsms::TimePoint> Endpoint::next_deadline() const noexcept {
    if (comm_ == CommState::wait_delay) {
        return retry_at_;
    }
    return std::nullopt;
}

void Endpoint::tick(Link& link, hsms::TimePoint now) {
    if (comm_ == CommState::wait_delay && now >= retry_at_) {
        send_s1f13(link, now);
    }
}

void Endpoint::send_s1f13(Link& link, hsms::TimePoint now) {
    auto e = encoder();
    encode_identity(e, identity());
    auto system = link.send(1, 13, true, body(e));
    if (!system) {
        comm_ = CommState::not_communicating; // the link isn't selected: wait for it
        s1f13_system_.reset();
        return;
    }
    (void)now;
    s1f13_system_ = *system;
    comm_ = CommState::wait_cra;
}

void Endpoint::wait_delay(hsms::TimePoint now) {
    s1f13_system_.reset();
    comm_ = CommState::wait_delay;
    retry_at_ = now + delay_;
}

bool Endpoint::communication(const hsms::Event& event, Link& link, hsms::TimePoint now) {
    if (std::holds_alternative<hsms::Selected>(event)) {
        if (enabled_ && comm_ != CommState::communicating) {
            send_s1f13(link, now);
        }
        return true;
    }
    if (std::holds_alternative<hsms::Deselected>(event) ||
        std::holds_alternative<hsms::Closed>(event)) {
        const bool was = comm_ == CommState::communicating;
        comm_ = enabled_ ? CommState::not_communicating : CommState::disabled;
        s1f13_system_.reset();
        if (was) {
            push(NotCommunicating{});
        }
        return false; // the derived class may care too (a pending S1F1, open commands)
    }
    if (const auto* timeout = std::get_if<hsms::ReplyTimeout>(&event)) {
        if (timeout->request.system_bytes == s1f13_system_ && comm_ == CommState::wait_cra) {
            wait_delay(now);
            return true;
        }
        return false;
    }
    if (const auto* rejected = std::get_if<hsms::Rejected>(&event)) {
        if (rejected->reject.system_bytes == s1f13_system_ && comm_ == CommState::wait_cra) {
            wait_delay(now);
            return true;
        }
        return false;
    }
    const auto* m = std::get_if<hsms::DataMessage>(&event);
    if (m == nullptr) {
        return false;
    }
    const hsms::Header& h = m->header;
    if (m->kind == hsms::DataMessage::Kind::reply && h.system_bytes == s1f13_system_) {
        s1f13_system_.reset();
        if (comm_ != CommState::wait_cra) {
            return true; // already communicating (both sides sent S1F13)
        }
        auto ack = secs::decode(m->body).and_then([](secs::ItemView v) { return decode_s1f14(v); });
        if (h.function() == 14 && ack && ack->ack == Commack::accepted) {
            comm_ = CommState::communicating;
            push(Communicating{});
        } else {
            wait_delay(now); // denied, aborted (S1F0) or malformed
        }
        return true;
    }
    if (m->kind != hsms::DataMessage::Kind::primary) {
        return !communicating(); // replies outside COMMUNICATING are dropped
    }
    if (h.stream() == 1 && h.function() == 13) {
        auto e = encoder();
        encode_s1f14(e, enabled_ ? Commack::accepted : Commack::denied,
                     enabled_ ? identity() : nullptr);
        (void)link.reply(h, body(e));
        if (enabled_ && comm_ != CommState::communicating) {
            comm_ = CommState::communicating;
            push(Communicating{});
        }
        return true;
    }
    if (!communicating()) {
        if (h.reply_expected()) {
            (void)link.abort(h);
        }
        return true;
    }
    return false;
}

} // namespace waferedge::gem
