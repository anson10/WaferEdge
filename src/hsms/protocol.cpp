#include "waferedge/hsms/protocol.hpp"

#include <algorithm>
#include <cstring>

namespace waferedge::hsms {

std::string_view stype_name(SType s) noexcept {
    switch (s) {
    case SType::data:
        return "data";
    case SType::select_req:
        return "Select.req";
    case SType::select_rsp:
        return "Select.rsp";
    case SType::deselect_req:
        return "Deselect.req";
    case SType::deselect_rsp:
        return "Deselect.rsp";
    case SType::linktest_req:
        return "Linktest.req";
    case SType::linktest_rsp:
        return "Linktest.rsp";
    case SType::reject_req:
        return "Reject.req";
    case SType::separate_req:
        return "Separate.req";
    }
    return "?";
}

std::string_view state_name(State s) noexcept {
    switch (s) {
    case State::not_connected:
        return "NOT CONNECTED";
    case State::not_selected:
        return "NOT SELECTED";
    case State::selected:
        return "SELECTED";
    }
    return "?";
}

std::string_view close_reason_name(CloseReason r) noexcept {
    switch (r) {
    case CloseReason::local:
        return "closed locally";
    case CloseReason::peer_separate:
        return "peer sent Separate.req";
    case CloseReason::transport:
        return "TCP connection lost";
    case CloseReason::t6_control:
        return "T6: no control response";
    case CloseReason::t7_not_selected:
        return "T7: not selected in time";
    case CloseReason::t8_intercharacter:
        return "T8: message stalled";
    case CloseReason::select_refused:
        return "Select refused";
    case CloseReason::bad_length:
        return "length below the header size";
    case CloseReason::too_long:
        return "message above the size limit";
    }
    return "?";
}

std::string_view send_error_name(SendError e) noexcept {
    switch (e) {
    case SendError::not_selected:
        return "not selected";
    case SendError::too_many_open:
        return "too many open transactions";
    case SendError::too_long:
        return "message too long";
    case SendError::control_pending:
        return "a control transaction is pending";
    case SendError::not_connected:
        return "not connected";
    }
    return "?";
}

Protocol::Protocol(Config config) : config_(config), reconnect_at_(TimePoint::min()) {}

// --- Transport ---------------------------------------------------------------------------

void Protocol::reset_connection() {
    in_begin_ = 0;
    in_end_ = 0;
    out_.clear();
    wants_close_ = false;
    for (auto& t : open_) {
        t.open = false;
    }
    control_ = Control::none;
    t7_deadline_.reset();
    linktest_at_.reset();
    closed_.reset();
}

void Protocol::on_connected(TimePoint now) {
    reset_connection();
    state_ = State::not_selected;
    ++stats_.connections;
    last_rx_ = now;
    t7_deadline_ = now + config_.t7;
    if (config_.role == Role::active) {
        start_control(Control::select, SType::select_req, now);
    }
}

void Protocol::on_disconnected(TimePoint now) {
    if (state_ == State::not_connected) {
        return; // we closed it ourselves; Closed was already reported
    }
    close_now(CloseReason::transport, now);
    out_.clear(); // nowhere to send it
}

void Protocol::on_connect_failed(TimePoint now) {
    reconnect_at_ = now + config_.t5;
}

void Protocol::close_now(CloseReason reason, TimePoint now) {
    state_ = State::not_connected;
    reconnect_at_ = now + config_.t5;
    for (auto& t : open_) {
        t.open = false;
    }
    control_ = Control::none;
    t7_deadline_.reset();
    linktest_at_.reset();
    // Unparsed input is dropped; the memory stays, so spans already handed out stay valid.
    in_begin_ = 0;
    in_end_ = 0;
    wants_close_ = true;
    closed_ = Closed{reason};
}

std::optional<Event> Protocol::take_closed() noexcept {
    std::optional<Event> e;
    if (closed_) {
        e = *closed_;
        closed_.reset();
    }
    return e;
}

std::span<std::uint8_t> Protocol::receive_buffer(std::size_t min_bytes) {
    if (in_begin_ > 0) {
        std::memmove(in_.data(), in_.data() + in_begin_, in_end_ - in_begin_);
        in_end_ -= in_begin_;
        in_begin_ = 0;
    }
    // Room for the rest of a frame whose length is known: one read can finish it. The
    // length was checked against max_message by the parser before this can be large.
    if (in_end_ >= kLengthSize) {
        const std::size_t frame = kLengthSize + read_length(in_.data());
        if (frame <= kLengthSize + config_.max_message && frame > in_end_) {
            min_bytes = std::max(min_bytes, frame - in_end_);
        }
    }
    if (in_.size() - in_end_ < min_bytes) {
        in_.resize(in_end_ + min_bytes);
    }
    return {in_.data() + in_end_, in_.size() - in_end_};
}

void Protocol::on_received(std::size_t n, TimePoint now) {
    if (state_ == State::not_connected) {
        return;
    }
    in_end_ += n;
    last_rx_ = now;
}

void Protocol::take_output(std::vector<std::uint8_t>& buffer) noexcept {
    buffer.clear();
    std::swap(buffer, out_);
}

// --- Events ------------------------------------------------------------------------------

std::optional<Event> Protocol::poll(TimePoint now) {
    if (closed_) {
        return take_closed();
    }
    if (state_ == State::not_connected) {
        return std::nullopt;
    }
    // Frames first: a reply that arrived with the deadline counts as on time.
    std::optional<Event> event;
    while (next_frame(now, event)) {
        if (closed_) {
            return take_closed();
        }
        if (event) {
            return event;
        }
    }
    event = check_timers(now);
    if (closed_) {
        return take_closed();
    }
    return event;
}

bool Protocol::partial_frame() const noexcept {
    return in_end_ > in_begin_; // whole frames are always parsed before this is asked
}

std::optional<Event> Protocol::check_timers(TimePoint now) {
    if (control_ != Control::none && now >= control_deadline_) {
        close_now(CloseReason::t6_control, now);
        return std::nullopt;
    }
    if (t7_deadline_ && now >= *t7_deadline_) {
        close_now(CloseReason::t7_not_selected, now);
        return std::nullopt;
    }
    if (partial_frame() && now >= last_rx_ + config_.t8) {
        close_now(CloseReason::t8_intercharacter, now);
        return std::nullopt;
    }
    for (auto& t : open_) {
        if (t.open && now >= t.deadline) {
            t.open = false;
            ++stats_.reply_timeouts;
            return ReplyTimeout{t.request};
        }
    }
    if (state_ == State::selected && linktest_at_ && now >= *linktest_at_ &&
        control_ == Control::none) {
        linktest_at_.reset();
        start_control(Control::linktest, SType::linktest_req, now);
        ++stats_.linktests_sent;
    }
    return std::nullopt;
}

std::optional<TimePoint> Protocol::next_deadline() const noexcept {
    if (closed_) {
        return TimePoint::min(); // poll now
    }
    if (state_ == State::not_connected) {
        return std::nullopt;
    }
    std::optional<TimePoint> next;
    const auto consider = [&next](TimePoint t) {
        if (!next || t < *next) {
            next = t;
        }
    };
    if (control_ != Control::none) {
        consider(control_deadline_);
    }
    if (t7_deadline_) {
        consider(*t7_deadline_);
    }
    if (partial_frame()) {
        consider(last_rx_ + config_.t8);
    }
    if (linktest_at_) {
        consider(*linktest_at_);
    }
    for (const auto& t : open_) {
        if (t.open) {
            consider(t.deadline);
        }
    }
    return next;
}

bool Protocol::next_frame(TimePoint now, std::optional<Event>& event) {
    event.reset();
    const std::size_t available = in_end_ - in_begin_;
    if (available < kLengthSize) {
        return false;
    }
    const std::uint8_t* p = in_.data() + in_begin_;
    const std::uint32_t length = read_length(p);
    // Checked before the frame is complete: a bad or huge length closes the connection at
    // once, without waiting for (or allocating) the bytes it announces.
    if (length < kHeaderSize) {
        close_now(CloseReason::bad_length, now);
        return true;
    }
    if (length > config_.max_message) {
        close_now(CloseReason::too_long, now);
        return true;
    }
    if (available < kLengthSize + length) {
        return false;
    }
    const Header h = read_header(p + kLengthSize);
    const std::span<const std::uint8_t> body(p + kLengthSize + kHeaderSize, length - kHeaderSize);
    in_begin_ += kLengthSize + length;
    if (in_begin_ == in_end_) {
        in_begin_ = 0; // nothing left: the next read starts at the front, no move needed
        in_end_ = 0;
    }
    event = handle(h, body, now);
    return true;
}

// --- Handling ----------------------------------------------------------------------------

std::optional<Event> Protocol::handle(const Header& h, std::span<const std::uint8_t> body,
                                      TimePoint now) {
    if (h.ptype != 0) {
        reject(h, RejectReason::ptype_not_supported);
        return std::nullopt;
    }
    if (!known_stype(h.stype)) {
        reject(h, RejectReason::stype_not_supported);
        return std::nullopt;
    }
    if (!h.is_data()) {
        return handle_control(h, now);
    }
    if (state_ != State::selected) {
        reject(h, RejectReason::entity_not_selected);
        return std::nullopt;
    }
    ++stats_.data_received;
    DataMessage m{h, body, DataMessage::Kind::primary};
    if (h.is_reply()) {
        m.kind = DataMessage::Kind::unmatched_reply;
        for (auto& t : open_) {
            if (t.open && t.request.system_bytes == h.system_bytes) {
                t.open = false;
                m.kind = DataMessage::Kind::reply;
                break;
            }
        }
    }
    return m;
}

std::optional<Event> Protocol::handle_control(const Header& h, TimePoint now) {
    const bool answers_ours = control_ != Control::none && h.system_bytes == control_system_;
    const auto respond = [&](SType s, std::uint8_t status) {
        write_frame(control_header(s, h.system_bytes, 0, status, h.session_id), {});
    };
    switch (static_cast<SType>(h.stype)) {
    case SType::select_req:
        if (state_ == State::selected) {
            respond(SType::select_rsp, static_cast<std::uint8_t>(SelectStatus::already_active));
            return std::nullopt;
        }
        respond(SType::select_rsp, static_cast<std::uint8_t>(SelectStatus::ok));
        become_selected(now);
        return Selected{};

    case SType::select_rsp:
        if (!answers_ours || control_ != Control::select) {
            reject(h, RejectReason::transaction_not_open);
            return std::nullopt;
        }
        control_ = Control::none;
        if (h.byte3 != static_cast<std::uint8_t>(SelectStatus::ok)) {
            close_now(CloseReason::select_refused, now);
            return std::nullopt;
        }
        if (state_ == State::selected) {
            return std::nullopt; // both sides selected at once; the peer's Select won
        }
        become_selected(now);
        return Selected{};

    case SType::deselect_req:
        if (state_ != State::selected) {
            respond(SType::deselect_rsp,
                    static_cast<std::uint8_t>(DeselectStatus::not_established));
            return std::nullopt;
        }
        respond(SType::deselect_rsp, static_cast<std::uint8_t>(DeselectStatus::ok));
        become_not_selected(now);
        return Deselected{};

    case SType::deselect_rsp:
        if (!answers_ours || control_ != Control::deselect) {
            reject(h, RejectReason::transaction_not_open);
            return std::nullopt;
        }
        control_ = Control::none;
        if (h.byte3 != static_cast<std::uint8_t>(DeselectStatus::ok) || state_ != State::selected) {
            return std::nullopt;
        }
        become_not_selected(now);
        return Deselected{};

    case SType::linktest_req:
        respond(SType::linktest_rsp, 0);
        return std::nullopt;

    case SType::linktest_rsp:
        if (!answers_ours || control_ != Control::linktest) {
            reject(h, RejectReason::transaction_not_open);
            return std::nullopt;
        }
        control_ = Control::none;
        if (state_ == State::selected && config_.linktest > Duration::zero()) {
            linktest_at_ = now + config_.linktest;
        }
        return std::nullopt;

    case SType::reject_req:
        ++stats_.rejects_received;
        if (answers_ours) {
            if (control_ == Control::select) {
                close_now(CloseReason::select_refused, now);
                return std::nullopt;
            }
            control_ = Control::none;
        } else {
            // A rejected data message ends its transaction: no reply will come.
            for (auto& t : open_) {
                if (t.open && t.request.system_bytes == h.system_bytes) {
                    t.open = false;
                }
            }
        }
        return Rejected{h};

    case SType::separate_req:
        close_now(CloseReason::peer_separate, now);
        return std::nullopt;

    case SType::data:
        break;
    }
    return std::nullopt; // unreachable: data and unknown STypes are handled by handle()
}

void Protocol::become_selected(TimePoint now) {
    state_ = State::selected;
    t7_deadline_.reset();
    if (config_.linktest > Duration::zero()) {
        linktest_at_ = now + config_.linktest;
    }
}

void Protocol::become_not_selected(TimePoint now) {
    state_ = State::not_selected;
    t7_deadline_ = now + config_.t7;
    linktest_at_.reset();
    for (auto& t : open_) {
        t.open = false; // no reply can arrive outside SELECTED
    }
}

// --- Sending -----------------------------------------------------------------------------

void Protocol::write_frame(const Header& h, std::span<const std::uint8_t> body) {
    const std::size_t at = out_.size();
    const auto length = static_cast<std::uint32_t>(kHeaderSize + body.size());
    out_.resize(at + kLengthSize + length);
    std::uint8_t* p = out_.data() + at;
    p[0] = static_cast<std::uint8_t>(length >> 24U);
    p[1] = static_cast<std::uint8_t>(length >> 16U);
    p[2] = static_cast<std::uint8_t>(length >> 8U);
    p[3] = static_cast<std::uint8_t>(length);
    write_header(h, p + kLengthSize);
    if (!body.empty()) {
        std::memcpy(p + kLengthSize + kHeaderSize, body.data(), body.size());
    }
}

void Protocol::reject(const Header& h, RejectReason reason) {
    if (h.stype == static_cast<std::uint8_t>(SType::reject_req)) {
        return; // never reject a Reject: two broken peers would loop
    }
    const std::uint8_t about = reason == RejectReason::ptype_not_supported ? h.ptype : h.stype;
    write_frame(control_header(SType::reject_req, h.system_bytes, about,
                               static_cast<std::uint8_t>(reason), h.session_id),
                {});
    ++stats_.rejects_sent;
}

void Protocol::start_control(Control kind, SType stype, TimePoint now) {
    control_ = kind;
    control_system_ = next_system_++;
    control_deadline_ = now + config_.t6;
    write_frame(control_header(stype, control_system_), {});
}

std::expected<std::uint32_t, SendError> Protocol::send(std::uint8_t stream, std::uint8_t function,
                                                       bool reply_expected,
                                                       std::span<const std::uint8_t> body,
                                                       TimePoint now) {
    if (state_ != State::selected) {
        return std::unexpected(state_ == State::not_connected ? SendError::not_connected
                                                              : SendError::not_selected);
    }
    if (body.size() > config_.max_message - kHeaderSize) {
        return std::unexpected(SendError::too_long);
    }
    const std::uint32_t system = next_system_++;
    const Header h = data_header(config_.session_id, stream, function, reply_expected, system);
    if (reply_expected) {
        auto* slot = std::ranges::find(open_, false, &Transaction::open);
        if (slot == open_.end()) {
            return std::unexpected(SendError::too_many_open);
        }
        *slot = Transaction{h, now + config_.t3, true};
    }
    write_frame(h, body);
    ++stats_.data_sent;
    return system;
}

std::expected<void, SendError> Protocol::reply(const Header& primary,
                                               std::span<const std::uint8_t> body) {
    if (state_ != State::selected) {
        return std::unexpected(state_ == State::not_connected ? SendError::not_connected
                                                              : SendError::not_selected);
    }
    if (body.size() > config_.max_message - kHeaderSize) {
        return std::unexpected(SendError::too_long);
    }
    write_frame(data_header(primary.session_id, primary.stream(),
                            static_cast<std::uint8_t>(primary.function() + 1), false,
                            primary.system_bytes),
                body);
    ++stats_.data_sent;
    return {};
}

std::expected<void, SendError> Protocol::abort(const Header& primary) {
    if (state_ != State::selected) {
        return std::unexpected(state_ == State::not_connected ? SendError::not_connected
                                                              : SendError::not_selected);
    }
    write_frame(data_header(primary.session_id, primary.stream(), 0, false, primary.system_bytes),
                {});
    ++stats_.data_sent;
    return {};
}

std::expected<void, SendError> Protocol::linktest(TimePoint now) {
    if (state_ == State::not_connected) {
        return std::unexpected(SendError::not_connected);
    }
    if (control_ != Control::none) {
        return std::unexpected(SendError::control_pending);
    }
    start_control(Control::linktest, SType::linktest_req, now);
    ++stats_.linktests_sent;
    return {};
}

std::expected<void, SendError> Protocol::deselect(TimePoint now) {
    if (state_ != State::selected) {
        return std::unexpected(state_ == State::not_connected ? SendError::not_connected
                                                              : SendError::not_selected);
    }
    if (control_ != Control::none) {
        return std::unexpected(SendError::control_pending);
    }
    start_control(Control::deselect, SType::deselect_req, now);
    return {};
}

void Protocol::separate(TimePoint now) {
    if (state_ == State::not_connected) {
        return;
    }
    write_frame(control_header(SType::separate_req, next_system_++), {});
    close_now(CloseReason::local, now);
}

void Protocol::close(TimePoint now) {
    if (state_ == State::not_connected) {
        return;
    }
    close_now(CloseReason::local, now);
}

} // namespace waferedge::hsms
