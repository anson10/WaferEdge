#pragma once

#include "waferedge/hsms/protocol.hpp"
#include "waferedge/hsms/session.hpp"

#include <cstdint>
#include <expected>
#include <span>

// How the GEM layer sends: data messages over an HSMS connection. Equipment and Host never
// touch HSMS directly, so the same GEM code runs on the fake-clock Protocol in tests and on a
// TCP Session in the emulator and the edge host (ADR-0011).
namespace waferedge::gem {

using hsms::SendError;

class Link {
public:
    Link() = default;
    Link(const Link&) = delete;
    Link& operator=(const Link&) = delete;
    Link(Link&&) = delete;
    Link& operator=(Link&&) = delete;
    virtual ~Link() = default;

    // A primary; returns its system bytes (replies and T3 timeouts carry them).
    virtual std::expected<std::uint32_t, SendError> send(std::uint8_t stream, std::uint8_t function,
                                                         bool reply_expected,
                                                         std::span<const std::uint8_t> body) = 0;
    virtual std::expected<void, SendError> reply(const hsms::Header& primary,
                                                 std::span<const std::uint8_t> body) = 0;
    virtual std::expected<void, SendError> abort(const hsms::Header& primary) = 0;
};

// A Protocol and the caller's clock: the tests' link.
class ProtocolLink final : public Link {
public:
    ProtocolLink(hsms::Protocol& protocol, const hsms::TimePoint& now) noexcept
        : protocol_(protocol), now_(now) {}
    std::expected<std::uint32_t, SendError> send(std::uint8_t stream, std::uint8_t function,
                                                 bool reply_expected,
                                                 std::span<const std::uint8_t> body) override {
        return protocol_.send(stream, function, reply_expected, body, now_);
    }
    std::expected<void, SendError> reply(const hsms::Header& primary,
                                         std::span<const std::uint8_t> body) override {
        return protocol_.reply(primary, body);
    }
    std::expected<void, SendError> abort(const hsms::Header& primary) override {
        return protocol_.abort(primary);
    }

private:
    hsms::Protocol& protocol_;
    const hsms::TimePoint& now_; // the test advances it
};

// A TCP session.
class SessionLink final : public Link {
public:
    explicit SessionLink(hsms::Session& session) noexcept : session_(session) {}
    std::expected<std::uint32_t, SendError> send(std::uint8_t stream, std::uint8_t function,
                                                 bool reply_expected,
                                                 std::span<const std::uint8_t> body) override {
        return session_.send(stream, function, reply_expected, body);
    }
    std::expected<void, SendError> reply(const hsms::Header& primary,
                                         std::span<const std::uint8_t> body) override {
        return session_.reply(primary, body);
    }
    std::expected<void, SendError> abort(const hsms::Header& primary) override {
        return session_.abort(primary);
    }

private:
    hsms::Session& session_;
};

} // namespace waferedge::gem
