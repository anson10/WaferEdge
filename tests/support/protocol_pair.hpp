#pragma once

// Two hsms::Protocols wired back to back on a fake clock, the tool's passive and the host's
// active, with a GEM Link for each: the edge tests drive GEM and the edge core through it
// without sockets or threads.
#include "waferedge/gem/link.hpp"
#include "waferedge/hsms/protocol.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

namespace waferedge::testing {

struct ProtocolPair {
    hsms::TimePoint now = hsms::TimePoint{} + std::chrono::hours(1);
    hsms::Protocol tool{config(hsms::Role::passive)};
    hsms::Protocol host{config(hsms::Role::active)};
    gem::ProtocolLink tool_link{tool, now};
    gem::ProtocolLink host_link{host, now};
    std::vector<std::uint8_t> wire;

    static hsms::Config config(hsms::Role role) {
        hsms::Config c;
        c.role = role;
        return c;
    }

    void connect() {
        host.on_connected(now);
        tool.on_connected(now);
    }

    // Moves bytes both ways until both sides are quiet; each side's HSMS events go to its
    // handler (taken by value: the tests pass lambdas capturing by reference). Returns false
    // if they never went quiet.
    template <typename OnTool, typename OnHost>
    bool pump(OnTool on_tool, OnHost on_host) {
        for (int round = 0; round < 50; ++round) {
            const bool a = pass(host, tool, on_tool);
            const bool b = pass(tool, host, on_host);
            if (!a && !b) {
                return true;
            }
        }
        return false;
    }

private:
    template <typename OnEvent>
    bool pass(hsms::Protocol& from, hsms::Protocol& to, OnEvent& on_event) {
        from.take_output(wire);
        if (!wire.empty()) {
            auto buffer = to.receive_buffer(wire.size());
            std::memcpy(buffer.data(), wire.data(), wire.size());
            to.on_received(wire.size(), now);
        }
        bool any = !wire.empty();
        while (auto e = to.poll(now)) {
            on_event(*e);
            any = true;
        }
        return any;
    }
};

} // namespace waferedge::testing
