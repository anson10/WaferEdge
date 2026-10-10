#pragma once

#include "waferedge/emulator/replay.hpp"
#include "waferedge/gem/equipment.hpp"
#include "waferedge/gem/link.hpp"
#include "waferedge/hsms/session.hpp"

#include <asio/awaitable.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/steady_timer.hpp>

#include <functional>
#include <optional>
#include <string_view>

// The tool emulator: passive HSMS equipment that replays wafer maps as S6F11 wafer reports
// on Replay's constant-rate schedule and honours S2F41 HOLD / RELEASE.
//
//   hsms::Session (passive) --events--> gem::Equipment --GEM events--> Tool
//   Tool's pacer coroutine: sleeps until the next wafer is due (or GEM's WAIT DELAY ends),
//   then reports every wafer whose slot has come.
//
// The operator puts the tool on-line as soon as communication is established (auto_online).
// The schedule runs while ON-LINE and pauses otherwise. Single-threaded, on the session's
// executor.
namespace waferedge::emulator {

struct ToolConfig {
    hsms::Config hsms{};        // role is forced to passive
    gem::EquipmentConfig gem{}; // initial control state: equipment off-line
    ReplayConfig replay{};
    bool auto_online = true;    // go on-line when communication is established
    bool stop_when_done = true; // separate and stop once every report is acknowledged
};

// One line per noteworthy event, for the command-line tool's log.
using Logger = std::function<void(std::string_view)>;

class Tool {
public:
    Tool(const asio::any_io_executor& executor, std::span<const WaferRecord> records,
         ToolConfig config, Logger log = {});
    Tool(const Tool&) = delete;
    Tool& operator=(const Tool&) = delete;
    Tool(Tool&&) = delete;
    Tool& operator=(Tool&&) = delete;
    ~Tool() = default;

    // Serves hosts on `acceptor` (one connection at a time) until stop(), or until every
    // wafer is reported and answered (stop_when_done). The acceptor must stay open.
    asio::awaitable<void> run(asio::ip::tcp::acceptor& acceptor);
    void stop();

    [[nodiscard]] const Replay& replay() const noexcept { return replay_; }
    [[nodiscard]] const gem::Equipment& equipment() const noexcept { return equipment_; }
    // Reports answered S6F12, and transactions that failed (T3, SxF0, Reject).
    [[nodiscard]] std::size_t acknowledged() const noexcept { return acknowledged_; }
    [[nodiscard]] std::size_t failed() const noexcept { return failed_; }
    // Times a report had to wait because HSMS's transaction window was full (backpressure).
    [[nodiscard]] std::size_t window_full() const noexcept { return window_full_; }

private:
    void on_hsms(const hsms::Event& event);
    void on_gem(const gem::Event& event);
    asio::awaitable<void> pacer();
    void send_due();
    void log(std::string_view line) const;

    ToolConfig config_;
    Replay replay_;
    gem::Equipment equipment_;
    hsms::Session session_;
    gem::SessionLink link_;
    asio::steady_timer wake_;
    Logger log_;
    std::size_t acknowledged_ = 0; // S6F12 received
    std::size_t failed_ = 0;
    std::size_t window_full_ = 0;
    // A wafer taken from the schedule but not yet sent (window full, or the link went down):
    // sent first, late, before the schedule moves on.
    std::optional<Replay::Due> blocked_;
    bool window_blocked_ = false; // blocked_ waits for a reply, not for a timer
    bool stopped_ = false;
};

} // namespace waferedge::emulator
