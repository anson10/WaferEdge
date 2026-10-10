// tool-emulator: passive HSMS equipment that replays a .wmap file as S6F11 wafer reports at a
// constant rate and honours S2F41 HOLD / RELEASE (docs/pipeline.md).
//
//   build/release/tools/tool-emulator data/waferlens_demo.wmap --port 5000 --rate 50
//
// Waits for a host to connect and select, establishes GEM communication, goes on-line, and
// reports one wafer per slot in tested_at order. Stops when every wafer is reported and
// answered, or on Ctrl-C, and prints what it sent and what the holds withheld.
#include "waferedge/emulator/tool.hpp"

#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/io_context.hpp>
#include <asio/signal_set.hpp>

#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string_view>

using namespace waferedge;

namespace {

template <typename T>
bool parse(std::string_view text, T& out) {
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), out);
    return ec == std::errc{} && end == text.data() + text.size();
}

// libc++ 18 has no floating-point from_chars.
bool parse(const char* text, double& out) {
    char* end = nullptr;
    out = std::strtod(text, &end);
    return end != text && *end == '\0';
}

// std::println is GCC 14 (libc++ 18 has it, so this can't share its name).
template <typename... Args>
void say(std::FILE* to, std::format_string<Args...> fmt, Args&&... args) {
    std::fputs(std::format(fmt, std::forward<Args>(args)...).c_str(), to);
    std::fputc('\n', to);
}

int usage() {
    std::fputs(
        "usage: tool-emulator <maps.wmap> [--port N] [--rate WAFERS_PER_S] [--limit N]\n"
        "                    [--device-id N] [--keep-running]\n"
        "  --port         TCP port to listen on (default 5000)\n"
        "  --rate         wafers per second, constant-rate schedule (default 10; 0: no pacing)\n"
        "  --limit        replay only the first N wafers in tested_at order\n"
        "  --device-id    HSMS session id of the data messages (default 0)\n"
        "  --keep-running don't stop when every wafer is answered\n",
        stderr);
    return 2;
}

double seconds_since(hsms::TimePoint t) {
    return std::chrono::duration<double>(hsms::Clock::now() - t).count();
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return usage();
    }
    const std::string_view path = argv[1];
    unsigned short port = 5000;
    emulator::ToolConfig config;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--keep-running") {
            config.stop_when_done = false;
        } else if (arg == "--port" && has_value && parse(argv[i + 1], port)) {
            ++i;
        } else if (arg == "--rate" && has_value && parse(argv[i + 1], config.replay.rate)) {
            ++i;
        } else if (arg == "--limit" && has_value && parse(argv[i + 1], config.replay.limit)) {
            ++i;
        } else if (arg == "--device-id" && has_value &&
                   parse(argv[i + 1], config.hsms.session_id)) {
            ++i;
        } else {
            return usage();
        }
    }

    auto maps = MapSet::load(path);
    if (!maps) {
        say(stderr, "tool-emulator: {}", maps.error());
        return 1;
    }
    say(stdout, "tool-emulator: {} wafers from {}, {} wafers/s, listening on port {}",
        config.replay.limit > 0 ? std::min(config.replay.limit, maps->size()) : maps->size(), path,
        config.replay.rate, port);

    asio::io_context io(1);
    asio::ip::tcp::acceptor acceptor(io, {asio::ip::tcp::v4(), port});
    const auto start = hsms::Clock::now();
    emulator::Tool tool(io.get_executor(), maps->records(), config, [&](std::string_view line) {
        say(stdout, "[{:9.3f} s] {}", seconds_since(start), line);
    });
    asio::signal_set signals(io, SIGINT, SIGTERM);
    signals.async_wait([&](const asio::error_code& ec, int) {
        if (!ec) {
            tool.stop();
        }
    });
    asio::co_spawn(io, tool.run(acceptor), [&](const std::exception_ptr& e) {
        signals.cancel();
        if (e) {
            std::rethrow_exception(e);
        }
    });
    io.run();

    const auto& stats = tool.replay().stats();
    say(stdout,
        "sent {} wafers ({} acknowledged, {} failed), withheld {} by {} hold(s), "
        "{} release(s), {:.1f} s",
        stats.sent, tool.acknowledged(), tool.failed(), stats.withheld, stats.holds, stats.releases,
        seconds_since(start));
    return 0;
}
