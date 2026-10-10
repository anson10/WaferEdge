// Replays every checked-in fuzz input through the fuzz target's body, so a crash the fuzzer
// once found stays a regression test in every build (not only the fuzz preset).
#include "gem_messages_target.hpp"
#include "hsms_frames_target.hpp"
#include "secs_item_target.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace {

std::size_t replay(const std::filesystem::path& dir,
                   int (*target)(const std::uint8_t*, std::size_t)) {
    std::size_t files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(entry.path(), std::ios::binary);
        const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                              std::istreambuf_iterator<char>());
        INFO(entry.path().filename().string());
        CHECK(target(bytes.data(), bytes.size()) == 0);
        ++files;
    }
    return files;
}

} // namespace

TEST_CASE("the SECS-II item fuzz corpus replays cleanly") {
    CHECK(replay(WAFEREDGE_SOURCE_DIR "/fuzz/corpus/secs_item", waferedge::fuzz::secs_item) > 0);
}

TEST_CASE("the HSMS framing fuzz corpus replays cleanly", "[hsms]") {
    CHECK(replay(WAFEREDGE_SOURCE_DIR "/fuzz/corpus/hsms_frames", waferedge::fuzz::hsms_frames) >
          0);
}

TEST_CASE("the GEM fuzz corpus replays cleanly", "[gem]") {
    CHECK(replay(WAFEREDGE_SOURCE_DIR "/fuzz/corpus/gem_messages", waferedge::fuzz::gem_messages) >
          0);
}
