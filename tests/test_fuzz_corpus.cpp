// Replays every checked-in fuzz input through the fuzz target's body, so a crash the fuzzer
// once found stays a regression test in every build (not only the fuzz preset).
#include "secs_item_target.hpp"

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

TEST_CASE("the SECS-II item fuzz corpus replays cleanly") {
    const std::filesystem::path dir = WAFEREDGE_SOURCE_DIR "/fuzz/corpus/secs_item";
    std::size_t files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        std::ifstream in(entry.path(), std::ios::binary);
        const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                              std::istreambuf_iterator<char>());
        INFO(entry.path().filename().string());
        CHECK(waferedge::fuzz::secs_item(bytes.data(), bytes.size()) == 0);
        ++files;
    }
    CHECK(files > 0);
}
