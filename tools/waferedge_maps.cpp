// Summarises a .wmap file and times the scalar features over all of it.
//
//   waferedge-maps data/waferlens_demo.wmap
//
// The timing is wall time on a steady clock over whole passes of the set (geometry tables
// built beforehand), repeated until at least a second has passed; maps/s is total maps over
// total time.
#include "waferedge/features.hpp"
#include "waferedge/machine.hpp"
#include "waferedge/map_file.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <format>
#include <map>
#include <string>
#include <utility>

using namespace waferedge;

namespace {

void print(const std::string& s) {
    std::fputs(s.c_str(), stdout);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fputs("usage: waferedge-maps <file.wmap>\n", stderr);
        return 2;
    }
    auto set = MapSet::load(argv[1]);
    if (!set) {
        std::fputs(std::format("error: {}\n", set.error()).c_str(), stderr);
        return 1;
    }
    const auto records = set->records();

    std::array<std::array<std::size_t, 4>, kPatternCount + 1> by_pattern{}; // [pattern][split]
    std::map<std::pair<int, int>, std::size_t> by_shape;
    std::size_t dies = 0;
    for (const auto& r : records) {
        const auto p =
            r.truth == Pattern::unknown ? kPatternCount : static_cast<std::size_t>(r.truth);
        ++by_pattern[p][static_cast<std::size_t>(r.split)];
        ++by_shape[{r.map.rows(), r.map.cols()}];
        dies += r.map.bins().size();
    }
    print(std::format("{}: {} maps, {} grid positions, {} shapes\n", argv[1], records.size(), dies,
                      by_shape.size()));
    print(std::format("{:<10} {:>8} {:>8} {:>8} {:>8}\n", "truth", "train", "val", "test",
                      "unsplit"));
    for (std::size_t p = 0; p <= kPatternCount; ++p) {
        const auto& c = by_pattern[p];
        if (c[0] + c[1] + c[2] + c[3] == 0) {
            continue;
        }
        const auto name = p == kPatternCount ? "unknown" : pattern_name(static_cast<Pattern>(p));
        print(std::format("{:<10} {:>8} {:>8} {:>8} {:>8}\n", name, c[0], c[1], c[2], c[3]));
    }
    int shown = 0;
    print("largest shape groups:");
    std::multimap<std::size_t, std::pair<int, int>, std::greater<>> largest;
    for (const auto& [shape, n] : by_shape) {
        largest.emplace(n, shape);
    }
    for (const auto& [n, shape] : largest) {
        if (shown++ == 5) {
            break;
        }
        print(std::format(" {}x{} ({})", shape.first, shape.second, n));
    }
    print("\n");

    GeometryCache cache;
    for (const auto& r : records) {
        (void)cache.get(r.map.rows(), r.map.cols());
    }
    using clock = std::chrono::steady_clock;
    std::uint64_t checksum = 0;
    std::size_t passes = 0;
    const auto start = clock::now();
    auto elapsed = clock::duration{};
    do {
        for (const auto& r : records) {
            const auto f = compute_features(r.map, cache.get(r.map.rows(), r.map.cols()));
            checksum += f.fails;
        }
        ++passes;
        elapsed = clock::now() - start;
    } while (elapsed < std::chrono::seconds(1));
    const double seconds = std::chrono::duration<double>(elapsed).count();
    const auto maps = static_cast<double>(passes * records.size());
    print(std::format("scalar features: {:.0f} maps/s, {:.0f} ns/map ({} passes, {:.2f} s, "
                      "checksum {})\n",
                      maps / seconds, seconds * 1e9 / maps, passes, seconds, checksum / passes));
    print(describe_machine());
    return 0;
}
