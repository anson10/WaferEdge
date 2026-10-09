#include "waferedge/wafer_map.hpp"

namespace waferedge {

std::string_view pattern_name(Pattern p) noexcept {
    switch (p) {
    case Pattern::none:
        return "none";
    case Pattern::center:
        return "center";
    case Pattern::donut:
        return "donut";
    case Pattern::edge_loc:
        return "edge_loc";
    case Pattern::edge_ring:
        return "edge_ring";
    case Pattern::loc:
        return "loc";
    case Pattern::near_full:
        return "near_full";
    case Pattern::random:
        return "random";
    case Pattern::scratch:
        return "scratch";
    case Pattern::unknown:
        return "unknown";
    }
    return "unknown";
}

std::string_view split_name(Split s) noexcept {
    switch (s) {
    case Split::train:
        return "train";
    case Split::val:
        return "val";
    case Split::test:
        return "test";
    case Split::unsplit:
        return "unsplit";
    }
    return "unsplit";
}

WaferMap::WaferMap(int rows, int cols, std::uint8_t fill)
    : bins_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), fill), rows_(rows),
      cols_(cols) {}

} // namespace waferedge
