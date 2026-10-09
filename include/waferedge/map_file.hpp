#pragma once

#include "waferedge/wafer_map.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

// The .wmap file: many wafer maps with their ids and ground truth, written by
// tools/export_maps.py (from WaferLens Parquet or WM-811K) and read here. Layout, all
// little-endian (ADR-0004):
//
//   header   32 bytes   "WAFERMAP", u32 version (1), u32 record size (32),
//                       u64 record count, u64 bin bytes
//   records  32 bytes each
//            i32 wafer_id, i32 lot_id, i64 tested_at (us since the Unix epoch, 0 if unknown),
//            u64 offset of the map's bins in the bin section,
//            u16 rows, u16 cols, u8 truth (Pattern), u8 split (Split), u16 reserved (0)
//   bins     u8 per die, row-major, maps back to back
namespace waferedge {

struct WaferRecord {
    std::int32_t wafer_id = 0;
    std::int32_t lot_id = 0;
    std::int64_t tested_at_us = 0;
    Pattern truth = Pattern::unknown;
    Split split = Split::unsplit;
    WaferMapView map;
};

// A loaded file: one buffer holding every map, and records whose views point into it.
// Movable, not copyable (a copy's views would point into the original's buffer).
class MapSet {
public:
    [[nodiscard]] static std::expected<MapSet, std::string> load(const std::filesystem::path& path);
    // Validates and indexes a file image already in memory.
    [[nodiscard]] static std::expected<MapSet, std::string> parse(std::vector<std::uint8_t> image);

    MapSet(MapSet&&) noexcept = default;
    MapSet& operator=(MapSet&&) noexcept = default;
    MapSet(const MapSet&) = delete;
    MapSet& operator=(const MapSet&) = delete;
    ~MapSet() = default;

    [[nodiscard]] std::span<const WaferRecord> records() const noexcept { return records_; }
    [[nodiscard]] std::size_t size() const noexcept { return records_.size(); }

private:
    MapSet() = default;
    std::vector<std::uint8_t> image_;
    std::vector<WaferRecord> records_;
};

// Writes records (and the maps their views point to) as a .wmap file image / file.
[[nodiscard]] std::vector<std::uint8_t> serialize(std::span<const WaferRecord> records);
[[nodiscard]] std::expected<void, std::string> save(const std::filesystem::path& path,
                                                    std::span<const WaferRecord> records);

} // namespace waferedge
