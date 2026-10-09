#include "waferedge/map_file.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <format>
#include <fstream>

namespace waferedge {

static_assert(std::endian::native == std::endian::little,
              "the .wmap reader and writer assume a little-endian host");

namespace {

constexpr std::array<char, 8> kMagic = {'W', 'A', 'F', 'E', 'R', 'M', 'A', 'P'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderSize = 32;
constexpr std::size_t kRecordSize = 32;

template <typename T>
T read_at(const std::uint8_t* p) noexcept {
    T value;
    std::memcpy(&value, p, sizeof value);
    return value;
}

template <typename T>
void write_at(std::uint8_t* p, T value) noexcept {
    std::memcpy(p, &value, sizeof value);
}

bool valid_pattern(std::uint8_t v) noexcept {
    return v < kPatternCount || v == static_cast<std::uint8_t>(Pattern::unknown);
}

} // namespace

std::expected<MapSet, std::string> MapSet::load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path.string()));
    }
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        return std::unexpected(std::format("{}: {}", path.string(), ec.message()));
    }
    std::vector<std::uint8_t> image(size);
    if (!in.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(size))) {
        return std::unexpected(std::format("{}: short read", path.string()));
    }
    return parse(std::move(image));
}

std::expected<MapSet, std::string> MapSet::parse(std::vector<std::uint8_t> image) {
    if (image.size() < kHeaderSize) {
        return std::unexpected("file shorter than the header");
    }
    const std::uint8_t* p = image.data();
    if (std::memcmp(p, kMagic.data(), kMagic.size()) != 0) {
        return std::unexpected("not a .wmap file (bad magic)");
    }
    if (const auto v = read_at<std::uint32_t>(p + 8); v != kVersion) {
        return std::unexpected(std::format("unsupported .wmap version {}", v));
    }
    if (const auto rs = read_at<std::uint32_t>(p + 12); rs != kRecordSize) {
        return std::unexpected(std::format("unexpected record size {}", rs));
    }
    const auto count = read_at<std::uint64_t>(p + 16);
    const auto bin_bytes = read_at<std::uint64_t>(p + 24);
    // Check the sizes without overflow: count and bin_bytes come from the file.
    const std::size_t after_header = image.size() - kHeaderSize;
    if (count > after_header / kRecordSize || bin_bytes != after_header - count * kRecordSize) {
        return std::unexpected("record count and bin size don't match the file size");
    }

    MapSet set;
    set.image_ = std::move(image);
    const std::uint8_t* records = set.image_.data() + kHeaderSize;
    const std::uint8_t* bins = records + count * kRecordSize;
    set.records_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t* r = records + i * kRecordSize;
        const auto offset = read_at<std::uint64_t>(r + 16);
        const auto rows = read_at<std::uint16_t>(r + 24);
        const auto cols = read_at<std::uint16_t>(r + 26);
        const auto truth = r[28];
        const auto split = r[29];
        const std::size_t dies = std::size_t{rows} * cols;
        if (offset > bin_bytes || dies > bin_bytes - offset) {
            return std::unexpected(std::format("record {}: map outside the bin section", i));
        }
        if (!valid_pattern(truth) || split > static_cast<std::uint8_t>(Split::unsplit)) {
            return std::unexpected(std::format("record {}: bad truth or split code", i));
        }
        set.records_.push_back(WaferRecord{
            .wafer_id = read_at<std::int32_t>(r),
            .lot_id = read_at<std::int32_t>(r + 4),
            .tested_at_us = read_at<std::int64_t>(r + 8),
            .truth = static_cast<Pattern>(truth),
            .split = static_cast<Split>(split),
            .map = WaferMapView({bins + offset, dies}, rows, cols),
        });
    }
    return set;
}

std::vector<std::uint8_t> serialize(std::span<const WaferRecord> records) {
    std::uint64_t bin_bytes = 0;
    for (const auto& r : records) {
        bin_bytes += r.map.bins().size();
    }
    std::vector<std::uint8_t> image(kHeaderSize + records.size() * kRecordSize + bin_bytes);
    std::uint8_t* p = image.data();
    std::memcpy(p, kMagic.data(), kMagic.size());
    write_at<std::uint32_t>(p + 8, kVersion);
    write_at<std::uint32_t>(p + 12, kRecordSize);
    write_at<std::uint64_t>(p + 16, records.size());
    write_at<std::uint64_t>(p + 24, bin_bytes);

    std::uint8_t* bins = p + kHeaderSize + records.size() * kRecordSize;
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& rec = records[i];
        std::uint8_t* r = p + kHeaderSize + i * kRecordSize;
        write_at(r, rec.wafer_id);
        write_at(r + 4, rec.lot_id);
        write_at(r + 8, rec.tested_at_us);
        write_at(r + 16, offset);
        write_at(r + 24, static_cast<std::uint16_t>(rec.map.rows()));
        write_at(r + 26, static_cast<std::uint16_t>(rec.map.cols()));
        r[28] = static_cast<std::uint8_t>(rec.truth);
        r[29] = static_cast<std::uint8_t>(rec.split);
        const auto map_bins = rec.map.bins();
        std::memcpy(bins + offset, map_bins.data(), map_bins.size());
        offset += map_bins.size();
    }
    return image;
}

std::expected<void, std::string> save(const std::filesystem::path& path,
                                      std::span<const WaferRecord> records) {
    const auto image = serialize(records);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.write(reinterpret_cast<const char*>(image.data()),
                   static_cast<std::streamsize>(image.size()))) {
        return std::unexpected(std::format("cannot write {}", path.string()));
    }
    return {};
}

} // namespace waferedge
