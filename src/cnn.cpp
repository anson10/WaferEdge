#include "waferedge/cnn.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <string_view>

namespace waferedge::cnn {

static_assert(std::endian::native == std::endian::little, "the .wcnn reader assumes little-endian");

namespace {

constexpr std::string_view kMagic = "WAFERCNN";
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeader = 8 + 4 + 4 + 8 + 32;
constexpr std::size_t kTensorHeader = 4 + 4 * 4;

std::uint64_t fnv1a64(std::span<const std::uint8_t> data) noexcept {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (const auto b : data) {
        h ^= b;
        h *= 0x100000001B3ULL;
    }
    return h;
}

template <typename T>
T read_at(const std::uint8_t* p) noexcept {
    T v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

} // namespace

std::expected<Model, std::string> parse_model(std::span<const std::uint8_t> image) {
    if (image.size() < kHeader || std::memcmp(image.data(), kMagic.data(), kMagic.size()) != 0) {
        return std::unexpected("not a .wcnn file");
    }
    if (read_at<std::uint32_t>(image.data() + 8) != kVersion) {
        return std::unexpected("unsupported .wcnn version");
    }
    const auto count = read_at<std::uint32_t>(image.data() + 12);
    const auto checksum = read_at<std::uint64_t>(image.data() + 16);
    const auto payload = image.subspan(kHeader);
    if (fnv1a64(payload) != checksum) {
        return std::unexpected("checksum mismatch: the file is damaged");
    }
    if (count != 2 * kConvLayers + 2) {
        return std::unexpected(
            std::format("expected {} tensors, found {}", 2 * kConvLayers + 2, count));
    }

    // The shapes the architecture needs, in file order: (weight, bias) per conv, then the
    // linear layer's.
    std::vector<std::vector<std::uint32_t>> want;
    for (const auto& [in, out] : kConvShapes) {
        want.push_back({static_cast<std::uint32_t>(out), static_cast<std::uint32_t>(in), 3, 3});
        want.push_back({static_cast<std::uint32_t>(out)});
    }
    want.push_back({kClasses, 256});
    want.push_back({kClasses});

    std::vector<std::vector<float>> tensors;
    std::size_t at = 0;
    for (std::size_t t = 0; t < count; ++t) {
        if (payload.size() - at < kTensorHeader) {
            return std::unexpected(std::format("tensor {}: truncated", t));
        }
        const auto ndim = read_at<std::uint32_t>(payload.data() + at);
        std::vector<std::uint32_t> dims;
        std::size_t elements = 1;
        for (std::uint32_t d = 0; d < std::min<std::uint32_t>(ndim, 4); ++d) {
            dims.push_back(read_at<std::uint32_t>(payload.data() + at + 4 + std::size_t{4} * d));
            elements *= dims.back();
        }
        if (dims != want[t]) {
            return std::unexpected(std::format("tensor {}: shape doesn't match FabEye's CNN", t));
        }
        at += kTensorHeader;
        if ((payload.size() - at) / sizeof(float) < elements) {
            return std::unexpected(std::format("tensor {}: truncated", t));
        }
        std::vector<float> values(elements);
        std::memcpy(values.data(), payload.data() + at, elements * sizeof(float));
        at += elements * sizeof(float);
        tensors.push_back(std::move(values));
    }
    if (at != payload.size()) {
        return std::unexpected("trailing bytes after the last tensor");
    }

    Model m;
    for (std::size_t i = 0; i < kConvLayers; ++i) {
        m.conv[i] = Conv{kConvShapes[i][0], kConvShapes[i][1], std::move(tensors[2 * i]),
                         std::move(tensors[2 * i + 1])};
    }
    m.fc_weight = std::move(tensors[std::size_t{2} * kConvLayers]);
    m.fc_bias = std::move(tensors[std::size_t{2} * kConvLayers + 1]);
    std::memcpy(m.onnx_sha256.data(), image.data() + 24, 32);
    return m;
}

std::expected<Model, std::string> load_model(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::unexpected(std::format("cannot open {}", path.string()));
    }
    std::vector<std::uint8_t> image((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    auto m = parse_model(image);
    if (!m) {
        return std::unexpected(std::format("{}: {}", path.string(), m.error()));
    }
    return m;
}

int nearest_source(int i, int n) noexcept {
    // Exactly OpenCV's resizeNN: inv_scale = dst / src, then floor(i * (1 / inv_scale)).
    const double inv_scale = static_cast<double>(kSide) / n;
    const double step = 1.0 / inv_scale;
    return std::min(static_cast<int>(std::floor(i * step)), n - 1);
}

void preprocess(WaferMapView map, std::span<float> out) noexcept {
    assert(out.size() == kInputSize);
    std::ranges::fill(out, 0.0F);
    std::array<int, kSide> col{};
    for (int x = 0; x < kSide; ++x) {
        col[static_cast<std::size_t>(x)] = nearest_source(x, map.cols());
    }
    for (int y = 0; y < kSide; ++y) {
        const int r = nearest_source(y, map.rows());
        for (int x = 0; x < kSide; ++x) {
            const std::uint8_t bin = map.at(r, col[static_cast<std::size_t>(x)]);
            // Channel 0 off wafer, 1 good, 2 every fail bin (FabEye maps bins >= 2 to 2).
            const std::size_t channel = std::min<std::size_t>(bin, 2);
            out[(channel * kSide + static_cast<std::size_t>(y)) * kSide +
                static_cast<std::size_t>(x)] = 1.0F;
        }
    }
}

std::array<float, kClasses> softmax(std::span<const float> logits) noexcept {
    std::array<float, kClasses> p{};
    const float top = *std::ranges::max_element(logits);
    float sum = 0;
    for (std::size_t k = 0; k < kClasses; ++k) {
        p[k] = std::exp(logits[k] - top);
        sum += p[k];
    }
    for (auto& v : p) {
        v /= sum;
    }
    return p;
}

Pattern predicted(std::span<const float> logits) noexcept {
    return static_cast<Pattern>(std::ranges::max_element(logits.first(kClasses)) - logits.begin());
}

} // namespace waferedge::cnn
