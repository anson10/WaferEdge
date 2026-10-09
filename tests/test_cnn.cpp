// FabEye's CNN, the reference path: preprocessing against OpenCV, each layer against a
// hand-checked or independent computation, the model file's checks. Runs in CI (no FabEye
// data needed); the comparison with ONNX Runtime's logits on real maps is the last test,
// which skips without the exported data (tools/export_cnn.py).
#include "support/synth.hpp"
#include "waferedge/cnn.hpp"
#include "waferedge/map_file.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

using namespace waferedge;
using Catch::Approx;

namespace {

std::vector<std::uint16_t> read_fixture() {
    std::ifstream in(WAFEREDGE_TEST_DATA "/cv2_nearest_64.bin", std::ios::binary);
    std::vector<std::uint16_t> v(512 * 64);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 2));
    REQUIRE(in);
    return v;
}

// An independent 3x3 convolution: explicit bounds checks instead of clipped row ranges.
std::vector<float> naive_conv(const std::vector<float>& in, int cin, int h, int w,
                              const std::vector<float>& weight, const std::vector<float>& bias,
                              int cout) {
    std::vector<float> out(static_cast<std::size_t>(cout * h * w));
    for (int co = 0; co < cout; ++co) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                double acc = bias[static_cast<std::size_t>(co)];
                for (int ci = 0; ci < cin; ++ci) {
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            const int yy = y + ky - 1;
                            const int xx = x + kx - 1;
                            if (yy >= 0 && yy < h && xx >= 0 && xx < w) {
                                acc += weight[static_cast<std::size_t>(
                                           ((co * cin + ci) * 3 + ky) * 3 + kx)] *
                                       in[static_cast<std::size_t>((ci * h + yy) * w + xx)];
                            }
                        }
                    }
                }
                out[static_cast<std::size_t>((co * h + y) * w + x)] =
                    static_cast<float>(std::max(acc, 0.0));
            }
        }
    }
    return out;
}

std::uint64_t fnv1a64(const std::vector<std::uint8_t>& data) {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (const auto b : data) {
        h ^= b;
        h *= 0x100000001B3ULL;
    }
    return h;
}

// A valid .wcnn image of FabEye's shapes, every weight `w`, every bias `b`.
std::vector<std::uint8_t> model_image(float w, float b) {
    std::vector<std::uint8_t> payload;
    auto tensor = [&](std::vector<std::uint32_t> dims, float value) {
        std::uint32_t header[5] = {static_cast<std::uint32_t>(dims.size()), 0, 0, 0, 0};
        std::size_t n = 1;
        for (std::size_t d = 0; d < dims.size(); ++d) {
            header[d + 1] = dims[d];
            n *= dims[d];
        }
        const auto* p = reinterpret_cast<const std::uint8_t*>(header);
        payload.insert(payload.end(), p, p + sizeof header);
        std::vector<float> values(n, value);
        const auto* v = reinterpret_cast<const std::uint8_t*>(values.data());
        payload.insert(payload.end(), v, v + n * sizeof(float));
    };
    for (const auto& [in, out] : cnn::kConvShapes) {
        tensor({static_cast<std::uint32_t>(out), static_cast<std::uint32_t>(in), 3, 3}, w);
        tensor({static_cast<std::uint32_t>(out)}, b);
    }
    tensor({cnn::kClasses, 256}, w);
    tensor({cnn::kClasses}, b);
    std::vector<std::uint8_t> image(8 + 4 + 4 + 8 + 32);
    std::memcpy(image.data(), "WAFERCNN", 8);
    const std::uint32_t version = 1;
    const std::uint32_t count = 18;
    const std::uint64_t checksum = fnv1a64(payload);
    std::memcpy(image.data() + 8, &version, 4);
    std::memcpy(image.data() + 12, &count, 4);
    std::memcpy(image.data() + 16, &checksum, 8);
    image.insert(image.end(), payload.begin(), payload.end());
    return image;
}

} // namespace

TEST_CASE("nearest_source reproduces cv2.resize INTER_NEAREST for every length 1..512") {
    const auto fixture = read_fixture();
    std::size_t differ = 0;
    for (int n = 1; n <= 512; ++n) {
        for (int i = 0; i < cnn::kSide; ++i) {
            differ +=
                cnn::nearest_source(i, n) == fixture[static_cast<std::size_t>((n - 1) * 64 + i)]
                    ? 0U
                    : 1U;
        }
    }
    CHECK(differ == 0);
    // Where the obvious integer formula goes wrong, ours doesn't.
    int naive_wrong = 0;
    for (int n = 1; n <= 512; ++n) {
        for (int i = 0; i < 64; ++i) {
            naive_wrong +=
                std::min(i * n / 64, n - 1) != fixture[static_cast<std::size_t>((n - 1) * 64 + i)]
                    ? 1
                    : 0;
        }
    }
    CHECK(naive_wrong > 0);
}

TEST_CASE("preprocess: one-hot, every pixel in exactly one channel, fail bins merged") {
    WaferMap map(2, 3);
    const std::uint8_t bins[] = {0, 1, 2, 5, 1, 0};
    for (int i = 0; i < 6; ++i) {
        map.set(i / 3, i % 3, bins[i]);
    }
    std::vector<float> x(cnn::kInputSize);
    cnn::preprocess(map, x);
    for (int y = 0; y < 64; ++y) {
        for (int c = 0; c < 64; ++c) {
            const auto at = [&](int ch) {
                return x[static_cast<std::size_t>((ch * 64 + y) * 64 + c)];
            };
            REQUIRE(at(0) + at(1) + at(2) == 1.0F);
            const std::uint8_t bin = map.at(cnn::nearest_source(y, 2), cnn::nearest_source(c, 3));
            CHECK(at(bin == 0 ? 0 : bin == 1 ? 1 : 2) == 1.0F);
        }
    }
    CHECK(x[static_cast<std::size_t>(2 * 4096 + 63 * 64 + 0)] == 1.0F); // bin 5 at (1, 0): fail
}

TEST_CASE("conv3x3_relu by hand: ones everywhere count the in-bounds taps") {
    const std::vector<float> in(9, 1.0F);
    const std::vector<float> weight(9, 1.0F);
    std::vector<float> out(9);
    cnn::conv3x3_relu(in, 1, 3, 3, weight, std::vector<float>{0.0F}, 1, out);
    CHECK(out == std::vector<float>{4, 6, 4, 6, 9, 6, 4, 6, 4});
    cnn::conv3x3_relu(in, 1, 3, 3, weight, std::vector<float>{-5.0F}, 1, out);
    CHECK(out == std::vector<float>{0, 1, 0, 1, 4, 1, 0, 1, 0}); // ReLU clamps the corners
}

TEST_CASE("conv3x3_relu equals an independent convolution on random data") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1, 1); // values only, not compared across platforms
    for (const auto& [cin, cout, h, w] :
         {std::array{3, 5, 7, 9}, std::array{4, 2, 1, 1}, std::array{2, 3, 16, 5}}) {
        std::vector<float> in(static_cast<std::size_t>(cin * h * w));
        std::vector<float> weight(static_cast<std::size_t>(cout * cin * 9));
        std::vector<float> bias(static_cast<std::size_t>(cout));
        for (auto* v : {&in, &weight, &bias}) {
            for (auto& x : *v) {
                x = u(rng);
            }
        }
        std::vector<float> out(static_cast<std::size_t>(cout * h * w));
        cnn::conv3x3_relu(in, cin, h, w, weight, bias, cout, out);
        const auto expected = naive_conv(in, cin, h, w, weight, bias, cout);
        for (std::size_t i = 0; i < out.size(); ++i) {
            CHECK(out[i] == Approx(expected[i]).margin(1e-5));
        }
    }
}

TEST_CASE("maxpool2, softmax and the predicted class") {
    const std::vector<float> in = {1, 2, 5,  0,  //
                                   3, 4, 1,  1,  //
                                   0, 0, -1, -2, //
                                   0, 9, -3, -4};
    std::vector<float> out(4);
    cnn::maxpool2(in, 1, 4, 4, out);
    CHECK(out == std::vector<float>{4, 5, 9, -1});

    const std::array<float, 9> logits = {1000, 999, 0, 0, 0, 0, 0, 0, 0}; // large: needs stability
    const auto p = cnn::softmax(logits);
    CHECK(std::accumulate(p.begin(), p.end(), 0.0F) == Approx(1.0F));
    CHECK(p[0] == Approx(1.0 / (1.0 + std::exp(-1.0))).epsilon(1e-5));
    CHECK(cnn::predicted(logits) == Pattern::none);
}

TEST_CASE("model files: valid ones load, damaged ones are refused with a reason") {
    const auto good = model_image(0.0F, 0.25F);
    const auto m = cnn::parse_model(good);
    REQUIRE(m.has_value());
    CHECK(m->conv[7].in == 256);
    CHECK(m->fc_bias[3] == 0.25F);
    // All-zero weights: every activation is ReLU(bias); the logits are the fc biases.
    std::vector<float> x(cnn::kInputSize, 1.0F);
    std::array<float, 9> logits{};
    cnn::ReferenceForward f;
    f.run(*m, x, logits);
    CHECK(logits ==
          std::array<float, 9>{0.25F, 0.25F, 0.25F, 0.25F, 0.25F, 0.25F, 0.25F, 0.25F, 0.25F});

    auto flipped = good;
    flipped.back() ^= 1;
    CHECK(cnn::parse_model(flipped).error().find("checksum") != std::string::npos);
    auto truncated = good;
    truncated.resize(truncated.size() - 4);
    CHECK_FALSE(cnn::parse_model(truncated).has_value());
    auto magic = good;
    magic[0] = 'X';
    CHECK_FALSE(cnn::parse_model(magic).has_value());
    CHECK_FALSE(cnn::load_model("/nonexistent.wcnn").has_value());
}

TEST_CASE("the reference forward pass matches ONNX Runtime's logits on real maps") {
    const std::filesystem::path data = WAFEREDGE_SOURCE_DIR "/data";
    if (!std::filesystem::exists(data / "fabeye_cnn.wcnn") ||
        !std::filesystem::exists(data / "fabeye_logits_waferlens.bin")) {
        SKIP("no exported FabEye data (python3 tools/export_cnn.py --out data)");
    }
    const auto model = cnn::load_model(data / "fabeye_cnn.wcnn");
    REQUIRE(model.has_value());
    auto set = MapSet::load(data / "waferlens_demo.wmap");
    REQUIRE(set.has_value());
    std::ifstream in(data / "fabeye_logits_waferlens.bin", std::ios::binary);
    char magic[8];
    std::uint32_t count = 0;
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(&count), 4);
    cnn::ReferenceForward forward;
    std::vector<float> x(cnn::kInputSize);
    std::array<float, 9> logits{};
    double max_diff = 0;
    // A few maps: this runs in Debug and sanitizer builds too; waferedge-cnn verify checks all
    // 1,500 on the release build.
    for (std::uint32_t k = 0; k < std::min<std::uint32_t>(count, 8); ++k) {
        std::int32_t wafer_id = 0;
        std::array<float, 9> expected{};
        in.read(reinterpret_cast<char*>(&wafer_id), 4);
        in.read(reinterpret_cast<char*>(expected.data()), sizeof expected);
        const auto it = std::ranges::find(set->records(), wafer_id, &WaferRecord::wafer_id);
        REQUIRE(it != set->records().end());
        cnn::preprocess(it->map, x);
        forward.run(*model, x, logits);
        for (std::size_t c = 0; c < 9; ++c) {
            max_diff = std::max(max_diff, static_cast<double>(std::abs(logits[c] - expected[c])));
        }
        CHECK(cnn::predicted(logits) == cnn::predicted(expected));
    }
    CHECK(max_diff < 1e-4);
}
