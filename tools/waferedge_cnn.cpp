// FabEye's CNN on WaferEdge's reference forward pass, or on the GPU engine.
//
//   waferedge-cnn verify <model.wcnn> <file.wmap> <logits.bin> [--gpu fp32|fp16]
//       our preprocessing + forward pass against ONNX Runtime's logits (tools/export_cnn.py):
//       max |difference| and whether every map gets the same class
//   waferedge-cnn eval <model.wcnn> <file.wmap> [train|val|test|all] [--gpu fp32|fp16]
//   waferedge-cnn layers <model.wcnn> <file.wmap> --gpu fp32|fp16
//       time per layer for a batch of 256 maps, bias + ReLU fused into the GEMM and not
//   waferedge-cnn agree <model.wcnn> <file.wmap> [train|val|test|all]
//       GPU fp16 against GPU fp32 on every map: maps that change class, largest logit change
//       classify every map of a split: per-class F1, macro-F1, maps/s
//
// On the CPU, maps are spread over all cores (ThreadPool), one ReferenceForward per thread;
// --gpu runs the GPU engine instead (cuda build only).
#include "waferedge/classifier.hpp"
#include "waferedge/cnn.hpp"
#include "waferedge/gpu_cnn.hpp"
#include "waferedge/machine.hpp"
#include "waferedge/map_file.hpp"
#include "waferedge/thread_pool.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace waferedge;

namespace {

void print(const std::string& s) {
    std::fputs(s.c_str(), stdout);
}

int fail(const std::string& message) {
    std::fputs(std::format("error: {}\n", message).c_str(), stderr);
    return 1;
}

struct Reference {
    std::int32_t wafer_id;
    std::array<float, cnn::kClasses> logits;
};

std::optional<std::vector<Reference>> read_logits(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::array<char, 8> magic{};
    std::uint32_t count = 0;
    if (!in.read(magic.data(), magic.size()) ||
        std::string_view(magic.data(), magic.size()) != "WCNNLOGT" ||
        !in.read(reinterpret_cast<char*>(&count), sizeof count)) {
        return std::nullopt;
    }
    std::vector<Reference> refs(count);
    for (auto& r : refs) {
        if (!in.read(reinterpret_cast<char*>(&r.wafer_id), sizeof r.wafer_id) ||
            !in.read(reinterpret_cast<char*>(r.logits.data()), sizeof r.logits)) {
            return std::nullopt;
        }
    }
    return refs;
}

// --gpu fp32|fp16, if given: which engine runs the network.
using Gpu = std::optional<std::string>;

// Logits of the given maps: the GPU engine with --gpu, otherwise all CPU cores.
std::vector<float> forward_all(const cnn::Model& model, const std::vector<WaferMapView>& maps,
                               double& seconds, const Gpu& gpu) {
#if WAFEREDGE_HAS_CUDA
    if (gpu) {
        gpu::CnnEngine engine(model,
                              *gpu == "fp16" ? gpu::CnnPrecision::fp16 : gpu::CnnPrecision::fp32);
        std::vector<float> logits(maps.size() * cnn::kClasses);
        engine.run(std::span(maps).first(std::min<std::size_t>(maps.size(), 64)),
                   std::span(logits).first(std::min<std::size_t>(maps.size(), 64) *
                                           cnn::kClasses)); // warm-up
        const auto start = std::chrono::steady_clock::now();
        if (!engine.run(maps, logits)) {
            throw std::runtime_error("GPU CNN: " + engine.error());
        }
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return logits;
    }
#else
    if (gpu) {
        throw std::runtime_error("--gpu needs the cuda build");
    }
#endif
    std::vector<float> logits(maps.size() * cnn::kClasses);
    ThreadPool pool;
    std::vector<cnn::ReferenceForward> forward(pool.size());
    std::vector<std::vector<float>> input(pool.size(), std::vector<float>(cnn::kInputSize));
    const auto start = std::chrono::steady_clock::now();
    pool.for_each(
        maps.size(), 4, [&](unsigned worker, std::size_t begin, std::size_t end) noexcept {
            for (std::size_t i = begin; i < end; ++i) {
                cnn::preprocess(maps[i], input[worker]);
                forward[worker].run(model, input[worker],
                                    std::span(logits).subspan(i * cnn::kClasses, cnn::kClasses));
            }
        });
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return logits;
}

int cmd_verify(const cnn::Model& model, const MapSet& set, const std::string& logits_path,
               const Gpu& gpu) {
    const auto refs = read_logits(logits_path);
    if (!refs) {
        return fail(std::format("{}: not a logits file", logits_path));
    }
    std::unordered_map<std::int32_t, WaferMapView> by_id;
    for (const auto& r : set.records()) {
        by_id.emplace(r.wafer_id, r.map);
    }
    std::vector<WaferMapView> maps;
    for (const auto& r : *refs) {
        const auto it = by_id.find(r.wafer_id);
        if (it == by_id.end()) {
            return fail(std::format("wafer {} is not in the map file", r.wafer_id));
        }
        maps.push_back(it->second);
    }
    double seconds = 0;
    const auto logits = forward_all(model, maps, seconds, gpu);
    double max_diff = 0;
    double sum_diff = 0;
    std::size_t same_class = 0;
    for (std::size_t i = 0; i < refs->size(); ++i) {
        const auto ours = std::span(logits).subspan(i * cnn::kClasses, cnn::kClasses);
        for (int k = 0; k < cnn::kClasses; ++k) {
            const double d = std::abs(static_cast<double>(ours[static_cast<std::size_t>(k)]) -
                                      (*refs)[i].logits[static_cast<std::size_t>(k)]);
            max_diff = std::max(max_diff, d);
            sum_diff += d;
        }
        same_class += cnn::predicted(ours) == cnn::predicted((*refs)[i].logits) ? 1U : 0U;
    }
    print(std::format(
        "{} maps vs ONNX Runtime: max |logit difference| {:.2e}, mean {:.2e}; same "
        "class on {} of {}; {:.0f} maps/s ({})\n",
        refs->size(), max_diff, sum_diff / static_cast<double>(refs->size() * cnn::kClasses),
        same_class, refs->size(), static_cast<double>(refs->size()) / seconds,
        gpu ? "GPU " + *gpu : std::format("CPU reference, {} threads", ThreadPool().size())));
    return same_class == refs->size() ? 0 : 1;
}

int cmd_eval(const cnn::Model& model, const MapSet& set, std::optional<Split> split,
             const Gpu& gpu) {
    std::vector<WaferMapView> maps;
    std::vector<Pattern> truth;
    for (const auto& r : set.records()) {
        if (!split || r.split == *split) {
            maps.push_back(r.map);
            truth.push_back(r.truth);
        }
    }
    double seconds = 0;
    const auto logits = forward_all(model, maps, seconds, gpu);
    std::vector<Pattern> pred;
    for (std::size_t i = 0; i < maps.size(); ++i) {
        pred.push_back(cnn::predicted(std::span(logits).subspan(i * cnn::kClasses, cnn::kClasses)));
    }
    const auto m = evaluate(truth, pred);
    print(std::format("{} maps: macro-F1 {:.3f}, accuracy {:.3f}; {:.0f} maps/s ({})\n",
                      maps.size(), m.macro_f1, m.accuracy,
                      static_cast<double>(maps.size()) / seconds,
                      gpu ? "GPU " + *gpu : "CPU reference"));
    for (std::size_t c = 0; c < kPatternCount; ++c) {
        if (m.support[c] > 0) {
            print(std::format("  {:<10} f1 {:.3f}  ({} maps)\n",
                              pattern_name(static_cast<Pattern>(c)), m.f1[c], m.support[c]));
        }
    }
    print(describe_machine());
    return 0;
}

std::vector<WaferMapView> select_maps(const MapSet& set, std::optional<Split> split) {
    std::vector<WaferMapView> maps;
    for (const auto& r : set.records()) {
        if (!split || r.split == *split) {
            maps.push_back(r.map);
        }
    }
    return maps;
}

int cmd_agree(const cnn::Model& model, const MapSet& set, std::optional<Split> split) {
    const auto maps = select_maps(set, split);
    double seconds = 0;
    const auto full = forward_all(model, maps, seconds, Gpu{"fp32"});
    const auto half = forward_all(model, maps, seconds, Gpu{"fp16"});
    std::size_t changed = 0;
    double max_diff = 0;
    for (std::size_t i = 0; i < maps.size(); ++i) {
        const auto a = std::span(full).subspan(i * cnn::kClasses, cnn::kClasses);
        const auto b = std::span(half).subspan(i * cnn::kClasses, cnn::kClasses);
        changed += cnn::predicted(a) == cnn::predicted(b) ? 0U : 1U;
        for (std::size_t k = 0; k < cnn::kClasses; ++k) {
            max_diff = std::max(max_diff, static_cast<double>(std::abs(a[k] - b[k])));
        }
    }
    print(std::format(
        "{} maps, GPU fp16 vs GPU fp32: {} change class, max |logit difference| {:.2e}\n",
        maps.size(), changed, max_diff));
    return 0;
}

int cmd_layers(const cnn::Model& model, const MapSet& set, const Gpu& gpu) {
#if WAFEREDGE_HAS_CUDA
    if (!gpu) {
        return fail("layers needs --gpu fp32|fp16");
    }
    auto maps = select_maps(set, std::nullopt);
    maps.resize(std::min<std::size_t>(maps.size(), 256));
    std::vector<float> logits(maps.size() * cnn::kClasses);
    gpu::CnnEngine engine(model,
                          *gpu == "fp16" ? gpu::CnnPrecision::fp16 : gpu::CnnPrecision::fp32);
    engine.set_timing(true);
    std::array<gpu::CnnTimings, 2> best{}; // fused, unfused: the fastest of 5 runs per layer
    for (int fused = 0; fused < 2; ++fused) {
        engine.set_fused(fused == 0);
        for (int run = 0; run < 6; ++run) {
            if (!engine.run(maps, logits)) {
                return fail(engine.error());
            }
            const auto t = engine.last_timings();
            auto& b = best[static_cast<std::size_t>(fused)];
            for (std::size_t l = 0; l < t.conv_ms.size(); ++l) {
                b.conv_ms[l] = run == 1 ? t.conv_ms[l] : std::min(b.conv_ms[l], t.conv_ms[l]);
            }
            for (std::size_t l = 0; l < t.pool_ms.size(); ++l) {
                b.pool_ms[l] = run == 1 ? t.pool_ms[l] : std::min(b.pool_ms[l], t.pool_ms[l]);
            }
            b.preprocess_ms =
                run == 1 ? t.preprocess_ms : std::min(b.preprocess_ms, t.preprocess_ms);
            b.head_ms = run == 1 ? t.head_ms : std::min(b.head_ms, t.head_ms);
        }
    }
    print(std::format(
        "GPU {}, {} maps, fastest of 5 runs (ms):\n| Layer | Fused | Unfused |\n|---|---|---|\n",
        *gpu, maps.size()));
    print(std::format("| preprocess | {:.3f} | {:.3f} |\n", best[0].preprocess_ms,
                      best[1].preprocess_ms));
    float total[2] = {best[0].preprocess_ms + best[0].head_ms,
                      best[1].preprocess_ms + best[1].head_ms};
    for (std::size_t l = 0; l < best[0].conv_ms.size(); ++l) {
        print(std::format("| conv {} ({} -> {}) | {:.3f} | {:.3f} |\n", l + 1,
                          cnn::kConvShapes[l][0], cnn::kConvShapes[l][1], best[0].conv_ms[l],
                          best[1].conv_ms[l]));
        total[0] += best[0].conv_ms[l];
        total[1] += best[1].conv_ms[l];
        if (l % 2 == 1) {
            print(std::format("| maxpool {} | {:.3f} | {:.3f} |\n", l / 2 + 1,
                              best[0].pool_ms[l / 2], best[1].pool_ms[l / 2]));
            total[0] += best[0].pool_ms[l / 2];
            total[1] += best[1].pool_ms[l / 2];
        }
    }
    print(std::format("| head | {:.3f} | {:.3f} |\n| **GPU total** | **{:.3f}** | **{:.3f}** |\n",
                      best[0].head_ms, best[1].head_ms, total[0], total[1]));
    return 0;
#else
    (void)model;
    (void)set;
    (void)gpu;
    return fail("layers needs the cuda build");
#endif
}

int run(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    Gpu gpu;
    if (args.size() >= 2 && args[args.size() - 2] == "--gpu") {
        gpu = args.back();
        if (*gpu != "fp32" && *gpu != "fp16") {
            return fail("--gpu takes fp32 or fp16");
        }
        args.resize(args.size() - 2);
    }
    if (args.size() < 3) {
        std::fputs("usage: waferedge-cnn verify <model.wcnn> <file.wmap> <logits.bin>\n"
                   "       waferedge-cnn eval <model.wcnn> <file.wmap> [train|val|test|all]\n",
                   stderr);
        return 2;
    }
    auto model = cnn::load_model(args[1]);
    if (!model) {
        return fail(model.error());
    }
    auto set = MapSet::load(args[2]);
    if (!set) {
        return fail(set.error());
    }
    if (args[0] == "verify" && args.size() == 4) {
        return cmd_verify(*model, *set, args[3], gpu);
    }
    if (args[0] == "layers") {
        return cmd_layers(*model, *set, gpu);
    }
    if (args[0] == "agree" || args[0] == "eval") {
        std::optional<Split> split;
        if (args.size() == 4 && args[3] != "all") {
            for (auto s : {Split::train, Split::val, Split::test, Split::unsplit}) {
                if (args[3] == split_name(s)) {
                    split = s;
                }
            }
            if (!split) {
                return fail(std::format("unknown split {}", args[3]));
            }
        }
        return args[0] == "agree" ? cmd_agree(*model, *set, split)
                                  : cmd_eval(*model, *set, split, gpu);
    }
    return fail("unknown command");
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        return fail(e.what());
    }
}
