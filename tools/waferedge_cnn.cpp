// FabEye's CNN on WaferEdge's reference forward pass, or on the GPU engine.
//
//   waferedge-cnn verify <model.wcnn> <file.wmap> <logits.bin> [--gpu fp32|fp16|int8]
//       our preprocessing + forward pass against ONNX Runtime's logits (tools/export_cnn.py):
//       max |difference| and whether every map gets the same class
//   waferedge-cnn eval <model.wcnn> <file.wmap> [train|val|test|all] [--gpu fp32|fp16|int8]
//   waferedge-cnn calibrate <model.wcnn> <file.wmap> <out.txt> [maps]
//       int8 activation scales: each convolution's largest output on the first `maps` (2,048)
//       validation maps, never test (GPU fp32 engine). --gpu int8 then needs --scales <out.txt>
//   waferedge-cnn layers <model.wcnn> <file.wmap> --gpu fp32|fp16|int8
//       time per layer for a batch of 256 maps, bias + ReLU fused into the GEMM and not
//   waferedge-cnn conformal <model.wcnn> <file.wmap> <conformal.txt> [split] [--gpu ...]
//       FabEye's prediction sets and auto-accept rule: coverage, worst class, set size,
//       accept rate and error, next to FabEye's own reported numbers
//   waferedge-cnn agree <model.wcnn> <file.wmap> [train|val|test|all] [--gpu fp16|int8 ...]
//       a lower precision (fp16 by default) against GPU fp32 on every map: maps that change
//       class, largest logit change
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
#include <sstream>
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

// --gpu fp32|fp16|int8 (int8 with --scales <file> from `calibrate`): the engine to run.
struct GpuChoice {
    std::string name;
    gpu::CnnPrecision precision = gpu::CnnPrecision::fp32;
    gpu::ActivationMax scales{};
};
using Gpu = std::optional<GpuChoice>;

std::optional<gpu::ActivationMax> read_scales(const std::string& path) {
    std::ifstream in(path);
    gpu::ActivationMax m{};
    std::size_t found = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line.starts_with('#')) {
            continue;
        }
        std::istringstream f(line);
        std::size_t layer = 0;
        float value = 0;
        if (f >> layer >> value && layer >= 1 && layer <= m.size()) {
            m[layer - 1] = value;
            ++found;
        }
    }
    return found == m.size() ? std::optional(m) : std::nullopt;
}

// Logits of the given maps: the GPU engine with --gpu, otherwise all CPU cores.
std::vector<float> forward_all(const cnn::Model& model, const std::vector<WaferMapView>& maps,
                               double& seconds, const Gpu& gpu) {
#if WAFEREDGE_HAS_CUDA
    if (gpu) {
        gpu::CnnEngine engine(model, gpu->precision, &gpu->scales);
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
        gpu ? "GPU " + gpu->name : std::format("CPU reference, {} threads", ThreadPool().size())));
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
                      gpu ? "GPU " + gpu->name : "CPU reference"));
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

int cmd_agree(const cnn::Model& model, const MapSet& set, std::optional<Split> split,
              const Gpu& gpu) {
    const auto maps = select_maps(set, split);
    double seconds = 0;
    const auto full =
        forward_all(model, maps, seconds, GpuChoice{"fp32", gpu::CnnPrecision::fp32, {}});
    const GpuChoice low = gpu ? *gpu : GpuChoice{"fp16", gpu::CnnPrecision::fp16, {}};
    const auto half = forward_all(model, maps, seconds, low);
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
    print(
        std::format("{} maps, GPU {} vs GPU fp32: {} change class, max |logit difference| {:.2e}\n",
                    maps.size(), low.name, changed, max_diff));
    return 0;
}

int cmd_layers(const cnn::Model& model, const MapSet& set, const Gpu& gpu) {
#if WAFEREDGE_HAS_CUDA
    if (!gpu) {
        return fail("layers needs --gpu fp32|fp16|int8");
    }
    auto maps = select_maps(set, std::nullopt);
    maps.resize(std::min<std::size_t>(maps.size(), 256));
    std::vector<float> logits(maps.size() * cnn::kClasses);
    gpu::CnnEngine engine(model, gpu->precision, &gpu->scales);
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
        gpu->name, maps.size()));
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

int cmd_conformal(const cnn::Model& model, const MapSet& set, const std::string& path,
                  std::optional<Split> split, const Gpu& gpu) {
    const auto cal = cnn::load_conformal(path);
    if (!cal) {
        return fail(cal.error());
    }
    if (cal->model_sha256 != model.onnx_sha256) {
        return fail("the conformal calibration is for a different model");
    }
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
    const auto a10 =
        cnn::evaluate_conformal(logits, truth, cal->thresholds_10, cal->accept_confidence);
    const auto a05 =
        cnn::evaluate_conformal(logits, truth, cal->thresholds_05, cal->accept_confidence);
    print(std::format("{} maps ({}), FabEye's conformal calibration:\n", maps.size(),
                      gpu ? "GPU " + gpu->name : "CPU reference"));
    print(std::format(
        "  90% target: coverage {:.4f} (FabEye {:.4f}), worst class {:.4f} (FabEye {:.4f}), "
        "mean set size {:.3f}\n",
        a10.coverage, cal->reported_coverage_10, a10.worst_class_coverage,
        cal->reported_worst_class_10, a10.mean_set_size));
    print(std::format(
        "  95% target: coverage {:.4f} (FabEye {:.4f}), worst class {:.4f} (FabEye {:.4f}), "
        "mean set size {:.3f}\n",
        a05.coverage, cal->reported_coverage_05, a05.worst_class_coverage,
        cal->reported_worst_class_05, a05.mean_set_size));
    print(std::format(
        "  auto-accept at confidence >= {:.3f}: {:.4f} accepted (FabEye {:.4f}), error among "
        "accepted {:.4f} (FabEye {:.4f})\n",
        cal->accept_confidence, a10.accept_rate, cal->reported_accept_rate,
        a10.error_among_accepted, cal->reported_accept_error));
    return 0;
}

int cmd_calibrate(const cnn::Model& model, const MapSet& set, const std::string& out_path,
                  std::size_t count) {
#if WAFEREDGE_HAS_CUDA
    std::vector<WaferMapView> maps;
    for (const auto& r : set.records()) {
        if (r.split == Split::val && maps.size() < count) {
            maps.push_back(r.map);
        }
    }
    if (maps.empty()) {
        return fail("no validation maps in this file");
    }
    gpu::CnnEngine engine(model, gpu::CnnPrecision::fp32);
    gpu::ActivationMax m{};
    if (!engine.activation_max(maps, m)) {
        return fail(engine.error());
    }
    std::ofstream out(out_path);
    out << std::format("# int8 calibration: each convolution's largest output (after ReLU) on the "
                       "first {} validation maps\n# (waferedge-cnn calibrate). layer max\n",
                       maps.size());
    for (std::size_t l = 0; l < m.size(); ++l) {
        out << std::format("{} {}\n", l + 1, m[l]);
        print(std::format("conv {}: max {:.4f}\n", l + 1, m[l]));
    }
    print(std::format("wrote {} ({} validation maps)\n", out_path, maps.size()));
    return out ? 0 : fail("cannot write " + out_path);
#else
    (void)model;
    (void)set;
    (void)out_path;
    (void)count;
    return fail("calibrate needs the cuda build");
#endif
}

int run(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    // Trailing options: --gpu fp32|fp16|int8 and --scales <file>, in either order.
    Gpu gpu;
    std::optional<std::string> scales_path;
    while (args.size() >= 2 &&
           (args[args.size() - 2] == "--gpu" || args[args.size() - 2] == "--scales")) {
        if (args[args.size() - 2] == "--gpu") {
            const auto& name = args.back();
            if (name != "fp32" && name != "fp16" && name != "int8") {
                return fail("--gpu takes fp32, fp16 or int8");
            }
            gpu = GpuChoice{name,
                            name == "fp32" ? gpu::CnnPrecision::fp32
                                           : (name == "fp16" ? gpu::CnnPrecision::fp16
                                                             : gpu::CnnPrecision::int8),
                            {}};
        } else {
            scales_path = args.back();
        }
        args.resize(args.size() - 2);
    }
    if (gpu && gpu->precision == gpu::CnnPrecision::int8) {
        const auto scales = scales_path ? read_scales(*scales_path) : std::nullopt;
        if (!scales) {
            return fail("--gpu int8 needs --scales <file> from `waferedge-cnn calibrate`");
        }
        gpu->scales = *scales;
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
    if (args[0] == "conformal" && args.size() >= 4) {
        std::optional<Split> split;
        if (args.size() == 5 && args[4] != "all") {
            for (auto sp : {Split::train, Split::val, Split::test, Split::unsplit}) {
                if (args[4] == split_name(sp)) {
                    split = sp;
                }
            }
        }
        return cmd_conformal(*model, *set, args[3], split, gpu);
    }
    if (args[0] == "calibrate" && args.size() >= 4) {
        return cmd_calibrate(*model, *set, args[3], args.size() == 5 ? std::stoul(args[4]) : 2048);
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
        return args[0] == "agree" ? cmd_agree(*model, *set, split, gpu)
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
