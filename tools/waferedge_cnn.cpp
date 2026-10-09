// FabEye's CNN on WaferEdge's reference forward pass.
//
//   waferedge-cnn verify <model.wcnn> <file.wmap> <logits.bin>
//       our preprocessing + forward pass against ONNX Runtime's logits (tools/export_cnn.py):
//       max |difference| and whether every map gets the same class
//   waferedge-cnn eval <model.wcnn> <file.wmap> [train|val|test|all]
//       classify every map of a split: per-class F1, macro-F1, maps/s
//
// Maps are spread over all cores (ThreadPool), one ReferenceForward per thread.
#include "waferedge/classifier.hpp"
#include "waferedge/cnn.hpp"
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

// Logits of the given maps, all cores.
std::vector<float> forward_all(const cnn::Model& model, const std::vector<WaferMapView>& maps,
                               double& seconds) {
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

int cmd_verify(const cnn::Model& model, const MapSet& set, const std::string& logits_path) {
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
    const auto logits = forward_all(model, maps, seconds);
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
    print(std::format("{} maps vs ONNX Runtime: max |logit difference| {:.2e}, mean {:.2e}; same "
                      "class on {} of {}; {:.0f} maps/s (reference forward, {} threads)\n",
                      refs->size(), max_diff,
                      sum_diff / static_cast<double>(refs->size() * cnn::kClasses), same_class,
                      refs->size(), static_cast<double>(refs->size()) / seconds,
                      ThreadPool().size()));
    return same_class == refs->size() ? 0 : 1;
}

int cmd_eval(const cnn::Model& model, const MapSet& set, std::optional<Split> split) {
    std::vector<WaferMapView> maps;
    std::vector<Pattern> truth;
    for (const auto& r : set.records()) {
        if (!split || r.split == *split) {
            maps.push_back(r.map);
            truth.push_back(r.truth);
        }
    }
    double seconds = 0;
    const auto logits = forward_all(model, maps, seconds);
    std::vector<Pattern> pred;
    for (std::size_t i = 0; i < maps.size(); ++i) {
        pred.push_back(cnn::predicted(std::span(logits).subspan(i * cnn::kClasses, cnn::kClasses)));
    }
    const auto m = evaluate(truth, pred);
    print(std::format("{} maps: macro-F1 {:.3f}, accuracy {:.3f}; {:.0f} maps/s\n", maps.size(),
                      m.macro_f1, m.accuracy, static_cast<double>(maps.size()) / seconds));
    for (std::size_t c = 0; c < kPatternCount; ++c) {
        if (m.support[c] > 0) {
            print(std::format("  {:<10} f1 {:.3f}  ({} maps)\n",
                              pattern_name(static_cast<Pattern>(c)), m.f1[c], m.support[c]));
        }
    }
    print(describe_machine());
    return 0;
}

int run(int argc, char** argv) {
    const std::vector<std::string> args(argv + 1, argv + argc);
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
        return cmd_verify(*model, *set, args[3]);
    }
    if (args[0] == "eval") {
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
        return cmd_eval(*model, *set, split);
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
