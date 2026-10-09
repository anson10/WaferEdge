// Summarises a .wmap file: counts by truth and split, the median of each spatial signal per
// truth class, and the throughput of each scalar stage over the whole set.
//
//   waferedge-maps data/waferlens_demo.wmap
//
// Timing is wall time on a steady clock over whole passes of the set (geometry tables and
// buffers warmed up beforehand), repeated until at least a second has passed; maps/s is total
// maps over total time.
#include "waferedge/backend.hpp"
#include "waferedge/clusters.hpp"
#include "waferedge/features.hpp"
#include "waferedge/gpu_features.hpp"
#include "waferedge/hough.hpp"
#include "waferedge/machine.hpp"
#include "waferedge/map_file.hpp"
#include "waferedge/randomness.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <exception>
#include <format>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace waferedge;

namespace {

void print(const std::string& s) {
    std::fputs(s.c_str(), stdout);
}

int run(int argc, char** argv) {
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
    ClusterFinder clusters;
    HoughTransform hough;

    // Median of each signal per truth class: a first look at what separates the patterns.
    struct Signals {
        std::vector<double> density, center, edge, sector, cluster_share, elongation, line, z;
    };
    std::array<Signals, kPatternCount + 1> signals;
    for (const auto& r : records) {
        const auto f = compute_features(r.map, cache.get(r.map.rows(), r.map.cols()));
        clusters.run(r.map);
        const auto line = hough.run(r.map);
        const Cluster* big = clusters.largest();
        auto& s = signals[r.truth == Pattern::unknown ? kPatternCount
                                                      : static_cast<std::size_t>(r.truth)];
        s.density.push_back(fail_density(f));
        s.center.push_back(center_ratio(f));
        s.edge.push_back(edge_ratio(f));
        s.sector.push_back(max_sector_ratio(f));
        s.cluster_share.push_back(big == nullptr ? 0.0 : static_cast<double>(big->size) / f.fails);
        s.elongation.push_back(big == nullptr ? 0.0 : shape(*big).elongation);
        s.line.push_back(line.line_dies == 0 || f.fails == 0
                             ? 0.0
                             : (static_cast<double>(line.votes) / line.line_dies) /
                                   fail_density(f));
        s.z.push_back(join_count_z(join_count(r.map)));
    }
    const auto median = [](std::vector<double>& v) {
        if (v.empty()) {
            return 0.0;
        }
        const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
        std::ranges::nth_element(v, mid);
        return *mid;
    };
    print("\nmedian signals by truth: fail density, zone ratios (center, edge), densest sector\n"
          "ratio, largest cluster's share of fails and its elongation, Hough line density ratio,\n"
          "join-count z\n");
    print(std::format("{:<10} {:>7} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6} {:>7}\n", "truth",
                      "density", "center", "edge", "sector", "cluster", "elong", "line", "z"));
    for (std::size_t p = 0; p <= kPatternCount; ++p) {
        auto& s = signals[p];
        if (s.density.empty()) {
            continue;
        }
        const auto name = p == kPatternCount ? "unknown" : pattern_name(static_cast<Pattern>(p));
        print(std::format("{:<10} {:>7.3f} {:>6.2f} {:>6.2f} {:>6.2f} {:>7.2f} {:>6.1f} {:>6.1f} "
                          "{:>7.1f}\n",
                          name, median(s.density), median(s.center), median(s.edge),
                          median(s.sector), median(s.cluster_share), median(s.elongation),
                          median(s.line), median(s.z)));
    }

    // Every backend must give the scalar counts, field for field, on every map.
    if (backend::Avx2::available()) {
        std::size_t differ = 0;
        for (const auto& r : records) {
            const auto& g = cache.get(r.map.rows(), r.map.cols());
            differ +=
                backend::Avx2::features(r.map, g) == backend::Scalar::features(r.map, g) ? 0U : 1U;
        }
        print(std::format("\navx2 features == scalar on {} of {} maps\n", records.size() - differ,
                          records.size()));
        if (differ != 0) {
            return 1;
        }
    }

#if WAFEREDGE_HAS_CUDA
    // The GPU, in batches of 4096 maps: every map compared with scalar, both kernels timed.
    if (backend::Cuda::available()) {
        constexpr std::size_t kBatch = 4096;
        std::vector<WaferMapView> views;
        std::vector<const Geometry*> geometries;
        for (const auto& r : records) {
            views.push_back(r.map);
            geometries.push_back(&cache.get(r.map.rows(), r.map.cols()));
        }
        std::vector<Features> out(records.size());
        for (const auto kernel :
             {gpu::FeatureKernel::shared_atomics, gpu::FeatureKernel::warp_aggregated}) {
            gpu::FeatureEngine engine(kernel);
            engine.set_timing(true);
            const auto name =
                kernel == gpu::FeatureKernel::shared_atomics ? "shared atomics" : "warp aggregated";
            double kernel_ms = 0;
            double copy_ms = 0;
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t at = 0; at < views.size(); at += kBatch) {
                const auto n = std::min(kBatch, views.size() - at);
                if (!engine.run(std::span(views).subspan(at, n),
                                std::span(geometries).subspan(at, n),
                                std::span(out).subspan(at, n))) {
                    print(std::format("cuda {}: {}\n", name, engine.error()));
                    return 1;
                }
                const auto t = engine.last_timings();
                kernel_ms += t.kernel_ms;
                copy_ms += t.upload_ms + t.download_ms;
            }
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::size_t differ = 0;
            for (std::size_t i = 0; i < records.size(); ++i) {
                differ += out[i] == backend::Scalar::features(views[i], *geometries[i]) ? 0U : 1U;
            }
            print(std::format("cuda {} features == scalar on {} of {} maps; one pass in batches of "
                              "{}: {:.0f} maps/s end to end (first pass, geometry uploads "
                              "included), kernels {:.1f} ms, copies {:.1f} ms of {:.1f} ms\n",
                              name, records.size() - differ, records.size(), kBatch,
                              static_cast<double>(records.size()) / seconds, kernel_ms, copy_ms,
                              seconds * 1e3));
            if (differ != 0) {
                return 1;
            }
        }
    }
#endif

    // Throughput of each stage alone, over whole passes of the set.
    print("\nthroughput, one stage at a time, scalar unless named (whole passes, >= 1 s each):\n");
    const auto time_stage = [&](std::string_view name, auto&& stage) {
        using clock = std::chrono::steady_clock;
        std::uint64_t checksum = 0;
        std::size_t passes = 0;
        const auto start = clock::now();
        auto elapsed = clock::duration{};
        do {
            for (const auto& r : records) {
                checksum += stage(r.map);
            }
            ++passes;
            elapsed = clock::now() - start;
        } while (elapsed < std::chrono::seconds(1));
        const double seconds = std::chrono::duration<double>(elapsed).count();
        const auto maps = static_cast<double>(passes * records.size());
        print(std::format("  {:<12} {:>9.0f} maps/s {:>8.0f} ns/map  (checksum {})\n", name,
                          maps / seconds, seconds * 1e9 / maps, checksum / passes));
    };
    time_stage("features", [&](WaferMapView m) -> std::uint64_t {
        return backend::Scalar::features(m, cache.get(m.rows(), m.cols())).fails;
    });
    if (backend::Avx2::available()) {
        // Same checksum as the scalar line: the counts are identical.
        time_stage("features avx2", [&](WaferMapView m) -> std::uint64_t {
            return backend::Avx2::features(m, cache.get(m.rows(), m.cols())).fails;
        });
    }
    time_stage("clusters", [&](WaferMapView m) -> std::uint64_t {
        clusters.run(m);
        return clusters.clusters().size();
    });
    time_stage("hough", [&](WaferMapView m) -> std::uint64_t { return hough.run(m).votes; });
    time_stage("join count",
               [&](WaferMapView m) -> std::uint64_t { return join_count(m).fail_joins; });
    print(describe_machine());
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fputs(std::format("error: {}\n", e.what()).c_str(), stderr);
        return 1;
    }
}
