// The signature engine: packs a batch of maps, uploads it once, runs the requested kernels
// (features, Hough) on that one upload, and downloads every result in one copy (docs/gpu.md).
//
// C++20 (nvcc 12.4's limit, ADR-0001). Kernels live in their own .cu files behind
// detail::launch_* (kernels.cuh).
#include "kernels.cuh"
#include "waferedge/backend.hpp"
#include "waferedge/gpu_signatures.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace waferedge::gpu {

namespace {

using detail::MapDesc;

std::size_t align16(std::size_t n) {
    return (n + 15) & ~std::size_t{15};
}

enum Event {
    kBeforeUpload,
    kAfterUpload,
    kAfterFeatures,
    kAfterHough,
    kAfterClusters,
    kAfterDownload,
    kEvents
};

} // namespace

struct SignatureEngine::Impl {
    FeatureKernel kernel = FeatureKernel::shared_atomics;
    cudaStream_t stream = nullptr;
    cudaEvent_t events[kEvents] = {};
    bool hough_tables = false; // cos / sin uploaded to constant memory
    // Pinned host staging (page-locked: the GPU copies it directly, ~2x pageable, Phase 0).
    std::uint8_t* host_in = nullptr;
    std::size_t host_in_cap = 0;
    std::uint8_t* host_out = nullptr;
    std::size_t host_out_cap = 0;
    // Device buffers, reused across runs. dev_out holds the features, the lines, then the
    // cluster summaries; dev_scratch the union-find trees of maps too big for shared memory.
    std::uint8_t* dev_in = nullptr;
    std::size_t dev_in_cap = 0;
    std::uint8_t* dev_out = nullptr;
    std::size_t dev_out_cap = 0;
    std::uint8_t* dev_scratch = nullptr;
    std::size_t dev_scratch_cap = 0;
    // Geometry tables of every Geometry seen, back to back; offset by Geometry::id.
    std::uint8_t* dev_geometry = nullptr;
    std::size_t geometry_cap = 0;
    std::size_t geometry_used = 0;
    std::unordered_map<std::uint64_t, std::uint32_t> geometry_offset;
    std::vector<std::uint8_t> staging; // geometry tables on their way up
    std::string error;
    RunTimings timings;
    bool timing = false;
    // CUDA Graphs: one executable graph per batch shape, replayed with one launch.
    using GraphKey = std::array<std::uintptr_t, 10>;
    static constexpr std::size_t kMaxGraphs = 32; // then start over: bounded memory
    bool graphs = false;
    std::map<GraphKey, cudaGraphExec_t> graph_cache;

    void clear_graphs() {
        for (auto& [key, exec] : graph_cache) {
            cudaGraphExecDestroy(exec);
        }
        graph_cache.clear();
    }

    void record(Event which) {
        if (timing) {
            cudaEventRecord(events[which], stream);
        }
    }

    float between(Event from, Event to) const {
        float ms = 0;
        cudaEventElapsedTime(&ms, events[from], events[to]);
        return ms;
    }

    bool ok(cudaError_t e, const char* what) {
        if (e == cudaSuccess) {
            return true;
        }
        error = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }

    // Pinned host or device buffer of at least `need` bytes; grows by half again when too
    // small, so a run of slowly growing batches doesn't reallocate every time.
    bool grow(std::uint8_t*& p, std::size_t& cap, std::size_t need, bool pinned) {
        if (need <= cap) {
            return true;
        }
        if (pinned) {
            cudaFreeHost(p);
        } else {
            cudaFree(p);
        }
        p = nullptr;
        cap = 0;
        const std::size_t want = need + need / 2;
        void* fresh = nullptr;
        if (!ok(pinned ? cudaMallocHost(&fresh, want) : cudaMalloc(&fresh, want),
                pinned ? "cudaMallocHost" : "cudaMalloc")) {
            return false;
        }
        p = static_cast<std::uint8_t*>(fresh);
        cap = want;
        return true;
    }

    // Device offset of a geometry's tables, uploading them on first use.
    bool geometry_of(const Geometry& g, std::uint32_t& offset) {
        if (const auto it = geometry_offset.find(g.id()); it != geometry_offset.end()) {
            offset = it->second;
            return true;
        }
        const std::size_t n = g.zones().size();
        const std::size_t at = align16(geometry_used);
        if (at + 3 * n > geometry_cap) {
            // Grow, keeping what is there: copy the old buffer on the device.
            const std::size_t want = std::max<std::size_t>({at + 3 * n, 2 * geometry_cap, 1 << 20});
            void* bigger = nullptr;
            if (!ok(cudaMalloc(&bigger, want), "cudaMalloc geometry")) {
                return false;
            }
            if (geometry_used > 0 &&
                !ok(cudaMemcpy(bigger, dev_geometry, geometry_used, cudaMemcpyDeviceToDevice),
                    "copy geometry")) {
                cudaFree(bigger);
                return false;
            }
            cudaFree(dev_geometry);
            dev_geometry = static_cast<std::uint8_t*>(bigger);
            geometry_cap = want;
        }
        staging.resize(3 * n);
        std::memcpy(staging.data(), g.zones().data(), n);
        std::memcpy(staging.data() + n, g.sectors().data(), n);
        std::memcpy(staging.data() + 2 * n, g.rings().data(), n);
        if (!ok(cudaMemcpy(dev_geometry + at, staging.data(), 3 * n, cudaMemcpyHostToDevice),
                "upload geometry")) {
            return false;
        }
        geometry_used = at + 3 * n;
        offset = static_cast<std::uint32_t>(at);
        geometry_offset.emplace(g.id(), offset);
        return true;
    }
};

SignatureEngine::SignatureEngine(FeatureKernel kernel) : impl_(std::make_unique<Impl>()) {
    impl_->kernel = kernel;
    impl_->ok(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
    for (auto& e : impl_->events) {
        impl_->ok(cudaEventCreate(&e), "cudaEventCreate");
    }
}

SignatureEngine::~SignatureEngine() {
    if (!impl_) {
        return; // moved from
    }
    impl_->clear_graphs();
    cudaFreeHost(impl_->host_in);
    cudaFreeHost(impl_->host_out);
    cudaFree(impl_->dev_in);
    cudaFree(impl_->dev_out);
    cudaFree(impl_->dev_scratch);
    cudaFree(impl_->dev_geometry);
    for (auto& e : impl_->events) {
        cudaEventDestroy(e);
    }
    cudaStreamDestroy(impl_->stream);
}

SignatureEngine::SignatureEngine(SignatureEngine&&) noexcept = default;
SignatureEngine& SignatureEngine::operator=(SignatureEngine&&) noexcept = default;

const std::string& SignatureEngine::error() const noexcept {
    return impl_->error;
}
RunTimings SignatureEngine::last_timings() const noexcept {
    return impl_->timings;
}
FeatureKernel SignatureEngine::kernel() const noexcept {
    return impl_->kernel;
}
void SignatureEngine::set_kernel(FeatureKernel kernel) noexcept {
    impl_->kernel = kernel;
}
void SignatureEngine::set_graphs(bool on) noexcept {
    impl_->graphs = on;
}
bool SignatureEngine::graphs() const noexcept {
    return impl_->graphs;
}
std::size_t SignatureEngine::cached_graphs() const noexcept {
    return impl_->graph_cache.size();
}
void SignatureEngine::set_timing(bool on) noexcept {
    impl_->timing = on;
    impl_->timings = {};
}

bool SignatureEngine::run(std::span<const WaferMapView> maps,
                          std::span<const Geometry* const> geometries, std::span<Features> features,
                          std::span<HoughLine> lines, std::span<ClusterSummary> clusters) {
    Impl& m = *impl_;
    if (!m.error.empty()) {
        return false; // a failed constructor, or an earlier error: the engine is unusable
    }
    const std::size_t count = maps.size();
    const bool want_features = !features.empty();
    const bool want_lines = !lines.empty();
    const bool want_clusters = !clusters.empty();
    if (count == 0 || (!want_features && !want_lines && !want_clusters)) {
        return true;
    }
    if (want_lines && !m.hough_tables) {
        if (!m.ok(detail::upload_hough_tables(), "upload Hough tables")) {
            return false;
        }
        m.hough_tables = true;
    }

    // Upload layout: descriptors, then each map's bins at a 16-byte aligned offset.
    // Result layout: features, lines, clusters (each if requested), one download.
    const std::size_t desc_bytes = align16(count * sizeof(MapDesc));
    std::size_t bin_bytes = 0;
    bool big_map = false;
    for (const auto& map : maps) {
        bin_bytes = align16(bin_bytes) + map.bins().size();
        big_map = big_map || map.bins().size() > detail::kClusterSharedPositions;
    }
    const std::size_t in_bytes = desc_bytes + align16(bin_bytes);
    const std::size_t features_bytes = want_features ? align16(count * sizeof(Features)) : 0;
    const std::size_t lines_bytes = want_lines ? align16(count * sizeof(HoughLine)) : 0;
    const std::size_t out_bytes =
        features_bytes + lines_bytes + (want_clusters ? count * sizeof(ClusterSummary) : 0);
    const bool need_scratch = want_clusters && big_map;
    if (!m.grow(m.host_in, m.host_in_cap, in_bytes, true) ||
        !m.grow(m.dev_in, m.dev_in_cap, in_bytes, false) ||
        !m.grow(m.host_out, m.host_out_cap, out_bytes, true) ||
        !m.grow(m.dev_out, m.dev_out_cap, out_bytes, false) ||
        (need_scratch &&
         !m.grow(m.dev_scratch, m.dev_scratch_cap, 8 * align16(bin_bytes), false))) {
        return false;
    }

    const auto pack_start = std::chrono::steady_clock::now();
    auto* descs = reinterpret_cast<MapDesc*>(m.host_in);
    std::uint8_t* bins = m.host_in + desc_bytes;
    std::size_t at = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t geometry = 0;
        if (want_features && !m.geometry_of(*geometries[i], geometry)) {
            return false;
        }
        at = align16(at);
        const auto map_bins = maps[i].bins();
        descs[i] = MapDesc{
            static_cast<std::uint32_t>(at), static_cast<std::uint32_t>(map_bins.size()), geometry,
            static_cast<std::uint16_t>(maps[i].rows()), static_cast<std::uint16_t>(maps[i].cols())};
        std::memcpy(bins + at, map_bins.data(), map_bins.size());
        at += map_bins.size();
    }
    if (m.timing) {
        m.timings = {};
        m.timings.pack_ms =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - pack_start)
                .count();
    }

    // One upload, the kernels, one download, all on the engine's stream (in order); events
    // time each step when timing is on.
    const auto* dev_descs = reinterpret_cast<const MapDesc*>(m.dev_in);
    const std::uint8_t* dev_bins = m.dev_in + desc_bytes;
    const auto grid = static_cast<unsigned>(count);
    const auto enqueue = [&]() -> bool {
        m.record(kBeforeUpload);
        if (!m.ok(cudaMemcpyAsync(m.dev_in, m.host_in, in_bytes, cudaMemcpyHostToDevice, m.stream),
                  "upload")) {
            return false;
        }
        m.record(kAfterUpload);
        if (want_features &&
            !m.ok(detail::launch_features(m.kernel == FeatureKernel::warp_aggregated, grid,
                                          dev_descs, dev_bins, m.dev_geometry,
                                          reinterpret_cast<Features*>(m.dev_out), m.stream),
                  "feature kernel launch")) {
            return false;
        }
        m.record(kAfterFeatures);
        if (want_lines &&
            !m.ok(detail::launch_hough(grid, dev_descs, dev_bins,
                                       reinterpret_cast<HoughLine*>(m.dev_out + features_bytes),
                                       m.stream),
                  "Hough kernel launch")) {
            return false;
        }
        m.record(kAfterHough);
        if (want_clusters &&
            !m.ok(detail::launch_clusters(
                      grid, dev_descs, dev_bins,
                      need_scratch ? reinterpret_cast<int*>(m.dev_scratch) : nullptr,
                      reinterpret_cast<ClusterSummary*>(m.dev_out + features_bytes + lines_bytes),
                      m.stream),
                  "cluster kernel launch")) {
            return false;
        }
        m.record(kAfterClusters);
        if (!m.ok(
                cudaMemcpyAsync(m.host_out, m.dev_out, out_bytes, cudaMemcpyDeviceToHost, m.stream),
                "download")) {
            return false;
        }
        m.record(kAfterDownload);
        return true;
    };

    if (m.graphs && !m.timing) {
        // Everything the graph freezes: sizes, which kernels, and every buffer address. A
        // buffer that grew has a new address, so its old graphs simply stop matching.
        const Impl::GraphKey key = {
            count,
            in_bytes,
            out_bytes,
            (want_features ? 1U : 0U) | (want_lines ? 2U : 0U) | (want_clusters ? 4U : 0U) |
                (need_scratch ? 8U : 0U) | (m.kernel == FeatureKernel::warp_aggregated ? 16U : 0U),
            reinterpret_cast<std::uintptr_t>(m.host_in),
            reinterpret_cast<std::uintptr_t>(m.host_out),
            reinterpret_cast<std::uintptr_t>(m.dev_in),
            reinterpret_cast<std::uintptr_t>(m.dev_out),
            reinterpret_cast<std::uintptr_t>(m.dev_scratch),
            reinterpret_cast<std::uintptr_t>(m.dev_geometry),
        };
        auto it = m.graph_cache.find(key);
        if (it == m.graph_cache.end()) {
            if (m.graph_cache.size() >= Impl::kMaxGraphs) {
                m.clear_graphs();
            }
            // Record instead of run: between Begin and EndCapture the stream collects the
            // copies and launches into a graph. The graph is then checked and prepared once
            // (instantiate), and every later batch of this shape replays it with one call.
            if (!m.ok(cudaStreamBeginCapture(m.stream, cudaStreamCaptureModeThreadLocal),
                      "begin capture")) {
                return false;
            }
            const bool recorded = enqueue();
            cudaGraph_t graph = nullptr;
            const cudaError_t ended = cudaStreamEndCapture(m.stream, &graph);
            if (!recorded || !m.ok(ended, "end capture")) {
                cudaGraphDestroy(graph);
                return false;
            }
            cudaGraphExec_t exec = nullptr;
            const cudaError_t made = cudaGraphInstantiate(&exec, graph, 0);
            cudaGraphDestroy(graph); // the executable graph keeps what it needs
            if (!m.ok(made, "instantiate graph")) {
                return false;
            }
            it = m.graph_cache.emplace(key, exec).first;
        }
        if (!m.ok(cudaGraphLaunch(it->second, m.stream), "graph launch")) {
            return false;
        }
    } else if (!enqueue()) {
        return false;
    }
    if (!m.ok(cudaStreamSynchronize(m.stream), "kernels")) {
        return false;
    }
    if (m.timing) {
        m.timings.upload_ms = m.between(kBeforeUpload, kAfterUpload);
        m.timings.features_ms = m.between(kAfterUpload, kAfterFeatures);
        m.timings.hough_ms = m.between(kAfterFeatures, kAfterHough);
        m.timings.clusters_ms = m.between(kAfterHough, kAfterClusters);
        m.timings.download_ms = m.between(kAfterClusters, kAfterDownload);
    }
    if (want_features) {
        std::memcpy(features.data(), m.host_out, count * sizeof(Features));
    }
    if (want_lines) {
        std::memcpy(lines.data(), m.host_out + features_bytes, count * sizeof(HoughLine));
    }
    if (want_clusters) {
        std::memcpy(clusters.data(), m.host_out + features_bytes + lines_bytes,
                    count * sizeof(ClusterSummary));
    }
    return true;
}

} // namespace waferedge::gpu

namespace waferedge {

bool backend::Cuda::available() noexcept {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError(); // clear the sticky "no device" error
        return false;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return false;
    }
    return prop.major * 10 + prop.minor >= 86; // built for sm_86 (PTX JIT covers newer)
}

Features backend::Cuda::features(WaferMapView map, const Geometry& geometry) noexcept {
    // A batch of one through a per-thread engine: what the concept needs (tests, and the
    // batch-size-1 point of the crossover benchmark). The pipeline batches.
    thread_local gpu::SignatureEngine engine;
    Features f;
    const Geometry* g = &geometry;
    if (!engine.run({&map, 1}, {&g, 1}, {&f, 1}, {})) {
        std::fprintf(stderr, "waferedge: CUDA feature backend failed: %s\n",
                     engine.error().c_str());
        std::abort(); // noexcept by the concept; a broken GPU path must not return made-up counts
    }
    return f;
}

} // namespace waferedge
