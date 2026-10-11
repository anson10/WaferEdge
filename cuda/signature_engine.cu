// The signature engine: packs a batch of maps, uploads it once, runs the requested kernels
// (features, Hough, clusters) on that one upload, and downloads every result in one copy.
// Two slots, each with its own buffers and stream, let the host pack the next batch while the
// GPU works on the current one (submit / collect, docs/gpu.md).
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
#include <span>
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
    // One batch's buffers and stream. Two slots let the host pack batch n+1 into one while the
    // GPU works on batch n from the other (submit / collect, docs/gpu.md).
    struct Slot {
        cudaStream_t stream = nullptr;
        cudaEvent_t events[kEvents] = {};
        // Pinned host staging (page-locked: the GPU copies it directly, ~2x pageable, Phase 0).
        std::uint8_t* host_in = nullptr;
        std::size_t host_in_cap = 0;
        std::uint8_t* host_out = nullptr;
        std::size_t host_out_cap = 0;
        // Device buffers, reused across batches. dev_out holds the features, the lines, then the
        // cluster summaries; dev_scratch the union-find trees of maps too big for shared memory.
        std::uint8_t* dev_in = nullptr;
        std::size_t dev_in_cap = 0;
        std::uint8_t* dev_out = nullptr;
        std::size_t dev_out_cap = 0;
        std::uint8_t* dev_scratch = nullptr;
        std::size_t dev_scratch_cap = 0;
        // The batch in flight: where collect() puts its results.
        std::size_t count = 0;
        std::size_t features_bytes = 0;
        std::size_t lines_bytes = 0;
        std::span<Features> features;
        std::span<HoughLine> lines;
        std::span<ClusterSummary> clusters;
        float pack_ms = 0;
        bool timed = false;
    };

    FeatureKernel kernel = FeatureKernel::shared_atomics;
    std::array<Slot, SignatureEngine::kSlots> slots;
    std::size_t oldest = 0;    // the slot collect() takes next
    std::size_t in_flight = 0; // batches submitted and not yet collected
    bool hough_tables = false; // cos / sin uploaded to constant memory
    // Geometry tables of every Geometry seen, back to back; offset by Geometry::id. Shared by
    // both slots (read only on the device).
    std::uint8_t* dev_geometry = nullptr;
    std::size_t geometry_cap = 0;
    std::size_t geometry_used = 0;
    std::unordered_map<std::uint64_t, std::uint32_t> geometry_offset;
    std::vector<std::uint8_t> staging; // geometry tables on their way up
    std::string error;
    RunTimings timings;
    bool timing = false;
    // CUDA Graphs: one executable graph per batch shape and slot, replayed with one launch.
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

    bool ok(cudaError_t e, const char* what) {
        if (e == cudaSuccess) {
            return true;
        }
        error = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }

    // Waits for every batch in flight (their results stay in their slots until collected).
    bool wait_all() {
        for (std::size_t i = 0; i < in_flight; ++i) {
            if (!ok(cudaStreamSynchronize(slots[(oldest + i) % slots.size()].stream), "kernels")) {
                return false;
            }
        }
        return true;
    }

    // Pinned host or device buffer of at least `need` bytes; grows by half again when too
    // small, so a run of slowly growing batches doesn't reallocate every time. Only a free
    // slot's buffers grow, so nothing in flight uses them.
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

    // Device offset of a geometry's tables, uploading them on first use. The upload is a
    // synchronous copy: rare (once per wafer shape), so it may stall behind a batch in flight.
    bool geometry_of(const Geometry& g, std::uint32_t& offset) {
        if (const auto it = geometry_offset.find(g.id()); it != geometry_offset.end()) {
            offset = it->second;
            return true;
        }
        const std::size_t n = g.zones().size();
        const std::size_t at = align16(geometry_used);
        if (at + 3 * n > geometry_cap) {
            // Grow, keeping what is there: copy the old buffer on the device. A batch in
            // flight may still read the old buffer, so wait for it before freeing.
            if (!wait_all()) {
                return false;
            }
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
    for (Impl::Slot& slot : impl_->slots) {
        impl_->ok(cudaStreamCreate(&slot.stream), "cudaStreamCreate");
        for (auto& e : slot.events) {
            impl_->ok(cudaEventCreate(&e), "cudaEventCreate");
        }
    }
}

SignatureEngine::~SignatureEngine() {
    if (!impl_) {
        return; // moved from
    }
    impl_->wait_all(); // copies in flight still use the pinned buffers
    impl_->clear_graphs();
    for (Impl::Slot& slot : impl_->slots) {
        cudaFreeHost(slot.host_in);
        cudaFreeHost(slot.host_out);
        cudaFree(slot.dev_in);
        cudaFree(slot.dev_out);
        cudaFree(slot.dev_scratch);
        for (auto& e : slot.events) {
            cudaEventDestroy(e);
        }
        cudaStreamDestroy(slot.stream);
    }
    cudaFree(impl_->dev_geometry);
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
std::size_t SignatureEngine::in_flight() const noexcept {
    return impl_->in_flight;
}

bool SignatureEngine::run(std::span<const WaferMapView> maps,
                          std::span<const Geometry* const> geometries, std::span<Features> features,
                          std::span<HoughLine> lines, std::span<ClusterSummary> clusters) {
    // Batches still in flight finish first (into their own outputs), then this one.
    while (impl_->in_flight > 0) {
        if (!collect()) {
            return false;
        }
    }
    return submit(maps, geometries, features, lines, clusters) &&
           (impl_->in_flight == 0 || collect());
}

bool SignatureEngine::submit(std::span<const WaferMapView> maps,
                             std::span<const Geometry* const> geometries,
                             std::span<Features> features, std::span<HoughLine> lines,
                             std::span<ClusterSummary> clusters) {
    Impl& m = *impl_;
    if (!m.error.empty()) {
        return false; // a failed constructor, or an earlier error: the engine is unusable
    }
    if (m.in_flight == m.slots.size()) {
        m.error = "submit: both slots are in flight; collect() first";
        return false;
    }
    const std::size_t count = maps.size();
    const bool want_features = !features.empty();
    const bool want_lines = !lines.empty();
    const bool want_clusters = !clusters.empty();
    if (count == 0 || (!want_features && !want_lines && !want_clusters)) {
        return true; // nothing to do: nothing in flight either
    }
    if (want_lines && !m.hough_tables) {
        if (!m.ok(detail::upload_hough_tables(), "upload Hough tables")) {
            return false;
        }
        m.hough_tables = true;
    }
    Impl::Slot& slot = m.slots[(m.oldest + m.in_flight) % m.slots.size()];

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
    if (!m.grow(slot.host_in, slot.host_in_cap, in_bytes, true) ||
        !m.grow(slot.dev_in, slot.dev_in_cap, in_bytes, false) ||
        !m.grow(slot.host_out, slot.host_out_cap, out_bytes, true) ||
        !m.grow(slot.dev_out, slot.dev_out_cap, out_bytes, false) ||
        (need_scratch &&
         !m.grow(slot.dev_scratch, slot.dev_scratch_cap, 8 * align16(bin_bytes), false))) {
        return false;
    }

    // Packing runs on the host while the other slot's batch (if any) is on the GPU.
    const auto pack_start = std::chrono::steady_clock::now();
    auto* descs = reinterpret_cast<MapDesc*>(slot.host_in);
    std::uint8_t* bins = slot.host_in + desc_bytes;
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
    slot.timed = m.timing;
    slot.pack_ms = m.timing ? std::chrono::duration<float, std::milli>(
                                  std::chrono::steady_clock::now() - pack_start)
                                  .count()
                            : 0;

    // One upload, the kernels, one download, all on the slot's stream (in order); events
    // time each step when timing is on.
    const auto record = [&](Event which) {
        if (slot.timed) {
            cudaEventRecord(slot.events[which], slot.stream);
        }
    };
    const auto* dev_descs = reinterpret_cast<const MapDesc*>(slot.dev_in);
    const std::uint8_t* dev_bins = slot.dev_in + desc_bytes;
    const auto grid = static_cast<unsigned>(count);
    const auto enqueue = [&]() -> bool {
        record(kBeforeUpload);
        if (!m.ok(cudaMemcpyAsync(slot.dev_in, slot.host_in, in_bytes, cudaMemcpyHostToDevice,
                                  slot.stream),
                  "upload")) {
            return false;
        }
        record(kAfterUpload);
        if (want_features &&
            !m.ok(detail::launch_features(m.kernel == FeatureKernel::warp_aggregated, grid,
                                          dev_descs, dev_bins, m.dev_geometry,
                                          reinterpret_cast<Features*>(slot.dev_out), slot.stream),
                  "feature kernel launch")) {
            return false;
        }
        record(kAfterFeatures);
        if (want_lines &&
            !m.ok(detail::launch_hough(grid, dev_descs, dev_bins,
                                       reinterpret_cast<HoughLine*>(slot.dev_out + features_bytes),
                                       slot.stream),
                  "Hough kernel launch")) {
            return false;
        }
        record(kAfterHough);
        if (want_clusters &&
            !m.ok(
                detail::launch_clusters(
                    grid, dev_descs, dev_bins,
                    need_scratch ? reinterpret_cast<int*>(slot.dev_scratch) : nullptr,
                    reinterpret_cast<ClusterSummary*>(slot.dev_out + features_bytes + lines_bytes),
                    slot.stream),
                "cluster kernel launch")) {
            return false;
        }
        record(kAfterClusters);
        if (!m.ok(cudaMemcpyAsync(slot.host_out, slot.dev_out, out_bytes, cudaMemcpyDeviceToHost,
                                  slot.stream),
                  "download")) {
            return false;
        }
        record(kAfterDownload);
        return true;
    };

    if (m.graphs && !slot.timed) {
        // Everything the graph freezes: sizes, which kernels, and every buffer address (each
        // slot has its own buffers, so its own graphs). A buffer that grew has a new address,
        // so its old graphs simply stop matching.
        const Impl::GraphKey key = {
            count,
            in_bytes,
            out_bytes,
            (want_features ? 1U : 0U) | (want_lines ? 2U : 0U) | (want_clusters ? 4U : 0U) |
                (need_scratch ? 8U : 0U) | (m.kernel == FeatureKernel::warp_aggregated ? 16U : 0U),
            reinterpret_cast<std::uintptr_t>(slot.host_in),
            reinterpret_cast<std::uintptr_t>(slot.host_out),
            reinterpret_cast<std::uintptr_t>(slot.dev_in),
            reinterpret_cast<std::uintptr_t>(slot.dev_out),
            reinterpret_cast<std::uintptr_t>(slot.dev_scratch),
            reinterpret_cast<std::uintptr_t>(m.dev_geometry),
        };
        auto it = m.graph_cache.find(key);
        if (it == m.graph_cache.end()) {
            if (m.graph_cache.size() >= Impl::kMaxGraphs) {
                // A graph launched for the other slot may still run: wait before destroying.
                if (!m.wait_all()) {
                    return false;
                }
                m.clear_graphs();
            }
            // Record instead of run: between Begin and EndCapture the stream collects the
            // copies and launches into a graph. The graph is then checked and prepared once
            // (instantiate), and every later batch of this shape replays it with one call.
            if (!m.ok(cudaStreamBeginCapture(slot.stream, cudaStreamCaptureModeThreadLocal),
                      "begin capture")) {
                return false;
            }
            const bool recorded = enqueue();
            cudaGraph_t graph = nullptr;
            const cudaError_t ended = cudaStreamEndCapture(slot.stream, &graph);
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
        if (!m.ok(cudaGraphLaunch(it->second, slot.stream), "graph launch")) {
            return false;
        }
    } else if (!enqueue()) {
        return false;
    }
    slot.count = count;
    slot.features_bytes = features_bytes;
    slot.lines_bytes = lines_bytes;
    slot.features = features;
    slot.lines = lines;
    slot.clusters = clusters;
    ++m.in_flight;
    return true;
}

bool SignatureEngine::collect() {
    Impl& m = *impl_;
    if (!m.error.empty()) {
        return false;
    }
    if (m.in_flight == 0) {
        m.error = "collect: no batch in flight";
        return false;
    }
    Impl::Slot& slot = m.slots[m.oldest];
    m.oldest = (m.oldest + 1) % m.slots.size();
    --m.in_flight;
    if (!m.ok(cudaStreamSynchronize(slot.stream), "kernels")) {
        return false;
    }
    if (slot.timed) {
        const auto between = [&](Event from, Event to) {
            float ms = 0;
            cudaEventElapsedTime(&ms, slot.events[from], slot.events[to]);
            return ms;
        };
        m.timings = {};
        m.timings.pack_ms = slot.pack_ms;
        m.timings.upload_ms = between(kBeforeUpload, kAfterUpload);
        m.timings.features_ms = between(kAfterUpload, kAfterFeatures);
        m.timings.hough_ms = between(kAfterFeatures, kAfterHough);
        m.timings.clusters_ms = between(kAfterHough, kAfterClusters);
        m.timings.download_ms = between(kAfterClusters, kAfterDownload);
    }
    const std::size_t count = slot.count;
    if (!slot.features.empty()) {
        std::memcpy(slot.features.data(), slot.host_out, count * sizeof(Features));
    }
    if (!slot.lines.empty()) {
        std::memcpy(slot.lines.data(), slot.host_out + slot.features_bytes,
                    count * sizeof(HoughLine));
    }
    if (!slot.clusters.empty()) {
        std::memcpy(slot.clusters.data(), slot.host_out + slot.features_bytes + slot.lines_bytes,
                    count * sizeof(ClusterSummary));
    }
    slot.features = {};
    slot.lines = {};
    slot.clusters = {};
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
