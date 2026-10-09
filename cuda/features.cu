// GPU feature counts for a batch of wafer maps: one thread block per map (docs/gpu.md).
//
// C++20 (nvcc 12.4's limit, ADR-0001). The host side packs the batch into one pinned buffer
// so a batch costs one upload, one kernel and one download, whatever its size.
#include "waferedge/backend.hpp"
#include "waferedge/gpu_features.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace waferedge::gpu {

namespace {

constexpr unsigned kThreads = 256; // 8 warps per block
constexpr unsigned kWarpSize = 32;
constexpr unsigned kNoBucket = 0xFF; // key of lanes with nothing to count

// The kernels fill Features as a flat array of 32-bit words; these offsets are its layout.
constexpr unsigned kWords = sizeof(Features) / sizeof(std::uint32_t);
constexpr unsigned kZoneDies = 2;
constexpr unsigned kZoneFails = kZoneDies + kZones;
constexpr unsigned kSectorDies = kZoneFails + kZones;
constexpr unsigned kSectorFails = kSectorDies + kSectors;
constexpr unsigned kRingDies = kSectorFails + kSectors;
constexpr unsigned kRingFails = kRingDies + kRings;
static_assert(kWords == kRingFails + kRings, "Features must be 32-bit counters, no padding");
static_assert(offsetof(Features, zone_dies) == kZoneDies * 4);
static_assert(offsetof(Features, sector_fails) == kSectorFails * 4);
static_assert(offsetof(Features, ring_fails) == kRingFails * 4);

// Per map, at the front of the uploaded buffer. Offsets are bytes from the start of the bins
// section (bins) and of the device geometry buffer (geometry: zones, then sectors, then rings,
// each `dies` bytes).
struct MapDesc {
    std::uint32_t bins;
    std::uint32_t dies; // grid positions, rows * cols
    std::uint32_t geometry;
    std::uint32_t unused;
};

__device__ void zero_counts(unsigned* counts) {
    for (unsigned i = threadIdx.x; i < kWords; i += blockDim.x) {
        counts[i] = 0;
    }
    __syncthreads(); // every counter is 0 before any thread adds to it
}

// Totals, then the block's counters to its slot in global memory.
__device__ void write_counts(unsigned* counts, Features* out) {
    __syncthreads(); // every thread's additions are in
    if (threadIdx.x == 0) {
        unsigned dies = 0;
        unsigned fails = 0;
        for (unsigned z = 0; z < kZones; ++z) {
            dies += counts[kZoneDies + z];
            fails += counts[kZoneFails + z];
        }
        counts[0] = dies;
        counts[1] = fails;
    }
    __syncthreads();
    auto* slot = reinterpret_cast<unsigned*>(out + blockIdx.x);
    for (unsigned i = threadIdx.x; i < kWords; i += blockDim.x) {
        slot[i] = counts[i];
    }
}

// v1. Thread t of the block handles dies t, t + 256, ...: in each step a warp reads 32
// consecutive bytes (one coalesced transaction). Every on-wafer die adds 1 to three counters
// in shared memory (six if it fails). Lanes of a warp that hit the same counter (5 zones for
// 32 lanes) are serialised by the hardware: that contention is what v2 removes.
__global__ void __launch_bounds__(kThreads)
    features_shared_atomics(const MapDesc* descs, const std::uint8_t* bins,
                            const std::uint8_t* geometry, Features* out) {
    __shared__ unsigned counts[kWords];
    zero_counts(counts);
    const MapDesc d = descs[blockIdx.x];
    const std::uint8_t* map = bins + d.bins;
    const std::uint8_t* zones = geometry + d.geometry;
    const std::uint8_t* sectors = zones + d.dies;
    const std::uint8_t* rings = sectors + d.dies;
    for (unsigned i = threadIdx.x; i < d.dies; i += blockDim.x) {
        const std::uint8_t bin = map[i];
        if (bin == 0) {
            continue;
        }
        atomicAdd(&counts[kZoneDies + zones[i]], 1U);
        atomicAdd(&counts[kSectorDies + sectors[i]], 1U);
        atomicAdd(&counts[kRingDies + rings[i]], 1U);
        if (bin >= 2) {
            atomicAdd(&counts[kZoneFails + zones[i]], 1U);
            atomicAdd(&counts[kSectorFails + sectors[i]], 1U);
            atomicAdd(&counts[kRingFails + rings[i]], 1U);
        }
    }
    write_counts(counts, out);
}

// Warp-aggregated increment: __match_any_sync gives each lane the mask of lanes holding the
// same key, so the lanes split into groups by bucket. The lowest lane of each group (its
// leader) adds the group's size; one atomic per distinct bucket instead of one per lane.
// Every lane of the warp must call it (the full mask), lanes with nothing to add pass
// kNoBucket.
__device__ void add_grouped(unsigned* counts, unsigned base, unsigned key) {
    const unsigned group = __match_any_sync(0xFFFFFFFFU, key);
    const unsigned leader = static_cast<unsigned>(__ffs(static_cast<int>(group)) - 1);
    if (key != kNoBucket && (threadIdx.x % kWarpSize) == leader) {
        atomicAdd(&counts[base + key], static_cast<unsigned>(__popc(group)));
    }
}

// v2. Warp w handles 32-die chunks w, w + 8, ...; lane l reads die chunk * 32 + l, the same
// coalesced pattern as v1. The loop condition depends only on the warp, so all 32 lanes
// stay together for the __match_any_sync calls (lanes past the end pass kNoBucket).
__global__ void __launch_bounds__(kThreads)
    features_warp_aggregated(const MapDesc* descs, const std::uint8_t* bins,
                             const std::uint8_t* geometry, Features* out) {
    __shared__ unsigned counts[kWords];
    zero_counts(counts);
    const MapDesc d = descs[blockIdx.x];
    const std::uint8_t* map = bins + d.bins;
    const std::uint8_t* zones = geometry + d.geometry;
    const std::uint8_t* sectors = zones + d.dies;
    const std::uint8_t* rings = sectors + d.dies;
    const unsigned lane = threadIdx.x % kWarpSize;
    const unsigned warps = blockDim.x / kWarpSize;
    for (unsigned chunk = threadIdx.x / kWarpSize; chunk * kWarpSize < d.dies; chunk += warps) {
        const unsigned i = chunk * kWarpSize + lane;
        unsigned zone = kNoBucket;
        unsigned sector = kNoBucket;
        unsigned ring = kNoBucket;
        bool fail = false;
        if (i < d.dies && map[i] != 0) {
            zone = zones[i];
            sector = sectors[i];
            ring = rings[i];
            fail = map[i] >= 2;
        }
        add_grouped(counts, kZoneDies, zone);
        add_grouped(counts, kSectorDies, sector);
        add_grouped(counts, kRingDies, ring);
        add_grouped(counts, kZoneFails, fail ? zone : kNoBucket);
        add_grouped(counts, kSectorFails, fail ? sector : kNoBucket);
        add_grouped(counts, kRingFails, fail ? ring : kNoBucket);
    }
    write_counts(counts, out);
}

std::size_t align16(std::size_t n) {
    return (n + 15) & ~std::size_t{15};
}

} // namespace

struct FeatureEngine::Impl {
    FeatureKernel kernel;
    cudaStream_t stream = nullptr;
    cudaEvent_t events[4] = {};
    // Pinned host staging (page-locked: the GPU copies it directly, ~2x pageable, Phase 0).
    std::uint8_t* host_in = nullptr;
    std::size_t host_in_cap = 0;
    Features* host_out = nullptr;
    std::size_t host_out_cap = 0;
    // Device buffers, reused across runs.
    std::uint8_t* dev_in = nullptr;
    std::size_t dev_in_cap = 0;
    Features* dev_out = nullptr;
    std::size_t dev_out_cap = 0;
    // Geometry tables of every shape seen, back to back; offset by Geometry::id.
    std::uint8_t* dev_geometry = nullptr;
    std::size_t geometry_cap = 0;
    std::size_t geometry_used = 0;
    std::unordered_map<std::uint64_t, std::uint32_t> geometry_offset;
    std::vector<std::uint8_t> staging; // geometry tables on their way up
    std::string error;
    RunTimings timings;
    bool timing = false;

    void record(int which) {
        if (timing) {
            cudaEventRecord(events[which], stream);
        }
    }

    bool ok(cudaError_t e, const char* what) {
        if (e == cudaSuccess) {
            return true;
        }
        error = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }

    template <typename T>
    bool grow_host(T*& p, std::size_t& cap, std::size_t need) {
        if (need <= cap) {
            return true;
        }
        cudaFreeHost(p);
        p = nullptr;
        cap = 0;
        const std::size_t want = std::max(need, need + need / 2);
        if (!ok(cudaMallocHost(reinterpret_cast<void**>(&p), want * sizeof(T)), "cudaMallocHost")) {
            return false;
        }
        cap = want;
        return true;
    }

    template <typename T>
    bool grow_device(T*& p, std::size_t& cap, std::size_t need) {
        if (need <= cap) {
            return true;
        }
        cudaFree(p);
        p = nullptr;
        cap = 0;
        const std::size_t want = std::max(need, need + need / 2);
        if (!ok(cudaMalloc(reinterpret_cast<void**>(&p), want * sizeof(T)), "cudaMalloc")) {
            return false;
        }
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
            std::uint8_t* bigger = nullptr;
            if (!ok(cudaMalloc(reinterpret_cast<void**>(&bigger), want), "cudaMalloc geometry")) {
                return false;
            }
            if (geometry_used > 0 &&
                !ok(cudaMemcpy(bigger, dev_geometry, geometry_used, cudaMemcpyDeviceToDevice),
                    "copy geometry")) {
                cudaFree(bigger);
                return false;
            }
            cudaFree(dev_geometry);
            dev_geometry = bigger;
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

FeatureEngine::FeatureEngine(FeatureKernel kernel) : impl_(std::make_unique<Impl>()) {
    impl_->kernel = kernel;
    impl_->ok(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
    for (auto& e : impl_->events) {
        impl_->ok(cudaEventCreate(&e), "cudaEventCreate");
    }
}

FeatureEngine::~FeatureEngine() {
    if (!impl_) {
        return; // moved from
    }
    cudaFreeHost(impl_->host_in);
    cudaFreeHost(impl_->host_out);
    cudaFree(impl_->dev_in);
    cudaFree(impl_->dev_out);
    cudaFree(impl_->dev_geometry);
    for (auto& e : impl_->events) {
        cudaEventDestroy(e);
    }
    cudaStreamDestroy(impl_->stream);
}

FeatureEngine::FeatureEngine(FeatureEngine&&) noexcept = default;
FeatureEngine& FeatureEngine::operator=(FeatureEngine&&) noexcept = default;

const std::string& FeatureEngine::error() const noexcept {
    return impl_->error;
}
RunTimings FeatureEngine::last_timings() const noexcept {
    return impl_->timings;
}
FeatureKernel FeatureEngine::kernel() const noexcept {
    return impl_->kernel;
}
void FeatureEngine::set_kernel(FeatureKernel kernel) noexcept {
    impl_->kernel = kernel;
}
void FeatureEngine::set_timing(bool on) noexcept {
    impl_->timing = on;
    impl_->timings = {};
}

bool FeatureEngine::run(std::span<const WaferMapView> maps,
                        std::span<const Geometry* const> geometries, std::span<Features> out) {
    Impl& m = *impl_;
    if (!m.error.empty()) {
        return false; // a failed constructor, or an earlier error: the engine is unusable
    }
    const std::size_t count = maps.size();
    if (count == 0) {
        return true;
    }

    // Layout of the upload: descriptors, then each map's bins at a 16-byte aligned offset.
    const std::size_t desc_bytes = align16(count * sizeof(MapDesc));
    std::size_t bin_bytes = 0;
    for (const auto& map : maps) {
        bin_bytes = align16(bin_bytes) + map.bins().size();
    }
    const std::size_t total = desc_bytes + align16(bin_bytes);
    if (!m.grow_host(m.host_in, m.host_in_cap, total) ||
        !m.grow_device(m.dev_in, m.dev_in_cap, total) ||
        !m.grow_host(m.host_out, m.host_out_cap, count) ||
        !m.grow_device(m.dev_out, m.dev_out_cap, count)) {
        return false;
    }

    const auto pack_start = std::chrono::steady_clock::now();
    auto* descs = reinterpret_cast<MapDesc*>(m.host_in);
    std::uint8_t* bins = m.host_in + desc_bytes;
    std::size_t at = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t geometry = 0;
        if (!m.geometry_of(*geometries[i], geometry)) {
            return false;
        }
        at = align16(at);
        const auto map_bins = maps[i].bins();
        descs[i] = MapDesc{static_cast<std::uint32_t>(at),
                           static_cast<std::uint32_t>(map_bins.size()), geometry, 0};
        std::memcpy(bins + at, map_bins.data(), map_bins.size());
        at += map_bins.size();
    }

    if (m.timing) {
        m.timings.pack_ms =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - pack_start)
                .count();
    }

    // One upload, one kernel, one download, all on the engine's stream; events time each
    // when timing is on.
    m.record(0);
    if (!m.ok(cudaMemcpyAsync(m.dev_in, m.host_in, total, cudaMemcpyHostToDevice, m.stream),
              "upload")) {
        return false;
    }
    m.record(1);
    const auto* dev_descs = reinterpret_cast<const MapDesc*>(m.dev_in);
    const std::uint8_t* dev_bins = m.dev_in + desc_bytes;
    const auto grid = static_cast<unsigned>(count);
    if (m.kernel == FeatureKernel::shared_atomics) {
        features_shared_atomics<<<grid, kThreads, 0, m.stream>>>(dev_descs, dev_bins,
                                                                 m.dev_geometry, m.dev_out);
    } else {
        features_warp_aggregated<<<grid, kThreads, 0, m.stream>>>(dev_descs, dev_bins,
                                                                  m.dev_geometry, m.dev_out);
    }
    if (!m.ok(cudaGetLastError(), "kernel launch")) {
        return false;
    }
    m.record(2);
    if (!m.ok(cudaMemcpyAsync(m.host_out, m.dev_out, count * sizeof(Features),
                              cudaMemcpyDeviceToHost, m.stream),
              "download")) {
        return false;
    }
    m.record(3);
    if (!m.ok(cudaStreamSynchronize(m.stream), "kernel")) {
        return false;
    }
    if (m.timing) {
        cudaEventElapsedTime(&m.timings.upload_ms, m.events[0], m.events[1]);
        cudaEventElapsedTime(&m.timings.kernel_ms, m.events[1], m.events[2]);
        cudaEventElapsedTime(&m.timings.download_ms, m.events[2], m.events[3]);
    }
    std::memcpy(out.data(), m.host_out, count * sizeof(Features));
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
    // batch-size-1 point of the crossover benchmark). The pipeline uses gpu::FeatureEngine.
    thread_local gpu::FeatureEngine engine;
    Features f;
    const Geometry* g = &geometry;
    if (!engine.run({&map, 1}, {&g, 1}, {&f, 1})) {
        std::fprintf(stderr, "waferedge: CUDA feature backend failed: %s\n",
                     engine.error().c_str());
        std::abort(); // noexcept by the concept; a broken GPU path must not return made-up counts
    }
    return f;
}

} // namespace waferedge
