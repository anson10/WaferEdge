// Feature counts on the GPU: one thread block per map, counters in shared memory
// (docs/gpu.md). Two versions of the counting, measured against each other.
#include "kernels.cuh"

#include <cstddef>
#include <cstdint>

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
    features_shared_atomics(const detail::MapDesc* descs, const std::uint8_t* bins,
                            const std::uint8_t* geometry, Features* out) {
    __shared__ unsigned counts[kWords];
    zero_counts(counts);
    const detail::MapDesc d = descs[blockIdx.x];
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
    features_warp_aggregated(const detail::MapDesc* descs, const std::uint8_t* bins,
                             const std::uint8_t* geometry, Features* out) {
    __shared__ unsigned counts[kWords];
    zero_counts(counts);
    const detail::MapDesc d = descs[blockIdx.x];
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

} // namespace

cudaError_t detail::launch_features(bool warp_aggregated, unsigned count, const MapDesc* descs,
                                    const std::uint8_t* bins, const std::uint8_t* geometry,
                                    Features* out, cudaStream_t stream) {
    if (warp_aggregated) {
        features_warp_aggregated<<<count, kThreads, 0, stream>>>(descs, bins, geometry, out);
    } else {
        features_shared_atomics<<<count, kThreads, 0, stream>>>(descs, bins, geometry, out);
    }
    return cudaGetLastError();
}

} // namespace waferedge::gpu
