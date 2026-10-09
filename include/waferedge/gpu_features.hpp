#pragma once

// Features for a batch of wafer maps on the GPU (built with the cuda preset only).
//
// C++20 and free of CUDA headers: included by the .cu implementation (nvcc 12.4 compiles at
// most C++20, ADR-0001) and by C++23 tests and benchmarks. CUDA state lives in Impl.
//
// One run(): pack every map's bins and a small descriptor into one pinned host buffer, one
// copy to the device, one kernel (one block per map), one copy back, one sync. On WSL2 each
// CUDA call costs ~0.1 ms (docs/gpu.md), so the count of calls per batch matters more than
// anything the kernel does for small batches.
#include "waferedge/features.hpp"
#include "waferedge/geometry.hpp"
#include "waferedge/wafer_map.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace waferedge::gpu {

enum class FeatureKernel : std::uint8_t {
    // v1, the default: every die does atomicAdd on the block's counters in shared memory.
    shared_atomics,
    // v2: lanes with the same bucket are grouped (__match_any_sync) and one leader per group
    // adds the group's count. Measured ~2x SLOWER than v1 on 40x40 maps (docs/gpu.md); kept as
    // the documented experiment.
    warp_aggregated,
};

// Where the time of the last run() went, from CUDA events on the stream (milliseconds).
// Off by default: on WSL2 each cudaEventRecord is a ~15 us driver call, and four of them added
// ~60 us to every batch (measured with Nsight Systems, docs/gpu.md).
struct RunTimings {
    float pack_ms = 0;     // host: copying the maps into the pinned upload buffer (steady clock)
    float upload_ms = 0;   // host -> device copy of bins and descriptors
    float kernel_ms = 0;   // the feature kernel
    float download_ms = 0; // device -> host copy of the features
};

class FeatureEngine {
public:
    explicit FeatureEngine(FeatureKernel kernel = FeatureKernel::shared_atomics);
    ~FeatureEngine();
    FeatureEngine(const FeatureEngine&) = delete;
    FeatureEngine& operator=(const FeatureEngine&) = delete;
    FeatureEngine(FeatureEngine&&) noexcept;
    FeatureEngine& operator=(FeatureEngine&&) noexcept;

    // Features of maps[i] with geometries[i] into out[i]. Preconditions: equal sizes, each
    // geometry has its map's shape. Buffers grow to the largest batch seen and are reused;
    // each Geometry's tables are uploaded once (cached by Geometry::id). Returns false on a
    // CUDA error; error() says which.
    bool run(std::span<const WaferMapView> maps, std::span<const Geometry* const> geometries,
             std::span<Features> out);

    [[nodiscard]] const std::string& error() const noexcept;
    // Timings of the last run() if set_timing(true) was in effect, else zeros.
    [[nodiscard]] RunTimings last_timings() const noexcept;
    void set_timing(bool on) noexcept;
    [[nodiscard]] FeatureKernel kernel() const noexcept;
    void set_kernel(FeatureKernel kernel) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace waferedge::gpu
