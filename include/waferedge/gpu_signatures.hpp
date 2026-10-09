#pragma once

// Spatial signatures for a batch of wafer maps on the GPU (built with the cuda preset only):
// the feature counts, the Hough line and the cluster summary, from one upload of the maps.
//
// C++20 and free of CUDA headers: included by the .cu implementation (nvcc 12.4 compiles at
// most C++20, ADR-0001) and by C++23 tests and benchmarks. CUDA state lives in Impl.
//
// One run(): pack every map's bins and a small descriptor into one pinned host buffer, one
// copy to the device, one kernel per requested signature (one block per map), one copy back,
// one sync. On WSL2 each CUDA call costs ~20-90 us (docs/gpu.md), and moving the maps costs
// more than computing on them, so every signature shares the one upload.
#include "waferedge/clusters.hpp"
#include "waferedge/features.hpp"
#include "waferedge/geometry.hpp"
#include "waferedge/hough.hpp"
#include "waferedge/wafer_map.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace waferedge::gpu {

enum class FeatureKernel : std::uint8_t {
    // v1, the default: every die does atomicAdd on the block's counters in shared memory.
    shared_atomics,
    // v2: lanes with the same bucket are grouped (__match_any_sync) and one leader per group
    // adds the group's count. Measured ~2x SLOWER than v1: 3x the instructions, and more
    // shared-atomic work, since Ampere already merges a warp's conflicting atomics in
    // hardware (docs/gpu.md). Kept as the documented experiment.
    warp_aggregated,
};

// Where the time of the last run() went (milliseconds). Pack is measured on the host's steady
// clock; the rest from CUDA events on the stream. Off by default: on WSL2 each
// cudaEventRecord is a ~15 us driver call, and they added ~60 us to every batch.
struct RunTimings {
    float pack_ms = 0;     // host: copying the maps into the pinned upload buffer
    float upload_ms = 0;   // host -> device copy of bins and descriptors
    float features_ms = 0; // feature kernel (0 if not requested)
    float hough_ms = 0;    // Hough kernel (0 if not requested)
    float clusters_ms = 0; // cluster kernel (0 if not requested)
    float download_ms = 0; // device -> host copy of the results
};

class SignatureEngine {
public:
    explicit SignatureEngine(FeatureKernel kernel = FeatureKernel::shared_atomics);
    ~SignatureEngine();
    SignatureEngine(const SignatureEngine&) = delete;
    SignatureEngine& operator=(const SignatureEngine&) = delete;
    SignatureEngine(SignatureEngine&&) noexcept;
    SignatureEngine& operator=(SignatureEngine&&) noexcept;

    // For each map i: features[i] (if features is not empty; needs geometries[i] with the
    // map's shape), lines[i], the strongest Hough line (if lines is not empty), and
    // clusters[i], the cluster count and largest cluster (if clusters is not empty). Each
    // output is either empty (not computed) or has one slot per map. Buffers grow to the
    // largest batch seen and are reused; geometry tables are uploaded once per Geometry
    // (cached by Geometry::id). Returns false on a CUDA error; error() says which.
    bool run(std::span<const WaferMapView> maps, std::span<const Geometry* const> geometries,
             std::span<Features> features, std::span<HoughLine> lines,
             std::span<ClusterSummary> clusters = {});

    [[nodiscard]] const std::string& error() const noexcept;
    // Timings of the last run() if set_timing(true) was in effect, else zeros. Timing runs
    // without CUDA Graphs (events are recorded between the steps).
    [[nodiscard]] RunTimings last_timings() const noexcept;
    void set_timing(bool on) noexcept;

    // CUDA Graphs: the first batch of each shape (count, sizes, requested signatures) is
    // recorded once into a graph; later batches of that shape replay it with one
    // cudaGraphLaunch instead of a copy, up to three launches and a copy (docs/gpu.md).
    void set_graphs(bool on) noexcept;
    [[nodiscard]] bool graphs() const noexcept;
    [[nodiscard]] std::size_t cached_graphs() const noexcept;
    [[nodiscard]] FeatureKernel kernel() const noexcept;
    void set_kernel(FeatureKernel kernel) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace waferedge::gpu
