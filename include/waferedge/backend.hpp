#pragma once

// Feature backends (ADR-0002): one concept, several implementations. Tests and benchmarks are
// templates over the concept, so each backend is checked against the scalar reference with
// the same code; the pipeline picks the fastest available one at run time.
#include "waferedge/features.hpp"

#include <concepts>
#include <string_view>

namespace waferedge {

template <typename B>
concept FeatureBackend = requires(WaferMapView map, const Geometry& geometry) {
    { B::name } -> std::convertible_to<std::string_view>;
    { B::available() } noexcept -> std::same_as<bool>;
    { B::features(map, geometry) } noexcept -> std::same_as<Features>;
};

namespace backend {

// The reference: one die at a time.
struct Scalar {
    static constexpr std::string_view name = "scalar";
    [[nodiscard]] static bool available() noexcept { return true; }
    [[nodiscard]] static Features features(WaferMapView map, const Geometry& geometry) noexcept {
        return compute_features(map, geometry);
    }
};

// 32 dies per instruction with 256-bit AVX2 vectors. Callable only when available() is true
// (the CPU has AVX2); the rest of the program is built for the baseline x86-64 CPU.
struct Avx2 {
    static constexpr std::string_view name = "avx2";
    [[nodiscard]] static bool available() noexcept;
    [[nodiscard]] static Features features(WaferMapView map, const Geometry& geometry) noexcept;
};

// The GPU, one map per call (a batch of one through gpu::SignatureEngine): for tests and the
// batch-size-1 end of the crossover benchmark. Real use batches maps (gpu_signatures.hpp).
// Available in builds with the cuda preset on a GPU of compute capability 8.6 or newer.
struct Cuda {
    static constexpr std::string_view name = "cuda";
    [[nodiscard]] static bool available() noexcept;
    [[nodiscard]] static Features features(WaferMapView map, const Geometry& geometry) noexcept;
};

} // namespace backend

static_assert(FeatureBackend<backend::Scalar>);
static_assert(FeatureBackend<backend::Avx2>);
static_assert(FeatureBackend<backend::Cuda>);

using FeaturesFn = Features (*)(WaferMapView, const Geometry&) noexcept;

// The fastest CPU backend this machine runs, chosen once. Never the GPU: one map at a time,
// a GPU call costs far more than the CPU's microsecond per map (docs/gpu.md).
[[nodiscard]] FeaturesFn best_features() noexcept;
[[nodiscard]] std::string_view best_features_name() noexcept;

} // namespace waferedge
