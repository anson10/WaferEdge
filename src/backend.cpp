#include "waferedge/backend.hpp"

namespace waferedge {

FeaturesFn best_features() noexcept {
    static const FeaturesFn best =
        backend::Avx2::available() ? &backend::Avx2::features : &backend::Scalar::features;
    return best;
}

std::string_view best_features_name() noexcept {
    return backend::Avx2::available() ? backend::Avx2::name : backend::Scalar::name;
}

#if !WAFEREDGE_HAS_CUDA
// Built without CUDA: the backend exists for the concept and the tests, never available.
bool backend::Cuda::available() noexcept {
    return false;
}

Features backend::Cuda::features(WaferMapView map, const Geometry& geometry) noexcept {
    return compute_features(map, geometry);
}
#endif

} // namespace waferedge
