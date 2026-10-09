#pragma once

// What the signature engine and its kernels share: the per-map descriptor at the front of
// every upload, and one host-side launch function per kernel (each .cu file owns its kernels;
// launching through a plain function keeps them out of the engine's translation unit).
#include "waferedge/features.hpp"
#include "waferedge/hough.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace waferedge::gpu::detail {

// 16 bytes per map, at the front of the uploaded buffer. bins: byte offset of the map in the
// bins section; dies: rows * cols; geometry: byte offset of its tables (zones, sectors, rings,
// each `dies` bytes) in the device geometry buffer (unused by kernels that don't need it).
struct MapDesc {
    std::uint32_t bins;
    std::uint32_t dies;
    std::uint32_t geometry;
    std::uint16_t rows;
    std::uint16_t cols;
};
static_assert(sizeof(MapDesc) == 16);

// Features of `count` maps into out[0..count), one block per map.
cudaError_t launch_features(bool warp_aggregated, unsigned count, const MapDesc* descs,
                            const std::uint8_t* bins, const std::uint8_t* geometry, Features* out,
                            cudaStream_t stream);

// The strongest Hough line of `count` maps into out[0..count), one block per map.
cudaError_t launch_hough(unsigned count, const MapDesc* descs, const std::uint8_t* bins,
                         HoughLine* out, cudaStream_t stream);

// Copies the Q14 cos / sin tables (hough_cos(), hough_sin()) to the GPU's constant memory and
// sets the Hough kernel's shared-memory carveout. Once per process is enough; calling it
// again is harmless.
cudaError_t upload_hough_tables();

} // namespace waferedge::gpu::detail
