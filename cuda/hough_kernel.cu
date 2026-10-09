// The Hough transform on the GPU: one block per map, one thread per angle (docs/gpu.md).
//
// The CPU version (src/hough.cpp) has every fail die vote once per angle. Here thread k of a
// block owns angle k's row of the vote table, so no two threads ever write the same counter:
// no atomics in the voting loop, and the result is deterministic. The votes are the same
// integers as on the CPU (Q14 cos / sin, doubled die coordinates, floor shift), so the line
// found is the same line, checked with == in the tests.
#include "kernels.cuh"

#include <cstdint>

namespace waferedge::gpu {

namespace {

constexpr unsigned kThreads = 192; // 6 warps; thread k owns angle (first + k) of a chunk
constexpr unsigned kWarps = kThreads / 32;
constexpr unsigned kAngles = static_cast<unsigned>(kHoughAngles);
// Shared memory per block, sized so three blocks fit an SM's 100 KB (with the 1 KB the driver
// keeps per block) AND all 180 angles of a 40 x 40 map (83 bins each) fit one chunk:
// 15,000 vote counters (30,000 B) + a 512-entry fail list (2 KB) + ~80 B ≈ 33.1 KB a block.
// With the earlier 24 KB budget a 40 x 40 map took 148 + 32 angles: in the second chunk one
// warp voted while five waited at the barrier (Nsight Compute: ~40% barrier stalls).
constexpr unsigned kTile = 512;            // grid positions per tile, so <= 512 fail dies
constexpr unsigned kVotes = 15000;         // 16-bit counters per chunk
constexpr int kShift = kHoughFracBits + 1; // Q14, and doubled coordinates: one die per bin

// 16-bit counters are enough: a counter holds the fail dies of a one-die-wide strip across
// the wafer, a few hundred at most even on WM-811K's largest 212 x 204 map.

__constant__ std::int32_t c_cos[kAngles];
__constant__ std::int32_t c_sin[kAngles];

// A candidate line as one number to maximise: more votes first; on equal votes the lower
// angle, then the lower bin. That is the CPU's rule (it scans angle by angle, bin by bin,
// and keeps the first maximum), so the GPU picks the same line among ties.
__device__ unsigned long long line_key(unsigned votes, unsigned angle, unsigned bin) {
    return (static_cast<unsigned long long>(votes) << 32) | (0xFFFFFFFFU - ((angle << 16) | bin));
}

// The maximum key over the block: within each warp by shuffles (lanes read each other's
// registers, no memory), then over the 6 warp results through shared memory.
__device__ unsigned long long block_max(unsigned long long key, unsigned long long* scratch) {
    for (unsigned delta = 16; delta > 0; delta /= 2) {
        key = max(key, __shfl_down_sync(0xFFFFFFFFU, key, delta));
    }
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = key;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned w = 1; w < kWarps; ++w) {
            key = max(key, scratch[w]);
        }
        scratch[0] = key;
    }
    __syncthreads();
    const unsigned long long result = scratch[0];
    __syncthreads(); // scratch is reused by the next call
    return result;
}

__global__ void __launch_bounds__(kThreads)
    hough_kernel(const detail::MapDesc* descs, const std::uint8_t* bins, HoughLine* out) {
    __shared__ std::uint16_t votes[kVotes];
    __shared__ short2 fails[kTile]; // doubled (x, y) of a tile's fail dies
    __shared__ unsigned fail_count;
    __shared__ unsigned line_dies;
    __shared__ unsigned long long scratch[kWarps];

    const detail::MapDesc d = descs[blockIdx.x];
    const std::uint8_t* map = bins + d.bins;
    const int rows = d.rows;
    const int cols = d.cols;
    const int offset = (rows + cols) / 2 + 1; // as on the CPU: |rho| <= offset
    const unsigned row_length = 2U * static_cast<unsigned>(offset) + 1U;
    // As many angles per chunk as fit the vote budget: all 180 up to rows + cols = 80 (every
    // WaferLens map, most of WM-811K), 43 for WM-811K's largest. Big maps take more chunks.
    const unsigned per_chunk = min(min(kAngles, kThreads), kVotes / row_length);

    unsigned long long best = 0;
    for (unsigned first = 0; first < kAngles; first += per_chunk) {
        for (unsigned i = threadIdx.x; i < per_chunk * row_length; i += kThreads) {
            votes[i] = 0;
        }
        const unsigned angle = first + threadIdx.x;
        const bool owner = threadIdx.x < per_chunk && angle < kAngles;
        // Each owner reads its cos / sin once and keeps them in registers.
        const int c = owner ? c_cos[angle] : 0;
        const int s = owner ? c_sin[angle] : 0;

        for (unsigned tile = 0; tile < d.dies; tile += kTile) {
            if (threadIdx.x == 0) {
                fail_count = 0;
            }
            __syncthreads(); // the counter is reset (and on the first tile, the votes zeroed)
            // All threads gather the tile's fail dies; the order in the list doesn't matter,
            // since the votes are counts.
            const unsigned end = min(tile + kTile, d.dies);
            for (unsigned i = tile + threadIdx.x; i < end; i += kThreads) {
                if (map[i] >= 2) {
                    const int r = static_cast<int>(i) / cols;
                    const int col = static_cast<int>(i) % cols;
                    fails[atomicAdd(&fail_count, 1U)] =
                        make_short2(static_cast<short>(2 * col - (cols - 1)),
                                    static_cast<short>((rows - 1) - 2 * r));
                }
            }
            __syncthreads(); // the list is complete
            if (owner) {
                std::uint16_t* row = votes + threadIdx.x * row_length;
                const unsigned n = fail_count;
                // Every owner reads the same fails[j] at the same time: one shared-memory
                // broadcast to the warp. Each writes only its own row: no atomics.
                for (unsigned j = 0; j < n; ++j) {
                    const short2 p = fails[j];
                    ++row[((p.x * c + p.y * s) >> kShift) + offset];
                }
            }
            __syncthreads(); // done with this tile's list
        }

        unsigned long long key = 0;
        if (owner) {
            const std::uint16_t* row = votes + threadIdx.x * row_length;
            unsigned top = 0;
            unsigned top_bin = 0;
            for (unsigned b = 0; b < row_length; ++b) {
                if (row[b] > top) { // strictly: the first maximum, as on the CPU
                    top = row[b];
                    top_bin = b;
                }
            }
            if (top > 0) {
                key = line_key(top, angle, top_bin);
            }
        }
        // block_max ends with __syncthreads, so the next chunk may zero the votes.
        best = max(best, block_max(key, scratch));
    }

    // best is the same in every thread, so this branch (and its barriers) is uniform.
    HoughLine line{};
    if (best != 0) {
        const unsigned packed = 0xFFFFFFFFU - static_cast<unsigned>(best & 0xFFFFFFFFULL);
        const unsigned angle = packed >> 16;
        const int rho = static_cast<int>(packed & 0xFFFFU) - offset;
        // How many on-wafer dies the line crosses: the most votes it could have had.
        if (threadIdx.x == 0) {
            line_dies = 0;
        }
        __syncthreads();
        const int c = c_cos[angle];
        const int s = c_sin[angle];
        unsigned mine = 0;
        for (unsigned i = threadIdx.x; i < d.dies; i += kThreads) {
            if (map[i] != 0) {
                const int x = 2 * (static_cast<int>(i) % cols) - (cols - 1);
                const int y = (rows - 1) - 2 * (static_cast<int>(i) / cols);
                mine += ((x * c + y * s) >> kShift) == rho ? 1U : 0U;
            }
        }
        atomicAdd(&line_dies, mine);
        __syncthreads();
        line.angle = static_cast<int>(angle);
        line.rho = rho;
        line.votes = static_cast<unsigned>(best >> 32);
        line.line_dies = line_dies;
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = line;
    }
}

} // namespace

cudaError_t detail::upload_hough_tables() {
    cudaError_t e = cudaMemcpyToSymbol(c_cos, hough_cos().data(), sizeof(c_cos));
    if (e == cudaSuccess) {
        e = cudaMemcpyToSymbol(c_sin, hough_sin().data(), sizeof(c_sin));
    }
    // Ask for the largest shared-memory share of the SM's L1 / shared split (the "carveout"),
    // so three 33 KB blocks fit. Once, here, not per launch: every driver call costs on WSL2.
    if (e == cudaSuccess) {
        e = cudaFuncSetAttribute(hough_kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared);
    }
    return e;
}

cudaError_t detail::launch_hough(unsigned count, const MapDesc* descs, const std::uint8_t* bins,
                                 HoughLine* out, cudaStream_t stream) {
    hough_kernel<<<count, kThreads, 0, stream>>>(descs, bins, out);
    return cudaGetLastError();
}

} // namespace waferedge::gpu
