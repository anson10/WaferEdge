// Connected clusters of fail dies on the GPU: lock-free union-find, one block per map
// (docs/gpu.md). The same clusters as src/clusters.cpp (8-connected), and the same canonical
// root, a cluster's smallest index (its first die in row-major order), so the GPU's summary
// is compared with the CPU's with ==.
#include "kernels.cuh"

#include <climits>
#include <cstdint>

namespace waferedge::gpu {

namespace {

constexpr unsigned kThreads = 256;
constexpr unsigned kWarps = kThreads / 32;
constexpr unsigned kShared = detail::kClusterSharedPositions;
constexpr int kNoDie = -1; // parent of a die that doesn't fail

// The root of i's tree. Parent links always point to a smaller index, so the walk ends.
// volatile: other threads change links while we walk, and every read must see memory, not a
// value the compiler kept in a register from an earlier iteration.
__device__ int find_root(const volatile int* parent, int i) {
    int p = parent[i];
    while (p != i) {
        i = p;
        p = parent[i];
    }
    return i;
}

// Joins the trees of a and b, with any number of other threads doing the same at once.
// Hang the larger root under the smaller with atomicMin: if b was still a root, done; if
// another thread linked b meanwhile, atomicMin returned b's new parent, so retry from there.
__device__ void unite(int* parent, int a, int b) {
    while (true) {
        a = find_root(parent, a);
        b = find_root(parent, b);
        if (a == b) {
            return;
        }
        if (a > b) {
            const int t = a;
            a = b;
            b = t;
        }
        const int old = atomicMin(&parent[b], a);
        if (old == b) {
            return;
        }
        b = old;
    }
}

// Block-wide maximum of a 64-bit key (warp shuffles, then the warp results in shared memory).
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
    __syncthreads();
    return result;
}

__device__ bool fails(const std::uint8_t* map, int i) {
    return map[i] >= 2;
}

__global__ void __launch_bounds__(kThreads)
    clusters_kernel(const detail::MapDesc* descs, const std::uint8_t* bins, int* scratch,
                    ClusterSummary* out) {
    // Maps up to 4096 positions (64 x 64) keep their trees in shared memory (32 KB); larger
    // ones use their slice of the global scratch (8 bytes per position, laid out like bins).
    __shared__ int s_parent[kShared];
    __shared__ unsigned s_size[kShared];
    __shared__ unsigned long long s_keys[kWarps];
    __shared__ unsigned s_clusters;
    __shared__ unsigned long long s_sum[5]; // r, c, r^2, c^2, r*c of the largest cluster
    __shared__ int s_box[4];                // min r, max r, min c, max c

    const detail::MapDesc d = descs[blockIdx.x];
    const std::uint8_t* map = bins + d.bins;
    const int n = static_cast<int>(d.dies);
    const int cols = d.cols;
    int* parent = s_parent;
    unsigned* size = s_size;
    if (d.dies > kShared) {
        parent = scratch + 2 * static_cast<std::size_t>(d.bins);
        size = reinterpret_cast<unsigned*>(parent + n);
    }
    const int t = static_cast<int>(threadIdx.x);
    const int stride = static_cast<int>(blockDim.x);

    // 1. Every fail die is its own tree.
    for (int i = t; i < n; i += stride) {
        parent[i] = fails(map, i) ? i : kNoDie;
        size[i] = 0;
    }
    if (t == 0) {
        s_clusters = 0;
        for (auto& v : s_sum) {
            v = 0;
        }
        s_box[0] = s_box[2] = INT_MAX;
        s_box[1] = s_box[3] = -1;
    }
    __syncthreads();

    // 2. Join each fail die with the fail neighbours before it (W, NW, N, NE), as the CPU
    // does; every pair of touching dies is joined once, from the later one.
    for (int i = t; i < n; i += stride) {
        if (!fails(map, i)) {
            continue;
        }
        const int r = i / cols;
        const int c = i % cols;
        if (c > 0 && fails(map, i - 1)) {
            unite(parent, i, i - 1);
        }
        if (r > 0) {
            for (int dc = -1; dc <= 1; ++dc) {
                const int cc = c + dc;
                if (cc >= 0 && cc < cols && fails(map, i - cols + dc)) {
                    unite(parent, i, i - cols + dc);
                }
            }
        }
    }
    __syncthreads();

    // 3. Point every die straight at its root (writes only shorten chains, so concurrent
    // walks still end at the same root).
    for (int i = t; i < n; i += stride) {
        if (parent[i] != kNoDie) {
            parent[i] = find_root(parent, i);
        }
    }
    __syncthreads();

    // 4. Size of each cluster, counted at its root; the number of roots is the cluster count.
    for (int i = t; i < n; i += stride) {
        const int root = parent[i];
        if (root != kNoDie) {
            atomicAdd(&size[root], 1U);
            if (root == i) {
                atomicAdd(&s_clusters, 1U);
            }
        }
    }
    __syncthreads();

    // 5. The largest cluster, ties to the smallest root (the CPU's first cluster of max size).
    unsigned long long key = 0;
    for (int i = t; i < n; i += stride) {
        if (parent[i] == i) {
            key = max(key, (static_cast<unsigned long long>(size[i]) << 32) |
                               (0xFFFFFFFFULL - static_cast<unsigned>(i)));
        }
    }
    const unsigned long long best = block_max(key, s_keys);

    // 6. Its integer moments and bounding box (the same values ClusterFinder sums).
    ClusterSummary summary{};
    if (best != 0) { // uniform across the block
        const int root = static_cast<int>(0xFFFFFFFFULL - (best & 0xFFFFFFFFULL));
        for (int i = t; i < n; i += stride) {
            if (parent[i] == root) {
                const long long r = i / cols;
                const long long c = i % cols;
                atomicAdd(&s_sum[0], static_cast<unsigned long long>(r));
                atomicAdd(&s_sum[1], static_cast<unsigned long long>(c));
                atomicAdd(&s_sum[2], static_cast<unsigned long long>(r * r));
                atomicAdd(&s_sum[3], static_cast<unsigned long long>(c * c));
                atomicAdd(&s_sum[4], static_cast<unsigned long long>(r * c));
                atomicMin(&s_box[0], static_cast<int>(r));
                atomicMax(&s_box[1], static_cast<int>(r));
                atomicMin(&s_box[2], static_cast<int>(c));
                atomicMax(&s_box[3], static_cast<int>(c));
            }
        }
        __syncthreads();
        summary.clusters = s_clusters;
        summary.largest.size = static_cast<std::uint32_t>(best >> 32);
        summary.largest.first = root;
        summary.largest.sum_r = static_cast<std::int64_t>(s_sum[0]);
        summary.largest.sum_c = static_cast<std::int64_t>(s_sum[1]);
        summary.largest.sum_rr = static_cast<std::int64_t>(s_sum[2]);
        summary.largest.sum_cc = static_cast<std::int64_t>(s_sum[3]);
        summary.largest.sum_rc = static_cast<std::int64_t>(s_sum[4]);
        summary.largest.min_r = static_cast<std::uint16_t>(s_box[0]);
        summary.largest.max_r = static_cast<std::uint16_t>(s_box[1]);
        summary.largest.min_c = static_cast<std::uint16_t>(s_box[2]);
        summary.largest.max_c = static_cast<std::uint16_t>(s_box[3]);
    }
    if (t == 0) {
        out[blockIdx.x] = summary;
    }
}

} // namespace

cudaError_t detail::launch_clusters(unsigned count, const MapDesc* descs, const std::uint8_t* bins,
                                    int* scratch, ClusterSummary* out, cudaStream_t stream) {
    clusters_kernel<<<count, kThreads, 0, stream>>>(descs, bins, scratch, out);
    return cudaGetLastError();
}

} // namespace waferedge::gpu
