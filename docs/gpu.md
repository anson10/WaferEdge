# GPU backend (phase 2a)

What runs on the GPU, how it is checked, and what it costs, measured. The short version:

- **Features**: the GPU computes 65,536 maps in **3 ms** (~22M maps/s), but getting the maps
  there takes 10× longer, so end to end it reaches **~1.6M maps/s, about 1.7× one CPU core**
  with AVX2, from batches of ~300 maps up. Too little work per byte to be worth PCIe.
- **Hough transform**: 180 votes per fail die is enough work per byte. The GPU passes one CPU
  core **from a batch of 4 maps**, and reaches **~1.2M maps/s, ~84× one core** (26–33× on
  the real datasets end to end). Equal to the CPU's line on all 103,698 real maps.
- **Clusters** (lock-free union-find): passes one core at ~64 maps, ~1.3M maps/s (~12×).
- **All three from one upload**: ~0.82M maps/s, **~66× one CPU core** doing the same work
  (~80 µs a map), ahead from a batch of ~4 maps.

Machine: RTX 3050 6 GB Laptop (sm_86, 20 SMs), driver / runtime 12.4, WSL2; Ryzen 5 7535HS.

## The model in one screen

A kernel launch is a **grid of blocks**; each block runs on one of the 20 SMs; its threads
run in **warps of 32** that execute one instruction together (SIMT: you write one thread's
code, the hardware runs it 32 wide, like the 32 byte lanes of an AVX2 register). An SM keeps
many warps in flight and switches to a ready one whenever another waits for memory: latency
is hidden by **occupancy**, not by large caches.

| Memory | Size here | Rough cost | Visible to |
|---|---|---|---|
| Registers | 64K × 32 bit per SM | 1 cycle | one thread |
| Shared memory | up to 100 KB per SM | ~20–30 cycles | one block |
| L2 | 2 MB | ~200 cycles | all |
| Global (GDDR6) | 6 GB | ~400+ cycles | all |
| PCIe to the host | | 11.7 GB/s pinned in this benchmark; ~20–90 µs per CUDA call on WSL2 | |

Reads are **coalesced** when the 32 lanes of a warp read consecutive addresses: one memory
transaction instead of 32.

## Kernel 1: features, one block per map

`cuda/features.cu`. The host packs a batch into one pinned buffer (a 16-byte descriptor per
map, then the maps' bins), so a batch costs one upload, one kernel, one download and one sync
whatever its size. Geometry tables are uploaded once per `Geometry` (cached by
`Geometry::id`, which is why `Geometry` became move-only with a unique id).

Block *b* handles map *b* with 256 threads. Thread *t* reads dies *t*, *t*+256, ... (each
warp reads 32 consecutive bytes: coalesced), looks up the die's zone, sector and ring, and
counts it into 46 counters in **shared memory**; at the end the block writes its 48 words
(the `Features` struct's layout, checked with `static_assert`s) to global memory.

Two versions of the counting, as with AVX2:

- **v1, shared atomics** (`ATOMS.ADD` in the SASS): every on-wafer die does three
  `atomicAdd`s (six if it fails). Lanes of a warp that hit the same counter (5 zones for 32
  lanes) are serialised by the hardware.
- **v2, warp-aggregated** (`MATCH.ANY` + `ATOMS.ADD`): `__match_any_sync` groups the lanes
  that hold the same bucket; one leader per group adds the group's size. One atomic per
  distinct bucket instead of one per lane: the AVX2 trick of counting a mask, in warp form.

**v2 is about 2× slower** (kernel time for 65,536 maps: 6.0–6.3 ms against 3.0–3.7 ms).
Nsight Compute on one launch of 4,096 maps says why (`ncu --set full`, base clock):

| | v1 shared atomics | v2 warp-aggregated |
|---|---|---|
| Kernel time | 216 µs | 503 µs |
| Warp instructions executed | 7.6 M | 23.4 M (3.1×) |
| Shared-memory atomic wavefronts | 1.18 M | 1.69 M (more, not fewer) |
| Top stall | long_scoreboard (global loads) | short_scoreboard (waiting on `MATCH.ANY`) |

In v1 an `atomicAdd` from 32 lanes is *one* warp instruction; the hardware already merges
lanes that hit the same address into a few internal passes. v2 replaced it with a
`MATCH.ANY` (variable latency: the short_scoreboard stalls) plus one atomic per group leader,
3–5 single-lane atomics where v1 had one instruction. Hand-made warp aggregation was a win on
older GPUs; on Ampere the hardware does it better. v1 is the default; v2 stays as the
measured, documented experiment.

### What limits v1 (Nsight Compute)

Occupancy is not the problem (100% theoretical, 95% achieved; 18 registers per thread). The
**LSU pipe** (load / store unit: every global load and shared atomic) is at 92% while DRAM is
at 28% (37 GB/s of ~190): too many small memory *instructions* (four byte loads and three to
six atomics per die), not too many bytes. Warps wait on global loads (long_scoreboard, ~50% of
stall samples) and on a full memory-instruction queue (mio_throttle). Active threads per warp
average 25.2 of 32 — that is π/4 = 78.5% of 32: the off-wafer corners of a disc in a square
grid skip the work. The textbook next step (4 dies per load with `uchar4`) isn't worth taking:
the kernel is 7% of the batch's time.

## Correctness

- `tests/test_gpu_features.cpp` (label `gpu`, run locally: `ctest --preset cuda -L gpu`):
  both kernels against the scalar reference with `==` on a batch of 50 maps of 10 shapes and
  5 densities; one map per call through `backend::Cuda`; a whole map in one bucket (maximum
  contention); every byte value as a bin; a 300×300 map; the real fixture as one batch; empty
  batches; buffer growth from 1 to 5,000 maps and back with cached geometry.
- `build/cuda/tools/waferedge-maps <file>` runs every map of a file through both kernels:
  **equal to scalar on 24,090 of 24,090 WaferLens maps and 79,608 of 79,608 WM-811K maps.**
- Not yet: `compute-sanitizer` (the GPU's ASan) cannot attach on this WSL2 setup ("Failed to
  initialize WDDM debugger interface"; it needs `EnableDebuggerInterface.bat` from the
  Windows CUDA toolkit). The full-output comparisons above would catch any out-of-bounds read
  that changed a count, not one that didn't; to be revisited before the CNN engine (2b).

## Crossover: CPU vs GPU by batch size

`build/cuda/bench/bench-gpu`: 40×40 random maps (10% fails). One iteration = one batch end to
end (wall time); CPU = AVX2, one thread. Median of 3 repetitions.

| Batch | CPU AVX2, 1 thread | GPU v1 | GPU latency per batch |
|---|---|---|---|
| 1 | 936k maps/s | 4.8k | 208 µs |
| 16 | 913k | 74k | 216 µs |
| 64 | 871k | 264k | 243 µs |
| 256 | 905k | 805k | 318 µs |
| 1,024 | 871k | 1.44M | 709 µs |
| 4,096 | 908k | 1.50M | 2.7 ms |
| 65,536 | 880k–908k | 1.58–1.60M | 41 ms |

**The GPU passes one CPU core at ~300 maps per batch and levels off at ~1.6M maps/s.** The
comparison against all CPU cores (the roadmap's "AVX2, all threads") is still to come; with
~6 cores at ~0.9M maps/s each, a multi-threaded CPU would likely beat the GPU at every batch
size for this kernel.

### Where the time goes

Same benchmark with CUDA events (`BM_gpu_*_timed`); the timing itself adds ~60 µs per batch
(below), so these rows explain the breakdown, they are not the headline:

| Batch | Pack (CPU) | Upload | Kernel | Download | Total |
|---|---|---|---|---|---|
| 1 | 0 µs | 23 µs | 83 µs | 89 µs | 304 µs |
| 256 | 29 µs | 43 µs | 92 µs | 91 µs | 361 µs |
| 4,096 | 1.4 ms | 0.58 ms | 0.33 ms | 0.18 ms | 2.9 ms |
| 65,536 | **26 ms (63%)** | 9.0 ms | **3.0 ms (7%)** | 1.2 ms | 41 ms |

- **Small batches are all overhead.** A single 1,600-die map shows 83 µs of "kernel": that
  is driver and scheduling time on WSL2, not computation.
- **Large batches are all data movement.** Copying 65,536 maps (105 MB, from 65,536 separate
  heap buffers) into the pinned buffer takes 26 ms on the CPU, the upload 9 ms, the kernel 3.
  The fix is not a faster kernel but not copying: maps that arrive from the network straight
  into pinned memory (phase 4's zero-copy receive path) remove the largest step; packing
  each die into 2 bits (only "on wafer" and "fail" matter here) would cut the upload 4×.

### A measurement trap: the timer changed the result

The first version recorded four CUDA events around every batch. Nsight Systems
(`nsys profile -t cuda`) showed what each CUDA call costs on WSL2, averaged over 839 batches
of 4 maps:

| Call | Average | Per batch |
|---|---|---|
| `cudaStreamSynchronize` (waiting for the GPU) | 220 µs | 1 |
| `cudaLaunchKernel` | 53 µs (native Linux: ~5 µs) | 1 |
| `cudaMemcpyAsync` | 26 µs | 2 |
| `cudaEventRecord` | 15 µs | 4 → ~60 µs |

The events were a quarter of a small batch's cost. Timing is now opt-in
(`SignatureEngine::set_timing`); without it a batch of one went from 297 µs to 208 µs and the
large-batch plateau from ~1.3M to ~1.6M maps/s. The remaining four calls per batch (copy,
launch, copy, sync) are the next target: a CUDA Graph replays them as one call.

On WSL2, Nsight Systems records the CPU side of CUDA calls but not the GPU's own kernel
timeline. Nsight Compute needs GPU performance-counter access enabled on the Windows side
(NVIDIA Control Panel → Developer → Manage GPU Performance Counters → allow all users; check
with `reg.exe query "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" /v
RmProfilingAdminOnly`, which must be `0x0`). It replays each kernel ~38 times and locks the
clock to base (1.16 GHz here), so its durations explain, the benchmarks measure.

## Reproduce

```sh
cmake --workflow --preset cuda                       # builds, runs all tests incl. gpu
ctest --preset cuda -L gpu                           # GPU tests only
build/cuda/tools/waferedge-maps data/wm811k_lot.wmap # GPU == CPU on every map, timings
build/cuda/bench/bench-gpu --benchmark_filter='BM_(cpu_avx2|gpu_shared_atomics)/' \
  --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
build/cuda/bench/bench-gpu --benchmark_filter=_timed/ --benchmark_repetitions=3 \
  --benchmark_report_aggregates_only=true             # the breakdown
build/cuda/bench/bench-gpu --benchmark_filter='BM_(cpu|gpu)_hough/' --benchmark_repetitions=3 \
  --benchmark_report_aggregates_only=true             # Hough crossover
ncu --set full --kernel-name regex:hough --launch-skip 1 --launch-count 1 -o hough \
  build/cuda/bench/bench-gpu --benchmark_filter='BM_gpu_hough/4096/' --benchmark_min_time=1x
ncu -i hough.ncu-rep | less                           # the profile, as text (or ncu-ui)
build/cuda/bench/bench-gpu --benchmark_filter='BM_(cpu_clusters|gpu_clusters|gpu_all_signatures)/' \
  --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
nsys profile -t cuda -o small build/cuda/bench/bench-gpu \
  --benchmark_filter='BM_gpu_shared_atomics/4/' && nsys stats -r cuda_api_sum small.nsys-rep
```

## Kernel 3: the Hough transform, one thread per angle

`cuda/hough_kernel.cu`. The CPU version has every fail die vote once per angle (180 angles,
Q14 fixed-point cos / sin, doubled integer coordinates, `src/hough.cpp`). The obvious GPU
port, one thread per die with `atomicAdd` into the vote table, puts its worst contention on
the peak: every die of a scratch votes for the same counter at the same angle.

Turned around: **thread k owns angle k's row of the vote table.** No two threads write the
same counter, so the voting loop has no atomics and the result is deterministic.

1. The block gathers the map's fail dies (doubled x, y) into a list in shared memory, a tile
   of 512 grid positions at a time.
2. Thread k loads cos[k], sin[k] from constant memory into registers once, then walks the
   list: `rho = (x·cos + y·sin) >> 15`, `++votes[k][rho]`. All threads read the same list
   entry at once (a shared-memory broadcast); each writes only its own row.
3. Each thread finds its row's first maximum; a block reduction (warp shuffles, then shared
   memory) picks the line with the most votes, ties to the lowest angle, then the lowest
   offset: the CPU's rule, encoded as one 64-bit key to maximise.
4. One more pass counts the on-wafer dies in the winning bin (`line_dies`).

Votes are 16-bit: a counter holds the fail dies of a one-die-wide strip, a few hundred at
most, even on WM-811K's 212×204 maps. Angles are processed in chunks that fit the vote budget:
all 180 for maps up to rows + cols = 80, down to 43 per chunk for WM-811K's largest.

**Correctness.** `tests/test_gpu_signatures.cpp`: lines equal to the CPU's (`==` on angle,
offset, votes, dies on the line) on 70 maps of 7 shapes, 5 densities, with and without
scratches; maps that need several chunks (212×204, 300×300, 97×131); the tie cases (no
fails; one fail die, where all 180 angles tie; two equal parallel scratches; a short
scratch's plateau of equal-vote angles); features and Hough from one upload equal to each
alone. Breaking the tie rule on purpose fails the tie test (`gpu angle 179 ... cpu angle 0`).
`waferedge-maps`: **equal on 24,090 of 24,090 WaferLens and 79,608 of 79,608 WM-811K maps.**

### One optimisation iteration, driven by the profile

The first version gave the vote table 24 KB of shared memory: three blocks per SM, but only
148 of a 40×40 map's 180 angles per chunk (83 bins a row). Nsight Compute:

| | 24 KB votes, 1,024-die list | 30 KB votes, 512-die list |
|---|---|---|
| Angles per chunk (40×40) | 148, then 32 | 180 |
| Blocks per SM (100 KB shared) | 3 | 3 |
| Kernel, 4,096 maps (base clock) | 3.03 ms | **2.04 ms (−33%)** |
| Barrier stalls | 36% of samples | 12% |
| Top stalls after | | short_scoreboard 28%, wait 25% |

In the second chunk one warp voted while five waited at the barrier. Sizing the table for all
180 angles of a 40×40 map, and the fail list down to fit three 33 KB blocks in 100 KB (with
the shared-memory carveout set to its maximum once, at start-up), removed the second chunk.
End to end the plateau rose from 1.03M to 1.22M maps/s (+19%: the kernel is part of the
batch). Still open: 35% of shared-memory accesses hit bank conflicts (rows of 83 16-bit
counters; padding the row stride is the usual fix).

### Crossover

`build/cuda/bench/bench-gpu --benchmark_filter='BM_(cpu|gpu)_hough/'`, 40×40 maps, 10% fails,
median of 3:

| Batch | CPU Hough, 1 thread | GPU Hough | GPU latency |
|---|---|---|---|
| 1 | 14.6k maps/s | 5.0k | 199 µs |
| 4 | 14.1k | **20.8k** | 192 µs |
| 64 | 14.1k | 263k | 244 µs |
| 1,024 | 14.5k | 1.22M | 842 µs |
| 65,536 | (~14.5k) | 1.15–1.22M | 54–57 ms |

Real data, batches of 4,096, every copy included (`waferedge-maps`): WaferLens 651k maps/s
against 19.9k on one CPU core (33×); WM-811K 401k against 15.4k (26×). With both kernels on
one upload, features add little (WaferLens: 608k maps/s for features + Hough).

**Why the Hough wins and the features don't**: arithmetic intensity. The CPU spends ~69 µs a
map on the Hough transform, so the ~200 µs a GPU call costs is paid back after 3–4 maps; it
spends ~1 µs on the features, so it takes ~300 maps and the gain stays small.

## Kernel 2: connected clusters, lock-free union-find

`cuda/cluster_kernel.cu`. Clusters are the hard one to parallelise: whether two dies belong
together can depend on a chain across the whole map (a U shape's arms meet only in its last
row). The CPU (`src/clusters.cpp`) runs union-find in scan order and always hangs the larger
root under the smaller, so a cluster's root is its first die. The GPU keeps that rule and
runs every union at once (the Playne–Hawick approach), one block per map:

1. every fail die is its own tree (`parent[i] = i`);
2. each fail die unites with its fail neighbours W, NW, N, NE: find both roots, then
   `old = atomicMin(&parent[larger], smaller)`; if `old` was still the larger root, done,
   otherwise another thread re-linked it meanwhile, so retry from `old`. Links only ever
   point to smaller indices, so no chain can loop, and when every thread is done each root is
   the smallest index of its cluster: the CPU's root;
3. every die points straight at its root (path compression);
4. sizes counted at the roots; the largest cluster (ties to the smallest root, the CPU's
   first) by block reduction; its integer moments and bounding box summed with shared atomics.

The block returns a `ClusterSummary` (count + the largest `Cluster`, what the classifier
uses), compared with `ClusterFinder::summary()` using `==`. Trees live in shared memory for
maps up to 4,096 positions (64×64: all of WaferLens, 97% of WM-811K); larger maps use a slice
of a global scratch, allocated only for batches that contain one. `find_root` reads through
`volatile` because other threads change links during the walk.

**Correctness.** 42 maps of 7 shapes × 6 densities; the hard cases: snakes (one cluster
winding through the map: longest chains, latest merges), an all-fail map, a checkerboard
(one cluster joined only diagonally), 400 tied single-die clusters, two equal blobs; maps over
the shared-memory limit (65×65, 212×204, a 150×150 snake). Breaking the tie rule fails the
test on exactly the two tie maps. `waferedge-maps`: **equal on 24,090 of 24,090 WaferLens and
79,608 of 79,608 WM-811K maps** (including WM-811K's maps over 4,096 positions).

**What bounds it** (Nsight Compute, 4,096 maps, 1.66 ms at base clock): divergence. Only 12.7
of 32 threads per warp are active on average: ~10% of dies fail and every phase skips the
rest, and `unite` loops a different number of times per lane (14% branch-resolving stalls,
26% barrier between the four phases). Occupancy 50% (shared memory). Gathering the fail dies
into a dense list first, as the Hough kernel does, is the obvious next step; at ~20% of an
all-signatures batch it isn't the bottleneck yet.

### All signatures, one upload

`bench-gpu`, 40×40 maps, median of 3, end to end:

| Batch | CPU clusters, 1 thread | GPU clusters | GPU features + Hough + clusters | its latency |
|---|---|---|---|---|
| 1 | 200k maps/s | 5.5k | 5.1k | 198 µs |
| 4 | 176k | 23.6k | 15.8k | 254 µs |
| 64 | 107k | 273k | 185k | 345 µs |
| 1,024 | 106k | 1.27M | 807k | 1.27 ms |
| 65,536 | (~104k) | 1.31–1.32M | 0.82–0.83M | 79 ms |

One CPU core needs ~80 µs per map for all three (AVX2 features ~1 µs, Hough ~69 µs, clusters
~9.5 µs): ~12.5k maps/s. The GPU passes it at a batch of ~4 and is ~66× faster at large
batches. Real data, batches of 4,096, all copies included: WaferLens 482k maps/s, WM-811K
247k (`waferedge-maps`, which also checks every map against the CPU).

## Next in phase 2a

- CPU on all cores in the crossover benchmarks.
- Streams overlapping the upload of one batch with the kernels of the previous; CUDA Graphs
  for the per-batch call overhead; the Hough kernel's bank conflicts.
