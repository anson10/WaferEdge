# GPU backend (phase 2a)

What runs on the GPU, how it is checked, and what it costs, measured. The short version:

- **Features**: the GPU computes 65,536 maps in **4 ms** (~16M maps/s), but packing and
  uploading them takes 6× longer, so end to end it peaks at **~2.7M maps/s at 1,024 maps, at
  most ~1.3× one CPU core** with AVX2 (2.1M), and falls below that core again at 65,536 maps.
  Too little work per byte to be worth PCIe.
- **Hough transform**: 180 votes per fail die is enough work per byte. The GPU passes one CPU
  core **between 4 and 16 maps a batch**, and reaches **~1.7M maps/s, ~49× one core** (~20× on
  the real datasets end to end). Equal to the CPU's line on all 103,698 real maps.
- **Clusters** (lock-free union-find): passes one core at ~64 maps, ~1.8M maps/s (~8×).
- **All three from one upload**: ~0.94M maps/s, ~32× one CPU core, and **~6× the whole
  CPU** (6 cores, 12 threads: ~150k maps/s), level with the whole CPU at **~16 maps** a batch.
- **Where things run** follows from that (ADR-0007): features on the CPU always (the whole
  CPU does 8.8M maps/s, the GPU never wins), Hough and clusters on the GPU from 16 maps.

Machine: RTX 3050 6 GB Laptop (sm_86, 20 SMs), WSL2; Ryzen 5 7535HS. Benchmarks: WSL2 Ubuntu
26.04, CUDA 13.3, driver 621.14, GCC 13.4, re-measured 2026-11-10 (ADR-0020). Sections marked
*CUDA 12.4* are Nsight Compute profiles and history from the first build (Ubuntu 22.04, driver
551.76); the laptop then ran at lower clocks, so their absolute times are higher.

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

**v2 is slower** (kernel time for 65,536 maps: 5.8 ms against 4.2 ms, `BM_gpu_*_timed`; on
CUDA 12.4 it was 6.0–6.3 against 3.0–3.7 ms). Nsight Compute on one launch of 4,096 maps says
why (`ncu --set full`, base clock; *CUDA 12.4*):

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

### What limits v1 (Nsight Compute, *CUDA 12.4*)

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
| 1 | 2.14M maps/s | 13.8k | 72 µs |
| 16 | 2.13M | 208k | 77 µs |
| 64 | 2.10M | 453k | 141 µs |
| 256 | 2.08M | 1.35M | 189 µs (CV 72%) |
| 1,024 | 2.11M | **2.74M** | 374 µs |
| 4,096 | 2.08M | 2.13M | 1.9 ms |
| 65,536 | 2.07M | 1.54M | 43 ms |

**The GPU passes one CPU core only between ~1,000 and ~4,000 maps per batch, by at most
1.3×**, and falls behind again at 65,536, where packing the maps dominates (next table).
Against all CPU cores (below) it never wins. (On CUDA 12.4, with the laptop at lower clocks,
one core did ~0.9M maps/s and the GPU passed it at ~300 maps, levelling off at ~1.6M.)

### Where the time goes

Same benchmark with CUDA events (`BM_gpu_*_timed`); the timing itself adds ~20–45 µs per
batch (below), so these rows explain the breakdown, they are not the headline:

| Batch | Pack (CPU) | Upload | Kernel | Download | Total |
|---|---|---|---|---|---|
| 1 | 0 µs | 15 µs | 20 µs | 12 µs | 115 µs |
| 256 | 13 µs | 81 µs | 71 µs | 11 µs | 208 µs |
| 4,096 | 0.84 ms | 0.52 ms | 0.22 ms | 0.11 ms | 1.84 ms |
| 65,536 | **17 ms (45%)** | 9.1 ms | **4.2 ms (11%)** | 1.1 ms | 39 ms |

- **Small batches are all overhead.** A single 1,600-die map shows 20 µs of "kernel" (83 µs
  on CUDA 12.4 and the older driver): driver and scheduling time on WSL2, not computation.
- **Large batches are all data movement.** Copying 65,536 maps (105 MB, from 65,536 separate
  heap buffers) into the pinned buffer takes 17 ms on the CPU, the upload 9 ms, the kernel 4.
  The fix is not a faster kernel but not copying: maps that arrive from the network straight
  into pinned memory (phase 4's zero-copy receive path) remove the largest step; packing
  each die into 2 bits (only "on wafer" and "fail" matter here) would cut the upload 4×.

### A measurement trap: the timer changed the result (*CUDA 12.4*)

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
build/cuda/bench/bench-gpu --benchmark_filter='BM_cpu_(features|all_signatures)_threads/' \
  --benchmark_repetitions=3 --benchmark_report_aggregates_only=true    # the whole CPU
build/cuda/bench/bench-gpu --benchmark_filter='BM_gpu_(all_signatures|all_signatures_graph)/' \
  --benchmark_repetitions=3 --benchmark_report_aggregates_only=true    # CUDA Graphs
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
148 of a 40×40 map's 180 angles per chunk (83 bins a row). Nsight Compute (*CUDA 12.4*):

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
| 1 | 34.7k maps/s | 5.5k | 183 µs |
| 4 | 34.3k | 23.3k | 172 µs |
| 16 | 34.5k | **92.9k** | 172 µs |
| 64 | 34.2k | 444k | 144 µs |
| 1,024 | 34.8k | 1.70M | 604 µs |
| 65,536 | (~34.6k) | 1.18M | 56 ms |

Real data, batches of 4,096, every copy included (`waferedge-maps`, three runs): WaferLens
845k–885k maps/s against 43.1k–43.7k on one CPU core (~20×); WM-811K 654k–657k against
33.3k–34.1k (~20×).

**Why the Hough wins and the features don't**: arithmetic intensity. The CPU spends ~29 µs a
map on the Hough transform, so the ~170 µs a GPU call costs is paid back after ~6 maps; it
spends ~0.5 µs on the features, so it takes ~1,000 maps and the gain stays small.

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

**What bounds it** (Nsight Compute, *CUDA 12.4*, 4,096 maps, 1.66 ms at base clock):
divergence. Only 12.7 of 32 threads per warp are active on average: ~10% of dies fail and
every phase skips the rest, and `unite` loops a different number of times per lane (14% branch-resolving stalls,
26% barrier between the four phases). Occupancy 50% (shared memory). Gathering the fail dies
into a dense list first, as the Hough kernel does, is the obvious next step; at ~20% of an
all-signatures batch it isn't the bottleneck yet.

### All signatures, one upload

`bench-gpu`, 40×40 maps, median of 3, end to end:

| Batch | CPU clusters, 1 thread | GPU clusters | GPU features + Hough + clusters | its latency |
|---|---|---|---|---|
| 1 | 408k maps/s | 5.6k | 8.1k | 123 µs |
| 4 | 382k | 23.3k | 28.9k | 138 µs |
| 64 | 231k | 441k | 380k | 168 µs |
| 1,024 | 228k | 1.81M | 938k | 1.09 ms |
| 65,536 | (~230k) | 1.29M | 739k | 89 ms |

One CPU core needs ~32 µs per map for all three (AVX2 features ~0.5 µs, Hough ~29 µs, clusters
~2.5–4.4 µs): ~30k maps/s. The GPU draws level with it at a batch of ~4 and is ~32× faster at
1,024 maps. Real data, batches of 4,096, all copies included: WaferLens 578k–589k maps/s,
WM-811K 378k–385k (`waferedge-maps`, three runs, which also checks every map against the CPU).

## The whole CPU, and the cost of waking threads

`ThreadPool` (`include/waferedge/thread_pool.hpp`): threads start once; a job hands out
chunks from an atomic counter (fast threads take more chunks: maps don't all cost the same);
per-thread Hough and cluster scratch, so no locks in the loop. Tested for every item exactly
once at sizes 0 … 100,003 and 1 … 12 threads, stable worker ids, 3,000 jobs back to back; the
pool tests run clean under ThreadSanitizer. (The first version broke its own contract with a
one-thread pool, calling the function with all n items instead of chunks of at most `grain`;
the exactly-once test caught it.)

`bench-gpu --benchmark_filter=BM_cpu_` (40×40 maps, median of 3, large batches):

| | 1 thread | 6 threads | 12 threads |
|---|---|---|---|
| Features (AVX2), 65,536 maps | 2.04M maps/s | 7.8M (3.8×) | 7.2M (3.5×; 8.8M spinning) |
| All three signatures, 16,384 maps | 29.1k | 123k (4.2×) | 150k (5.2×) |

The second hardware thread per core adds 22% for all three (Hough's dependent arithmetic
leaves gaps the sibling thread fills) and nothing measurable for the features (7.2–8.8M
across runs). On CUDA 12.4's run, at lower clocks, it was 28% and 50%.

**Small batches: waking threads is the cost.** With workers asleep on a condition variable,
a batch of 4 maps took 101 µs on 6 threads and 176 µs on 12, against 2.0 µs on one thread:
waking a thread through the kernel (a futex, the scheduler, the core leaving an idle state)
costs ~100–180 µs under WSL2, the same order as a GPU call. **Spinning before sleeping**
(polling an atomic with the `pause` hint for up to 200 µs; release / acquire ordering hands
the job over, the mutex still guards the sleepers so no wake-up is lost):

| Features, 6 threads | Sleeping workers | Spinning workers |
|---|---|---|
| batch 4 | 101 µs | **2.1 µs** |
| batch 64 | 100 µs | **9.0 µs** |
| batch 1,024 | 186 µs | 132 µs |

**Spinning on all 12 hardware threads gains less and varies more**: a batch of 4 took 5.6 µs
(CV 18%) against 2.1 µs on 6 threads; in the first measurement (CUDA 12.4 run) it backfired
completely, 238 µs. With every logical CPU spinning, anything else on the machine preempts a
spinner and the caller waits for it. Low-latency systems spin on fewer threads than CPUs,
often pinned; ADR-0007 takes 6 of 12.

### GPU vs the whole CPU, all three signatures

| Batch | Best CPU (6 or 12 threads, spinning) | GPU | |
|---|---|---|---|
| 1 | 31.7k maps/s (32 µs) | 8.1k (123 µs) | CPU 3.9× |
| 4 | 71.0k | 28.9k | CPU 2.5× |
| 16 | 109k | 115k | GPU: crossover |
| 64 | 125k | 380k | GPU 3.0× |
| 1,024 | 150k | 938k | GPU 6.3× |
| 16,384 | 153k | 903k | GPU 5.9× |

For features alone the whole CPU (8.8M maps/s) beats the GPU (2.7M at best) at every batch
size.

## CUDA Graphs: one call per batch

Each batch made five driver calls (an upload, up to three kernel launches, a download) plus
the sync, and on WSL2 each call costs ~30 µs of CPU time before the GPU even starts. A CUDA
Graph records the sequence once and replays it with one `cudaGraphLaunch`.

`SignatureEngine::set_graphs(true)`: the first batch of each shape runs under **stream
capture** (between `cudaStreamBeginCapture` and `cudaStreamEndCapture` the copies and
launches are recorded, not run); the graph is instantiated (validated and prepared once) and
cached. A graph freezes sizes, launch dimensions and addresses, so the cache key holds the
batch count, byte sizes, requested signatures, the feature kernel variant and every buffer
address: a buffer that grows gets a new address and its old graphs stop matching. The cache
is capped at 32 graphs. Timing mode runs without graphs (events go between the steps). Test:
replays equal the CPU for all three signatures across new shapes, repeats, buffer growth and
timing mode.

`bench-gpu --benchmark_filter='BM_gpu_(all_signatures|all_signatures_graph)/'`, 40×40 maps,
median of 3; the CUDA 12.4 columns are the first build's run, at lower clocks:

| All three signatures, batch | CUDA 12.4: individual calls | CUDA 12.4: graph | CUDA 13.3: individual calls | CUDA 13.3: graph |
|---|---|---|---|---|
| 1 | 195 µs | **110 µs** (−44%) | 123 µs | 121 µs |
| 4 | 238 µs | **123 µs** | 138 µs | 139 µs |
| 16 | 265 µs | **117 µs** (2.3×) | 139 µs | 143 µs |
| 64 | 367 µs | **162 µs** | 168 µs | 164 µs |
| 1,024 | 1.27 ms | 1.13 ms | 1.09 ms | 1.02 ms |
| 65,536 | 78 ms | 78 ms | 89 ms | 87 ms |

**On CUDA 13.3 the graph gains nothing for all three signatures.** Individual calls got
cheaper (195 → 123 µs at one map) and a graph launch dearer: run back to back on the same day,
the CUDA 12.4 build replays a batch of one in 99 µs and the 13.3 build in 119–122 µs
(ADR-0020). Clusters alone still gain at batch 1: 178 → 112 µs (CV 14%; 163 → 99 µs on CUDA
12.4). Large batches don't change: copying and computing dominate there. Nsight Systems on
CUDA 12.4, CPU time in CUDA API calls per batch of 16 (average over ~3,000 batches): 3 launches
× 29 µs + 2 copies × 28.5 µs ≈ 144 µs individually, against one `cudaGraphLaunch` of 63 µs.

**With CUDA 12.4 graphs moved the crossover with the whole CPU from ~16 maps to ~4**; with 13.3
it is ~16 maps either way, and a single wafer is faster on the CPU (32 µs vs 121 µs).

## Overlapping the CPU's packing with the GPU's work

At large batches the CPU packing the maps into pinned memory is the biggest step (63% of a
features batch of 65,536, table above), and while the CPU packs, the GPU waits; while the GPU
works, the CPU waits in `cudaStreamSynchronize`. With two sets of buffers the two can run
at the same time: the CPU packs batch n+1 while the GPU copies and computes batch n.

`SignatureEngine::submit` / `collect`: the engine has two **slots**, each with its own pinned
upload and download buffers, device buffers and CUDA stream. `submit()` packs a batch into
the free slot and enqueues its upload, kernels and download on that slot's stream, without
waiting. `collect()` waits for the oldest batch's stream and copies its results out.

```cpp
engine.submit(batch[0]...);
for (n = 1; n < N; ++n) {
    engine.submit(batch[n]...); // packs n on the CPU while batch n - 1 runs on the GPU
    engine.collect();           // waits for batch n - 1
}
engine.collect();
```

`run()` is `submit` then `collect`. What needed care:
- **Each slot has its own streams and buffers**, so a batch never overwrites memory the other
  is still copying. A slot's buffers grow only while it is free.
- **Shared state** is the geometry tables (read only on the device) and the graph cache.
  Growing the geometry buffer frees the old one, so it first waits for the batch in flight;
  so does emptying a full graph cache (a graph may still be running on the other slot).
  Each slot records its own CUDA Graphs (the buffer addresses are part of the key).
- **Uploading a new geometry** is a synchronous copy, so it may wait behind the batch in
  flight: rare (once per wafer shape), and correct either way.
- **Two batches at most**: a third `submit()` without a `collect()` is an error, not a wait.

The best it can do: per batch, max(pack, GPU) instead of pack + GPU. For the features at
65,536 maps that is ~26 ms instead of ~39 ms (pack 26 ms, upload + kernel + download 13 ms),
about 1.5× the throughput; for all three signatures, where the kernels take longer, the
packing hides behind the GPU instead. Small batches gain little: their cost is the driver
calls, which CUDA Graphs already cut.

Tested (GPU-labelled): seven batches of mixed sizes and shapes, two in flight, a new shape
while a batch runs, with and without graphs; every result equals the CPU; `run()` after
`submit()` collects the earlier batch first; misuse (collect with nothing in flight, a third
submit) is an error.

### Measured

Laptop, `cuda` preset (waferedge 1.0.0, GCC 13.4, CUDA 13.3, driver 621.14), 40×40 maps, all
three signatures, medians of 3 repetitions; time per batch in the steady state, so maps/s is
the batch size over it. `graph` runs one batch at a time with a CUDA Graph; `overlapped`
keeps one batch in flight while the next is packed:

| Batch | One at a time, graph | Overlapped | Overlapped, graph | Gain (graph → overlapped graph) |
|---|---|---|---|---|
| 1 | 121 µs (8.3k maps/s) | 69 µs | **66 µs** (15.1k) | 1.8× |
| 16 | 143 µs (112k) | 72 µs | **71 µs** (224k) | 2.0× |
| 64 | 164 µs (389k) | 96 µs (CV 22%) | **85 µs** (752k) | 1.9× |
| 256 | 297 µs (862k) | 210 µs | **212 µs** (1.21M) | 1.4× |
| 1,024 | 1.02 ms (1.00M) | 859 µs | **860 µs** (1.19M) | 1.2× |
| 4,096 | 4.41 ms (930k) | 3.19 ms | **3.22 ms** (1.27M) | 1.4× |
| 16,384 | 18.4 ms (893k) | 12.2 ms | **12.3 ms** (1.33M) | 1.5× |
| 65,536 | 86.6 ms (757k) | 49.1 ms | **48.7 ms** (1.35M) | 1.8× |

- **Large batches gain what the model predicted:** 1.5× at 16,384 maps, from ~890k to
  ~1.33M maps/s; the packing now hides behind the GPU's copies and kernels. At 65,536 the
  gain is 1.8×.
- **Small batches gain too, for a different reason:** with one batch in flight, the next
  batch's driver calls go in while the previous one is still running, so the driver's CPU
  work per call on WSL2 overlaps the GPU's work instead of adding to it. A batch of 1 to 64
  maps costs 66–85 µs instead of 121–164 µs.
- **That is throughput, not latency.** Each batch still waits for its own upload, kernels
  and download, and in the overlapped loop it is collected one iteration later. For a wafer
  that has to be decided now, one batch at a time stays the lower-latency choice; overlap
  pays when maps arrive faster than one batch's round trip (re-scoring history, many tools).
- **Graphs add little to overlap on CUDA 13.3** (69–96 µs without, 66–85 µs with, for 1 to
  64 maps). On CUDA 12.4, overlap without graphs made small batches *slower* (168–201 µs
  against 121–165 µs one at a time with a graph): five driver calls per batch on two
  interleaved streams cost more than they hid. Cheaper individual calls on the new driver
  removed that penalty.
- In the CUDA 12.4 run the one-at-a-time 65,536 row was unstable (268 and 487 ms right after a
  full build) and wasn't compared; here it was steady (CV 5%).

## Batching: latency against throughput

A batch costs a fixed overhead (driver calls; ~120 µs on WSL2, with or without a CUDA Graph
on CUDA 13.3) plus a part that grows with the number of maps. Bigger batches spread the
overhead over more maps, so throughput rises, but every map in the batch waits for the whole
batch, so latency rises too. From the tables above (all three signatures, 40×40, CUDA Graphs,
one batch at a time):

| Batch | Throughput | Latency of the batch | |
|---|---|---|---|
| 1 | ~8k maps/s | 121 µs | overhead only |
| 16 | ~112k maps/s | 143 µs | 16× the maps for 18% more time |
| 64 | ~389k maps/s | 164 µs | |
| 1,024 | ~1.0M maps/s | 1.02 ms | |
| 65,536 | ~0.76M maps/s | 87 ms | saturated: latency only grows |

So for a stream of wafers the batch should be as large as the arrival rate fills within the
latency budget, and no larger:
- **Below ~1,000 maps the GPU is overhead-bound**: a batch of 16 costs about what a batch of
  1 costs. If wafers arrive faster than ~1 per 120 µs, batching them is free.
- **Above ~1,000 maps it is throughput-bound**: doubling the batch doubles its latency and
  adds nothing.
- **A batch must close on a deadline as well as a size**, or a slow tool's wafers wait for
  maps that don't come. ADR-0007 closes signature batches at 4,096 maps or 2 ms; the edge
  host's detectors close at B maps or D µs from the oldest wafer (ADR-0016).

In the edge pipeline (docs/pipeline.md, laptop, the *CUDA 12.4* run) that trade-off measured end to end: at 100
to 10,000 wafers/s, the GPU rules with batch 64 / 2 ms took 2.2–3.5 ms median sort →
decision, almost all of it the deadline (a batch of 64 never filled at those rates), against
0.3–0.5 ms for the rules on one CPU core. A tool sorts a wafer every few seconds to minutes,
so at the edge the CPU wins; the GPU pays where maps arrive by the thousand per second
(re-scoring history, many tools on one host), or for the CNN.

## Reproduce the overlap

```sh
ctest --preset cuda -L gpu -R overlapped
build/cuda/bench/bench-gpu \
  --benchmark_filter='BM_gpu_all_signatures_(graph|overlapped|overlapped_graph)/' \
  --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
```

Still open: the Hough kernel's shared-memory bank conflicts (Kernel 3 above).
