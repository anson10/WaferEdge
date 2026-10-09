# GPU backend (phase 2a)

What runs on the GPU, how it is checked, and what it costs, measured. The short version so
far: the GPU computes features for 65,536 maps in **3 ms** (~22 million maps/s), but getting
the maps there takes 10× longer, so end to end it reaches **~1.6M maps/s, about 1.7× one CPU
core with AVX2, from batches of ~300 maps up**. Feature counting is too little work per byte
to be worth shipping over PCIe; the Hough transform and the CNN are where the GPU should pay.

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

**v2 is about 2× slower** (kernel time for 65,536 maps: 6.0–6.3 ms against 3.0–3.7 ms). The
likely reason is that six `MATCH.ANY` per 32 dies cost more than Ampere's hardware handling
of shared-atomic conflicts; that is a hypothesis until Nsight Compute can read the counters
(below). v1 is the default; v2 stays as the measured, documented experiment.

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
(`FeatureEngine::set_timing`); without it a batch of one went from 297 µs to 208 µs and the
large-batch plateau from ~1.3M to ~1.6M maps/s. The remaining four calls per batch (copy,
launch, copy, sync) are the next target: a CUDA Graph replays them as one call.

On WSL2, Nsight Systems records the CPU side of CUDA calls but not the GPU's own kernel
timeline; Nsight Compute needs GPU performance-counter access enabled on the Windows side
(NVIDIA Control Panel → Developer → Manage GPU Performance Counters → allow all users, then
`wsl --shutdown`).

## Reproduce

```sh
cmake --workflow --preset cuda                       # builds, runs all tests incl. gpu
ctest --preset cuda -L gpu                           # GPU tests only
build/cuda/tools/waferedge-maps data/wm811k_lot.wmap # GPU == scalar on every map, timings
build/cuda/bench/bench-gpu --benchmark_filter='BM_(cpu_avx2|gpu_shared_atomics)/' \
  --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
build/cuda/bench/bench-gpu --benchmark_filter=_timed/ --benchmark_repetitions=3 \
  --benchmark_report_aggregates_only=true             # the breakdown
nsys profile -t cuda -o small build/cuda/bench/bench-gpu \
  --benchmark_filter='BM_gpu_shared_atomics/4/' && nsys stats -r cuda_api_sum small.nsys-rep
```

## Next in phase 2a

- Nsight Compute on v1 and v2 (why v2 is slower: atomic throughput, `MATCH` issue cost,
  occupancy), once counters are enabled.
- CPU AVX2 on all cores in the crossover benchmark.
- Kernel 3, the Hough transform (~85% of the classifier's CPU time, 180 votes per fail die:
  much more work per byte than the features); kernel 2, connected components.
- Streams overlapping the upload of one batch with the kernel of the previous; CUDA Graphs
  for the per-batch call overhead.
