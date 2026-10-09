# WaferEdge

**How fast can a spatial excursion be caught, and the lot put on hold, at the equipment,
before the next wafer is processed?**

[WaferLens](https://github.com/anson10/waferLens) finds excursions in batch: its pattern alarm
caught all 13 spatial excursions in its simulated fab, a median of 52 hours after they began,
most of it waiting for wafer sort. WaferEdge moves that check to the tool. Tool emulators send
wafer sort results over SEMI HSMS / SECS-II; an edge host decodes them without copying, finds
spatial signatures on the wafer maps (CPU with AVX2, or batched on the GPU with CUDA), runs
[FabEye](https://github.com/anson10/FabEye)'s CNN on its own CUDA inference engine, and sends
a lot hold (S2F41) back. Accuracy, throughput and tail latency from sort result to hold are
measured.

```
 Tool emulator ──HSMS/SECS-II (TCP)──► Edge host
 (replays WaferLens / WM-811K maps)    ├─ net thread: Asio coroutines, zero-copy SECS-II decode
        ▲                              ├─ lock-free SPSC rings, no allocation on the hot path
        │                              ├─ analytics: wafer-map signatures, CPU (AVX2) or GPU (CUDA)
        │                              ├─ inference: FabEye's CNN on our own CUDA kernels
        └── S5F1 alarm / S2F41 HOLD ◄──└─ decision rule → lot hold; latency histogram
```

## Status

Phase 1 of [the roadmap](ROADMAP.md) is complete: wafer maps load from both datasets
([docs/data.md](docs/data.md)); scalar detectors (zones, sectors, rings, clusters, a Hough
scratch search, a join-count randomness test, [docs/signatures.md](docs/signatures.md)) feed
an explainable rule classifier evaluated against FabEye's CNN
([docs/evaluation.md](docs/evaluation.md)); the feature counts also run on AVX2
([docs/avx2.md](docs/avx2.md)). Phase 2a (CUDA) is under way: the feature counts run on the
GPU in batches, with the Hough transform and the clusters: all three from one upload ~66× one
CPU core at large batches ([docs/gpu.md](docs/gpu.md)).

## Build

Needs CMake ≥ 3.25, Ninja and GCC 13 (Ubuntu 24.04 has it; on 22.04 use the
`ubuntu-toolchain-r/test` PPA). Clang 18 with libc++ 18 for the `clang` and `fuzz` presets;
CUDA 12.4 and GCC 11 (nvcc's host compiler, see ADR-0001) for the `cuda` preset.

```sh
cmake --workflow --preset dev        # configure, build, test (Debug, GCC 13)
cmake --workflow --preset asan       # the same under AddressSanitizer + UBSan
cmake --workflow --preset cuda       # Release with the CUDA backend, GPU tests included
build/dev/tools/waferedge-info       # the build and machine every number is measured on
```

GitHub's runners have no GPU: CI compiles the CUDA code and runs the CPU tests; the tests
labelled `gpu` run locally (`ctest --preset cuda`).

## Measurements

Every number here names the machine and the command that produced it.

**Features on the GPU** ([docs/gpu.md](docs/gpu.md)), 40×40 maps, end to end per batch
(pack, upload, kernel, download), `cuda` preset:

| Batch | CPU AVX2, 1 thread | GPU | GPU latency |
|---|---|---|---|
| 1 | 936k maps/s | 4.8k | 208 µs |
| 1,024 | 871k | 1.44M | 709 µs |
| 65,536 | ~0.9M | ~1.6M | 41 ms (kernel 3 ms, CPU packing 26 ms) |

**Hough transform on the GPU** (scratch search, 180 votes per fail die), same benchmark:

| Batch | CPU, 1 thread | GPU | GPU latency |
|---|---|---|---|
| 1 | 14.6k maps/s | 5.0k | 199 µs |
| 4 | 14.1k | 20.8k | 192 µs |
| 1,024 | 14.5k | 1.22M | 842 µs |

Real data, every copy included: WaferLens 651k vs 19.9k maps/s, WM-811K 401k vs 15.4k
(`build/cuda/tools/waferedge-maps <file>`, which also checks GPU == CPU on every map).

**All three signatures** (features, Hough, clusters) on one upload: ~0.82M maps/s against
~12.5k on one CPU core (`BM_gpu_all_signatures`); real data with every copy: WaferLens 482k,
WM-811K 247k maps/s.

`build/cuda/bench/bench-gpu`. Counting features is too little work per byte to win over PCIe;
the Hough transform is enough.

**Pattern classification** ([docs/evaluation.md](docs/evaluation.md)): macro-F1 on the same maps.

| Data | Rules (WaferEdge) | CNN (FabEye) | Trivial (density threshold) |
|---|---|---|---|
| WM-811K test, lot-disjoint, 25,875 maps | 0.655 | 0.858 | 0.118 |
| WaferLens, simulated, 24,090 maps | 0.645 | 0.909 | 0.132 |

The CNN is clearly more accurate. The rules explain every call in one line and run at ~13k
maps/s on one core; they hold up on none, edge-ring, center and near-full and lose on
edge-loc, loc and scratch.

*Machine A*: Ryzen 5 7535HS (6 cores, AVX2), RTX 3050 Laptop 6 GB (sm_86), WSL2 Ubuntu 22.04,
CUDA 12.4, GCC 13.

**Features** (die and fail counts by radial zone, angular sector and ring; every backend must
match the scalar reference bit for bit), `release` preset, one thread; real-data rows are
ranges over several runs, which swing ±25% on this laptop ([docs/avx2.md](docs/avx2.md)):

| Input | Scalar | AVX2 | Command |
|---|---|---|---|
| WaferLens demo, 24,090 maps (24² to 40²) | 303–314k maps/s | 1.03–1.13M | `build/release/tools/waferedge-maps data/waferlens_demo.wmap` |
| WM-811K, 79,608 maps (341 shapes) | 180–208k | 531–860k | `build/release/tools/waferedge-maps data/wm811k_lot.wmap` |
| Synthetic 24² / 64², 10% fails | 599k / 64k | 1.89M / 347k | `build/release/bench/bench-features --benchmark_filter=BM_features` |

Per stage on WaferLens: clusters 146k, join count 196k, Hough transform 19k maps/s
(docs/signatures.md).

`waferedge-maps` times whole passes over the file on a steady clock for at least a second;
the benchmark reports the median of 5 repetitions (`--benchmark_repetitions=5`). Run-to-run
spread on this laptop is up to ~20%, so treat differences under that as noise.

**Host→device copy** (`cudaMemcpy`), median of 5 repetitions, `cuda` preset:

| Size | Pageable | Pinned |
|---|---|---|
| 3.2 KB (one 40×40 map of int16 bins) | 99 µs | 94 µs |
| 1 MiB | 3.8 GiB/s | 5.2 GiB/s |
| 16 MiB | 3.8 GiB/s | 8.9 GiB/s |
| 64 MiB | 4.1 GiB/s | 8.8 GiB/s |

Pinned memory doubles large-copy bandwidth, so batches are staged in pinned buffers. A single
small copy costs about 0.1 ms whatever the memory. That is high for a 3.2 KB copy; the likely
cause is WSL2's GPU paravirtualisation (each CUDA call crosses into the Windows driver), not
measured on native Linux here. On this machine, sending maps to the GPU
one at a time pays that cost per map, which raises the batch size at which the GPU pays off.

```sh
cmake --workflow --preset cuda
build/cuda/bench/bench-transfer --benchmark_min_time=0.5s --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

## Decisions

Architecture decision records live in [docs/adr](docs/adr).

## Licence

MIT. The SEMI standards (E5, E30, E37) are not reproduced here; WaferEdge implements a subset
of them from public descriptions and is not certified.
