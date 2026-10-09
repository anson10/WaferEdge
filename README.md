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

Phase 1 of [the roadmap](ROADMAP.md): wafer maps load from both datasets
([docs/data.md](docs/data.md)); the scalar detectors (zones, sectors, rings, clusters, a Hough
scratch search, a join-count randomness test) feed an explainable rule classifier with
thresholds fitted on WM-811K train ([docs/signatures.md](docs/signatures.md), ADR-0004,
ADR-0005). The AVX2 backend comes next.

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

**Scalar features** (fail counts by radial zone and angular sector, the reference every other
backend must match bit for bit), `release` preset:

| Input | Maps/s | Command |
|---|---|---|
| WaferLens demo, 24,090 maps (24² to 40²) | 414k | `build/release/tools/waferedge-maps data/waferlens_demo.wmap` |
| WM-811K, 79,608 maps (341 shapes) | 267k | `build/release/tools/waferedge-maps data/wm811k_lot.wmap` |
| Synthetic 24² / 30² / 40² / 64², 10% fails | 756k / 444k / 244k / 67k | `build/release/bench/bench-features` |

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
