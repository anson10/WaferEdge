# WaferEdge

**How fast can a spatial excursion be caught, and the lot put on hold, at the equipment,
before the next wafer is processed?**

[WaferLens](https://github.com/anson10/waferLens) finds excursions in batch: its pattern alarm
caught all 13 spatial excursions in its simulated fab, a median of 52 hours after they began,
most of it waiting for wafer sort. WaferEdge moves that check to the tool. Tool emulators send
wafer sort results over SEMI HSMS / SECS-II; an edge host decodes them without copying, finds
spatial signatures on the wafer maps (CPU with AVX2, or batched on the GPU with CUDA), runs
[FabEye](https://github.com/anson10/FabEye)'s CNN on its own CUDA inference engine, and sends
a lot hold (S2F41) back to the tool.

## The answer, in short

- **The hold reaches the tool within milliseconds of the sort result.** With the rules on one
  CPU core, sort result → decision is 0.13–0.16 ms at the median and 0.26–0.31 ms at p99, at
  1,000–5,000 wafers/s (laptop). A tool sorts a wafer every few seconds to minutes, so the
  hold always lands before the lot's next wafer.
- **What decides how early an excursion is caught is the hold rule and the detector, not the
  machinery.** Replayed over WaferLens's 24,090 wafers, every one of its 13 spatial
  excursions is held. After an excursion's first patterned wafer reaches sort, the default
  rule (3 of a lot's last 5 wafers) holds after a median 8.1 h, slower than WaferLens's batch
  alarm (3.0 h). Adding a rule that counts across lots (2 of 5, or 3 in 24 h) holds after
  2.0 h with 31 patterned wafers sorted first, against 72 for WaferLens's alarm, for 5 false
  holds out of 211. With a perfect detector the cross-lot rule alone matches WaferLens's
  alarm; the rest of the gap is the rules detector missing patterned wafers.
- **The CNN is more accurate than the rules** (macro-F1 0.858 vs 0.655 on WM-811K's test
  lots) and runs at ~19,000–26,000 maps/s on WaferEdge's own GPU kernels. **At a tool's
  wafer rate the CPU rules are the faster choice**: the GPU's batch deadline and driver calls
  cost more than classifying one map.

Every number below says which machine and which command produced it.

```
 Tool emulator ──HSMS/SECS-II (TCP)──► Edge host
 (replays WaferLens / WM-811K maps)    ├─ net thread: Asio coroutines, zero-copy SECS-II decode
        ▲                              ├─ lock-free SPSC rings, no allocation on the hot path
        │                              ├─ analytics: wafer-map signatures, CPU (AVX2) or GPU (CUDA)
        │                              ├─ inference: FabEye's CNN on our own CUDA kernels
        └──────── S2F41 HOLD ◄─────────└─ decision rule (per lot, and across lots) → lot hold
```

## Results

*Laptop*: Ryzen 5 7535HS (6 cores / 12 threads, AVX2), RTX 3050 Laptop 6 GB (sm_86), WSL2
Ubuntu 26.04, CUDA 13.3, GCC 13, driver 621.14; re-measured 2026-11-10 (ADR-0020 compares
the old and new builds on the same day). *Container*: a 4-thread cloud VM (Xeon @ 2.1 GHz,
AVX2, no GPU). `build/<preset>/tools/waferedge-info` prints the machine and build to quote with a
number.

### Accuracy against FabEye's CNN

Macro-F1 on the same maps ([docs/evaluation.md](docs/evaluation.md)):

| Data | Rules (WaferEdge) | CNN (FabEye) | Trivial (density threshold) |
|---|---|---|---|
| WM-811K test, lot-disjoint, 25,875 maps | 0.655 | 0.858 | 0.118 |
| WaferLens, simulated, 24,090 maps | 0.645 | 0.909 | 0.132 |

The rules explain every call in one line and run at ~30,000 maps/s on one core; they hold
up on none, edge-ring, center and near-full and lose on edge-loc, loc and scratch. FabEye's
CNN on WaferEdge's engine keeps FabEye's macro-F1 in fp32 and fp16 and its conformal coverage
in every precision (90% target: 0.8925 / 0.8926 / 0.8951 for fp32 / fp16 / int8;
[docs/inference.md](docs/inference.md)).

`python3 tools/evaluate.py ...` (the sequence is in docs/evaluation.md),
`build/cuda/tools/waferedge-cnn conformal ...`.

### CPU against GPU, and the crossover

All three signatures (features, Hough line, clusters) from one upload, 40×40 maps, laptop,
end to end per batch ([docs/gpu.md](docs/gpu.md)):

| Batch | Whole CPU (6 or 12 threads, spinning pool) | GPU | GPU with CUDA Graphs |
|---|---|---|---|
| 1 | 31.8k maps/s (31 µs) | 8.1k (123 µs) | 8.2k (122 µs) |
| 16 | 110k | 110k | 113k (142 µs) |
| 1,024 | 152k | 986k | 916k (1.12 ms) |
| 16,384 | 151k | 875k | 886k (18.5 ms) |

The GPU draws level with the whole CPU at ~16 maps a batch and is ~6× it at large batches.
CUDA Graphs no longer move the crossover earlier: with CUDA 13.3 each graph launch costs
~20–25 µs more than with 12.4 (ADR-0020). Packing the next batch while the GPU works on this
one (two buffer sets and streams) adds 1.3–2.1× throughput: ~1.30M maps/s at 16,384 maps,
and 62–87 µs per batch of 1–64 maps with CUDA Graphs
(`BM_gpu_all_signatures_overlapped_graph`). For the feature counts alone the whole CPU
(8.5M maps/s) beats the GPU (2.3M) at every size: too little work per byte to pay for PCIe. Below ~1,000 maps a batch is all
overhead, above it all throughput, so batches close on a size or a deadline (ADR-0007).

`build/cuda/bench/bench-gpu --benchmark_filter='BM_(cpu_all_signatures_threads|gpu_all_signatures|gpu_all_signatures_graph|gpu_all_signatures_overlapped_graph)/' --benchmark_repetitions=3`
(medians).

FabEye's CNN, end to end per batch, laptop (GPU SM clock median 1.46 GHz for WaferEdge's runs
and 1.67 GHz for the yardsticks'; wall clock on this laptop, read as ±10–20%):

| Batch | ONNX Runtime CPU | PyTorch fp16 (GPU) | WaferEdge fp16 | WaferEdge int8 |
|---|---|---|---|---|
| 1 | 895 maps/s | 605 | 2,076 | 3,080 (0.32 ms) |
| 256 | 1,092 | 9,756 | 19,448 | 25,516 |

`build/cuda/bench/bench-cnn --benchmark_filter='fused:1/graph:0' --benchmark_repetitions=3`,
`python3 tools/bench_cnn_yardsticks.py`.

### Decoding the equipment protocol

SECS-II wafer reports decode in place to views over the receive buffer
([docs/secs.md](docs/secs.md)), container, medians of 3:

| Message | Bytes | Decode and read every field | Encode | Allocations |
|---|---|---|---|---|
| Wafer report, 40×40 map | 1,653 | 118 ns | 120 ns | 0 |
| Wafer report, 200×200 map | 40,053 | 120 ns | 1.46 µs | 0 |
| S2F41 HOLD | 29 | 40 ns | 59 ns | 0 |

About 8M wafer reports/s on one core, whatever the map size. An HSMS transaction over TCP
loopback takes ~7 µs (p99 20 µs), of which the protocol is ~0.2 µs; 0 allocations per
transaction. The decoder, the HSMS framing and the GEM layer are fuzzed with libFuzzer in CI.

`build/release/bench/bench-secs`, `build/release/bench/bench-hsms`.

### Tail latency, sort result to hold

The tool emulator and the edge host in one process, HSMS over TCP, latency from each wafer's
*scheduled* time in an open-loop schedule, so a stall delays the wafers behind it and
coordinated omission is counted ([docs/pipeline.md](docs/pipeline.md), ADR-0015). Laptop, one
worker, CPU rules, median of 3 runs per row:

| Maps | Wafers/s | Dropped | Decision p50 | p99 | p99.9 | HOLD at the tool, p50 |
|---|---|---|---|---|---|---|
| synthetic 40×40 | 1,000 | 0 | 162 µs | 309 µs | 428 µs | 229 µs |
| synthetic 40×40 | 5,000 | 0 | 130 µs | 258 µs | 477 µs | 213 µs |
| WaferLens demo | 1,000 | 0 | 176 µs | 350 µs | 477 µs | 358 µs |
| WaferLens demo | 10,000 | 0 | 130 µs | 311 µs | 709 µs | 218 µs |

Overload turns into counted drops, not an unbounded queue: one worker keeps up with 30,000
synthetic maps/s on the laptop and starts dropping at 40,000. WSL2 can pause the whole
process: in the earlier measurement two runs hit a 1.1–1.5 s pause that six reruns didn't
reproduce, and here one of the four WaferLens runs at 10,000/s had a 2.9 ms p99.9; the
open-loop schedule makes such stalls visible, and docs/pipeline.md reports them rather than
hiding them. With the GPU, the CNN in fp16 (batch 8, 0.5 ms deadline) takes 1.1 ms median
at 1,000 wafers/s.

`build/release/bench/bench-latency`, `... --maps data/waferlens_demo.wmap`,
`build/cuda/bench/bench-latency --maps data/waferlens_demo.wmap --rates 1000 --detector
cnn-fp16 --batch 8 --deadline-us 500`.

### The closed loop on WaferLens's excursions

WaferLens's demo set replayed in sort-time order through the edge host's own detector and
decision rule; a hold lands 5 ms after its wafer's sort (ADR-0017, ADR-0018). Hours from an
excursion's first patterned wafer at sort to the hold (medians over the 13 excursions):

| Monitor | Excursions held | In-line h | Patterned wafers sorted first (of 1,891) | Holds | False holds |
|---|---|---|---|---|---|
| WaferLens batch alarm (≥ 3 in 24 h, all lots) | 13 | 3.0 | 72 | – | – |
| Rules, 3 of 5 per lot (default) | 13 | 8.1 | 147 | 122 | 2 |
| Rules, 2 of 5 per lot | 13 | 2.5 | 37 | 194 | 2 |
| Rules, 2 of 5, or 3 in 24 h across lots | 13 | **2.0** | **31** | 211 | 5 |
| Perfect detector, 3 of 5, or 3 in 24 h | 13 | 3.2 | 39 | 206 | 0 |
| No hold | 0 | – | 1,891 | 0 | 0 |

The default rule's 8.1 h is a negative result, kept: a per-lot window fills slowly because
an excursion spreads its wafers over many lots. The defaults stay 3 of 5 with the cross-lot
rule off, because these numbers come from one simulated fab. The replay is deterministic;
the container and the laptop give the same tables.

`build/release/tools/waferedge-closedloop data/waferlens_demo.wmap --excursions
data/waferlens_excursions.csv --sweep` (and `--detector truth`); the data from
`tools/export_maps.py` (docs/data.md).

## Python

The CPU signatures, the rule classifier and the clusters on NumPy arrays, read in place
through the buffer protocol ([docs/python.md](docs/python.md)):

```python
import waferedge as we
rules = we.Classifier.load("config/rules.txt")
rules.classify(wafer_map)            # a 2-D uint8 array -> 'edge_loc'
we.features_batch(stack)             # a (N, rows, cols) stack -> count arrays
```

`CXX=g++-13 pip install .`, then `pytest python/tests`. From Python: ~0.94M maps/s for the
feature counts and ~17k maps/s for classification on one container core
(`python3 tools/bench_python.py`).

## Build

Needs CMake ≥ 3.25, Ninja and GCC 13 (Ubuntu 24.04 and 26.04 package it; on 22.04 use the
`ubuntu-toolchain-r/test` PPA). Clang 18 with libc++ 18 for the `clang` and `fuzz` presets;
CUDA 13.x for the `cuda` preset, with GCC 13 as nvcc's host compiler (ADR-0020).

```sh
cmake --workflow --preset dev        # configure, build, test (Debug, GCC 13)
cmake --workflow --preset asan       # the same under AddressSanitizer + UBSan
cmake --workflow --preset tsan       # ThreadSanitizer (the rings, the edge host)
cmake --workflow --preset cuda       # Release with the CUDA backend, GPU tests included
build/dev/tools/waferedge-info       # the build and machine every number is measured on
```

Run the loop by hand: `build/release/tools/tool-emulator data/waferlens_demo.wmap --port 5000
--rate 50` in one terminal, `build/release/tools/waferedge-edge --port 5000` in another
([docs/pipeline.md](docs/pipeline.md)).

## Limitations

- **Simulated fab.** The excursions, their timing and their ground truth come from
  WaferLens's simulator; WM-811K is real but has no process history. The closed-loop numbers
  are one simulated fab's, and the rule settings aren't tuned on them.
- **GPU tests run locally only.** GitHub's runners have no GPU: CI compiles the CUDA code and
  runs the CPU tests; the tests labelled `gpu` run with `ctest --preset cuda -L gpu`.
- **One laptop under WSL2.** Each CUDA call costs ~20–90 µs there, GPU clocks move with power
  and heat (kernels are compared in cycles), and the process can pause for a second now and
  then. Native Linux would have smaller overheads; it isn't measured here.
- **A subset of SEMI E5 / E30 / E37**, from public descriptions; not certified. The edge host
  is the GEM host side and the emulator the equipment side; S5F1 alarms go from equipment to
  host, so the host doesn't raise them.

## Decisions and docs

Architecture decision records: [docs/adr](docs/adr) (0001–0020). The roadmap and what is
done: [ROADMAP.md](ROADMAP.md). Per area: [data](docs/data.md),
[signatures](docs/signatures.md), [AVX2](docs/avx2.md), [GPU](docs/gpu.md),
[inference](docs/inference.md), [evaluation](docs/evaluation.md), [SECS/HSMS/GEM](docs/secs.md),
[pipeline](docs/pipeline.md), [Python](docs/python.md).

## Licence

MIT. The SEMI standards (E5, E30, E37) are not reproduced here; WaferEdge implements a subset
of them from public descriptions and is not certified.
