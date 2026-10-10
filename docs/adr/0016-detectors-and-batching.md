# ADR-0016: Workers run a pluggable detector on batches closed by size or deadline

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
Phase 4's experiments compare what the edge host can run on each wafer: the rule classifier
on the CPU (AVX2 features), the same rules from the GPU signature engine, and FabEye's CNN,
on the CPU reference forward pass or on WaferEdge's GPU engine in fp32, fp16 or int8. The
GPU pays a fixed cost per call: on WSL2 every CUDA call is ~0.1 ms, a whole CNN batch only a
little more. So the GPU wants batches, while the hold wants each wafer decided at once. Until
now a worker called the rule classifier on one wafer at a time.

## Options considered
1. **A separate worker type per detector** (a CPU worker, a GPU worker): duplicates the ring,
   doorbell and verdict logic for each.
2. **One worker, one batch interface for every detector**: `Detector::detect(maps, out)`.
   CPU detectors loop over the batch; GPU detectors make one call. The worker decides when a
   batch starts.
3. **A central batcher thread feeding a GPU thread**: one more hand-off (and wake-up) on the
   path of every wafer, and a thread that only waits.

When a batch starts:
- **Fixed size:** good throughput, unbounded latency at low load (a batch of 64 at 100
  wafers/s holds the first wafer for 0.64 s).
- **Deadline only:** bounded latency, and batches as large as whatever arrived in time.
- **Size or deadline, whichever comes first:** the bound of the deadline and the efficiency
  of the size. ADR-0007 used the same rule for throughput runs (4,096 maps or 2 ms).

## Decision
Option 2, with batches closed by size *or* deadline:
- `pipeline::Detector` (`detector.hpp`) has `name()` and `detect(maps, out)`. Detectors:
  - `RulesDetector` and `CnnCpuDetector`, per map.
  - The GPU detectors, only in the cuda build: `rules-gpu` (signature engine + the same
    rules; the join-count z stays on the CPU) and `cnn-fp32|fp16|int8` (the CNN engine).
  - `PairDetector`: the first detector decides, and the second's pattern is recorded in the
    verdict for agreement.
  - `detector_factory("cnn-fp16+rules", resources)` builds any of them by name, and names
    what is missing: a model, the int8 scales, or the cuda build.
- `SpscRing::peek(i)` / `pop(n)`: the worker reads the waiting wafers in place and frees a
  whole batch at once.
- `AnalyzerConfig::batch` (1..256, the ring's size) and `deadline`. A batch starts when
  `batch` wafers wait, or when the oldest has waited `deadline` since the network thread
  received it. While a partial batch waits, the worker spins: deadlines are milliseconds,
  and a futex wait has no timeout. `batch = 1` is the previous behaviour, so it is the
  default.
- `SignalExtractor::compose()` turns any detector outputs (features, z, largest cluster,
  Hough line) into the signals. The CPU and GPU rules therefore share one formula, and a GPU
  test checks them equal map for map.
- Each worker builds its own detector from the factory: a GPU engine (stream, buffers,
  graphs) per worker, never shared between threads.

## Consequences
- Every detector is measured by the same benchmark, `bench-latency --detector … --batch …
  --deadline-us …`, and scored in the same run against truth: per-wafer accuracy, and the
  precision and recall of the held lots. A faster detector that holds the wrong lots shows as
  such.
- The batch's views and results are fixed arrays in the worker, so the hot path still makes
  no allocation (`waferedge-alloc-tests "[edge]"` runs through the new path). The GPU
  engines grow their buffers to the largest batch once.
- Batching never helps the CPU rules: it only adds the wait for the batch to fill (container:
  median 0.1–0.3 ms at batch 1, 1.3–1.7 ms at batch 16–64 with a 2 ms deadline). It exists for
  the GPU, where one call per wafer would cost more than the work.
- A failed detector call (a CUDA error) gives `unknown` verdicts for its batch and is
  counted. The rule never holds a lot on `unknown`.
- The largest batch is the ring's 256 slots. For latency that is plenty: at 0.1 ms per call,
  a batch of 64 already makes the call overhead small. The throughput runs of ADR-0007 used
  larger batches outside the pipeline.
