# ADR-0007: Where each signature runs, and how the GPU is batched

- **Status:** Accepted
- **Date:** 2026-10-09

## Context
Phase 2a put every classical signature on the GPU (features, Hough line, clusters) and
measured them against the CPU on one core and on all cores (docs/gpu.md). The pipeline
(phase 4) must decide, per wafer or per batch, where each signature runs and how long it may
wait for other wafers to form a batch. Constraints:

- **The latency budget is generous.** A lot must be held before the next wafer is processed;
  a sort result to a hold decision has seconds to minutes, not microseconds. Throughput
  matters for replay and catch-up (WaferLens's stress profile: 125,000 wafers).
- **Every GPU call has a fixed cost** on this machine (WSL2): ~200 µs per batch end to end
  (one upload, kernels, one download, one sync), measured with Nsight Systems and the
  benchmarks.
- **Waking sleeping CPU threads costs too**: ~100–250 µs per job on WSL2; spinning removes it
  but burns cores, and spinning on every hardware thread backfires (preemption).

Measurements (40×40 maps, end to end per batch, `bench-gpu`, docs/gpu.md):

| | Whole CPU (best of 6 / 12 threads) | GPU | Crossover |
|---|---|---|---|
| Features alone | up to 5.4M maps/s | up to 1.6M | none: CPU wins at every batch size |
| Hough alone | (~14.5k per core) | up to 1.22M | ~4 maps vs one core |
| All three | 14k (batch 1) … 87k | 5k (batch 1) … 0.82M | **~16 maps**; ~9.4× at large batches |

## Options considered
1. **Everything on the GPU, fixed batch size** (e.g. 4,096): simplest; a wafer waits for 4,095
   others, which at a tool's rate (a wafer every minute or so) is hours.
2. **Everything on the CPU**: no GPU dependency; ~87k maps/s on all cores is plenty for one
   tool but leaves the GPU idle and loses 9× on replay.
3. **Placement by measured cost, deadline-based batches**: features on the CPU always; Hough
   and clusters on the CPU for small batches and on the GPU for large ones; a batch closes when
   it is full **or** its oldest wafer has waited a deadline, whichever comes first.
4. **Adaptive**: learn the crossover at run time from timings. More moving parts than the
   problem needs; the crossover is a property of the machine, measurable once.

## Decision
Option 3.

- **Features run on the CPU** (AVX2, `best_features()`), always: the GPU never wins for them.
- **Hough and clusters**: below **16 maps** in a batch on the CPU (pool threads), from 16 up
  on the GPU (`SignatureEngine`, one upload for both). 16 is this machine's measured
  crossover; it is a configuration value with the measurement documented, not a constant in
  the code.
- **Batches close on size or deadline**: at most 4,096 maps (the GPU's plateau starts at
  ~1,024; larger batches only add latency) or when the oldest wafer has waited **2 ms**,
  whichever comes first. At a tool's real rate every batch closes on the deadline with one
  wafer and runs on the CPU in ~70 µs; during replay batches fill and go to the GPU.
- **CPU pool threads spin briefly** (~200 µs) before sleeping, on **fewer threads than
  hardware threads** (e.g. 6 of 12), so a burst of wafers doesn't pay the wake-up cost and
  spinning threads aren't preempted by the rest of the system.

## Consequences
- A single wafer's signatures cost ~70 µs on the CPU instead of ~200 µs on the GPU, plus at
  most the 2 ms deadline; both are negligible against the minutes the hold decision has.
- Replay throughput is the GPU's: ~0.8M maps/s for all signatures, ~9× the whole CPU.
- Two code paths per signature (CPU and GPU) are already tested for equality on every real
  map, so switching between them at the crossover can't change a result.
- The numbers are this laptop's (WSL2 inflates both GPU-call and thread-wake costs). On a
  native Linux edge box the crossover will move down; re-measure with `bench-gpu` and set the
  configuration, don't change the design.
- Phase 4's latency measurements will show whether the 2 ms deadline is right; if the hold
  path ever needs less, CUDA Graphs (one call per batch) and pinned receive buffers (no
  packing) are the levers measured as the largest costs.
