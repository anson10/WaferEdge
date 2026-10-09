# WaferEdge — Roadmap

Goal: an in-line wafer-map watchdog at the equipment edge. Tool emulators send wafer sort
results over HSMS / SECS-II; an edge host decodes them without copying, detects spatial
signatures (CPU AVX2 or CUDA), and sends a lot hold back, with accuracy, throughput and tail
latency measured. Next to the classical detectors, FabEye's CNN runs on **WaferEdge's own CUDA
inference engine** (implicit-GEMM convolution, kernel fusion, int8 on tensor cores). A showcase
for advanced C++ (low-latency / HPC techniques), CUDA and ML inference, in the semiconductor
domain. About 4–5 weeks for phases 0–5; phase 6 (a local LLM writing the incident note on the
same engine) is a stretch goal.

Every phase ends green in CI, with tests and at least one measured number. ADRs are numbered
when written (docs/adr/); the roadmap names future ones by topic.

```
 Tool emulator ──HSMS/SECS-II (TCP)──► Edge host
 (replays WaferLens / WM-811K maps)    ├─ net thread: Asio coroutines, zero-copy SECS-II decode
        ▲                              ├─ lock-free SPSC rings, no allocation on the hot path
        │                              ├─ analytics: wafer-map signatures, CPU (AVX2) or GPU (CUDA)
        │                              ├─ inference: FabEye's CNN on our own CUDA kernels (fp32 / fp16 / int8)
        └── S5F1 alarm / S2F41 HOLD ◄──└─ decision rule → lot hold; latency histogram
```

---

## Phase 0 — Foundation (1–2 days)

Decisions first (ask the user, then record in ADRs):
- [x] C++ standard: install GCC 13 for C++23 (`std::expected`, `std::print`) or stay on C++20
- [x] CUDA learning mode: the user writes the kernels with guidance, or Claude writes them
- [x] GitHub repo `anson10/WaferEdge` created (ask before creating; public, MIT)

Setup:
- [x] Layout: `include/waferedge/`, `src/`, `cuda/`, `tests/`, `bench/`, `fuzz/`, `tools/`,
      `python/`, `docs/adr/`, `docs/` (see docs/context.md for the module sketch)
- [x] CMake ≥ 3.25 with `CMakePresets.json`: `dev` (Debug, g++), `release`, `asan`, `tsan`,
      `cuda` (CUDA on), `fuzz` (clang + libFuzzer); compilers pinned to `g++` / `gcc` in presets
- [x] Warnings as errors (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion`), clang-format,
      clang-tidy config
- [x] FetchContent: Catch2 v3, Google Benchmark, Asio (standalone) (no fmt: C++23 has `std::format`)
- [x] CI (GitHub Actions): GCC + Clang build/test, ASan/UBSan job, CUDA compile-only job
      (nvidia/cuda container or the CUDA toolkit action), later fuzz smoke run
- [x] pre-commit: whitespace, clang-format check
- [x] ADR-0001: why C++ (and which standard) and CUDA; ADR-0002: repo layout and backends
      behind one interface; ADR-0003: dependency policy (FetchContent, pinned)
- [x] README skeleton: the question, the architecture sketch, status
- [x] CLAUDE.md "Commands" section filled in

## Phase 1 — Wafer-map core, CPU (3–4 days)

The domain model and classical spatial analysis, scalar first (the reference), then AVX2.
- [x] `WaferMap`: dense grid of bin codes (0 off wafer, 1 pass, ≥2 fail), view types
      (`std::span`/`mdspan`-like), no per-die allocation; loaders for WaferLens Parquet maps
      (via a small Python export script to a compact binary format, or Arrow C++) and WM-811K
      (`tools/export_maps.py` → `.wmap`, `MapSet::load`; ADR-0004)
- [x] Features per map: yield, fail density by **radial zone** (centre → edge) and **angular
      sector**, edge-ring ratio, centre ratio
      (exact integer zones/sectors per shape; scalar reference: ~414k maps/s on WaferLens)
- [x] Defect **clustering**: connected components on fail dies (union-find), cluster sizes,
      largest cluster, its centroid and shape (elongation, for scratches) (docs/signatures.md)
- [x] **Scratch** detection: Hough transform over fail dies (line-shaped chains)
- [x] **Spatial randomness** test: is the fail field random or clustered (e.g. nearest-
      neighbour or join-count statistic against a binomial null)
- [x] Rule-based classifier: features → {none, center, donut, edge-loc, edge-ring, loc,
      scratch, random, near-full}, thresholds fitted on a training split only
      (decision list as data, `explain()`, `config/rules.txt`; 'none' weighted to its natural share)
- [x] Tests: hand-built maps per pattern; properties (rotation of a map rotates sectors,
      cluster labels partition the fail dies, empty / full / single-die maps)
- [x] AVX2 backend for the features; tested equal to scalar on random maps
      (target attributes + run-time dispatch, ADR-0006; equal on all 103,698 real maps)
- [x] Benchmark (Google Benchmark): maps/s scalar vs AVX2, by map size (24², 30², 40², 64²)
      (3.2× / 3.7× / 4.0× / 5.4×; docs/avx2.md)
- [x] **Evaluation**: on WaferLens's 24,090 sorted maps against `wafer_pattern_truth`, and on
      WM-811K's labelled lot-disjoint test set: per-pattern recall/precision, macro-F1, next to
      FabEye's CNN and a trivial baseline (yield threshold). Report where rules win and lose.
      (rules 0.655 / CNN 0.858 / trivial 0.118 on WM-811K test; 0.645 / 0.909 / 0.132 on WaferLens)
- [x] `docs/evaluation.md` with the numbers and how to regenerate them
- [x] ADR-0004: wafer-map file format and exact integer geometry
- [x] ADR-0005: classical, explainable signatures next to the CNN (why, and what each is for)

## Phase 2a — CUDA backend for the classical detectors (4–6 days; the learning-heavy phase)

Same features and clustering as phase 1, batched on the GPU, tested equal to the scalar
reference, profiled.
- [x] CUDA build integration (`CMAKE_CUDA_ARCHITECTURES 86`), a `Backend` concept the CPU and
      GPU paths both satisfy
- [x] Kernel 1: per-map features (radial / angular histograms) with **shared-memory atomics**,
      one block per map; batches of N maps (equal to scalar on all 103,698 real maps; a
      warp-aggregated v2 measured 2x slower; crossover vs one AVX2 core at ~300 maps, docs/gpu.md)
- [x] Kernel 2: **connected-component labelling** on GPU (iterative union-find, e.g. the
      "label equivalence" / Playne–Hawick approach); compare with CPU union-find
      (atomicMin union-find; summary == CPU on all real maps; ~12× one core)
- [x] Kernel 3: **Hough transform** voting with atomics, then a peak search
      (one thread per angle instead: no atomics; == CPU on all real maps; 26–84× one core)
- [ ] Memory: pinned host buffers, device buffer pools, **CUDA streams** overlapping copy and
      compute, no `cudaMalloc` per batch
      (pinned buffers, reused device buffers, no per-batch allocation, CUDA Graphs per batch shape:
      done; overlapping packing of one batch with the GPU work of the previous: still open)
- [x] Tests (local, GPU-labelled): bit-for-bit equal features and equivalent labellings to the
      scalar reference on random batches
- [x] Profiling with Nsight Compute / Systems: occupancy, memory throughput, a roofline-style
      explanation of each kernel's bound; one optimisation iteration with before/after numbers
      (v1 LSU-bound, v2 2x slower explained; Hough −33% from removing a barrier-bound chunk)
- [x] Benchmark: throughput (maps/s) and latency per batch for CPU scalar, CPU AVX2 (all
      threads), GPU, across batch sizes 1 … 65,536; **find the crossover batch size**
      (all signatures: GPU passes the whole CPU at ~16 maps, ~9.4× at large batches)
- [ ] `docs/gpu.md`: what each kernel does, the profile, the crossover, and the latency vs
      throughput trade-off of batching
- [x] ADR-0007: GPU batching strategy (fixed batch, deadline-based, or adaptive): placement by
      measured cost, batches close at 4,096 maps or a 2 ms deadline

## Phase 2b — Inference engine: FabEye's CNN on our own kernels (5–7 days)

Run FabEye's trained CNN without ONNX Runtime or PyTorch: the core skills of an LLM inference
engine (GEMM, tensor cores, quantisation, fusion), learned on a model we know. The model
facts (layers, preprocessing, calibration) are in docs/context.md.
- [ ] Weights: export from FabEye's ONNX / checkpoint to a simple binary format (script in
      `tools/`), **fold BatchNorm into the convolutions** at export; a C++ loader with checks
      (shapes, a hash) — FabEye's repo is read, never modified
- [ ] Preprocessing in C++ exactly as FabEye's (`cv2.resize` INTER_NEAREST to 64×64, one-hot
      channels off / good / fail); a test against FabEye's Python output on real maps
- [ ] CPU reference forward pass (scalar, readable): conv3×3 + ReLU, maxpool, global average
      pool, linear, softmax; **matches ONNX Runtime's logits** on 1,000 maps (max abs diff
      tolerance stated)
- [ ] **GEMM ladder** (the CUDA learning core): naive → shared-memory tiling → register
      blocking → vectorised loads → **tensor cores** (WMMA / `mma.sync`, fp16 in, fp32 accumulate);
      each step benchmarked against cuBLAS (% of cuBLAS throughput), cuBLAS used only as the
      yardstick
- [ ] Convolution as **implicit GEMM** on top of it; **fused** conv + bias + ReLU (+ maxpool)
      kernels; layer timings before/after fusion
- [ ] **int8 quantisation**: per-channel weight scales, activation scales calibrated on held-out
      maps (never the test set), int8 tensor-core GEMM
- [ ] Batching with streams and pinned memory (shared with phase 2a); CUDA Graphs for a fixed
      batch shape
- [ ] **Accuracy must survive**: fp32 / fp16 / int8 macro-F1 on FabEye's lot-disjoint test set,
      and **conformal coverage** with FabEye's calibration (90% sets still cover ≈ 90%; the
      selective-accept rule at 0.688 still keeps error ≈ 2%). If int8 breaks coverage, report it
      and recalibrate, don't hide it
- [ ] Benchmark: wafers/s and per-batch latency against ONNX Runtime CPU (FabEye's ~412/s) and
      GPU, PyTorch, and TensorRT if installable; Nsight profile of the hottest kernel
- [ ] The CNN as a third detector in the pipeline (phase 4) next to the rules; agreement / fusion
      of rule-based and CNN signals reported
- [ ] `docs/inference.md` (the GEMM ladder table, quantisation results, coverage check)
- [ ] ADR: own kernels instead of TensorRT/ONNX Runtime (why: learning and control; what
      it costs), and the int8 scheme

## Phase 3 — SECS-II codec and HSMS transport (4–5 days)

The equipment protocol, from the bytes up.
- [ ] SECS-II **item codec**: all formats (List, Binary, Boolean, ASCII, I1–I8, U1–U8, F4/F8,
      JIS-8), length bytes, nesting; **zero-copy decode** to views over the receive buffer;
      encode into a reusable buffer
- [ ] `constexpr` format tables; `std::expected`-style (or own `Result`) error handling, no
      exceptions on the hot path
- [ ] **HSMS** (SEMI E37) transport on Asio coroutines: 10-byte header, data and control
      messages, Select / Deselect / Linktest / Separate, T3 / T5 / T6 / T7 / T8 timers,
      reconnect; active (host) and passive (equipment) roles
- [ ] HSMS connection state machine as an explicit, tested type
- [ ] GEM subset (SEMI E30): S1F1/F2 (are you there), S1F13/F14 (establish communication),
      S6F11/F12 (event report: carries the wafer map and lot/wafer ids), S5F1/F2 (alarm),
      S2F41/F42 (host command: HOLD / RELEASE lot); communication and control state machines
- [ ] Tests: round-trip properties (encode → decode is identity) on random item trees; golden
      byte vectors for known messages; timer behaviour with a fake clock
- [ ] **Fuzzing**: libFuzzer target on the decoder (and HSMS framing), run in CI for a fixed
      time; corpus checked in; any crash becomes a regression test
- [ ] Benchmark: messages/s decoded and encoded, allocation count per message (should be 0)
- [ ] `docs/secs.md`: the subset implemented, message layouts used, what is out of scope
- [ ] ADRs: zero-copy views and error handling; Asio coroutines for HSMS

## Phase 4 — The edge pipeline (3–4 days)

Putting it together, closed loop, with honest tail latency.
- [ ] **Tool emulator** (passive HSMS equipment): replays WaferLens maps in `tested_at` order
      as S6F11 events at a configurable rate, honours S2F41 HOLD (stops sending that lot)
- [ ] **Edge host**: network thread → **lock-free SPSC ring** → analytics thread(s) → decision
      thread → S2F41 back; cache-line aligned slots, no false sharing, optional thread pinning
- [ ] Lock-free queue tests: TSan, a stress test with a checker, a benchmark against a mutex
      queue
- [ ] **Allocation-free hot path**: `std::pmr` / arena buffers; a test that fails on any
      allocation per message
- [ ] Decision rule: hold the lot after k wafers with the same signature within a window
      (k configurable); also raise S5F1 alarms
- [ ] **Latency measurement**: per wafer, from the emulator's send timestamp to the HOLD being
      received, on a steady clock; HDR-style histogram; constant-rate load so coordinated
      omission is accounted for; p50 / p99 / p99.9 / max
- [ ] Experiments: rules on CPU vs GPU vs the CNN (fp16 / int8), batch size / deadline, load
      (maps/s) vs tail latency
- [ ] **Closed-loop result** on a WaferLens spatial excursion: wafers processed before the hold,
      against WaferLens's batch pattern alarm (median 52 h, mostly sort lag) and against no hold
- [ ] `docs/pipeline.md` and an ADR (threading and queue design)

## Phase 5 — Python bindings and launch (2–3 days)

- [ ] pybind11 module: features, classifier and clustering on NumPy arrays (zero-copy buffer
      protocol); wheel via scikit-build-core; tested from pytest
- [ ] Optional: a WaferLens-side script comparing WaferEdge's rule-based patterns with FabEye's
      CNN on the same wafers (lives in WaferEdge, reads WaferLens data; WaferLens unchanged)
- [ ] README: the question, architecture, results tables (accuracy vs FabEye, CPU vs GPU
      throughput and crossover, decode rate, tail latency, closed-loop result), how to
      reproduce each, limitations (simulated data, GPU tests local only)
- [ ] Release workflow (as in Path-Finding-Visualiser): tag `v1.0.0` → Linux build artifacts
      (and the Python wheel); release notes in `docs/release-notes/`
- [ ] Website project page and CV line (the user's personal site: ~/personal-web-mig/ansonantony-v2)

## Phase 6 (stretch) — LLM incident note on the same engine (1–2 weeks)

Only after phases 0–5 ship. When the edge host holds a lot, a small local LLM turns the
verified facts into a short note for the engineer. The LLM summarises facts; it never diagnoses.
- [ ] Model choice (ADR): Qwen2.5 0.5B / 1.5B or TinyLlama 1.1B instruct; fits 6 GB with int8 /
      int4 weights; licence checked
- [ ] Transformer kernels on the phase-2b GEMM: RMSNorm, RoPE, softmax, **fused attention**
      (FlashAttention-style tiling), KV cache, int8 / int4 weight dequantisation, sampling
- [ ] Tokenizer (BPE) in C++ or a pinned library; logits match the reference implementation
      (llama.cpp or Hugging Face) on fixed prompts
- [ ] Prompt from structured facts only: pattern, confidence and prediction set (CNN), zone
      statistics and k-in-a-row rule (rules), lot / wafer ids, optional WaferLens commonality
      suspect
- [ ] **Faithfulness check, automatic**: every number, pattern, chamber and id in the note must
      appear in the input facts; hallucination rate over ≥ 500 generated notes, before and after
      prompt / decoding fixes
- [ ] Benchmark: tokens/s (prefill and decode) and memory against llama.cpp on the same GPU;
      note latency stays off the hold path (the hold never waits for the note)
- [ ] `docs/llm.md`, ADR on the model and quantisation

---

## Rules for the whole project
- Every phase ends green in CI and with a measured number.
- Every speed claim names the machine, the build type and the command that produced it.
- No backend ships without a test against the scalar reference.
- Commits: `type(scope): subject` + 3–5 bullets; one feature branch per checklist group;
  no Claude attribution.
