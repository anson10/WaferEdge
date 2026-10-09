# Inference engine (phase 2b)

FabEye's CNN on WaferEdge's own code: first a CPU reference that provably matches ONNX
Runtime, then CUDA kernels checked against it (the GEMM ladder, convolution as GEMM, fusion,
int8). Why own kernels at all: ADR-0008.

## The model

FabEye's `WaferCNN`, the deployed lot-disjoint model (`~/FabEye/serving/wafer_cnn.onnx`,
SHA-256 checked against FabEye's calibration file):

| Stage | Output | Multiply-adds |
|---|---|---|
| input: one-hot (off wafer, good, fail) | 3 × 64 × 64 | |
| block 1: conv 3→32, conv 32→32, maxpool | 32 × 32 × 32 | 3.5M + 37.7M |
| block 2: conv 32→64, conv 64→64, maxpool | 64 × 16 × 16 | 18.9M + 37.7M |
| block 3: conv 64→128, conv 128→128, maxpool | 128 × 8 × 8 | 18.9M + 37.7M |
| block 4: conv 128→256, conv 256→256, maxpool | 256 × 4 × 4 | 18.9M + 37.7M |
| global average pool, linear 256→9 | 9 logits | 2.3k |

Every conv is 3×3, padding 1, followed by ReLU; BatchNorm is folded into it. 1,174,569
parameters, ~211M multiply-adds (~0.42 GFLOP) per map.

## Export: folding BatchNorm

At inference BatchNorm is a fixed per-channel affine map, `y = γ(x − μ)/√(σ² + ε) + β`, right
after a linear convolution, so the two collapse into one convolution:

```
s  = γ / √(σ² + ε)
W' = W · s          (each output channel's weights scaled)
b' = (b − μ) · s + β
```

`tools/export_cnn.py` folds FabEye's PyTorch checkpoint this way and compares the result with
the weights PyTorch's ONNX exporter folded into the deployed model: **max difference 4.8e-7**
(two independent folds agree to float32 rounding). The `.wcnn` file holds the 18 tensors, an
FNV-1a checksum and the ONNX model's SHA-256; the loader refuses damaged files and wrong
shapes.

## Preprocessing: matching OpenCV exactly

FabEye resizes each map to 64×64 with `cv2.resize(..., INTER_NEAREST)`, then one-hot encodes
it. OpenCV computes a source index as `floor(i · (1 / (64 / n)))` in double precision. The
obvious exact formula `i · n / 64` gives a **different source pixel at lengths 186, 198, 210,
234, 246, …** (where `1 / (64 / n)` rounds just below `n / 64`), and WM-811K has maps up to 212
dies a side. `cnn::nearest_source` uses OpenCV's arithmetic and is tested against cv2's output
for every source length 1..512 (`tests/data/cv2_nearest_64.bin`, written by
`export_cnn.py --cv2-fixture`).

## The reference forward pass

`cnn::ReferenceForward` (`src/cnn_reference.cpp`): direct 3×3 convolutions, ReLU, 2×2 max
pooling, global average pooling, the linear layer. Plain loops on purpose; their order (output
channel, input channel, kernel tap, row, then along the row innermost) lets the compiler
vectorise the inner loop without obscuring anything. Unit tests: hand-checked convolution,
an independent bounds-checked convolution on random data, max pooling, a numerically stable
softmax, model-file damage.

**Against ONNX Runtime** (`waferedge-cnn verify`; reference logits from FabEye's serving path,
written by `export_cnn.py`):

| Maps | Max \|Δlogit\| | Mean \|Δlogit\| | Same class |
|---|---|---|---|
| 1,000 WM-811K test maps | 7.6e-6 | 1.0e-6 | 1,000 / 1,000 |
| 500 WaferLens maps | 8.6e-6 | 8.2e-7 | 500 / 500 |

Differences of ~1e-5 are summation order (ONNX Runtime's kernels add in another order and with
fused multiply-adds), three orders of magnitude below anything that changes a prediction.

**FabEye's accuracy, reproduced** (`waferedge-cnn eval ... test`, all 25,875 lot-disjoint test
maps): **macro-F1 0.858, accuracy 0.965**, and every per-class F1 equal to the ONNX Runtime
column of docs/evaluation.md (none 0.983, center 0.888, donut 0.899, edge_loc 0.804, edge_ring
0.978, loc 0.750, near_full 0.906, random 0.807, scratch 0.706).

**Speed**: 60–65 maps/s on 12 threads (Ryzen 5 7535HS, release build, ~5 maps/s per thread,
~2.3 GFLOP/s per core). This is the floor the GPU ladder starts from; FabEye's ONNX Runtime
CPU path does ~412 maps/s at batch 32 (FabEye's README).

## Reproduce

```sh
python3 tools/export_maps.py waferlens ~/waferLens/data/demo data/waferlens_demo.wmap
python3 tools/export_maps.py wm811k ~/FabEye/data/wm811k/processed_lot.pkl data/wm811k_lot.wmap
python3 tools/export_cnn.py --out data           # weights + ONNX Runtime reference logits
cmake --workflow --preset release
build/release/tools/waferedge-cnn verify data/fabeye_cnn.wcnn data/wm811k_lot.wmap data/fabeye_logits_wm811k.bin
build/release/tools/waferedge-cnn verify data/fabeye_cnn.wcnn data/waferlens_demo.wmap data/fabeye_logits_waferlens.bin
build/release/tools/waferedge-cnn eval data/fabeye_cnn.wcnn data/wm811k_lot.wmap test

# GPU (cuda build): fp32 / fp16 / int8, conformal coverage, agreement
cmake --workflow --preset cuda
build/cuda/tools/waferedge-cnn calibrate data/fabeye_cnn.wcnn data/wm811k_lot.wmap data/fabeye_int8_scales.txt
I8="--gpu int8 --scales data/fabeye_int8_scales.txt"
build/cuda/tools/waferedge-cnn eval data/fabeye_cnn.wcnn data/wm811k_lot.wmap test $I8
build/cuda/tools/waferedge-cnn conformal data/fabeye_cnn.wcnn data/wm811k_lot.wmap data/fabeye_conformal.txt test $I8
build/cuda/tools/waferedge-cnn agree data/fabeye_cnn.wcnn data/wm811k_lot.wmap all $I8
build/cuda/bench/bench-cnn --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
```

## The GEMM ladder

Every 3×3 convolution is a matrix multiply once the data is lined up (weights: C_out × C_in·9;
input patches: C_in·9 × H·W·batch); ~97% of the CNN's arithmetic is GEMM. `cuda/gemm.cu`
builds C = A·B (row-major) one rung at a time, each tested against a CPU reference that
accumulates in double (`tests/test_gpu_gemm.cpp`: random matrices, sizes with partial tiles on
every side; fp16 inputs rounded the same way on both sides for the tensor-core rung), and
measured against cuBLAS (`bench/bench_gemm.cpp`; cuBLAS is linked into the benchmark only).

| Rung | The idea |
|---|---|
| 1 Naive | a thread per element of C, A and B read from global memory |
| 2 Tiled | 32×32 tiles of A and B in shared memory, each value used by 32 threads |
| 3 Register-blocked | 128×128 block tiles; each thread computes an 8×8 patch in registers: 16 shared loads feed 64 multiply-adds (0.25 per FMA instead of 2) |
| 4 Vectorised | rung 3 with `float4` global loads and A transposed in shared memory (two `float4` reads for a thread's 8 values) |
| 5 Tensor cores | WMMA: fp16 A and B, fp32 accumulation; 4 warps own a 64×64 tile, each `mma_sync` does a 16×16×16 product |

### Measuring on a laptop GPU: count cycles, not seconds

Under load this GPU runs at ~0.96–1.28 GHz, not its 2.1 GHz maximum, throttled by its power cap
and temperature (`nvidia-smi` reports throttle reasons 0x24: software power cap + thermal
slowdown). The clock depends on the kernel and on what ran before, so two kernels timed in
seconds may have run at different clocks. Even Nsight Compute's clock lock doesn't hold here:
our tensor-core kernel and cuBLASLt's ran at 1.15 and 0.96 GHz in the same session. A first
wall-clock comparison put our tensor-core kernel 5–23% *ahead* of cuBLAS; in cycles it is
behind. **Comparisons below are cycles** (`ncu --metrics sm__cycles_elapsed.max`, one launch
of each kernel at 2048³), which don't depend on the clock.

| Rung (2048³) | Cycles | FLOP / cycle | vs cuBLAS, same precision |
|---|---|---|---|
| 1 Naive | 54.6M | 314 | 10% |
| 2 Tiled | 46.2M | 372 | 12% |
| 3 Register-blocked | 10.3M | 1,671 | 53% |
| 4 Vectorised | 7.53M | 2,281 | 72% |
| cuBLAS `sgemm` (fp32) | 5.43M | 3,165 | 100% |
| 5 Tensor cores (WMMA, fp16 → fp32) | 2.16M | 7,950 | **92%** (2.5× cuBLAS fp32) |
| cuBLAS `GemmEx` / cuBLASLt best (fp16 → fp32) | 2.00M | 8,590 | 100% |

At 4096³ the tensor-core rung is ~80% of cuBLASLt's best (17.4M vs 14.0M cycles): cuBLAS's
kernel (CUTLASS `s16816gemm_f16_256x128_32x3`) uses 256×128 tiles, 218 registers per thread, a
3-stage `cp.async` pipeline (next tiles load while the current ones compute) and has no
shared-memory bank conflicts; ours has 64×64 tiles, one stage and 20M bank conflicts. Those
are the next rungs if the conv kernels need them.

### Why each rung is faster (Nsight Compute, 2048³)

| | Load/store pipe | FMA pipe | Top stall | |
|---|---|---|---|---|
| 1 Naive | 99% | 19% | lg_throttle 56% (global load queue full) | L1 hit rate 87%: the cache does the reuse the code doesn't |
| 2 Tiled | 81% | 8% | mio_throttle 58% (shared-memory queue full) | 2 shared loads per FMA; 1,024-thread blocks: 67% occupancy |
| 3 Register-blocked | 51% | 39% | not_selected / selected (ready to issue) | 112 registers: 32% occupancy; 67M bank conflicts |
| 4 Vectorised | 30% | 48% | not_selected, short_scoreboard | bank conflicts halved (33.6M) by the transposed A |

Rungs 1 and 2 are bound by the *number* of load instructions, not by DRAM bandwidth: tiling
moved the queue from global to shared memory, which is why it gained only ~20%. Register
blocking is the big step: the stalls turn from "waiting for memory" into "waiting for an issue
slot", what a compute-bound kernel looks like.

Wall-clock GFLOPS for reference (`bench-gemm`, median of 3; clocks vary, see above): at 4096³
naive ~470, tiled ~610, register-blocked ~2,650, vectorised ~3,600, cuBLAS sgemm ~4,150; tensor
cores 6,000–11,300 depending on the clock, cuBLAS fp16 ~10,000–11,700.

## The CNN on the GPU

`gpu::CnnEngine` (`include/waferedge/gpu_cnn.hpp`, `cuda/cnn_kernels.cu`, `cuda/cnn_engine.cu`):
raw bins up, logits down.

- **Implicit GEMM.** A 3×3 convolution is `out = W · P`: W the weights (C_out × C_in·9), P the
  patches (C_in·9 × images·H·W). P is never built (im2col would write 9× the input): while a
  tile of P is staged in shared memory, each element is gathered from its row k → (channel,
  ky, kx) and column p → (image, y, x), or 0 in the padding.
- **Layout [channel][image][y][x]**: a GEMM's output (channel × pixels) is already the next
  layer's input. No transposes.
- **Fusion**: bias + ReLU in the GEMM's epilogue, on values still in registers, instead of a
  second kernel that reads and writes every activation again.
- **Preprocessing on the GPU**: the raw bins (0.6–1.6 KB a map) go up instead of the 48 KB
  one-hot tensor; a kernel does OpenCV's nearest rule (same double-precision arithmetic, same
  pixels) and the one-hot encoding.
- **fp32**: an implicit-GEMM kernel in the register-blocked style (64×64 tiles, 4×4 per
  thread). **fp16**: the WMMA rung with the gather, fp16 weights (K padded to a multiple of 32)
  and activations, fp32 accumulation, an epilogue through shared memory. **int8**: below.
- Then 2×2 max pooling, and average pooling + the linear layer (one block per image).

**Correctness** (`tests/test_gpu_cnn.cpp`, a random model so no FabEye data is needed): GPU fp32
logits equal the CPU reference within 1e-4 with the same class on maps from 1×1 to 212×204,
including lengths where OpenCV's rounding matters; fp16 within 3%; fused equal to unfused bit
for bit in fp32; batches across the 1,024-map chunk boundary equal single-map runs.

**FabEye's model** (`waferedge-cnn verify | eval | agree ... --gpu fp32|fp16`):

| | vs ONNX Runtime: max \|Δlogit\| | Same class (1,500 maps) | WM-811K test macro-F1 |
|---|---|---|---|
| GPU fp32 | 7.6e-6 / 7.2e-6 | 1,500 / 1,500 | **0.858** (per class = FabEye) |
| GPU fp16 | 3.9e-3 / 5.7e-3 | 1,500 / 1,500 | **0.858** (per class = FabEye) |

Over every map, fp16 against fp32: **3 of 79,608 WM-811K maps change class** (0.004%, near-ties;
none in the test split), **0 of 24,090 WaferLens maps**; largest logit change 8.1e-3. Conformal
coverage: below.

### Speed

End to end per batch (`bench-cnn`, 40×40 maps; bins up, preprocessing, network, logits down),
against FabEye's model on ONNX Runtime CPU and PyTorch on this GPU (`tools/bench_cnn_yardsticks.py`,
FabEye's checkpoint, cuDNN autotuning; **model time only, inputs already on the GPU**: the best
case for the yardsticks). Maps/s, median of 3, all columns in one session in which the SM clock
sat at 0.83–0.99 GHz (power and thermal cap; wall clock on this laptop, read as ±10–20%):

| Batch | ONNX Runtime CPU | PyTorch fp32 | PyTorch fp16 | WaferEdge fp32 | WaferEdge fp16 | WaferEdge int8 |
|---|---|---|---|---|---|---|
| 1 | 540 | 304 | 254 | 500 | 964 | **1,212** (0.83 ms) |
| 16 | 657 | 2,906 | 3,789 | 3,074 | 8,913 | 9,845 |
| 64 | 710 | 3,562 | 5,942 | 3,196 | 11,184 | 12,439 |
| 256 | 790 | 3,886 | 6,283 | 3,408 | 10,960 | **12,960** |
| 1,024 | — | 4,174 | 6,585 | 3,218 | 10,838 | 13,379 |

- **Large batches: fp16 ~1.7× PyTorch fp16 and int8 ~2×**, although PyTorch's numbers exclude
  the upload and preprocessing ours include. fp32 stays below PyTorch fp32 (77–88%): its kernel
  is the plain register-blocked rung, bound by FMAs rather than the gather.
- **One map: 0.55–1.0 ms** in fp16 / int8 depending on the clock (0.55 ms int8 at a higher clock
  in the before/after run below), ~4× PyTorch's 3.3–3.9 ms: no framework overhead per layer.
  That's the in-line case, a wafer at a time at the tool.
- **~16× FabEye's serving path** (ONNX Runtime CPU) at batch 256 in int8.

### The gather, rewritten: −74% (fp16) and −70% (int8) of the convolution cycles

The first version of these kernels (PR #13, #14) kept the tensor cores ~10% busy: warps
stalled on the gather's integer arithmetic and on its loads, with only 12 (fp16) or 16 (int8)
of 48 warps resident per SM to hide them. Five changes, each measured (Nsight Compute, sum of
the 8 convolutions' `sm__cycles_elapsed.max`, batch 256; cycles, because the clock drifts):

| Step | fp16 | int8 | What changed |
|---|---|---|---|
| before | 66.6M | 47.0M | occupancy 25% / 33%, shared memory the limit |
| 1. K tap-major | 29.3M | 27.3M | one tap per K step: bounds check and offset once per step |
| 2. 1-D grid | 27.5M | 27.3M | blocks sharing input run together (conv 8 DRAM 284 → 38 MB) |
| 3. per-warp epilogue | 19.0M | 25.7M | 17 KB float tile → 1.25 KB per warp: occupancy 49% / 57% |
| 4. packed, transposed staging | 18.7M | 14.0M | 16-byte stores, no bank conflicts; int8 K step 16 → 32 |
| 5. skip padding warps | **17.6M** | **13.9M** | conv 1–2 have 32 output channels: half the tile is zeros |

1. **K ordered tap-major.** The patch matrix's K was PyTorch's order, `channel * 9 + tap`: every
   element changed tap, so every element paid two divisions, four compares and a 64-bit
   address. With `k = tap * in_channels + channel` (the engine reorders the weights once, at
   upload; CUTLASS and cuDNN order implicit GEMM the same way), a K step of 8, 16 or 32 lies
   inside one tap for every layer but the first (3 channels; it keeps a per-element path). The
   tap's bounds check becomes one bit of a 9-bit mask computed per pixel at the start, and the
   elements are plain loads one channel plane apart. Half the cycles gone in one change.
2. **Block order.** The grid was (pixel tiles, channel tiles): all pixel tiles of channel tile 0
   ran, then all of tile 1 re-read the same input. Tap-major order made that visible: a
   channel's 9 taps are now far apart in K, and conv 8's 4 MB input (2 MB L2) was read from
   DRAM 4 times (284 MB). A 1-D grid with the channel tile as the fast index runs the blocks
   that share pixels together: 38 MB.
3. **Occupancy.** The tensor-core kernels staged their epilogue through a 64 × 64 float tile,
   17 KB of shared memory, which held an SM to 3 blocks (fp16). Each warp now writes its four
   16 × 16 fragments one at a time through its own 1.25 KB scratch (aliased onto the weight
   tile, free after the K loop). **The first attempt was slower** (fp16 28.0M, int8 32.7M) with
   9× the DRAM writes: the loop over `acc[i][j]` wasn't unrolled, so the compiler moved the
   accumulators to local memory (`ptxas -v`: a 128-byte stack frame, 0 bytes "spilled"; local
   memory is DRAM). `#pragma unroll` gives every index a compile-time value and puts them back
   in registers.
4. **Shared-memory stores.** int8 stored its transposed patch tile a byte at a time: 72M bank
   conflicts in conv 2 alone (fp16: 7.7M), and its 16-deep K step (forced by the 32-byte
   alignment of int8 fragment loads) doubled the barriers. Both tensor-core kernels now give
   each thread 16 consecutive k of one pixel, packed in registers and written as 16-byte
   stores; row pitches of 80 (fp16) and 48 (int8) bytes put 8 lanes' stores in 8 different
   bank groups. int8 steps K by 32, staged as two 16-deep halves in separate aligned arrays.
5. **Padding rows.** conv 1 and 2 have 32 output channels in a 64-row tile: two of the four
   warps multiplied zeros. They now skip the MMA (a warp-uniform branch) and still stage and
   synchronise. With the tensor pipe at 36% busy in fp16, this was now worth −19% on conv 2.

Per layer, after (cycles, batch 256; before → after):

| Layer | fp32 | fp16 | int8 |
|---|---|---|---|
| conv 1 (3 → 32) | 2.9M → 2.8M | 3.0M → 1.8M | 2.6M → 1.9M |
| conv 2 (32 → 32) | 22.3M → 19.9M | 19.0M → 4.2M | 14.1M → 4.0M |
| conv 3 (32 → 64) | 5.9M → 5.3M | 5.2M → 1.4M | 3.8M → 1.1M |
| conv 4 (64 → 64) | 11.5M → 10.7M | 10.0M → 2.6M | 7.0M → 1.9M |
| conv 5 (64 → 128) | 5.8M → 5.3M | 5.1M → 1.3M | 3.4M → 1.0M |
| conv 6 (128 → 128) | 11.4M → 10.5M | 10.2M → 2.5M | 6.5M → 1.8M |
| conv 7 (128 → 256) | 5.9M → 5.4M | 4.7M → 1.3M | 3.3M → 0.9M |
| conv 8 (256 → 256) | 11.7M → 10.6M | 9.4M → 2.6M | 6.4M → 1.7M |
| **total** | **77.2M → 70.6M** | **66.6M → 17.6M** | **47.0M → 14.2M** |

- **End to end, 2.4× in fp16 and int8** (old and new binaries interleaved, two rounds, batch
  256): fp16 5.2k → 12.6k maps/s, int8 6.7k → 16.2k; one map in int8 0.94 → 0.55 ms. fp32
  +3%: its kernel is FMA-bound, the gather was a smaller share of its time.
- **Tap-major order costs DRAM traffic** where a layer's input outgrows L2: fp32 conv 4 moves
  522 MB instead of 134 (a channel's 9 taps are no longer read back to back). fp16 and int8,
  which move half or a quarter of the bytes, aren't bound by it (conv 2 runs at ~30 GB/s of
  ~190).
- **Where it stands**: fp16 conv 2 has the tensor pipe 36% busy and stalls on the pipe and on
  barriers; int8 issues 2.9 instructions per cycle (of 4). The next gains are in the
  instruction count (the gather's loads, one per element) and in overlapping the next tile's
  loads with this tile's MMAs (double buffering).
- Accuracy unchanged: fp32 vs ONNX Runtime 7.63e-6, conformal coverage identical in all three
  precisions; fp16 vs fp32 changes 3 of 79,608 maps instead of 2 (a different summation order
  tips one more near-tie).
- Fusion of bias + ReLU saved 11% with the first kernels (`waferedge-cnn layers`); with these,
  wall-clock fused-vs-unfused pairs moved with the clock by more than the effect, so it isn't
  re-quoted.

## Conformal prediction: the stricter check

FabEye doesn't only predict a class: it returns a **prediction set** that contains the true
class with ~90% (or 95%) probability, and an **auto-accept** flag. Both are calibrated on
validation lots against the fp32 model's probabilities (`~/FabEye/serving/calibration.json`):
class k is in the set when `1 − p_k ≤ q_k` (one threshold per class), and a wafer is accepted
without review when its top probability is ≥ 0.688. A lower precision can keep the top class
and still shift probabilities enough to drop the true class from a set, so coverage is
checked separately from accuracy.

`waferedge-cnn conformal` mirrors FabEye's `describe()` (`src/cnn_conformal.cpp`; the
calibration exported as plain numbers to `data/fabeye_conformal.txt`, its model hash checked).
On our fp32 logits it **reproduces FabEye's published numbers to four decimals**, which
validates both the evaluation and the engine:

| WM-811K test, 25,875 maps | FabEye (published) | fp32 | fp16 | int8 |
|---|---|---|---|---|
| macro-F1 | 0.858 | 0.858 | 0.858 | 0.860 |
| Coverage, 90% target | 0.8925 | 0.8925 | 0.8926 | **0.8951** |
| Worst-class coverage, 90% | 0.8621 | 0.8621 | 0.8621 | 0.8621 |
| Mean set size, 90% | | 0.931 | 0.931 | 0.933 |
| Coverage, 95% target | 0.9467 | 0.9467 | 0.9467 | 0.9477 |
| Worst-class coverage, 95% | 0.8750 | 0.8750 | 0.8750 | 0.8750 |
| Auto-accepted | 96.17% | 96.17% | 96.17% | 96.22% |
| Error among accepted | 1.86% | 1.86% | 1.86% | **1.79%** |

**Coverage held in both reduced precisions**, the worst class didn't move, and the auto-accept
rule still keeps its error under FabEye's 2% target. int8's +0.002 macro-F1 is noise (122
changed predictions netting slightly positive), not an improvement: the claim is "unchanged".

## int8

Each number becomes an 8-bit integer and a scale, `x ≈ x_int · s`:

- **Weights: symmetric per output channel**, `s_w[c] = max |W[c, :]| / 127` (each filter keeps
  its own range).
- **Activations: symmetric per layer**, `s_a = max activation / 127`, the maxima **calibrated on
  2,048 validation maps, never test** (`waferedge-cnn calibrate`, from the fp32 engine with a
  max-reduction after each convolution; written to `data/fabeye_int8_scales.txt`). After ReLU
  activations are ≥ 0, so signed int8 uses 0…127: 7 bits. Unsigned activations would add a
  bit; accuracy didn't call for it.
- **Input**: the one-hot 0 / 1 is exactly 0 / 127 (`s = 1/127`): no rounding at the first layer.
- **Kernel**: WMMA on `signed char` fragments, int32 accumulation (exact). Epilogue:
  `y = acc · (s_w[c] · s_in) + bias → ReLU → round(y / s_out)` clamped to 0…127. int8 fragment
  loads need 32-byte aligned addresses, which the second 16 of a 32-wide row break: a 32-deep
  K step is staged as two 16-deep halves in separate arrays, and the patch tile is stored
  transposed (read as a column-major fragment) so every fragment starts aligned. The head
  dequantises the last block and runs in float.

**Agreement over every map** (`waferedge-cnn agree --gpu int8`): **122 of 79,608 WM-811K maps
(0.15%) and 5 of 24,090 WaferLens maps change class** against fp32 (fp16: 3 and 0); largest
logit change 0.32. Against ONNX Runtime: max |Δlogit| 0.21, same class on 998 of 1,000 test
references and 500 of 500 WaferLens. Tests: the GPU calibration equals each layer's maximum
computed on the CPU; int8 logits within 5% (relative) of the CPU reference on a random model
(2.2% measured); an int8 engine without scales refuses.

**Speed.** With the first kernels, int8 was only 16% faster than fp16 end to end (−15 to −28%
cycles per conv, DRAM traffic halved): both were bound by the gather, with the tensor cores
~10% busy, so halving the bytes moved little. After the gather rewrite above, int8 is 20–25%
faster than fp16 per conv in cycles (14.2M vs 17.6M) and 11–29% end to end at batch 64 and up.
Still not 2×: the gather issues one load per element in both precisions, and that instruction
count, not the multiply, sets the pace.

TensorRT isn't installed on this machine (no Python package, no `trtexec`): not measured.

## Next

CUDA Graphs and streams for the engine (one map is ~0.5–1 ms, much of it launch overhead on
WSL2), then the CNN joins the pipeline (phase 4) as the third detector next to the rules.
Kernel-side, the next steps are double buffering (load the next K step while this one
multiplies) and wider gather loads.
