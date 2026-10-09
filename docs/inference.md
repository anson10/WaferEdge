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

Over every map, fp16 against fp32: **2 of 79,608 WM-811K maps change class** (0.0025%, near-ties;
none in the test split), **0 of 24,090 WaferLens maps**; largest logit change 8.1e-3. Conformal
coverage: below.

### Speed

End to end per batch (`bench-cnn`, 40×40 maps; bins up, preprocessing, network, logits down),
against FabEye's model on ONNX Runtime CPU and PyTorch on this GPU (`tools/bench_cnn_yardsticks.py`,
FabEye's checkpoint, cuDNN autotuning; **model time only, inputs already on the GPU**: the best
case for the yardsticks). Maps/s, median of 3 (wall clock; this GPU's clock drifts, so read as
±10%):

| Batch | ONNX Runtime CPU | PyTorch fp32 | PyTorch fp16 | WaferEdge fp32 | WaferEdge fp16 |
|---|---|---|---|---|---|
| 1 | 452 | 282 | 275 | 743 | **915** (1.09 ms) |
| 16 | 656 | 3,006 | 3,870 | 2,714 | 4,063 |
| 64 | 606 | 3,724 | 6,258 | 3,106 | 4,752 |
| 256 | 743 | 3,909 | 6,560 | 3,231 | 4,899 |
| 1,024 | — | 4,215 | 6,790 | 3,472 | 5,047 |

- **One map: ~3.3× PyTorch** (1.1 ms vs 3.6 ms): no framework overhead per layer. That's the
  in-line case (a wafer at a time at the tool).
- **Large batches: ~74–76% of PyTorch fp16, ~83–90% of fp32**, even though PyTorch's numbers
  exclude the upload and preprocessing ours include. cuDNN picks tuned kernels per layer; ours
  is one generic implicit GEMM.
- **~6.6× FabEye's serving path** (ONNX Runtime CPU) at batch 256, ~80× the CPU reference.

### Where the time goes (`waferedge-cnn layers`, 256 maps, fastest of 5 runs, ms)

| Layer | fp32 fused | fp32 unfused | fp16 fused | fp16 unfused |
|---|---|---|---|---|
| preprocess | 0.41 | 0.23 | 0.18 | 0.28 |
| conv 1 (3 → 32) | **2.25** | **4.51** | **1.85** | **3.04** |
| conv 2 (32 → 32) | 17.00 | 19.02 | 11.62 | 13.03 |
| maxpool 1 | 1.46 | 1.46 | 0.73 | 0.73 |
| conv 3 (32 → 64) | 4.40 | 5.46 | 3.29 | 3.95 |
| conv 4 (64 → 64) | 8.53 | 9.50 | 6.27 | 6.97 |
| maxpool 2 | 0.73 | 0.73 | 0.37 | 0.37 |
| conv 5 (64 → 128) | 4.34 | 4.80 | 3.26 | 3.62 |
| conv 6 (128 → 128) | 8.57 | 9.02 | 6.41 | 6.84 |
| maxpool 3 | 0.37 | 0.37 | 0.19 | 0.19 |
| conv 7 (128 → 256) | 4.42 | 4.62 | 2.99 | 3.20 |
| conv 8 (256 → 256) | 8.83 | 9.11 | 6.03 | 6.25 |
| maxpool 4 | 0.19 | 0.19 | 0.09 | 0.10 |
| head | 0.08 | 0.08 | 0.06 | 0.06 |
| **GPU total** | **61.6** | **69.1** | **43.4** | **48.6** |

- **Fusion saves 11%**, most where activations are big and the GEMM is small: conv 1 (K = 27)
  halves, since the separate bias + ReLU pass over 33.5M activations cost as much as the
  convolution.
- **Conv 2 is 27% of the time** with the same 37.7M multiply-adds per map as conv 4, 6, 8: its
  32 output channels fill half of a 64-row tile (Nsight Compute: 18.7M cycles vs conv 4's 9.8M).
- **Max pooling is 3% of the time**: fusing it into the convolution isn't worth it yet.
- Every conv kernel keeps the tensor cores only ~10% busy (the plain GEMM rung: 38%); stalls
  are `wait` (the gather's divide / modulo) and `long_scoreboard` (the gathered loads). The
  cost of implicit GEMM is the gather, not the multiply.

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
  loads need 32-byte aligned addresses, which a 16-element step inside a 32-wide tile breaks:
  the kernel steps K by 16 and stores the patch tile transposed (read as a column-major
  fragment) so every fragment starts aligned. The head dequantises the last block and runs in
  float.

**Agreement over every map** (`waferedge-cnn agree --gpu int8`): **122 of 79,608 WM-811K maps
(0.15%) and 5 of 24,090 WaferLens maps change class** against fp32 (fp16: 2 and 0); largest
logit change 0.32. Against ONNX Runtime: max |Δlogit| 0.21, same class on 998 of 1,000 test
references and 500 of 500 WaferLens. Tests: the GPU calibration equals each layer's maximum
computed on the CPU; int8 logits within 5% (relative) of the CPU reference on a random model
(2.2% measured); an int8 engine without scales refuses.

**Speed.** End to end (`bench-cnn`, same session): +16% over fp16 at batch 64–256 (e.g. 5,259
vs 4,546 maps/s at 256), 1.02 vs 1.11 ms for one map. In cycles (Nsight Compute, batch 256):

| | fp16 | int8 | |
|---|---|---|---|
| conv 2 | 16.5M cycles | 14.1M | −15% |
| conv 4 | 9.8M cycles | 7.0M | −28% |
| DRAM traffic, conv 4 | 67 MB | 33 MB | halved |
| tensor pipe busy | ~10% | ~7% | |

int8 doesn't approach 2× because the multiply was never the bottleneck: the kernels are bound
by the implicit-GEMM gather (stalls on its integer arithmetic and scattered loads, tensor
cores ~10% busy). int8 gains by moving half the bytes. A wall-clock per-layer comparison of
fp16 and int8 run back to back came out 4× apart because the GPU throttled during the first
run; the cycles above are the comparison to trust.

TensorRT isn't installed on this machine (no Python package, no `trtexec`): not measured.

## Next

The gather is the bottleneck of every precision: precomputed offsets for k → (channel, tap),
32-row tiles for the 32-channel layers, vectorised patch loads. Then the CNN joins the
pipeline (phase 4) as the third detector next to the rules.
