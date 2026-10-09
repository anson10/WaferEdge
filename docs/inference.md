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

## Next

Convolution as implicit GEMM on top of these kernels, fusion (conv + bias + ReLU + pool), then
int8.
