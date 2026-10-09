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

## Next

The GEMM ladder on the GPU (naive → shared-memory tiling → register blocking → vectorised
loads → tensor cores), each step measured as % of cuBLAS, then convolution as implicit GEMM,
fusion and int8.
