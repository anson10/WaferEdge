# ADR-0008: FabEye's CNN on our own kernels, checked against ONNX Runtime

- **Status:** Accepted (the int8 scheme is added when phase 2b reaches it)
- **Date:** 2026-10-09

## Context
WaferEdge runs FabEye's CNN next to the classical detectors (phase 2b). FabEye serves it with
ONNX Runtime on the CPU (~412 wafers/s at batch 32, FabEye's README); PyTorch on this GPU
reaches ~4,363 wafers/s. The project's other goal is to learn what an inference engine does:
GEMM, tensor cores, convolution as GEMM, fusion, quantisation (the same pieces an LLM engine
is made of, phase 6), on a model whose training and evaluation are already understood.

## Options considered
1. **ONNX Runtime (CPU or CUDA execution provider)**: the model as served, no new code; a
   black box, and a ~100 MB dependency in the shipped path.
2. **TensorRT**: the fastest NVIDIA path, with fusion and int8 built in; a black box too, a
   large and version-pinned dependency, and nothing learned about how it gets there.
3. **cuDNN / cuBLAS calls for each layer**: fast building blocks, our graph; the interesting
   kernels stay someone else's.
4. **Our own kernels, from a plain CPU reference up**: the most work; every step measured
   against a yardstick (cuBLAS for GEMM, ONNX Runtime / PyTorch / TensorRT end to end).

## Decision
Option 4, with the yardsticks used only to measure, never in the shipped path.

- **The weights** come from FabEye's checkpoint, with BatchNorm folded into the convolutions
  by `tools/export_cnn.py` (`W' = W·s`, `b' = (b − μ)·s + β`, `s = γ/√(σ² + ε)`) and checked
  against the weights PyTorch's ONNX exporter folded (max difference 4.8e-7). The `.wcnn` file
  carries a checksum and the SHA-256 of FabEye's deployed ONNX model, so the engine can prove
  which model it runs.
- **The reference** is a plain CPU forward pass (`ReferenceForward`), readable on purpose. It is
  checked against **ONNX Runtime's logits** on 1,000 WM-811K test maps and 500 WaferLens maps
  (max |Δlogit| 7.6e-6 and 8.6e-6, same class on every map) and must reproduce FabEye's
  macro-F1 on the test split. Every faster kernel is then checked against this reference.
- **Preprocessing** reproduces OpenCV's `INTER_NEAREST` exactly (checked against cv2 for every
  source length 1..512), not "nearest neighbour" in general: the obvious integer formula picks
  a different source pixel at some lengths WM-811K has.
- Floating-point results are compared with a stated tolerance (summation order differs between
  implementations), and the predicted class must agree on every map.

## Consequences
- Every speed-up in the GEMM ladder, fusion and int8 is measured against cuBLAS / ONNX Runtime
  and checked against the reference; a negative result (slower than cuBLAS, or int8 losing
  accuracy or conformal coverage) is reported, not hidden.
- More code to own than calling a library; the reference and the logit files make any
  regression visible immediately.
- If the engine ever needs to ship to production rather than to a portfolio, ONNX Runtime's
  CUDA provider is the fallback with the same model file.
