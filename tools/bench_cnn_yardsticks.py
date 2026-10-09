#!/usr/bin/env python3
"""FabEye's CNN on the yardsticks, measured on this machine (docs/inference.md).

    python3 tools/bench_cnn_yardsticks.py

ONNX Runtime CPU: FabEye's deployed model as it serves it (all cores).
PyTorch GPU: FabEye's WaferCNN and checkpoint, eval mode, fp32 and fp16 (model.half()),
cuDNN autotuning on (torch.backends.cudnn.benchmark).

Inputs are already preprocessed one-hot tensors (and already on the GPU for PyTorch): the
model's time only, the best case for the yardsticks. WaferEdge's numbers (bench-cnn) include
uploading the raw bins, preprocessing on the GPU and downloading the logits. Time on
time.perf_counter (monotonic), median of 5 runs of ~1 s each. FabEye is only read.
"""

import statistics
import sys
import time
from pathlib import Path

import numpy as np

FABEYE = Path.home() / "FabEye"
BATCHES = [1, 16, 64, 256, 1024]


def timed(fn, items):
    fn()  # warm-up
    runs = []
    for _ in range(5):
        n = 0
        start = time.perf_counter()
        while time.perf_counter() - start < 1.0:
            fn()
            n += 1
        runs.append(items * n / (time.perf_counter() - start))
    return statistics.median(runs)


def onehot(batch):
    rng = np.random.default_rng(1)
    m = rng.choice([0, 1, 2], size=(batch, 64, 64), p=[0.2, 0.72, 0.08])
    return np.stack([m == 0, m == 1, m == 2], axis=1).astype(np.float32)


def main():
    import onnxruntime as ort
    import torch

    sys.path.insert(0, str(FABEYE))
    from models.wm_models import WaferCNN

    sess = ort.InferenceSession(str(FABEYE / "serving" / "wafer_cnn.onnx"), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    model = WaferCNN()
    model.load_state_dict(torch.load(FABEYE / "checkpoints" / "wm_lot_cnn_seed0.pt", map_location="cpu"))
    model.eval().cuda()
    half = WaferCNN()
    half.load_state_dict(model.state_dict())
    half.eval().cuda().half()
    torch.backends.cudnn.benchmark = True

    print(f"onnxruntime {ort.__version__}, torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
    print("| Batch | ONNX Runtime CPU | PyTorch GPU fp32 | PyTorch GPU fp16 |")
    print("|---|---|---|---|")
    for b in BATCHES:
        x = onehot(b)
        ort_rate = timed(lambda: sess.run(None, {name: x}), b) if b <= 256 else float("nan")
        xg = torch.from_numpy(x).cuda()
        xh = xg.half()

        def torch_run(m, inp):
            with torch.no_grad():
                m(inp)
            torch.cuda.synchronize()

        fp32 = timed(lambda: torch_run(model, xg), b)
        fp16 = timed(lambda: torch_run(half, xh), b)
        print(f"| {b} | {ort_rate:,.0f} | {fp32:,.0f} | {fp16:,.0f} |")


if __name__ == "__main__":
    main()
