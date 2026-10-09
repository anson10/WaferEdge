#!/usr/bin/env python3
"""Export FabEye's CNN for WaferEdge's inference engine, with the references to check it.

    python3 tools/export_cnn.py --out data
    python3 tools/export_cnn.py --cv2-fixture tests/data/cv2_nearest_64.bin

Writes (FabEye is only read):
  data/fabeye_cnn.wcnn            weights, BatchNorm folded into the convolutions (by us, from
                                  FabEye's PyTorch checkpoint, checked against the weights the
                                  ONNX exporter folded), with the ONNX model's SHA-256
  data/fabeye_logits_wm811k.bin   ONNX Runtime logits of the first 1,000 WM-811K test maps
  data/fabeye_logits_waferlens.bin  ... of 500 WaferLens maps (every 48th, all patterns)
  --cv2-fixture FILE              cv2.resize INTER_NEAREST source indices for 64 outputs, for
                                  every source length 1..512 (the C++ preprocessing test)

Needs numpy, torch, onnx, onnxruntime, opencv-python, and data/wm811k_lot.wmap and
data/waferlens_demo.wmap (tools/export_maps.py).
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

FABEYE = Path.home() / "FabEye"
ONNX = FABEYE / "serving" / "wafer_cnn.onnx"
CHECKPOINT = FABEYE / "checkpoints" / "wm_lot_cnn_seed0.pt"
CALIBRATION = FABEYE / "serving" / "calibration.json"
BN_EPS = 1e-5  # nn.BatchNorm2d's default; WaferCNN doesn't change it
CONV_LAYERS = [(3, 32), (32, 32), (32, 64), (64, 64), (64, 128), (128, 128), (128, 256), (256, 256)]
SPLIT_TEST = 2


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def fold_checkpoint():
    """Conv + BatchNorm pairs of FabEye's checkpoint, folded into conv weights and biases."""
    import torch

    sd = {k: v.numpy().astype(np.float64) for k, v in torch.load(CHECKPOINT, map_location="cpu").items()
          if k != "num_batches_tracked" and not k.endswith("num_batches_tracked")}
    tensors = []
    # features.<i> is a Conv2d when (i % 7) in (0, 3), followed by its BatchNorm2d at i + 1.
    convs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("features.") and sd[k].ndim == 4})
    for i in convs:
        w, b = sd[f"features.{i}.weight"], sd[f"features.{i}.bias"]
        gamma, beta = sd[f"features.{i + 1}.weight"], sd[f"features.{i + 1}.bias"]
        mean, var = sd[f"features.{i + 1}.running_mean"], sd[f"features.{i + 1}.running_var"]
        s = gamma / np.sqrt(var + BN_EPS)  # BatchNorm at inference is y = s (x - mean) + beta
        tensors.append((w * s[:, None, None, None]).astype(np.float32))
        tensors.append(((b - mean) * s + beta).astype(np.float32))
    tensors.append(sd["head.1.weight"].astype(np.float32))  # Linear(256, 9): (9, 256)
    tensors.append(sd["head.1.bias"].astype(np.float32))
    return tensors


def onnx_weights():
    import onnx
    from onnx import numpy_helper

    g = onnx.load(ONNX).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    out = []
    for node in g.node:
        if node.op_type in ("Conv", "Gemm"):
            out += [init[node.input[1]], init[node.input[2]]]
    return out


def write_model(path: Path, tensors, onnx_sha256: bytes):
    payload = b"".join(
        struct.pack("<I4I", t.ndim, *(list(t.shape) + [0] * (4 - t.ndim))) + t.astype("<f4").tobytes()
        for t in tensors)
    header = b"WAFERCNN" + struct.pack("<IIQ", 1, len(tensors), fnv1a64(payload)) + onnx_sha256
    path.write_bytes(header + payload)


def read_wmap(path):
    raw = Path(path).read_bytes()
    n = int.from_bytes(raw[16:24], "little")
    rec = np.dtype([("wafer_id", "<i4"), ("lot_id", "<i4"), ("t", "<i8"), ("offset", "<u8"), ("rows", "<u2"),
                    ("cols", "<u2"), ("truth", "u1"), ("split", "u1"), ("x", "<u2")])
    records = np.frombuffer(raw, dtype=rec, count=n, offset=32)
    bins = np.frombuffer(raw, dtype=np.uint8, offset=32 + 32 * n)
    return records, [bins[r["offset"]:r["offset"] + int(r["rows"]) * int(r["cols"])].reshape(r["rows"], r["cols"])
                     for r in records]


def onnx_logits(maps):
    """FabEye's serving path: fail bins to 2, cv2 INTER_NEAREST to 64x64, one-hot, ONNX Runtime."""
    import cv2
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    x = []
    for m in maps:
        m = cv2.resize(np.minimum(m, 2).astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)
        x.append(np.stack([m == 0, m == 1, m == 2]).astype(np.float32))
    return sess.run(None, {sess.get_inputs()[0].name: np.stack(x)})[0]


def write_logits(path: Path, wafer_ids, logits):
    rec = np.zeros(len(wafer_ids), dtype=[("wafer_id", "<i4"), ("logits", "<f4", (9,))])
    rec["wafer_id"], rec["logits"] = wafer_ids, logits
    path.write_bytes(b"WCNNLOGT" + struct.pack("<I", len(rec)) + rec.tobytes())


def cv2_fixture(path: Path):
    import cv2

    rows = []
    for n in range(1, 513):
        across = cv2.resize(np.arange(n, dtype=np.uint16)[None, :], (64, 64), interpolation=cv2.INTER_NEAREST)
        down = cv2.resize(np.arange(n, dtype=np.uint16)[:, None], (64, 64), interpolation=cv2.INTER_NEAREST)
        assert (across == across[0]).all() and (down.T == across).all(), "rows and columns map alike"
        rows.append(across[0])
    path.write_bytes(np.stack(rows).astype("<u2").tobytes())
    print(f"{path}: cv2 {cv2.__version__} INTER_NEAREST indices, source lengths 1..512")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="directory for the model and the reference logits")
    ap.add_argument("--cv2-fixture", type=Path)
    args = ap.parse_args()
    if args.cv2_fixture:
        cv2_fixture(args.cv2_fixture)
    if not args.out:
        return

    onnx_sha = hashlib.sha256(ONNX.read_bytes()).digest()
    if onnx_sha.hex() != json.loads(CALIBRATION.read_text())["model_sha256"]:
        sys.exit("FabEye's ONNX model doesn't match its calibration file")
    ours, theirs = fold_checkpoint(), onnx_weights()
    assert [t.shape for t in ours] == [t.shape for t in theirs]
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ours, theirs))
    print(f"BatchNorm folded from the checkpoint vs the ONNX exporter's: max |diff| {diff:.2e}")
    assert diff < 1e-5, "our fold disagrees with the ONNX model"
    assert [(t.shape[1], t.shape[0]) for t in ours[:16:2]] == CONV_LAYERS
    args.out.mkdir(parents=True, exist_ok=True)
    write_model(args.out / "fabeye_cnn.wcnn", ours, onnx_sha)
    print(f"{args.out / 'fabeye_cnn.wcnn'}: {len(ours)} tensors, {sum(t.size for t in ours):,} parameters")

    rec, maps = read_wmap(args.out / "wm811k_lot.wmap")
    test = np.flatnonzero(rec["split"] == SPLIT_TEST)[:1000]
    write_logits(args.out / "fabeye_logits_wm811k.bin", rec["wafer_id"][test], onnx_logits([maps[i] for i in test]))
    rec, maps = read_wmap(args.out / "waferlens_demo.wmap")
    pick = np.arange(0, len(maps), 48)[:500]
    write_logits(args.out / "fabeye_logits_waferlens.bin", rec["wafer_id"][pick], onnx_logits([maps[i] for i in pick]))
    print("reference logits: 1,000 WM-811K test maps, 500 WaferLens maps (ONNX Runtime CPU)")


if __name__ == "__main__":
    main()
