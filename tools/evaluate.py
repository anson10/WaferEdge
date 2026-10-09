#!/usr/bin/env python3
"""Score the rule classifier next to FabEye's CNN and a trivial baseline, on the same maps.

    build/release/tools/waferedge-classify eval data/wm811k_lot.wmap config/rules.txt all \\
        --csv data/rules_wm811k.csv
    build/release/tools/waferedge-classify eval data/waferlens_demo.wmap config/rules.txt all \\
        --csv data/rules_waferlens.csv
    python3 tools/evaluate.py data/wm811k_lot.wmap data/rules_wm811k.csv \\
        data/waferlens_demo.wmap data/rules_waferlens.csv

Methods:
  rules    WaferEdge's rule classifier, predictions read from the CSVs above.
  cnn      FabEye's CNN (~/FabEye/serving/wafer_cnn.onnx) on ONNX Runtime, FabEye's own
           preprocessing (fail bins >= 2 mapped to 2, cv2 INTER_NEAREST to 64x64, one-hot).
  trivial  "none" below a fail-density threshold, otherwise the most common defect class of
           WM-811K train; threshold fitted on WM-811K train with the same 'none' weight.

Prints markdown tables: per-class F1 and macro-F1 on WM-811K test (lot-disjoint) and on every
WaferLens map. Macro-F1 averages over the classes present in the truth. Needs numpy,
opencv-python and onnxruntime. FabEye is only read.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PATTERNS = ["none", "center", "donut", "edge_loc", "edge_ring", "loc", "near_full", "random", "scratch"]
SPLITS = ["train", "val", "test", "unsplit"]
# WM-811K's 147,431 labelled 'none' maps minus the 44,089 in val + test: the uncapped train
# count, over the 10,000 FabEye kept (ADR-0005). The same weight fits config/rules.txt.
NONE_WEIGHT = 103342 / 10000
FABEYE = Path.home() / "FabEye" / "serving"


def read_wmap(path):
    raw = Path(path).read_bytes()
    assert raw[:8] == b"WAFERMAP", f"{path}: not a .wmap file"
    n = int.from_bytes(raw[16:24], "little")
    rec = np.dtype([("wafer_id", "<i4"), ("lot_id", "<i4"), ("tested_at_us", "<i8"),
                    ("offset", "<u8"), ("rows", "<u2"), ("cols", "<u2"), ("truth", "u1"),
                    ("split", "u1"), ("reserved", "<u2")])
    records = np.frombuffer(raw, dtype=rec, count=n, offset=32)
    bins = np.frombuffer(raw, dtype=np.uint8, offset=32 + 32 * n)
    maps = [bins[r["offset"]:r["offset"] + int(r["rows"]) * int(r["cols"])].reshape(r["rows"], r["cols"])
            for r in records]
    return records, maps


def cnn_predict(maps, batch=256):
    import cv2
    import onnxruntime as ort

    model = FABEYE / "wafer_cnn.onnx"
    cal = json.loads((FABEYE / "calibration.json").read_text())
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    assert digest == cal["model_sha256"], "FabEye's ONNX model doesn't match its calibration"
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = []
    for i in range(0, len(maps), batch):
        x = []
        for m in maps[i:i + batch]:
            m = np.minimum(m, 2).astype(np.uint8)  # WaferLens fail bins 2..5 -> FabEye's 2
            m = cv2.resize(m, (64, 64), interpolation=cv2.INTER_NEAREST)
            x.append(np.stack([m == 0, m == 1, m == 2]).astype(np.float32))
        logits = sess.run(None, {name: np.stack(x)})[0]
        out.append(logits.argmax(1))
    return np.concatenate(out)


def density(m):
    dies = (m > 0).sum()
    return (m >= 2).sum() / dies if dies else 0.0


def weighted_macro_f1(truth, pred, weight):
    f1s = []
    for c in range(len(PATTERNS)):
        w_true = weight[truth == c]
        if w_true.sum() == 0:
            continue
        tp = weight[(truth == c) & (pred == c)].sum()
        p = tp / weight[pred == c].sum() if (pred == c).any() else 0.0
        r = tp / w_true.sum()
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def fit_trivial(dens, truth):
    """Threshold on fail density; above it, the most common defect class."""
    defect = int(np.bincount(truth[truth != 0], minlength=len(PATTERNS)).argmax())
    weight = np.where(truth == 0, NONE_WEIGHT, 1.0)
    best = (-1.0, 0.0)
    for t in np.quantile(dens, np.linspace(0, 1, 201)):
        pred = np.where(dens >= t, defect, 0)
        best = max(best, (weighted_macro_f1(truth, pred, weight), float(t)))
    return best[1], defect


def scores(truth, pred):
    present = [c for c in range(len(PATTERNS)) if (truth == c).any()]
    f1 = {}
    for c in present:
        tp = ((truth == c) & (pred == c)).sum()
        p = tp / (pred == c).sum() if (pred == c).any() else 0.0
        r = tp / (truth == c).sum()
        f1[c] = 2 * p * r / (p + r) if p + r else 0.0
    return f1, float(np.mean(list(f1.values()))), float((truth == pred).mean())


def read_rules(csv_path, records):
    by_id = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            by_id[int(row["wafer_id"])] = PATTERNS.index(row["pred"])
    return np.array([by_id[int(w)] for w in records["wafer_id"]])


def table(title, truth, preds):
    print(f"\n### {title}\n")
    results = {name: scores(truth, p) for name, p in preds.items()}
    present = list(next(iter(results.values()))[0])
    print("| Class | Support | " + " | ".join(preds) + " |")
    print("|---|---|" + "---|" * len(preds))
    for c in present:
        cells = " | ".join(f"{results[n][0][c]:.3f}" for n in preds)
        print(f"| {PATTERNS[c]} | {(truth == c).sum():,} | {cells} |")
    print("| **macro-F1** | | " + " | ".join(f"**{results[n][1]:.3f}**" for n in preds) + " |")
    print("| accuracy | | " + " | ".join(f"{results[n][2]:.3f}" for n in preds) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wm811k_wmap")
    ap.add_argument("wm811k_rules_csv")
    ap.add_argument("waferlens_wmap")
    ap.add_argument("waferlens_rules_csv")
    args = ap.parse_args()

    wm_rec, wm_maps = read_wmap(args.wm811k_wmap)
    train = wm_rec["split"] == SPLITS.index("train")
    test = wm_rec["split"] == SPLITS.index("test")
    wm_dens = np.array([density(m) for m in wm_maps])
    t, defect = fit_trivial(wm_dens[train], wm_rec["truth"][train].astype(int))
    print(f"trivial baseline: fail density >= {t:.3f} -> {PATTERNS[defect]}, else none "
          "(fitted on WM-811K train)")

    test_idx = np.flatnonzero(test)
    truth = wm_rec["truth"][test].astype(int)
    preds = {
        "rules": read_rules(args.wm811k_rules_csv, wm_rec[test]),
        "cnn (FabEye)": cnn_predict([wm_maps[i] for i in test_idx]),
        "trivial": np.where(wm_dens[test] >= t, defect, 0),
    }
    table(f"WM-811K test, lot-disjoint ({len(truth):,} maps)", truth, preds)

    wl_rec, wl_maps = read_wmap(args.waferlens_wmap)
    truth = wl_rec["truth"].astype(int)
    wl_dens = np.array([density(m) for m in wl_maps])
    preds = {
        "rules": read_rules(args.waferlens_rules_csv, wl_rec),
        "cnn (FabEye)": cnn_predict(wl_maps),
        "trivial": np.where(wl_dens >= t, defect, 0),
    }
    table(f"WaferLens demo, simulated ({len(truth):,} maps; no near_full in the truth)", truth, preds)


if __name__ == "__main__":
    main()
