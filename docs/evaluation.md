# Evaluation: rule classifier vs FabEye's CNN vs a trivial baseline

The phase-1 question: how well do explainable, classical detectors classify wafer-map
patterns, next to FabEye's CNN, on the same maps? Short answer: clearly worse than the CNN
(macro-F1 0.655 vs 0.858 on real maps), far better than a trivial baseline (0.118), strong on
a few patterns and weak on others. The reasons are below; ADR-0005 has the design.

## Setup

- **Data** (docs/data.md): WM-811K with FabEye's lot-disjoint split (no lot in two splits),
  and the 24,090 simulated WaferLens maps with injected-pattern ground truth.
- **Rules**: thresholds fitted on WM-811K **train only** (27,830 maps), 'none' weighted back to
  its uncapped count (ADR-0005). WaferLens is never used for fitting.
- **CNN**: FabEye's deployed model (`~/FabEye/serving/wafer_cnn.onnx`, hash checked against its
  calibration file) on ONNX Runtime, with FabEye's preprocessing (fail bins mapped to 2,
  `cv2.resize` INTER_NEAREST to 64×64, one-hot). It also trained on WM-811K train.
- **Trivial**: 'none' below a fail-density threshold, otherwise the most common defect class
  of WM-811K train (edge_ring); threshold fitted on train with the same 'none' weight.
- **Metric**: per-class F1, macro-F1 = mean F1 over the classes present in the truth (all 9 on
  WM-811K, 8 on WaferLens, which has no near_full), and accuracy. One implementation scores
  all three methods (`tools/evaluate.py`).
- **Sanity check**: the script reproduces FabEye's published numbers (0.858 on WM-811K test,
  0.909 on WaferLens), so data, preprocessing and metric agree with FabEye's evaluation.

## Results

### WM-811K test, lot-disjoint (25,875 maps)

| Class | Support | rules | cnn (FabEye) | trivial |
|---|---|---|---|---|
| none | 22,082 | 0.974 | 0.983 | 0.905 |
| center | 643 | 0.690 | 0.888 | 0.000 |
| donut | 87 | 0.620 | 0.899 | 0.000 |
| edge_loc | 808 | 0.401 | 0.804 | 0.000 |
| edge_ring | 1,424 | 0.885 | 0.978 | 0.156 |
| loc | 510 | 0.395 | 0.750 | 0.000 |
| near_full | 24 | 0.889 | 0.906 | 0.000 |
| random | 123 | 0.704 | 0.807 | 0.000 |
| scratch | 174 | 0.340 | 0.706 | 0.000 |
| **macro-F1** | | **0.655** | **0.858** | **0.118** |
| accuracy | | 0.927 | 0.965 | 0.791 |

Rules on the other WM-811K splits: train 0.698, validation 0.672 (the fit is not badly
overfitted: 0.698 → 0.672 → 0.655).

### WaferLens demo, simulated (24,090 maps; no near_full in the truth)

| Class | Support | rules | cnn (FabEye) | trivial |
|---|---|---|---|---|
| none | 22,199 | 0.998 | 0.996 | 0.949 |
| center | 215 | 0.986 | 0.993 | 0.000 |
| donut | 367 | 0.543 | 1.000 | 0.000 |
| edge_loc | 634 | 0.737 | 0.991 | 0.000 |
| edge_ring | 164 | 0.910 | 1.000 | 0.107 |
| loc | 133 | 0.382 | 0.810 | 0.000 |
| random | 315 | 0.248 | 0.637 | 0.000 |
| scratch | 63 | 0.360 | 0.844 | 0.000 |
| **macro-F1** | | **0.645** | **0.909** | **0.132** |
| accuracy | | 0.964 | 0.988 | 0.861 |

## Where the rules hold and where they lose

**Hold**: none (0.974 real, 0.998 simulated; slightly above the CNN on WaferLens), edge_ring
(0.885 / 0.910), center on simulated maps (0.986), near_full (0.889). These are the patterns
defined by one radial or density signal, which is what the zones and rings measure.

**Lose**:
- **edge_loc, loc, scratch** (0.34–0.40 real). They overlap in signal space: an edge-loc blob is
  a loc blob near the edge, a short scratch is an elongated loc, and real 'none' maps carry
  edge fails (median edge ratio 2.0). A threshold per signal can't draw those boundaries;
  the CNN sees the shape directly.
- **donut on WaferLens** (0.543): 227 of 367 donuts are called loc. Every one of them passes
  the donut rule's hole condition (inner ≤ 1.83) and fails its ring condition (ring ≥ 2.17,
  theirs ≈ 1.8–2.1). The ratio signals have a ceiling: a zone's fail density over the wafer's
  can't exceed 1 / (wafer fail density). The simulated donut is thick (median wafer fail
  density 0.46, ceiling ≈ 2.2), so the threshold learned from WM-811K's small donuts sits right
  at the ceiling. A signal that compares the ring with the hole (ring − inner density) instead
  of with the whole wafer wouldn't saturate; that is a candidate change, to be judged on
  validation.
- **random on WaferLens** (0.248): 266 of 315 are called near_full. The simulated random field
  is dense (median fail density 0.72, docs/signatures.md), above the near_full threshold
  learned from real maps. The CNN makes the same mistake for the same reason (0.637), as
  FabEye's own report noted: it's the simulator's random pattern, not a detector bug.

**A miss, explained.** `waferedge-classify explain data/waferlens_demo.wmap config/rules.txt 10570`:

```
wafer 10570 (lot 423, 24x24, truth edge_ring)
  edge 2.812   sector 1.185   z 4.185   cluster_size 0.2407   cluster_rho 0.01964 ...
loc -> loc: cluster_size 0.241 >= 0.00557, cluster_rho 0.0196 <= 0.831, z 4.18 >= 3.52
```

Its edge ratio (2.81 ≥ 2.34) and even spread (sector 1.19 ≤ 1.59) pass the edge_ring rule, but
its join-count z (4.18) is under that rule's fitted 5.39, so it falls through to loc. That
one line is the point of the rules: a wrong call can be read, argued with and fixed.

## Speed

Signals plus rules, one core, `release` preset (Ryzen 5 7535HS, WSL2): 12.9k maps/s on
WM-811K test, 14.2k maps/s on WaferLens (`waferedge-classify eval` prints it). The Hough
transform is most of that cost (docs/signatures.md). For scale, FabEye's README gives ~412
wafers/s for the CNN on ONNX Runtime CPU at batch 32; phase 2b measures the CNN on WaferEdge's
own GPU engine on this machine.

## Reproduce

```sh
cmake --workflow --preset release
python3 tools/export_maps.py waferlens ~/waferLens/data/demo data/waferlens_demo.wmap
python3 tools/export_maps.py wm811k ~/FabEye/data/wm811k/processed_lot.pkl data/wm811k_lot.wmap

# fit (writes config/rules.txt, checked in), then score each split
build/release/tools/waferedge-classify fit data/wm811k_lot.wmap config/rules.txt --none-weight 10.3342
build/release/tools/waferedge-classify eval data/wm811k_lot.wmap config/rules.txt test

# the comparison tables above
build/release/tools/waferedge-classify eval data/wm811k_lot.wmap config/rules.txt all --csv data/rules_wm811k.csv
build/release/tools/waferedge-classify eval data/waferlens_demo.wmap config/rules.txt all --csv data/rules_waferlens.csv
python3 tools/evaluate.py data/wm811k_lot.wmap data/rules_wm811k.csv data/waferlens_demo.wmap data/rules_waferlens.csv
```

`evaluate.py` needs numpy, opencv-python and onnxruntime, and reads FabEye's model from
`~/FabEye/serving` (never modified).
