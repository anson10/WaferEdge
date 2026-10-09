# Data

WaferEdge reads wafer maps from `.wmap` files (format: `include/waferedge/map_file.hpp`,
ADR-0004). They are exported from the sister projects, which are only read, and land in
`data/` (git-ignored).

## Export

Needs Python with `numpy` and `pyarrow`.

```sh
# WaferLens simulated fab (run `make simulate` in ~/waferLens first): 24,090 maps
python3 tools/export_maps.py waferlens ~/waferLens/data/demo data/waferlens_demo.wmap

# WM-811K, FabEye's labelled lot-disjoint split: 79,608 maps
python3 tools/export_maps.py wm811k ~/FabEye/data/wm811k/processed_lot.pkl data/wm811k_lot.wmap

# Check a file: counts by truth and split, shapes, and scalar feature throughput
build/release/tools/waferedge-maps data/waferlens_demo.wmap
```

The `dev` (1,000 wafers) and `stress` (125,000) WaferLens profiles export the same way from
`~/waferLens/data/dev` and `~/waferLens/data/stress`.

## What the files hold

| | WaferLens `demo` | WM-811K (FabEye lot split) |
|---|---|---|
| Maps | 24,090 | 79,608 |
| Shapes | 3 (24×24, 30×30, 40×40) | 341 (most common 25×27) |
| Bins | 0 off, 1 pass, 2–5 fail bins | 0 off, 1 pass, 2 fail |
| Truth | injected pattern per wafer (`wafer_pattern_truth` → excursion's `spatial_pattern`); other wafers `none` | FabEye's 9 classes |
| Split | `unsplit` (an evaluation set) | train / val / test, lot-disjoint; train `none` capped at 10,000 |
| Order | `tested_at` (the order a tool emulator replays) | FabEye's row order |
| Ids | `wafer_id`, `lot_id` from WaferLens | row index; lot names as sorted integer ids |

Truth by class:

| Truth | WaferLens | WM-811K train / val / test |
|---|---|---|
| none | 22,199 | 10,000 / 22,007 / 22,082 |
| center | 215 | 3,075 / 576 / 643 |
| donut | 367 | 381 / 87 / 87 |
| edge_loc | 634 | 3,616 / 765 / 808 |
| edge_ring | 164 | 6,704 / 1,552 / 1,424 |
| loc | 133 | 2,485 / 598 / 510 |
| near_full | 0 | 103 / 22 / 24 |
| random | 315 | 617 / 126 / 123 |
| scratch | 63 | 849 / 170 / 174 |

The test fixture `tests/data/waferlens_sample.wmap` is the first three wafers of each
WaferLens class (`--per-pattern 3`), 23 KB.
