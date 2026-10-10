#!/usr/bin/env python3
"""Export wafer maps with ground truth to a .wmap file (layout: include/waferedge/map_file.hpp).

    export_maps.py waferlens ~/waferLens/data/demo data/waferlens_demo.wmap
    export_maps.py wm811k ~/FabEye/data/wm811k/processed_lot.pkl data/wm811k_lot.wmap

WaferLens: every sorted wafer, in tested_at order (the order a tool emulator replays them),
truth from wafer_pattern_truth + excursions_ground_truth (wafers not listed: none), split
"unsplit". WM-811K: FabEye's labelled, lot-disjoint maps with its train/val/test split; lot
names become integer ids in sorted order, wafer_id is the row index, tested_at is unknown (0).

--per-pattern N keeps the first N wafers of each truth class (a small fixture for tests).
--excursions FILE (WaferLens only) also writes the spatial excursions and which wafer shows
which, for the closed-loop evaluation (docs/pipeline.md):
    excursion,<id>,<pattern>,<start_us>,<end_us>     one per spatial-pattern excursion
    wafer,<wafer_id>,<excursion_id>                  one per wafer showing a pattern
Needs numpy and pyarrow; the sources are only read, never modified.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PATTERNS = ["none", "center", "donut", "edge_loc", "edge_ring", "loc", "near_full", "random", "scratch"]
FABEYE_CLASSES = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
SPLITS = {0: 0, 1: 1, 2: 2}  # FabEye's train / val / test ids are ours
UNSPLIT = 3

HEADER = np.dtype([("magic", "S8"), ("version", "<u4"), ("record_size", "<u4"),
                   ("count", "<u8"), ("bin_bytes", "<u8")])
RECORD = np.dtype([("wafer_id", "<i4"), ("lot_id", "<i4"), ("tested_at_us", "<i8"),
                   ("offset", "<u8"), ("rows", "<u2"), ("cols", "<u2"), ("truth", "u1"),
                   ("split", "u1"), ("reserved", "<u2")])
assert HEADER.itemsize == 32 and RECORD.itemsize == 32


def write_wmap(path, wafer_ids, lot_ids, tested_at_us, truths, splits, maps):
    records = np.zeros(len(maps), dtype=RECORD)
    offset = 0
    for i, m in enumerate(maps):
        if m.ndim != 2 or m.min() < 0 or m.max() > 255:
            sys.exit(f"map {i}: expected a 2-D grid of bin codes 0..255")
        records[i] = (wafer_ids[i], lot_ids[i], tested_at_us[i], offset, m.shape[0], m.shape[1],
                      truths[i], splits[i], 0)
        offset += m.size
    header = np.array([(b"WAFERMAP", 1, RECORD.itemsize, len(maps), offset)], dtype=HEADER)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(records.tobytes())
        for m in maps:
            f.write(np.ascontiguousarray(m, dtype=np.uint8).tobytes())
    return offset


def from_waferlens(data_dir):
    import pyarrow.parquet as pq

    maps_t = pq.read_table(data_dir / "wafer_maps.parquet").sort_by("tested_at")
    wafers = pq.read_table(data_dir / "wafers.parquet", columns=["wafer_id", "lot_id"]).to_pydict()
    lot_of = dict(zip(wafers["wafer_id"], wafers["lot_id"]))
    exc = pq.read_table(data_dir / "excursions_ground_truth.parquet",
                        columns=["excursion_id", "spatial_pattern"]).to_pydict()
    pattern_of_exc = dict(zip(exc["excursion_id"], exc["spatial_pattern"]))
    truth_t = pq.read_table(data_dir / "wafer_pattern_truth.parquet").to_pydict()
    truth_of = {}
    for wafer, excursion in zip(truth_t["wafer_id"], truth_t["excursion_id"]):
        name = pattern_of_exc[excursion]
        if truth_of.get(wafer, name) != name:
            sys.exit(f"wafer {wafer} carries two different patterns")
        truth_of[wafer] = name

    wafer_ids = maps_t.column("wafer_id").to_pylist()
    tested = maps_t.column("tested_at").cast("int64").to_pylist()  # us since epoch
    maps = [np.array(m, dtype=np.int16) for m in maps_t.column("bin_map").to_pylist()]
    truths = [PATTERNS.index(truth_of.get(w, "none")) for w in wafer_ids]
    lots = [lot_of[w] for w in wafer_ids]
    return wafer_ids, lots, tested, truths, [UNSPLIT] * len(maps), maps


def write_excursions(data_dir, path):
    """The spatial excursions (id, pattern, start and end in us since the epoch) and the
    excursion each patterned wafer shows, from WaferLens's ground-truth tables."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    exc = pq.read_table(data_dir / "excursions_ground_truth.parquet")
    exc = exc.filter(pc.equal(exc.column("excursion_type"), "spatial_pattern"))
    to_us = lambda col: exc.column(col).cast(pa.timestamp("us")).cast("int64").to_pylist()
    ids = exc.column("excursion_id").to_pylist()
    patterns = exc.column("spatial_pattern").to_pylist()
    starts, ends = to_us("start_time"), to_us("end_time")
    truth = pq.read_table(data_dir / "wafer_pattern_truth.parquet").to_pydict()
    with open(path, "w") as f:
        f.write("# kind,id,...: excursion,id,pattern,start_us,end_us | wafer,wafer_id,excursion_id\n")
        for i, p, a, b in sorted(zip(ids, patterns, starts, ends)):
            f.write(f"excursion,{i},{p},{a},{b}\n")
        for w, e in sorted(zip(truth["wafer_id"], truth["excursion_id"])):
            f.write(f"wafer,{w},{e}\n")
    print(f"{path}: {len(ids)} spatial excursions, {len(truth['wafer_id'])} patterned wafers")


def from_wm811k(pkl):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    if list(d["classes"]) != FABEYE_CLASSES:
        sys.exit(f"unexpected class order {d['classes']}")
    lot_names = sorted(set(d["lots"]))
    lot_id = {name: i for i, name in enumerate(lot_names)}
    n = len(d["maps"])
    return (list(range(n)), [lot_id[x] for x in d["lots"]], [0] * n,
            [int(y) for y in d["y"]], [SPLITS[int(s)] for s in d["split"]], d["maps"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["waferlens", "wm811k"])
    ap.add_argument("input", type=Path, help="WaferLens data directory, or FabEye's processed_lot.pkl")
    ap.add_argument("output", type=Path)
    ap.add_argument("--per-pattern", type=int, default=0, metavar="N",
                    help="keep only the first N wafers of each truth class")
    ap.add_argument("--excursions", type=Path, metavar="FILE",
                    help="WaferLens: also write the spatial excursions and their wafers (CSV)")
    args = ap.parse_args()
    if args.excursions and args.source != "waferlens":
        ap.error("--excursions needs the waferlens source")

    load = from_waferlens if args.source == "waferlens" else from_wm811k
    cols = load(args.input.expanduser())
    if args.per_pattern:
        seen, keep = {}, []
        for i, t in enumerate(cols[3]):
            if seen.get(t, 0) < args.per_pattern:
                seen[t] = seen.get(t, 0) + 1
                keep.append(i)
        cols = tuple([c[i] for i in keep] for c in cols)

    bin_bytes = write_wmap(args.output, *cols)
    counts = np.bincount(np.array(cols[3]), minlength=len(PATTERNS))
    print(f"{args.output}: {len(cols[5])} maps, {bin_bytes} dies")
    print("  " + ", ".join(f"{p} {c}" for p, c in zip(PATTERNS, counts) if c))
    if args.excursions:
        write_excursions(args.input.expanduser(), args.excursions)


if __name__ == "__main__":
    main()
