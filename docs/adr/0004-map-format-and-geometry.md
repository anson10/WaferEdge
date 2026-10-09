# ADR-0004: One .wmap file format, and exact integer die geometry

- **Status:** Accepted
- **Date:** 2026-10-09

## Context
Phase 1 reads two datasets: WaferLens's simulated maps (Parquet, `list<list<int16>>`, 24,090
maps in three shapes, ground truth in two other tables) and WM-811K (a pickled pandas frame
in FabEye, 79,608 labelled maps in 341 shapes with FabEye's lot-disjoint split). Later the
tool emulator replays maps, the GPU path batches them and the Python wheel exposes them. The
C++ side needs them fast to load, with no per-map allocation, and identical across sources.

The features (fail density by radial zone and angular sector) need each die's zone and
sector. Three backends (scalar, AVX2, CUDA) will compute them and must agree; with `sqrt`
and `atan2` in floating point, dies on a zone or sector boundary can land on different sides
in different backends (fused multiply-add, different `atan2` implementations), and "equal"
would become "equal within a tolerance, except at boundaries".

## Options considered
For loading:
1. **Arrow / Parquet C++**: reads WaferLens directly, but is a large dependency (and doesn't
   read WM-811K's pickle), and nested lists come back as Arrow arrays to convert anyway.
2. **A small binary format written by a Python exporter**: one file per dataset; Python does
   the source-specific work (joins for truth, lot ids, splits), C++ reads a flat layout.
3. **CSV / NPY per map**: simple, but many files or a slow text parse.

For geometry:
1. Floating-point radius and `atan2` per die per map.
2. **Exact integer zone/sector per grid position, tabled once per shape.**

## Decision
Loading: option 2. `tools/export_maps.py` writes `.wmap` files (layout in
`include/waferedge/map_file.hpp`): a 32-byte header, a fixed 32-byte record per map (ids,
`tested_at`, truth, split, shape, offset) and one byte arena of bins. `MapSet::load` reads the
whole file into one buffer and hands out `WaferMapView`s into it: one allocation per file,
none per map. Records are validated against the file size with overflow-safe checks, so a
corrupt file is an error, never an out-of-bounds read. WaferLens maps are written in
`tested_at` order (replay order); WaferLens has no split (it is an evaluation set), WM-811K
keeps FabEye's train / val / test.

Geometry: option 2. With doubled coordinates `X = 2c - (cols-1)`, `Y = (rows-1) - 2r` and
per-axis normalisation (WM-811K dies aren't square, so the wafer is an ellipse in die units),
the zone is `floor(K * (X^2 rows^2 + Y^2 cols^2) / (rows^2 cols^2))` in 64-bit integers, and
the octant comes from signs and one comparison of `X * rows` against `Y * cols`. Zones are
equal-area rings (K = 5), sectors 45-degree octants, half-open counter-clockwise. A
`Geometry` per shape holds both as byte tables; `compute_features` is then integer counting
over bins, zone and sector tables, and returns counts only. Densities and ratios are derived
from counts in one function each.

## Consequences
- Backends can be compared with `==` on the `Features` struct, bit for bit.
- Rotating a square map by 90 degrees provably keeps zones and moves sectors by two; the
  tests check it for every die rather than "approximately".
- Features are a gather plus increments: easy to vectorise (AVX2 histogramming) and to map to
  shared-memory atomics on the GPU. Tables cost 2 bytes per grid position per shape (341
  WM-811K shapes: a few hundred kB).
- A new source needs only an exporter function; the C++ side doesn't change.
- The exported files are build inputs, not repository content (`/data/` is ignored); a
  24-map WaferLens fixture is checked in for tests. If the format changes, the version field
  goes up and old files are rejected with a clear message.
