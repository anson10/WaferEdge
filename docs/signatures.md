# Spatial signatures

The classical detectors, scalar reference versions. Each produces integers (counts, moments,
votes), so the AVX2 and CUDA versions in later phases are tested with `==` against these.
The classifier that turns them into a pattern comes next (phase 1, `feat/rule-classifier`).

| Detector | Header | Output | What it is for |
|---|---|---|---|
| Zone / sector features | `features.hpp` | dies and fails per radial zone (5 equal-area rings) and angular sector (8 octants) | center, donut, edge-ring (radial profile); edge-loc, loc (one-sided) |
| Clusters | `clusters.hpp` | 8-connected fail clusters with integer moments; shape: centroid, elongation, angle | blobs (loc, center) vs lines (scratch); share of fails in the largest cluster |
| Hough transform | `hough.hpp` | the strongest straight line: angle, offset, fail votes, dies on the line | broken scratches that clustering splits into pieces |
| Join count | `randomness.hpp` | fail-fail neighbour pairs against a random-placement null; z score | random fields (z ≈ 0) vs any spatial structure (z ≫ 0) |

## Design notes

**Clusters.** Union-find in two passes over the grid. Every union hangs the larger root under
the smaller, so a cluster's root is its first die in row-major order and clusters come out
numbered by that die: the labelling is a function of the map alone, and a GPU labelling (which
merges in a different order) can be compared label for label. The test compares against an
independent breadth-first flood fill on random maps. Shape uses dies as unit squares
(variance 1/12 per axis) so elongation, `sqrt(major / minor variance)`, is exactly `L` for a
straight line of `L` dies and 1 for a single die or a square.

**Hough transform.** 180 angles at 1°, offsets in die-width bins. cos and sin are Q14
fixed-point table entries and die centres are the doubled integer coordinates of the
geometry (ADR-0004), so every vote is integer arithmetic plus a floor shift. A short line
votes equally for a small fan of angles around its true normal; ties go to the lowest angle,
so tests allow ±2°. `line_dies` counts the on-wafer dies in the winning bin, so
`votes / line_dies` against the wafer's fail density says how much denser than the wafer
the line is.

**Join count.** Under the null, the map's `F` fails are a random draw without replacement from
its `N` on-wafer dies (the wafer's own yield, no spatial structure). The mean and variance of
the number of fail-fail joins have a closed form (in `randomness.hpp`); the test checks it
against exact enumeration of every placement on two small wafers, and checks that z over 200
random fields has mean ≈ 0 and standard deviation ≈ 1.

## What separates the patterns

Median of each signal per truth class (`build/release/tools/waferedge-maps <file>`):
fail density; center and edge zone ratio (zone fail density over the wafer's; 1 = even);
densest sector ratio; the largest cluster's share of all fails and its elongation; the Hough
line's fail density over the wafer's; join-count z.

WaferLens demo (simulated, 24,090 maps):

| Truth | Density | Center | Edge | Sector | Cluster | Elong. | Line | z |
|---|---|---|---|---|---|---|---|---|
| none | 0.098 | 0.91 | 1.43 | 1.44 | 0.07 | 2.1 | 3.0 | -0.1 |
| center | 0.303 | 2.53 | 0.41 | 1.16 | 0.76 | 1.1 | 1.8 | 15.9 |
| donut | 0.463 | 0.77 | 0.28 | 1.10 | 0.93 | 1.0 | 1.5 | 20.3 |
| edge_loc | 0.157 | 0.57 | 1.92 | 2.56 | 0.46 | 3.8 | 5.5 | 9.8 |
| edge_ring | 0.334 | 0.35 | 2.84 | 1.15 | 0.80 | 1.0 | 2.6 | 10.2 |
| loc | 0.197 | 1.79 | 0.80 | 2.29 | 0.54 | 1.2 | 2.4 | 12.6 |
| random | 0.723 | 1.00 | 1.02 | 1.08 | 1.00 | 1.0 | 1.1 | -0.2 |
| scratch | 0.125 | 1.37 | 1.04 | 2.16 | 0.35 | 5.8 | 4.6 | 4.8 |

WM-811K (real, FabEye's lot split, all splits, 79,608 maps):

| Truth | Density | Center | Edge | Sector | Cluster | Elong. | Line | z |
|---|---|---|---|---|---|---|---|---|
| none | 0.098 | 0.80 | 2.00 | 1.48 | 0.08 | 2.2 | 3.0 | 0.7 |
| center | 0.250 | 1.52 | 1.56 | 1.31 | 0.30 | 1.4 | 2.2 | 3.8 |
| donut | 0.258 | 1.81 | 0.86 | 1.50 | 0.66 | 1.4 | 2.1 | 18.8 |
| edge_loc | 0.156 | 0.60 | 2.18 | 1.86 | 0.27 | 2.8 | 3.8 | 6.6 |
| edge_ring | 0.140 | 0.37 | 3.62 | 1.21 | 0.71 | 1.1 | 6.1 | 12.1 |
| loc | 0.142 | 0.87 | 1.50 | 2.00 | 0.29 | 1.7 | 3.2 | 6.7 |
| near_full | 0.878 | 1.01 | 1.00 | 1.09 | 1.00 | 1.1 | 1.1 | 3.8 |
| random | 0.477 | 1.03 | 1.16 | 1.22 | 0.86 | 1.2 | 1.3 | 2.0 |
| scratch | 0.092 | 0.79 | 1.80 | 1.81 | 0.20 | 5.1 | 4.7 | 4.7 |

What this says before any classifier is fitted:
- On the simulated maps each injected pattern moves the signal it was designed for: center
  ratio for center, edge ratio for edge-ring, sector ratio for edge-loc and loc, elongation
  and line for scratch, z ≈ 0 with high density for random.
- WM-811K is harder. Real donuts are small: their ring falls partly inside zone 0, so the
  donut's center ratio (1.81) is *higher* than the center class's (1.52); radial zones alone
  won't separate them, cluster shape and z will have to. WM-811K's `none` maps carry many
  edge fails (edge ratio 2.0), which blurs edge-loc and edge-ring against none.
- Edge-ring maps have a high line ratio (6.1): a line tangent to the ring collects many fail
  dies. The classifier must look at the edge ratio before calling a line a scratch.
- Random and near-full differ mostly by density (0.48 vs 0.88); both have small z.
- WaferLens's "random" excursion is dense (median density 0.72), closer to WM-811K's
  near-full than to its random: one reason FabEye's CNN calls strong random fields near-full.

## Throughput (scalar, release preset, Ryzen 5 7535HS, WSL2)

Each stage alone, whole passes over the file, `waferedge-maps`:

| Stage | WaferLens (maps/s) | WM-811K (maps/s) |
|---|---|---|
| Zone / sector features | 401k | 274k |
| Clusters | 146k | 106k |
| Hough (180 angles) | 19k | 13k |
| Join count | 196k | 148k |

The Hough transform costs ~10× the rest together (every fail die votes 180 times): the first
candidate for the GPU (phase 2a, kernel 3). `build/release/bench/bench-features` gives the
same stages on synthetic maps by size (24² to 64²).
