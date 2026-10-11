# WaferEdge from Python

`waferedge` is WaferEdge's C++23 wafer-map code on NumPy arrays: the die and fail counts by
zone, sector and ring, the 13 signals and the rule classifier (ADR-0005), the connected
fail clusters, and the Hough line of a scratch. The edge pipeline, the protocol stack and the
GPU paths stay in C++; this module is for analysis in notebooks and scripts.

## Install

From a checkout (needs CMake ≥ 3.25 and a C++23 compiler: GCC 13 or Clang 18):

```sh
CXX=g++-13 pip install .            # builds the wheel with scikit-build-core
pip install pytest && pytest python/tests
```

## Use

```python
import numpy as np
import waferedge as we

data = we.read_wmap("tests/data/waferlens_sample.wmap")  # from tools/export_maps.py
m = data["maps"][0]                   # 2-D uint8: 0 off-wafer, 1 pass, >= 2 fail
we.features(m)                        # dies, fails, zone/sector/ring counts
we.signals(m)                         # 13 floats, names in we.SIGNALS
rules = we.Classifier.load("config/rules.txt")
rules.classify(m), rules.explain(m)   # 'edge_loc', and why
labels, clusters = we.clusters(m)     # int32 label per die, one dict per cluster
we.hough_line(m)                      # angle, rho, votes, dies on the line

stack = np.stack([x for x in data["maps"] if x.shape == m.shape])
we.features_batch(stack)              # arrays: dies (N,), zone_fails (N, 5), ...
rules.classify_batch(stack)           # one pattern name per map
```

**Zero copy, or an error.** Maps are read in place through the buffer protocol: a 2-D
(rows, cols) map or a 3-D (maps, rows, cols) stack of equal-sized maps, C-contiguous
`uint8`. A different dtype or a strided view (a transpose, a slice with a step) raises
`TypeError` instead of being copied silently; `np.ascontiguousarray(a, dtype=np.uint8)`
makes one that fits. Results are new arrays. The batch functions release the GIL while they
run, so threads can classify different stacks at once.

`read_wmap` copies each map into its own array (maps in a file differ in size).

## Tests

`python/tests/test_waferedge.py`: counts against NumPy's, batches against one-by-one calls,
the dtype / layout errors, clusters and a scratch by hand, and the rule classifier on the
WaferLens sample (batches equal single calls, accuracy against the truth).
