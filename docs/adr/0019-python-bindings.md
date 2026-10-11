# ADR-0019: Python bindings read maps in place, or refuse them

- **Status:** Accepted
- **Date:** 2026-10-11

## Context
Phase 5 asks for the features, the classifier and the clustering from Python, on NumPy
arrays, as a wheel tested from pytest. The stack names pybind11 and scikit-build-core
(CLAUDE.md). Two questions remained: what the module accepts, and what it covers.

Wafer maps are small (24² to 64² bytes) and many; a stack of 65,536 40×40 maps is 105 MB.
NumPy hands out arrays of any dtype and any layout (a transpose or a slice with a step is a
strided view of the same memory).

## Options considered
1. **`py::array_t<uint8_t, forcecast>`**: accepts anything and converts. Easy to call, but a
   float array or a transposed view is copied without the caller knowing, and the copy can
   cost more than the computation.
2. **The buffer protocol, uint8 and C-contiguous only, otherwise TypeError**: the module
   reads the caller's memory in place as `WaferMapView`s, which are already spans over bytes.
   The caller converts once, explicitly, with `np.ascontiguousarray(a, dtype=np.uint8)`.
3. **Bind everything** (the GPU engine, the edge host, the protocol stack): more surface, and
   a wheel that needs CUDA to build. The edge pipeline is a service, not a library call.

## Decision
Option 2, for the CPU code that is useful in analysis: features (one map or a stack), the
13 signals, the rule classifier (classify, explain, load / parse), the clusters with their
label grid, the Hough line, and a `.wmap` reader. One extension module, `waferedge._core`,
linked against the same `waferedge` library as the tools; scikit-build-core builds it from the
CMake project with `WAFEREDGE_PYTHON=ON` (tests, benchmarks and tools off). pybind11 v2.13.6
is fetched at its tag like every other dependency (ADR-0003).

## Consequences
- No hidden copies: what goes in is read where it lies. Batches take a 3-D stack, so one call
  crosses into C++ for N maps, and they release the GIL while they run.
- Callers with other dtypes or layouts convert first; the error message says how.
- A stack needs maps of one shape. Files of mixed shapes (WM-811K has 341) are grouped by
  shape first; `read_wmap` returns one array per map for that reason.
- The library is built position-independent when the module is on.
- The GPU paths stay C++-only. If they are wanted from Python later, they need a CUDA-built
  wheel, which is a separate decision.
- The wheel built by the release workflow is a plain `linux_x86_64` wheel for one Python
  version, attached to the GitHub release; not on PyPI (that would need manylinux builds).
