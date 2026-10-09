# ADR-0002: Repository layout, and backends behind one interface

- **Status:** Accepted
- **Date:** 2026-10-09

## Context
The same computation (wafer-map features, clustering, later the CNN) exists in up to three
versions: a scalar reference, AVX2, and CUDA. The project's first rule is that every faster
version is tested against the reference on the same inputs, and benchmarked against it. The
layout has to make that easy and keep CUDA optional, because GitHub's runners have no GPU and
contributors may have no CUDA toolkit.

## Options considered
1. **One directory per module, backends as subdirectories** (`signature/scalar`,
   `signature/avx2`, `signature/cuda`), all in one library.
2. **Layered by language**: `include/` + `src/` for C++, `cuda/` for kernels, built into
   one library with CUDA sources added only when `WAFEREDGE_CUDA` is on.
3. **A separate CUDA library** that the core links optionally.

## Decision
Option 2, the layout from the design sketch:

| Path | What lives there |
|---|---|
| `include/waferedge/` | Public headers, one per module (`core`, `signature`, `secs`, `hsms`, `gem`, `infer`, `pipeline`) |
| `src/` | Host implementation (C++23), including the scalar and AVX2 backends |
| `cuda/` | `.cu` kernels (C++20, see ADR-0001), compiled only with `WAFEREDGE_CUDA=ON` |
| `tests/` | Catch2 tests; GPU tests carry the ctest label `gpu` |
| `bench/` | Google Benchmark binaries, each printing the machine (`describe_machine()`) |
| `fuzz/` | libFuzzer targets (phase 3) |
| `tools/` | Executables: `waferedge-info`, later `tool-emulator`, `edge-host`, export scripts |
| `python/` | pybind11 bindings (phase 5) |
| `docs/`, `docs/adr/` | Design notes, results, decisions |

Directories appear when their first file lands, not as empty placeholders.

Backends share one interface: a C++20 concept per computation (for example, something that
turns a batch of `WaferMap` views into feature rows), with `scalar`, `avx2` and `cuda` types
that satisfy it. Tests and benchmarks are templated over the concept, so adding a backend
adds one line to each, and a backend that drifts from the reference fails the same test the
others pass. The concept, not virtual functions, because the pipeline picks a backend at
build/configure time and the hot loops should inline.

## Consequences
- The CPU build has no CUDA dependency at all; CI builds and tests it on plain runners, and a
  separate job compiles the CUDA code.
- `cuda/` mirrors the module names in `include/`, so finding the GPU version of something is
  a matter of looking in the other directory.
- One library keeps linking simple. If the CUDA part grows large (phase 2b, 6), splitting it
  into `waferedge_cuda` is a mechanical change to `src/CMakeLists.txt`.
- The concept has to be designed before the second backend exists; phase 1 starts with the
  scalar type alone and extracts the concept when AVX2 lands.
