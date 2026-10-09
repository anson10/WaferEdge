# ADR-0001: C++23 on GCC 13 for the host, CUDA 12.4 (C++20) for the GPU

- **Status:** Accepted
- **Date:** 2026-10-09

## Context
WaferEdge has to show three things with measured requirements behind them: decoding SECS-II
messages without copying or allocating, a pipeline whose tail latency depends on lock-free
hand-offs, and GPU kernels (classical detectors, then a CNN inference engine) with explicit
control of memory and streams. The language must give that control and be what the target
roles use: equipment software and fab analytics in Germany are mostly C++ on the tool side,
and NVIDIA's own stack is C++ with CUDA.

The machine is WSL2 Ubuntu 22.04 with GCC 11.4 as the system compiler and CUDA 12.4 on an
RTX 3050 Laptop (compute capability 8.6). CUDA 12.4 accepts host compilers up to GCC 13.

Two C++23 library features matter here:
- `std::expected` for errors on the hot path (the SECS-II decoder in phase 3 must not throw).
  GCC 11 doesn't have it.
- `std::format`, so text output needs no extra dependency. GCC 13 has it; `std::print`
  arrived only in GCC 14, so we format into a string and write it ourselves.

## Options considered
1. **C++20 on GCC 11**: nothing to install; we'd write and test our own `Result<T, E>` and
   pull in fmt.
2. **C++23 on GCC 13** (toolchain PPA): `std::expected`, `std::format`, deducing `this`,
   `std::to_underlying`; still within what nvcc 12.4 accepts as a host compiler.
3. **Clang 18 as the main compiler**: also C++23, but nvcc 12.4 doesn't accept it as a host
   compiler, so the CUDA build would need GCC anyway.

## Decision
Option 2. The host code is C++23 built with GCC 13 (`g++-13`, pinned in the presets).
Clang 18 builds the same code in CI and drives the libFuzzer preset, so the code stays
portable across both compilers. Clang 18 uses **libc++ 18**, not GCC's libstdc++: libstdc++
only exposes `std::expected` when `__cpp_concepts >= 202002L`, which Clang reports from
version 19 on. A side benefit is that CI compiles against two standard library
implementations.

nvcc 12.4 compiles at most C++20, so `.cu` files are C++20. The rule that follows: **headers
included by `.cu` files stay C++20 and free of C++23 types**; a `.cu` exposes plain functions
(spans, raw pointers, error codes) and a C++23 `.cpp` wraps them in `std::expected`. The CUDA
runtime API is a C API, so host code that only calls it (device queries, copies) is an
ordinary `.cpp` and needs no nvcc.

nvcc's host compiler is **GCC 11** (Ubuntu 22.04's system compiler), not GCC 13. With GCC 13,
nvcc's front end treats `_Float128` as a built-in type, and glibc 2.35's `bits/floatn.h`
then fails to compile (`typedef __float128 _Float128`: "invalid combination of type
specifiers"). GCC 13 itself copes with that header; nvcc's emulation of it doesn't. Since
`.cu` files are C++20 behind plain interfaces, GCC 11 is enough for them, and GCC 13's
libstdc++ links GCC 11 objects (the library is backward compatible).

## Consequences
- Errors are `std::expected` everywhere on the host; no home-made result type to maintain.
- Anyone building WaferEdge needs GCC 13 (Ubuntu 24.04 ships it; on 22.04 the toolchain PPA).
  The README says so.
- Two language levels in one tree: a C++23 header pulled into a `.cu` file fails to compile.
  That failure is loud, so the rule enforces itself. If a later CUDA whose nvcc accepts
  C++23 is installed, the boundary can move.
- Two host compilers in the CUDA build (GCC 13 for `.cpp`, GCC 11 under nvcc). Moving to
  Ubuntu 24.04 (glibc 2.39) or a newer CUDA removes the need; the preset is the only place
  that names GCC 11.
- If GCC 13 turns out to be a problem (a compiler bug on the hot path, a CUDA mismatch), the
  fallback is C++20 with a `Result` type; only error-handling code would change.
