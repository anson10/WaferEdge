# ADR-0006: AVX2 through per-function target attributes and run-time dispatch

- **Status:** Accepted
- **Date:** 2026-10-09

## Context
The feature counts have an AVX2 implementation (docs/avx2.md) that is 3–5× the scalar one.
The release artifacts, the Python wheel (phase 5) and CI binaries must still run on any
x86-64 CPU, with or without AVX2, and the AVX2 code must never execute on a CPU without it.
Backends are compared through one concept (ADR-0002).

## Options considered
1. **Build everything with `-mavx2` (or `-march=native`)**: simplest, fastest everywhere the
   compiler can vectorise; the binary dies with "illegal instruction" on CPUs without AVX2,
   and `-march=native` binaries built on one machine may not run on another.
2. **Compile only `features_avx2.cpp` with `-mavx2`**: looks contained, but isn't. That file
   includes headers with inline functions (the standard library, our own); its copies of
   them may be compiled with AVX2 instructions, and the linker keeps one copy of each inline
   function for the whole program. If it keeps the AVX2 one, scalar code calls it too: an ODR
   problem that crashes only on older CPUs, only sometimes, and only after a rebuild.
3. **Per-function `[[gnu::target("avx2")]]` plus run-time dispatch**: only the marked kernel
   functions may use AVX2; the rest of the file, including every inline function it pulls in,
   is compiled for the baseline CPU. `backend::Avx2::available()` asks the CPU
   (`__builtin_cpu_supports("avx2")`), and `best_features()` picks the fastest available
   backend once.
4. **Function multiversioning (`target_clones`)**: the compiler builds and dispatches several
   versions of one function automatically; convenient for auto-vectorised code, but our kernel
   is written with intrinsics and has no portable source to clone from.

## Decision
Option 3. Intrinsics live only in functions marked `[[gnu::target("avx2")]]` (GCC and Clang
both support the attribute); helpers that take or return `__m256i` carry it too, so their
calling convention matches. Callers go through `backend::Avx2::features`, which tests and
benchmarks call only when `available()` is true, and the pipeline through `best_features()`.
On non-x86 builds the backend exists, reports itself unavailable and falls back to scalar.

## Consequences
- One binary runs everywhere and uses AVX2 where the CPU has it; CI's tests skip the AVX2
  cases on a runner without AVX2 instead of crashing.
- The dispatch costs an indirect call per map (one function pointer), negligible next to
  ~0.5–3 µs of work per map.
- Every intrinsic-using function must remember the attribute; forgetting it is a compile
  error ("inlining failed ... target specific option mismatch"), not a silent bug.
- AVX-512 (not on this CPU) or a CUDA backend joins the same way: one more backend type, one
  more `available()`, one more line in the test and benchmark lists.
