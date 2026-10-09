# ADR-0003: Dependencies through FetchContent at pinned versions

- **Status:** Accepted
- **Date:** 2026-10-09

## Context
WaferEdge needs a small set of libraries: Catch2 (tests), Google Benchmark (micro-benchmarks),
standalone Asio (HSMS networking, phase 3), later pybind11 (phase 5). It builds on a WSL2
laptop, on GitHub's Ubuntu runners and inside NVIDIA's CUDA container, which have different
system packages. Builds must be reproducible: a benchmark number from today must come from
the same library code next month.

## Options considered
1. **System packages** (`apt install catch2 libbenchmark-dev libasio-dev`): fast configure,
   but versions differ between Ubuntu 22.04, 24.04 and the CUDA container (Catch2 v2 vs v3).
2. **A package manager** (vcpkg or Conan): pinned and cached, but one more tool to install,
   bootstrap in CI and explain, for four libraries.
3. **CMake FetchContent at pinned tags**: CMake alone; versions written in
   `cmake/Dependencies.cmake`.
4. **Vendoring** (copying sources into the repo): pinned and offline, but bloats the history
   and hides updates.

## Decision
Option 3. Every dependency is declared in `cmake/Dependencies.cmake` with a release tag and
`GIT_SHALLOW`, and nothing is vendored:

| Library | Version | Used for |
|---|---|---|
| Catch2 | v3.7.1 | tests (same as Path-Finding-Visualiser) |
| Google Benchmark | v1.9.1 | micro-benchmarks |
| Asio (standalone) | 1.30.2 | HSMS transport with C++20 coroutines |
| pybind11 | added in phase 5 | Python bindings |

Third-party headers are included as `SYSTEM`, so our `-Werror` doesn't fail on their
warnings. CUDA itself and the compilers are system installs, named in the presets.

## Consequences
- A fresh configure downloads the sources (about a minute); later configures reuse them.
  Each preset has its own build tree and so its own copy; acceptable at this size.
- Upgrading a library is a one-line change in one file, reviewed like any other change.
- A git tag can in principle be moved upstream. If that ever matters, pin to commit hashes
  or to release tarballs with `URL_HASH`.
- Offline builds need a populated `build/` tree or `FETCHCONTENT_BASE_DIR` pointing at one.
