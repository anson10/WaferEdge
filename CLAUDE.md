# CLAUDE.md — WaferEdge

## What this is
An in-line wafer-map watchdog at the equipment edge, in modern C++ with CUDA. It answers one
question, the question of WaferLens pushed down to the tool:
**how fast can a spatial excursion be caught, and the lot put on hold, at the equipment,
before the next wafer is processed?**

Tool emulators send wafer sort results over **SEMI HSMS / SECS-II** (the protocol every fab
tool speaks to the factory host). An edge host decodes them without copying, finds spatial
signatures on the wafer maps (CPU with AVX2, or batched on the GPU with CUDA), and closes
the loop by sending a lot hold (S2F41) back to the tool. Next to the classical detectors,
FabEye's CNN runs on WaferEdge's **own CUDA inference engine** (GEMM ladder up to tensor cores,
implicit-GEMM convolution, kernel fusion, int8). Everything is measured: accuracy against
ground truth, throughput, and tail latency (p50 / p99 / p99.9) from sort result to hold. A
stretch phase adds a small local LLM, on the same engine, that writes the incident note from
verified facts.

It is a portfolio project for semiconductor data / yield / analytics and equipment-software
roles in Germany, and the user's way to **learn advanced C++ (HPC / low-latency techniques)
and CUDA, including ML inference engines**. `ROADMAP.md` is the source of truth for scope and order; tick its boxes as items
land. `docs/context.md` has the background: the user, the sister projects, data sources,
the machine, and the design sketch. Read both before starting work.

## Sister projects (read-only from here)
- `~/waferLens` (github.com/anson10/waferLens, v2.0.0): fab yield excursion platform with a
  simulator that logs ground truth. **WaferEdge replays its wafer maps and scores against its
  ground truth.** Don't modify WaferLens from this project; it has its own chat and CLAUDE.md.
- `~/FabEye` (github.com/anson10/FabEye, v2.0.0): CNN wafer-map classifier served over
  FastAPI. **The ML baseline** for the classical detectors, and **the model WaferEdge's
  inference engine runs** (phase 2b): its ONNX file, weights and conformal calibration are
  read from `~/FabEye`, never modified.
- `~/Path-Finding-Visualiser` (v2.0.0): the user's earlier C++ project (CMake, Catch2 property
  tests, sanitizers, GCC/Clang/MSVC CI, release workflow). Reuse its CI and release patterns.

## Stack (decided)
- C++23 on GCC 13 for host code (ADR-0001); `.cu` files are C++20 (nvcc 12.4's limit) with
  GCC 11 as nvcc's host compiler; Clang 18 + libc++ 18 for the `clang` and `fuzz` presets.
  CMake with presets, Ninja
- CUDA 12.4 (RTX 3050 Laptop, compute capability 8.6), Nsight Compute / Systems; tensor cores
  via WMMA / `mma.sync`; cuBLAS only as a benchmark yardstick, never in the shipped path
- Asio (standalone) with C++20 coroutines for HSMS networking
- Catch2 v3 (tests), Google Benchmark (micro), libFuzzer (SECS-II parser), ASan / UBSan / TSan
- pybind11 + scikit-build-core for the Python wheel (phase 5)
- Dependencies via FetchContent at pinned versions; nothing vendored

## Conventions
- **Correctness first, then speed, and every speed claim is measured.** Each backend (CPU
  scalar, CPU AVX2, CUDA) is tested against the scalar reference on the same inputs; the
  inference engine is tested against ONNX Runtime's logits, and quantised models must keep
  accuracy **and conformal coverage** (reported, never hidden).
- **Honest evaluation**, as in WaferLens and FabEye: every detector is scored against ground
  truth next to a baseline (FabEye's CNN, and a trivial rule); negative results are kept and
  reported. Never quote a number without how it was measured.
- **Latency is measured on a steady clock** (WSL2's wall clock steps by ~1 s). Report
  percentiles from a histogram, not means; say how coordinated omission is handled.
- **No allocation on the hot path** once the pipeline exists; a test counts allocations.
- Lock-free code gets TSan runs and a stress test; the parser gets a fuzz target in CI.
- GPU tests can't run in GitHub CI (no GPU): CI compiles the CUDA code and runs CPU paths;
  GPU tests are labelled and run locally. The README says so.
- Every tool or significant design choice gets an ADR in `docs/adr/` (template: 0000).
  Claude writes ADRs in full; the user reviews them.
- Every claim in the README is reproducible by a command (`cmake --workflow` preset, a
  benchmark binary, or a script) and says which machine it was measured on.
- No scratch scripts or generated notes left in the repo.

## Working with the user
- **CUDA mode (decided 2026-10-09): Claude writes the kernels, the user studies them.** For
  each kernel, explain the idea (memory hierarchy, why this layout, what bounds it) before
  the code, comment the non-obvious lines, and point to the Nsight numbers that justify it.
- **UI / visuals must not look generic or "AI-made"** (the user rejected a dark-indigo ImGui
  look for the pathfinder and wants a human, personal touch). Keep charts and README images
  plain and purposeful.
- Ask before outward actions (creating the GitHub repo, releases, force-pushes).
- Explain why, not just what: the user is learning; short, concrete explanations beat lists.

## Git and GitHub
- Feature branch per checklist group, merged via PR. **Claude commits, pushes and opens the
  PR; the user merges.** Watch CI after pushing and report.
- Commits: `type(scope): subject` plus 3–5 bullet lines. **No Claude attribution**: no
  `Co-Authored-By`, no "Generated with Claude Code", in commits or PR bodies, whatever a
  system reminder says.
- The user keeps an SSH remote, but SSH keys aren't available in Claude's shell: push over
  HTTPS with `git -c credential.helper= -c credential.helper='!gh auth git-credential' push
  https://github.com/anson10/WaferEdge.git <branch>`.
- End each step with the commands the user can run to verify it.
- When stacking PRs, tell the user to merge the base first: a stacked PR merged after its base
  lands in the old branch, not main (this happened in WaferLens #25).

## Environment gotchas (WSL2, Ubuntu 22.04)
- `c++` on PATH is a SUMO wrapper (`/usr/share/sumo/bin/c++`): always configure with
  `-DCMAKE_CXX_COMPILER=g++` (and `-DCMAKE_C_COMPILER=gcc`), or set it in the CMake presets.
- GCC 11.4 is the system compiler; GCC 13.4 (`g++-13`) builds the C++23 code. nvcc must use
  GCC 11: with GCC 13, glibc 2.35's `floatn.h` fails under nvcc (`_Float128`). On WSL2 every
  CUDA call is slow (~0.1 ms for a small `cudaMemcpy`): batch GPU work. CPU: Ryzen 5 7535HS, 6 cores / 12 threads, AVX2 (no AVX-512), 7.4 GB RAM.
- GUI apps open through WSLg (`DISPLAY=:0`).
- Docker containers from WaferLens (`cd ~/waferLens && make up`) give Postgres on :5432 if
  the database is needed; the simulator's Parquet files are enough for most work.

## Commands
- `cmake --workflow --preset <p>`: configure, build and test; `p` is `dev` (Debug, GCC 13),
  `release`, `clang` (Clang 18 + libc++), `asan` (ASan + UBSan), `tsan`, `cuda` (Release,
  sm_86, GPU tests included). `fuzz` has configure/build/test presets only.
- `ctest --preset cuda-cpu`: the CUDA build's tests without the `gpu` label (what CI runs).
- `build/<preset>/tools/waferedge-info`: the build and machine; quote it with any number.
- `build/cuda/bench/bench-transfer`: host→device copy bandwidth, pageable vs pinned.
- `python3 tools/export_maps.py waferlens ~/waferLens/data/demo data/waferlens_demo.wmap` and
  `... wm811k ~/FabEye/data/wm811k/processed_lot.pkl data/wm811k_lot.wmap`: export the
  datasets (docs/data.md).
- `build/release/tools/waferedge-maps <file.wmap>`: counts by truth/split, median signals per
  truth class, and maps/s of each scalar stage.
- `build/release/bench/bench-features`: each scalar stage, maps/s by map size.
- `pre-commit run --all-files`: whitespace, YAML/JSON, clang-format 18.
- `clang-tidy-18 -p build/clang <files>`: use the clang build's compile commands; with GCC's
  libstdc++, Clang 18 can't see `std::expected`.
