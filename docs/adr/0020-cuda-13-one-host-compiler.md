# ADR-0020: CUDA 13.3 with GCC 13 as nvcc's host compiler

- **Status:** Accepted
- **Date:** 2026-11-10

## Context
ADR-0001 chose CUDA 12.4 on WSL2 Ubuntu 22.04 and made GCC 11 nvcc's host compiler, because
nvcc 12.4 with GCC 13 fails on glibc 2.35's `bits/floatn.h` (`_Float128`). It also said that
Ubuntu 24.04 or a newer CUDA would remove the need, and that the `cuda` preset is the only
place naming GCC 11.

The laptop has moved to Ubuntu 26.04 (glibc 2.43, GCC 15 as the system compiler; GCC 11 and
13 still packaged). Two things broke:
- **CUDA 12.4 no longer compiles any `.cu` file.** glibc 2.43's `bits/mathcalls.h` declares
  `cospi`, `sinpi` and `rsqrt` with exception specifications that clash with CUDA 12.4's
  `crt/math_functions.h`, whatever the host compiler.
- **GCC 11 under nvcc no longer links with GCC 13 objects.** 22.04 shipped one libstdc++ for
  every GCC; 26.04 ships a separate `libstdc++.so` per GCC version. CMake adds GCC 11's
  library directory to the link (nvcc's implicit link directories), so the GCC 13 objects
  link against GCC 11's libstdc++ and miss its newer symbols
  (`std::__cxx11::basic_string::_M_replace_cold`, `std::to_chars(_Float128)`).

The Windows driver was updated to 621.14 (CUDA 13.5 capable) so a newer toolkit can run.

## Options considered
1. **Keep CUDA 12.4 and GCC 11**: can't compile on 26.04 (the glibc clash above).
2. **CUDA 13.3 with GCC 11 under nvcc**: compiles, but the link fails on 26.04 as above
   unless the link line is patched by hand.
3. **CUDA 13.3 with GCC 13 under nvcc**: one host compiler for `.cpp` and `.cu`; nvcc 13.3
   handles glibc's `_Float128` and the math declarations.

## Decision
Option 3. The `cuda` preset sets `CMAKE_CUDA_HOST_COMPILER` to `g++-13`, and CI's CUDA job
uses `nvcr.io/nvidia/cuda:13.3.1-devel-ubuntu24.04`, whose system compiler is GCC 13, so it
needs no toolchain PPA. `.cu` files stay C++20: nvcc 13.3 lists `-std=c++23` but ignores it
with a GCC 13 host ("not supported with the configured host compiler"), so the boundary of
ADR-0001 stays where it is.

## Consequences
- One host compiler in the CUDA build; GCC 11 is no longer needed anywhere.
- On the laptop (26.04, CUDA 13.3.73, GCC 13.4, sm_86) the `cuda` preset builds without
  warnings and passes all 221 tests, the 28 `gpu` tests included.
- The CUDA build needs CUDA 13.x and a driver for it (580 or newer on Windows for WSL2).
  CUDA 13 dropped GPUs older than compute capability 7.5; sm_86 is unaffected.
- **CUDA Graph launches cost ~20–25 µs more** with the 13.3 runtime (below). Graphs no longer
  move the CPU/GPU crossover earlier; overlapping batches still pays.
- The README's numbers were re-measured on the new setup. The detailed tables in `docs/` keep
  the setup they were measured on (CUDA 12.4, driver 551.76) and are re-measured when next
  touched.

## Measured: the old build against the new, same day
The old 22.04 install was kept, so its CUDA 12.4 binaries (built 2026-10-11) and the new CUDA
13.3 ones ran back to back on 2026-11-10, alternating, on the same driver (621.14). Medians of
3 repetitions, two or three rounds each:

| | Old (22.04, CUDA 12.4, nvcc host GCC 11) | New (26.04, CUDA 13.3, GCC 13) |
|---|---|---|
| All signatures, 1 CPU thread, 1 map | 31.6–32.7 µs | 31.6–33.2 µs |
| All signatures, GPU, no graph, 1 / 16,384 maps | 131–134 µs / 18.2–18.5 ms | 127–133 µs / 18.7 ms |
| All signatures, GPU with a graph, 1 map | **99 µs** | 119–122 µs |
| All signatures, GPU with a graph, 16 maps | **109 µs** | 136–139 µs |
| Overlapped with a graph, 1 / 16,384 maps | **52–54 µs** / 12.5–12.6 ms | 62 µs / 12.4–12.5 ms |
| CNN, int8, batch 1 / 256 | 3.03k / 25.2k maps/s | 3.16–3.18k / 25.1–25.3k maps/s |
| Sort → decision p50 / p99, synthetic, 1,000/s | 156–164 / 293–328 µs | 161–170 / 305–319 µs |

Only graph launches changed. The numbers first published (Ubuntu 22.04, October) are slower
than both columns: 71 µs for the CPU row, 198 µs for the GPU row without a graph and a GPU
clock capped at 0.83–0.99 GHz in the CNN run, against 1.46 GHz median now. That difference is
the laptop's state (power and clocks under Windows), not the toolchain.
- If a later GCC or CUDA breaks this pairing again, the preset and the CI image are still the
  only two places to change.
