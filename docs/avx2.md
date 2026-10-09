# AVX2 feature backend

The zone / sector / ring counts (ADR-0004) computed 32 dies at a time with 256-bit AVX2
vectors. **3.2–5.4× the scalar reference on synthetic maps, about 3.5× on the real datasets,
equal to scalar field for field on all 103,698 real maps.** This page explains how it works,
what was tried and measured, and what it means for the pipeline. Code:
`src/features_avx2.cpp`; build and dispatch decision: ADR-0006.

## The idea

AVX2 gives 16 registers of 256 bits; as bytes, one register holds 32 lanes, and one
instruction operates on all of them. Our bins and the geometry's zone / sector / ring ids are
bytes, so one instruction can look at 32 dies.

The features are a **histogram**: per die, increment the counter of its zone, sector and ring
(23 buckets, dies and fails). A histogram is the awkward case for SIMD: 32 lanes that want to
increment counters, some of them the same counter, have no conflict-free vector instruction.
The scalar loop has its own problem: neighbouring dies hit the same counter, so each
increment waits for the previous store to the same address.

**Turn it around.** Instead of each die finding its bucket, each bucket asks all 32 dies:

```
for each bucket k:
    in = (table[i..i+31] == k)           vpcmpeqb: 0xFF where equal, 0x00 elsewhere
    dies_k  -= in & ~off_wafer           0xFF is -1 as a signed byte: subtracting adds 1
    fails_k -= in & fail
```

No memory writes, no conflicts; each of the 32 lanes counts its own column of dies. Two masks
per block, computed once: `off_wafer = (bin == 0)`, and `fail = (bin >= 2)`, written as
`max(bin, 2) == bin` because AVX2 has no unsigned byte compare.

## Three practical problems

**Byte counters overflow.** A lane holds 0..255 and gains at most 1 per block, so after 255
blocks the counters are emptied into 64-bit totals. `vpsadbw` against zero ("sum of absolute
differences") adds each group of 8 bytes into a 64-bit lane: a horizontal byte sum in one
instruction.

**Registers.** 23 buckets × 2 counters = 46 vectors; there are 16 registers. If counters don't
fit, the compiler *spills* them: store to the stack, reload, update, store again, every block,
which is a dependency chain through memory. So the work is split into passes of K buckets, and
the data (a few KB, in L1 cache) is read once per pass. How many per pass is a measurement,
not a guess (below).

**The tail.** A map's size is rarely a multiple of 32 (25×27 = 675 leaves 3). A 32-byte load
past the end of the map is undefined behaviour and can fault at a page boundary (the last map
of a `MapSet` ends at the end of its buffer), so the last `n % 32` dies go through the same
scalar code the reference uses (`src/features_detail.hpp`).

## How many buckets per pass

Same kernel, buckets per pass varied; maps/s, median of 5, `release` preset, synthetic maps
(10% fails), two runs each where shown:

| Buckets per pass | Passes | 24² | 64² | Inner loop |
|---|---|---|---|---|
| 2 | 12 | 1.61–1.69M | 269–298k | no spills |
| **3** | **9** | **1.75M** | **302–325k** | **no spills** |
| 4 | 7 | 1.41–1.45M | 245–248k | spills |
| 5 | 5 | 1.49M | 235k | ~3 of 10 counters spilled |
| 5 / 8 / 10 | 3 | 1.35M | 236k | heavy spills |

Reading the data three times more often (9 passes instead of 3) is *faster*: L1 reloads are
independent and overlap, while a spilled counter is a serial store → load → add → store chain.
At 2 per pass the extra passes start to cost more than the registers they free. The chosen
inner loop (3 buckets), from `objdump -d build/release/src/CMakeFiles/waferedge.dir/features_avx2.cpp.o`:

```
vmovdqu  (%rdi,%rbx),%ymm0         32 bins
vmovdqu  (%rsi,%rbx),%ymm4         32 table entries
vpmaxub  0x60(%rsp),%ymm0,%ymm3    max(bin, 2)  (a constant read, not a spill)
vpcmpeqb %ymm13,%ymm0,%ymm1        off-wafer mask
vpcmpeqb %ymm4,%ymm15,%ymm2        in bucket k
vpcmpeqb %ymm3,%ymm0,%ymm0         fail mask
...                                two more bucket compares
vpandn   %ymm2,%ymm1,%ymm5         in bucket and on the wafer
vpand    %ymm2,%ymm0,%ymm2         in bucket and failing
vpsubb   %ymm5,%ymm7,%ymm5         counters stay in ymm7..ymm12
...
```

20 instructions for 96 die-bucket checks; the six counters never leave their registers.

## Testing it

- `tests/test_backends.cpp` is one template over every backend, so the CUDA backend (phase
  2a) joins with one more type. AVX2 must equal scalar with `==` on the whole `Features`
  struct: 18 shapes (single dies, 1×31 / 1×32 / 1×33 for the tail, odd WM-811K shapes, up to
  300×300) × 5 fail densities; every byte value 0..255 as a bin (128..255 are negative as
  signed bytes, which a signed "≥ 2" would get wrong); the real fixture.
- **The overflow test was too weak at first.** Changing the flush limit from 255 to 256
  blocks (a counter would wrap to 0) still passed every test: on a real wafer shape the
  tables change from block to block, so no single lane ever counts near 255. The test now
  uses explicit tables (`Geometry::from_tables`) with every die in bucket 0, 255 to 600
  blocks; the same mutation now fails with `5 == 8197` (every lane wrapped to 0, only the 5
  tail dies counted). A test that can't fail proves nothing; breaking the code on purpose is
  how to find out.
- `waferedge-maps` compares AVX2 with scalar on every map of a file:
  `avx2 features == scalar on 24090 of 24090 maps` (WaferLens), `79608 of 79608` (WM-811K).

## Results

Synthetic maps, `build/release/bench/bench-features --benchmark_filter=BM_features
--benchmark_repetitions=5` (median, coefficient of variation ≤ 7%):

| Map | Scalar | AVX2 | Speed-up |
|---|---|---|---|
| 24² | 599k maps/s | 1.89M | 3.2× |
| 30² | 371k | 1.35M | 3.7× |
| 40² | 199k | 793k | 4.0× |
| 64² | 64k | 347k | 5.4× |

Real data, whole passes over the file, `build/release/tools/waferedge-maps <file>`, several
runs (whole-file timings on this laptop swing ±25% between runs: WSL2, boost clocks and other
load; the synthetic benchmark above is steadier because it reports a median of repetitions):

| Data | Scalar | AVX2 | Speed-up |
|---|---|---|---|
| WaferLens, 24,090 maps, 3 runs | 303k–314k maps/s | 1.03M–1.13M | 3.3–3.7× |
| WM-811K, 79,608 maps, 5 runs | 180k–208k | 531k–860k | 2.6–4.7× (median of runs ≈ 3.9×) |

Larger maps gain more: the per-map costs (nine flushes, the scalar tail) are spread over more
blocks. Ryzen 5 7535HS (Zen 3+), WSL2, GCC 13.4, one thread.

## What it means for the pipeline: Amdahl's law

The classifier (signals + rules) runs at ~11.6k maps/s before and after: features were ~5% of
its time, the Hough transform ~85% (docs/signatures.md). Making 5% of the work 4× faster
saves under 4% of the total. The next speed-up has to come from the Hough transform, which is
why it is the first GPU kernel in phase 2a; the AVX2 features matter where features run alone
(a first screen on every wafer) and as the CPU side of the CPU-vs-GPU comparison.
