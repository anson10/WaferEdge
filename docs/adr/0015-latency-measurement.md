# ADR-0015: Latency is measured from the tool's schedule, in one process, into log-linear histograms

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
WaferEdge's question is how fast a spatial excursion is caught and the lot held. The answer
is a latency distribution, and its tail matters more than its median: one wafer stuck behind
a stall is one more wafer processed on a bad tool. Three things can make such a measurement
lie:
- **Coordinated omission.** If the load generator waits for the system, a stall delays the
  next sample instead of appearing in it, and the tail looks far better than it is.
- **Clocks.** WSL2's wall clock steps by about a second. Timestamps taken in two processes,
  or on two clocks, can't simply be subtracted.
- **Summaries.** A mean hides the tail. Keeping every sample and sorting works, but costs
  memory, and allocating while recording would itself disturb the system being measured.

## Options considered
1. **Measure from the send time, closed loop** (the next wafer goes out when the last one is
   answered): simple, and it suffers from coordinated omission.
2. **Measure from the emulator's open-loop schedule** (wafer k is due at start + k / rate,
   ADR-0012), whether or not it actually went out on time. A late emulator, a full HSMS
   window, a backed-up ring and a busy worker all show up as latency of the wafers behind
   them.
3. **Two processes** (tool-emulator and waferedge-edge) with timestamps matched afterwards.
   CLOCK_MONOTONIC is shared by all processes on one Linux machine, so this would work, but it
   needs a log from each side and a join.
4. **One process**: the emulator on one thread, the edge host's network thread and workers on
   others, all reading the same `steady_clock`. The edge host reports each verdict through an
   observer callback, and the benchmark joins it with the emulator's schedule by wafer id.

For the summary:
- **Sorted samples:** exact, O(n) memory, and the sort runs at the end.
- **Fixed-width buckets** (as `bench-emulator` used: 1 µs up to 100 ms): either coarse at
  the low end or huge for a wide range.
- **Log-linear buckets, HdrHistogram style:** relative precision everywhere, fixed size.

## Decision
Options 2 and 4: `bench-latency` measures from the scheduled time in one process.
`pipeline::LatencyHistogram` splits each power-of-two range into 128 equal sub-buckets
(exact below 256 ns). That's 7,424 counters, 58 KB, covering 1 ns to the whole 64-bit range,
each value stored to within 1/128 of itself. A percentile reports its bucket's upper edge,
so it is never below the true value. Min and max are exact.

Two latencies are reported:
- **Sort → decision**, for every analysed wafer: the moment a HOLD would go out if this
  wafer completed the rule.
- **Sort → HOLD at the tool**, for every hold: from the scheduled time of the wafer that
  completed the rule to the tool receiving the S2F41. This is the end-to-end number, with
  fewer samples.

The stage breakdown (emulator lateness, scheduled → received, received → analysed,
analysed → decided) explains each total.

## Consequences
- The measurement includes the load generator: the emulator's own lateness is reported as a
  stage, so it is a visible floor under every total, not a hidden one.
- Wafers dropped because every ring was full have no verdict and so no latency. Each row
  reports them as a count next to the percentiles. Their "latency" is infinite: no hold can
  come from them.
- One process means the emulator competes with the edge host for cores. On a 4-thread
  container that shapes the tails (see docs/pipeline.md); the laptop's 12 threads leave room.
- The histogram's tests check bucket arithmetic at every power of two, and check percentiles
  of 100,000 log-uniform samples against the sorted exact ones (never below, within 1/128).
- Recording is O(1) and allocation-free, so the histogram can move into the edge host itself
  if live percentiles are wanted later.
