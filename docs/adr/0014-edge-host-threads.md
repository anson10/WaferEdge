# ADR-0014: The edge host runs one network thread and N analytics threads joined by SPSC rings

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
The edge host connects to a tool as the HSMS / GEM host, classifies every wafer report and
holds a lot (S2F41) when it sees an excursion. Three kinds of work compete for time: the
network (HSMS framing, GEM, the S6F12 that must answer each report within the tool's T3),
the detectors (tens to hundreds of microseconds per map, more on the GPU path with batching),
and the decision to hold. The network must never wait for a detector, or the tool's
transactions time out. The hot path must not allocate (CLAUDE.md), and every stage's time
must be measurable for the latency work later in phase 4.

## Options considered
1. **One thread for everything** (classify inside the GEM event handler): no hand-offs, but a
   slow map delays every later HSMS message, and only one core works.
2. **Network thread, analytics threads, decision thread**, as first sketched in the roadmap:
   the decision is a table lookup of nanoseconds, so its own thread buys nothing and adds a
   hand-off (a wake-up of microseconds) to every verdict.
3. **Network thread + N analytics threads, the decision on the network thread**: each worker
   has one SPSC ring in (wafer slots) and one out (verdicts), so every ring keeps exactly one
   producer and one consumer (ADR-0013). The network thread is the one place where all
   workers' verdicts meet, which the per-lot rule needs: a lot's wafers go to different
   workers.

How a thread waits for an empty ring was a second choice:
- **Spin only**: the lowest latency, a core burnt per worker even when the tool is idle (a
  real tool sends a wafer every minute or so).
- **Mutex + condition variable** per ring: a lock on every push (ADR-0013 measured 27 µs per
  round trip against 0.7 µs).
- **Spin a while, then sleep on a futex**, waking only when someone sleeps.

## Decision
Option 3, in `pipeline::`:
- `EdgeCore`: the network thread's logic without I/O (GEM host events in, wafers out, verdicts
  back, the decision, the HOLD), so tests drive it on a fake clock and count its allocations.
  `EdgeHost` runs it on an `hsms::Session` (active) with Asio.
- `Analyzer`: one worker. Wafer slots are fixed-size (a 32-character lot id, up to 64 × 64
  bins), so a map is copied once, from the HSMS receive buffer into its slot, and nothing is
  allocated per wafer. Wafers go round robin to the workers whose ring has room. If every ring
  is full the wafer is **dropped and counted**: the S6F12 has already gone back, and stalling
  the network thread would stall the tool's transactions. Larger maps and longer lot ids are
  refused and counted, never truncated.
- Workers wait by spinning for `spin` (50 µs default), then sleeping on a `Doorbell`: an
  epoch `std::atomic<uint32_t>` to `wait` on and a `sleeping` flag. Both sides touch the flag
  with an exchange (an RMW reads the latest value, so either the producer sees the sleeper or
  the sleeper's recheck sees the push), and the producer makes the futex call only when a
  sleeper announced itself.
- The network thread sleeps in epoll, so workers wake it through an **eventfd** that Asio
  reads. A `pending` flag (again an exchange on both sides) turns a burst of verdicts into one
  `write()`.
- `DecisionRule`: hold a lot once k of its last W verdicts show the same pattern other than
  `none` (default 3 of 5). One flagged wafer is often a classifier false positive or a one-off
  defect. State is a fixed table of the 64 most recently seen lots, scanned linearly (a tool
  works through one lot or a few at a time); the least recently seen is forgotten when full.

## Consequences
- The hot path allocates nothing once warm: `waferedge-alloc-tests "[edge]"` runs 100 rounds
  of report → classify → decide → HOLD → HCACK on one thread (the analyzer polled by hand) and
  fails on any allocation; it does fail when a logger that formats strings is plugged in.
- The doorbell's correctness is tested, not argued only: a two-thread stress test makes the
  consumer sleep thousands of times and a watchdog counts lost wake-ups (none), under TSan in
  CI. With the textbook mistake put in on purpose (the producer reads the flag with a relaxed
  load instead of an exchange), the test lost a wake-up in 1 of 5 runs on the 4-thread cloud
  container. The test finds this race only by chance; the exchange is what makes the code
  correct.
- Two ordering bugs were found by the closed-loop test over TCP and fixed. (1) Timestamps: one
  `now` read before draining could precede a verdict pushed during the drain, so stamps are
  now read when each step happens. (2) The end of a replay: the host acknowledges a report
  before analysing it, so the emulator, which stops once every S6F12 is in, can leave before
  the last lot's HOLD. A real tool keeps running; the test stops the emulator only when the
  edge host has settled, and the command-line tools show the effect as a hold "decided" but
  not "answered".
- Verdicts from different workers can reach the decision out of wafer order. The rule counts
  within a window of recent verdicts, so a reordering of a few wafers barely changes it.
- A forgotten lot that reappears starts afresh, and a second HOLD gets HCACK 5 ("already in
  condition") from the tool: harmless, and counted.
- GPU workers (batching maps per launch) fit the same shape: a worker that takes several
  slots before classifying. The decision and the network thread don't change.
- The roadmap's "also raise S5F1 alarms" doesn't fit GEM: S5F1 goes from the equipment to the
  host, so the host can't raise one at the tool. The edge host logs each hold instead; if the
  fab needs a host-side alarm, that goes to the factory's MES, outside this project.
