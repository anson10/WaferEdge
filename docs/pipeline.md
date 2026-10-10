# The edge pipeline

Phase 4 puts the pieces together into a closed loop: a tool emulator sorts wafers and reports
them over HSMS / SECS-II, the edge host finds spatial signatures and holds the lot, and the
time from sort to hold is measured. This page grows with the phase; so far it covers the
**tool emulator** (`include/waferedge/emulator/`, `src/emulator/`, `tools/tool_emulator.cpp`;
ADR-0012), the **SPSC ring** the edge host's threads hand wafers over with
(`include/waferedge/pipeline/spsc_ring.hpp`; ADR-0013), the **edge host** itself
(`include/waferedge/pipeline/`, `src/pipeline/`, `tools/edge_host.cpp`; ADR-0014), and the
**latency** from sort result to hold (`bench/bench_latency.cpp`; ADR-0015).

```
 tool-emulator (passive HSMS, GEM equipment)          waferedge-edge (active HSMS, GEM host)
   Replay: constant-rate schedule ──S6F11 wafer report──►  decode → classify → decide
   holds: skip the lot's wafers ◄──S2F41 HOLD / RELEASE──  hold the lot
```

## The tool emulator

`emulator::Tool` is passive HSMS equipment running `gem::Equipment`. It replays a `.wmap`
file in `tested_at` order, one S6F11 wafer report (CEID 100, report 10: lot, wafer, rows,
cols, bins; docs/secs.md) per slot of a constant-rate schedule.
- It goes ON-LINE as soon as GEM communication is established (the operator's switch,
  automated), and the schedule runs only while ON-LINE.
- It answers S2F41 HOLD / RELEASE with HCACK 0 (done), 5 (already held / not held) or 6
  (no such lot).
- A held lot's remaining wafers are skipped and counted as **withheld**: the wafers a real
  tool would not have processed, which is the payoff the closed loop is measured by.
- It stops once every wafer is reported and answered (S6F12, or a failed transaction) and a
  linger has passed (`--linger`, 1 s by default; see the edge host below), or on Ctrl-C.

Lots are named from the `.wmap` lot ids, `LOT-000281`. The logic is split as in HSMS and GEM:
`emulator::Replay` holds the schedule, the holds and the per-wafer log without any I/O and
is tested on a fake clock; `emulator::Tool` wires it to a `gem::Equipment` on an
`hsms::Session`.

### Pacing: open loop, so overload shows

Wafer k is due at `start + k / rate`, whether or not the host kept up (ADR-0012). A sender
that falls behind catches up: the late wafers go out back to back, each with its *scheduled*
time recorded. Phase 4 measures latency from that scheduled time, so a stalled host shows as
late wafers. A closed loop (send the next wafer when the last is acknowledged) would hide the
stall: the generator would simply wait, and the wafers that should have queued behind it
would never be measured. That is coordinated omission, and the reason for this design.

When HSMS's 64-slot transaction window is full (64 reports awaiting S6F12), the next wafer
waits for a reply (`gem::GemError::busy`) and goes out late; `Tool::window_full()` counts
how often. The benchmark below found this case: the first version treated a full window as
a dead link and stopped for good.

### How good a load generator it is

The emulator's own lateness (actual send time minus scheduled time) is a floor under every
latency measured with it. `build/release/bench/bench-emulator` runs the emulator and a
`gem::Host` in one process, on two threads with their own `io_context`, over TCP on
127.0.0.1, with 40×40 maps, for about 3 s per rate. Release, GCC 13.3, **4-thread cloud
container, not the laptop**:

| Target wafers/s | Achieved | Late p50 | p99 | p99.9 | max | Window full |
|---|---|---|---|---|---|---|
| 100 | 100 | 124 µs | 295 µs | 924 µs | 924 µs | 0 |
| 1,000 | 1,000 | 72 µs | 232 µs | 1.7 ms | 2.7 ms | 0 |
| 10,000 | 10,000 | 19 µs | 273 µs | 1.7 ms | 2.9 ms | 0 |
| 50,000 | 50,000 | 25 µs | 188 µs | 1.8 ms | 2.7 ms | 79 |
| 200,000 | 199,994 | 30 µs | 1.2 ms | 2.8 ms | 3.3 ms | 3,813 |

- The rate holds up to 200,000 wafers/s, far above any real tool (a few wafers per minute)
  and above what the detectors process per core; the emulator won't be the bottleneck.
- The median lateness is *worse* at low rates: an idle thread takes longer to wake from its
  timer than a busy one. The tails (p99.9 around 1–3 ms) are the container's scheduler; they
  bound how finely phase 4 can resolve tail latency here.
- Lateness histograms use 1 µs buckets; percentiles are read from the histogram, not
  averaged.

### Interoperability: an independent GEM host

`tools/secsgem_host.py` drives the emulator from secsgem 0.3.0, a separate Python HSMS / GEM
implementation: it receives the wafer reports, holds a lot after its first wafer, and checks
none of that lot's wafers arrive after the HCACK. On the WaferLens sample: 22 reports
decoded with the right map sizes, HCACK 0, LOT-000281's other 2 wafers withheld, none late.
secsgem logs one warning, "unexpected S1F14": both sides sent S1F13 at the same moment
(E30 allows it), and its own request's answer arrived after it was already communicating.

## The SPSC ring: hand-offs between threads

The edge host's threads (network, detectors, decision) pass wafers along through
`pipeline::SpscRing<T, N>`: a bounded, lock-free ring for exactly one producer and one
consumer thread.

```
   head (consumer writes)                       tail (producer writes)
     v                                            v
   [ slot | slot | slot | ...               ... | slot ]   N slots, N a power of two
   empty: head == tail     full: tail - head == N     slot of counter c: c & (N - 1)
```

- **One writer per counter**, so no locks and no compare-and-swap: the producer fills a slot
  and publishes it with a *release* store of `tail`; the consumer's *acquire* load of `tail`
  then guarantees it sees the finished slot. The same pair on `head` tells the producer a
  slot has been read before it is reused. x86 hides a missing release (the stress test passed
  with a relaxed store on this machine), so the ring's tests run under ThreadSanitizer, which
  reported the race on the slot at once when the release was removed on purpose.
- **Cache lines:** `head` and `tail` each on their own 64-byte line, so the two threads'
  writes don't invalidate each other's line (false sharing). Each side also keeps a private
  copy of the other's counter and reloads it only when the ring looks full or empty.
- **In place:** `begin_push` / `commit_push` and `front` / `pop` give out the slot itself,
  so a wafer map can be written once, straight into the ring.
- **Bounded:** no allocation after construction; a full ring is backpressure the producer
  sees (`begin_push` returns nullptr).

Tests (`tests/test_spsc_ring.cpp`): empty / full / FIFO, counters wrapping round the slots,
in-place slots, the cache-line layout, and two-thread stress tests with a checker on every
message (1 M messages through a 16-slot ring, 100,000 2 KB slots checked byte for byte,
the uncached variant through a 2-slot ring), under TSan in the `tsan` preset and CI.

`build/release/bench/bench-ring`, against a `std::mutex` + `std::condition_variable` queue and
against the ring without cached counters; threads pinned to CPUs 0 and 1; throughput is the
median of 3 runs with the range; **4-thread cloud container, not the laptop**:

| Payload | SPSC ring | ring, no cached counters | mutex + condvar | ring / mutex |
|---|---|---|---|---|
| 16 B | 28.8 M/s (23.6–32.7) | 11.7 M/s (10.7–25.1) | 4.4 M/s (4.2–11.2) | 7x |
| 1,600 B (a 40×40 map) | 4.6 M/s (4.4–4.6) | 3.4 M/s (3.3–3.6) | 1.4 M/s (1.4–1.5) | 3x |

| Round trip, 16 B (ping-pong) | p50 | p99 | p99.9 |
|---|---|---|---|
| SPSC ring | 740 ns | 1.1 µs | 19 µs |
| ring, no cached counters | 850 ns | 1.3 µs | 22 µs |
| mutex + condvar | 27 µs | 65 µs | 217 µs |

- The mutex queue's cost is mostly waking the other thread (a futex system call per hand-off
  when it sleeps): ~27 µs per round trip against ~0.7 µs. For the closed loop that is the
  difference between a hand-off being free and costing more than the whole HSMS transaction.
- Caching the counters helps, by less than its reputation and noisily here (the uncached
  ring's runs range from 10.7 to 25.1 M/s).
- At 1,600 bytes every queue is bound by copying the map; the in-place API removes that copy.

On the laptop (Ryzen 5 7535HS, 6 cores / 12 threads, WSL2, release build), pinned to
separate cores (`--cpus 0 2`) and to the two hyperthreads of one core (`--cpus 0 1`):

| Payload | CPUs | SPSC ring | ring, no cached counters | mutex + condvar |
|---|---|---|---|---|
| 16 B | 0, 2 | 23.7 M/s (16.3–25.4) | 17.2 M/s (16.1–18.0) | 6.8 M/s (6.8–6.8) |
| 16 B | 0, 1 | 22.7 M/s (22.0–24.6) | 17.2 M/s (16.3–17.3) | 6.9 M/s (6.4–6.9) |
| 1,600 B | 0, 2 | 7.9 M/s (7.5–8.4) | 8.2 M/s (7.7–8.3) | 2.8 M/s (0.9–3.0) |
| 1,600 B | 0, 1 | 7.4 M/s (7.1–7.8) | 8.5 M/s (8.4–8.6) | 3.2 M/s (3.1–3.2) |

| Round trip, 16 B | CPUs | p50 | p99 | p99.9 |
|---|---|---|---|---|
| SPSC ring | 0, 2 | 340 ns | 490 ns | 1.9 µs |
| SPSC ring | 0, 1 | 320 ns | 490 ns | 2.7 µs |
| ring, no cached counters | 0, 2 | 170 ns | 480 ns | 1.8 µs |
| ring, no cached counters | 0, 1 | 320 ns | 490 ns | 9.2 µs |
| mutex + condvar | 0, 2 | 101 µs | 221 µs | 506 µs |
| mutex + condvar | 0, 1 | 103 µs | 230 µs | 591 µs |

- The mutex queue is ~4x slower under WSL2 than in the container (~100 µs per round trip):
  its futex wake-up costs more there. The ring makes no system call, so it is faster on the
  laptop (~0.33 µs), and the gap grows to ~300x.
- Cached counters: 1.4x the throughput of 16-byte messages, nothing on the round trip. In
  ping-pong the ring holds at most one message, so the cached copy of the other side's
  counter is always stale and gets reloaded anyway; the cache pays off only when a backlog
  builds, which is when the edge host needs throughput. (The uncached ring's 170 ns median at
  CPUs 0 and 2 is one run; at 0 and 1 it matches the cached ring.)
- Hyperthread siblings and separate cores measure the same within noise. The six cores of
  the 7535HS share one L3, so a cache line moves between cores quickly; thread placement
  should matter more on a machine with several L3 domains.

## The edge host

```
 network thread (Asio)                                    analytics threads (N)
  hsms::Session + gem::Host
  S6F11 decoded in place ──map copied once──► SpscRing<WaferSlot> ──► extract signals, classify
                                              (doorbell: spin, then futex)        │
  DecisionRule (per lot, k of last W) ◄─eventfd─ SpscRing<Verdict> ◄───────────────┘
  → S2F41 HOLD
```

- **Network thread.** `pipeline::EdgeCore` does the network thread's work without I/O: GEM
  host events in, wafers out to the workers, verdicts back, the decision, the HOLD.
  `pipeline::EdgeHost` runs it on an active `hsms::Session`. A wafer report is decoded in the
  receive buffer (no copy) and its map copied once, into a free slot of the next worker's
  ring (round robin). Slots are fixed-size: a lot id of up to 32 characters and up to 64 × 64
  bins. Larger maps and longer ids are counted and skipped, never truncated.
- **Analytics threads.** `pipeline::Analyzer` runs `SignalExtractor` and the fitted
  `RuleClassifier` on the slot in place, pushes a verdict, and signals the network thread.
  With nothing to do it spins for `--spin-us` (50 µs by default), then sleeps on a
  `pipeline::Doorbell`.
- **Waking threads without losing a wake-up.** The doorbell is an epoch to `atomic::wait` on
  and a "sleeping" flag. The consumer announces itself (an exchange), rechecks the ring, and
  only then sleeps. The producer, after each push, exchanges the flag back and makes the
  futex call only if a sleeper had announced itself. Because both sides use a read-modify-
  write, one of them always sees the other: either the producer sees the sleeper, or the
  sleeper's recheck sees the push. The network thread sleeps in epoll instead, so verdicts
  reach it through an eventfd that Asio reads. A `pending` flag turns a burst of verdicts into
  one `write()`.
- **Decision rule.** A lot is held once k of its last W verdicts show the same pattern other
  than `none` (`--k 3 --window 5` by default). It runs on the network thread, the one place
  where every worker's verdicts meet. Its state is a fixed table of the 64 most recently seen
  lots.
- **Full rings.** If every worker's ring is full (256 wafers each), the wafer is dropped and
  counted. The S6F12 has already gone back, and stalling the network thread would stall the
  tool.

### What the tests check

- `tests/test_edge.cpp`:
  - The decision rule: k in a window, `none` never counts, different patterns don't add up,
    the window slides, lots are independent, the least recently seen lot is forgotten.
  - The doorbell: 200,000 messages, the consumer made to sleep thousands of times, and a
    watchdog that counts lost wake-ups (none). With a relaxed load put in on purpose instead
    of the producer's exchange, it lost a wake-up in 1 of 5 runs: a race that tests catch
    only sometimes, which is why the design rests on the RMW argument.
  - The analyzer, on the caller's thread and on its own thread.
  - The edge core through GEM on a fake clock: the HOLD reaches the tool and its HCACK
    returns; oversized maps, long lot ids and full rings are counted.
- `tests/test_edge_host.cpp`: the closed loop over TCP against the tool emulator with two
  workers and k = 1. The set of lots held must equal the set of lots in which the classifier,
  run directly on the file, finds a pattern. Every hold must be answered with HCACK 0, and its
  timestamps must be in order.
- `waferedge-alloc-tests "[edge]"`: 100 rounds of report → classify → decide → HOLD → HCACK
  on one thread, with zero allocations once warm.

All of them run under TSan and ASan in CI.

The closed-loop test found two ordering bugs, both fixed:
- **Timestamps.** A single `now`, read before draining, could precede a verdict pushed during
  the drain. Stamps are now read when each step happens.
- **The end of a replay.** The host acknowledges a report before analysing it. The emulator
  stops once every S6F12 is in, so it can leave before the last lot's HOLD arrives. A real
  tool keeps running. The test stops the emulator only once the edge host has settled, and
  the emulator now lingers after its last answer (`ToolConfig::linger`, `--linger`, 1 s by
  default) so the last lots' HOLDs still land.

### Run it

```sh
build/release/tools/tool-emulator tests/data/waferlens_sample.wmap --port 5000 --rate 20 &
build/release/tools/waferedge-edge --port 5000 --k 1 --window 1 --workers 2
```

On the sample (24 wafers, three per WaferLens class) with k = 1, the edge host holds 11 lots
and the emulator withholds 10 wafers; all 11 holds are answered (the 11th lot had no wafers
left to withhold). The edge host prints the median time of each stage for the answered holds (received → analysed →
decided → S2F41 sent → S2F42). These are only ~10 samples, at 20 wafers/s, where every wafer
finds the threads asleep, so wake-ups dominate. In the cloud container, received → analysed
(waking the worker and classifying a 40 × 40 map) took ~0.2–0.3 ms with the worker asleep,
and ~0.13 ms with the worker spinning through the 50 ms gap (`--spin-us 100000`). Phase 4's
latency item measures this properly, with histograms from the tool's schedule. The laptop
showed the same picture: received → analysed 232 µs, → decided 91 µs, → S2F41 sent 3 µs,
→ S2F42 419 µs (medians of 10 holds, 2 workers, default spin).
`--cpus NET,W1,...` pins the network thread and the workers.

## Latency: sort result to hold

`build/release/bench/bench-latency` runs the tool emulator and the edge host in one process:
the emulator on one thread, the edge host's network thread on another, the workers on their
own. All of them read the same steady clock, and the two sides talk HSMS over TCP on
127.0.0.1 (ADR-0015). Latency is measured from each wafer's **scheduled** time in the
emulator's open-loop schedule, so a stall anywhere shows in the wafers queued behind it.
Percentiles come from `pipeline::LatencyHistogram`: log-linear buckets, never below the true
value and at most 1/128 above it.

- **Decision**: scheduled sort time → the network thread has the wafer's verdict, i.e. the
  moment a HOLD would go out. Every analysed wafer gives a sample.
- **HOLD**: for each lot held, the scheduled time of the wafer that completed the rule → the
  tool receiving the S2F41.
- **Dropped**: wafers that found every ring full. They have no verdict and so no latency;
  they are counted, never hidden.

Synthetic 40 × 40 maps (lots of 25; every tenth lot gets a centre blob from its 6th wafer,
held after 3 of 5), at least 3 s and 2,000 wafers per rate, default worker spin (50 µs).
**4-thread cloud container, not the laptop**:

| Wafers/s | Workers | Dropped | Decision p50 | p99 | p99.9 | max | HOLD p50 | HOLD max |
|---|---|---|---|---|---|---|---|---|
| 100 | 1 | 0 | 582 µs | 3.4 ms | 15.1 ms | 25.0 ms | 721 µs | 2.8 ms |
| 1,000 | 1 | 0 | 387 µs | 23.7 ms | 36.4 ms | 38.2 ms | 475 µs | 1.5 ms |
| 5,000 | 1 | 0 | 107 µs | 39.8 ms | 51.9 ms | 54.1 ms | 157 µs | 54.4 ms |
| 10,000 | 1 | 0 | 108 µs | 2.3 ms | 4.8 ms | 5.8 ms | 156 µs | 4.1 ms |
| 20,000 | 1 | 0 | 190 µs | 9.0 ms | 12.2 ms | 13.3 ms | 276 µs | 12.2 ms |
| 40,000 | 1 | 15,755 of 80,000 | 8.0 ms | 10.4 ms | 13.0 ms | 15.6 ms | 8.0 ms | 15.8 ms |
| 40,000 | 2 | 0 | 119 µs | 2.2 ms | 4.0 ms | 4.8 ms | 176 µs | 2.9 ms |
| 80,000 | 2 | 30,145 of 160,000 | 7.7 ms | 12.8 ms | 21.6 ms | 22.1 ms | 7.6 ms | 15.3 ms |

Where the time goes (p50 / p99):

| Wafers/s | Workers | Emulator late | Scheduled → received | Received → analysed | Analysed → decided |
|---|---|---|---|---|---|
| 100 | 1 | 138 µs / 1.4 ms | 324 µs / 2.6 ms | 177 µs / 721 µs | 60 µs / 264 µs |
| 100, spinning workers | 1 | 154 µs / 2.8 ms | 356 µs / 5.5 ms | 32 µs / 208 µs | 30 µs / 226 µs |
| 10,000 | 1 | 23 µs / 161 µs | 55 µs / 602 µs | 37 µs / 1.6 ms | 15 µs / 51 µs |
| 40,000 | 1 | 25 µs / 217 µs | 70 µs / 741 µs | 7.9 ms / 10.0 ms | 15 µs / 259 µs |

What the numbers say:
- **At low rates, waking threads is the cost.** At 100 wafers/s every thread sleeps between
  wafers. With the worker asleep, received → analysed is 177 µs. With the worker spinning
  through the 10 ms gaps (`--spin-us 20000`) it is 32 µs, about what classifying the map
  costs. The ~145 µs difference is the wake-up. The rest of the median is the emulator and
  the network thread waking from epoll. Busy rates are *faster* at the median (~0.1 ms at
  5,000–10,000 wafers/s) because nothing has time to fall asleep.
- **Overload turns into drops, not an unbounded queue.** One worker classifies ~32,000 of
  these maps per second. At 40,000/s its ring fills: 20% of the wafers are dropped, and the
  others wait at most a full ring, 256 wafers × ~31 µs ≈ 8 ms (Little's law). That is where
  the median jumps to. A second worker doubles the capacity (40,000/s with no drops; drops
  again at 80,000/s). Dropped wafers are the price of keeping the network thread and the
  tool's transactions on time (ADR-0014); the next step is to size workers so a real tool
  never gets near this.
- **The tails here are the container's.** The p99 of 24–40 ms at 1,000–5,000/s follows the
  emulator's own lateness (p99 up to 15 ms): four virtual CPUs shared by the emulator, the
  network thread, a spinning worker and the OS. The same rows on the laptop, with 12
  hardware threads, are the numbers to quote.
- **A late hold can still be too late for its lot.** At 40,000 wafers/s a lot of 25 takes
  0.6 ms to sort, less than the HOLD's latency, so the hold reaches the tool after the lot is
  done and withholds nothing. A real tool sorts a wafer every minute or so, which leaves
  four to five orders of magnitude of slack. The closed-loop result will count withheld wafers
  at realistic rates.

On real maps, `--maps data/waferlens_demo.wmap` replays the WaferLens demo set (24,090
wafers, exported as in docs/data.md) instead of the synthetic ones.

### On the laptop

Ryzen 5 7535HS (6 cores / 12 threads), WSL2, release build, threads not pinned, same
commands:

| Maps | Wafers/s | Workers | Dropped | Decision p50 | p99 | p99.9 | HOLD p50 | Emulator late p99 |
|---|---|---|---|---|---|---|---|---|
| synthetic | 100 | 1 | 0 | 532 µs | **1.13 s** | 1.30 s | 717 µs | **1.12 s** |
| synthetic | 1,000 | 1 | 0 | 338 µs | 680 µs | 1.4 ms | 487 µs | 188 µs |
| synthetic | 5,000 | 1 | 0 | 258 µs | 2.3 ms | 10.7 ms | 379 µs | 226 µs |
| synthetic | 10,000 | 1 | **12,866** | 299 µs | **1.48 s** | 1.48 s | 4.1 ms | **1.46 s** |
| synthetic | 20,000 | 1 | 580 | 6.5 ms | 16.6 ms | 18.1 ms | 6.5 ms | 510 µs |
| synthetic | 40,000 | 2 | 4,045 | 11.0 ms | 20.8 ms | 21.5 ms | 10.3 ms | 324 µs |
| synthetic | 80,000 | 2 | 85,626 | 13.2 ms | 20.6 ms | 22.7 ms | 12.5 ms | 1.5 ms |
| WaferLens demo | 100 | 1 | 0 | 528 µs | 1.1 ms | 3.6 ms | (no hold) | 356 µs |
| WaferLens demo | 1,000 | 1 | 0 | 375 µs | 827 µs | 4.1 ms | 639 µs | 240 µs |
| WaferLens demo | 10,000 | 1 | 0 | 309 µs | 15.2 ms | 16.1 ms | 532 µs | 208 µs |

- **Two runs hit a stall of over a second, in the whole process.** In the bold rows the
  emulator's *own* lateness has the same p99 as the total, and the HSMS window filled up
  (64 reports unanswered; "window full" 9 and 4,383 times). The tool thread and the edge
  host's network thread stopped together for 1.1–1.5 s, which points at the process or the
  WSL2 VM being paused, not at the pipeline. At 10,000/s the open-loop schedule shows what
  follows: ~14,000 overdue wafers go out at once, flood the rings, and 12,866 are dropped.
  A closed-loop benchmark would have hidden the stall. It didn't come back: six more runs at
  100 and 10,000/s had p99 of 0.9–9.8 ms, and `bench-emulator` alone was clean up to
  50,000/s. So it was a rare pause of the process or the VM (WSL2, Windows), not something
  the pipeline does. That is also why each benchmark row is worth repeating before quoting.
- **Without the stalls the laptop is steadier than the container:** p99 of 0.7–2.3 ms at
  1,000–5,000/s against 24–40 ms there, because the emulator's lateness stays below
  0.25 ms at p99.
- **Its worker saturates sooner:** about 20,000 synthetic maps/s for one worker (container:
  ~32,000), and two workers already drop at 40,000/s. Not because of core sharing: pinned to
  separate cores (`--cpus 0,2,4`), one worker still dropped 2,289 wafers at 20,000/s. It is
  the laptop's per-core speed at this work: signals + rules ran at ~14,000 real WaferLens
  maps/s on one core (docs/evaluation.md), the Hough transform the largest part.
- **Real WaferLens maps** hold 10,000 wafers/s on one worker without drops; the p99 of 15 ms
  is the ring backing up near the worker's capacity. At 100/s the first 2,000 wafers in
  `tested_at` order contain no excursion that trips 3 of 5, so no hold.

## Reproduce

```sh
cmake --workflow --preset release
build/dev/tests/waferedge-tests "[emulator]"         # Replay on a fake clock, Tool over TCP
build/release/bench/bench-emulator                   # the emulator's rate and lateness
build/dev/tests/waferedge-tests "[ring]"             # the ring's tests (and under tsan)
cmake --workflow --preset tsan                       # every test under ThreadSanitizer
build/release/bench/bench-ring --cpus 0 2              # the ring vs a mutex queue, pinned
build/dev/tests/waferedge-tests "[decision],[doorbell],[analyzer],[edge]"
build/dev/tests/waferedge-alloc-tests "[edge]"       # the edge hot path allocates nothing
build/dev/tests/waferedge-tests "[histogram]"        # the latency histogram
build/release/bench/bench-latency                    # sort -> decision / HOLD by load
build/release/bench/bench-latency --workers 2 --rates 40000,80000 --seconds 2
build/release/tools/tool-emulator tests/data/waferlens_sample.wmap --port 5000 --rate 5 &
python3 -m venv /tmp/secsgem-venv && /tmp/secsgem-venv/bin/pip install secsgem==0.3.0
/tmp/secsgem-venv/bin/python tools/secsgem_host.py --port 5000 --hold LOT-000281
```

The full WaferLens demo set (24,090 wafers) replays the same way once exported
(`python3 tools/export_maps.py waferlens ~/waferLens/data/demo data/waferlens_demo.wmap`).
