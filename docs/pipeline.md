# The edge pipeline

Phase 4 puts the pieces together into a closed loop: a tool emulator sorts wafers and reports
them over HSMS / SECS-II, the edge host finds spatial signatures and holds the lot, and the
time from sort to hold is measured. This page grows with the phase; so far it covers the
**tool emulator** (`include/waferedge/emulator/`, `src/emulator/`, `tools/tool_emulator.cpp`;
ADR-0012) and the **SPSC ring** the edge host's threads will hand wafers over with
(`include/waferedge/pipeline/spsc_ring.hpp`; ADR-0013).

```
 tool-emulator (passive HSMS, GEM equipment)          edge host (phase 4, next)
   Replay: constant-rate schedule ──S6F11 wafer report──►  decode → detect → decide
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
- It stops once every wafer is reported and answered (S6F12, or a failed transaction), or on
  Ctrl-C.

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
  ring's runs range from 10.7 to 25.1 M/s): these virtual CPUs may share caches. The laptop's
  separate per-core L2 caches should show the effect more clearly.
- At 1,600 bytes every queue is bound by copying the map; the in-place API removes that copy.

## Reproduce

```sh
cmake --workflow --preset release
build/dev/tests/waferedge-tests "[emulator]"         # Replay on a fake clock, Tool over TCP
build/release/bench/bench-emulator                   # the emulator's rate and lateness
build/dev/tests/waferedge-tests "[ring]"             # the ring's tests (and under tsan)
cmake --workflow --preset tsan                       # every test under ThreadSanitizer
build/release/bench/bench-ring                       # the ring vs a mutex queue
build/release/tools/tool-emulator tests/data/waferlens_sample.wmap --port 5000 --rate 5 &
python3 -m venv /tmp/secsgem-venv && /tmp/secsgem-venv/bin/pip install secsgem==0.3.0
/tmp/secsgem-venv/bin/python tools/secsgem_host.py --port 5000 --hold LOT-000281
```

The full WaferLens demo set (24,090 wafers) replays the same way once exported
(`python3 tools/export_maps.py waferlens ~/waferLens/data/demo data/waferlens_demo.wmap`).
