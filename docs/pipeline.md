# The edge pipeline

Phase 4 puts the pieces together into a closed loop: a tool emulator sorts wafers and reports
them over HSMS / SECS-II, the edge host finds spatial signatures and holds the lot, and the
time from sort to hold is measured. This page grows with the phase; so far it covers the
**tool emulator**. Code: `include/waferedge/emulator/`, `src/emulator/`,
`tools/tool_emulator.cpp`; pacing decision: ADR-0012.

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

## Reproduce

```sh
cmake --workflow --preset release
build/dev/tests/waferedge-tests "[emulator]"         # Replay on a fake clock, Tool over TCP
build/release/bench/bench-emulator                   # the table above
build/release/tools/tool-emulator tests/data/waferlens_sample.wmap --port 5000 --rate 5 &
python3 -m venv /tmp/secsgem-venv && /tmp/secsgem-venv/bin/pip install secsgem==0.3.0
/tmp/secsgem-venv/bin/python tools/secsgem_host.py --port 5000 --hold LOT-000281
```

The full WaferLens demo set (24,090 wafers) replays the same way once exported
(`python3 tools/export_maps.py waferlens ~/waferLens/data/demo data/waferlens_demo.wmap`).
