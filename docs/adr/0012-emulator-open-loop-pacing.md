# ADR-0012: The tool emulator paces wafers open-loop, on a constant-rate schedule

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
Phase 4 measures how long a wafer takes from sort to lot hold, and how that latency grows
with load. The tool emulator is the load generator: it replays wafer maps as S6F11 reports.
How it decides *when* to send the next wafer decides whether the measurement is honest. A
real tool sorts wafers at its own pace, whether or not the host has caught up; a host that
falls behind doesn't slow the tool down, it just reacts later.

## Options considered
1. **Closed loop**: send the next wafer when the previous one is acknowledged (as the HSMS
   round-trip benchmark does). Simple, and it never overloads the host, because a slow host
   slows the generator. That is coordinated omission: when the host stalls, the wafers that
   would have queued behind the stall are never sent, so the latency samples never contain
   the stall. Percentiles come out far better than what a tool would see.
2. **Open loop, constant rate**: wafer k is due at `start + k / rate`. A sender that falls
   behind catches up: the wafers go out back to back, each recorded with its *scheduled*
   time, and latency is measured from that time. A stall then shows as late wafers, as it
   would on a real tool.
3. **Open loop, Poisson arrivals**: random inter-arrival times at the same mean rate, closer
   to a fab with many tools, harder to reason about for a single tool.

## Decision
Option 2. `emulator::Replay` keeps the schedule and records each wafer's scheduled and actual
send times; latency in phase 4 is measured from the scheduled time. The schedule starts when
the tool goes ON-LINE and is re-based (not caught up) after the tool was off-line or the link
was down, as a real tool doesn't sort while it can't report.

A held lot's remaining wafers are skipped, counted as withheld, and take no slot: the tool
sorts the next lot's wafer in that time. When HSMS's transaction window is full (64 reports
awaiting S6F12), the wafer waits for a reply and goes out late; its lateness shows the
backpressure instead of hiding it.

## Consequences
- The emulator's own lateness (actual minus scheduled) is a floor under every latency
  measured with it. `bench-emulator` measures it: in the 4-thread cloud container, median
  20–120 µs, p99 0.2–0.3 ms (1.2 ms at 200,000 wafers/s), max ~3 ms; the achieved rate
  matches the target up to 200,000 wafers/s. The laptop's numbers replace these.
- Overload is visible: a host that can't keep up makes wafers later and later, rather than
  making the generator slower.
- Writing `bench-emulator` exposed a bug the closed-loop tests could not: a full transaction
  window was reported as a dead link and stopped the schedule for good. `gem::GemError::busy`
  now separates the two.
- Poisson arrivals (option 3) remain possible later as another schedule in `Replay`; nothing
  else depends on the spacing being constant.
