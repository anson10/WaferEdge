# ADR-0011: A GEM subset with a predefined report, sans-I/O over a Link

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
GEM (SEMI E30) is the behaviour layer above SECS-II and HSMS: which messages a tool serves,
and when. Its communication state machine (S1F13 establish communication, with a retry
delay) and control state machine (off-line / on-line, local / remote) decide whether the
tool will execute the edge host's lot hold at all. WaferEdge needs one event (a wafer was
sorted, with its map), one command (HOLD / RELEASE a lot), alarms, and the error replies a
real tool sends. The full standard has hundreds of messages and capabilities (dynamic event
reports, process programs, spooling, limits monitoring, ...). Like HSMS (ADR-0010), the
state machines must be testable on a fake clock, and the hot path (report → ack → hold →
answer) must not allocate.

## Options considered
1. **Full dynamic data collection**: the host defines reports (S2F33), links them to events
   (S2F35) and enables them (S2F37); every S6F11 is decoded against that definition. What
   real fabs do, and a large state machine with its own storage and failure modes.
2. **One predefined report**: CEID 100 "wafer sorted" always carries report 10, a fixed
   list (LOTID, WAFERID, ROWS, COLS, BINS). Pre-configured tools ship like this; the host
   decodes it with a fixed layout, as views.
3. For commands: the GEM layer accepts them and replies itself (HCACK 4, "will finish later",
   then an event on completion), or hands each command to the app and lets it choose the HCACK.

For structure: GEM inside the HSMS handler, or GEM endpoints that take HSMS events and send
through an interface, with no I/O of their own.

## Decision
Option 2, and the app answers each command. `gem::Equipment` and `gem::Host` derive from
`gem::Endpoint`, which holds the communication state machine, a fixed queue of GEM events
and a reused body buffer. They take HSMS events (`on_hsms`) and the time (`tick`), and
send through `gem::Link`, an interface with two implementations: `ProtocolLink` over a
fake-clock `hsms::Protocol` (tests) and `SessionLink` over a TCP `hsms::Session`.

The equipment answers by itself everything the standard decides (S1F13/S1F14, S1F1 off-line
→ S1F0, S1F15/S1F17, HCACK 1 for an unknown command, 2 in ON-LINE LOCAL, 3 for a missing or
non-ASCII LOTID, S9F3 / S9F5 / S9F7 for unknown streams, unknown functions and malformed
bodies, S9F9 on its own T3 timeouts). It passes only valid HOLD / RELEASE commands in
ON-LINE REMOTE to the app, which answers with `Equipment::answer` (done, already in that
condition, no such lot).

## Consequences
- The wafer report decodes in place: the map reaches the detectors as a `WaferMapView` over
  the HSMS receive buffer. Report, ack, hold and answer allocate nothing once warm
  (`tests/test_gem_alloc.cpp`).
- Both state machines are tested on a fake clock, including the retry after a denied or
  unanswered S1F13, and over TCP once end to end.
- A host that expects to define its own reports (S2F33/35/37) gets S9F5 (unknown function)
  from our equipment. A real tool in phase 4 would need option 1; the emulator and the edge
  host are both ours, so the predefined report is enough for the closed loop.
- If the app never answers a LotCommandReceived, the host's T3 expires: the contract is
  documented and the emulator (phase 4) answers every command.
- Out of scope: spooling (S6F23), process programs (S7), terminal services (S10), limits
  monitoring, equipment constants and status variables (S1F3, S2F13), clock (S2F17/31).
