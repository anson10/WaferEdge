# ADR-0018: An optional cross-lot hold rule next to the per-lot window

- **Status:** Accepted
- **Date:** 2026-10-11

## Context
The closed-loop replay (ADR-0017) held all 13 of WaferLens's spatial excursions. But after
an excursion's first patterned wafer reached sort, the default per-lot rule (3 of a lot's
last 5 wafers) took a median 8.1 h to hold, against 3.0 h for WaferLens's batch alarm. An
excursion upstream (a chamber, a reticle) marks wafers in many lots, a few per lot, so a
window inside one lot fills slowly. WaferLens's alarm counts the same pattern across all
lots: at least max(3, 3× background per day) wafers within 24 h.

The edge host sees every wafer a tool sorts, so it can count across lots as well.

## Options considered
1. **Lower k in the per-lot rule** (2 of 5): 2.5 h, but every lot needs two patterned wafers
   of its own, and a lot with one is never held.
2. **A cross-lot count per pattern:** hold a lot whose wafer shows pattern p once K wafers
   with p, from any lots, arrived within a time window H. Next to the per-lot rule; either
   rule can hold.
3. **WaferLens's background-scaled threshold** (3× the pattern's background rate per day):
   needs a background estimate per pattern and tool, i.e. history the edge host doesn't
   keep. With the rules detector the background is small (2–5 false holds in the replay), so
   a fixed K is close to it.

## Decision
Option 2, off by default: `DecisionConfig::tool_k` and `tool_window_us`.
- Per pattern, a ring of the last K arrival times (K ≤ 32), 9 patterns: 2.3 KB, sized once,
  nothing allocated per wafer (the edge allocation test still passes).
- The rule fires when the oldest of the last K arrivals is within H of the newest. It then
  holds the lot of the wafer that completed it, and later lots on their first wafer with that
  pattern while the count stays at K within H.
- Wafers of held lots still count: they are evidence that the excursion is on.
- Time: the replay uses the sort time (`tested_at`); the live edge host uses the time the
  network thread received the report, on the steady clock. With several workers, verdicts
  can arrive slightly out of order; the ring then compares by arrival order, which is off by
  at most the workers' skew (milliseconds against a window of hours).
- Options: `waferedge-closedloop --tool-k K --tool-window-h H`, `waferedge-edge --tool-k K
  --tool-window-s S`. `waferedge-closedloop --detector truth` runs the rules on the true
  patterns, which separates what the rule costs from what the detector's misses cost.

## Consequences
On WaferLens's demo set (docs/pipeline.md has the tables), in-line median from the first
patterned wafer at sort to the hold:

- With the rules detector: 3 of 5 alone 8.1 h; 3 of 5 plus 3 in 24 h 5.3 h; 2 of 5 plus
  3 in 24 h 2.0 h with 31 patterned wafers sorted before the holds (WaferLens: 3.0 h and 72).
  False holds go from 2 to 5 of 211.
- With the true patterns: 3 of 5 4.6 h; plus 3 in 24 h 3.2 h, close to WaferLens's 3.0 h
  with the same kind of rule. The rest of the gap with the rules detector (3.2 → 5.3 h) is
  the detector missing patterned wafers.
- The defaults don't change: the numbers come from one simulated fab, and picking K, H, k
  and W on them would tune the rule to it. A deployment picks them from its own false-hold
  budget.
- More lots are held: 211 instead of 122 holds, 4,076 instead of 1,771 wafers withheld. Each
  hold is a lot an engineer releases, so the hold count is reported next to the speed.
- In a live run with time compressed (the emulator at hundreds of wafers per second), a
  window of hours covers the whole run: the rule then behaves as "K wafers since start".
  The replay, on sort time, is where the window means what it says.
