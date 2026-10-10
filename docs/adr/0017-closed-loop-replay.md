# ADR-0017: The closed-loop result is a sort-time replay of the live decision rule

- **Status:** Accepted
- **Date:** 2026-10-10

## Context
The closed-loop result asks what WaferEdge's holds would have done to WaferLens's 13
spatial excursions. The comparison is with WaferLens's batch pattern alarm (caught 13 of 13,
median 52 h after the excursion began, of which ~45 h is sort lag) and with no hold at all.
WaferLens's demo set has 24,090 wafers sorted over weeks of fab time, with ground truth:
which excursion each patterned wafer shows, and when each excursion began. The live loop
(tool emulator → HSMS → edge host → S2F41) is measured in milliseconds (ADR-0015); a tool
sorts a wafer every few minutes.

## Options considered
1. **Run the live loop in real time:** weeks per run.
2. **Run the live loop with time compressed** (the emulator at thousands of wafers/s): the
   holds and verdicts stay the same, but a hold now lands several wafers later than it would
   in real time (milliseconds of latency against microseconds between wafers). It would
   undercount the wafers a hold withholds.
3. **Replay in sort-time order with the live loop's own parts:** the same `Detector` gives
   every wafer's verdict, and the same `DecisionRule` sees them in `tested_at` order. A hold
   lands at its triggering wafer's sort time plus the measured latency. Later wafers of the
   held lot are withheld, as the emulator withholds them.

## Decision
Option 3: `pipeline::replay_closed_loop` and the `waferedge-closedloop` tool.
- Inputs:
  - the `.wmap` file (maps, lots, sort times);
  - `tools/export_maps.py --excursions`, from WaferLens's ground truth: each spatial
    excursion's id, pattern and start, and which excursion each patterned wafer shows;
  - `config/waferlens_pattern_alarms.csv`: WaferLens's published alarm times per excursion,
    copied from its docs with the source named. WaferEdge compares against WaferLens's
    numbers rather than recomputing them.
- Per excursion, in hours after it began:
  - when its first patterned wafer reached sort (the floor for any sort-based monitor;
    printed next to WaferLens's own figure as a check);
  - WaferEdge's first hold credited to it;
  - WaferLens's alarm.

  Then how many of its patterned wafers were sorted before each, and with no hold (all).
- A hold is credited to the excursion that the triggering wafer shows. If that wafer shows
  none, it is a false hold. WaferLens credits an alarm to any excursion with the same pattern
  in its window, so overlapping excursions can be credited early (its id 22 alarmed at 0 h).
  WaferEdge's crediting is stricter.
- `--sweep` reruns the rule for several k-of-window settings on the same verdicts: holding
  sooner against holding wrongly.

## Consequences
- The replay is the live loop as long as a hold lands before the lot's next wafer is sorted.
  The tool prints the smallest gap between a lot's sorted wafers next to the latency it
  assumed. A test also runs the WaferLens sample both live (emulator + edge host over TCP)
  and replayed, and requires the same holds and the same withheld wafers.
- WaferEdge sits at sort, so it can't beat sort lag: for both monitors, the hours until the
  first patterned wafer reaches sort are the same. What it can shorten is the part after
  that, which is ~3 h (median) for WaferLens's batch alarm. The result reports both parts.
- A held lot's withheld wafers are the ones that tool doesn't sort. Holding a lot doesn't stop
  the chamber that caused the excursion: other lots keep arriving until someone acts on the
  hold (WaferLens's pattern-led commonality names the chamber). So the headline is the time
  to the first hold, and the patterned wafers sorted before it, not the withheld count.
- Every detector can be replayed (`--detector`): the CNN pairs from ADR-0016 included.
