# ADR-0005: Explainable rule classifier next to the CNN, thresholds fitted on train only

- **Status:** Accepted
- **Date:** 2026-10-09

## Context
WaferEdge needs a pattern call per wafer, at the tool, fast enough to hold a lot before the
next wafer is processed. FabEye's CNN already classifies wafer maps (macro-F1 0.858 on
WM-811K's lot-disjoint test set) and will run on WaferEdge's own inference engine in phase 2b.
The question for phase 1 is whether classical detectors (zones, sectors, clusters, a Hough
line, a randomness test) are worth keeping next to it, and in what form.

Constraints: an equipment engineer must be able to see why a lot was held; the detectors run
on every wafer, so they must be cheap and allocation-free; the evaluation must be honest
(thresholds never tuned on the data they are scored on).

## Options considered
1. **CNN only**: the most accurate; opaque, needs the GPU engine or ~400 maps/s on a CPU core.
2. **A learned model on the detector signals** (gradient-boosted trees, logistic regression):
   likely more accurate than hand rules, but its decisions are as hard to read as the CNN's
   and it adds a training dependency to the C++ side.
3. **An ordered decision list with hand-designed structure and fitted thresholds**: each
   rule names a pattern and 1–3 conditions on named signals ("edge ratio ≥ t, sector ratio ≤
   t, join-count z ≥ t"); the first rule that holds wins.

## Decision
Option 3, alongside the CNN rather than instead of it.

- **Structure by hand, thresholds by data.** The rule order and each condition's signal and
  direction come from what the detectors are for (docs/signatures.md); only the thresholds are
  fitted, by coordinate ascent on macro-F1, candidates at quantiles of each signal.
- **Rules are data.** One table drives classification, `explain()` (the rule that fired, each
  condition with its value and threshold), fitting and the thresholds file
  `config/rules.txt` (shortest round-trip doubles, so the file reproduces the fit exactly).
- **Fitted on WM-811K train only**, FabEye's lot-disjoint split, so the CNN and the rules
  learn from the same maps. WaferLens is never used for fitting: it is an out-of-distribution
  test (simulated maps).
- **'none' is weighted back to its natural share during fitting.** FabEye capped train 'none'
  at 10,000 maps (for CNN training); val and test keep the natural ~85%. Fitted on the capped
  set, macro-F1 rewards thresholds that call borderline wafers defective, which floods the
  test set with false alarms. Each train 'none' map counts as 103,342 / 10,000 = 10.33 maps:
  WM-811K has 147,431 labelled 'none' maps, 44,089 of them in val and test, so the uncapped
  train split held 103,342. On validation this lifts macro-F1 from 0.631 to 0.672.
  Disclosure: this change was made after the unweighted fit had been scored on the test set
  once (0.614); the weight uses only dataset counts, and validation alone shows the same gain,
  but the decision was prompted by a test-set look. Development since uses validation only.

## Consequences
- Rules reach macro-F1 0.655 on WM-811K test against the CNN's 0.858, and 0.645 against 0.909
  on WaferLens (docs/evaluation.md): the CNN is clearly better, and that result is kept.
- What the rules give in exchange: every call explains itself in one line (which the phase-6
  incident note can quote), they run at ~13–14k maps/s on one core including the Hough
  transform, and they need no GPU and no training framework at run time.
- In the pipeline (phase 4) the rules are a fast, explainable first opinion; the CNN on the
  GPU engine is the accurate one; agreement between the two is reported.
- If a learned model on the same signals is wanted later (option 2), the signals CSV from
  `waferedge-classify eval --csv` is its training data; it would be a new ADR.
