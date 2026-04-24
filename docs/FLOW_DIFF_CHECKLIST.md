# Flow Difference Validation Checklist

## 1) Interval Boundary Correctness
- Run short simulation with small metrics interval.
- Verify interval reset happens once at boundary (`reset_interval_counters` set, then cleared next step).
- Confirm `entered/left/sampledSeconds` do not bleed across intervals.

## 2) Per-Step Ordering Sanity
- Instrument/log timestamps around:
  - move/request
  - departure processing
  - request processing
  - wait/accept transition
- Confirm observed order matches `L0..L9` declaration.

## 3) Queue Saturation Behavior
- Create saturated downstream segment.
- Verify Python behavior: request retries each step.
- Compare expected SUMO behavior from doc: deferred admissible time / full recheck behavior.

## 4) Junction Conflict Behavior
- Create contested junction scenario (competing movements).
- Verify Python behavior is primarily signal-gate + queue constraints.
- Mark as divergence if conflict-priority behavior expected from SUMO doc cannot be reproduced.

## 5) Teleport/Stuck Outcomes
- Create gridlock-like scenario.
- Verify Python: packet death after wait threshold.
- Compare against SUMO doc expectation: multiple teleport branches (remove/jump/multi-step).

## 6) Trigger/Output Parity
- Verify Python outputs only interval collector parquet outputs.
- Confirm no calibrator-equivalent trigger path is executed in Python runtime.
- Treat SUMO detector/calibrator comparisons as doc-level unless upstream SUMO source is available.

## 7) Confidence Tagging
- For each validated gap, tag:
  - `confirmed-local` (Python source + reproducible behavior)
  - `doc-confirmed-sumo` (in `MESOSIM_RUNTIME_CALL_FLOW.md`)
  - `needs-upstream-sumo-source` (function-level confirmation unavailable locally)
