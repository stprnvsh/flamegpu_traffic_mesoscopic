# Python Runtime Accuracy Upgrade (Toward SUMO Mesosim)

Based on:
- `src/PYTHON_RUNTIME_CALL_FLOW.md`
- `sumo/src/mesosim/MESOSIM_RUNTIME_CALL_FLOW.md`
- `docs/FLOW_IMPLEMENTATION_DIFF_REPORT.md`

This document defines what to implement in the Python FLAMEGPU runtime to improve traffic-simulation accuracy with minimal architecture changes.

## 1) Current Accuracy Gaps (from diff report)

- High impact:
  - Event-driven timing vs fixed per-step retry behavior
  - Simplified junction/right-of-way gating
  - Simplified queue admission (`accept/reject` only)
  - Teleport as hard death threshold only
- Medium impact:
  - No calibrator-like trigger subsystem
  - No detector-style output projection (only interval edge parquet)
  - Parser connection/signal movement extraction is incomplete

## 2) Implement Now (P1/P2)

## P1: Queue Admission + Event-Time Semantics
Target files:
- `src/core/messages.py`
- `src/core/agents.py`
- `src/core/model.py`

Implement:
- Extend `entry_accept` message with:
  - `admit_status` (`ACCEPT`, `DEFER`, `REJECT`)
  - `retry_time` (float)
  - optional `reason_code`
- Add packet vars:
  - `next_retry_time`
  - `next_event_time`
- Update RTC functions:
  - `process_edge_requests`: send deferred response instead of silent reject
  - `wait_for_entry`: honor deferred retry time
  - `move_and_request`: suppress re-request until retry/event time is due
- Keep current layer order (`L0..L9`), emulate event-driven timing with per-packet due times.

Acceptance criteria:
- Under saturation, request spam is reduced (no unconditional per-step resend).
- Deferred entries occur at/after computed retry times.
- Normal uncongested behavior remains unchanged.

## P2: Junction Gate + Teleport Semantics
Target files:
- `src/core/agents.py`
- `src/core/simulation.py`

Implement:
- Add minimal junction gating metadata:
  - `priority_rank`
  - `junction_block_until`
  - optional `conflict_group`
- Enforce gate in `process_edge_requests` before acceptance.
- Replace single teleport death branch with reasoned outcomes:
  - `jam`
  - `disconnected`
  - `route_end`
- Add counters in results (`_collect_results`) for teleport reasons.

Acceptance criteria:
- Contested junction scenarios show deterministic priority behavior.
- Teleport outcomes are reason-classified and visible in results.
- Current default behavior preserved when new controls are disabled.

## 3) Implement Next (P3/P4)

## P3: Parser Connectivity + TLS Movement Mapping
Target files:
- `src/input/sumo_parser.py`
- `src/core/simulation.py`

Implement:
- Parse richer connection data (`from`, `to`, `tl`, `linkIndex`, lane/movement metadata if present).
- Replace `_extract_green_edges` placeholder with connection-backed movement extraction.
- Validate trip/flow route connectivity; mark disconnected demand explicitly.

Acceptance criteria:
- Signal phases produce real green movement mappings when network data supports it.
- Disconnected routes are detected and reported (not silently treated as normal).

## P4: Trigger/Output Parity Layer (minimal)
Target files:
- `src/core/simulation.py`
- `src/core/metrics.py`
- `src/core/model.py`

Implement:
- Optional host-side calibrator-like function for interval target inflow correction.
- Detector-view projection from existing interval metrics (no second metrics pipeline).

Acceptance criteria:
- Calibrator-like correction can be toggled on/off by config.
- Detector-view output matches edge interval metrics for mapped edges.

## 4) File-by-File Change Map

- `src/core/messages.py`
  - extend `entry_accept` schema (`admit_status`, `retry_time`, `reason_code`)
- `src/core/agents.py`
  - packet vars: `next_retry_time`, `next_event_time`, teleport reason fields
  - edge vars: minimal junction gating metadata
  - RTC updates: `move_and_request`, `wait_for_entry`, `process_edge_requests`
- `src/core/model.py`
  - keep existing layers; only add minimal optional layer if junction reservation bus is needed
- `src/core/simulation.py`
  - new config toggles
  - teleport counters in results
  - optional calibrator hook registration
- `src/input/sumo_parser.py`
  - connection detail extraction + TLS movement mapping + connectivity validation
- `src/core/metrics.py`
  - detector-view projection and parity output adapter

## 5) Validation Checklist (what to run)

Use and extend `docs/FLOW_DIFF_CHECKLIST.md`:
- interval reset correctness
- per-step order sanity
- saturation retry timing behavior
- contested junction behavior
- teleport reason distribution
- route connectivity validation
- detector-view parity against interval metrics

## 6) Explicit Non-Goals (for this upgrade cycle)

- No full event-queue architecture rewrite.
- No full SUMO conflict engine replication (`MSLink::opened` depth).
- No full SUMO calibrator internals; only minimal host-side parity layer.
- No assumption of unavailable upstream SUMO source files beyond documented behavior.
