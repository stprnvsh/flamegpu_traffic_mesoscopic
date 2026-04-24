# Project Tracker

## 2026-04-24
- Added `sumo/src/mesosim/MESOSIM_RUNTIME_CALL_FLOW.md`.
- Scope: mesoscopic runtime function-call mapping (entry, init, per-step, movement, queue/junction, triggers, outputs, known gaps).
- Added `src/PYTHON_RUNTIME_CALL_FLOW.md`.
- Scope: Python runtime function-call mapping for `src/core`, `src/input`, and `src/traffic_models` (entry/init, per-step layers, messages, parsing integration, output hooks, known gaps).
- Added `docs/FLOW_IMPLEMENTATION_DIFF_REPORT.md`.
- Added `docs/FLOW_DIFF_CHECKLIST.md`.
- Scope: implementation difference analysis and validation checklist between Python runtime flow and SUMO mesosim flow docs.
- Added `docs/PYTHON_RUNTIME_ACCURACY_UPGRADE.md`.
- Scope: phased implementation roadmap for improving Python FLAMEGPU runtime accuracy toward SUMO mesosim behavior.
- Updated runtime implementation for phased upgrade in `src/core/messages.py`, `src/core/agents.py`, `src/core/model.py`, `src/core/simulation.py`, and `src/input/sumo_parser.py`.
- Scope: deferred admission + due-time packet scheduling, minimal junction gate, teleport reason tagging, connection/movement parsing, route connectivity validation, calibrator-lite hook, detector-view projection, and runtime KPI outputs.
- Implemented `sumo-accuracy-flamegpu-speed-plan` in strict phase order across `src/core/messages.py`, `src/core/agents.py`, `src/core/simulation.py`, `src/input/sumo_parser.py`, and `src/traffic_models/fundamental_diagram.py`.
- Scope: bucketized signal hot path, TLS segment-id alignment, movement multiplicity preservation, inter-edge legality rejection tagging, BFS OD route expansion, priority rank mapping from parsed edge metadata, and KPI regression gate helper.
- Implemented `sumo-full-conflict-event-plan` across `src/core/messages.py`, `src/core/agents.py`, `src/core/model.py`, `src/core/simulation.py`, `src/input/sumo_parser.py`, plus `tests/parity_runner.py` and `examples/parity_benchmark.py`.
- Scope: movement-level runtime/parser containers, due-time action/event fields, deterministic contested-segment arbitration path, movement legality timing semantics, SUMO-aligned jam/headway defaults, and parity/throughput smoke harness.
- Implemented `flamegpu-reroute-teleport-parity` phases in runtime and parity harness.
- Scope: teleport state machine fields, single-jump/multi-step delayed retry behavior, reroute eligibility during teleport, disconnected cooldown retry policy, visible teleport lifecycle metrics, and three-scenario parity gate helper.
