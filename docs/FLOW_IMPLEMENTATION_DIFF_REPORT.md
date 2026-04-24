# Python vs SUMO Flow Implementation Difference Report

Compared:
- `src/PYTHON_RUNTIME_CALL_FLOW.md`
- `sumo/src/mesosim/MESOSIM_RUNTIME_CALL_FLOW.md`

SUMO source availability in this repo:
- `sumo/src/mesosim/MESOSIM_RUNTIME_CALL_FLOW.md` exists.
- Referenced SUMO `.cpp/.h` files do not exist locally, so those are marked `requires-upstream-file`.

## Diff Matrix (Category-First)

- **entry/init path**
  - Python: `MesoscopicSimulation.build_model/load_network/load_demand/initialize` -> `MesoscopicTrafficModel.build`.
  - SUMO: `main` -> `NLBuilder::init/build` -> `MSNet::simulate`.
  - Mismatch: `different mechanism`
  - Impact: `medium`

- **per-step scheduling model**
  - Python: fixed FLAMEGPU layers (`L0..L9`) in `src/core/model.py`.
  - SUMO: event-driven scheduler (`MELoop::simulate/checkCar/changeSegment`) with due-time queue.
  - Mismatch: `different mechanism`
  - Impact: `high`

- **movement transition logic**
  - Python: `move_and_request` -> `send_departure` -> `process_edge_requests` -> `wait_for_entry`.
  - SUMO: `changeSegment` calling `send/receive` with event-time returns.
  - Mismatch: `different mechanism`
  - Impact: `high`

- **queue admission/full handling**
  - Python: per-step accept/reject in `process_edge_requests`; retry by re-requesting each step.
  - SUMO: `hasSpaceFor` supports now/later/blocked semantics and recheck scheduling.
  - Mismatch: `simplified`
  - Impact: `high`

- **junction/right-of-way**
  - Python: signal gate (`green_signal` + `update_green_flag`) and queue checks.
  - SUMO: `mayProceed` + `isOpen` + `MSLink::opened` conflict logic.
  - Mismatch: `simplified`
  - Impact: `high`

- **teleport/gridlock**
  - Python: hard wait threshold death (`teleport_threshold` in packet RTC).
  - SUMO: `teleportVehicle` with multiple branches (remove/jump/multi-step).
  - Mismatch: `different mechanism`
  - Impact: `high`

- **rerouting**
  - Python: explicit `Packet.try_reroute` layer and `edge_status` messages.
  - SUMO doc: no dedicated per-step reroute phase in mesosim flow.
  - Mismatch: `different mechanism`
  - Impact: `medium`

- **insertion/removal lifecycle**
  - Python: host spawn function + packet death in RTC + final host result collection.
  - SUMO: `determineCandidates` -> `emitVehicles` -> `tryInsert` and `removePending`.
  - Mismatch: `different mechanism`
  - Impact: `medium`

- **events/triggers (calibrator)**
  - Python: no calibrator trigger subsystem in documented runtime.
  - SUMO: explicit `METriggeredCalibrator` construction/execution.
  - Mismatch: `missing`
  - Impact: `medium`

- **outputs/detectors/metrics timing**
  - Python: interval collector + parquet save (`IntervalEdgeDataCollector`).
  - SUMO: detector pipeline (`MEInductLoop` + `MSDetectorControl::writeOutput`).
  - Mismatch: `different mechanism`
  - Impact: `medium`

## Python Evidence (Direct Source)

- Initialization and host-step registration:
  - `src/core/simulation.py`: `initialize`, `self.model.build()`, `add_step_function(...)`, `create_simulation()`.
- Per-step loop and collector timing:
  - `src/core/simulation.py`: `run`, `IntervalEdgeDataCollector`, `self.simulation.step()`.
- Layer order:
  - `src/core/model.py`: `_define_layers`, `L0_reset_counters` through `L9_wait`.
- Teleport/death and retry behavior:
  - `src/core/agents.py`: `teleport_threshold`, `setAllowAgentDeath`, repeated `message_out.setKey(next_segment)`.
- Message producers/consumers:
  - `src/core/messages.py`: definitions for `entry_request`, `entry_accept`, `departure_notice`, `edge_status`, `green_signal`.
- Parser integration and signal extraction limitation:
  - `src/input/sumo_parser.py`: `SUMONetworkParser`, `SUMORouteParser`, `parse_sumo_network`, `parse_sumo_routes`, `_extract_green_edges` currently returning empty list.

## SUMO Evidence Classification

- **verified-in-doc**
  - All SUMO function names and flow steps documented in `sumo/src/mesosim/MESOSIM_RUNTIME_CALL_FLOW.md`.

- **requires-upstream-file**
  - Any behavior requiring direct inspection of paths like:
    - `sumo/src/mesosim/MELoop.cpp`
    - `sumo/src/mesosim/MESegment.cpp`
    - `sumo/src/mesosim/MEVehicle.cpp`
    - `sumo/src/microsim/MSLink.cpp`
    - `sumo/src/netload/NLBuilder.cpp`
    - `sumo/src/microsim/MSNet.cpp`

## Prioritized Gaps

- **High**
  - Event-driven scheduler vs fixed-layer stepping.
  - Junction conflict depth (`MSLink::opened`-style logic absent in Python runtime path).
  - Queue future-admission scheduling vs step retry loop.
  - Teleport branch richness vs hard-threshold death.

- **Medium**
  - Missing calibrator trigger subsystem in Python flow.
  - Detector XML pipeline vs interval parquet collector.
  - Rerouting mechanism mismatch (explicit Python layer vs no equivalent documented phase).

- **Low**
  - Runner/hook placement differences (host orchestration style differences that do not necessarily imply semantic mismatch alone).
