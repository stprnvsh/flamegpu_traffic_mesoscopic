# Python Runtime Call Flow

## 1) Key Files And Responsibilities
- `src/core/__init__.py`: re-exports core runtime API (`MesoscopicTrafficModel`, `MesoscopicSimulation`, helpers).
- `src/core/simulation.py`: host-side orchestrator (`MesoscopicSimulation`) for build, initialize, run, collect, save.
- `src/core/model.py`: FLAMEGPU model assembly (`MesoscopicTrafficModel.build`) and execution layer definition.
- `src/core/agents.py`: RTC agent behaviors for `EdgeQueue`, `Packet`, `SignalController`.
- `src/core/messages.py`: message schema and bucket/bruteforce channels.
- `src/core/metrics.py`: interval aggregation and parquet output (`IntervalEdgeDataCollector`).
- `src/input/__init__.py`: parser entry exports.
- `src/input/sumo_parser.py`: SUMO network/routes parsing and conversion to `NetworkData`/`DemandData`.
- `src/traffic_models/__init__.py`: analytical model exports.
- `src/traffic_models/fundamental_diagram.py`: FD math models and travel-time/state calculations.
- `src/traffic_models/queue_models.py`: queue dynamics models.
- `src/traffic_models/junction_models.py`: junction capacity/delay/control models.

## 2) Ordered Call Flow (Concrete Functions)

### Entry + Initialization Path
1. External runner calls `MesoscopicSimulation.build_model(...)`, `load_network(...)`, `load_demand(...)`, then `initialize(...)`.
2. `MesoscopicSimulation.initialize` computes route array size from demand and writes `packet_config.max_route_length`.
3. `MesoscopicSimulation.initialize` creates and builds model:
   - `self.model = MesoscopicTrafficModel(self._model_config)`
   - `self.model.build()`
4. `MesoscopicTrafficModel.build` runs in order:
   - `_define_messages()` -> `define_messages(...)`
   - `_define_agents()` -> `define_edge_queue_agent(...)`, `define_packet_agent(...)`, `define_signal_controller_agent(...)`
   - `_define_environment()`
   - `_define_layers()`
5. `MesoscopicSimulation.initialize` attaches host step functions (in add order):
   - `create_time_update_function()`
   - `create_logging_function(...)`
   - `create_edge_data_function(...)`
   - `create_spawn_packets_function(...)`
6. `MesoscopicSimulation.initialize` compiles model:
   - `self.simulation = self.model.create_simulation()`
   - configures steps/seed/CUDA settings.
7. Initial populations are created:
   - `_create_edge_agents()`
   - `_create_signal_agents()`

### Per-Step Execution Path
1. `MesoscopicSimulation.run` loops with `self.simulation.step()`.
2. Interval boundary logic in host loop:
   - `IntervalEdgeDataCollector.collect(current_time)`
   - set env `reset_interval_counters = 1` for that step.
3. Device layer order from `MesoscopicTrafficModel._define_layers`:
   - `L0_reset_counters`: `EdgeQueue.reset_interval_counters`
   - `L1_move`: `Packet.move_and_request`
   - `L2_departure`: `Packet.send_departure`
   - `L3_process_departures`: `EdgeQueue.process_departures`
   - `L4_signal`: `SignalController.update_signal`
   - `L5_green_flag`: `EdgeQueue.update_green_flag`
   - `L6_broadcast`: `EdgeQueue.broadcast_status`
   - `L7_reroute`: `Packet.try_reroute`
   - `L8_process_requests`: `EdgeQueue.process_edge_requests`
   - `L9_wait`: `Packet.wait_for_entry`
4. End of run:
   - final collector call
   - `_collect_results()`
   - `IntervalEdgeDataCollector.save(...)`

### Message/Dataflow Path
- `entry_request` (bucket by target segment):
  - written by `Packet.move_and_request` and retry in `Packet.wait_for_entry`
  - read by `EdgeQueue.process_edge_requests`
- `entry_accept` (bucket by packet `agent_id`):
  - written by `EdgeQueue.process_edge_requests`
  - read by `Packet.wait_for_entry`
- `departure_notice` (bucket by current segment):
  - written by `Packet.send_departure`
  - read by `EdgeQueue.process_departures`
- `green_signal` (bruteforce):
  - written by `SignalController.update_signal`
  - read by `EdgeQueue.update_green_flag`
- `edge_status` (bucket by `from_node`):
  - written by `EdgeQueue.broadcast_status`
  - read by `Packet.try_reroute`

### Input Parsing + Integration Path
1. `parse_sumo_network(...)` -> `SUMONetworkParser.parse(...)`
   - `_parse_nodes`, `_parse_edges`, `_parse_traffic_lights`, `_parse_connections`
   - `_to_network_data` (optionally segmentizes edges via `split_edge_into_segments`)
   - returns `NetworkData`.
2. `parse_sumo_routes(...)` -> `SUMORouteParser.parse(...)`
   - `_parse_routes`, `_parse_vehicles`, `_parse_flows`, `_parse_trips`
   - `_group_vehicles`
   - returns `DemandData`.
3. `MesoscopicSimulation.initialize` consumes these outputs:
   - `create_spawn_packets_function(self.demand.departures, self.network.edge_id_map, ..., edge_first_segment, segments)`
   - `_create_edge_agents` builds one `EdgeQueue` per segment when `network.segments` exists.
   - `_create_signal_agents` maps signal `green_edges` through `edge_id_map`.

## 3) Notable Branching Paths
- Segment vs edge mode:
  - `_create_edge_agents` and `create_spawn_packets_function` branch on segment data availability.
- Packet lifecycle branch:
  - `move_and_request`: traveling decrement vs ready/wait loop.
  - teleport/death when wait exceeds threshold.
  - route-complete death when no `next_segment`.
- Admission branch:
  - `wait_for_entry` transitions on `entry_accept`, else re-requests.
- Interval metrics branch:
  - reset gate controlled by env `reset_interval_counters`.
- Flow parsing branch:
  - `_parse_flows` handles `period` / `vehsPerHour` / `number`.
- Signal optional branch:
  - `_create_signal_agents` is no-op when `network.signals` is empty.
- Rerouting behavior branch:
  - GPU-side `Packet.try_reroute` active in waiting state; host rerouting path is not used in main run path.

## 4) Traffic Models Package Role
- `src/traffic_models/*` is an analytical model library (FD/queue/junction math APIs).
- Runtime in `src/core/*` does not directly import or invoke these classes/functions.
- Operational runtime behavior is implemented in FLAMEGPU RTC agent functions in `src/core/agents.py`.

## Unresolved Gaps Requiring Direct File Read
- Exact top-level runner variants (`run_sumo_network.py`, examples, tests) may apply different setup sequences before calling `MesoscopicSimulation`.
- `_extract_green_edges` in parser is currently simplified (returns empty list), so signal edge mapping semantics depend on further implementation if required.
