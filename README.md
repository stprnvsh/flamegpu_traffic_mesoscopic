# FLAMEGPU2 Mesoscopic Traffic Simulation

A GPU-accelerated mesoscopic traffic simulation using FLAMEGPU2, comparable to SUMO's mesoscopic mode but leveraging GPU parallelism for large-scale city simulations.

## Features

- **GPU-Accelerated**: Runs on NVIDIA GPUs via FLAMEGPU2
- **SUMO Compatible**: Parses `.net.xml`, `.rou.xml`, `.trips.xml`, and `.sumocfg` files
- **Mesoscopic Model**: Packet-based simulation with queue dynamics
- **GPU-Side Rerouting**: Dynamic rerouting computed entirely on GPU
- **Traffic Signals**: Full signal phase support
- **Teleporting**: SUMO-compatible stuck vehicle handling
- **Agent Compaction**: Efficient memory management for high-throughput simulations

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install FLAMEGPU2 (CUDA 12.0)
pip install pyflamegpu
# Or for specific CUDA version:
pip install --extra-index-url https://whl.flamegpu.com/whl/cuda120/ pyflamegpu
```

**Requirements:**
- Python 3.10+
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11.0+ (matching pyflamegpu version)

## Quick Start

### Run from SUMO files

```bash
# Basic usage
python run_sumo_network.py <network.net.xml> <routes.rou.xml> [duration_seconds]

# Example: 24-hour simulation
python run_sumo_network.py arbon.net.xml routes_arbon.xml 86400

# With SUMO config file
python run_sumo_network.py simulation.sumocfg

# Override config files
python run_sumo_network.py simulation.sumocfg network.xml routes.xml 3600
```

### Run from trips file

```bash
# Trips files (origin-destination) are supported
python run_sumo_network.py network.net.xml trips.trips.xml 86400

# For better results, convert trips to routes first:
duarouter -n network.net.xml --route-files trips.trips.xml -o routes.rou.xml
```

### Python API

```python
from src.input.sumo_parser import parse_sumo_network, parse_sumo_routes
from src.core.simulation import MesoscopicSimulation, SimulationConfig

# Load SUMO files
network = parse_sumo_network("network.net.xml")
demand = parse_sumo_routes("routes.rou.xml", edge_id_map=network.edge_id_map)

# Configure simulation
config = SimulationConfig(
    duration=86400.0,      # 24 hours
    time_step=1.0,         # 1 second steps
    verbose=True,
)

# Run simulation
sim = MesoscopicSimulation(config)
sim.build_model()
sim.load_network(network)
sim.load_demand(demand)
sim.initialize()
results = sim.run()

# Export results
sim.export_results("results.json")
```

## Architecture

### Agent Types

| Agent | Purpose | Key Variables |
|-------|---------|---------------|
| **Packet** | Group of vehicles | `curr_edge`, `next_edge`, `route[]`, `remaining_time` |
| **EdgeQueue** | Road segment | `capacity`, `curr_count`, `travel_time`, `is_green` |
| **SignalController** | Traffic light | `phase_index`, `phase_durations[]`, `cycle_length` |

### Execution Layers (per step)

1. `send_departure` - Packets notify edges when leaving
2. `process_departures` - Edges update vehicle counts
3. `move_and_request` - Packets travel and request next edge
4. `update_signal` - Signals advance phases
5. `update_green_flag` - Edges read signal state
6. `broadcast_status` - Edges broadcast congestion (for rerouting)
7. `try_reroute` - Stuck packets find alternatives (GPU-side)
8. `process_edge_requests` - Edges accept/reject entry requests
9. `wait_for_entry` - Packets check acceptance and transition

### GPU-Side Rerouting

When packets are stuck (waiting >60s), they autonomously find alternative routes:

```cpp
// Packets scan edge_status messages to find alternatives
for (const auto& msg : FLAMEGPU->message_in(curr_node)) {
    if (msg.getVariable<int>("available_capacity") > 0) {
        // Found alternative edge from current node
        FLAMEGPU->setVariable<int>("next_edge", msg.getVariable<int>("edge_id"));
    }
}
```

## Project Structure

```
├── src/
│   ├── core/
│   │   ├── agents.py       # Agent definitions (RTC CUDA code)
│   │   ├── messages.py     # Message type definitions
│   │   ├── model.py        # FLAMEGPU2 model builder
│   │   └── simulation.py   # Simulation runner
│   ├── input/
│   │   └── sumo_parser.py  # SUMO file parsers
│   └── traffic_models/
│       ├── fundamental_diagram.py  # Speed-density models
│       ├── queue_models.py         # Queue dynamics
│       └── junction_models.py      # Junction capacity
├── examples/
│   └── toy_simulation.py   # Simple example
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── run_sumo_network.py     # Main entry point
```

## Configuration

### SimulationConfig Options

```python
SimulationConfig(
    duration=3600.0,           # Simulation duration [s]
    time_step=1.0,             # Time step [s] (use 5.0 for faster runs)
    output_interval=60.0,      # Logging interval [s]
    verbose=True,              # Print progress
    random_seed=42,            # For reproducibility
    
    # SUMO mesoscopic parameters
    tau_ff=1.4,                # Free-flow TAU factor
    tau_fj=1.4,                # Free-to-jam TAU factor
    tau_jf=2.0,                # Jam-to-free TAU factor
    tau_jj=1.4,                # Jam-to-jam TAU factor
    
    # Rerouting (GPU-side)
    rerouting_enabled=True,    # Enable GPU rerouting
    rerouting_period=60.0,     # Check interval [s]
)
```

### SUMO Config File Support

The simulation reads mesoscopic parameters from `.sumocfg` files:

```xml
<configuration>
    <mesoscopic>
        <mesosim value="true"/>
        <meso-tauff value="1.4"/>
        <meso-taufj value="1.4"/>
    </mesoscopic>
    <processing>
        <time-to-teleport value="180"/>
    </processing>
</configuration>
```

## Performance Tuning

### Quick Optimizations

```python
# 1. Increase time step (5x speedup)
config = SimulationConfig(time_step=5.0)

# 2. Reduce logging (10-20% speedup)
config = SimulationConfig(output_interval=600.0, verbose=False)

# 3. Agent compaction is enabled by default for large simulations
```

### Network Encoding

SUMO files with non-UTF-8 characters (German umlauts) need conversion:

```bash
iconv -f LATIN1 -t UTF-8 network.net.xml > network_utf8.net.xml
```

## Output

Results are exported as JSON:

```json
{
  "steps": 86400,
  "final_packet_count": 370,
  "packets_traveling": 370,
  "packets_waiting": 0,
  "edge_stats": [
    {"edge_id": 0, "curr_count": 5, "capacity": 50, "travel_time": 12.5}
  ]
}
```

## Documentation

Detailed documentation in `docs/`:

1. [Introduction](docs/01_introduction.md) - Overview and key features
2. [Agent Design](docs/02_agent_environment_design.md) - Agent types and environment
3. [Traffic Models](docs/03_traffic_flow_models.md) - Fundamental diagrams and queue models
4. [Simulation Loop](docs/04_simulation_loop.md) - Execution order and state transitions
5. [Data Input](docs/05_data_input.md) - SUMO file parsing
6. [FLAMEGPU Implementation](docs/07_flamegpu_implementation.md) - RTC code details
7. [GPU Optimization](docs/10_gpu_optimization.md) - Performance tuning

## Testing

```bash
pytest tests/ -v
```

## License

MIT License

## Acknowledgments

- [FLAMEGPU2](https://flamegpu.com/) - GPU agent-based simulation framework
- [SUMO](https://eclipse.dev/sumo/) - Traffic simulation reference
