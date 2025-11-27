# GPU-Accelerated Mesoscopic Traffic Simulation with FLAMEGPU2

This repository contains documentation for implementing a city-scale mesoscopic traffic simulation using the FLAMEGPU2 Python API, comparable to SUMO's mesoscopic mode.

## Documentation Structure

The documentation is organized into the following sections:

1. **[01_introduction.md](docs/01_introduction.md)** - Overview and key features
2. **[02_agent_environment_design.md](docs/02_agent_environment_design.md)** - Agent types and environment design
3. **[03_traffic_flow_models.md](docs/03_traffic_flow_models.md)** - Governing equations and traffic flow models
4. **[04_simulation_loop.md](docs/04_simulation_loop.md)** - Simulation loop and agent interaction details
5. **[05_data_input.md](docs/05_data_input.md)** - SUMO network and route file parsing
6. **[06_emissions.md](docs/06_emissions.md)** - Optional emission modeling
7. **[07_flamegpu_implementation.md](docs/07_flamegpu_implementation.md)** - FLAMEGPU2 Python API implementation details
8. **[08_toy_example.md](docs/08_toy_example.md)** - End-to-end example with a simple network
9. **[09_output_design.md](docs/09_output_design.md)** - Output metrics and data collection
10. **[10_gpu_optimization.md](docs/10_gpu_optimization.md)** - GPU optimization strategies
11. **[11_scaling_performance.md](docs/11_scaling_performance.md)** - Scaling and performance considerations
12. **[12_error_handling.md](docs/12_error_handling.md)** - Error handling and edge cases

## Key Concepts

- **Packet agents**: Groups of vehicles traveling together
- **EdgeQueue agents**: Road segments as queues with capacity
- **SignalController agents**: Traffic light controllers at junctions
- **Message-based interaction**: Packets request entry to edges, edges accept/reject based on capacity and signals
- **State-based packet behavior**: TRAVELING and WAITING states for packet movement

## Important Implementation Notes

- **Route Array Lookup**: See [04_simulation_loop.md](04_simulation_loop.md) for complete route advancement logic
- **Departure Messages**: Packets must send departure notices when leaving edges to maintain accurate edge counts
- **Signal Communication**: See [07_flamegpu_implementation.md](07_flamegpu_implementation.md) for signal-to-edge message passing
- **Error Handling**: Review [12_error_handling.md](12_error_handling.md) for common edge cases and validation strategies

