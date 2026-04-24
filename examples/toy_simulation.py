#!/usr/bin/env python3
"""
Toy Example: Simple Mesoscopic Traffic Simulation

This example demonstrates a basic 3-edge, 2-junction network:

    A ─────────[Edge 0]───────> J1 ─────[Edge 2]────> C
                                 ↑
    B ─────────[Edge 1]─────────┘

Two streams of traffic merge at junction J1 and proceed to C.

This matches the toy example from docs/08_toy_example.md

Usage:
    python toy_simulation.py

Requirements:
    - pyflamegpu (install with: pip install pyflamegpu)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Check if FLAMEGPU is available
try:
    import pyflamegpu
    FLAMEGPU_AVAILABLE = True
    print("✓ pyflamegpu is available")
except ImportError:
    FLAMEGPU_AVAILABLE = False
    print("✗ pyflamegpu is not available - running in mock mode")

from core.simulation import (
    MesoscopicSimulation,
    SimulationConfig,
    create_simple_network,
    create_simple_demand,
    NetworkData,
    DemandData,
)
from core.model import ModelConfig


def create_toy_network() -> NetworkData:
    """
    Create the toy network from the documentation
    
    Network:
    - Edge 0: A -> J1, 500m, 20 m/s, capacity 40
    - Edge 1: B -> J1, 300m, 15 m/s, capacity 24
    - Edge 2: J1 -> C, 400m, 20 m/s, capacity 32
    """
    edges = [
        {
            'id': 'edge_0',
            'length': 500.0,      # meters
            'speed': 20.0,        # m/s (~72 km/h)
            'lanes': 1,
            'to_node': 'J1',
            'from_node': 'A',
        },
        {
            'id': 'edge_1', 
            'length': 300.0,
            'speed': 15.0,        # m/s (~54 km/h)
            'lanes': 1,
            'to_node': 'J1',
            'from_node': 'B',
        },
        {
            'id': 'edge_2',
            'length': 400.0,
            'speed': 20.0,
            'lanes': 1,
            'to_node': 'C',
            'from_node': 'J1',
        },
    ]
    
    nodes = [
        {'id': 'A'},
        {'id': 'B'},
        {'id': 'J1'},
        {'id': 'C'},
    ]
    
    # No signals in this example (unsignalized merge)
    signals = []
    
    return create_simple_network(edges, nodes, signals)


def create_toy_demand() -> DemandData:
    """
    Create the toy demand from the documentation
    
    Demand:
    - At t=0: 10 vehicles from A to C (via J1)
    - At t=0: 5 vehicles from B to C (via J1)
    """
    departures = [
        # (time, origin_edge, route, count)
        (0.0, 'edge_0', ['edge_0', 'edge_2'], 10),  # A -> J1 -> C
        (0.0, 'edge_1', ['edge_1', 'edge_2'], 5),   # B -> J1 -> C
    ]
    
    return create_simple_demand(departures)


def run_simulation_mock():
    """
    Run a mock simulation without FLAMEGPU
    
    This demonstrates the expected behavior using Python calculations.
    """
    print("\n" + "="*60)
    print("Running MOCK Simulation (no GPU)")
    print("="*60)
    
    network = create_toy_network()
    demand = create_toy_demand()
    
    print(f"\nNetwork:")
    print(f"  Edges: {network.num_edges}")
    for i, eid in enumerate(network.edge_ids):
        print(f"    {eid}: {network.edge_lengths[i]:.0f}m, "
              f"{network.edge_speeds[i]:.1f} m/s, "
              f"capacity {network.edge_capacities[i]}")
    
    print(f"\nDemand:")
    print(f"  Total vehicles: {demand.total_vehicles}")
    for dep in demand.departures:
        print(f"    t={dep[0]:.0f}s: {dep[3]} vehicles on route {dep[2]}")
    
    # Calculate expected times
    print("\n" + "-"*60)
    print("Expected Timeline (from documentation):")
    print("-"*60)
    
    # Edge travel times
    tt_edge0 = 500 / 20  # 25s
    tt_edge1 = 300 / 15  # 20s
    tt_edge2 = 400 / 20  # 20s
    
    print(f"\nFree-flow travel times:")
    print(f"  Edge 0 (A→J1): {tt_edge0:.0f}s")
    print(f"  Edge 1 (B→J1): {tt_edge1:.0f}s")
    print(f"  Edge 2 (J1→C): {tt_edge2:.0f}s")
    
    print(f"\nPacket progression:")
    print(f"  t=0:   P_A (10 veh) starts on Edge 0, P_B (5 veh) starts on Edge 1")
    print(f"  t=20:  P_B arrives at J1, requests entry to Edge 2")
    print(f"         Edge 2 accepts (capacity 32 > 5)")
    print(f"         P_B enters Edge 2")
    print(f"  t=25:  P_A arrives at J1, requests entry to Edge 2")
    print(f"         Edge 2 accepts (32 - 5 = 27 > 10)")
    print(f"         P_A enters Edge 2")
    print(f"  t=40:  P_B exits Edge 2 (destination reached)")
    print(f"  t=45:  P_A exits Edge 2 (destination reached)")
    
    print(f"\nTotal travel times:")
    print(f"  P_A: {tt_edge0 + tt_edge2:.0f}s (free-flow)")
    print(f"  P_B: {tt_edge1 + tt_edge2:.0f}s (free-flow)")
    
    print("\n" + "="*60)
    print("Mock simulation complete!")
    print("="*60)


def run_simulation_gpu():
    """
    Run actual GPU simulation with FLAMEGPU2
    """
    print("\n" + "="*60)
    print("Running GPU Simulation with FLAMEGPU2")
    print("="*60)
    
    # Create simulation
    config = SimulationConfig(
        duration=60.0,      # 60 seconds
        time_step=1.0,      # 1 second steps
        verbose=True,
    )
    
    sim = MesoscopicSimulation(config)
    
    # Build model
    sim.build_model()
    
    # Load network and demand
    network = create_toy_network()
    demand = create_toy_demand()
    
    sim.load_network(network)
    sim.load_demand(demand)
    
    # Initialize
    sim.initialize()
    
    # Run
    results = sim.run()
    
    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("GPU simulation complete!")
    print("="*60)


def main():
    """Main entry point"""
    print("="*60)
    print("Mesoscopic Traffic Simulation - Toy Example")
    print("="*60)
    
    if FLAMEGPU_AVAILABLE:
        run_simulation_gpu()
    else:
        run_simulation_mock()
        print("\nTo run with GPU acceleration, install pyflamegpu:")
        print("  pip install pyflamegpu")


if __name__ == "__main__":
    main()

