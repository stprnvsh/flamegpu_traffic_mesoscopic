#!/usr/bin/env python3
"""
Run FLAMEGPU2 Mesoscopic Simulation from SUMO Network and Route Files

Usage:
    python run_sumo_network.py <network.net.xml> <routes.rou.xml> [duration]
    python run_sumo_network.py <simulation.sumocfg> [duration]
    python run_sumo_network.py <simulation.sumocfg> <network.xml> <routes.xml> [duration]

Example:
    python run_sumo_network.py cologne.net.xml cologne.rou.xml 3600
    python run_sumo_network.py sumo_base.sumocfg
    python run_sumo_network.py sumo_base.sumocfg arbon.net.xml routes_arbon.xml 7200
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from input.sumo_parser import (
    parse_sumo_network, 
    parse_sumo_routes, 
    SUMOConfigParser,
    SUMOMesoConfig
)
from core.simulation import MesoscopicSimulation, SimulationConfig
from core.model import ModelConfig, EnvironmentConfig


def print_meso_config(config: SUMOMesoConfig):
    """Print mesoscopic configuration parameters"""
    print("\n  Mesoscopic Parameters:")
    print(f"    TAU factors: ff={config.tau_ff}, fj={config.tau_fj}, jf={config.tau_jf}, jj={config.tau_jj}")
    print(f"    Edge segment length: {config.meso_edgelength}m")
    print(f"    Jam threshold: {config.jam_threshold}")
    print(f"    Multi-queue: {config.multi_queue}, Junction control: {config.junction_control}")
    print(f"    Minor penalty: {config.minor_penalty}s, TLS penalty: {config.tls_penalty}s")
    
    print("\n  Rerouting Parameters:")
    print(f"    Probability: {config.rerouting_probability * 100:.0f}%")
    print(f"    Period: {config.rerouting_period}s")
    print(f"    Algorithm: {config.routing_algorithm}")
    print(f"    Threshold: factor={config.rerouting_threshold_factor}, const={config.rerouting_threshold_constant}s")
    
    print("\n  Processing:")
    print(f"    Time to teleport: {config.time_to_teleport}s")
    print(f"    Time to impatience: {config.time_to_impatience}s")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    # Check if first argument is a .sumocfg file
    first_arg = sys.argv[1]
    meso_config = None
    
    print("=" * 60)
    print("FLAMEGPU2 Mesoscopic Traffic Simulation")
    print("=" * 60)
    
    if first_arg.endswith('.sumocfg'):
        # Parse SUMO config file for mesoscopic parameters
        print(f"\nParsing SUMO config: {first_arg}")
        
        config_parser = SUMOConfigParser()
        meso_config = config_parser.parse(first_arg)
        print_meso_config(meso_config)
        
        # Check if network and routes are provided on command line (override config)
        if len(sys.argv) >= 4 and sys.argv[2].endswith('.xml') and sys.argv[3].endswith('.xml'):
            # Format: config.sumocfg network.xml routes.xml [duration]
            net_file = sys.argv[2]
            route_file = sys.argv[3]
            duration = float(sys.argv[4]) if len(sys.argv) > 4 else meso_config.end_time
            print(f"\n  Using command-line network/routes (overriding config)")
        else:
            # Use files from config
            input_files = config_parser.get_input_files(first_arg)
            net_file = input_files.get("network")
            route_file = input_files.get("routes")
            
            if not net_file:
                print("Error: No network file specified in config or command line")
                print("Usage: python run_sumo_network.py config.sumocfg network.xml routes.xml [duration]")
                sys.exit(1)
            
            if not route_file:
                print("Error: No route file specified in config or command line")
                print("Usage: python run_sumo_network.py config.sumocfg network.xml routes.xml [duration]")
                sys.exit(1)
            
            # Duration from command line or config
            duration = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].endswith('.xml') else meso_config.end_time
        
    else:
        # Old format: network.xml routes.xml [duration]
        if len(sys.argv) < 3:
            print(__doc__)
            sys.exit(1)
        
        net_file = first_arg
        route_file = sys.argv[2]
        duration = float(sys.argv[3]) if len(sys.argv) > 3 else 3600.0
    
    # Parse network
    print(f"\nParsing network: {net_file}")
    network = parse_sumo_network(net_file)
    print(f"  Edges: {network.num_edges}")
    print(f"  Nodes: {network.num_nodes}")
    print(f"  Signals: {network.num_signals}")
    
    # Parse routes
    print(f"\nParsing routes: {route_file}")
    demand = parse_sumo_routes(route_file, edge_id_map=network.edge_id_map)
    print(f"  Total vehicles: {demand.total_vehicles}")
    print(f"  Packet departures: {demand.num_departures}")
    
    # Configure simulation with meso parameters if available
    if meso_config:
        # Enable rerouting if probability > 0 in config
        rerouting_enabled = meso_config.rerouting_probability > 0
        
        config = SimulationConfig(
            duration=duration,
            time_step=meso_config.step_length,
            verbose=True,
            # Pass SUMO meso parameters
            tau_ff=meso_config.tau_ff,
            tau_fj=meso_config.tau_fj,
            tau_jf=meso_config.tau_jf,
            tau_jj=meso_config.tau_jj,
            random_seed=meso_config.seed,
            # Rerouting
            rerouting_enabled=rerouting_enabled,
            rerouting_period=meso_config.rerouting_period,
            rerouting_probability=meso_config.rerouting_probability,
        )
        
        if rerouting_enabled:
            print(f"\n  Rerouting: ENABLED (period={meso_config.rerouting_period}s, prob={meso_config.rerouting_probability*100:.0f}%)")
    else:
        config = SimulationConfig(
            duration=duration,
            time_step=1.0,
            verbose=True,
        )
    
    # Create and run simulation
    print(f"\nInitializing simulation for {duration}s...")
    sim = MesoscopicSimulation(config)
    sim.build_model()
    sim.load_network(network)
    sim.load_demand(demand)
    sim.initialize()
    
    print("\nRunning simulation...")
    results = sim.run()
    
    # Print results
    print("\n" + "=" * 60)
    print("Simulation Results")
    print("=" * 60)
    print(f"  Simulation steps: {results['steps']}")
    print(f"  Final packets (traveling): {results['packets_traveling']}")
    print(f"  Final packets (waiting): {results['packets_waiting']}")
    print(f"  Total packets remaining: {results['final_packet_count']}")
    
    # Edge statistics summary
    if results.get('edge_stats'):
        occupied = sum(1 for e in results['edge_stats'] if e['curr_count'] > 0)
        total_vehicles = sum(e['curr_count'] for e in results['edge_stats'])
        print(f"  Edges with vehicles: {occupied}/{network.num_edges}")
        print(f"  Total vehicles on network: {total_vehicles}")
    
    # Export results
    output_file = "simulation_results.json"
    sim.export_results(output_file)
    print(f"\nResults exported to {output_file}")
    
    return results


if __name__ == "__main__":
    main()
