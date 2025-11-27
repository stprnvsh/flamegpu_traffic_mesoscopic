"""
Simulation Runner for FLAMEGPU2 Mesoscopic Traffic Simulation

This module provides the main simulation interface including:
- Configuration management
- Network initialization
- Demand loading
- Simulation execution
- Output collection

Reference: FLAMEGPU2 Simulation Documentation
https://docs.flamegpu.com/guide/running-a-simulation/
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time

# Try to import pyflamegpu
try:
    import pyflamegpu
    FLAMEGPU_AVAILABLE = True
except ImportError:
    FLAMEGPU_AVAILABLE = False

from .model import (
    MesoscopicTrafficModel,
    ModelConfig,
    EnvironmentConfig,
    create_time_update_function,
    create_spawn_packets_function,
    create_logging_function,
    create_rerouting_function,
)


@dataclass
class SimulationConfig:
    """Configuration for simulation execution"""
    # Time settings
    duration: float = 3600.0              # Total simulation duration [seconds]
    time_step: float = 1.0                # Time step size [seconds]
    warmup_time: float = 0.0              # Warmup period (no logging)
    
    # Output settings
    output_interval: float = 3600.0         # Interval for logging [seconds]
    output_dir: str = "./output"          # Output directory
    output_format: str = "csv"            # 'csv' or 'xml'
    
    # GPU settings
    device_id: int = 0                    # CUDA device ID
    
    # Verbosity
    verbose: bool = True                  # Print progress
    
    # Random seed
    random_seed: Optional[int] = None     # For reproducibility
    
    # SUMO Mesoscopic TAU factors (travel time adjustment)
    tau_ff: float = 1.4                   # Free-flow to free-flow factor
    tau_fj: float = 1.4                   # Free-flow to jam factor
    tau_jf: float = 2.0                   # Jam to free-flow factor
    tau_jj: float = 1.4                   # Jam to jam factor
    
    # Rerouting parameters
    rerouting_enabled: bool = False       # Enable dynamic rerouting
    rerouting_period: float = 60.0        # Rerouting check interval [s]
    rerouting_probability: float = 0.7    # Fraction of vehicles that can reroute


@dataclass
class NetworkData:
    """Processed network data ready for simulation"""
    # Edge data (required)
    edge_ids: List[str]                   # Original edge IDs
    edge_id_map: Dict[str, int]           # ID -> index mapping
    edge_lengths: List[float]             # Edge lengths [m]
    edge_speeds: List[float]              # Free-flow speeds [m/s]
    edge_capacities: List[int]            # Max vehicles
    edge_lanes: List[int]                 # Lane counts
    edge_to_nodes: List[int]              # Downstream node indices
    
    # Node data (required)
    node_ids: List[str]                   # Original node IDs
    node_id_map: Dict[str, int]           # ID -> index mapping
    
    # Optional fields with defaults
    edge_from_nodes: List[int] = field(default_factory=list)  # Upstream node indices
    edge_signal_ids: List[int] = field(default_factory=list)  # Signal ID (-1 if none)
    node_adjacency: Dict[int, List[int]] = field(default_factory=dict)  # For rerouting
    signals: List[Dict[str, Any]] = field(default_factory=list)  # Signal definitions
    
    @property
    def num_edges(self) -> int:
        return len(self.edge_ids)
    
    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)
    
    @property
    def num_signals(self) -> int:
        return len(self.signals)


@dataclass
class DemandData:
    """Processed demand data ready for simulation"""
    departures: List[Tuple[float, str, List[str], int]]  # (time, origin, route, count)
    
    @property
    def total_vehicles(self) -> int:
        return sum(d[3] for d in self.departures)
    
    @property
    def num_departures(self) -> int:
        return len(self.departures)


class MesoscopicSimulation:
    """
    Main simulation runner for mesoscopic traffic simulation
    
    This class handles:
    - Model creation and configuration
    - Network and demand data loading
    - Agent initialization
    - Simulation execution
    - Output collection
    
    Usage:
        sim = MesoscopicSimulation()
        sim.load_network(network_data)
        sim.load_demand(demand_data)
        sim.run(duration=3600)
        results = sim.get_results()
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulation
        
        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()
        self.model = None
        self.simulation = None
        self.network = None
        self.demand = None
        self._initialized = False
        self._results = {}
        
    def build_model(self, model_config: Optional[ModelConfig] = None, max_route_length: Optional[int] = None):
        """
        Build the FLAMEGPU2 model
        
        Args:
            model_config: Optional model configuration
            max_route_length: Override max route length (auto-detected from demand if not provided)
        """
        if not FLAMEGPU_AVAILABLE:
            raise RuntimeError("pyflamegpu is not available. Install with: pip install pyflamegpu")
        
        # Update environment config with simulation config
        if model_config is None:
            model_config = ModelConfig()
        model_config.environment_config.time_step = self.config.time_step
        
        # Set max route length if provided (will be set dynamically in initialize() otherwise)
        if max_route_length is not None:
            model_config.packet_config.max_route_length = max_route_length
        
        self._model_config = model_config  # Store for later modification
        self._model_built = False  # Defer actual build until we have demand data
        
    def load_network(self, network: NetworkData):
        """
        Load network data
        
        Args:
            network: Processed network data
        """
        self.network = network
        
        if self.config.verbose:
            print(f"Loaded network: {network.num_edges} edges, "
                  f"{network.num_nodes} nodes, {network.num_signals} signals")
    
    def load_demand(self, demand: DemandData):
        """
        Load demand data
        
        Args:
            demand: Processed demand data
        """
        self.demand = demand
        
        if self.config.verbose:
            print(f"Loaded demand: {demand.num_departures} departures, "
                  f"{demand.total_vehicles} total vehicles")
    
    def initialize(self):
        """
        Initialize the simulation with loaded data
        """
        if self.network is None:
            raise RuntimeError("Network not loaded. Call load_network() first.")
        if self.demand is None:
            raise RuntimeError("Demand not loaded. Call load_demand() first.")
        if not hasattr(self, '_model_config') or self._model_config is None:
            raise RuntimeError("Model not configured. Call build_model() first.")
        
        # Calculate max route length from demand data (round up to next power of 2 for efficiency)
        max_route_in_demand = max(len(d[2]) for d in self.demand.departures) if self.demand.departures else 10
        # Round up to next power of 2, minimum 64
        import math
        max_route_length = max(64, 2 ** math.ceil(math.log2(max_route_in_demand + 1)))
        
        # Update config with actual max route length
        self._model_config.packet_config.max_route_length = max_route_length
        
        if self.config.verbose:
            print(f"Max route length in demand: {max_route_in_demand} edges")
            print(f"Using route array size: {max_route_length}")
        
        # Now actually build the model with correct route length
        self.model = MesoscopicTrafficModel(self._model_config)
        self.model.build()
        
        # Add standard host functions
        self.model.add_step_function(create_time_update_function())
        self.model.add_step_function(create_logging_function(self.config.output_interval))
        
        # Add spawn function BEFORE creating CUDASimulation (model gets compiled at that point)
        spawn_func = create_spawn_packets_function(
            self.demand.departures, 
            self.network.edge_id_map,
            self.network.edge_lengths,
            self.network.edge_speeds,
            self.network.edge_to_nodes,  # For GPU-side rerouting (dest_node)
            max_route_length  # Pass the calculated max route length
        )
        if spawn_func:
            self.model.add_step_function(spawn_func)
        
        # Note: Rerouting via host function is complex in FLAMEGPU2 since agent
        # modification requires device-host sync. Instead, stuck vehicles are
        # teleported after waiting too long (implemented in agent function).
        if self.config.rerouting_enabled and self.config.verbose:
            print(f"Teleporting enabled for stuck vehicles (threshold=180s)")
        
        # Create CUDA simulation (this compiles the model)
        self.simulation = self.model.create_simulation()
        
        # Configure simulation
        self.simulation.SimulationConfig().steps = int(self.config.duration / self.config.time_step)
        
        if self.config.random_seed is not None:
            self.simulation.SimulationConfig().random_seed = self.config.random_seed
        
        # Enable agent compaction for better performance with many spawns/deaths
        # This keeps active agent count low even with 700k+ total spawns
        self.simulation.CUDAConfig().agent_min_dead_before_compact = 1000
        
        # Create initial agents (edge properties stored in agent variables, not env arrays)
        self._create_edge_agents()
        self._create_signal_agents()
        
        self._initialized = True
        
        if self.config.verbose:
            print("Simulation initialized")
    
    def _create_edge_agents(self):
        """Create EdgeQueue agents from network data"""
        import pyflamegpu
        
        # Get agent description from model
        agent_desc = self.model.model.getAgent("EdgeQueue")
        
        # Create population
        num_edges = len(self.network.edge_ids)
        pop = pyflamegpu.AgentVector(agent_desc, num_edges)
        
        for i in range(num_edges):
            agent = pop[i]
            agent.setVariableInt("edge_id", i)
            agent.setVariableInt("capacity", self.network.edge_capacities[i])
            agent.setVariableInt("curr_count", 0)
            agent.setVariableFloat("length", self.network.edge_lengths[i])
            agent.setVariableFloat("free_speed", self.network.edge_speeds[i])
            agent.setVariableInt("signal_id", self.network.edge_signal_ids[i])
            agent.setVariableInt("is_green", 1)  # Default to green
            travel_time = self.network.edge_lengths[i] / self.network.edge_speeds[i] if self.network.edge_speeds[i] > 0 else 1.0
            agent.setVariableFloat("travel_time", travel_time)
            # Node info for GPU-side rerouting
            from_node = self.network.edge_from_nodes[i] if i < len(self.network.edge_from_nodes) else -1
            agent.setVariableInt("from_node", from_node)
            agent.setVariableInt("out_node", self.network.edge_to_nodes[i])
            agent.setVariableInt("lane_count", self.network.edge_lanes[i])
        
        # Add population to simulation
        self.simulation.setPopulationData(pop)
    
    def _create_signal_agents(self):
        """Create SignalController agents from network data"""
        if not self.network.signals:
            return  # No signals to create
        
        import pyflamegpu
        
        # Get agent description from model
        agent_desc = self.model.model.getAgent("SignalController")
        
        # Create population
        num_signals = len(self.network.signals)
        pop = pyflamegpu.AgentVector(agent_desc, num_signals)
        
        max_phases = 32
        max_edges_per_phase = 16
        
        for idx, signal in enumerate(self.network.signals):
            agent = pop[idx]
            agent.setVariableInt("node_id", signal["node_id"])
            agent.setVariableInt("phase_index", 0)
            agent.setVariableInt("phase_count", len(signal["phases"]))
            agent.setVariableFloat("time_to_phase_end", signal["phases"][0]["duration"] if signal["phases"] else 30.0)
            agent.setVariableFloat("cycle_length", signal["cycle_length"])
            
            # Build phase durations array (pad to max_phases)
            phase_durations = [0.0] * max_phases
            for i, phase in enumerate(signal["phases"]):
                if i < max_phases:
                    phase_durations[i] = phase["duration"]
            agent.setVariableArrayFloat("phase_durations", phase_durations)
            
            # Build green edges array (max_phases × max_edges_per_phase)
            # Initialize with -1 (invalid)
            phase_green_edges = [-1] * (max_phases * max_edges_per_phase)
            for i, phase in enumerate(signal["phases"]):
                if i >= max_phases:
                    break
                for j, edge_id in enumerate(phase.get("green_edges", [])):
                    if j < max_edges_per_phase:
                        edge_idx = self.network.edge_id_map.get(edge_id, -1)
                        phase_green_edges[i * max_edges_per_phase + j] = edge_idx
            agent.setVariableArrayInt("phase_green_edges", phase_green_edges)
        
        # Add population to simulation
        self.simulation.setPopulationData(pop)
    
    def run(self, duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the simulation
        
        Args:
            duration: Optional override for simulation duration
            
        Returns:
            Results dictionary
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")
        
        if duration is not None:
            steps = int(duration / self.config.time_step)
            self.simulation.SimulationConfig().steps = steps
        
        if self.config.verbose:
            print(f"Starting simulation for {self.config.duration}s...")
            start_time = time.time()
        
        # Run simulation
        self.simulation.simulate()
        
        if self.config.verbose:
            elapsed = time.time() - start_time
            sim_time = self.config.duration
            print(f"Simulation completed in {elapsed:.2f}s "
                  f"({sim_time/elapsed:.1f}x real-time)")
        
        # Collect results
        self._collect_results()
        
        return self._results
    
    def _collect_results(self):
        """Collect simulation results"""
        import pyflamegpu
        
        # Get packet counts - Packet agent uses states, count both
        packet_desc = self.model.model.getAgent("Packet")
        
        traveling_pop = pyflamegpu.AgentVector(packet_desc)
        self.simulation.getPopulationData(traveling_pop, "traveling")
        
        waiting_pop = pyflamegpu.AgentVector(packet_desc)
        self.simulation.getPopulationData(waiting_pop, "waiting")
        
        total_packets = traveling_pop.size() + waiting_pop.size()
        
        self._results = {
            "duration": self.config.duration,
            "steps": self.simulation.getStepCounter(),
            "final_packet_count": total_packets,
            "packets_traveling": traveling_pop.size(),
            "packets_waiting": waiting_pop.size(),
            "network_edges": self.network.num_edges,
            "total_demand": self.demand.total_vehicles,
        }
        
        # Collect per-edge statistics - EdgeQueue uses default state
        edge_desc = self.model.model.getAgent("EdgeQueue")
        edge_pop = pyflamegpu.AgentVector(edge_desc)
        self.simulation.getPopulationData(edge_pop)
        
        edge_stats = []
        for i in range(edge_pop.size()):
            agent = edge_pop[i]
            edge_stats.append({
                "edge_id": agent.getVariableInt("edge_id"),
                "curr_count": agent.getVariableInt("curr_count"),
                "travel_time": agent.getVariableFloat("travel_time"),
            })
        
        self._results["edge_stats"] = edge_stats
    
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results"""
        return self._results
    
    def get_agent_count(self, agent_type: str, state: str = None) -> int:
        """Get current count of agents of given type"""
        if self.simulation and self.model:
            import pyflamegpu
            agent_desc = self.model.model.getAgent(agent_type)
            
            # Check if agent uses states
            if agent_type == "Packet":
                # Packet uses states - count both if no state specified
                if state:
                    pop = pyflamegpu.AgentVector(agent_desc)
                    self.simulation.getPopulationData(pop, state)
                    return pop.size()
                else:
                    traveling = pyflamegpu.AgentVector(agent_desc)
                    waiting = pyflamegpu.AgentVector(agent_desc)
                    self.simulation.getPopulationData(traveling, "traveling")
                    self.simulation.getPopulationData(waiting, "waiting")
                    return traveling.size() + waiting.size()
            else:
                # Default state
                pop = pyflamegpu.AgentVector(agent_desc)
                self.simulation.getPopulationData(pop)
                return pop.size()
        return 0
    
    def export_results(self, filepath: str):
        """
        Export results to file
        
        Args:
            filepath: Output file path
        """
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self._results, f, indent=2)
        
        if self.config.verbose:
            print(f"Results exported to {filepath}")


# =============================================================================
# Helper Functions
# =============================================================================

def create_simple_network(
    edges: List[Dict[str, Any]],
    nodes: List[Dict[str, Any]],
    signals: Optional[List[Dict[str, Any]]] = None
) -> NetworkData:
    """
    Create NetworkData from simple dictionaries
    
    Args:
        edges: List of edge dictionaries with 'id', 'length', 'speed', 'lanes', 'to_node'
        nodes: List of node dictionaries with 'id'
        signals: Optional list of signal dictionaries
        
    Returns:
        NetworkData instance
    """
    edge_ids = [e['id'] for e in edges]
    edge_id_map = {e['id']: i for i, e in enumerate(edges)}
    
    node_ids = [n['id'] for n in nodes]
    node_id_map = {n['id']: i for i, n in enumerate(nodes)}
    
    # Calculate capacities from jam density
    jam_density = 0.15  # veh/m
    edge_capacities = [int(e['length'] * e.get('lanes', 1) * jam_density) for e in edges]
    
    # Map signal IDs to edges
    edge_signal_ids = [-1] * len(edges)
    if signals:
        for sig_idx, sig in enumerate(signals):
            for phase in sig.get('phases', []):
                for edge_id in phase.get('green_edges', []):
                    if edge_id in edge_id_map:
                        edge_signal_ids[edge_id_map[edge_id]] = sig_idx
    
    return NetworkData(
        edge_ids=edge_ids,
        edge_id_map=edge_id_map,
        edge_lengths=[e['length'] for e in edges],
        edge_speeds=[e['speed'] for e in edges],
        edge_capacities=edge_capacities,
        edge_lanes=[e.get('lanes', 1) for e in edges],
        edge_to_nodes=[node_id_map.get(e.get('to_node', ''), -1) for e in edges],
        edge_signal_ids=edge_signal_ids,
        node_ids=node_ids,
        node_id_map=node_id_map,
        signals=signals or [],
    )


def create_simple_demand(
    departures: List[Tuple[float, str, List[str], int]]
) -> DemandData:
    """
    Create DemandData from simple tuples
    
    Args:
        departures: List of (time, origin, route, count) tuples
        
    Returns:
        DemandData instance
    """
    return DemandData(departures=departures)

