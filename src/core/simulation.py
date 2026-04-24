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
from .metrics import (
    MetricsConfig,
    create_edge_data_function,
)


@dataclass
class SimulationConfig:
    """Configuration for simulation execution"""
    # Time settings
    duration: float = 3600.0              # Total simulation duration [seconds]
    time_step: float = 1.0                # Time step size [seconds]
    warmup_time: float = 0.0              # Warmup period (no logging)
    
    # Output settings
    output_interval: float = 60.0         # Interval for console logging [seconds]
    output_dir: str = "./output"          # Output directory
    output_format: str = "parquet"        # 'parquet', 'csv' or 'json'
    
    # Metrics collection settings
    metrics_interval: float = 900.0       # Interval for metrics aggregation [seconds]
    metrics_file: str = "simulation_metrics.parquet"  # Metrics output file
    collect_edge_metrics: bool = True     # Collect per-edge metrics
    
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
    enable_calibrator_lite: bool = False
    calibrator_gain: float = 0.05


@dataclass
class SegmentData:
    """Data for a single segment (SUMO meso-style ~100m chunks)"""
    segment_id: str                       # Format: "edge_id:segment_idx"
    edge_id: str                          # Parent edge ID
    edge_idx: int                         # Parent edge index
    segment_idx: int                      # Segment index within edge (0, 1, 2...)
    length: float                         # Segment length [m]
    speed: float                          # Speed limit [m/s]
    capacity: int                         # Max vehicles
    lanes: int                            # Number of lanes
    next_segment: int                     # Next segment index (-1 if last in edge)
    from_node: int                        # Start node index
    to_node: int                          # End node index (only meaningful for last segment)
    signal_id: int                        # Signal ID (-1 if none, only for last segment)


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
    edge_priorities: List[int] = field(default_factory=list)
    edge_from_nodes: List[int] = field(default_factory=list)  # Upstream node indices
    edge_signal_ids: List[int] = field(default_factory=list)  # Signal ID (-1 if none)
    node_adjacency: Dict[int, List[int]] = field(default_factory=dict)  # For rerouting
    signals: List[Dict[str, Any]] = field(default_factory=list)  # Signal definitions
    edge_successors: Dict[int, List[int]] = field(default_factory=dict)
    movement_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = field(default_factory=dict)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    movement_successors: Dict[int, List[int]] = field(default_factory=dict)
    movement_signal_map: Dict[Tuple[str, int], int] = field(default_factory=dict)  # (tl_id, linkIndex)->movement_id
    movement_conflicts: Dict[int, List[int]] = field(default_factory=dict)  # movement_id -> conflicting movement_ids
    movement_priority: Dict[int, int] = field(default_factory=dict)  # movement_id -> priority rank
    
    # Segment data (for SUMO meso compatibility)
    segments: List[SegmentData] = field(default_factory=list)  # Segment list
    segment_id_map: Dict[str, int] = field(default_factory=dict)  # segment_id -> index
    edge_to_segments: Dict[int, List[int]] = field(default_factory=dict)  # edge_idx -> [segment indices]
    edge_first_segment: List[int] = field(default_factory=list)  # edge_idx -> first segment index
    edge_last_segment: List[int] = field(default_factory=list)   # edge_idx -> last segment index
    
    @property
    def num_edges(self) -> int:
        return len(self.edge_ids)
    
    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)
    
    @property
    def num_signals(self) -> int:
        return len(self.signals)
    
    @property
    def num_segments(self) -> int:
        return len(self.segments)


@dataclass
class DemandData:
    """Processed demand data ready for simulation"""
    departures: List[Tuple[float, str, List[str], int]]  # (time, origin, route, count)
    invalid_routes: List[Dict[str, Any]] = field(default_factory=list)
    
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
        self._metrics_function = None
        
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
        model_config.environment_config.time_to_teleport = 180.0
        model_config.environment_config.time_to_teleport_disconnected = 30.0
        
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
        
        # Console logging (frequent, for monitoring)
        self.model.add_step_function(create_logging_function(self.config.output_interval))
        
        # Metrics collection for parquet output (less frequent)
        from .metrics import create_edge_data_function
        self._metrics_function = create_edge_data_function(
            output_interval=self.config.metrics_interval,
            edge_ids=self.network.edge_ids,
            edge_lengths=self.network.edge_lengths,
            edge_lanes=self.network.edge_lanes,
            edge_speeds=self.network.edge_speeds,
            output_file=self.config.metrics_file
        )
        if self._metrics_function:
            self.model.add_step_function(self._metrics_function)
        
        # Add spawn function BEFORE creating CUDASimulation (model gets compiled at that point)
        spawn_func = create_spawn_packets_function(
            self.demand.departures, 
            self.network.edge_id_map,
            self.network.edge_lengths,
            self.network.edge_speeds,
            self.network.edge_to_nodes,  # For GPU-side rerouting (dest_node)
            max_route_length,  # Pass the calculated max route length
            # Segment data for SUMO meso compatibility
            edge_first_segment=self.network.edge_first_segment if self.network.segments else None,
            segments=self.network.segments if self.network.segments else None,
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
        """Create EdgeQueue (Segment) agents from network data
        
        Note: EdgeQueue agents now represent SEGMENTS (~100m), not full edges.
        This matches SUMO's mesoscopic model for better traffic flow.
        """
        import pyflamegpu
        
        # Get agent description from model
        agent_desc = self.model.model.getAgent("EdgeQueue")
        
        # Check if we have segment data
        if self.network.segments:
            # Use segment-based approach (SUMO meso style)
            num_segments = len(self.network.segments)
            print(f"Creating {num_segments} segment agents (from {len(self.network.edge_ids)} edges)")
            pop = pyflamegpu.AgentVector(agent_desc, num_segments)
            
            # Get tau parameters from config
            tau_ff = getattr(self.config, 'tau_ff', 1.4)
            tau_fj = getattr(self.config, 'tau_fj', 1.4)
            tau_jf = getattr(self.config, 'tau_jf', 2.0)
            tau_jj = getattr(self.config, 'tau_jj', 2.0)
            
            for i, seg in enumerate(self.network.segments):
                agent = pop[i]
                agent.setVariableInt("edge_id", i)  # Segment index as ID
                agent.setVariableInt("edge_idx", seg.edge_idx)  # Parent edge
                agent.setVariableInt("segment_idx", seg.segment_idx)  # Segment position in edge
                agent.setVariableInt("next_segment", seg.next_segment)  # Next segment (-1 if last)
                agent.setVariableInt("capacity", seg.capacity)
                agent.setVariableInt("curr_count", 0)
                agent.setVariableFloat("length", seg.length)
                agent.setVariableFloat("free_speed", seg.speed)
                agent.setVariableInt("signal_id", seg.signal_id)
                agent.setVariableInt("is_green", 1)
                travel_time = seg.length / seg.speed if seg.speed > 0 else 1.0
                agent.setVariableFloat("travel_time", travel_time)
                agent.setVariableInt("from_node", seg.from_node)
                agent.setVariableInt("out_node", seg.to_node)
                agent.setVariableInt("lane_count", seg.lanes)
                prio = self.network.edge_priorities[seg.edge_idx] if seg.edge_idx < len(self.network.edge_priorities) else 0
                agent.setVariableInt("priority_rank", prio)
                agent.setVariableInt("conflict_group", seg.to_node if seg.to_node >= 0 else -1)
                agent.setVariableFloat("junction_block_until", 0.0)
                
                # SUMO Mesoscopic 4-tau headway parameters
                agent.setVariableFloat("tau_ff", tau_ff)
                agent.setVariableFloat("tau_fj", tau_fj)
                agent.setVariableFloat("tau_jf", tau_jf)
                agent.setVariableFloat("tau_jj", tau_jj)
                
                agent.setVariableFloat("jam_threshold", 0.8)
                agent.setVariableFloat("block_time", 0.0)
                agent.setVariableInt("is_jammed", 0)
                
                # Interval metrics
                agent.setVariableFloat("interval_sampled_seconds", 0.0)
                agent.setVariableInt("interval_entered", 0)
                agent.setVariableInt("interval_left", 0)
        else:
            # Fallback: use edge-based approach (old behavior)
            num_edges = len(self.network.edge_ids)
            print(f"Creating {num_edges} edge agents (no segment data)")
            pop = pyflamegpu.AgentVector(agent_desc, num_edges)
            
            tau_ff = getattr(self.config, 'tau_ff', 1.4)
            tau_fj = getattr(self.config, 'tau_fj', 1.4)
            tau_jf = getattr(self.config, 'tau_jf', 2.0)
            tau_jj = getattr(self.config, 'tau_jj', 2.0)
            
            for i in range(num_edges):
                agent = pop[i]
                agent.setVariableInt("edge_id", i)
                agent.setVariableInt("edge_idx", i)  # Edge index = segment index in fallback
                agent.setVariableInt("segment_idx", 0)  # Only one "segment" per edge
                agent.setVariableInt("next_segment", -1)  # No next segment
                agent.setVariableInt("capacity", self.network.edge_capacities[i])
                agent.setVariableInt("curr_count", 0)
                agent.setVariableFloat("length", self.network.edge_lengths[i])
                agent.setVariableFloat("free_speed", self.network.edge_speeds[i])
                agent.setVariableInt("signal_id", self.network.edge_signal_ids[i])
                agent.setVariableInt("is_green", 1)
                travel_time = self.network.edge_lengths[i] / self.network.edge_speeds[i] if self.network.edge_speeds[i] > 0 else 1.0
                agent.setVariableFloat("travel_time", travel_time)
                from_node = self.network.edge_from_nodes[i] if i < len(self.network.edge_from_nodes) else -1
                agent.setVariableInt("from_node", from_node)
                agent.setVariableInt("out_node", self.network.edge_to_nodes[i])
                agent.setVariableInt("lane_count", self.network.edge_lanes[i])
                prio = self.network.edge_priorities[i] if i < len(self.network.edge_priorities) else 0
                agent.setVariableInt("priority_rank", prio)
                agent.setVariableInt("conflict_group", self.network.edge_to_nodes[i] if i < len(self.network.edge_to_nodes) else -1)
                agent.setVariableFloat("junction_block_until", 0.0)
                
                agent.setVariableFloat("tau_ff", tau_ff)
                agent.setVariableFloat("tau_fj", tau_fj)
                agent.setVariableFloat("tau_jf", tau_jf)
                agent.setVariableFloat("tau_jj", tau_jj)
                
                agent.setVariableFloat("jam_threshold", 0.8)
                agent.setVariableFloat("block_time", 0.0)
                agent.setVariableInt("is_jammed", 0)
                
                agent.setVariableFloat("interval_sampled_seconds", 0.0)
                agent.setVariableInt("interval_entered", 0)
                agent.setVariableInt("interval_left", 0)
        
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
                        if edge_idx >= 0 and self.network.segments and edge_idx < len(self.network.edge_last_segment):
                            phase_green_edges[i * max_edges_per_phase + j] = self.network.edge_last_segment[edge_idx]
                        else:
                            phase_green_edges[i * max_edges_per_phase + j] = edge_idx
            agent.setVariableArrayInt("phase_green_edges", phase_green_edges)
        
        # Add population to simulation
        self.simulation.setPopulationData(pop)
    
    def run(self, duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Run the simulation with interval-based data collection
        
        Args:
            duration: Optional override for simulation duration
            
        Returns:
            Results dictionary
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call initialize() first.")
        
        sim_duration = duration or self.config.duration
        total_steps = int(sim_duration / self.config.time_step)
        steps_per_interval = int(self.config.metrics_interval / self.config.time_step)
        
        if self.config.verbose:
            print(f"Starting simulation for {sim_duration}s...")
            start_time = time.time()
        else:
            start_time = time.time()
        
        # Create interval data collector for per-edge per-interval data
        from .metrics import IntervalEdgeDataCollector
        self._interval_collector = IntervalEdgeDataCollector(
            simulation=self.simulation,
            model=self.model,
            network_data=self.network,
            output_interval=self.config.metrics_interval,
            output_file=self.config.metrics_file
        )
        
        # Run step-by-step to collect interval data
        import pyflamegpu
        step = 0
        next_collection = steps_per_interval
        
        while step < total_steps:
            # Check if we need to collect BEFORE this step (so we get data before reset)
            if step == next_collection:
                current_time = step * self.config.time_step
                self._interval_collector.collect(current_time)
                if self.config.enable_calibrator_lite:
                    self._apply_calibrator_lite()
                self._interval_collector.last_collection_time = current_time  # Update for next interval
                next_collection += steps_per_interval
                # Set reset flag for THIS step
                self.simulation.setEnvironmentPropertyInt("reset_interval_counters", 1)
                # Reset spawn tracker for next interval
                try:
                    from .model import SpawnTracker
                    SpawnTracker.get_instance().reset()
                except ImportError:
                    pass
            else:
                # Clear reset flag
                self.simulation.setEnvironmentPropertyInt("reset_interval_counters", 0)
            
            # Run one step
            self.simulation.step()
            step += 1
        
        # Collect final partial interval if any
        final_time = step * self.config.time_step
        if final_time > self._interval_collector.last_collection_time:
            self._interval_collector.collect(final_time)
            if self.config.enable_calibrator_lite:
                self._apply_calibrator_lite()
            self._interval_collector.last_collection_time = final_time
        
        if self.config.verbose:
            elapsed = time.time() - start_time
            print(f"Simulation completed in {elapsed:.2f}s "
                  f"({sim_duration/elapsed:.1f}x real-time)")
        else:
            elapsed = time.time() - start_time
        
        # Collect final results
        self._collect_results()
        self._results["wall_time_seconds"] = elapsed
        self._results["sim_seconds_per_wall_second"] = (sim_duration / elapsed) if elapsed > 0 else 0.0
        
        # Save all metrics to Parquet
        self._interval_collector.save(self.config.metrics_file)
        
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
            "invalid_routes": len(self.demand.invalid_routes) if hasattr(self.demand, "invalid_routes") else 0,
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
        self._results["detector_view"] = self._build_detector_view()
        self._results["teleport_reasons_visible"] = self._count_visible_teleport_reasons(traveling_pop, waiting_pop)
        self._results["teleport_lifecycle_visible"] = self._count_visible_teleport_lifecycle(traveling_pop, waiting_pop)
    
    def _apply_calibrator_lite(self):
        """Bounded host-side correction to edge travel_time."""
        import pyflamegpu
        edge_desc = self.model.model.getAgent("EdgeQueue")
        edge_pop = pyflamegpu.AgentVector(edge_desc)
        self.simulation.getPopulationData(edge_pop)
        gain = max(0.0, min(0.5, self.config.calibrator_gain))
        for i in range(edge_pop.size()):
            agent = edge_pop[i]
            curr = agent.getVariableInt("curr_count")
            cap = max(1, agent.getVariableInt("capacity"))
            occ = float(curr) / float(cap)
            base = agent.getVariableFloat("length") / max(0.1, agent.getVariableFloat("free_speed"))
            tt = agent.getVariableFloat("travel_time")
            target = base * (1.0 + 0.8 * occ)
            corrected = (1.0 - gain) * tt + gain * target
            if corrected < base:
                corrected = base
            if corrected > base * 4.0:
                corrected = base * 4.0
            agent.setVariableFloat("travel_time", corrected)
        self.simulation.setPopulationData(edge_pop)
    
    def _build_detector_view(self) -> List[Dict[str, Any]]:
        """Project detector-style rows from interval edge metrics."""
        if not hasattr(self, "_interval_collector") or self._interval_collector is None:
            return []
        edge_df = self._interval_collector.get_edge_dataframe()
        if edge_df is None or edge_df.empty:
            return []
        latest = edge_df[edge_df["interval_end"] == edge_df["interval_end"].max()]
        out = []
        for _, row in latest.iterrows():
            out.append({
                "detector_id": f"det_{row['id']}",
                "edge_id": row["id"],
                "interval_end": float(row["interval_end"]),
                "flow": float(row["flow"]),
                "speed": float(row["speed"]),
                "occupancy": float(row["occupancy"]),
            })
        return out
    
    def _count_visible_teleport_reasons(self, traveling_pop, waiting_pop) -> Dict[str, int]:
        counts = {"jam": 0, "disconnected": 0, "route_end": 0}
        for pop in (traveling_pop, waiting_pop):
            for i in range(pop.size()):
                reason = pop[i].getVariableInt("teleport_reason")
                if reason == 1:
                    counts["jam"] += 1
                elif reason == 2:
                    counts["disconnected"] += 1
                elif reason == 3:
                    counts["route_end"] += 1
        return counts

    def _count_visible_teleport_lifecycle(self, traveling_pop, waiting_pop) -> Dict[str, int]:
        counts = {
            "single_jump": 0,
            "multi_step": 0,
            "reroutes_while_teleporting": 0,
            "failed_reentry_or_disconnected": 0,
            "teleport_hops": 0,
        }
        for pop in (traveling_pop, waiting_pop):
            for i in range(pop.size()):
                agent = pop[i]
                counts["single_jump"] += agent.getVariableInt("teleport_single_count")
                counts["multi_step"] += agent.getVariableInt("teleport_multi_count")
                counts["reroutes_while_teleporting"] += agent.getVariableInt("reroutes_while_teleporting")
                counts["teleport_hops"] += agent.getVariableInt("teleport_hops")
                if agent.getVariableInt("is_disconnected") == 1:
                    counts["failed_reentry_or_disconnected"] += 1
        return counts
    
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
    
    def export_results(self, filepath: str, format: str = None):
        """
        Export results to file
        
        Args:
            filepath: Output file path
            format: Output format ('json', 'parquet'). Auto-detected from extension if None.
        """
        if format is None:
            if filepath.endswith('.parquet'):
                format = 'parquet'
            else:
                format = 'json'
        
        if format == 'parquet':
            try:
                import pandas as pd
                # Convert results to DataFrame-friendly format
                summary = {k: v for k, v in self._results.items() if k != 'edge_stats'}
                df = pd.DataFrame([summary])
                df.to_parquet(filepath.replace('.parquet', '_summary.parquet'), index=False)
                
                # Save edge stats separately
                if 'edge_stats' in self._results:
                    df_edges = pd.DataFrame(self._results['edge_stats'])
                    df_edges.to_parquet(filepath.replace('.parquet', '_edges_final.parquet'), index=False)
                
                if self.config.verbose:
                    print(f"Results exported to {filepath}")
            except ImportError:
                print("pandas not available, falling back to JSON")
                format = 'json'
        
        if format == 'json':
            import json
            with open(filepath, 'w') as f:
                json.dump(self._results, f, indent=2)
            if self.config.verbose:
                print(f"Results exported to {filepath}")
    
    def get_metrics_dataframe(self):
        """
        Get network-level metrics as a pandas DataFrame
        
        Returns:
            pandas DataFrame with time-series network metrics, or None if not available
        """
        if hasattr(self, '_interval_collector') and self._interval_collector:
            return self._interval_collector.get_network_dataframe()
        return None
    
    def get_edge_data(self):
        """
        Get per-edge metrics as a pandas DataFrame (SUMO edgeData style)
        
        Columns include:
        - edge_id: Original edge ID  
        - vehicle_count: Vehicles on edge
        - density: Vehicles per km per lane
        - occupancy: Fraction of capacity used
        - speed: Average speed [m/s]
        - speed_relative: Speed relative to free flow
        - flow: Vehicles per hour
        - travel_time: Average travel time [s]
        
        Returns:
            pandas DataFrame with per-edge metrics, or None if not available
        """
        # Return per-edge per-interval data from collector
        if hasattr(self, '_interval_collector') and self._interval_collector:
            return self._interval_collector.get_edge_dataframe()
        return None
    
    def save_metrics(self, filepath: str = None):
        """
        Save collected metrics to Parquet files
        
        Args:
            filepath: Output path (defaults to config.metrics_file)
        """
        # Saving is handled by the interval collector in run()
        pass


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
        edge_priorities=[e.get('priority', 0) for e in edges],
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


def evaluate_kpi_gate(baseline: Dict[str, Any], candidate: Dict[str, Any], max_regression_pct: float = 5.0) -> Dict[str, Any]:
    """Simple rollout gate: fail if throughput regresses over threshold."""
    base_tp = float(baseline.get("sim_seconds_per_wall_second", 0.0) or 0.0)
    cand_tp = float(candidate.get("sim_seconds_per_wall_second", 0.0) or 0.0)
    if base_tp <= 0.0:
        return {"pass": True, "regression_pct": 0.0, "reason": "no-baseline-throughput"}
    regression_pct = ((base_tp - cand_tp) / base_tp) * 100.0
    return {
        "pass": regression_pct <= max_regression_pct,
        "regression_pct": regression_pct,
        "base_throughput": base_tp,
        "candidate_throughput": cand_tp,
    }

