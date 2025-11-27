"""
Model Definition for FLAMEGPU2 Mesoscopic Traffic Simulation

This module creates and configures the complete FLAMEGPU2 model including:
- Agent definitions
- Message definitions
- Environment properties
- Execution layers
- Host functions

Reference: FLAMEGPU2 Model Documentation
https://docs.flamegpu.com/guide/creating-a-model/
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import sys

# Try to import pyflamegpu - if not available, provide stubs for testing
try:
    import pyflamegpu
    FLAMEGPU_AVAILABLE = True
except ImportError:
    FLAMEGPU_AVAILABLE = False
    print("Warning: pyflamegpu not available. Using stub implementation.")

from .agents import (
    define_edge_queue_agent,
    define_packet_agent,
    define_signal_controller_agent,
    EdgeQueueConfig,
    PacketConfig,
    SignalControllerConfig,
)
from .messages import define_messages, MessageConfig


@dataclass
class EnvironmentConfig:
    """Configuration for environment properties"""
    time_step: float = 1.0                    # Simulation time step [seconds]
    max_edges: int = 100000                   # Maximum number of edges (now unlimited - RTC doesn't use env arrays)
    jam_density: float = 0.15                 # Jam density [veh/m]
    wave_speed: float = 5.0                   # Backward wave speed [m/s]
    congestion_threshold: float = 0.8         # Occupancy threshold for congestion
    
    # Fundamental diagram parameters
    fd_model: str = "newell_daganzo"          # 'greenshields' or 'newell_daganzo'
    
    # SUMO compatibility
    tau_ff: float = 1.4                       # Free-free TAU factor
    tau_fj: float = 1.4                       # Free-jam TAU factor
    tau_jf: float = 2.0                       # Jam-free TAU factor
    tau_jj: float = 1.4                       # Jam-jam TAU factor


@dataclass
class ModelConfig:
    """Complete model configuration"""
    name: str = "MesoscopicTrafficModel"
    
    # Component configs
    edge_config: EdgeQueueConfig = field(default_factory=EdgeQueueConfig)
    packet_config: PacketConfig = field(default_factory=PacketConfig)
    signal_config: SignalControllerConfig = field(default_factory=SignalControllerConfig)
    message_config: MessageConfig = field(default_factory=MessageConfig)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)


class MesoscopicTrafficModel:
    """
    FLAMEGPU2 Mesoscopic Traffic Simulation Model
    
    This class encapsulates the complete model definition including:
    - Agent types (EdgeQueue, Packet, SignalController)
    - Message types (entry_request, entry_accept, departure_notice, green_signal)
    - Environment properties
    - Execution layers
    
    Usage:
        model = MesoscopicTrafficModel()
        model.build()
        simulation = model.create_simulation()
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model
        
        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.model = None
        self.agents = {}
        self.messages = {}
        self.layers = []
        self._host_functions = []
        self._init_functions = []
        self._step_functions = []
        self._exit_functions = []
        
    def build(self) -> 'MesoscopicTrafficModel':
        """
        Build the complete FLAMEGPU2 model
        
        Returns:
            self for method chaining
        """
        if not FLAMEGPU_AVAILABLE:
            raise RuntimeError("pyflamegpu is not available. Install with: pip install pyflamegpu")
        
        # Create model
        self.model = pyflamegpu.ModelDescription(self.config.name)
        
        # Define messages first (agents reference them)
        self._define_messages()
        
        # Define agents
        self._define_agents()
        
        # Define environment
        self._define_environment()
        
        # Define execution layers
        self._define_layers()
        
        return self
    
    def _define_messages(self):
        """Define all message types"""
        self.messages = define_messages(self.model, self.config.message_config)
    
    def _define_agents(self):
        """Define all agent types"""
        self.agents["EdgeQueue"] = define_edge_queue_agent(
            self.model, self.config.edge_config)
        self.agents["Packet"] = define_packet_agent(
            self.model, self.config.packet_config)
        self.agents["SignalController"] = define_signal_controller_agent(
            self.model, self.config.signal_config)
    
    def _define_environment(self):
        """Define environment properties - ONLY scalars, no arrays!
        
        Large arrays cause CUDA JIT linking errors on newer GPUs (sm_120+).
        All edge-specific data is stored in EdgeQueue agent variables and
        passed through messages, following the Boids example pattern.
        """
        env = self.model.Environment()
        cfg = self.config.environment_config
        
        # Simulation parameters (scalars only)
        env.newPropertyFloat("time_step", cfg.time_step)
        env.newPropertyFloat("current_time", 0.0)
        
        # Traffic flow parameters (scalars)
        env.newPropertyFloat("jam_density", cfg.jam_density)
        env.newPropertyFloat("wave_speed", cfg.wave_speed)
        env.newPropertyFloat("congestion_threshold", cfg.congestion_threshold)
        
        # SUMO TAU factors (scalars)
        env.newPropertyFloat("tau_ff", cfg.tau_ff)
        env.newPropertyFloat("tau_fj", cfg.tau_fj)
        env.newPropertyFloat("tau_jf", cfg.tau_jf)
        env.newPropertyFloat("tau_jj", cfg.tau_jj)
        
        # NOTE: Edge property arrays removed - they cause CUDA JIT errors on sm_120+
        # All edge data is stored in EdgeQueue agent variables instead:
        # - length, free_speed, capacity, travel_time are EdgeQueue variables
        # - Packets receive travel_time through acceptance messages
        
    def _define_layers(self):
        """
        Define execution layers to control agent function ordering
        
        Execution order per step:
        1. L1_departure - Packets send departure notices
        2. L2_process_departures - Edges process departures
        3. L3_move - Packets move and send entry requests
        4. L4_signal - Signals update phases and broadcast green
        5. L5_green_flag - Edges update green flags
        6. L6_broadcast - Edges broadcast status (for GPU rerouting)
        7. L7_reroute - Stuck packets try to find alternatives
        8. L8_process_requests - Edges process entry requests
        9. L9_wait - Waiting packets check for acceptance
        """
        # Layer 1: Packets send departure notices (traveling state)
        layer1 = self.model.newLayer("L1_departure")
        layer1.addAgentFunction("Packet", "send_departure")
        
        # Layer 2: Edges process departures
        layer2 = self.model.newLayer("L2_process_departures")
        layer2.addAgentFunction("EdgeQueue", "process_departures")
        
        # Layer 3: Packets move and request entry (traveling → waiting)
        layer3 = self.model.newLayer("L3_move")
        layer3.addAgentFunction("Packet", "move_and_request")
        
        # Layer 4: Signals update and broadcast green
        layer4 = self.model.newLayer("L4_signal")
        layer4.addAgentFunction("SignalController", "update_signal")
        
        # Layer 5: Edges update green flags
        layer5 = self.model.newLayer("L5_green_flag")
        layer5.addAgentFunction("EdgeQueue", "update_green_flag")
        
        # Layer 6: Edges broadcast status for GPU-side rerouting
        layer6 = self.model.newLayer("L6_broadcast")
        layer6.addAgentFunction("EdgeQueue", "broadcast_status")
        
        # Layer 7: Stuck packets try to find alternative routes (GPU-side)
        layer7 = self.model.newLayer("L7_reroute")
        layer7.addAgentFunction("Packet", "try_reroute")
        
        # Layer 8: Edges process entry requests
        layer8 = self.model.newLayer("L8_process_requests")
        layer8.addAgentFunction("EdgeQueue", "process_edge_requests")
        
        # Layer 9: Waiting packets check for acceptance (waiting → traveling)
        layer9 = self.model.newLayer("L9_wait")
        layer9.addAgentFunction("Packet", "wait_for_entry")
        
        self.layers = [layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9]
    
    def add_init_function(self, func: Callable):
        """Add an initialization function"""
        self._init_functions.append(func)
        if self.model:
            self.model.addInitFunction(func)
    
    def add_step_function(self, func: Callable):
        """Add a step function (runs each simulation step)"""
        self._step_functions.append(func)
        if self.model:
            self.model.addStepFunction(func)
    
    def add_exit_function(self, func: Callable):
        """Add an exit function (runs at simulation end)"""
        self._exit_functions.append(func)
        if self.model:
            self.model.addExitFunction(func)
    
    def create_simulation(self):
        """
        Create a CUDASimulation from this model
        
        Returns:
            pyflamegpu.CUDASimulation
        """
        if not self.model:
            raise RuntimeError("Model not built. Call build() first.")
        
        if not FLAMEGPU_AVAILABLE:
            raise RuntimeError("pyflamegpu is not available")
        
        return pyflamegpu.CUDASimulation(self.model)
    
    def get_model_description(self):
        """Get the underlying ModelDescription"""
        return self.model


def create_model(config: Optional[ModelConfig] = None) -> MesoscopicTrafficModel:
    """
    Factory function to create and build a mesoscopic traffic model
    
    Args:
        config: Optional model configuration
        
    Returns:
        Built MesoscopicTrafficModel
    """
    model = MesoscopicTrafficModel(config)
    model.build()
    return model


# =============================================================================
# Host Functions (must subclass pyflamegpu.HostFunction)
# =============================================================================

def create_time_update_function():
    """
    Create a host function that updates simulation time
    """
    if not FLAMEGPU_AVAILABLE:
        return None
    
    class TimeUpdateFunction(pyflamegpu.HostFunction):
        def run(self, host_api):
            step = host_api.getStepCounter()
            dt = host_api.environment.getPropertyFloat("time_step")
            current_time = step * dt
            host_api.environment.setPropertyFloat("current_time", current_time)
    
    return TimeUpdateFunction()


def create_spawn_packets_function(departures: List[tuple], edge_id_map: Dict[str, int],
                                   edge_lengths: List[float], edge_speeds: List[float],
                                   edge_to_nodes: List[int] = None,
                                   max_route_length: int = 256):
    """
    Create a host function that spawns packets according to departure schedule
    
    Args:
        departures: List of (time, origin_edge_id, route_list, count) tuples
        edge_id_map: Mapping from edge string IDs to integer indices
        edge_lengths: List of edge lengths (indexed by edge_id)
        edge_speeds: List of edge speeds (indexed by edge_id)
        edge_to_nodes: List of destination node indices (for rerouting)
        
    Returns:
        Host function
    """
    if not FLAMEGPU_AVAILABLE:
        return None
    
    # Sort departures by time
    sorted_deps = sorted(departures, key=lambda x: x[0])
    
    # Store edge data locally (not in CUDA env arrays - they cause JIT errors)
    _edge_lengths = list(edge_lengths)
    _edge_speeds = list(edge_speeds)
    _edge_to_nodes = list(edge_to_nodes) if edge_to_nodes else []
    _max_route = max_route_length  # Capture for closure
    
    class SpawnPacketsFunction(pyflamegpu.HostFunction):
        def __init__(self):
            super().__init__()
            self.depart_index = 0
            self.spawned_count = 0
        
        def run(self, host_api):
            step = host_api.getStepCounter()
            dt = host_api.environment.getPropertyFloat("time_step")
            current_time = step * dt
            
            # Spawn all packets scheduled for this time
            while (self.depart_index < len(sorted_deps) and 
                   sorted_deps[self.depart_index][0] <= current_time + dt * 0.5):
                
                t, origin, route_list, count = sorted_deps[self.depart_index]
                
                # Convert route to edge indices
                route_indices = [edge_id_map.get(e, -1) for e in route_list]
                
                # Pad route to max length (must match PacketConfig.max_route_length)
                padded_route = (route_indices + [-1] * _max_route)[:_max_route]
                
                # Get edge properties from local Python data (not CUDA env)
                origin_idx = route_indices[0] if route_indices else -1
                if origin_idx < 0 or origin_idx >= len(_edge_lengths):
                    self.depart_index += 1
                    continue
                
                length = _edge_lengths[origin_idx]
                speed = _edge_speeds[origin_idx]
                
                # Create packet agent using HostAgentAPI.newAgent()
                packet = host_api.agent("Packet", "traveling").newAgent()
                packet.setVariableInt("size", count)
                packet.setVariableInt("curr_edge", origin_idx)
                packet.setVariableInt("next_edge", route_indices[1] if len(route_indices) > 1 else -1)
                packet.setVariableArrayInt("route", padded_route)
                packet.setVariableInt("route_idx", 0)
                packet.setVariableInt("route_length", len(route_indices))
                packet.setVariableFloat("remaining_time", length / speed if speed > 0 else 1.0)
                packet.setVariableFloat("entry_time", current_time)
                packet.setVariableInt("destination", route_indices[-1] if route_indices else -1)
                packet.setVariableFloat("wait_time", 0.0)
                
                # Set curr_node (end of start edge) and dest_node (end of destination edge) for rerouting
                dest_idx = route_indices[-1] if route_indices else -1
                dest_node = _edge_to_nodes[dest_idx] if (dest_idx >= 0 and dest_idx < len(_edge_to_nodes)) else -1
                curr_node = _edge_to_nodes[origin_idx] if (origin_idx >= 0 and origin_idx < len(_edge_to_nodes)) else -1
                packet.setVariableInt("curr_node", curr_node)
                packet.setVariableInt("dest_node", dest_node)
                
                self.spawned_count += 1
                self.depart_index += 1
                
                # Debug: print first few spawns
                if self.spawned_count <= 5:
                    print(f"  [Spawn] t={current_time:.1f}s packet #{self.spawned_count}: edge={origin_idx}, size={count}, route_len={len(route_indices)}")
    
    return SpawnPacketsFunction()


def create_logging_function(output_interval: float = 60.0):
    """
    Create a host function that logs simulation metrics periodically
    
    Args:
        output_interval: Time interval between outputs [seconds]
        
    Returns:
        Host function
    """
    if not FLAMEGPU_AVAILABLE:
        return None
    
    class LoggingFunction(pyflamegpu.HostFunction):
        def __init__(self):
            super().__init__()
            self.last_log_time = 0.0
            self.last_packet_count = 0
            self.total_spawned = 0
        
        def run(self, host_api):
            current_time = host_api.environment.getPropertyFloat("current_time")
            
            if current_time - self.last_log_time >= output_interval:
                # Get agent counts - Packet uses explicit states
                traveling = host_api.agent("Packet", "traveling").count()
                waiting = host_api.agent("Packet", "waiting").count()
                packet_count = traveling + waiting
                
                # Calculate deaths (negative delta means deaths exceeded spawns)
                delta = packet_count - self.last_packet_count
                
                # Sample a few packets to check their state
                if traveling > 0:
                    traveling_api = host_api.agent("Packet", "traveling")
                    avg_route_idx = traveling_api.sumInt("route_idx") / traveling
                    avg_route_len = traveling_api.sumInt("route_length") / traveling
                    avg_rem_time = traveling_api.sumFloat("remaining_time") / traveling
                    print(f"[t={current_time:.1f}s] Packets: {packet_count} Δ={delta:+d} | idx={avg_route_idx:.1f}/{avg_route_len:.1f} rem={avg_rem_time:.1f}s")
                else:
                    print(f"[t={current_time:.1f}s] Packets: {packet_count} Δ={delta:+d}")
                
                self.last_packet_count = packet_count
                self.last_log_time = current_time
    
    return LoggingFunction()


def create_rerouting_function(
    adjacency: Dict[int, List[int]],  # node_id -> list of outgoing edge_ids
    edge_to_node: List[int],          # edge_id -> destination node_id
    edge_from_node: List[int],        # edge_id -> origin node_id  
    edge_lengths: List[float],
    edge_speeds: List[float],
    max_route_length: int = 256,
    reroute_interval: float = 60.0,   # How often to check for rerouting
    wait_threshold: float = 30.0,     # Reroute if waiting longer than this
):
    """
    Create a host function that reroutes stuck packets using Dijkstra's algorithm
    
    Args:
        adjacency: Node to outgoing edges mapping
        edge_to_node: Edge to destination node mapping
        edge_from_node: Edge to origin node mapping
        edge_lengths: Edge lengths for travel time calculation
        edge_speeds: Edge speeds for travel time calculation
        max_route_length: Maximum route length for packets
        reroute_interval: Time between rerouting checks [seconds]
        wait_threshold: Reroute packets waiting longer than this [seconds]
        
    Returns:
        Host function for rerouting
    """
    if not FLAMEGPU_AVAILABLE:
        return None
    
    import heapq
    
    # Build reverse mapping: destination_node -> edges ending there
    edges_to_node = {}
    for edge_id, to_node in enumerate(edge_to_node):
        if to_node not in edges_to_node:
            edges_to_node[to_node] = []
        edges_to_node[to_node].append(edge_id)
    
    def dijkstra_shortest_path(from_edge: int, to_edge: int, 
                                edge_travel_times: List[float]) -> List[int]:
        """
        Compute shortest path from from_edge to to_edge using Dijkstra
        
        Returns:
            List of edge IDs forming the path, or empty list if no path
        """
        if from_edge < 0 or to_edge < 0:
            return []
        if from_edge == to_edge:
            return [to_edge]
        
        # Start from the node at the end of from_edge
        start_node = edge_to_node[from_edge]
        target_node = edge_to_node[to_edge]
        
        # Priority queue: (distance, node, path)
        pq = [(0.0, start_node, [from_edge])]
        visited = set()
        
        while pq:
            dist, node, path = heapq.heappop(pq)
            
            if node in visited:
                continue
            visited.add(node)
            
            # Check if we reached destination edge
            if node == target_node:
                return path
            
            # Explore outgoing edges from this node
            if node in adjacency:
                for next_edge in adjacency[node]:
                    next_node = edge_to_node[next_edge]
                    if next_node not in visited:
                        tt = edge_travel_times[next_edge] if next_edge < len(edge_travel_times) else float('inf')
                        new_path = path + [next_edge]
                        if len(new_path) < max_route_length:
                            heapq.heappush(pq, (dist + tt, next_node, new_path))
        
        return []  # No path found
    
    class ReroutingFunction(pyflamegpu.HostFunction):
        def __init__(self):
            super().__init__()
            self.last_reroute_time = 0.0
            self.reroute_count = 0
        
        def run(self, host_api):
            current_time = host_api.environment.getPropertyFloat("current_time")
            
            # Only reroute periodically
            if current_time - self.last_reroute_time < reroute_interval:
                return
            self.last_reroute_time = current_time
            
            # Build travel time array (could be enhanced to use actual congestion)
            edge_travel_times = [l / s if s > 0 else float('inf') 
                                  for l, s in zip(edge_lengths, edge_speeds)]
            
            # Get packets in traveling state
            traveling_api = host_api.agent("Packet", "traveling")
            traveling_count = traveling_api.count()
            
            if traveling_count == 0:
                return
            
            # Get packet population for modification
            # getPopulationData() returns an AgentVector directly
            packet_pop = host_api.agent("Packet", "traveling").getPopulationData()
            
            rerouted_this_step = 0
            
            for i in range(len(packet_pop)):
                agent = packet_pop[i]
                wait_time = agent.getVariableFloat("wait_time")
                
                # Only reroute if waiting too long
                if wait_time < wait_threshold:
                    continue
                
                curr_edge = agent.getVariableInt("curr_edge")
                destination = agent.getVariableInt("destination")
                
                if destination < 0 or curr_edge < 0:
                    continue
                
                # Find new route from current edge to destination
                new_route = dijkstra_shortest_path(curr_edge, destination, edge_travel_times)
                
                if len(new_route) < 2:
                    # No alternative path found, skip
                    continue
                
                # Check if new route is different from current next_edge
                old_next = agent.getVariableInt("next_edge")
                new_next = new_route[1] if len(new_route) > 1 else -1
                
                if new_next == old_next:
                    # Same route, no point rerouting
                    continue
                
                # Update packet's route
                padded_route = (new_route + [-1] * max_route_length)[:max_route_length]
                agent.setVariableArrayInt("route", padded_route)
                agent.setVariableInt("route_length", len(new_route))
                agent.setVariableInt("route_idx", 0)  # Reset to start of new route
                agent.setVariableInt("next_edge", new_next)
                agent.setVariableFloat("wait_time", 0.0)  # Reset wait time
                
                rerouted_this_step += 1
            
            # Write back modified population
            if rerouted_this_step > 0:
                host_api.agent("Packet", "traveling").setPopulationData(packet_pop)
                self.reroute_count += rerouted_this_step
                print(f"  [Reroute] t={current_time:.1f}s: Rerouted {rerouted_this_step} packets (total: {self.reroute_count})")
    
    return ReroutingFunction()

