"""
SUMO File Parsers for Network and Demand Data

This module parses SUMO XML files and converts them to the format
required by the FLAMEGPU2 mesoscopic simulation.

Supported files:
- .net.xml - Network definition (nodes, edges, lanes, signals)
- .rou.xml - Routes and demand (vehicles, flows, routes)
- .trips.xml - Trip definitions (origin-destination pairs)
- .sumocfg - SUMO configuration file with simulation parameters

Reference: SUMO Documentation
https://sumo.dlr.de/docs/index.html
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import re

# Import core types
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.simulation import NetworkData, DemandData, SegmentData


def split_edge_into_segments(
    edge_idx: int,
    edge_id: str,
    edge_length: float,
    edge_speed: float,
    edge_lanes: int,
    edge_capacity: int,
    edge_from_node: int,
    edge_to_node: int,
    edge_signal_id: int,
    segment_length: float = 100.0,
    jam_density: float = 0.15,
    min_capacity: int = 5,
) -> List[SegmentData]:
    """
    Split an edge into SUMO-style segments (~100m each).
    
    In SUMO meso, edges are divided into segments for finer-grained traffic flow.
    Each segment acts as an independent queue with its own capacity and headway.
    
    Args:
        edge_idx: Index of the parent edge
        edge_id: ID of the parent edge
        edge_length: Total edge length [m]
        edge_speed: Speed limit [m/s]
        edge_lanes: Number of lanes
        edge_capacity: Total edge capacity
        edge_from_node: Start node index
        edge_to_node: End node index
        edge_signal_id: Signal ID (-1 if none)
        segment_length: Target segment length [m] (SUMO default: 98-100m)
        jam_density: Jam density for capacity calculation [veh/m/lane]
        min_capacity: Minimum capacity per segment
        
    Returns:
        List of SegmentData objects
    """
    # Calculate number of segments (SUMO formula)
    num_segments = max(1, round(edge_length / segment_length))
    actual_segment_length = edge_length / num_segments
    
    segments = []
    for seg_idx in range(num_segments):
        # Segment ID format matches SUMO: "edge_id:segment_idx"
        seg_id = f"{edge_id}:{seg_idx}"
        
        # Calculate segment capacity
        seg_capacity = max(min_capacity, int(actual_segment_length * edge_lanes * jam_density))
        
        # Next segment is seg_idx + 1, or -1 if this is the last segment
        next_seg = -1  # Will be set after all segments created
        
        # From/to nodes: only meaningful for first/last segments
        from_node = edge_from_node if seg_idx == 0 else -1
        to_node = edge_to_node if seg_idx == num_segments - 1 else -1
        
        # Signal only on last segment (junction)
        signal_id = edge_signal_id if seg_idx == num_segments - 1 else -1
        
        segments.append(SegmentData(
            segment_id=seg_id,
            edge_id=edge_id,
            edge_idx=edge_idx,
            segment_idx=seg_idx,
            length=actual_segment_length,
            speed=edge_speed,
            capacity=seg_capacity,
            lanes=edge_lanes,
            next_segment=-1,  # Will be set later
            from_node=from_node,
            to_node=to_node,
            signal_id=signal_id,
        ))
    
    return segments


@dataclass
class SUMOMesoConfig:
    """SUMO Mesoscopic simulation parameters from .sumocfg
    
    Reference: https://sumo.dlr.de/docs/Simulation/Meso.html
    """
    # Mesoscopic TAU factors (headway times in seconds)
    # These control the minimum time between vehicles leaving a segment
    tau_ff: float = 1.4      # Free-flow to free-flow headway
    tau_fj: float = 1.4      # Free-flow to jam headway
    tau_jf: float = 2.0      # Jam to free-flow headway  
    tau_jj: float = 2.0      # Jam to jam headway (SUMO default is 2.0)
    
    # Queue/segment parameters
    meso_edgelength: float = 98.0    # Segment length for queues [m]
    jam_threshold: float = -1.0       # Jam threshold (negative = auto-calculate)
    multi_queue: bool = True          # Use multi-queue model per segment
    junction_control: bool = True     # Junction-based flow control
    lane_queue: bool = False          # Per-lane queuing (vs per-edge)
    overtaking: bool = False          # Allow faster vehicles to overtake
    
    # Penalties (added to travel time)
    tls_penalty: float = 0.0          # Traffic light penalty [s]
    tls_flow_penalty: float = 0.0     # TLS flow-based penalty factor
    minor_penalty: float = 0.0        # Minor road priority penalty [s]
    
    # Rerouting parameters  
    rerouting_probability: float = 0.0   # Fraction of vehicles that can reroute
    rerouting_period: float = 60.0       # Rerouting check interval [s]
    rerouting_adaptation_interval: float = 1.0  # Edge weight adaptation interval [s]
    rerouting_adaptation_steps: int = 180  # Number of steps for moving average
    rerouting_adaptation_weight: float = 0.5  # Exponential smoothing weight
    rerouting_threshold_factor: float = 1.0  # Threshold factor to trigger reroute
    rerouting_threshold_constant: float = 0.0  # Constant threshold to trigger reroute
    routing_algorithm: str = "astar"  # Routing algorithm (dijkstra, astar, CH)
    weights_random_factor: float = 1.0  # Random factor for edge weights
    weights_priority_factor: float = 0.0  # Priority-based weight factor
    
    # Processing parameters
    time_to_teleport: float = 180.0   # Teleport stuck vehicles after N seconds
    time_to_impatience: float = 7.0   # Time until impatience kicks in [s]
    teleport_disconnected: float = 1.0  # Teleport on disconnected edges [s]
    ignore_junction_blocker: float = 1.0  # Ignore blocking at junctions
    
    # Time parameters
    step_length: float = 1.0          # Simulation step length [s]
    begin_time: float = 0.0           # Simulation start time [s]
    end_time: float = 86400.0         # Simulation end time [s]
    
    # Random seed
    seed: int = 42
    
    # Scale factor for demand
    scale: float = 1.0


class SUMOConfigParser:
    """
    Parser for SUMO .sumocfg configuration files
    
    Extracts mesoscopic, routing, and processing parameters.
    
    Usage:
        parser = SUMOConfigParser()
        config = parser.parse("simulation.sumocfg")
    """
    
    def parse(self, filepath: str) -> SUMOMesoConfig:
        """Parse a .sumocfg file and return configuration"""
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        config = SUMOMesoConfig()
        
        # Parse mesoscopic section
        meso = root.find("mesoscopic")
        if meso is not None:
            config.tau_ff = float(meso.findtext("meso-tauff", default=str(config.tau_ff)) 
                                  or meso.find("meso-tauff").get("value", str(config.tau_ff)) 
                                  if meso.find("meso-tauff") is not None else config.tau_ff)
            config.tau_fj = self._get_value(meso, "meso-taufj", config.tau_fj)
            config.tau_jf = self._get_value(meso, "meso-taujf", config.tau_jf)
            config.tau_jj = self._get_value(meso, "meso-taujj", config.tau_jj)
            config.meso_edgelength = self._get_value(meso, "meso-edgelength", config.meso_edgelength)
            config.jam_threshold = self._get_value(meso, "meso-jam-threshold", config.jam_threshold)
            config.multi_queue = self._get_bool(meso, "meso-multi-queue", config.multi_queue)
            config.junction_control = self._get_bool(meso, "meso-junction-control", config.junction_control)
            config.lane_queue = self._get_bool(meso, "meso-lane-queue", config.lane_queue)
            config.overtaking = self._get_bool(meso, "meso-overtaking", config.overtaking)
            config.tls_penalty = self._get_value(meso, "meso-tls-penalty", config.tls_penalty)
            config.tls_flow_penalty = self._get_value(meso, "meso-tls-flow-penalty", config.tls_flow_penalty)
            config.minor_penalty = self._get_value(meso, "meso-minor-penalty", config.minor_penalty)
        
        # Parse processing section
        processing = root.find("processing")
        if processing is not None:
            config.time_to_teleport = self._get_value(processing, "time-to-teleport", config.time_to_teleport)
            config.time_to_impatience = self._get_value(processing, "time-to-impatience", config.time_to_impatience)
            config.teleport_disconnected = self._get_value(processing, "time-to-teleport.disconnected", config.teleport_disconnected)
            config.ignore_junction_blocker = self._get_value(processing, "ignore-junction-blocker", config.ignore_junction_blocker)
            config.scale = self._get_value(processing, "scale", config.scale)
        
        # Parse routing section (some params may be in routing subsection)
        routing = root.find("routing")
        if routing is not None:
            config.rerouting_probability = self._get_value(routing, "device.rerouting.probability", config.rerouting_probability)
            config.rerouting_period = self._get_value(routing, "device.rerouting.period", config.rerouting_period)
            config.rerouting_adaptation_interval = self._get_value(routing, "device.rerouting.adaptation-interval", config.rerouting_adaptation_interval)
            config.rerouting_adaptation_steps = int(self._get_value(routing, "device.rerouting.adaptation-steps", config.rerouting_adaptation_steps))
            config.rerouting_adaptation_weight = self._get_value(routing, "device.rerouting.adaptation-weight", config.rerouting_adaptation_weight)
            config.rerouting_threshold_factor = self._get_value(routing, "device.rerouting.threshold.factor", config.rerouting_threshold_factor)
            config.rerouting_threshold_constant = self._get_value(routing, "device.rerouting.threshold.constant", config.rerouting_threshold_constant)
            config.routing_algorithm = self._get_str(routing, "routing-algorithm", config.routing_algorithm)
            config.weights_random_factor = self._get_value(routing, "weights.random-factor", config.weights_random_factor)
            config.weights_priority_factor = self._get_value(routing, "weights.priority-factor", config.weights_priority_factor)
        
        # Parse time section
        time_section = root.find("time")
        if time_section is not None:
            config.step_length = self._get_value(time_section, "step-length", config.step_length)
            config.begin_time = self._get_value(time_section, "begin", config.begin_time)
            config.end_time = self._get_value(time_section, "end", config.end_time)
        
        # Parse random seed
        random_section = root.find("random_number")
        if random_section is not None:
            config.seed = int(self._get_value(random_section, "seed", config.seed))
        
        return config
    
    def _get_value(self, parent, name: str, default: float) -> float:
        """Get float value from element with 'value' attribute"""
        elem = parent.find(name)
        if elem is not None:
            val = elem.get("value")
            if val is not None:
                try:
                    return float(val)
                except ValueError:
                    pass
        return default
    
    def _get_bool(self, parent, name: str, default: bool) -> bool:
        """Get boolean value from element"""
        elem = parent.find(name)
        if elem is not None:
            val = elem.get("value", "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
        return default
    
    def _get_str(self, parent, name: str, default: str) -> str:
        """Get string value from element"""
        elem = parent.find(name)
        if elem is not None:
            val = elem.get("value")
            if val is not None:
                return val
        return default
    
    def get_input_files(self, filepath: str) -> Dict[str, str]:
        """Extract input file paths from config (relative to config location)"""
        tree = ET.parse(filepath)
        root = tree.getroot()
        config_dir = Path(filepath).parent
        
        files = {}
        input_section = root.find("input")
        if input_section is not None:
            net_elem = input_section.find("net-file")
            if net_elem is not None:
                files["network"] = str(config_dir / net_elem.get("value", ""))
            
            route_elem = input_section.find("route-files")
            if route_elem is not None:
                files["routes"] = str(config_dir / route_elem.get("value", ""))
        
        return files


@dataclass
class SUMOEdge:
    """Parsed SUMO edge data"""
    id: str
    from_node: str
    to_node: str
    length: float
    speed: float
    lanes: int
    priority: int = 0
    type_id: str = ""


@dataclass
class SUMONode:
    """Parsed SUMO node/junction data"""
    id: str
    x: float
    y: float
    type: str  # 'priority', 'traffic_light', etc.


@dataclass
class SUMOSignal:
    """Parsed SUMO traffic light data"""
    id: str
    node_id: str
    phases: List[Dict[str, Any]]
    cycle_length: float


class SUMONetworkParser:
    """
    Parser for SUMO .net.xml network files
    
    Extracts:
    - Edges (road segments)
    - Nodes (junctions)
    - Traffic lights (signal controllers)
    - Connection topology
    - Segments (SUMO meso-style ~100m chunks)
    
    Usage:
        parser = SUMONetworkParser()
        network = parser.parse("network.net.xml")
    """
    
    def __init__(self, 
                 jam_density: float = 0.15,  # Realistic jam density
                 min_capacity: int = 20,     # Minimum capacity per edge to prevent bottlenecks
                 default_speed: float = 13.89,
                 skip_internal: bool = True,
                 segment_length: float = 100.0,  # SUMO meso default ~98-100m
                 use_segments: bool = True):  # Enable segment-based simulation
        """
        Args:
            jam_density: Jam density for capacity calculation [veh/m/lane]
            min_capacity: Minimum capacity per edge (prevents tiny edges from blocking)
            default_speed: Default speed if not specified [m/s]
            skip_internal: Skip internal/connector edges
            segment_length: Target segment length for SUMO meso compatibility [m]
            use_segments: Whether to split edges into segments
        """
        self.jam_density = jam_density
        self.min_capacity = min_capacity
        self.default_speed = default_speed
        self.skip_internal = skip_internal
        self.segment_length = segment_length
        self.use_segments = use_segments
        
        self.edges: Dict[str, SUMOEdge] = {}
        self.nodes: Dict[str, SUMONode] = {}
        self.signals: Dict[str, SUMOSignal] = {}
        self.connections: Dict[str, List[str]] = {}  # from_edge -> [to_edges]
        self.connection_records: List[Dict[str, Any]] = []
        self.junction_requests: Dict[str, List[Dict[str, Any]]] = {}
    
    def parse(self, filepath: str) -> NetworkData:
        """
        Parse a SUMO network file
        
        Args:
            filepath: Path to .net.xml file
            
        Returns:
            NetworkData ready for simulation
        """
        # Handle encoding issues (SUMO files may have Latin-1 chars like ö, ü, ä)
        try:
            tree = ET.parse(filepath)
        except ET.ParseError:
            # Try reading with Latin-1 and converting to UTF-8
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
            # Parse from string
            root = ET.fromstring(content)
            tree = ET.ElementTree(root)
        root = tree.getroot()
        
        # Parse nodes (junctions)
        self._parse_nodes(root)
        
        # Parse edges
        self._parse_edges(root)
        
        # Parse connections
        self._parse_connections(root)
        
        # Parse traffic lights (needs connection linkIndex mapping)
        self._parse_traffic_lights(root)
        
        # Convert to NetworkData
        return self._to_network_data()
    
    def _parse_nodes(self, root: ET.Element):
        """Parse junction elements"""
        for junction in root.findall('.//junction'):
            node_id = junction.get('id', '')
            
            # Skip internal junctions
            if self.skip_internal and node_id.startswith(':'):
                continue
            
            self.nodes[node_id] = SUMONode(
                id=node_id,
                x=float(junction.get('x', 0)),
                y=float(junction.get('y', 0)),
                type=junction.get('type', 'priority')
            )
            # Parse right-of-way request matrix hints (if present).
            reqs = []
            for req in junction.findall('request'):
                reqs.append({
                    "index": int(req.get("index", "-1")),
                    "response": req.get("response", ""),
                    "foes": req.get("foes", ""),
                    "cont": int(req.get("cont", "0")),
                })
            self.junction_requests[node_id] = reqs
    
    def _parse_edges(self, root: ET.Element):
        """Parse edge elements"""
        for edge_elem in root.findall('.//edge'):
            edge_id = edge_elem.get('id', '')
            
            # Skip internal edges
            if self.skip_internal and edge_id.startswith(':'):
                continue
            
            # Skip function edges (like internal connections)
            if edge_elem.get('function') == 'internal':
                continue
            
            # Get lanes for detailed info
            lanes = edge_elem.findall('lane')
            num_lanes = len(lanes) if lanes else 1
            
            # Get length and speed from first lane
            if lanes:
                length = float(lanes[0].get('length', 100))
                speed = float(lanes[0].get('speed', self.default_speed))
            else:
                length = float(edge_elem.get('length', 100))
                speed = float(edge_elem.get('speed', self.default_speed))
            
            self.edges[edge_id] = SUMOEdge(
                id=edge_id,
                from_node=edge_elem.get('from', ''),
                to_node=edge_elem.get('to', ''),
                length=length,
                speed=speed,
                lanes=num_lanes,
                priority=int(edge_elem.get('priority', 0)),
                type_id=edge_elem.get('type', '')
            )
    
    def _parse_traffic_lights(self, root: ET.Element):
        """Parse traffic light logic"""
        for tl in root.findall('.//tlLogic'):
            tl_id = tl.get('id', '')
            
            phases = []
            cycle_length = 0.0
            
            for phase in tl.findall('phase'):
                duration = float(phase.get('duration', 30))
                state = phase.get('state', '')
                
                phases.append({
                    'duration': duration,
                    'state': state,
                    'green_edges': self._extract_green_edges(tl_id, state)
                })
                
                cycle_length += duration
            
            # Find the junction for this signal
            node_id = tl_id  # Usually same as junction ID
            
            self.signals[tl_id] = SUMOSignal(
                id=tl_id,
                node_id=node_id,
                phases=phases,
                cycle_length=cycle_length
            )
    
    def _extract_green_edges(self, tl_id: str, state: str) -> List[str]:
        """
        Extract which edges have green in a given state
        
        State string uses characters: G/g (green), r (red), y (yellow)
        Position corresponds to connection index
        """
        green_edges = []
        for rec in self.connection_records:
            if rec.get("tl") != tl_id:
                continue
            link_index = rec.get("linkIndex", -1)
            if link_index < 0 or link_index >= len(state):
                continue
            if state[link_index] in ("g", "G"):
                green_edges.append(rec["from"])
        return green_edges
    
    def _parse_connections(self, root: ET.Element):
        """Parse connection elements to build topology"""
        for conn in root.findall('.//connection'):
            from_edge = conn.get('from', '')
            to_edge = conn.get('to', '')
            
            # Skip internal connections
            if self.skip_internal and (from_edge.startswith(':') or to_edge.startswith(':')):
                continue
            
            if from_edge not in self.connections:
                self.connections[from_edge] = []
            
            if to_edge not in self.connections[from_edge]:
                self.connections[from_edge].append(to_edge)
            self.connection_records.append({
                "id": len(self.connection_records),
                "from": from_edge,
                "to": to_edge,
                "fromLane": conn.get("fromLane", ""),
                "toLane": conn.get("toLane", ""),
                "dir": conn.get("dir", ""),
                "tl": conn.get("tl", ""),
                "linkIndex": int(conn.get("linkIndex", "-1")),
                "via": conn.get("via", ""),
            })
    
    def _to_network_data(self) -> NetworkData:
        """Convert parsed data to NetworkData format"""
        # Create ordered lists
        edge_list = list(self.edges.values())
        node_list = list(self.nodes.values())
        
        edge_ids = [e.id for e in edge_list]
        edge_id_map = {e.id: i for i, e in enumerate(edge_list)}
        
        node_ids = [n.id for n in node_list]
        node_id_map = {n.id: i for i, n in enumerate(node_list)}
        
        # Calculate capacities (with minimum to prevent bottlenecks on short edges)
        edge_capacities = [
            max(self.min_capacity, int(e.length * e.lanes * self.jam_density))
            for e in edge_list
        ]
        
        # Map edges to signals
        edge_signal_ids = []
        for edge in edge_list:
            signal_id = -1
            to_node = edge.to_node
            # Check if to_node has a signal
            if to_node in self.signals or to_node in [s.node_id for s in self.signals.values()]:
                for sig_idx, sig in enumerate(self.signals.values()):
                    if sig.node_id == to_node:
                        signal_id = sig_idx
                        break
            edge_signal_ids.append(signal_id)
        
        # Convert signals to list format
        signals_list = []
        for sig in self.signals.values():
            signals_list.append({
                'id': sig.id,
                'node_id': node_id_map.get(sig.node_id, -1),
                'phases': sig.phases,
                'cycle_length': sig.cycle_length
            })
        
        # Build from_nodes list
        edge_from_nodes = [node_id_map.get(e.from_node, -1) for e in edge_list]
        edge_to_nodes_list = [node_id_map.get(e.to_node, -1) for e in edge_list]
        
        # Build node adjacency for rerouting: node_id -> list of outgoing edge_ids
        node_adjacency = {}
        for edge_idx, edge in enumerate(edge_list):
            from_node_idx = node_id_map.get(edge.from_node, -1)
            if from_node_idx >= 0:
                if from_node_idx not in node_adjacency:
                    node_adjacency[from_node_idx] = []
                node_adjacency[from_node_idx].append(edge_idx)
        edge_successors = {}
        movement_map = {}
        movement_successors = {}
        movement_signal_map = {}
        movement_conflicts = {}
        movement_priority = {}
        connections = []
        for from_edge, to_edges in self.connections.items():
            from_idx = edge_id_map.get(from_edge, -1)
            if from_idx < 0:
                continue
            edge_successors[from_idx] = [edge_id_map[e] for e in to_edges if e in edge_id_map]
        for rec in self.connection_records:
            from_idx = edge_id_map.get(rec["from"], -1)
            to_idx = edge_id_map.get(rec["to"], -1)
            if from_idx >= 0 and to_idx >= 0:
                key = (from_idx, to_idx)
                if key not in movement_map:
                    movement_map[key] = []
                movement_map[key].append({
                    "id": rec.get("id", -1),
                    "tl": rec.get("tl", ""),
                    "linkIndex": rec.get("linkIndex", -1),
                    "dir": rec.get("dir", ""),
                    "fromLane": rec.get("fromLane", ""),
                    "toLane": rec.get("toLane", ""),
                    "via": rec.get("via", ""),
                })
                mov_id = rec.get("id", -1)
                connections.append({
                    "id": mov_id,
                    "from_edge": from_idx,
                    "to_edge": to_idx,
                    "from_lane": int(rec.get("fromLane", "0") or 0),
                    "to_lane": int(rec.get("toLane", "0") or 0),
                    "dir": rec.get("dir", ""),
                    "tl": rec.get("tl", ""),
                    "linkIndex": rec.get("linkIndex", -1),
                    "via": rec.get("via", ""),
                })
                if rec.get("tl", ""):
                    movement_signal_map[(rec.get("tl", ""), rec.get("linkIndex", -1))] = mov_id
                movement_priority[mov_id] = edge_list[from_idx].priority if from_idx < len(edge_list) else 0
                movement_conflicts[mov_id] = []
        # Build simple movement successor/conflict structures
        for c in connections:
            movement_successors[c["id"]] = []
        from_groups: Dict[int, List[int]] = {}
        for c in connections:
            from_groups.setdefault(c["from_edge"], []).append(c["id"])
        for c in connections:
            next_from = c["to_edge"]
            movement_successors[c["id"]] = from_groups.get(next_from, [])
            # Basic conflict model: same to_edge conflicts (merge), same from_edge alternative turns conflict.
            conflicts = set(from_groups.get(c["from_edge"], []))
            for other in connections:
                if other["id"] != c["id"] and other["to_edge"] == c["to_edge"]:
                    conflicts.add(other["id"])
            conflicts.discard(c["id"])
            movement_conflicts[c["id"]] = sorted(conflicts)
        
        # Generate segments if enabled
        segments: List[SegmentData] = []
        segment_id_map: Dict[str, int] = {}
        edge_to_segments: Dict[int, List[int]] = {}
        edge_first_segment: List[int] = []
        edge_last_segment: List[int] = []
        
        if self.use_segments:
            segment_idx = 0
            for edge_idx, edge in enumerate(edge_list):
                # Get edge properties
                edge_cap = edge_capacities[edge_idx]
                from_node = edge_from_nodes[edge_idx]
                to_node = edge_to_nodes_list[edge_idx]
                signal_id = edge_signal_ids[edge_idx]
                
                # Split this edge into segments
                edge_segments = split_edge_into_segments(
                    edge_idx=edge_idx,
                    edge_id=edge.id,
                    edge_length=edge.length,
                    edge_speed=edge.speed,
                    edge_lanes=edge.lanes,
                    edge_capacity=edge_cap,
                    edge_from_node=from_node,
                    edge_to_node=to_node,
                    edge_signal_id=signal_id,
                    segment_length=self.segment_length,
                    jam_density=self.jam_density,
                    min_capacity=max(3, self.min_capacity // 4),  # Smaller min for segments
                )
                
                # Record first and last segment indices for this edge
                first_seg_idx = segment_idx
                edge_first_segment.append(first_seg_idx)
                
                seg_indices_for_edge = []
                for seg in edge_segments:
                    segments.append(seg)
                    segment_id_map[seg.segment_id] = segment_idx
                    seg_indices_for_edge.append(segment_idx)
                    segment_idx += 1
                
                edge_last_segment.append(segment_idx - 1)
                edge_to_segments[edge_idx] = seg_indices_for_edge
                
                # Now set next_segment pointers within the edge
                for i, seg_global_idx in enumerate(seg_indices_for_edge):
                    if i < len(seg_indices_for_edge) - 1:
                        # Point to next segment in same edge
                        segments[seg_global_idx].next_segment = seg_indices_for_edge[i + 1]
                    else:
                        # Last segment in edge - next_segment stays -1
                        # Will need to look up next edge's first segment at runtime
                        segments[seg_global_idx].next_segment = -1
        
        return NetworkData(
            edge_ids=edge_ids,
            edge_id_map=edge_id_map,
            edge_lengths=[e.length for e in edge_list],
            edge_speeds=[e.speed for e in edge_list],
            edge_capacities=edge_capacities,
            edge_lanes=[e.lanes for e in edge_list],
            edge_to_nodes=edge_to_nodes_list,
            edge_priorities=[e.priority for e in edge_list],
            node_ids=node_ids,
            node_id_map=node_id_map,
            # Optional fields
            edge_from_nodes=edge_from_nodes,
            edge_signal_ids=edge_signal_ids,
            node_adjacency=node_adjacency,
            signals=signals_list,
            edge_successors=edge_successors,
            movement_map=movement_map,
            connections=connections,
            movement_successors=movement_successors,
            movement_signal_map=movement_signal_map,
            movement_conflicts=movement_conflicts,
            movement_priority=movement_priority,
            # Segment fields
            segments=segments,
            segment_id_map=segment_id_map,
            edge_to_segments=edge_to_segments,
            edge_first_segment=edge_first_segment,
            edge_last_segment=edge_last_segment,
        )


class SUMORouteParser:
    """
    Parser for SUMO route/demand files (.rou.xml, .trips.xml)
    
    Supports:
    - Individual vehicle definitions (<vehicle>)
    - Flow definitions (<flow>) - vehicle generators
    - Route definitions (<route>)
    - Trip definitions (<trip>) - origin-destination pairs
    
    Usage:
        parser = SUMORouteParser(edge_id_map)
        demand = parser.parse("routes.rou.xml")
        # or
        demand = parser.parse("trips.trips.xml")
    """
    
    def __init__(self,
                 edge_id_map: Optional[Dict[str, int]] = None,
                 edge_successors: Optional[Dict[int, List[int]]] = None,
                 grouping_window: float = 5.0,
                 max_packet_size: int = 50):
        """
        Args:
            edge_id_map: Mapping from edge IDs to indices
            grouping_window: Time window for grouping vehicles [s]
            max_packet_size: Maximum vehicles per packet
        """
        self.edge_id_map = edge_id_map or {}
        self.edge_successors = edge_successors or {}
        self.grouping_window = grouping_window
        self.max_packet_size = max_packet_size
        
        self.routes: Dict[str, List[str]] = {}  # route_id -> edge list
        self.vehicles: List[Dict[str, Any]] = []
        self.invalid_routes: List[Dict[str, Any]] = []
    
    def _expand_route_if_needed(self, from_edge: str, to_edge: str) -> List[str]:
        """Expand OD pair into edge path using BFS on edge_successors."""
        if from_edge == to_edge:
            return [from_edge]
        if not self.edge_id_map or not self.edge_successors:
            return [from_edge, to_edge]
        start = self.edge_id_map.get(from_edge, -1)
        goal = self.edge_id_map.get(to_edge, -1)
        if start < 0 or goal < 0:
            return [from_edge, to_edge]
        from collections import deque
        q = deque([start])
        parent = {start: -1}
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for nxt in self.edge_successors.get(cur, []):
                if nxt not in parent:
                    parent[nxt] = cur
                    q.append(nxt)
        if goal not in parent:
            return [from_edge, to_edge]
        idx_to_edge = {v: k for k, v in self.edge_id_map.items()}
        route_idxs = []
        cur = goal
        while cur != -1:
            route_idxs.append(cur)
            cur = parent[cur]
        route_idxs.reverse()
        return [idx_to_edge[i] for i in route_idxs if i in idx_to_edge]
    
    def parse(self, filepath: str) -> DemandData:
        """
        Parse a SUMO routes file
        
        Args:
            filepath: Path to .rou.xml file
            
        Returns:
            DemandData ready for simulation
        """
        # Handle encoding issues (SUMO files may have Latin-1 chars)
        try:
            tree = ET.parse(filepath)
        except ET.ParseError:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
            root = ET.fromstring(content)
            tree = ET.ElementTree(root)
        root = tree.getroot()
        
        # Parse route definitions
        self._parse_routes(root)
        
        # Parse vehicles
        self._parse_vehicles(root)
        
        # Parse flows
        self._parse_flows(root)
        
        # Parse trips (origin-destination pairs)
        self._parse_trips(root)
        
        # Group into packets
        departures = self._group_vehicles()
        
        return DemandData(departures=departures, invalid_routes=self.invalid_routes)
    
    def _parse_routes(self, root: ET.Element):
        """Parse route definitions"""
        for route in root.findall('.//route'):
            route_id = route.get('id', '')
            edges_str = route.get('edges', '')
            edges = edges_str.split()
            
            if route_id:
                self.routes[route_id] = edges
    
    def _parse_vehicles(self, root: ET.Element):
        """Parse individual vehicle definitions"""
        for veh in root.findall('.//vehicle'):
            veh_id = veh.get('id', '')
            depart = float(veh.get('depart', 0))
            
            # Get route
            route_ref = veh.get('route', '')
            route_elem = veh.find('route')
            
            if route_ref and route_ref in self.routes:
                edges = self.routes[route_ref]
            elif route_elem is not None:
                edges = route_elem.get('edges', '').split()
            else:
                continue
            
            self.vehicles.append({
                'id': veh_id,
                'depart': depart,
                'route': edges,
                'type': veh.get('type', 'default')
            })
    
    def _parse_trips(self, root: ET.Element):
        """Parse trip definitions (origin-destination pairs)
        
        Trips have 'from' and 'to' edges, optionally with 'via' intermediate edges.
        If no route calculator is available, stores just origin-destination as route.
        """
        for trip in root.findall('.//trip'):
            trip_id = trip.get('id', '')
            depart = float(trip.get('depart', 0))
            from_edge = trip.get('from', '')
            to_edge = trip.get('to', '')
            via = trip.get('via', '')  # Intermediate edges
            
            if not from_edge or not to_edge:
                continue
            
            # Build route from trip
            if via:
                # via is space-separated list of intermediate edges
                edges = [from_edge] + via.split() + [to_edge]
            else:
                edges = self._expand_route_if_needed(from_edge, to_edge)
            
            self.vehicles.append({
                'id': trip_id,
                'depart': depart,
                'route': edges,
                'type': trip.get('type', 'default'),
                'is_trip': True  # Flag that this might need route expansion
            })
    
    def _parse_flows(self, root: ET.Element):
        """Parse flow definitions (vehicle generators)"""
        for flow in root.findall('.//flow'):
            flow_id = flow.get('id', '')
            begin = float(flow.get('begin', 0))
            end = float(flow.get('end', 3600))
            
            # Get rate
            period = flow.get('period')
            veh_per_hour = flow.get('vehsPerHour')
            probability = flow.get('probability')
            number = flow.get('number')
            
            # Get route
            route_ref = flow.get('route', '')
            from_edge = flow.get('from', '')
            to_edge = flow.get('to', '')
            route_elem = flow.find('route')
            
            if route_ref and route_ref in self.routes:
                edges = self.routes[route_ref]
            elif route_elem is not None:
                edges = route_elem.get('edges', '').split()
            elif from_edge and to_edge:
                edges = self._expand_route_if_needed(from_edge, to_edge)
            else:
                continue
            
            # Generate vehicles from flow
            if period:
                interval = float(period)
                times = self._generate_times_from_period(begin, end, interval)
            elif veh_per_hour:
                rate = float(veh_per_hour)
                times = self._generate_times_from_rate(begin, end, rate)
            elif number:
                count = int(number)
                times = self._generate_times_uniform(begin, end, count)
            else:
                continue
            
            for i, t in enumerate(times):
                self.vehicles.append({
                    'id': f"{flow_id}_{i}",
                    'depart': t,
                    'route': edges,
                    'type': flow.get('type', 'default')
                })
    
    def _is_route_connected(self, route: List[str]) -> bool:
        if len(route) <= 1:
            return True
        if not self.edge_id_map or not self.edge_successors:
            return True
        for idx in range(len(route) - 1):
            a = self.edge_id_map.get(route[idx], -1)
            b = self.edge_id_map.get(route[idx + 1], -1)
            if a < 0 or b < 0:
                return False
            if b not in self.edge_successors.get(a, []):
                return False
        return True
    
    def _generate_times_from_period(self, begin: float, end: float, 
                                     period: float) -> List[float]:
        """Generate departure times with fixed period"""
        times = []
        t = begin
        while t < end:
            times.append(t)
            t += period
        return times
    
    def _generate_times_from_rate(self, begin: float, end: float,
                                   rate: float) -> List[float]:
        """Generate departure times from hourly rate"""
        if rate <= 0:
            return []
        period = 3600.0 / rate
        return self._generate_times_from_period(begin, end, period)
    
    def _generate_times_uniform(self, begin: float, end: float,
                                 count: int) -> List[float]:
        """Generate uniformly distributed departure times"""
        if count <= 0:
            return []
        interval = (end - begin) / count
        return [begin + i * interval for i in range(count)]
    
    def _group_vehicles(self) -> List[Tuple[float, str, List[str], int]]:
        """
        Group vehicles into packets based on departure time and route
        
        Strategy: Sort by (route, time) so vehicles with the same route
        are adjacent, then group by time window within each route.
        Finally, sort output by departure time.
        
        Returns:
            List of (time, origin, route, count) tuples
        """
        if not self.vehicles:
            return []
        
        # Sort by (route_tuple, departure_time) so same-route vehicles are adjacent
        sorted_vehicles = sorted(
            self.vehicles, 
            key=lambda v: (tuple(v['route']), v['depart'])
        )
        
        departures = []
        i = 0
        
        while i < len(sorted_vehicles):
            v = sorted_vehicles[i]
            if not self._is_route_connected(v["route"]):
                self.invalid_routes.append({"vehicle_id": v["id"], "route": list(v["route"])})
                i += 1
                continue
            t = v['depart']
            route = tuple(v['route'])
            origin = v['route'][0] if v['route'] else ''
            
            # Count vehicles in same window with same route
            count = 1
            j = i + 1
            
            while (j < len(sorted_vehicles) and 
                   tuple(sorted_vehicles[j]['route']) == route and
                   sorted_vehicles[j]['depart'] <= t + self.grouping_window and
                   count < self.max_packet_size):
                count += 1
                j += 1
            
            departures.append((t, origin, list(route), count))
            i = j
        
        # Sort final output by departure time for proper simulation order
        departures.sort(key=lambda x: x[0])
        
        return departures


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_sumo_network(filepath: str, **kwargs) -> NetworkData:
    """
    Parse a SUMO network file
    
    Args:
        filepath: Path to .net.xml file
        **kwargs: Additional arguments for parser
        
    Returns:
        NetworkData
    """
    parser = SUMONetworkParser(**kwargs)
    return parser.parse(filepath)


def parse_sumo_routes(filepath: str, 
                      edge_id_map: Optional[Dict[str, int]] = None,
                      edge_successors: Optional[Dict[int, List[int]]] = None,
                      **kwargs) -> DemandData:
    """
    Parse a SUMO routes or trips file
    
    Args:
        filepath: Path to .rou.xml or .trips.xml file
        edge_id_map: Optional mapping from edge IDs to indices
        **kwargs: Additional arguments for parser
        
    Returns:
        DemandData
        
    Note:
        Trips (.trips.xml) contain origin-destination pairs without full routes.
        If routes need to be computed, pass the network's edge_id_map.
    """
    parser = SUMORouteParser(edge_id_map=edge_id_map, edge_successors=edge_successors, **kwargs)
    return parser.parse(filepath)

