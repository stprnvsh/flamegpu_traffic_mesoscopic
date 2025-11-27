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
from core.simulation import NetworkData, DemandData


@dataclass
class SUMOMesoConfig:
    """SUMO Mesoscopic simulation parameters from .sumocfg"""
    # Mesoscopic TAU factors (travel time adjustment)
    tau_ff: float = 1.4      # Free-flow to free-flow
    tau_fj: float = 1.4      # Free-flow to jam
    tau_jf: float = 2.0      # Jam to free-flow  
    tau_jj: float = 1.4      # Jam to jam
    
    # Queue/segment parameters
    meso_edgelength: float = 98.0    # Segment length for queues [m]
    jam_threshold: float = -1.0       # Jam threshold (negative = auto)
    multi_queue: bool = True          # Use multi-queue model
    junction_control: bool = True     # Junction-based control
    lane_queue: bool = False          # Per-lane queuing
    overtaking: bool = False          # Allow overtaking
    
    # Penalties
    tls_penalty: float = 0.0          # Traffic light penalty [s]
    tls_flow_penalty: float = 0.0     # TLS flow-based penalty
    minor_penalty: float = 0.0        # Minor road penalty [s]
    
    # Rerouting parameters  
    rerouting_probability: float = 0.0   # Fraction of vehicles that reroute
    rerouting_period: float = 60.0       # Rerouting check interval [s]
    rerouting_adaptation_interval: float = 60.0  # Travel time adaptation interval
    rerouting_adaptation_steps: int = 1  # Number of adaptation steps
    rerouting_threshold_factor: float = 1.0  # Threshold to trigger reroute
    rerouting_threshold_constant: float = 0.0  # Constant threshold
    routing_algorithm: str = "dijkstra"  # Routing algorithm
    
    # Processing parameters
    time_to_teleport: float = -1.0    # Teleport stuck vehicles (-1 = disabled)
    time_to_impatience: float = 1e12  # Time until impatience kicks in
    teleport_disconnected: float = -1.0  # Teleport on disconnected
    
    # Time parameters
    step_length: float = 1.0          # Simulation step length [s]
    begin_time: float = 0.0           # Simulation start time [s]
    end_time: float = 86400.0         # Simulation end time [s]
    
    # Random seed
    seed: int = 42


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
        
        # Parse routing section
        routing = root.find("routing")
        if routing is not None:
            config.rerouting_probability = self._get_value(routing, "device.rerouting.probability", config.rerouting_probability)
            config.rerouting_period = self._get_value(routing, "device.rerouting.period", config.rerouting_period)
            config.rerouting_adaptation_interval = self._get_value(routing, "device.rerouting.adaptation-interval", config.rerouting_adaptation_interval)
            config.rerouting_adaptation_steps = int(self._get_value(routing, "device.rerouting.adaptation-steps", config.rerouting_adaptation_steps))
            config.rerouting_threshold_factor = self._get_value(routing, "device.rerouting.threshold.factor", config.rerouting_threshold_factor)
            config.rerouting_threshold_constant = self._get_value(routing, "device.rerouting.threshold.constant", config.rerouting_threshold_constant)
            config.routing_algorithm = self._get_str(routing, "routing-algorithm", config.routing_algorithm)
        
        # Parse processing section
        processing = root.find("processing")
        if processing is not None:
            config.time_to_teleport = self._get_value(processing, "time-to-teleport", config.time_to_teleport)
            config.time_to_impatience = self._get_value(processing, "time-to-impatience", config.time_to_impatience)
            config.teleport_disconnected = self._get_value(processing, "time-to-teleport.disconnected", config.teleport_disconnected)
        
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
    
    Usage:
        parser = SUMONetworkParser()
        network = parser.parse("network.net.xml")
    """
    
    def __init__(self, 
                 jam_density: float = 0.15,  # Realistic jam density
                 min_capacity: int = 20,     # Minimum capacity per edge to prevent bottlenecks
                 default_speed: float = 13.89,
                 skip_internal: bool = True):
        """
        Args:
            jam_density: Jam density for capacity calculation [veh/m/lane]
            min_capacity: Minimum capacity per edge (prevents tiny edges from blocking)
            default_speed: Default speed if not specified [m/s]
            skip_internal: Skip internal/connector edges
        """
        self.jam_density = jam_density
        self.min_capacity = min_capacity
        self.default_speed = default_speed
        self.skip_internal = skip_internal
        
        self.edges: Dict[str, SUMOEdge] = {}
        self.nodes: Dict[str, SUMONode] = {}
        self.signals: Dict[str, SUMOSignal] = {}
        self.connections: Dict[str, List[str]] = {}  # from_edge -> [to_edges]
    
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
        
        # Parse traffic lights
        self._parse_traffic_lights(root)
        
        # Parse connections
        self._parse_connections(root)
        
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
        # This is simplified - full implementation would need connection mapping
        green_edges = []
        # Would need to cross-reference with connection data
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
        
        return NetworkData(
            edge_ids=edge_ids,
            edge_id_map=edge_id_map,
            edge_lengths=[e.length for e in edge_list],
            edge_speeds=[e.speed for e in edge_list],
            edge_capacities=edge_capacities,
            edge_lanes=[e.lanes for e in edge_list],
            edge_to_nodes=edge_to_nodes_list,
            node_ids=node_ids,
            node_id_map=node_id_map,
            # Optional fields
            edge_from_nodes=edge_from_nodes,
            edge_signal_ids=edge_signal_ids,
            node_adjacency=node_adjacency,
            signals=signals_list,
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
                 grouping_window: float = 5.0,
                 max_packet_size: int = 50):
        """
        Args:
            edge_id_map: Mapping from edge IDs to indices
            grouping_window: Time window for grouping vehicles [s]
            max_packet_size: Maximum vehicles per packet
        """
        self.edge_id_map = edge_id_map or {}
        self.grouping_window = grouping_window
        self.max_packet_size = max_packet_size
        
        self.routes: Dict[str, List[str]] = {}  # route_id -> edge list
        self.vehicles: List[Dict[str, Any]] = []
    
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
        
        return DemandData(departures=departures)
    
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
                # Just origin and destination - for mesoscopic we need actual route
                # If we have network with adjacency, we could compute route here
                # For now, store as simple 2-edge "route" (will need routing)
                edges = [from_edge, to_edge]
            
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
                edges = [from_edge, to_edge]  # Simplified
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
    parser = SUMORouteParser(edge_id_map=edge_id_map, **kwargs)
    return parser.parse(filepath)

