# Data Input Pipeline (SUMO Network and Routes)

To initialize the simulation, we need to import the road network and traffic demand from SUMO files.

## Parsing the Network (.net.xml)

SUMO's network XML contains nodes (junctions), edges (road segments connecting nodes), lanes (details per edge), and traffic lights. We will use it to create our EdgeQueue and SignalController agents, and to set up environment structures like adjacency lists.

### Creating EdgeQueue Agents

Each `<edge>` in the net file has attributes like `id`, `from`, `to`, `length`, `speed`, and possibly `numLanes`. We create one EdgeQueue agent per edge:

- **edge_id**: Map the SUMO edge IDs (strings) to an integer index using a dictionary mapping `edgeID -> index`
- **capacity**: If `numLanes` is given and if we assume jam density ~ 0.15 vehicles/m (i.e. 150 veh/km per lane), we can compute `capacity = jam_density * length * lanes`
- **length** and **free_speed**: directly from the file (SUMO's speed is often m/s; length in m)
- **curr_count**: initial vehicles on the edge. Typically at time 0 we assume empty network (`curr_count=0`) unless we have initial traffic
- **out_node**: the `to` node of the edge
- **signal_id**: Determine if the `to` node has a traffic light. In the net XML, junctions with `type="traffic_light"` usually indicate a signal

### Example XML Parsing Code

```python
import xml.etree.ElementTree as ET

def parse_network(net_file):
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    edge_id_map = {}  # SUMO edge ID -> integer index
    edges_data = []
    edge_lengths = []
    edge_speeds = []
    
    # Parse edges
    for idx, edge_elem in enumerate(root.findall('edge')):
        if 'function' in edge_elem.attrib:  # Skip internal edges
            continue
            
        edge_id_str = edge_elem.get('id')
        edge_id_map[edge_id_str] = idx
        
        length = float(edge_elem.get('length'))
        speed = float(edge_elem.get('speed'))
        lanes = int(edge_elem.get('numLanes', 1))
        to_node = edge_elem.get('to')
        
        # Calculate capacity
        jam_density = 0.15  # veh/m per lane
        capacity = int(jam_density * length * lanes)
        
        edges_data.append({
            'edge_id': idx,
            'capacity': capacity,
            'length': length,
            'free_speed': speed,
            'to_node': to_node,
            'signal_id': -1  # Will be updated if node has signal
        })
        
        edge_lengths.append(length)
        edge_speeds.append(speed)
    
    return edge_id_map, edges_data, edge_lengths, edge_speeds
```

### Creating SignalController Agents

For each `<junction>` with `type="traffic_light"`:

- **node_id**: from file (string or number). Map to an index if needed
- Determine the controlled edges (incoming to this junction)
- Parse `<tlLogic>` for this signal: extract the cycle. For each `<phase duration="..." state="...">`, record the duration and which edges are green in that phase
- Populate the SignalController's `cycle_plan`: For example, `cycle_plan = [(dur1, [edgeA, edgeB]), (dur2, [edgeC])]` where `[edgeA, edgeB]` are green in phase1, etc.
- Set `phase_index = 0` and `time_to_phase_end = dur1` initially

### Building CSR/Topology

We can optionally build a CSR representation of the network graph. In practice, since vehicles follow predetermined routes, we might not need to compute shortest paths on the fly. But for potential routing or aggregate analysis, having adjacency can help.

## Parsing Demand (.rou.xml)

The routes file describes individual vehicles or flows (vehicle generators).

### Vehicle Grouping

The instructions say to model "vehicle groups as variable-sized packets based on demand." This implies if multiple vehicles share the same route and depart around the same time, we could bundle them into one Packet to reduce simulation entities.

**If the .rou.xml has explicit `<vehicle>` entries**: We can post-process: look at sequential vehicles in the list – if two have the same origin edge and depart times that differ by less than some threshold (say 5 seconds) and maybe same route, combine them into one Packet with size equal to count of those vehicles.

**If the .rou.xml defines flows via `<flow>` tags**: We can directly treat each flow as generating packets. For example, if depart rate is 360 veh/hour (one every 10s), instead of 1 each 10s, we might spawn one packet every 50s of 5 vehicles.

### Route Assignment

Each Packet needs its route (sequence of edges). The .rou.xml gives edges list. We can store this in the Packet agent as an array variable of fixed max length (say max 20-32 edges) and an int `route_length`. Initialize it with the edge sequence and length. Also store `route_index` indicating which index the Packet is currently on.

### Spawning Packets

We choose to spawn during simulation via a host function for clarity. We can store the departure info in a Python list or environment structure. After parsing .rou.xml, we have a list of `(depart_time, origin_edge, route)` for each vehicle (or packet). We sort by `depart_time`. Then we define a host function `spawn_packets()` that runs each simulation step and spawns all vehicles whose `depart_time` equals the current step time.

### Example Grouping Code

```python
departures = []  # list of (time, origin_edge, route_edge_list, count)
for veh in vehicles:
    time = float(veh.get('depart'))
    route_edges = veh.find('route').get('edges').split()
    origin = edge_id_map[route_edges[0]]
    departures.append((time, origin, route_edges))

# sort by time
departures.sort(key=lambda x: x[0])

# grouping
grouped_deps = []
i = 0
while i < len(departures):
    t, origin, route = departures[i]
    count = 1
    j = i + 1
    # group subsequent vehicles within 1s window on same origin & route
    while j < len(departures) and departures[j][0] <= t + 1 and departures[j][1] == origin and departures[j][2] == route:
        count += 1
        j += 1
    grouped_deps.append((t, origin, route, count))
    i = j
```

### Host Function for Spawning

```python
depart_index = 0  # index in grouped_deps list

def spawn_packets_host():
    nonlocal depart_index
    current_time = simulation.getStepCount() * dt
    
    while depart_index < len(grouped_deps) and abs(grouped_deps[depart_index][0] - current_time) < 1e-9:
        t, origin, route, count = grouped_deps[depart_index]
        
        # create a new Packet agent
        new_agent = simulation.newAgent("Packet")
        new_agent.setVariableInt("size", count)
        new_agent.setVariableInt("curr_edge", origin)
        
        # Prepare route array (pad with -1 if route is shorter than max)
        edge_indices = [edge_id_map[e] for e in route]
        route_array = edge_indices + [-1] * (32 - len(edge_indices))
        new_agent.setVariableArrayInt("route", route_array)
        new_agent.setVariableInt("route_length", len(edge_indices))
        new_agent.setVariableInt("route_idx", 0)
        new_agent.setVariableInt("next_edge", edge_indices[1] if len(edge_indices)>1 else -1)
        
        # Set travel time for first edge
        length = edge_lengths[origin]
        speed = edge_speeds[origin]
        new_agent.setVariableFloat("remaining_time", length/speed)
        new_agent.setVariableFloat("entry_time", current_time)
        new_agent.setState("traveling")
        
        # IMPORTANT: Increment origin edge's curr_count since vehicles are now on it
        # This needs to be done by accessing the EdgeQueue agent for origin edge
        # In practice, you might need to maintain a reference or use environment macro
        
        depart_index += 1
```

**Note on Initial Edge Count**: When spawning packets, the origin edge's `curr_count` should be incremented. This can be done by:
1. Maintaining a Python-side mapping of edge_id -> EdgeQueue agent reference
2. Using an environment macro property to track initial counts
3. Having a separate initialization step that sets initial counts based on spawn schedule

## Agent Initialization Summary

- **EdgeQueue agents**: Created for each road in network, with static properties (capacity, length, etc.)
- **SignalController agents**: Created for each traffic light node, with cycle definitions
- **Packet agents**: Not all created at t=0, but spawned as simulation runs, based on the demand schedule parsed from routes

We should also initialize environment properties like `current_time = 0` and `dt` (time step). If using macro properties for logging (like arrays for edge flow counts), initialize those as well (size = num_edges, initial zeros).

