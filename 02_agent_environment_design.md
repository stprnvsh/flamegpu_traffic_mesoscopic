# Agent and Environment Design

## Agents

We use three primary agent types to model the system: EdgeQueue, Packet, and SignalController (for junctions with lights). Each agent type encapsulates state variables relevant to its role.

### EdgeQueue Agent

Represents a road segment (edge in the network) with a traffic queue. Each EdgeQueue agent has:

- `edge_id` (int): Unique identifier (index) for the edge (e.g. mapping from SUMO edge IDs)
- `capacity` (int): Maximum vehicles the edge can hold (e.g. based on jam density × length × lanes)
- `length` (float): Length of the road segment (meters or km)
- `free_speed` (float): Free-flow speed on this edge (m/s or km/h)
- `curr_count` (int): Current number of vehicles (or equivalent) on the edge (updated as packets enter/leave)
- `travel_time` (float): Current travel time for the edge (seconds), computed from the fundamental diagram given current occupancy
- `out_node` (int): Identifier of the junction at the downstream end of this edge (useful to link to signals or next edges)
- Optionally, `lane_count` (int) if multi-lane detail is needed (for capacity calculations), or separate queues for different turning lanes

### Packet Agent

Represents a group of one or more vehicles traveling together on a route. A Packet's state includes:

- `packet_id` (int): Unique agent ID (assigned automatically by FLAMEGPU2)
- `size` (int): Number of vehicles in this packet (the group size)
- `curr_edge` (int): The edge ID on which the packet is currently traveling
- `next_edge` (int): The next edge ID in the packet's route. If `next_edge = -1`, it indicates no next edge (the destination is reached)
- `remaining_time` (float): Remaining travel time on the current edge (seconds). This counts down each time step as the packet moves. When it reaches 0, the packet is at the end of the edge
- `entry_time` (float): The simulation time when the packet entered the current edge (used to calculate actual travel time upon exit)
- `state` (enumerated): Two major states – TRAVELING and WAITING. In TRAVELING state, the packet is en route on an edge; in WAITING state, it has finished an edge and is queued at a junction waiting to enter the next edge
- (Optional) `route_id` or route list: If routes are pre-determined, we may store a reference to the route
- (Optional) `dest_node` (int): Destination node (junction or sink) for the packet

### SignalController Agent

Represents a traffic light controller at a junction (intersection). We create a SignalController for each signalized intersection in the network. Its variables include:

- `node_id` (int): Junction identifier (to associate with incoming/outgoing edges)
- `phase_index` (int): Current phase of the signal (index into the cycle program)
- `time_to_phase_end` (float): Time remaining until the current phase ends (seconds)
- `cycle_plan` (array or data structure): The definition of the signal phases and timing. For example, a list of phases, where each phase has a duration and a set of approaches that have green
- `is_signalized` (bool): A flag to distinguish between signalized junctions and unsignalized ones
- (Optional) Conflict matrix or priority info: For unsignalized junctions, one might include data about which incoming edges conflict, and priority rules

## Environment (Global State)

Besides agents, we define global properties to represent static or aggregated data accessible to agents:

### Network Topology

We represent the road network graph for fast lookup. For example, we can use adjacency lists or CSR (Compressed Sparse Row) structure to map from a junction to its outgoing edges. We can store arrays like `edge_outgoing[edge_id] = [list of next_edge_ids]` or `node_outgoing[node_id] = [edges]` as environment constants.

### Edge Properties Arrays

Instead of each EdgeQueue agent storing all static info (speed limit, length, etc.), we can store some of it in environment arrays indexed by `edge_id`. However, since we've already included those as agent variables, this duplication isn't strictly needed.

### Simulation Parameters

Such as time step size `dt` (simulation delta time in seconds), and perhaps constants for the fundamental diagram model (e.g. parameters for speed-density equation, jam density, etc.). These can be set as environment properties that agents can read.

### Global Counters or Logging Aids

We may use environment macro properties (arrays that agents can update in limited ways) to accumulate outputs like total vehicles that exited each edge in an interval, sum of travel times, etc., for computing metrics.

## Agent State & Function Structure

In FLAMEGPU2, each agent type can have one or more agent functions defining its behavior, and we organize execution in layers or a dependency graph to control order. We will use agent states for Packets to handle the travel vs. waiting behaviors distinctly. For example, we define states "traveling" and "waiting" for the Packet agent, and ensure that certain functions execute only for agents in the relevant state (by setting function initial/end states). The simulation loop will then be implemented by scheduling these functions in an order (e.g., move packets, then process junctions, etc.).

