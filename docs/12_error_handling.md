# Error Handling and Edge Cases

This document covers error handling strategies and edge cases that should be considered when implementing the mesoscopic traffic simulation.

## Edge Count Consistency

### Problem
Edge `curr_count` must remain consistent: it should equal the sum of all packets currently on that edge. Errors can occur if:
- Departure messages are lost or not processed
- Packets are spawned but origin edge count isn't incremented
- Multiple packets leave simultaneously and counts conflict

### Solutions

1. **Validation Function**: Add a host function that periodically validates edge counts:
```python
@pyflamegpu.host_function
def validate_edge_counts(sim):
    # Iterate all Packet agents and count vehicles per edge
    packet_counts = {}
    for packet in sim.getAgent("Packet"):
        edge = packet.getVariableInt("curr_edge")
        size = packet.getVariableInt("size")
        packet_counts[edge] = packet_counts.get(edge, 0) + size
    
    # Compare with EdgeQueue curr_count
    for edge in sim.getAgent("EdgeQueue"):
        edge_id = edge.getVariableInt("edge_id")
        expected = packet_counts.get(edge_id, 0)
        actual = edge.getVariableInt("curr_count")
        if abs(expected - actual) > 0:
            print(f"WARNING: Edge {edge_id} count mismatch: expected {expected}, got {actual}")
```

2. **Atomic Operations**: Ensure departure/arrival updates are atomic if possible

3. **Initial Count Tracking**: When spawning packets, maintain a separate counter for initial edge occupancy

## Route Array Bounds

### Problem
Packets may have routes longer than the allocated array size (e.g., 32 edges). This causes:
- Route truncation
- Incorrect next_edge lookups
- Packets getting stuck

### Solutions

1. **Route Length Validation**: Check during parsing:
```python
MAX_ROUTE_LENGTH = 32
if len(route_edges) > MAX_ROUTE_LENGTH:
    print(f"WARNING: Route too long ({len(route_edges)} edges), truncating")
    route_edges = route_edges[:MAX_ROUTE_LENGTH]
```

2. **Dynamic Route Storage**: Use environment arrays or route_id lookup instead of per-agent arrays

3. **Route Compression**: Store only key waypoints, compute intermediate edges on-the-fly

## Signal Phase Edge Mapping

### Problem
Mapping signal phases to edges can be complex:
- Multiple edges may share the same phase index
- Edge-to-phase mapping may be ambiguous
- Phase changes may not align with simulation steps

### Solutions

1. **Explicit Edge Lists**: Store list of edge IDs for each phase in SignalController:
```python
signal_agent.newVariableArrayInt("phase_0_green_edges", max_edges_per_phase)
signal_agent.newVariableArrayInt("phase_1_green_edges", max_edges_per_phase)
# ... etc
```

2. **Lookup Table**: Use environment array mapping `(signal_id, phase_index) -> [edge_ids]`

3. **Phase Duration Validation**: Ensure phase durations are multiples of time step (or handle fractional steps)

## Capacity Overflow

### Problem
If packets are too large relative to edge capacity, they may never be accepted:
- Large packet (e.g., 100 vehicles) on edge with capacity 50
- Packet waits indefinitely
- Simulation deadlocks

### Solutions

1. **Packet Size Validation**: During grouping, ensure packet size doesn't exceed typical edge capacity:
```python
MAX_PACKET_SIZE = 50  # Reasonable upper bound
if packet_size > MAX_PACKET_SIZE:
    # Split into multiple packets
    num_packets = (packet_size + MAX_PACKET_SIZE - 1) // MAX_PACKET_SIZE
    for i in range(num_packets):
        size = min(MAX_PACKET_SIZE, packet_size - i * MAX_PACKET_SIZE)
        # Create packet with size
```

2. **Partial Entry**: Allow packets to partially enter edges (split packets)

3. **Impatience Timer**: Packets waiting too long force entry or take alternative route

## Negative Remaining Time

### Problem
If `remaining_time` becomes negative due to large time steps or numerical errors:
- Packets may overshoot edges
- Travel time calculations become invalid

### Solutions

1. **Clamp Values**: Ensure `remaining_time >= 0`:
```python
rem = max(0.0, rem - dt)
```

2. **Adaptive Time Steps**: Reduce time step if many packets finish simultaneously

3. **Validation**: Check for negative values and correct:
```python
if rem < 0:
    rem = 0  # Clamp to zero
    # Optionally log warning
```

## Missing Edge References

### Problem
If `next_edge` references an edge that doesn't exist:
- Packets get stuck in WAITING state
- Simulation may crash on array access

### Solutions

1. **Route Validation**: Validate all edges in route exist during parsing:
```python
for edge_id_str in route_edges:
    if edge_id_str not in edge_id_map:
        raise ValueError(f"Route contains unknown edge: {edge_id_str}")
```

2. **Bounds Checking**: In packet functions, check edge ID is valid:
```python
if next_edge < 0 or next_edge >= num_edges:
    # Route complete or invalid
    return pyflamegpu.DEAD
```

3. **Default Behavior**: If edge doesn't exist, mark packet as DEAD or reroute

## Simultaneous Arrivals

### Problem
Multiple packets may arrive at the same edge simultaneously, causing:
- Race conditions in capacity checks
- Unfair ordering (depends on agent ID or processing order)

### Solutions

1. **FIFO Queue**: Maintain arrival timestamps in requests:
```python
msg_req.newVariableFloat("arrival_time")  # When packet reached edge end
```

2. **Sort by Timestamp**: In edge processing, sort requests by arrival_time before accepting

3. **Round-Robin**: Alternate between incoming edges if multiple feeds

## Empty Network

### Problem
If no vehicles are spawned or all complete early:
- Simulation continues unnecessarily
- GPU resources wasted

### Solutions

1. **Early Termination**: Check if any Packet agents exist:
```python
@pyflamegpu.host_function
def check_termination(sim):
    packet_count = sim.getAgent("Packet").count()
    if packet_count == 0:
        sim.exit()  # Terminate simulation
```

2. **Spawn Tracking**: Track remaining spawns and terminate when done and network empty

## Memory Limits

### Problem
Very large networks or many packets may exceed GPU memory:
- Agent creation fails
- Simulation crashes

### Solutions

1. **Memory Estimation**: Pre-calculate memory requirements:
```python
edge_memory = num_edges * edge_vars_size
packet_memory = max_packets * packet_vars_size
total = edge_memory + packet_memory
if total > gpu_memory_limit:
    raise MemoryError(f"Estimated memory {total} exceeds limit")
```

2. **Packet Aggregation**: Increase grouping to reduce packet count

3. **Batch Processing**: Process network in regions if possible

## Numerical Precision

### Problem
Floating-point errors in time calculations:
- `remaining_time` may not reach exactly 0
- Travel times may accumulate errors

### Solutions

1. **Tolerance Checks**: Use small epsilon:
```python
if rem <= 1e-6:  # Effectively zero
    rem = 0
```

2. **Use Fixed-Point**: Consider integer time steps (microseconds) if precision critical

3. **Periodic Correction**: Periodically recalculate travel times from scratch

## Summary

Key validation points:
- Edge count consistency
- Route array bounds
- Signal phase mapping
- Capacity constraints
- Edge reference validity
- Memory limits
- Numerical precision

Implement validation functions and error checks throughout the simulation to catch issues early.

