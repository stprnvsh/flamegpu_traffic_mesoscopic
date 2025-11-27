# Simulation Loop and Agent Interaction

We use a discrete time-step simulation, where each time step (e.g. 1 second) all agents update in parallel according to their roles. FLAMEGPU2 allows us to control execution order through layers.

## Layer Ordering

A logical ordering of events in each step:

1. **Packet Movement & Request Stage** (Traveling state)
2. **Traffic Signal Update Stage**
3. **Edge Queue Processing Stage**
4. **Packet Transition/Advancement Stage** (Waiting state)

## Stage 1: Packet Movement & Request (Traveling state)

All Packet agents in TRAVELING state update their position on the current edge. Each packet will decrement its `remaining_time` by `dt` (the time step).

- If after decrementing, `remaining_time > 0`, the packet is still in the middle of the edge; it simply stays in TRAVELING state and continues next step
- If `remaining_time <= 0`, the packet has reached the end of its current edge this step:
  - Send a departure message to the current edge to decrement its `curr_count`
  - If there is a next edge: the packet sends a request message to move into that edge's queue, then transitions to WAITING state
  - If there is no next edge (the packet's route ended): the packet is removed from the simulation (marked as DEAD)

### Example Agent Function

```python
@pyflamegpu.agent_function
def move_and_request(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageBucket, depart_out: pyflamegpu.MessageBucket):
    # Decrement remaining travel time
    dt = pyflamegpu.environment.getPropertyFloat("time_step")
    rem = pyflamegpu.getVariableFloat("remaining_time")
    rem -= dt
    pyflamegpu.setVariableFloat("remaining_time", rem)
    
    if rem > 0:
        return pyflamegpu.ALIVE  # stay in Traveling state
    
    # If remaining_time <= 0: reached end of edge
    curr_edge = pyflamegpu.getVariableInt("curr_edge")
    packet_size = pyflamegpu.getVariableInt("size")
    
    # Send departure notice to current edge
    depart_out.setKey(curr_edge)
    depart_out.setVariableInt("size", packet_size)
    
    next_edge = pyflamegpu.getVariableInt("next_edge")
    if next_edge == -1:
        return pyflamegpu.DEAD  # Destination reached
    
    # Send request to enter next_edge
    message_out.setKey(next_edge)
    message_out.setVariableInt("size", packet_size)
    message_out.setVariableUInt("agent_id", pyflamegpu.getID())
    
    return pyflamegpu.ALIVE  # Transition to WAITING state
```

Note: The function signature shows two message outputs. In FLAMEGPU2, you may need to handle departures in a separate function or use a different approach. See implementation details section.

## Stage 2: Traffic Signal Update

Each SignalController agent updates the signal phase:

- Decrement `time_to_phase_end` by `dt`
- If `time_to_phase_end` reaches 0, advance `phase_index` to the next phase in the cycle and reset `time_to_phase_end` to that phase's duration
- Determine which edges have green in the new phase

### Example Agent Function

```python
@pyflamegpu.agent_function
def update_signal(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    time_left = pyflamegpu.getVariableFloat("time_to_phase_end")
    dt = pyflamegpu.environment.getPropertyFloat("time_step")
    time_left -= dt
    
    if time_left <= 0:
        # Advance phase
        phase_index = pyflamegpu.getVariableInt("phase_index")
        phase_index = (phase_index + 1) % pyflamegpu.getVariableInt("phase_count")
        pyflamegpu.setVariableInt("phase_index", phase_index)
        
        # Reset timer
        duration = pyflamegpu.getVariableArrayFloat("phase_durations", phase_index)
        time_left = duration
    
    pyflamegpu.setVariableFloat("time_to_phase_end", time_left)
    return pyflamegpu.ALIVE
```

## Stage 3: Edge Queue Processing

Each EdgeQueue agent processes incoming requests from packets and determines which packets can enter:

1. Process departure messages first (if any) to decrement `curr_count`
2. Sum incoming vehicle requests
3. Check the signal state if this edge is controlled by a light
4. Check the space/capacity on this edge:
   - If `available_space >= total_request`: accept all packets
   - If `available_space < total_request`: partial acceptance (FIFO by arrival)
5. Update `curr_count` with allowed vehicles
6. Send acknowledgment messages for accepted packets

### Example Agent Function

```python
@pyflamegpu.agent_function
def process_edge_requests(depart_in: pyflamegpu.MessageBucket, request_in: pyflamegpu.MessageBucket, response_out: pyflamegpu.MessageBruteForce):
    edge_id = pyflamegpu.getVariableInt("edge_id")
    
    # First, process departures to decrement curr_count
    curr = pyflamegpu.getVariableInt("curr_count")
    for msg in depart_in:
        depart_size = msg.getVariableInt("size")
        curr = max(0, curr - depart_size)  # Ensure non-negative
    pyflamegpu.setVariableInt("curr_count", curr)
    
    # Sum incoming vehicle requests
    total_in = 0
    requests = []
    for msg in request_in:
        total_in += msg.getVariableInt("size")
        requests.append((msg.getVariableUInt("agent_id"), msg.getVariableInt("size")))
    
    if total_in == 0:
        return pyflamegpu.ALIVE
    
    # Check signal
    if pyflamegpu.getVariableInt("signal_id") != -1:
        green = pyflamegpu.getVariableInt("is_green")
        if green == 0:
            return pyflamegpu.ALIVE  # Red light, do not accept
    
    # Capacity check (curr already updated from departures)
    cap = pyflamegpu.getVariableInt("capacity")
    available = cap - curr
    
    accepted_ids = []
    if available > 0:
        running = 0
        for (req_id, size) in requests:
            if running + size <= available:
                running += size
                accepted_ids.append(req_id)
            else:
                break
    
    # Update edge occupancy
    pyflamegpu.setVariableInt("curr_count", curr + running)
    
    # Send response for each accepted packet
    for aid in accepted_ids:
        response = response_out.append()
        response.setVariableUInt("agent_id", aid)
        response.setVariableInt("edge_id", edge_id)
    
    return pyflamegpu.ALIVE
```

## Stage 4: Packet Transition/Advancement (Waiting state)

Packet agents currently in WAITING state respond to the outcomes of the edge processing:

- If accepted: transition to TRAVELING state, set `curr_edge = next_edge`, determine new `next_edge`, reset `remaining_time`
- If not accepted: remain in WAITING state, resend request message

### Example Agent Function

```python
@pyflamegpu.agent_function
def wait_for_entry(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageBucket):
    accepted = False
    my_id = pyflamegpu.getID()
    
    for msg in pyflamegpu.message_in:
        if msg.getVariableUInt("agent_id") == my_id:
            accepted = True
            break
    
    if not accepted:
        # Still waiting, resend request
        next_edge = pyflamegpu.getVariableInt("next_edge")
        pyflamegpu.message_out.setKey(next_edge)
        pyflamegpu.message_out.setVariableInt("size", pyflamegpu.getVariableInt("size"))
        pyflamegpu.message_out.setVariableUInt("agent_id", my_id)
        return pyflamegpu.ALIVE
    
    # If accepted: transition to traveling on next edge
    curr_edge = pyflamegpu.getVariableInt("next_edge")
    pyflamegpu.setVariableInt("curr_edge", curr_edge)
    
    # Determine following edge
    route_index = pyflamegpu.getVariableInt("route_idx")
    route_length = pyflamegpu.getVariableInt("route_length")
    route = pyflamegpu.getVariableArrayInt("route")
    
    # Advance route index
    pyflamegpu.setVariableInt("route_idx", route_index + 1)
    
    # Get next edge from route array
    if route_index + 1 < route_length:
        next_edge2 = route[route_index + 1]
        pyflamegpu.setVariableInt("next_edge", next_edge2)
    else:
        pyflamegpu.setVariableInt("next_edge", -1)  # Route complete
    
    # Compute initial remaining_time for curr_edge
    # Get edge properties from environment arrays
    edge_lengths = pyflamegpu.environment.getPropertyArrayFloat("edge_lengths")
    edge_speeds = pyflamegpu.environment.getPropertyArrayFloat("edge_speeds")
    length = edge_lengths[curr_edge]
    free_speed = edge_speeds[curr_edge]
    travel_t = length / free_speed
    pyflamegpu.setVariableFloat("remaining_time", travel_t)
    pyflamegpu.setVariableFloat("entry_time", pyflamegpu.environment.getPropertyFloat("current_time"))
    
    return pyflamegpu.ALIVE
```

## Message Usage Summary

- **entry_request** – a Bucket message keyed by `edge_id`, carrying `size` and `agent_id`. Emitted by Packet when finishing an edge (and by waiting ones repeatedly) to request entry to next edge. Received by EdgeQueue agents.

- **acceptance** (or entry_grant) – a BruteForce message carrying `agent_id` (and possibly `edge_id`) for packets that are allowed to enter. Emitted by EdgeQueue after processing requests. Received by Packet (waiting) agents.

- **departure_notice** – a Bucket message keyed by `edge_id`, carrying `size`. Emitted by Packet when finishing an edge to inform the old EdgeQueue that vehicles have left. Received by EdgeQueue agents to decrement `curr_count`.

## Conflict Resolution

Our design inherently handles conflicts via the EdgeQueue logic and signals:

- If two packets from different edges want the same next edge, the EdgeQueue of that next edge will accept one or both based on available space
- If two packets want to cross an intersection (different destinations), if there's a signal, one will be red and wait

