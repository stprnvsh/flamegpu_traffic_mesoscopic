# FLAMEGPU2 Python Implementation Details

With the model design in place, we now map it fully onto FLAMEGPU2's Python API.

## Model Definition

Using `pyflamegpu.ModelDescription` to create the model and then adding components:

```python
import pyflamegpu

model = pyflamegpu.ModelDescription("CityTrafficModel")

# Define Message types
# 1. Entry request message (Bucket by edge_id)
msg_req = model.newMessageBucket("entry_request")
num_edges = total_number_of_edges  # obtained from parsing network
msg_req.setUpperBound(num_edges - 1)  # keys 0..num_edges-1
msg_req.newVariableInt("size")
msg_req.newVariableUInt("agent_id")

# 2. Acceptance message (BruteForce for simplicity)
msg_acc = model.newMessageBruteForce("entry_accept")
msg_acc.newVariableUInt("agent_id")
msg_acc.newVariableInt("edge_id")

# 3. Departure notice message (Bucket by edge_id)
msg_depart = model.newMessageBucket("departure_notice")
msg_depart.setUpperBound(num_edges - 1)
msg_depart.newVariableInt("size")
```

## Agent Definitions

### EdgeQueue Agent

```python
edge_agent = model.newAgent("EdgeQueue")
edge_agent.newVariableInt("edge_id")
edge_agent.newVariableInt("capacity")
edge_agent.newVariableInt("curr_count")
edge_agent.newVariableFloat("length")
edge_agent.newVariableFloat("free_speed")
edge_agent.newVariableInt("signal_id")   # -1 if no signal
edge_agent.newVariableInt("is_green")    # 1 or 0
edge_agent.newVariableFloat("travel_time")

# Agent function for processing requests:
edge_fn = edge_agent.newRTCFunction("process_edge_requests", process_edge_requests)
edge_fn.setMessageInput("departure_notice")  # Process departures first
edge_fn.setMessageInput("entry_request")
edge_fn.setMessageOutput("entry_accept")
```

### Packet Agent

```python
packet_agent = model.newAgent("Packet")
packet_agent.newVariableInt("size")
packet_agent.newVariableInt("curr_edge")
packet_agent.newVariableInt("next_edge")
packet_agent.newVariableFloat("remaining_time")
packet_agent.newVariableFloat("entry_time")
packet_agent.newVariableArrayInt("route", 32)
packet_agent.newVariableInt("route_length")
packet_agent.newVariableInt("route_idx")

# State definitions:
packet_agent.newState("traveling")
packet_agent.newState("waiting")
packet_agent.setInitialState("traveling")

# Agent functions:
move_fn = packet_agent.newFunction("move_and_request", move_and_request)
move_fn.setInitialState("traveling")
move_fn.setEndState("waiting")
move_fn.setMessageOutput("departure_notice")  # Output departure when finishing edge
move_fn.setMessageOutput("entry_request")     # Output request for next edge
move_fn.setMessageOutputOptional(True)         # Only output if finishing edge

wait_fn = packet_agent.newFunction("wait_for_entry", wait_for_entry)
wait_fn.setInitialState("waiting")
wait_fn.setEndState("traveling")
wait_fn.setMessageInput("entry_accept")
wait_fn.setMessageOutput("entry_request")
wait_fn.setMessageOutputOptional(True)
```

### SignalController Agent

```python
signal_agent = model.newAgent("SignalController")
signal_agent.newVariableInt("node_id")
signal_agent.newVariableInt("phase_index")
signal_agent.newVariableInt("phase_count")
signal_agent.newVariableFloat("time_to_phase_end")
signal_agent.newVariableArrayFloat("phase_durations", 10)

# Agent function:
sig_fn = signal_agent.newFunction("update_signal", update_signal)
```

## Signal-to-Edge Communication

For signal control, we can use messages from signal to edges:

```python
green_msg = model.newMessageBruteForce("green_signal")
green_msg.newVariableInt("edge_id")

# Add a secondary function for EdgeQueue
edge_recv = edge_agent.newFunction("update_green_flag", update_green_flag)
edge_recv.setMessageInput("green_signal")

# Implementation of update_green_flag:
@pyflamegpu.agent_function
def update_green_flag(message_in: pyflamegpu.MessageBruteForce, message_out: pyflamegpu.MessageNone):
    edge_id = pyflamegpu.getVariableInt("edge_id")
    is_green = 0  # Default to red
    
    # Check if this edge received a green signal
    for msg in message_in:
        if msg.getVariableInt("edge_id") == edge_id:
            is_green = 1
            break
    
    pyflamegpu.setVariableInt("is_green", is_green)
    return pyflamegpu.ALIVE

# SignalController outputs green signals:
@pyflamegpu.agent_function
def update_signal(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageBruteForce):
    # ... phase update logic ...
    
    # Output green signals for edges in current phase
    phase_index = pyflamegpu.getVariableInt("phase_index")
    green_edges = get_green_edges_for_phase(phase_index)  # Lookup function
    
    for edge_id in green_edges:
        msg = message_out.append()
        msg.setVariableInt("edge_id", edge_id)
    
    return pyflamegpu.ALIVE
```

## Execution Configuration

We add layers to enforce ordering:

```python
# Layering
layer1 = model.newLayer("L1_move")
layer1.addAgentFunction("Packet", "move_and_request")

layer2 = model.newLayer("L2_signal")
layer2.addAgentFunction("SignalController", "update_signal")

layer3a = model.newLayer("L3a_set_green")
layer3a.addAgentFunction("EdgeQueue", "update_green_flag")

layer3b = model.newLayer("L3b_process")
layer3b.addAgentFunction("EdgeQueue", "process_edge_requests")

layer4 = model.newLayer("L4_waiting")
layer4.addAgentFunction("Packet", "wait_for_entry")
```

## Environment Setup

```python
env = model.Environment()
env.newPropertyFloat("time_step", 1.0)
env.newPropertyFloat("current_time", 0.0)

# Edge property arrays for packet lookup (read-only, can be marked const)
env.newPropertyArrayFloat("edge_lengths", edge_lengths_list)  # Length for each edge
env.newPropertyArrayFloat("edge_speeds", edge_speeds_list)     # Free speed for each edge

# For logging outputs:
env.newMacroPropertyFloat("edge_travel_time", num_edges)
env.newMacroPropertyFloat("edge_flow", num_edges)
env.newMacroPropertyFloat("edge_CO2", num_edges)
```

## Running the Simulation

```python
simulation = pyflamegpu.CUDASimulation(model)
simulation.initialise()

# Add host function for spawning vehicles
model.addStepFunction(spawn_packets_host)

simulation.simulate()
```

## Host Function for Spawning

```python
@pyflamegpu.host_function
def spawn_packets(simulation):
    global depart_index, grouped_deps
    current_step = simulation.getStepCounter()
    current_time = current_step * simulation.getSimulationConfig().time_step
    
    # Check departures at this time
    while depart_index < len(grouped_deps) and abs(grouped_deps[depart_index][0] - current_time) < 1e-6:
        t, origin, route_list, count = grouped_deps[depart_index]
        
        # Create packet agent
        agent = simulation.Agent("Packet")
        agent.setVariableInt("size", count)
        agent.setVariableInt("curr_edge", origin)
        agent.setVariableInt("next_edge", route_list[1] if len(route_list)>1 else -1)
        agent.setVariableArrayInt("route", [edge_id_map[e] for e in route_list] + [-1]*(32-len(route_list)))
        agent.setVariableInt("route_idx", 0)
        agent.setVariableInt("route_length", len(route_list))
        agent.setVariableFloat("remaining_time", edge_length[origin] / edge_speed[origin])
        agent.setVariableFloat("entry_time", current_time)
        agent.setState("traveling")
        simulation.addAgent(agent)
        
        depart_index += 1
```

## Initialization Function

```python
@pyflamegpu.init_function
def init_model(sim):
    for each edge in edges_data:
        agent = sim.Agent("EdgeQueue")
        agent.setVariableInt("edge_id", ...)
        # ... set other variables ...
        sim.addAgent(agent)
    
    for each signal in signals_data:
        agent = sim.Agent("SignalController")
        # ... set other variables ...
        sim.addAgent(agent)
```

## Validation

Once implemented, one should verify on a small network that:

- Packets move and queue correctly (no starvation or lost messages)
- Edge counts `curr_count` remain sensible (not negative or exceeding capacity beyond allowed)
- The simulation stops when no more vehicles or when time is up

