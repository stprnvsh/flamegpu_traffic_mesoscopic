"""
Agent Definitions for FLAMEGPU2 Mesoscopic Traffic Simulation

This module defines the three primary agent types:
1. EdgeQueue - Road segments with queue dynamics
2. Packet - Groups of vehicles traveling together
3. SignalController - Traffic signal controllers

Each agent has:
- Variable definitions (state)
- Agent functions (behavior)
- State definitions (for Packet only)

Reference: FLAMEGPU2 Agent Documentation
https://docs.flamegpu.com/guide/defining-agents/
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# Agent Function Code (RTC - Runtime Compiled)
# =============================================================================

# These are the CUDA/C++ agent function code strings that will be compiled
# at runtime by FLAMEGPU2's RTC (Runtime Compilation) system.

MOVE_AND_REQUEST_CODE = """
FLAMEGPU_AGENT_FUNCTION(move_and_request, flamegpu::MessageNone, flamegpu::MessageBucket) {
    const float dt = FLAMEGPU->environment.getProperty<float>("time_step");
    const float teleport_threshold = 180.0f;  // Teleport if waiting > 180s (like SUMO)
    
    // Check if already waiting for acceptance (don't decrement time again)
    const int ready = FLAMEGPU->getVariable<int>("ready_to_move");
    if (ready == 1) {
        // Increment wait time for teleporting decision
        float wait_time = FLAMEGPU->getVariable<float>("wait_time");
        wait_time += dt;
        FLAMEGPU->setVariable<float>("wait_time", wait_time);
        
        // TELEPORT: If waiting too long, die (simulates reaching destination via teleport)
        if (wait_time > teleport_threshold) {
            return flamegpu::DEAD;  // Teleport to destination
        }
        
        // Already waiting - check if we should die (route complete)
        const int next_edge = FLAMEGPU->getVariable<int>("next_edge");
        const int route_idx = FLAMEGPU->getVariable<int>("route_idx");
        const int route_length = FLAMEGPU->getVariable<int>("route_length");
        
        if (next_edge == -1 || next_edge < 0 || route_idx >= route_length - 1) {
            return flamegpu::DEAD;
        }
        
        // Resend request
        const int packet_size = FLAMEGPU->getVariable<int>("size");
        const int curr_edge = FLAMEGPU->getVariable<int>("curr_edge");
        FLAMEGPU->message_out.setKey(next_edge);
        FLAMEGPU->message_out.setVariable<int>("size", packet_size);
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("agent_id", FLAMEGPU->getID());
        FLAMEGPU->message_out.setVariable<int>("from_edge", curr_edge);
        return flamegpu::ALIVE;
    }
    
    // Decrement remaining travel time
    float rem = FLAMEGPU->getVariable<float>("remaining_time");
    rem -= dt;
    FLAMEGPU->setVariable<float>("remaining_time", rem);
    
    // If still traveling, continue
    if (rem > 0.0f) {
        return flamegpu::ALIVE;
    }
    
    // Reached end of current edge - mark as ready to transition
    FLAMEGPU->setVariable<int>("ready_to_move", 1);
    
    const int curr_edge = FLAMEGPU->getVariable<int>("curr_edge");
    const int packet_size = FLAMEGPU->getVariable<int>("size");
    const int next_edge = FLAMEGPU->getVariable<int>("next_edge");
    const int route_idx = FLAMEGPU->getVariable<int>("route_idx");
    const int route_length = FLAMEGPU->getVariable<int>("route_length");
    
    // If no next edge OR we've completed the route, destination reached - die
    if (next_edge == -1 || next_edge < 0 || route_idx >= route_length - 1) {
        return flamegpu::DEAD;
    }
    
    // Send request to enter next edge
    FLAMEGPU->message_out.setKey(next_edge);
    FLAMEGPU->message_out.setVariable<int>("size", packet_size);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("from_edge", curr_edge);
    
    return flamegpu::ALIVE;
}
"""

SEND_DEPARTURE_CODE = """
FLAMEGPU_AGENT_FUNCTION(send_departure, flamegpu::MessageNone, flamegpu::MessageBucket) {
    // Only send departure ONCE when first finishing an edge
    // ready_to_move=1 means we already sent departure and are waiting for acceptance
    const int ready = FLAMEGPU->getVariable<int>("ready_to_move");
    if (ready == 1) {
        // Already sent departure, waiting for acceptance - don't send again
        return flamegpu::ALIVE;
    }
    
    const float rem = FLAMEGPU->getVariable<float>("remaining_time");
    
    if (rem <= 0.0f) {
        const int curr_edge = FLAMEGPU->getVariable<int>("curr_edge");
        const int packet_size = FLAMEGPU->getVariable<int>("size");
        
        // Send departure notice to current edge
        FLAMEGPU->message_out.setKey(curr_edge);
        FLAMEGPU->message_out.setVariable<int>("size", packet_size);
    }
    
    return flamegpu::ALIVE;
}
"""

def get_wait_for_entry_code(max_route_length: int) -> str:
    """Generate WAIT_FOR_ENTRY_CODE with dynamic route array size"""
    return f"""
FLAMEGPU_AGENT_FUNCTION(wait_for_entry, flamegpu::MessageBucket, flamegpu::MessageBucket) {{
    // Check if we're actually ready to transition (finished current edge)
    const int ready = FLAMEGPU->getVariable<int>("ready_to_move");
    if (ready == 0) {{
        // Not ready - still traveling on current edge, just return
        return flamegpu::ALIVE;
    }}
    
    const flamegpu::id_t my_id = FLAMEGPU->getID();
    bool accepted = false;
    int accepted_edge = -1;
    float accepted_travel_time = 0.0f;
    
    // Check if we received an acceptance message (O(1) bucket lookup by our ID)
    // EdgeQueue includes travel_time and out_node in acceptance so we don't need env arrays
    int accepted_out_node = -1;
    for (const auto& msg : FLAMEGPU->message_in(static_cast<unsigned int>(my_id))) {{
        accepted = true;
        accepted_edge = msg.getVariable<int>("edge_id");
        accepted_travel_time = msg.getVariable<float>("travel_time");
        accepted_out_node = msg.getVariable<int>("out_node");
        break;  // Only one acceptance per packet
    }}
    
    if (!accepted) {{
        // Still waiting, resend request (only if we have a valid next edge)
        const int next_edge = FLAMEGPU->getVariable<int>("next_edge");
        
        // If no valid next edge, we shouldn't be waiting - die
        if (next_edge < 0) {{
            return flamegpu::DEAD;
        }}
        
        const int packet_size = FLAMEGPU->getVariable<int>("size");
        const int curr_edge = FLAMEGPU->getVariable<int>("curr_edge");
        
        FLAMEGPU->message_out.setKey(next_edge);
        FLAMEGPU->message_out.setVariable<int>("size", packet_size);
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("agent_id", my_id);
        FLAMEGPU->message_out.setVariable<int>("from_edge", curr_edge);
        
        return flamegpu::ALIVE;
    }}
    
    // Accepted - transition to traveling on new edge
    FLAMEGPU->setVariable<int>("curr_edge", accepted_edge);
    FLAMEGPU->setVariable<int>("curr_node", accepted_out_node);  // Track current node for rerouting
    FLAMEGPU->setVariable<int>("ready_to_move", 0);  // Reset flag - now traveling again
    FLAMEGPU->setVariable<float>("wait_time", 0.0f);  // Reset wait time
    
    // Advance route index
    int route_idx = FLAMEGPU->getVariable<int>("route_idx");
    route_idx += 1;
    FLAMEGPU->setVariable<int>("route_idx", route_idx);
    
    // Get next edge from route (array size determined dynamically from routes data)
    const int route_length = FLAMEGPU->getVariable<int>("route_length");
    if (route_idx + 1 < route_length) {{
        const int next_edge2 = FLAMEGPU->getVariable<int, {max_route_length}>("route", route_idx + 1);
        FLAMEGPU->setVariable<int>("next_edge", next_edge2);
    }} else {{
        FLAMEGPU->setVariable<int>("next_edge", -1);  // Route complete
    }}
    
    // Travel time comes from acceptance message (EdgeQueue already calculated it)
    FLAMEGPU->setVariable<float>("remaining_time", accepted_travel_time);
    
    // Record entry time
    const float current_time = FLAMEGPU->environment.getProperty<float>("current_time");
    FLAMEGPU->setVariable<float>("entry_time", current_time);
    
    return flamegpu::ALIVE;
}}
"""

# Default for backwards compatibility
WAIT_FOR_ENTRY_CODE = get_wait_for_entry_code(256)

def get_try_reroute_code(max_route_length: int) -> str:
    """Generate TRY_REROUTE_CODE for GPU-side local rerouting"""
    return f"""
FLAMEGPU_AGENT_FUNCTION(try_reroute, flamegpu::MessageBucket, flamegpu::MessageNone) {{
    // Only try rerouting if stuck (waiting too long)
    const float wait_time = FLAMEGPU->getVariable<float>("wait_time");
    const float reroute_threshold = 60.0f;  // Try reroute after 60s waiting
    
    if (wait_time < reroute_threshold) {{
        return flamegpu::ALIVE;  // Not stuck long enough
    }}
    
    const int curr_node = FLAMEGPU->getVariable<int>("curr_node");
    if (curr_node < 0) {{
        return flamegpu::ALIVE;  // Don't know current node
    }}
    
    const int current_next = FLAMEGPU->getVariable<int>("next_edge");
    const int dest_node = FLAMEGPU->getVariable<int>("dest_node");
    
    // Scan edge_status messages from curr_node to find alternatives
    int best_edge = -1;
    int best_capacity = 0;
    float best_time = 1e9f;
    
    for (const auto& msg : FLAMEGPU->message_in(curr_node)) {{
        const int edge_id = msg.getVariable<int>("edge_id");
        const int to_node = msg.getVariable<int>("to_node");
        const int avail_cap = msg.getVariable<int>("available_capacity");
        const float travel_time = msg.getVariable<float>("travel_time");
        
        // Skip current blocked edge
        if (edge_id == current_next) {{
            continue;
        }}
        
        // Only consider edges with available capacity
        if (avail_cap > 0) {{
            // Prefer edges that lead toward destination (simple heuristic)
            // For now, just pick the one with most capacity
            if (avail_cap > best_capacity || 
                (avail_cap == best_capacity && travel_time < best_time)) {{
                best_edge = edge_id;
                best_capacity = avail_cap;
                best_time = travel_time;
            }}
        }}
    }}
    
    // If found a better alternative, update route
    if (best_edge >= 0 && best_edge != current_next) {{
        FLAMEGPU->setVariable<int>("next_edge", best_edge);
        FLAMEGPU->setVariable<float>("wait_time", 0.0f);  // Reset wait time
        
        // Update route array to reflect the detour
        // For simplicity, just update next_edge; full route recomputation would be ideal
        // The packet will continue from the new edge
    }}
    
    return flamegpu::ALIVE;
}}
"""

TRY_REROUTE_CODE = get_try_reroute_code(256)

PROCESS_EDGE_REQUESTS_CODE = """
FLAMEGPU_AGENT_FUNCTION(process_edge_requests, flamegpu::MessageBucket, flamegpu::MessageBucket) {
    const int edge_id = FLAMEGPU->getVariable<int>("edge_id");
    int curr_count = FLAMEGPU->getVariable<int>("curr_count");
    const int capacity = FLAMEGPU->getVariable<int>("capacity");
    const float travel_time = FLAMEGPU->getVariable<float>("travel_time");
    const int out_node = FLAMEGPU->getVariable<int>("out_node");  // For packet rerouting
    
    // Check signal state if controlled
    const int signal_id = FLAMEGPU->getVariable<int>("signal_id");
    if (signal_id != -1) {
        const int is_green = FLAMEGPU->getVariable<int>("is_green");
        if (is_green == 0) {
            // Red light - don't accept any requests
            return flamegpu::ALIVE;
        }
    }
    
    // Calculate available space
    int available = capacity - curr_count;
    if (available <= 0) {
        return flamegpu::ALIVE;  // No space
    }
    
    // Process requests (iterate messages for this edge)
    int accepted_count = 0;
    
    for (const auto& msg : FLAMEGPU->message_in(edge_id)) {
        const int req_size = msg.getVariable<int>("size");
        const flamegpu::id_t req_id = msg.getVariable<flamegpu::id_t>("agent_id");
        
        if (accepted_count + req_size <= available) {
            // Accept this request
            accepted_count += req_size;
            
            // Send acceptance message with travel_time and out_node (for rerouting)
            // Key by agent_id for O(1) packet lookup (instead of O(n) BruteForce scan)
            FLAMEGPU->message_out.setKey(static_cast<unsigned int>(req_id));
            FLAMEGPU->message_out.setVariable<int>("edge_id", edge_id);
            FLAMEGPU->message_out.setVariable<float>("travel_time", travel_time);
            FLAMEGPU->message_out.setVariable<int>("out_node", out_node);
        }
    }
    
    // Update edge count
    FLAMEGPU->setVariable<int>("curr_count", curr_count + accepted_count);
    
    return flamegpu::ALIVE;
}
"""

PROCESS_DEPARTURES_CODE = """
FLAMEGPU_AGENT_FUNCTION(process_departures, flamegpu::MessageBucket, flamegpu::MessageNone) {
    const int edge_id = FLAMEGPU->getVariable<int>("edge_id");
    int curr_count = FLAMEGPU->getVariable<int>("curr_count");
    
    // Process all departure messages for this edge
    for (const auto& msg : FLAMEGPU->message_in(edge_id)) {
        const int depart_size = msg.getVariable<int>("size");
        curr_count -= depart_size;
    }
    
    // Ensure non-negative
    if (curr_count < 0) curr_count = 0;
    
    FLAMEGPU->setVariable<int>("curr_count", curr_count);
    
    // Update travel time based on current density
    const float length = FLAMEGPU->getVariable<float>("length");
    const float free_speed = FLAMEGPU->getVariable<float>("free_speed");
    const int capacity = FLAMEGPU->getVariable<int>("capacity");
    
    // Simple congestion model: speed decreases linearly with occupancy
    float occupancy = (float)curr_count / (float)capacity;
    if (occupancy > 1.0f) occupancy = 1.0f;
    
    float speed = free_speed * (1.0f - 0.5f * occupancy);  // Speed drops to 50% at capacity
    if (speed < 1.0f) speed = 1.0f;  // Minimum speed
    
    float travel_time = length / speed;
    FLAMEGPU->setVariable<float>("travel_time", travel_time);
    
    return flamegpu::ALIVE;
}
"""

UPDATE_SIGNAL_CODE = """
FLAMEGPU_AGENT_FUNCTION(update_signal, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    const float dt = FLAMEGPU->environment.getProperty<float>("time_step");
    float time_left = FLAMEGPU->getVariable<float>("time_to_phase_end");
    int phase_index = FLAMEGPU->getVariable<int>("phase_index");
    const int phase_count = FLAMEGPU->getVariable<int>("phase_count");
    
    time_left -= dt;
    
    if (time_left <= 0.0f) {
        // Advance to next phase
        phase_index = (phase_index + 1) % phase_count;
        FLAMEGPU->setVariable<int>("phase_index", phase_index);
        
        // Get duration of new phase
        const float duration = FLAMEGPU->getVariable<float, 32>("phase_durations", phase_index);
        time_left = duration;
    }
    
    FLAMEGPU->setVariable<float>("time_to_phase_end", time_left);
    
    // Output green signal for edges in current phase
    // Each signal controls specific edges based on phase
    // We output messages that edges will read
    const int node_id = FLAMEGPU->getVariable<int>("node_id");
    
    // Get green edges for this phase from array
    // phase_green_edges stores edge IDs that are green for each phase
    // Format: [phase0_edge0, phase0_edge1, ..., phase1_edge0, ...]
    const int max_edges_per_phase = 16;
    for (int i = 0; i < max_edges_per_phase; i++) {
        int edge_id = FLAMEGPU->getVariable<int, 512>("phase_green_edges", phase_index * max_edges_per_phase + i);
        if (edge_id >= 0) {
            FLAMEGPU->message_out.setVariable<int>("edge_id", edge_id);
            FLAMEGPU->message_out.setVariable<int>("node_id", node_id);
        }
    }
    
    return flamegpu::ALIVE;
}
"""

UPDATE_GREEN_FLAG_CODE = """
FLAMEGPU_AGENT_FUNCTION(update_green_flag, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    const int edge_id = FLAMEGPU->getVariable<int>("edge_id");
    const int signal_id = FLAMEGPU->getVariable<int>("signal_id");
    
    // If not controlled by a signal, always green
    if (signal_id == -1) {
        FLAMEGPU->setVariable<int>("is_green", 1);
        return flamegpu::ALIVE;
    }
    
    // Check if we received a green signal
    int is_green = 0;
    for (const auto& msg : FLAMEGPU->message_in) {
        if (msg.getVariable<int>("edge_id") == edge_id) {
            is_green = 1;
            break;
        }
    }
    
    FLAMEGPU->setVariable<int>("is_green", is_green);
    
    return flamegpu::ALIVE;
}
"""

# Broadcast edge status for GPU-side rerouting
BROADCAST_STATUS_CODE = """
FLAMEGPU_AGENT_FUNCTION(broadcast_status, flamegpu::MessageNone, flamegpu::MessageBucket) {
    const int edge_id = FLAMEGPU->getVariable<int>("edge_id");
    const int from_node = FLAMEGPU->getVariable<int>("from_node");
    const int to_node = FLAMEGPU->getVariable<int>("out_node");
    const int capacity = FLAMEGPU->getVariable<int>("capacity");
    const int curr_count = FLAMEGPU->getVariable<int>("curr_count");
    const float travel_time = FLAMEGPU->getVariable<float>("travel_time");
    
    // Broadcast status keyed by from_node (so packets can find alternatives from a node)
    if (from_node >= 0) {
        FLAMEGPU->message_out.setKey(from_node);
        FLAMEGPU->message_out.setVariable<int>("edge_id", edge_id);
        FLAMEGPU->message_out.setVariable<int>("to_node", to_node);
        FLAMEGPU->message_out.setVariable<int>("available_capacity", capacity - curr_count);
        FLAMEGPU->message_out.setVariable<float>("travel_time", travel_time);
    }
    
    return flamegpu::ALIVE;
}
"""


# =============================================================================
# Agent Configuration Dataclasses
# =============================================================================

@dataclass
class EdgeQueueConfig:
    """Configuration for EdgeQueue agent type"""
    max_edges: int = 100000  # Now unlimited - RTC doesn't use env arrays
    
    # Variable names and types
    variables: Dict[str, str] = field(default_factory=lambda: {
        'edge_id': 'int',
        'capacity': 'int',
        'curr_count': 'int',
        'length': 'float',
        'free_speed': 'float',
        'signal_id': 'int',      # -1 if not signalized
        'is_green': 'int',       # 1 = green, 0 = red
        'travel_time': 'float',
        'out_node': 'int',
        'lane_count': 'int',
    })


@dataclass
class PacketConfig:
    """Configuration for Packet agent type"""
    max_route_length: int = 256  # Default, will be overridden dynamically based on routes
    
    # Variable names and types
    variables: Dict[str, str] = field(default_factory=lambda: {
        'size': 'int',
        'curr_edge': 'int',
        'next_edge': 'int',
        'remaining_time': 'float',
        'entry_time': 'float',
        'route_idx': 'int',
        'route_length': 'int',
    })
    
    # Array variables
    array_variables: Dict[str, tuple] = field(default_factory=lambda: {
        'route': ('int', 32),  # Fixed-size route array
    })
    
    # Agent states
    states: List[str] = field(default_factory=lambda: ['traveling', 'waiting'])
    initial_state: str = 'traveling'


@dataclass
class SignalControllerConfig:
    """Configuration for SignalController agent type"""
    max_phases: int = 32
    max_edges_per_phase: int = 16
    
    # Variable names and types
    variables: Dict[str, str] = field(default_factory=lambda: {
        'node_id': 'int',
        'phase_index': 'int',
        'phase_count': 'int',
        'time_to_phase_end': 'float',
        'cycle_length': 'float',
    })
    
    # Array variables
    array_variables: Dict[str, tuple] = field(default_factory=lambda: {
        'phase_durations': ('float', 32),
        'phase_green_edges': ('int', 512),  # 32 phases × 16 edges per phase
    })


# =============================================================================
# Agent Definition Functions
# =============================================================================

def define_edge_queue_agent(model, config: Optional[EdgeQueueConfig] = None):
    """
    Define the EdgeQueue agent in a FLAMEGPU2 model
    
    Args:
        model: pyflamegpu.ModelDescription
        config: Optional EdgeQueueConfig
        
    Returns:
        The EdgeQueue agent description
    """
    if config is None:
        config = EdgeQueueConfig()
    
    agent = model.newAgent("EdgeQueue")
    
    # Define variables
    agent.newVariableInt("edge_id")
    agent.newVariableInt("capacity")
    agent.newVariableInt("curr_count", 0)
    agent.newVariableFloat("length")
    agent.newVariableFloat("free_speed")
    agent.newVariableInt("signal_id", -1)
    agent.newVariableInt("is_green", 1)  # Default to green (unsignalized)
    agent.newVariableFloat("travel_time")
    agent.newVariableInt("from_node", -1)  # Origin node for rerouting
    agent.newVariableInt("out_node")       # Destination node (to_node)
    agent.newVariableInt("lane_count", 1)
    
    # Define agent functions using RTC
    fn_process_departures = agent.newRTCFunction("process_departures", PROCESS_DEPARTURES_CODE)
    fn_process_departures.setMessageInput("departure_notice")
    
    fn_update_green = agent.newRTCFunction("update_green_flag", UPDATE_GREEN_FLAG_CODE)
    fn_update_green.setMessageInput("green_signal")
    
    fn_process_requests = agent.newRTCFunction("process_edge_requests", PROCESS_EDGE_REQUESTS_CODE)
    fn_process_requests.setMessageInput("entry_request")
    fn_process_requests.setMessageOutput("entry_accept")
    
    # Broadcast status for GPU-side rerouting
    fn_broadcast = agent.newRTCFunction("broadcast_status", BROADCAST_STATUS_CODE)
    fn_broadcast.setMessageOutput("edge_status")
    
    return agent


def define_packet_agent(model, config: Optional[PacketConfig] = None):
    """
    Define the Packet agent in a FLAMEGPU2 model
    
    Args:
        model: pyflamegpu.ModelDescription
        config: Optional PacketConfig
        
    Returns:
        The Packet agent description
    """
    if config is None:
        config = PacketConfig()
    
    agent = model.newAgent("Packet")
    
    # Define variables
    agent.newVariableInt("size", 1)
    agent.newVariableInt("curr_edge")
    agent.newVariableInt("next_edge", -1)
    agent.newVariableFloat("remaining_time")
    agent.newVariableFloat("entry_time", 0.0)
    agent.newVariableInt("route_idx", 0)
    agent.newVariableInt("route_length")
    agent.newVariableInt("ready_to_move", 0)  # Flag: 1 = finished edge, waiting to move
    agent.newVariableInt("destination", -1)  # Final destination edge for rerouting
    agent.newVariableFloat("wait_time", 0.0)  # Time spent waiting for next edge (for reroute trigger)
    agent.newVariableInt("curr_node", -1)    # Current node (end of curr_edge) for finding alternatives
    agent.newVariableInt("dest_node", -1)    # Destination node for choosing direction
    
    # Route array (fixed size)
    agent.newVariableArrayInt("route", config.max_route_length)
    
    # Define states
    agent.newState("traveling")
    agent.newState("waiting")
    agent.setInitialState("traveling")
    
    # Define agent functions
    
    # 1. Send departure notice (traveling state only, runs before moving)
    fn_depart = agent.newRTCFunction("send_departure", SEND_DEPARTURE_CODE)
    fn_depart.setInitialState("traveling")
    fn_depart.setEndState("traveling")
    fn_depart.setMessageOutput("departure_notice")
    fn_depart.setMessageOutputOptional(True)
    
    # 2. Move and request (traveling → waiting transition)
    fn_move = agent.newRTCFunction("move_and_request", MOVE_AND_REQUEST_CODE)
    fn_move.setInitialState("traveling")
    fn_move.setEndState("waiting")
    fn_move.setMessageOutput("entry_request")
    fn_move.setMessageOutputOptional(True)
    fn_move.setAllowAgentDeath(True)
    
    # 3. Try reroute (waiting state, GPU-side local rerouting)
    # Reads edge_status to find alternatives if stuck
    reroute_code = get_try_reroute_code(config.max_route_length)
    fn_reroute = agent.newRTCFunction("try_reroute", reroute_code)
    fn_reroute.setInitialState("waiting")
    fn_reroute.setEndState("waiting")  # Stay waiting, but with updated next_edge
    fn_reroute.setMessageInput("edge_status")
    
    # 4. Wait for entry (waiting → traveling transition)
    # Use dynamic route length from config
    wait_code = get_wait_for_entry_code(config.max_route_length)
    fn_wait = agent.newRTCFunction("wait_for_entry", wait_code)
    fn_wait.setInitialState("waiting")
    fn_wait.setEndState("traveling")
    fn_wait.setMessageInput("entry_accept")
    fn_wait.setAllowAgentDeath(True)  # Can die if no valid next edge
    fn_wait.setMessageOutput("entry_request")
    fn_wait.setMessageOutputOptional(True)
    
    return agent


def define_signal_controller_agent(model, config: Optional[SignalControllerConfig] = None):
    """
    Define the SignalController agent in a FLAMEGPU2 model
    
    Args:
        model: pyflamegpu.ModelDescription
        config: Optional SignalControllerConfig
        
    Returns:
        The SignalController agent description
    """
    if config is None:
        config = SignalControllerConfig()
    
    agent = model.newAgent("SignalController")
    
    # Define variables
    agent.newVariableInt("node_id")
    agent.newVariableInt("phase_index", 0)
    agent.newVariableInt("phase_count")
    agent.newVariableFloat("time_to_phase_end")
    agent.newVariableFloat("cycle_length")
    
    # Array variables
    agent.newVariableArrayFloat("phase_durations", config.max_phases)
    agent.newVariableArrayInt("phase_green_edges", 
                              config.max_phases * config.max_edges_per_phase)
    
    # Define agent function
    fn_update = agent.newRTCFunction("update_signal", UPDATE_SIGNAL_CODE)
    fn_update.setMessageOutput("green_signal")
    
    return agent

