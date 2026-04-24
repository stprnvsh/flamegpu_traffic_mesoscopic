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
        
        // TELEPORT: If waiting too long, die
        // Note: departure notice was sent before, so curr_count already decremented
        if (wait_time > teleport_threshold) {
            return flamegpu::DEAD;
        }
        
        // Get next segment to request
        const int next_segment = FLAMEGPU->getVariable<int>("next_segment");
        const int route_idx = FLAMEGPU->getVariable<int>("route_idx");
        const int route_length = FLAMEGPU->getVariable<int>("route_length");
        
        // If no next segment AND at end of route, destination reached - die
        // Note: departure notice was already sent, so curr_count decremented
        if (next_segment < 0 && route_idx >= route_length - 1) {
            return flamegpu::DEAD;
        }
        
        // Resend request to next segment
        const int packet_size = FLAMEGPU->getVariable<int>("size");
        const int curr_segment = FLAMEGPU->getVariable<int>("curr_segment");
        const int is_jammed = FLAMEGPU->getVariable<int>("is_jammed");
        
        if (next_segment >= 0) {
            FLAMEGPU->message_out.setKey(next_segment);
            FLAMEGPU->message_out.setVariable<int>("size", packet_size);
            FLAMEGPU->message_out.setVariable<flamegpu::id_t>("agent_id", FLAMEGPU->getID());
            FLAMEGPU->message_out.setVariable<int>("from_edge", curr_segment);
            FLAMEGPU->message_out.setVariable<int>("is_jammed", is_jammed);
        }
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
    
    // Reached end of current segment - mark as ready to transition
    // This triggers send_departure in the next layer to send departure notice
    FLAMEGPU->setVariable<int>("ready_to_move", 1);
    
    const int curr_segment = FLAMEGPU->getVariable<int>("curr_segment");
    const int packet_size = FLAMEGPU->getVariable<int>("size");
    const int next_segment = FLAMEGPU->getVariable<int>("next_segment");
    const int route_idx = FLAMEGPU->getVariable<int>("route_idx");
    const int route_length = FLAMEGPU->getVariable<int>("route_length");
    const int is_jammed = FLAMEGPU->getVariable<int>("is_jammed");
    
    // DON'T DIE YET - wait for send_departure to run first!
    // The send_departure function will send departure notice this step,
    // then next step we'll check route completion and die if needed.
    // This ensures curr_count is properly decremented.
    
    // Send request to enter next segment (if there is one)
    if (next_segment >= 0) {
        FLAMEGPU->message_out.setKey(next_segment);
        FLAMEGPU->message_out.setVariable<int>("size", packet_size);
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("agent_id", FLAMEGPU->getID());
        FLAMEGPU->message_out.setVariable<int>("from_edge", curr_segment);
        FLAMEGPU->message_out.setVariable<int>("is_jammed", is_jammed);
    }
    
    return flamegpu::ALIVE;
}
"""

SEND_DEPARTURE_CODE = """
FLAMEGPU_AGENT_FUNCTION(send_departure, flamegpu::MessageNone, flamegpu::MessageBucket) {
    // Send departure when: ready_to_move==1 (just finished segment) AND departure_sent==0 (haven't sent yet)
    const int ready = FLAMEGPU->getVariable<int>("ready_to_move");
    const int sent = FLAMEGPU->getVariable<int>("departure_sent");
    
    if (ready == 1 && sent == 0) {
        const int curr_segment = FLAMEGPU->getVariable<int>("curr_segment");
        const int packet_size = FLAMEGPU->getVariable<int>("size");
        
        // Send departure notice to current segment
        FLAMEGPU->message_out.setKey(curr_segment);
        FLAMEGPU->message_out.setVariable<int>("size", packet_size);
        
        // Mark as sent so we don't send again
        FLAMEGPU->setVariable<int>("departure_sent", 1);
    }
    
    return flamegpu::ALIVE;
}
"""

def get_wait_for_entry_code(max_route_length: int) -> str:
    """Generate WAIT_FOR_ENTRY_CODE with dynamic route array size
    
    Handles segment-based movement:
    - Route contains FIRST segment of each edge (one entry per edge in route)
    - Each segment has next_segment (-1 if last in edge)
    - Only increment route_idx when entering a NEW EDGE (not same-edge segment)
    - Acceptance message includes next_segment info
    """
    return f"""
FLAMEGPU_AGENT_FUNCTION(wait_for_entry, flamegpu::MessageBucket, flamegpu::MessageBucket) {{
    // Check if we're actually ready to transition (finished current segment)
    const int ready = FLAMEGPU->getVariable<int>("ready_to_move");
    if (ready == 0) {{
        // Not ready - still traveling on current segment, just return
        return flamegpu::ALIVE;
    }}
    
    const flamegpu::id_t my_id = FLAMEGPU->getID();
    bool accepted = false;
    int accepted_segment = -1;
    float accepted_travel_time = 0.0f;
    int accepted_next_segment = -1;  // Next segment in same edge, or -1 if last
    int accepted_edge_idx = -1;      // Edge index of this segment
    int accepted_out_node = -1;
    
    // Check if we received an acceptance message (O(1) bucket lookup by our ID)
    for (const auto& msg : FLAMEGPU->message_in(static_cast<unsigned int>(my_id))) {{
        accepted = true;
        accepted_segment = msg.getVariable<int>("edge_id");  // Actually segment_id
        accepted_travel_time = msg.getVariable<float>("travel_time");
        accepted_next_segment = msg.getVariable<int>("next_segment");
        accepted_edge_idx = msg.getVariable<int>("edge_idx");
        accepted_out_node = msg.getVariable<int>("out_node");
        break;  // Only one acceptance per packet
    }}
    
    if (!accepted) {{
        // Still waiting, resend request
        const int next_segment = FLAMEGPU->getVariable<int>("next_segment");
        
        // If no valid next segment, we shouldn't be waiting - die
        if (next_segment < 0) {{
            return flamegpu::DEAD;
        }}
        
        const int packet_size = FLAMEGPU->getVariable<int>("size");
        const int curr_segment = FLAMEGPU->getVariable<int>("curr_segment");
        const int is_jammed = FLAMEGPU->getVariable<int>("is_jammed");
        
        FLAMEGPU->message_out.setKey(next_segment);
        FLAMEGPU->message_out.setVariable<int>("size", packet_size);
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("agent_id", my_id);
        FLAMEGPU->message_out.setVariable<int>("from_edge", curr_segment);
        FLAMEGPU->message_out.setVariable<int>("is_jammed", is_jammed);
        
        return flamegpu::ALIVE;
    }}
    
    // Check if we're moving to a segment in the SAME edge or a NEW edge
    const int old_edge = FLAMEGPU->getVariable<int>("curr_edge");
    bool is_new_edge = (accepted_edge_idx != old_edge);
    
    // Accepted - transition to traveling on new segment
    FLAMEGPU->setVariable<int>("curr_segment", accepted_segment);
    FLAMEGPU->setVariable<int>("curr_edge", accepted_edge_idx);  // Update edge for metrics
    FLAMEGPU->setVariable<int>("curr_node", accepted_out_node);
    FLAMEGPU->setVariable<int>("ready_to_move", 0);  // Reset flag - now traveling again
    FLAMEGPU->setVariable<int>("departure_sent", 0);  // Reset for next segment
    FLAMEGPU->setVariable<float>("wait_time", 0.0f);  // Reset wait time
    
    int route_idx = FLAMEGPU->getVariable<int>("route_idx");
    const int route_length = FLAMEGPU->getVariable<int>("route_length");
    
    // Only advance route_idx when entering a NEW EDGE (route contains first segment per edge)
    if (is_new_edge) {{
        route_idx += 1;
        FLAMEGPU->setVariable<int>("route_idx", route_idx);
    }}
    
    // Determine next segment:
    // - If accepted_next_segment >= 0, that's the next segment in same edge (don't use route)
    // - If accepted_next_segment < 0 (last segment), get first segment of next edge from route
    if (accepted_next_segment >= 0) {{
        // More segments in current edge - use next_segment pointer
        FLAMEGPU->setVariable<int>("next_segment", accepted_next_segment);
    }} else if (route_idx + 1 < route_length) {{
        // Last segment in edge - get first segment of next edge from route
        const int next_seg = FLAMEGPU->getVariable<int, {max_route_length}>("route", route_idx + 1);
        FLAMEGPU->setVariable<int>("next_segment", next_seg);
    }} else {{
        // Route complete - no more edges/segments
        FLAMEGPU->setVariable<int>("next_segment", -1);
    }}
    
    // Travel time comes from acceptance message
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
    """Generate TRY_REROUTE_CODE for GPU-side local rerouting with jam awareness
    
    Note: In segment mode, this works at the segment level, not edge level.
    """
    return f"""
FLAMEGPU_AGENT_FUNCTION(try_reroute, flamegpu::MessageBucket, flamegpu::MessageNone) {{
    // Only try rerouting if stuck (waiting too long)
    const float wait_time = FLAMEGPU->getVariable<float>("wait_time");
    const float reroute_threshold = 60.0f;  // Try reroute after 60s waiting
    const float current_time = FLAMEGPU->environment.getProperty<float>("current_time");
    
    if (wait_time < reroute_threshold) {{
        return flamegpu::ALIVE;  // Not stuck long enough
    }}
    
    const int curr_node = FLAMEGPU->getVariable<int>("curr_node");
    if (curr_node < 0) {{
        return flamegpu::ALIVE;  // Don't know current node
    }}
    
    const int current_next = FLAMEGPU->getVariable<int>("next_segment");
    const int dest_node = FLAMEGPU->getVariable<int>("dest_node");
    
    // Scan edge_status messages from curr_node to find alternatives
    // Note: edge_status still broadcasts edge-level info (first segment of each edge)
    int best_segment = -1;
    float best_score = -1e9f;
    
    for (const auto& msg : FLAMEGPU->message_in(curr_node)) {{
        const int segment_id = msg.getVariable<int>("edge_id");  // First segment of edge
        const int to_node = msg.getVariable<int>("to_node");
        const int avail_cap = msg.getVariable<int>("available_capacity");
        const float travel_time = msg.getVariable<float>("travel_time");
        const int is_jammed = msg.getVariable<int>("is_jammed");
        const float block_time = msg.getVariable<float>("block_time");
        
        // Skip current blocked segment
        if (segment_id == current_next) {{
            continue;
        }}
        
        // Skip segments with no capacity
        if (avail_cap <= 0) {{
            continue;
        }}
        
        // Skip segments that are blocked (headway not satisfied)
        if (block_time > current_time + 5.0f) {{
            continue;
        }}
        
        // Score segments: prefer non-jammed, high capacity, low travel time
        float score = (float)avail_cap * 10.0f;
        score -= travel_time;
        if (is_jammed == 0) {{
            score += 50.0f;
        }}
        
        if (score > best_score) {{
            best_segment = segment_id;
            best_score = score;
        }}
    }}
    
    // If found a better alternative, update route
    if (best_segment >= 0 && best_segment != current_next) {{
        FLAMEGPU->setVariable<int>("next_segment", best_segment);
        FLAMEGPU->setVariable<float>("wait_time", 0.0f);
    }}
    
    return flamegpu::ALIVE;
}}
"""

TRY_REROUTE_CODE = get_try_reroute_code(256)

PROCESS_EDGE_REQUESTS_CODE = """
FLAMEGPU_AGENT_FUNCTION(process_edge_requests, flamegpu::MessageBucket, flamegpu::MessageBucket) {
    // This agent represents a SEGMENT (not full edge)
    // segment_id is stored in edge_id for compatibility
    const int segment_id = FLAMEGPU->getVariable<int>("edge_id");  // Actually segment index
    const int edge_idx = FLAMEGPU->getVariable<int>("edge_idx");   // Parent edge index
    const int next_segment = FLAMEGPU->getVariable<int>("next_segment");  // Next segment in edge (-1 if last)
    
    int curr_count = FLAMEGPU->getVariable<int>("curr_count");
    const int capacity = FLAMEGPU->getVariable<int>("capacity");
    const float length = FLAMEGPU->getVariable<float>("length");
    const float free_speed = FLAMEGPU->getVariable<float>("free_speed");
    const int out_node = FLAMEGPU->getVariable<int>("out_node");
    const float current_time = FLAMEGPU->environment.getProperty<float>("current_time");
    
    // Get tau parameters for SUMO-style headway calculation
    const float tau_ff = FLAMEGPU->getVariable<float>("tau_ff");
    const float tau_fj = FLAMEGPU->getVariable<float>("tau_fj");
    const float tau_jf = FLAMEGPU->getVariable<float>("tau_jf");
    const float tau_jj = FLAMEGPU->getVariable<float>("tau_jj");
    const int is_jammed = FLAMEGPU->getVariable<int>("is_jammed");
    float block_time = FLAMEGPU->getVariable<float>("block_time");
    
    // Check signal state if controlled (only last segment in edge has signal)
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
    
    // Calculate travel time for this segment
    // SUMO formula: segment_length / speed (basic, adjusted by queue if needed)
    float travel_time = length / free_speed;
    
    // Process requests (iterate messages for this segment)
    int accepted_count = 0;
    
    for (const auto& msg : FLAMEGPU->message_in(segment_id)) {
        const int req_size = msg.getVariable<int>("size");
        const flamegpu::id_t req_id = msg.getVariable<flamegpu::id_t>("agent_id");
        const int from_segment = msg.getVariable<int>("from_edge");  // Actually from_segment
        const int from_jammed = msg.getVariable<int>("is_jammed");
        
        // Check capacity
        if (accepted_count + req_size > available) {
            continue;  // No more space
        }
        
        // Check headway (block_time) - SUMO meso style
        if (current_time < block_time) {
            continue;  // Must wait for headway
        }
        
        // Accept this request
        accepted_count += req_size;
        
        // Calculate headway based on jam states (4-tau model)
        // SUMO: tau depends on current segment jam and next segment jam
        float tau;
        if (is_jammed == 0 && from_jammed == 0) {
            tau = tau_ff;  // Free to free
        } else if (is_jammed == 0 && from_jammed == 1) {
            tau = tau_fj;  // Free to jam
        } else if (is_jammed == 1 && from_jammed == 0) {
            tau = tau_jf;  // Jam to free
        } else {
            tau = tau_jj;  // Jam to jam
        }
        
        // Update block_time for next vehicle (headway constraint)
        // Add vehicle length factor (~5m / speed gives time gap)
        const float veh_length_time = 5.0f / (free_speed + 0.1f);
        block_time = current_time + tau + veh_length_time;
        
        // Send acceptance message with segment info
        FLAMEGPU->message_out.setKey(static_cast<unsigned int>(req_id));
        FLAMEGPU->message_out.setVariable<int>("edge_id", segment_id);  // Actually segment_id
        FLAMEGPU->message_out.setVariable<float>("travel_time", travel_time);
        FLAMEGPU->message_out.setVariable<int>("out_node", out_node);
        FLAMEGPU->message_out.setVariable<int>("next_segment", next_segment);  // Next segment in edge
        FLAMEGPU->message_out.setVariable<int>("edge_idx", edge_idx);  // Parent edge for metrics
    }
    
    // Update block_time
    FLAMEGPU->setVariable<float>("block_time", block_time);
    
    // Update segment count
    FLAMEGPU->setVariable<int>("curr_count", curr_count + accepted_count);
    
    // Store current travel time for status broadcasts
    FLAMEGPU->setVariable<float>("travel_time", travel_time);
    
    // Track interval metrics: vehicles entered
    int interval_entered = FLAMEGPU->getVariable<int>("interval_entered");
    FLAMEGPU->setVariable<int>("interval_entered", interval_entered + accepted_count);
    
    return flamegpu::ALIVE;
}
"""

PROCESS_DEPARTURES_CODE = """
FLAMEGPU_AGENT_FUNCTION(process_departures, flamegpu::MessageBucket, flamegpu::MessageNone) {
    const int edge_id = FLAMEGPU->getVariable<int>("edge_id");
    int curr_count = FLAMEGPU->getVariable<int>("curr_count");
    const float dt = FLAMEGPU->environment.getProperty<float>("time_step");
    const float current_time = FLAMEGPU->environment.getProperty<float>("current_time");
    
    // Process all departure messages for this edge
    int departed_count = 0;
    for (const auto& msg : FLAMEGPU->message_in(edge_id)) {
        const int depart_size = msg.getVariable<int>("size");
        departed_count += depart_size;
        curr_count -= depart_size;
    }
    
    // Ensure non-negative
    if (curr_count < 0) curr_count = 0;
    
    // Track interval metrics: vehicles left
    int interval_left = FLAMEGPU->getVariable<int>("interval_left");
    FLAMEGPU->setVariable<int>("interval_left", interval_left + departed_count);
    
    // Accumulate sampled seconds (vehicle-seconds on this edge this timestep)
    float interval_sampled = FLAMEGPU->getVariable<float>("interval_sampled_seconds");
    interval_sampled += (float)curr_count * dt;
    FLAMEGPU->setVariable<float>("interval_sampled_seconds", interval_sampled);
    
    FLAMEGPU->setVariable<int>("curr_count", curr_count);
    
    // Get edge parameters
    const float length = FLAMEGPU->getVariable<float>("length");
    const float free_speed = FLAMEGPU->getVariable<float>("free_speed");
    const int capacity = FLAMEGPU->getVariable<int>("capacity");
    const float jam_threshold = FLAMEGPU->getVariable<float>("jam_threshold");
    
    // Calculate occupancy and jam state (SUMO-style)
    float occupancy = (float)curr_count / (float)capacity;
    if (occupancy > 1.0f) occupancy = 1.0f;
    
    // Update jam state: jammed if occupancy > threshold
    const int is_jammed = (occupancy > jam_threshold) ? 1 : 0;
    FLAMEGPU->setVariable<int>("is_jammed", is_jammed);
    
    // Calculate mean speed based on jam state
    // In free-flow: use free_speed
    // In jam: speed decreases proportionally to congestion above threshold
    float speed;
    if (is_jammed == 0) {
        speed = free_speed;
    } else {
        // Linear decrease from free_speed at jam_threshold to min_speed at capacity
        float congestion_factor = (occupancy - jam_threshold) / (1.0f - jam_threshold);
        speed = free_speed * (1.0f - 0.7f * congestion_factor);  // Down to 30% at full
    }
    if (speed < 1.0f) speed = 1.0f;  // Minimum speed
    
    // Travel time = length / speed (base travel time)
    float travel_time = length / speed;
    FLAMEGPU->setVariable<float>("travel_time", travel_time);
    
    // Update block_time decay (allows next vehicle entry after headway)
    float block_time = FLAMEGPU->getVariable<float>("block_time");
    if (block_time < current_time) {
        block_time = current_time;  // Reset if expired
    }
    FLAMEGPU->setVariable<float>("block_time", block_time);
    
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
    const int is_jammed = FLAMEGPU->getVariable<int>("is_jammed");
    const float block_time = FLAMEGPU->getVariable<float>("block_time");
    
    // Broadcast status keyed by from_node (so packets can find alternatives from a node)
    if (from_node >= 0) {
        FLAMEGPU->message_out.setKey(from_node);
        FLAMEGPU->message_out.setVariable<int>("edge_id", edge_id);
        FLAMEGPU->message_out.setVariable<int>("to_node", to_node);
        FLAMEGPU->message_out.setVariable<int>("available_capacity", capacity - curr_count);
        FLAMEGPU->message_out.setVariable<float>("travel_time", travel_time);
        FLAMEGPU->message_out.setVariable<int>("is_jammed", is_jammed);
        FLAMEGPU->message_out.setVariable<float>("block_time", block_time);
    }
    
    return flamegpu::ALIVE;
}
"""


# =============================================================================
# Agent Configuration Dataclasses
# =============================================================================

@dataclass
class EdgeQueueConfig:
    """Configuration for EdgeQueue (Segment) agent type
    
    Note: EdgeQueue now represents a SEGMENT (~100m), not a full edge.
    The name is kept for backwards compatibility.
    """
    max_edges: int = 100000  # Now represents max segments
    
    # Variable names and types
    variables: Dict[str, str] = field(default_factory=lambda: {
        'edge_id': 'int',         # Actually segment_id (index in segment list)
        'edge_idx': 'int',        # Parent edge index (for metrics)
        'segment_idx': 'int',     # Segment index within edge (0, 1, 2...)
        'next_segment': 'int',    # Next segment in same edge (-1 if last)
        'capacity': 'int',
        'curr_count': 'int',
        'length': 'float',        # Segment length (~100m)
        'free_speed': 'float',
        'signal_id': 'int',       # -1 if not signalized (only last segment has signal)
        'is_green': 'int',        # 1 = green, 0 = red
        'travel_time': 'float',
        'out_node': 'int',        # End node (only meaningful for last segment)
        'from_node': 'int',       # Start node (only meaningful for first segment)
        'lane_count': 'int',
    })


@dataclass
class PacketConfig:
    """Configuration for Packet agent type
    
    Note: Routes now contain SEGMENT indices (not edge indices).
    Packets move segment-by-segment through the network.
    """
    max_route_length: int = 256  # Default, will be overridden dynamically based on routes
    
    # Variable names and types
    variables: Dict[str, str] = field(default_factory=lambda: {
        'size': 'int',
        'curr_edge': 'int',       # Current edge index (for metrics/reporting)
        'curr_segment': 'int',    # Current segment index
        'next_segment': 'int',    # Next segment to enter
        'remaining_time': 'float',
        'entry_time': 'float',
        'route_idx': 'int',
        'route_length': 'int',
        'is_jammed': 'int',       # Current segment jam state (for 4-tau)
    })
    
    # Array variables
    array_variables: Dict[str, tuple] = field(default_factory=lambda: {
        'route': ('int', 32),  # Fixed-size route array (contains segment indices)
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
    Define the EdgeQueue (Segment) agent in a FLAMEGPU2 model
    
    Note: EdgeQueue agents now represent SEGMENTS (~100m), not full edges.
    This matches SUMO's mesoscopic model for better traffic flow.
    
    Args:
        model: pyflamegpu.ModelDescription
        config: Optional EdgeQueueConfig
        
    Returns:
        The EdgeQueue (Segment) agent description
    """
    if config is None:
        config = EdgeQueueConfig()
    
    agent = model.newAgent("EdgeQueue")
    
    # Define variables - now represents a SEGMENT
    agent.newVariableInt("edge_id")            # Segment index (used as ID for messages)
    agent.newVariableInt("edge_idx")           # Parent edge index (for metrics)
    agent.newVariableInt("segment_idx", 0)     # Segment position within edge (0, 1, 2...)
    agent.newVariableInt("next_segment", -1)   # Next segment in same edge (-1 if last)
    agent.newVariableInt("capacity")
    agent.newVariableInt("curr_count", 0)
    agent.newVariableFloat("length")           # Segment length (~100m)
    agent.newVariableFloat("free_speed")
    agent.newVariableInt("signal_id", -1)      # Only last segment in edge has signal
    agent.newVariableInt("is_green", 1)        # Default to green
    agent.newVariableFloat("travel_time")
    agent.newVariableInt("from_node", -1)      # Only meaningful for first segment
    agent.newVariableInt("out_node")           # Only meaningful for last segment
    agent.newVariableInt("lane_count", 1)
    
    # SUMO Mesoscopic 4-tau headway parameters
    agent.newVariableFloat("tau_ff", 1.4)       # Free-flow to free-flow headway [s]
    agent.newVariableFloat("tau_fj", 1.4)       # Free-flow to jammed headway [s]
    agent.newVariableFloat("tau_jf", 2.0)       # Jammed to free-flow headway [s]
    agent.newVariableFloat("tau_jj", 2.0)       # Jammed to jammed headway [s]
    agent.newVariableFloat("jam_threshold", 0.5)  # Occupancy threshold for jammed state
    agent.newVariableFloat("block_time", 0.0)   # Time when next vehicle can enter
    agent.newVariableInt("is_jammed", 0)        # Current jam state (0=free, 1=jammed)
    
    # Interval-based metrics tracking (SUMO edgeData style)
    agent.newVariableFloat("interval_sampled_seconds", 0.0)  # Cumulative veh-seconds
    agent.newVariableInt("interval_entered", 0)              # Vehicles entered
    agent.newVariableInt("interval_left", 0)                 # Vehicles left
    
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
    
    # Reset interval counters (triggered by host function)
    reset_code = """
FLAMEGPU_AGENT_FUNCTION(reset_interval_counters, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Check if reset is requested
    const int do_reset = FLAMEGPU->environment.getProperty<int>("reset_interval_counters");
    if (do_reset == 1) {
        FLAMEGPU->setVariable<float>("interval_sampled_seconds", 0.0f);
        FLAMEGPU->setVariable<int>("interval_entered", 0);
        FLAMEGPU->setVariable<int>("interval_left", 0);
    }
    return flamegpu::ALIVE;
}
"""
    fn_reset = agent.newRTCFunction("reset_interval_counters", reset_code)
    
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
    agent.newVariableInt("curr_edge")         # Current edge index (for metrics/reporting)
    agent.newVariableInt("curr_segment")      # Current segment index (for segment mode)
    agent.newVariableInt("next_segment", -1)  # Next segment to enter
    agent.newVariableFloat("remaining_time")
    agent.newVariableFloat("entry_time", 0.0)
    agent.newVariableInt("route_idx", 0)
    agent.newVariableInt("route_length")
    agent.newVariableInt("ready_to_move", 0)  # Flag: 1 = finished segment, waiting to move
    agent.newVariableInt("departure_sent", 0)  # Flag: 1 = departure notice sent for current segment
    agent.newVariableInt("destination", -1)   # Final destination edge for rerouting
    agent.newVariableFloat("wait_time", 0.0)  # Time spent waiting for next segment (for teleport)
    agent.newVariableInt("curr_node", -1)     # Current node for rerouting
    agent.newVariableInt("dest_node", -1)     # Destination node for rerouting
    agent.newVariableInt("is_jammed", 0)      # Current segment jam state (for 4-tau)
    
    # Route array (fixed size) - contains segment indices in segment mode, edge indices otherwise
    agent.newVariableArrayInt("route", config.max_route_length)
    
    # Define states
    agent.newState("traveling")
    agent.newState("waiting")
    agent.setInitialState("traveling")
    
    # Define agent functions
    
    # 1. Send departure notice (waiting state - after move_and_request transitions packet)
    fn_depart = agent.newRTCFunction("send_departure", SEND_DEPARTURE_CODE)
    fn_depart.setInitialState("waiting")
    fn_depart.setEndState("waiting")
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

