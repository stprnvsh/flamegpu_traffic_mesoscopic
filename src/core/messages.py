"""
Message Definitions for FLAMEGPU2 Mesoscopic Traffic Simulation

This module defines the message types used for inter-agent communication:
1. entry_request - Packets requesting entry to edges (Bucket by edge_id)
2. entry_accept - Edges accepting packets (BruteForce)
3. departure_notice - Packets notifying departure from edges (Bucket by edge_id)
4. green_signal - Signals broadcasting green state (Bucket by controlled id)

Message Types in FLAMEGPU2:
- MessageBruteForce: All agents see all messages (O(n×m))
- MessageBucket: Messages grouped by key (O(n×m/k) average)
- MessageSpatial: Messages filtered by spatial proximity

Reference: FLAMEGPU2 Message Documentation
https://docs.flamegpu.com/guide/defining-messages/
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MessageConfig:
    """Configuration for message types"""
    max_edges: int = 100000  # Upper bound for bucket message keys (matches EnvironmentConfig)


def define_messages(model, config: Optional[MessageConfig] = None):
    """
    Define all message types for the mesoscopic traffic model
    
    Args:
        model: pyflamegpu.ModelDescription
        config: Optional MessageConfig
        
    Returns:
        Dict of message descriptions
    """
    if config is None:
        config = MessageConfig()
    
    messages = {}
    
    # =========================================================================
    # 1. Entry Request Message (Bucket by segment_id)
    # =========================================================================
    # Sent by Packets to request entry to a SEGMENT
    # Keyed by target segment_id for efficient lookup
    msg_request = model.newMessageBucket("entry_request")
    msg_request.setUpperBound(config.max_edges - 1)       # Now max segments
    msg_request.newVariableInt("size")                    # Packet size (vehicles)
    msg_request.newVariableID("agent_id")                 # Requesting packet's ID
    msg_request.newVariableInt("from_edge")               # Origin segment (for headway calc)
    msg_request.newVariableInt("is_jammed")               # Origin segment jam state (for 4-tau)
    msg_request.newVariableInt("from_node")               # Origin node for movement legality checks
    msg_request.newVariableFloat("next_action_time")      # Due-time for event semantics
    msg_request.newVariableInt("action_type")             # 0=none,1=request,2=move
    msg_request.newVariableInt("event_seq")               # Monotonic event sequence per packet
    messages["entry_request"] = msg_request
    
    # =========================================================================
    # 2. Entry Accept Message (Bucket by agent_id)
    # =========================================================================
    # Sent by Segment to accepted Packets
    # Bucket keyed by requesting packet's agent_id for O(1) lookup
    # Includes segment info so packets know where to go next
    msg_accept = model.newMessageBucket("entry_accept")
    msg_accept.setUpperBound(2000000)  # Max concurrent packets
    msg_accept.newVariableInt("admit_status")              # 1=accept, 0=defer, -1=reject
    msg_accept.newVariableFloat("retry_time")              # Earliest retry time for deferred requests
    msg_accept.newVariableInt("reason_code")               # Optional reason code
    msg_accept.newVariableInt("edge_id")                  # Segment ID (index in segment list)
    msg_accept.newVariableFloat("travel_time")            # Travel time for this segment
    msg_accept.newVariableInt("out_node")                 # End node (for rerouting)
    msg_accept.newVariableInt("next_segment")             # Next segment in same edge (-1 if last)
    msg_accept.newVariableInt("edge_idx")                 # Parent edge index (for metrics)
    messages["entry_accept"] = msg_accept
    
    # =========================================================================
    # 3. Departure Notice Message (Bucket by edge_id)
    # =========================================================================
    # Sent by Packets to notify departure from current edge
    # Keyed by edge_id so edges only process their departures
    msg_depart = model.newMessageBucket("departure_notice")
    msg_depart.setUpperBound(config.max_edges - 1)
    msg_depart.newVariableInt("size")                     # Departing packet size
    messages["departure_notice"] = msg_depart
    
    # =========================================================================
    # 4. Edge Status Message (Bucket by from_node) - For GPU rerouting
    # =========================================================================
    # Sent by EdgeQueue to broadcast their topology and congestion status
    # Packets use this to find alternative routes when stuck
    msg_status = model.newMessageBucket("edge_status")
    msg_status.setUpperBound(50000)  # Max nodes
    msg_status.newVariableInt("edge_id")                   # This edge's ID
    msg_status.newVariableInt("to_node")                   # Where this edge leads
    msg_status.newVariableInt("available_capacity")        # How much space left
    msg_status.newVariableFloat("travel_time")             # Current travel time
    msg_status.newVariableInt("is_jammed")                 # Jam state (0/1) for routing
    msg_status.newVariableFloat("block_time")              # When edge can accept next vehicle
    messages["edge_status"] = msg_status
    
    # =========================================================================
    # 5. Green Signal Message (Bucket by controlled segment/edge id)
    # =========================================================================
    # Sent by SignalControllers to indicate green status for controlled ids
    msg_green = model.newMessageBucket("green_signal")
    msg_green.setUpperBound(config.max_edges - 1)
    msg_green.newVariableInt("edge_id")                   # Edge that has green
    msg_green.newVariableInt("node_id")                   # Junction node ID
    messages["green_signal"] = msg_green

    # =========================================================================
    # 6. Movement Request/Resolution (Conflict Arbitration)
    # =========================================================================
    msg_mreq = model.newMessageBucket("movement_request")
    msg_mreq.setUpperBound(50000)  # conflict_group/node keyed
    msg_mreq.newVariableID("agent_id")
    msg_mreq.newVariableInt("target_segment")
    msg_mreq.newVariableInt("priority_rank")
    msg_mreq.newVariableFloat("request_time")
    msg_mreq.newVariableInt("event_seq")
    messages["movement_request"] = msg_mreq

    msg_mres = model.newMessageBucket("movement_resolution")
    msg_mres.setUpperBound(2000000)  # keyed by agent id
    msg_mres.newVariableInt("permit")            # 1=permit,0=defer,-1=reject
    msg_mres.newVariableFloat("defer_until")
    msg_mres.newVariableInt("reason_code")
    msg_mres.newVariableInt("target_segment")
    messages["movement_resolution"] = msg_mres
    
    return messages


def get_message_statistics(simulation) -> dict:
    """
    Get statistics about message counts in the simulation
    
    Args:
        simulation: Running CUDASimulation
        
    Returns:
        Dict with message counts
    """
    stats = {}
    
    # In FLAMEGPU2, message counts can be queried from the simulation
    # This is a placeholder - actual implementation depends on API
    
    return stats

