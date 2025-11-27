"""
Message Definitions for FLAMEGPU2 Mesoscopic Traffic Simulation

This module defines the message types used for inter-agent communication:
1. entry_request - Packets requesting entry to edges (Bucket by edge_id)
2. entry_accept - Edges accepting packets (BruteForce)
3. departure_notice - Packets notifying departure from edges (Bucket by edge_id)
4. green_signal - Signals broadcasting green state (BruteForce)

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
    # 1. Entry Request Message (Bucket by edge_id)
    # =========================================================================
    # Sent by Packets to request entry to an edge
    # Keyed by target edge_id for efficient lookup
    msg_request = model.newMessageBucket("entry_request")
    msg_request.setUpperBound(config.max_edges - 1)
    msg_request.newVariableInt("size")                    # Packet size (vehicles)
    msg_request.newVariableID("agent_id")                 # Requesting packet's ID
    msg_request.newVariableInt("from_edge")               # Origin edge (for priority)
    messages["entry_request"] = msg_request
    
    # =========================================================================
    # 2. Entry Accept Message (Bucket by agent_id)
    # =========================================================================
    # Sent by EdgeQueue to accepted Packets
    # Bucket keyed by requesting packet's agent_id for O(1) lookup
    # Includes travel_time so packets don't need to look up env arrays (scales to any network)
    msg_accept = model.newMessageBucket("entry_accept")
    msg_accept.setUpperBound(2000000)  # Max concurrent packets (agent IDs can grow with large demand)
    msg_accept.newVariableInt("edge_id")                  # Edge that accepted
    msg_accept.newVariableFloat("travel_time")            # Current travel time for this edge
    msg_accept.newVariableInt("out_node")                 # Destination node of this edge (for rerouting)
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
    messages["edge_status"] = msg_status
    
    # =========================================================================
    # 5. Green Signal Message (BruteForce)
    # =========================================================================
    # Sent by SignalControllers to indicate green status for edges
    # BruteForce because signals output multiple edges
    msg_green = model.newMessageBruteForce("green_signal")
    msg_green.newVariableInt("edge_id")                   # Edge that has green
    msg_green.newVariableInt("node_id")                   # Junction node ID
    messages["green_signal"] = msg_green
    
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

