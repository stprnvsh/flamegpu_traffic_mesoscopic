"""
Queue Dynamics Models for Mesoscopic Traffic Simulation

This module implements various queue models that determine how vehicles
accumulate and discharge at bottlenecks (junctions, lane drops, etc.).

Queue modeling is critical for mesoscopic accuracy because:
1. It determines how congestion propagates upstream (spillback)
2. It affects travel time variability under congestion
3. It controls throughput at capacity-constrained locations

Mathematical Background:
------------------------
Queue theory in traffic applies to vehicle accumulation at:
- Traffic signals (deterministic service with periodic availability)
- Unsignalized junctions (gap acceptance)
- Bottlenecks (capacity reduction points)

Key relationships:
- Arrival rate λ(t): vehicles arriving at queue tail [veh/s]
- Service rate μ(t): vehicles departing queue head [veh/s]
- Queue length Q(t) = ∫[λ(τ) - μ(τ)]dτ

For stability: long-term average λ < μ

References:
-----------
[1] Newell, G.F. (1965). "Approximation methods for queues with application to
    the fixed-cycle traffic light"
[2] Akcelik, R. (1980). "Time-dependent expressions for delay, stop rate and 
    queue length at traffic signals"
[3] Daganzo, C.F. (1997). "Fundamentals of Transportation and Traffic Operations"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict
import math


class QueueState(Enum):
    """Queue operational states"""
    EMPTY = "empty"                 # No vehicles waiting
    BUILDING = "building"           # Queue growing
    STEADY = "steady"               # Arrival ≈ service
    DISCHARGING = "discharging"     # Queue shrinking
    SPILLBACK = "spillback"         # Queue reached upstream limit


@dataclass
class QueueStatus:
    """Current queue state information"""
    length_vehicles: int       # Number of vehicles in queue
    length_meters: float       # Physical queue length [m]
    wait_time: float          # Current expected wait time [s]
    state: QueueState
    service_rate: float       # Current service rate [veh/s]
    arrival_rate: float       # Current arrival rate [veh/s]
    time_to_clearance: float  # Time until queue clears (if discharging) [s]
    spillback_risk: float     # Probability/proximity to spillback (0-1)


@dataclass
class VehicleInQueue:
    """Representation of a vehicle (or packet) in queue"""
    id: int
    size: int                 # Number of vehicles if packet
    arrival_time: float       # When entered queue [s]
    priority: int = 0         # For priority-based service
    turn_movement: int = 0    # Which movement (for turn-based capacity)


class QueueModel(ABC):
    """
    Abstract base class for queue models
    
    Queue models determine:
    1. How long vehicles wait before being served
    2. What order vehicles are served in
    3. Whether spillback occurs
    """
    
    @abstractmethod
    def add_vehicle(self, vehicle: VehicleInQueue, current_time: float) -> None:
        """Add a vehicle to the queue"""
        pass
    
    @abstractmethod
    def serve_vehicles(self, available_capacity: float, 
                       current_time: float, dt: float) -> List[VehicleInQueue]:
        """
        Serve vehicles from queue given available capacity
        
        Args:
            available_capacity: How many vehicles can be served [veh]
            current_time: Current simulation time [s]
            dt: Time step duration [s]
            
        Returns:
            List of vehicles served this time step
        """
        pass
    
    @abstractmethod
    def get_status(self, current_time: float) -> QueueStatus:
        """Get current queue status"""
        pass
    
    @abstractmethod
    def get_wait_time(self, arrival_time: float, 
                      current_time: float) -> float:
        """Estimate wait time for a vehicle that just arrived"""
        pass


class PointQueueModel(QueueModel):
    """
    Point Queue (Vertical Queue) Model
    
    The simplest queue model where the queue has no physical length.
    Vehicles stack "vertically" at a point.
    
    Mathematical formulation:
    -------------------------
    Queue evolution:
        Q(t + Δt) = max(0, Q(t) + λ(t)×Δt - μ(t)×Δt)
    
    Wait time (FIFO):
        W = Q / μ  (Little's Law in steady state)
    
    For time-varying arrival (e.g., signal):
        W(t) = ∫ q(τ)dτ where q(τ) is queue at time τ
    
    Advantages:
    - Simple implementation
    - Low computational cost
    - Good for short segments
    
    Disadvantages:
    - No physical queue length
    - Cannot model spillback directly
    - Underestimates delay in congestion
    
    Accuracy rating: ★★☆☆☆ (use for simple scenarios)
    """
    
    def __init__(self, capacity: float, vehicle_length: float = 7.0):
        """
        Args:
            capacity: Maximum service rate [veh/s]
            vehicle_length: Average vehicle length + gap [m]
        """
        self.capacity = capacity
        self.vehicle_length = vehicle_length
        self.queue: List[VehicleInQueue] = []
        self.arrival_count = 0
        self.departure_count = 0
        self.total_wait_time = 0.0
        self._last_arrival_rate = 0.0
    
    def add_vehicle(self, vehicle: VehicleInQueue, current_time: float) -> None:
        """Add vehicle to end of queue (FIFO)"""
        vehicle.arrival_time = current_time
        self.queue.append(vehicle)
        self.arrival_count += vehicle.size
    
    def serve_vehicles(self, available_capacity: float,
                       current_time: float, dt: float) -> List[VehicleInQueue]:
        """Serve vehicles from front of queue"""
        served = []
        remaining_capacity = available_capacity
        
        while self.queue and remaining_capacity > 0:
            vehicle = self.queue[0]
            
            if vehicle.size <= remaining_capacity:
                # Serve entire vehicle/packet
                self.queue.pop(0)
                served.append(vehicle)
                remaining_capacity -= vehicle.size
                
                # Track statistics
                wait = current_time - vehicle.arrival_time
                self.total_wait_time += wait * vehicle.size
                self.departure_count += vehicle.size
            else:
                # Partial service (split packet)
                # In practice, we might not split - vehicle waits
                break
        
        return served
    
    def get_status(self, current_time: float) -> QueueStatus:
        """Get current queue status"""
        queue_length = sum(v.size for v in self.queue)
        queue_meters = queue_length * self.vehicle_length
        
        # Estimate wait time using Little's Law
        if self.capacity > 0:
            wait_time = queue_length / self.capacity
        else:
            wait_time = float('inf') if queue_length > 0 else 0.0
        
        # Determine state
        if queue_length == 0:
            state = QueueState.EMPTY
        elif self._last_arrival_rate > self.capacity:
            state = QueueState.BUILDING
        elif self._last_arrival_rate < self.capacity and queue_length > 0:
            state = QueueState.DISCHARGING
        else:
            state = QueueState.STEADY
        
        # Time to clearance
        if state == QueueState.DISCHARGING:
            net_rate = self.capacity - self._last_arrival_rate
            time_to_clear = queue_length / net_rate if net_rate > 0 else float('inf')
        else:
            time_to_clear = float('inf')
        
        return QueueStatus(
            length_vehicles=queue_length,
            length_meters=queue_meters,
            wait_time=wait_time,
            state=state,
            service_rate=self.capacity,
            arrival_rate=self._last_arrival_rate,
            time_to_clearance=time_to_clear,
            spillback_risk=0.0  # Point queue has no spillback
        )
    
    def get_wait_time(self, arrival_time: float, current_time: float) -> float:
        """Estimate wait time for new arrival"""
        queue_length = sum(v.size for v in self.queue)
        if self.capacity > 0:
            return queue_length / self.capacity
        return float('inf') if queue_length > 0 else 0.0
    
    def get_average_wait(self) -> float:
        """Get average wait time of all served vehicles"""
        if self.departure_count > 0:
            return self.total_wait_time / self.departure_count
        return 0.0


class SpatialQueueModel(QueueModel):
    """
    Spatial Queue Model with Physical Queue Length
    
    Models queue as having physical extent, enabling spillback detection.
    Queue grows backward from junction at rate determined by jam density.
    
    Mathematical formulation:
    -------------------------
    Queue length evolution:
        L_q(t + Δt) = L_q(t) + [λ(t) - μ(t)] × Δt / ρ_jam
    
    where ρ_jam is jam density (veh/m)
    
    Spillback condition:
        L_q(t) ≥ L_edge  (queue reaches upstream junction)
    
    Queue propagation speed:
        w_queue = -w (backward wave speed from fundamental diagram)
    
    Travel time through queue:
        T_queue = L_q / w + service_time
    
    Advantages:
    - Realistic queue lengths
    - Natural spillback modeling
    - Better delay estimates
    
    Disadvantages:
    - More complex
    - Requires edge length information
    
    Accuracy rating: ★★★★☆ (recommended for mesoscopic)
    """
    
    def __init__(self, 
                 capacity: float,
                 edge_length: float,
                 jam_density: float = 0.15,
                 wave_speed: float = 5.0,
                 vehicle_length: float = 7.0):
        """
        Args:
            capacity: Maximum service rate [veh/s]
            edge_length: Length of the edge [m]
            jam_density: Jam density [veh/m]
            wave_speed: Backward wave speed [m/s]
            vehicle_length: Average vehicle length + gap [m]
        """
        self.capacity = capacity
        self.edge_length = edge_length
        self.jam_density = jam_density
        self.wave_speed = wave_speed
        self.vehicle_length = vehicle_length
        
        self.queue: List[VehicleInQueue] = []
        self.queue_length_meters = 0.0
        self._last_arrival_rate = 0.0
        
        # Statistics
        self.arrival_count = 0
        self.departure_count = 0
        self.total_wait_time = 0.0
        self.spillback_events = 0
    
    @property
    def queue_length_vehicles(self) -> int:
        """Number of vehicles in queue"""
        return sum(v.size for v in self.queue)
    
    @property
    def is_spillback(self) -> bool:
        """Check if queue has reached upstream junction"""
        return self.queue_length_meters >= self.edge_length
    
    @property
    def available_storage(self) -> float:
        """Remaining storage capacity [vehicles]"""
        remaining_length = self.edge_length - self.queue_length_meters
        return max(0, remaining_length * self.jam_density)
    
    def add_vehicle(self, vehicle: VehicleInQueue, current_time: float) -> bool:
        """
        Add vehicle to queue if space available
        
        Returns:
            True if added, False if spillback prevents entry
        """
        # Check storage capacity
        vehicle_space = vehicle.size / self.jam_density
        
        if self.queue_length_meters + vehicle_space > self.edge_length:
            self.spillback_events += 1
            return False  # Spillback - cannot accept
        
        vehicle.arrival_time = current_time
        self.queue.append(vehicle)
        self.queue_length_meters += vehicle_space
        self.arrival_count += vehicle.size
        
        return True
    
    def serve_vehicles(self, available_capacity: float,
                       current_time: float, dt: float) -> List[VehicleInQueue]:
        """
        Serve vehicles from queue head
        
        Queue shrinks at service rate limited by wave speed
        """
        served = []
        remaining_capacity = available_capacity
        
        # Max queue reduction by wave speed
        max_length_reduction = self.wave_speed * dt
        length_reduced = 0.0
        
        while self.queue and remaining_capacity > 0:
            vehicle = self.queue[0]
            vehicle_space = vehicle.size / self.jam_density
            
            # Check if we can serve this vehicle (wave speed limit)
            if length_reduced + vehicle_space > max_length_reduction:
                break  # Wave hasn't reached this vehicle yet
            
            if vehicle.size <= remaining_capacity:
                self.queue.pop(0)
                served.append(vehicle)
                remaining_capacity -= vehicle.size
                
                # Update queue length
                self.queue_length_meters -= vehicle_space
                length_reduced += vehicle_space
                
                # Statistics
                wait = current_time - vehicle.arrival_time
                self.total_wait_time += wait * vehicle.size
                self.departure_count += vehicle.size
            else:
                break
        
        # Ensure non-negative
        self.queue_length_meters = max(0.0, self.queue_length_meters)
        
        return served
    
    def get_status(self, current_time: float) -> QueueStatus:
        """Get current queue status with spillback risk"""
        queue_veh = self.queue_length_vehicles
        
        # Wait time estimation (queue traverse + service)
        if self.capacity > 0:
            traverse_time = self.queue_length_meters / self.wave_speed
            service_time = queue_veh / self.capacity
            wait_time = traverse_time + service_time
        else:
            wait_time = float('inf') if queue_veh > 0 else 0.0
        
        # State determination
        if queue_veh == 0:
            state = QueueState.EMPTY
        elif self.is_spillback:
            state = QueueState.SPILLBACK
        elif self._last_arrival_rate > self.capacity:
            state = QueueState.BUILDING
        elif self._last_arrival_rate < self.capacity and queue_veh > 0:
            state = QueueState.DISCHARGING
        else:
            state = QueueState.STEADY
        
        # Time to clearance
        if state == QueueState.DISCHARGING:
            net_rate = self.capacity - self._last_arrival_rate
            time_to_clear = queue_veh / net_rate if net_rate > 0 else float('inf')
        else:
            time_to_clear = float('inf')
        
        # Spillback risk (proximity to edge length)
        spillback_risk = min(1.0, self.queue_length_meters / self.edge_length)
        
        return QueueStatus(
            length_vehicles=queue_veh,
            length_meters=self.queue_length_meters,
            wait_time=wait_time,
            state=state,
            service_rate=self.capacity,
            arrival_rate=self._last_arrival_rate,
            time_to_clearance=time_to_clear,
            spillback_risk=spillback_risk
        )
    
    def get_wait_time(self, arrival_time: float, current_time: float) -> float:
        """Estimate wait time including queue traversal"""
        traverse_time = self.queue_length_meters / self.wave_speed
        service_time = self.queue_length_vehicles / self.capacity if self.capacity > 0 else float('inf')
        return traverse_time + service_time


class SUMOMesoQueueModel(QueueModel):
    """
    SUMO Mesoscopic Queue Model Implementation
    
    Replicates SUMO's mesoscopic queue behavior for maximum compatibility.
    
    SUMO Meso Approach:
    -------------------
    1. Each edge has segments of fixed length (default 100m)
    2. Vehicles occupy segments, not point positions
    3. Travel time = base_time × TAU_factor
    4. Blocking occurs when downstream segment full
    
    TAU Factors:
    - TAUFF: free → free (1.4)
    - TAUFJ: free → jam (1.4)
    - TAUJF: jam → free (2.0) - recovery is slower
    - TAUJJ: jam → jam (1.4)
    
    Jam Threshold:
    - Default: occupancy > 0.8 of capacity
    
    Segment Model:
    - Each segment is a separate queue
    - Vehicles move segment-to-segment
    - Segment travel time based on TAU factors
    """
    
    # SUMO default parameters
    DEFAULT_SEGMENT_LENGTH = 100.0  # meters
    DEFAULT_TAUFF = 1.4
    DEFAULT_TAUFJ = 1.4
    DEFAULT_TAUJF = 2.0
    DEFAULT_TAUJJ = 1.4
    DEFAULT_JAM_THRESHOLD = 0.8
    
    def __init__(self,
                 capacity: float,
                 edge_length: float,
                 free_speed: float,
                 segment_length: float = DEFAULT_SEGMENT_LENGTH,
                 jam_density: float = 0.15,
                 tau_ff: float = DEFAULT_TAUFF,
                 tau_fj: float = DEFAULT_TAUFJ,
                 tau_jf: float = DEFAULT_TAUJF,
                 tau_jj: float = DEFAULT_TAUJJ,
                 jam_threshold: float = DEFAULT_JAM_THRESHOLD):
        """
        Args:
            capacity: Edge capacity [veh/s]
            edge_length: Total edge length [m]
            free_speed: Free-flow speed [m/s]
            segment_length: Length of each segment [m]
            jam_density: Jam density [veh/m]
            tau_*: SUMO TAU factors
            jam_threshold: Occupancy threshold for "jammed"
        """
        self.capacity = capacity
        self.edge_length = edge_length
        self.free_speed = free_speed
        self.segment_length = segment_length
        self.jam_density = jam_density
        
        # TAU factors
        self.tau_ff = tau_ff
        self.tau_fj = tau_fj
        self.tau_jf = tau_jf
        self.tau_jj = tau_jj
        self.jam_threshold = jam_threshold
        
        # Create segments
        self.num_segments = max(1, int(math.ceil(edge_length / segment_length)))
        self.actual_segment_length = edge_length / self.num_segments
        
        # Per-segment capacity
        self.segment_capacity = self.jam_density * self.actual_segment_length
        
        # Initialize segments (list of queues, 0 = upstream, -1 = downstream)
        self.segments: List[List[VehicleInQueue]] = [[] for _ in range(self.num_segments)]
        
        # Per-segment state
        self.segment_jammed: List[bool] = [False] * self.num_segments
        
        # Statistics
        self.arrival_count = 0
        self.departure_count = 0
        self.total_wait_time = 0.0
        self._last_arrival_rate = 0.0
    
    def _get_segment_occupancy(self, segment_idx: int) -> float:
        """Get occupancy ratio of a segment"""
        vehicle_count = sum(v.size for v in self.segments[segment_idx])
        return vehicle_count / self.segment_capacity
    
    def _update_segment_states(self) -> None:
        """Update jammed status for all segments"""
        for i in range(self.num_segments):
            occupancy = self._get_segment_occupancy(i)
            self.segment_jammed[i] = occupancy >= self.jam_threshold
    
    def _get_tau_factor(self, curr_segment: int, next_segment: int) -> float:
        """Get TAU factor for transition between segments"""
        curr_jammed = self.segment_jammed[curr_segment]
        
        # Next segment (or exit if at edge end)
        if next_segment >= self.num_segments:
            next_jammed = False  # Assume exit is free
        else:
            next_jammed = self.segment_jammed[next_segment]
        
        if not curr_jammed and not next_jammed:
            return self.tau_ff
        elif not curr_jammed and next_jammed:
            return self.tau_fj
        elif curr_jammed and not next_jammed:
            return self.tau_jf
        else:
            return self.tau_jj
    
    def get_segment_travel_time(self, segment_idx: int) -> float:
        """Calculate travel time for a segment with TAU adjustment"""
        base_time = self.actual_segment_length / self.free_speed
        tau = self._get_tau_factor(segment_idx, segment_idx + 1)
        return base_time * tau
    
    def add_vehicle(self, vehicle: VehicleInQueue, current_time: float) -> bool:
        """Add vehicle to first segment (upstream entry)"""
        vehicle.arrival_time = current_time
        
        # Check if first segment can accept
        first_segment_count = sum(v.size for v in self.segments[0])
        if first_segment_count + vehicle.size > self.segment_capacity:
            return False  # Entry blocked
        
        self.segments[0].append(vehicle)
        self.arrival_count += vehicle.size
        self._update_segment_states()
        
        return True
    
    def serve_vehicles(self, available_capacity: float,
                       current_time: float, dt: float) -> List[VehicleInQueue]:
        """
        Process all segments and serve vehicles from edge exit
        
        Vehicles move through segments based on travel times
        """
        served = []
        self._update_segment_states()
        
        # Process from downstream to upstream (so moves don't conflict)
        for seg_idx in range(self.num_segments - 1, -1, -1):
            segment = self.segments[seg_idx]
            if not segment:
                continue
            
            # Check vehicles that have completed their segment time
            remaining = []
            for vehicle in segment:
                time_in_segment = current_time - vehicle.arrival_time
                required_time = self.get_segment_travel_time(seg_idx)
                
                if time_in_segment >= required_time:
                    if seg_idx == self.num_segments - 1:
                        # Exit edge (serve)
                        if len(served) < available_capacity:
                            served.append(vehicle)
                            wait = current_time - vehicle.arrival_time
                            self.total_wait_time += wait * vehicle.size
                            self.departure_count += vehicle.size
                        else:
                            remaining.append(vehicle)  # Wait for capacity
                    else:
                        # Move to next segment if space
                        next_seg = seg_idx + 1
                        next_count = sum(v.size for v in self.segments[next_seg])
                        if next_count + vehicle.size <= self.segment_capacity:
                            vehicle.arrival_time = current_time  # Reset for next segment
                            self.segments[next_seg].append(vehicle)
                        else:
                            remaining.append(vehicle)  # Blocked
                else:
                    remaining.append(vehicle)
            
            self.segments[seg_idx] = remaining
        
        return served
    
    def get_status(self, current_time: float) -> QueueStatus:
        """Get overall edge queue status"""
        total_vehicles = sum(sum(v.size for v in seg) for seg in self.segments)
        
        # Queue length from downstream end
        queue_length_meters = 0.0
        for seg_idx in range(self.num_segments - 1, -1, -1):
            seg_count = sum(v.size for v in self.segments[seg_idx])
            if seg_count > 0:
                queue_length_meters = (self.num_segments - seg_idx) * self.actual_segment_length
                break
        
        # State based on last segment
        last_seg_occupancy = self._get_segment_occupancy(self.num_segments - 1)
        if total_vehicles == 0:
            state = QueueState.EMPTY
        elif self.segment_jammed[0]:  # Upstream segment jammed = spillback
            state = QueueState.SPILLBACK
        elif last_seg_occupancy > 0.5:
            state = QueueState.BUILDING
        else:
            state = QueueState.STEADY
        
        # Simplified wait time estimate
        if self.capacity > 0:
            wait_time = total_vehicles / self.capacity
        else:
            wait_time = float('inf') if total_vehicles > 0 else 0.0
        
        # Spillback risk
        first_seg_occupancy = self._get_segment_occupancy(0)
        spillback_risk = first_seg_occupancy
        
        return QueueStatus(
            length_vehicles=total_vehicles,
            length_meters=queue_length_meters,
            wait_time=wait_time,
            state=state,
            service_rate=self.capacity,
            arrival_rate=self._last_arrival_rate,
            time_to_clearance=float('inf'),  # Complex to estimate
            spillback_risk=spillback_risk
        )
    
    def get_wait_time(self, arrival_time: float, current_time: float) -> float:
        """Estimate wait time including all segment traversals"""
        total_time = 0.0
        for seg_idx in range(self.num_segments):
            total_time += self.get_segment_travel_time(seg_idx)
        
        # Add queueing delay at exit
        last_seg_vehicles = sum(v.size for v in self.segments[-1])
        if self.capacity > 0:
            total_time += last_seg_vehicles / self.capacity
        
        return total_time
    
    def get_edge_travel_time(self) -> float:
        """Get total edge travel time with current conditions"""
        total = 0.0
        for seg_idx in range(self.num_segments):
            total += self.get_segment_travel_time(seg_idx)
        return total


# =============================================================================
# Queue Coordination for Multi-Edge Systems
# =============================================================================

@dataclass
class SpillbackInfo:
    """Information about spillback from downstream edge"""
    is_blocked: bool
    blocking_edge_id: int
    available_capacity: float
    time_to_unblock: float


class QueueCoordinator:
    """
    Coordinates queues across multiple edges for spillback propagation
    
    This is essential for accurate mesoscopic simulation - queues don't
    exist in isolation but affect each other.
    """
    
    def __init__(self):
        self.edge_queues: Dict[int, QueueModel] = {}
        self.downstream_edges: Dict[int, List[int]] = {}  # edge_id -> [downstream edge_ids]
        self.upstream_edges: Dict[int, List[int]] = {}    # edge_id -> [upstream edge_ids]
    
    def register_edge(self, edge_id: int, queue: QueueModel,
                      downstream: List[int], upstream: List[int]) -> None:
        """Register an edge and its connections"""
        self.edge_queues[edge_id] = queue
        self.downstream_edges[edge_id] = downstream
        self.upstream_edges[edge_id] = upstream
    
    def check_spillback(self, edge_id: int) -> SpillbackInfo:
        """
        Check if downstream edges are causing spillback
        
        Returns spillback info for capacity adjustment
        """
        downstream = self.downstream_edges.get(edge_id, [])
        
        for ds_id in downstream:
            ds_queue = self.edge_queues.get(ds_id)
            if ds_queue:
                status = ds_queue.get_status(0)  # current_time not needed for check
                
                if status.state == QueueState.SPILLBACK:
                    return SpillbackInfo(
                        is_blocked=True,
                        blocking_edge_id=ds_id,
                        available_capacity=0.0,
                        time_to_unblock=status.time_to_clearance
                    )
                
                # Check if nearly full (spillback imminent)
                if status.spillback_risk > 0.95:
                    return SpillbackInfo(
                        is_blocked=True,
                        blocking_edge_id=ds_id,
                        available_capacity=0.0,
                        time_to_unblock=float('inf')
                    )
        
        return SpillbackInfo(
            is_blocked=False,
            blocking_edge_id=-1,
            available_capacity=float('inf'),
            time_to_unblock=0.0
        )
    
    def propagate_spillback(self, edge_id: int, current_time: float) -> None:
        """
        Propagate spillback effects upstream
        
        When an edge is blocked, upstream edges should reduce their
        effective capacity (or block entirely).
        """
        spillback = self.check_spillback(edge_id)
        
        if spillback.is_blocked:
            # Mark this edge as affected
            queue = self.edge_queues.get(edge_id)
            if queue:
                # Reduce effective service rate
                # Implementation depends on queue type
                pass
            
            # Recursively propagate upstream
            for us_id in self.upstream_edges.get(edge_id, []):
                self.propagate_spillback(us_id, current_time)


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_queue_delay_webster(arrival_rate: float, 
                                   capacity: float,
                                   cycle_time: float,
                                   green_ratio: float) -> float:
    """
    Webster's delay formula for signalized intersections
    
    d = (C(1-λ)²) / (2(1-λx)) + x² / (2q(1-x))
    
    where:
        C = cycle time [s]
        λ = green ratio (g/C)
        x = degree of saturation (q/c)
        q = arrival rate [veh/s]
    
    Returns average delay per vehicle [s]
    """
    if capacity <= 0:
        return float('inf')
    
    x = arrival_rate / capacity  # Degree of saturation
    
    if x >= 1.0:
        return float('inf')  # Oversaturated
    
    # First term: uniform delay
    term1 = (cycle_time * (1 - green_ratio) ** 2) / (2 * (1 - green_ratio * x))
    
    # Second term: random delay
    term2 = (x ** 2) / (2 * arrival_rate * (1 - x)) if arrival_rate > 0 else 0
    
    return term1 + term2


def calculate_queue_length_akcelik(arrival_rate: float,
                                    capacity: float,
                                    analysis_period: float) -> float:
    """
    Akcelik's overflow queue formula
    
    Estimates queue length considering time-varying demand
    
    Returns expected queue length [vehicles]
    """
    if capacity <= 0:
        return arrival_rate * analysis_period
    
    x = arrival_rate / capacity
    
    if x <= 1.0:
        # Undersaturated: random queue only
        return x ** 2 / (2 * (1 - x)) if x < 1 else float('inf')
    else:
        # Oversaturated: deterministic growth
        overflow = (arrival_rate - capacity) * analysis_period
        return overflow + capacity / 2  # Overflow + average cycling queue

