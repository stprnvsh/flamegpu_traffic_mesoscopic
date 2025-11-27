"""
Junction Models for Mesoscopic Traffic Simulation

This module implements capacity and delay models for various junction types:
- Signalized intersections
- Unsignalized intersections (priority, stop, yield)
- Roundabouts
- Merge/diverge sections

Junction modeling is critical because junctions are typically the capacity
bottlenecks in urban networks.

Mathematical Background:
------------------------
Junction capacity depends on:
1. Saturation flow (s): maximum discharge rate during green/opportunity
2. Lost time: time lost to starting/stopping
3. Conflict resolution: how competing movements share capacity

Key relationships:
- Capacity = Saturation_flow × Effective_green / Cycle_time  (signalized)
- Capacity = f(conflicting_flow, critical_gap, follow_up_time)  (unsignalized)

References:
-----------
[1] Highway Capacity Manual (HCM) 2010 & 2016
[2] Webster, F.V. (1958). "Traffic Signal Settings"
[3] Akcelik, R. (1981). "Traffic Signals: Capacity and Timing Analysis"
[4] Troutbeck, R.J. (1986). "Average Delay at an Unsignalized Intersection"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set
import math


class MovementType(Enum):
    """Standard turn movement types"""
    LEFT = "left"
    THROUGH = "through"
    RIGHT = "right"
    U_TURN = "u_turn"


class SignalState(Enum):
    """Traffic signal states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    FLASHING_YELLOW = "flashing_yellow"
    FLASHING_RED = "flashing_red"


class PriorityType(Enum):
    """Priority at unsignalized junctions"""
    MAJOR = "major"         # Has right-of-way
    MINOR = "minor"         # Must yield
    EQUAL = "equal"         # Equal priority (e.g., uncontrolled)


@dataclass
class TurnMovement:
    """
    Represents a turn movement at a junction
    
    A movement is defined by its origin edge, destination edge, and turn type.
    """
    from_edge: int
    to_edge: int
    movement_type: MovementType
    num_lanes: int = 1
    
    # Capacity parameters
    saturation_flow: float = 1800.0     # veh/hr/lane (base)
    effective_green: float = 0.0        # seconds (for signalized)
    capacity: float = 0.0               # veh/hr (calculated)
    
    # Signal timing (for signalized)
    phase_index: int = 0
    signal_state: SignalState = SignalState.RED
    
    # Priority (for unsignalized)
    priority: PriorityType = PriorityType.MAJOR
    critical_gap: float = 4.5           # seconds
    follow_up_time: float = 2.5         # seconds
    
    # Current state
    queue_length: int = 0
    is_blocked: bool = False
    
    def __post_init__(self):
        """Calculate capacity if not set"""
        if self.capacity <= 0 and self.effective_green > 0:
            # Will be calculated by junction model
            pass


@dataclass
class ConflictPair:
    """
    Defines conflicting movements at a junction
    
    Two movements conflict if they cannot both proceed simultaneously.
    """
    movement_1: TurnMovement
    movement_2: TurnMovement
    conflict_type: str  # 'crossing', 'merging', 'diverging'
    conflict_severity: float = 1.0  # 1.0 = full conflict, 0.5 = partial


@dataclass
class SignalPhase:
    """
    A signal phase defines which movements have green simultaneously
    """
    phase_id: int
    duration: float                     # seconds
    green_movements: List[int]          # indices of movements with green
    yellow_time: float = 3.0           # seconds
    all_red_time: float = 2.0          # seconds
    min_green: float = 5.0             # minimum green time
    max_green: float = 60.0            # maximum green time
    
    @property
    def total_time(self) -> float:
        """Total phase time including clearance"""
        return self.duration + self.yellow_time + self.all_red_time
    
    @property
    def effective_green(self) -> float:
        """Effective green time (adjusted for startup/clearance losses)"""
        startup_loss = 2.0  # seconds
        return max(0, self.duration - startup_loss + self.yellow_time * 0.5)


@dataclass
class SignalTiming:
    """
    Complete signal timing plan for an intersection
    """
    phases: List[SignalPhase]
    cycle_length: float = 0.0          # Will be calculated
    offset: float = 0.0                 # For coordination [seconds from reference]
    
    def __post_init__(self):
        """Calculate cycle length"""
        if self.cycle_length <= 0:
            self.cycle_length = sum(p.total_time for p in self.phases)
    
    def get_phase_at_time(self, time: float) -> Tuple[SignalPhase, float]:
        """
        Get active phase at a given time
        
        Returns:
            (active_phase, time_into_phase)
        """
        # Adjust for offset
        adjusted_time = (time - self.offset) % self.cycle_length
        
        cumulative = 0.0
        for phase in self.phases:
            if cumulative + phase.total_time > adjusted_time:
                time_into_phase = adjusted_time - cumulative
                return phase, time_into_phase
            cumulative += phase.total_time
        
        # Should not reach here if phases cover full cycle
        return self.phases[0], 0.0
    
    def get_green_ratio(self, movement_index: int) -> float:
        """Calculate green ratio (g/C) for a movement"""
        total_green = 0.0
        for phase in self.phases:
            if movement_index in phase.green_movements:
                total_green += phase.effective_green
        return total_green / self.cycle_length if self.cycle_length > 0 else 0


@dataclass
class JunctionCapacity:
    """Result of junction capacity analysis"""
    movement_capacities: Dict[int, float]  # movement_index -> capacity [veh/hr]
    total_capacity: float                   # sum of all capacities
    critical_v_c_ratio: float              # highest volume/capacity ratio
    level_of_service: str                   # A-F rating
    average_delay: float                    # seconds/vehicle


class JunctionModel(ABC):
    """
    Abstract base class for junction models
    
    A junction model calculates:
    1. Capacity for each movement
    2. Delay for vehicles
    3. Queue dynamics at approaches
    """
    
    def __init__(self, junction_id: int, movements: List[TurnMovement]):
        self.junction_id = junction_id
        self.movements = movements
        self.conflicts: List[ConflictPair] = []
        self._build_conflict_matrix()
    
    @abstractmethod
    def _build_conflict_matrix(self) -> None:
        """Identify conflicting movements"""
        pass
    
    @abstractmethod
    def calculate_capacity(self) -> JunctionCapacity:
        """Calculate capacity for all movements"""
        pass
    
    @abstractmethod
    def calculate_delay(self, volumes: Dict[int, float]) -> Dict[int, float]:
        """
        Calculate delay for each movement given demand volumes
        
        Args:
            volumes: movement_index -> demand volume [veh/hr]
            
        Returns:
            movement_index -> average delay [seconds]
        """
        pass
    
    @abstractmethod
    def can_proceed(self, movement_index: int, current_time: float) -> bool:
        """Check if a vehicle on given movement can proceed"""
        pass
    
    def get_movement_by_edges(self, from_edge: int, to_edge: int) -> Optional[TurnMovement]:
        """Find movement by origin and destination edges"""
        for m in self.movements:
            if m.from_edge == from_edge and m.to_edge == to_edge:
                return m
        return None
    
    def conflicts_with(self, movement1_idx: int, movement2_idx: int) -> bool:
        """Check if two movements conflict"""
        for conflict in self.conflicts:
            m1, m2 = conflict.movement_1, conflict.movement_2
            if (self.movements.index(m1) == movement1_idx and 
                self.movements.index(m2) == movement2_idx):
                return True
            if (self.movements.index(m1) == movement2_idx and 
                self.movements.index(m2) == movement1_idx):
                return True
        return False


class SignalizedJunction(JunctionModel):
    """
    Signalized Intersection Model
    
    Implements HCM methodology for signalized intersection capacity and delay.
    
    Mathematical formulation:
    -------------------------
    
    Capacity per lane group:
        c = s × (g/C)
    
    where:
        s = saturation flow rate [veh/hr/lane]
        g = effective green time [s]
        C = cycle length [s]
    
    Saturation flow adjustment:
        s = s_0 × f_w × f_HV × f_g × f_p × f_bb × f_a × f_LU × f_LT × f_RT × f_Lpb × f_Rpb
    
    where s_0 = 1900 veh/hr/lane (ideal conditions) and f_* are adjustment factors
    
    Control delay (HCM uniform + incremental):
        d = d_1 × PF + d_2 + d_3
    
    Uniform delay:
        d_1 = 0.5C(1-g/C)² / (1 - min(1,X)×g/C)
    
    Incremental delay:
        d_2 = 900T[(X-1) + √((X-1)² + 8kIX/(cT))]
    
    where:
        X = v/c (volume-to-capacity ratio)
        T = analysis period [hr]
        k = incremental delay factor
        I = upstream filtering factor
    
    Level of Service:
        A: d ≤ 10s
        B: 10 < d ≤ 20s
        C: 20 < d ≤ 35s
        D: 35 < d ≤ 55s
        E: 55 < d ≤ 80s
        F: d > 80s
    """
    
    # Base saturation flow (ideal conditions)
    BASE_SATURATION_FLOW = 1900.0  # veh/hr/lane
    
    # LOS delay thresholds
    LOS_THRESHOLDS = {
        'A': 10, 'B': 20, 'C': 35, 'D': 55, 'E': 80
    }
    
    def __init__(self, junction_id: int, movements: List[TurnMovement],
                 signal_timing: SignalTiming):
        self.signal_timing = signal_timing
        super().__init__(junction_id, movements)
        self._assign_phases_to_movements()
    
    def _build_conflict_matrix(self) -> None:
        """
        Build conflict matrix for signal phase validation
        
        Conflicting movements should not be in the same phase
        """
        self.conflicts = []
        
        for i, m1 in enumerate(self.movements):
            for j, m2 in enumerate(self.movements[i+1:], i+1):
                if self._movements_conflict(m1, m2):
                    self.conflicts.append(ConflictPair(
                        movement_1=m1,
                        movement_2=m2,
                        conflict_type=self._determine_conflict_type(m1, m2)
                    ))
    
    def _movements_conflict(self, m1: TurnMovement, m2: TurnMovement) -> bool:
        """
        Determine if two movements conflict
        
        Movements conflict if they cross paths or merge into same lane
        """
        # Same origin - no conflict (they're sequential on same approach)
        if m1.from_edge == m2.from_edge:
            return False
        
        # Same destination - potential merge conflict
        if m1.to_edge == m2.to_edge:
            return True
        
        # Cross-traffic conflicts (simplified - would need geometry for exact)
        # Left turns typically conflict with opposing through movements
        if m1.movement_type == MovementType.LEFT and m2.movement_type == MovementType.THROUGH:
            return True
        if m2.movement_type == MovementType.LEFT and m1.movement_type == MovementType.THROUGH:
            return True
        
        return False
    
    def _determine_conflict_type(self, m1: TurnMovement, m2: TurnMovement) -> str:
        """Determine type of conflict between movements"""
        if m1.to_edge == m2.to_edge:
            return 'merging'
        return 'crossing'
    
    def _assign_phases_to_movements(self) -> None:
        """Assign phase information to movements based on signal timing"""
        for phase in self.signal_timing.phases:
            for move_idx in phase.green_movements:
                if move_idx < len(self.movements):
                    self.movements[move_idx].phase_index = phase.phase_id
                    self.movements[move_idx].effective_green = phase.effective_green
    
    def calculate_saturation_flow(self, movement: TurnMovement,
                                   heavy_vehicle_pct: float = 0.02,
                                   grade_pct: float = 0.0,
                                   parking_activity: bool = False,
                                   bus_blockage: float = 0.0,
                                   area_type: str = 'urban') -> float:
        """
        Calculate adjusted saturation flow rate (HCM methodology)
        
        Args:
            movement: The turn movement
            heavy_vehicle_pct: Fraction of heavy vehicles (0-1)
            grade_pct: Approach grade (%, positive = uphill)
            parking_activity: Whether parking maneuvers occur
            bus_blockage: Bus stops per hour
            area_type: 'urban' or 'cbd'
            
        Returns:
            Adjusted saturation flow [veh/hr/lane]
        """
        s0 = self.BASE_SATURATION_FLOW
        
        # Heavy vehicle factor
        E_T = 2.0  # PCE for trucks
        f_HV = 100 / (100 + heavy_vehicle_pct * 100 * (E_T - 1))
        
        # Grade factor
        f_g = 1 - grade_pct / 200
        
        # Parking factor (simplified)
        f_p = 0.9 if parking_activity else 1.0
        
        # Bus blockage factor
        f_bb = max(0.5, 1 - bus_blockage * 14.4 / 3600 / movement.num_lanes)
        
        # Area type factor
        f_a = 0.9 if area_type == 'cbd' else 1.0
        
        # Turn type factors
        if movement.movement_type == MovementType.LEFT:
            f_turn = 0.95  # Protected left
        elif movement.movement_type == MovementType.RIGHT:
            f_turn = 0.85  # Right turn (pedestrian conflicts)
        else:
            f_turn = 1.0
        
        # Combined adjustment
        s = s0 * f_HV * f_g * f_p * f_bb * f_a * f_turn
        
        return s
    
    def calculate_capacity(self) -> JunctionCapacity:
        """
        Calculate capacity for all movements using HCM methodology
        """
        capacities = {}
        total = 0.0
        max_vc = 0.0
        
        for idx, movement in enumerate(self.movements):
            # Get saturation flow
            s = self.calculate_saturation_flow(movement)
            
            # Get green ratio
            g_C = self.signal_timing.get_green_ratio(idx)
            
            # Capacity = s × g/C × num_lanes
            c = s * g_C * movement.num_lanes
            capacities[idx] = c
            total += c
            
            # Update movement
            movement.capacity = c
            movement.saturation_flow = s
        
        # Determine LOS based on critical movement
        avg_delay = self._calculate_intersection_delay(capacities)
        los = self._determine_los(avg_delay)
        
        return JunctionCapacity(
            movement_capacities=capacities,
            total_capacity=total,
            critical_v_c_ratio=max_vc,
            level_of_service=los,
            average_delay=avg_delay
        )
    
    def calculate_delay(self, volumes: Dict[int, float]) -> Dict[int, float]:
        """
        Calculate delay for each movement (HCM control delay)
        
        Args:
            volumes: movement_index -> volume [veh/hr]
            
        Returns:
            movement_index -> delay [seconds]
        """
        delays = {}
        C = self.signal_timing.cycle_length
        T = 0.25  # Analysis period [hr]
        k = 0.5   # Incremental delay factor (pretimed)
        I = 1.0   # Upstream filtering factor
        
        for idx, movement in enumerate(self.movements):
            v = volumes.get(idx, 0)
            c = movement.capacity
            g_C = self.signal_timing.get_green_ratio(idx)
            
            if c <= 0:
                delays[idx] = float('inf') if v > 0 else 0
                continue
            
            X = v / c  # Volume-to-capacity ratio
            
            # Uniform delay (d1)
            numerator = 0.5 * C * (1 - g_C) ** 2
            denominator = 1 - min(1.0, X) * g_C
            d1 = numerator / max(0.01, denominator)
            
            # Progression factor (assume random arrivals)
            PF = 1.0
            
            # Incremental delay (d2)
            if X > 0:
                term1 = X - 1
                term2 = math.sqrt((X - 1) ** 2 + 8 * k * I * X / (c * T))
                d2 = 900 * T * (term1 + term2)
            else:
                d2 = 0
            
            # Initial queue delay (d3) - assume zero
            d3 = 0
            
            # Total delay
            delays[idx] = d1 * PF + d2 + d3
        
        return delays
    
    def _calculate_intersection_delay(self, capacities: Dict[int, float]) -> float:
        """Calculate intersection-wide average delay (weighted by assumed equal volumes)"""
        if not capacities:
            return 0
        
        # Assume each movement has moderate volume for baseline
        volumes = {idx: c * 0.7 for idx, c in capacities.items()}  # 70% v/c
        delays = self.calculate_delay(volumes)
        
        if delays:
            return sum(delays.values()) / len(delays)
        return 0
    
    def _determine_los(self, delay: float) -> str:
        """Determine Level of Service from delay"""
        for los, threshold in self.LOS_THRESHOLDS.items():
            if delay <= threshold:
                return los
        return 'F'
    
    def can_proceed(self, movement_index: int, current_time: float) -> bool:
        """Check if movement has green at current time"""
        if movement_index >= len(self.movements):
            return False
        
        phase, time_into_phase = self.signal_timing.get_phase_at_time(current_time)
        
        # Check if this movement is green in current phase
        if movement_index in phase.green_movements:
            # Check if within green portion (not yellow/all-red)
            if time_into_phase < phase.duration:
                return True
        
        return False
    
    def get_time_to_green(self, movement_index: int, current_time: float) -> float:
        """Calculate time until movement gets green"""
        if self.can_proceed(movement_index, current_time):
            return 0.0
        
        C = self.signal_timing.cycle_length
        current_pos = (current_time - self.signal_timing.offset) % C
        
        # Find next green for this movement
        cumulative = 0.0
        for _ in range(2):  # Check up to 2 cycles
            for phase in self.signal_timing.phases:
                if movement_index in phase.green_movements:
                    if cumulative >= current_pos:
                        return cumulative - current_pos
                cumulative += phase.total_time
        
        return C  # Should find it within one cycle


class UnsignalizedJunction(JunctionModel):
    """
    Unsignalized Intersection Model (Two-Way Stop, All-Way Stop, Yield)
    
    Implements HCM methodology for unsignalized intersection capacity.
    
    Mathematical formulation:
    -------------------------
    
    Gap acceptance capacity (HCM 2010):
        c_x = v_c × e^(-v_c × t_c/3600) / (1 - e^(-v_c × t_f/3600))
    
    where:
        v_c = conflicting flow rate [veh/hr]
        t_c = critical gap [s]
        t_f = follow-up time [s]
    
    For multiple conflicting streams, use impedance factors:
        c_m = c_x × ∏(p_0,i) for all higher-priority movements i
    
    where p_0,i = 1 - v_i/c_i (probability stream i has no queue)
    
    Critical gaps (HCM base values):
    - Left turn from major: 4.1s
    - Right turn from minor: 6.2s
    - Through from minor: 6.5s
    - Left turn from minor: 7.1s
    
    Follow-up times:
    - All movements: typically 2.2-3.5s
    """
    
    # Base critical gaps and follow-up times (HCM 2010, 2-lane roads)
    CRITICAL_GAPS = {
        (PriorityType.MAJOR, MovementType.LEFT): 4.1,
        (PriorityType.MINOR, MovementType.RIGHT): 6.2,
        (PriorityType.MINOR, MovementType.THROUGH): 6.5,
        (PriorityType.MINOR, MovementType.LEFT): 7.1,
    }
    
    FOLLOW_UP_TIMES = {
        (PriorityType.MAJOR, MovementType.LEFT): 2.2,
        (PriorityType.MINOR, MovementType.RIGHT): 3.3,
        (PriorityType.MINOR, MovementType.THROUGH): 4.0,
        (PriorityType.MINOR, MovementType.LEFT): 3.5,
    }
    
    def __init__(self, junction_id: int, movements: List[TurnMovement],
                 control_type: str = 'twsc'):  # 'twsc', 'awsc', 'yield'
        self.control_type = control_type
        super().__init__(junction_id, movements)
        self._assign_critical_gaps()
    
    def _build_conflict_matrix(self) -> None:
        """Build conflict relationships based on priority"""
        self.conflicts = []
        
        # Minor movements conflict with major movements
        for i, m1 in enumerate(self.movements):
            for j, m2 in enumerate(self.movements[i+1:], i+1):
                if self._movements_conflict_unsignalized(m1, m2):
                    self.conflicts.append(ConflictPair(
                        movement_1=m1,
                        movement_2=m2,
                        conflict_type='gap_acceptance',
                        conflict_severity=1.0
                    ))
    
    def _movements_conflict_unsignalized(self, m1: TurnMovement, 
                                          m2: TurnMovement) -> bool:
        """Check for conflicts at unsignalized junction"""
        # Minor yields to major
        if m1.priority == PriorityType.MINOR and m2.priority == PriorityType.MAJOR:
            return True
        if m2.priority == PriorityType.MINOR and m1.priority == PriorityType.MAJOR:
            return True
        
        # Left turns yield to opposing through
        if (m1.movement_type == MovementType.LEFT and 
            m2.movement_type == MovementType.THROUGH and
            m1.from_edge != m2.from_edge):
            return True
        
        return False
    
    def _assign_critical_gaps(self) -> None:
        """Assign appropriate critical gaps and follow-up times"""
        for movement in self.movements:
            key = (movement.priority, movement.movement_type)
            
            # Get values from tables, with defaults
            movement.critical_gap = self.CRITICAL_GAPS.get(key, 5.0)
            movement.follow_up_time = self.FOLLOW_UP_TIMES.get(key, 3.0)
    
    def calculate_potential_capacity(self, movement: TurnMovement,
                                      conflicting_flow: float) -> float:
        """
        Calculate potential capacity using gap acceptance formula
        
        c = v_c × e^(-v_c × t_c/3600) / (1 - e^(-v_c × t_f/3600))
        
        Args:
            movement: The turn movement
            conflicting_flow: Total conflicting flow [veh/hr]
            
        Returns:
            Potential capacity [veh/hr]
        """
        v_c = conflicting_flow
        t_c = movement.critical_gap
        t_f = movement.follow_up_time
        
        if v_c <= 0:
            # No conflicting flow - capacity is very high
            return 3600 / t_f  # Limited by follow-up time
        
        # Gap acceptance formula
        exp_tc = math.exp(-v_c * t_c / 3600)
        exp_tf = math.exp(-v_c * t_f / 3600)
        
        if exp_tf >= 1:
            return 0  # No capacity
        
        c = v_c * exp_tc / (1 - exp_tf)
        
        return max(0, c)
    
    def calculate_movement_capacity(self, movement: TurnMovement,
                                     conflicting_flow: float,
                                     impedance_factor: float = 1.0) -> float:
        """
        Calculate actual movement capacity with impedance
        
        c_m = c_p × f_imp
        
        where f_imp accounts for queueing of higher-priority movements
        """
        c_p = self.calculate_potential_capacity(movement, conflicting_flow)
        return c_p * impedance_factor * movement.num_lanes
    
    def calculate_capacity(self, 
                           conflicting_flows: Optional[Dict[int, float]] = None) -> JunctionCapacity:
        """
        Calculate capacity for all movements
        
        Args:
            conflicting_flows: movement_index -> conflicting flow [veh/hr]
                              If None, assumes moderate conflicting flows
        """
        if conflicting_flows is None:
            # Default: assume 300 veh/hr conflicting flow for minor movements
            conflicting_flows = {
                i: 300 if m.priority == PriorityType.MINOR else 0
                for i, m in enumerate(self.movements)
            }
        
        capacities = {}
        total = 0.0
        
        # Calculate in priority order
        major_movements = [i for i, m in enumerate(self.movements) 
                          if m.priority == PriorityType.MAJOR]
        minor_movements = [i for i, m in enumerate(self.movements)
                          if m.priority == PriorityType.MINOR]
        
        # Major movements: no impedance
        for idx in major_movements:
            movement = self.movements[idx]
            c_flow = conflicting_flows.get(idx, 0)
            c = self.calculate_movement_capacity(movement, c_flow, 1.0)
            capacities[idx] = c
            total += c
            movement.capacity = c
        
        # Minor movements: with impedance from queued majors
        for idx in minor_movements:
            movement = self.movements[idx]
            c_flow = conflicting_flows.get(idx, 300)
            
            # Calculate impedance (simplified)
            impedance = 0.9  # Assume some impedance from major queue
            
            c = self.calculate_movement_capacity(movement, c_flow, impedance)
            capacities[idx] = c
            total += c
            movement.capacity = c
        
        # Average delay estimation
        avg_delay = self._estimate_average_delay(capacities, conflicting_flows)
        los = self._determine_los(avg_delay)
        
        return JunctionCapacity(
            movement_capacities=capacities,
            total_capacity=total,
            critical_v_c_ratio=0,  # Would need volumes
            level_of_service=los,
            average_delay=avg_delay
        )
    
    def calculate_delay(self, volumes: Dict[int, float],
                        conflicting_flows: Optional[Dict[int, float]] = None) -> Dict[int, float]:
        """
        Calculate control delay for each movement (HCM)
        
        d = 3600/c + 900T[(v/c - 1) + √((v/c-1)² + (3600/c)(v/c)/(450T))]
        """
        if conflicting_flows is None:
            conflicting_flows = {}
        
        capacity_result = self.calculate_capacity(conflicting_flows)
        delays = {}
        T = 0.25  # Analysis period
        
        for idx, movement in enumerate(self.movements):
            v = volumes.get(idx, 0)
            c = capacity_result.movement_capacities.get(idx, 1)
            
            if c <= 0:
                delays[idx] = float('inf') if v > 0 else 0
                continue
            
            x = v / c  # Volume-to-capacity
            
            # HCM delay formula for unsignalized
            term1 = 3600 / c
            term2 = x - 1
            term3 = math.sqrt((x - 1) ** 2 + (3600 / c) * x / (450 * T))
            
            d = term1 + 900 * T * (term2 + term3)
            delays[idx] = max(0, d)
        
        return delays
    
    def _estimate_average_delay(self, capacities: Dict[int, float],
                                 conflicting_flows: Dict[int, float]) -> float:
        """Estimate average delay for LOS determination"""
        # Use 70% of capacity as typical volume
        # Calculate delay directly to avoid recursion
        volumes = {idx: c * 0.7 for idx, c in capacities.items()}
        delays = {}
        T = 0.25  # Analysis period
        
        for idx, movement in enumerate(self.movements):
            v = volumes.get(idx, 0)
            c = capacities.get(idx, 1)
            
            if c <= 0:
                delays[idx] = float('inf') if v > 0 else 0
                continue
            
            x = v / c  # Volume-to-capacity
            
            # HCM delay formula for unsignalized
            term1 = 3600 / c
            term2 = x - 1
            term3 = math.sqrt((x - 1) ** 2 + (3600 / c) * x / (450 * T))
            
            d = term1 + 900 * T * (term2 + term3)
            delays[idx] = max(0, d)
        
        if delays:
            return sum(delays.values()) / len(delays)
        return 0
    
    def _determine_los(self, delay: float) -> str:
        """LOS for unsignalized (different thresholds than signalized)"""
        thresholds = {'A': 10, 'B': 15, 'C': 25, 'D': 35, 'E': 50}
        for los, threshold in thresholds.items():
            if delay <= threshold:
                return los
        return 'F'
    
    def can_proceed(self, movement_index: int, current_time: float) -> bool:
        """
        Check if movement can proceed based on gap acceptance
        
        Simplified: major movements always can, minor depends on conflicts
        """
        if movement_index >= len(self.movements):
            return False
        
        movement = self.movements[movement_index]
        
        # Major movements can always proceed (simplified)
        if movement.priority == PriorityType.MAJOR:
            return True
        
        # Minor movements: check if blocked by conflicts
        if movement.is_blocked:
            return False
        
        # Would need gap information for full implementation
        return True


class RoundaboutJunction(JunctionModel):
    """
    Roundabout (Rotary) Junction Model
    
    Implements HCM 2010 roundabout capacity methodology.
    
    Mathematical formulation:
    -------------------------
    
    Entry capacity (single-lane roundabout):
        c_e = 1130 × e^(-0.001 × v_c)
    
    where v_c = circulating flow [pcu/hr]
    
    For multi-lane:
        c_e = 1130 × e^(-0.0010 × v_c) for single-lane entry
        c_e = 1130 × e^(-0.0007 × v_c) for multi-lane entry
    
    Follow-up headway: typically 2.6-3.1s
    Critical headway: typically 4.1-4.6s
    """
    
    def __init__(self, junction_id: int, movements: List[TurnMovement],
                 circulating_lanes: int = 1,
                 inscribed_diameter: float = 40.0):
        self.circulating_lanes = circulating_lanes
        self.inscribed_diameter = inscribed_diameter
        super().__init__(junction_id, movements)
    
    def _build_conflict_matrix(self) -> None:
        """Roundabout conflicts: entering vs circulating"""
        self.conflicts = []
        # All entries conflict with circulating traffic
        # (handled implicitly through capacity formula)
    
    def calculate_entry_capacity(self, circulating_flow: float,
                                  entry_lanes: int = 1) -> float:
        """
        Calculate entry capacity given circulating flow
        
        Args:
            circulating_flow: Flow in circulating roadway [veh/hr]
            entry_lanes: Number of entry lanes
            
        Returns:
            Entry capacity [veh/hr]
        """
        # Coefficient depends on entry geometry
        if entry_lanes == 1:
            A = 1130
            B = 0.0010
        else:
            A = 1130
            B = 0.0007
        
        c_e = A * math.exp(-B * circulating_flow)
        
        return c_e * entry_lanes
    
    def calculate_capacity(self,
                           circulating_flows: Optional[Dict[int, float]] = None) -> JunctionCapacity:
        """Calculate entry capacity for each approach"""
        if circulating_flows is None:
            circulating_flows = {i: 500 for i in range(len(self.movements))}
        
        capacities = {}
        total = 0.0
        
        for idx, movement in enumerate(self.movements):
            c_flow = circulating_flows.get(idx, 500)
            c = self.calculate_entry_capacity(c_flow, movement.num_lanes)
            capacities[idx] = c
            total += c
            movement.capacity = c
        
        avg_delay = 15.0  # Simplified
        los = 'C' if avg_delay < 25 else 'D'
        
        return JunctionCapacity(
            movement_capacities=capacities,
            total_capacity=total,
            critical_v_c_ratio=0,
            level_of_service=los,
            average_delay=avg_delay
        )
    
    def calculate_delay(self, volumes: Dict[int, float]) -> Dict[int, float]:
        """Calculate delay using gap acceptance formula"""
        # Simplified: use unsignalized formula structure
        delays = {}
        for idx, movement in enumerate(self.movements):
            v = volumes.get(idx, 0)
            c = movement.capacity or 1000
            
            if c <= 0:
                delays[idx] = float('inf') if v > 0 else 0
                continue
            
            x = v / c
            # Simplified delay formula
            delays[idx] = 5 + 15 * x + 50 * max(0, x - 0.9) ** 2
        
        return delays
    
    def can_proceed(self, movement_index: int, current_time: float) -> bool:
        """Check if entry can proceed (gap available in circulating)"""
        if movement_index >= len(self.movements):
            return False
        return not self.movements[movement_index].is_blocked


# =============================================================================
# Merge and Diverge Models
# =============================================================================

class MergeModel:
    """
    Freeway/Highway Merge Model
    
    Models capacity at merge sections where two streams combine.
    
    Mathematical formulation:
    -------------------------
    
    Merge capacity (HCM):
        c_merge = min(c_freeway, c_ramp + c_mainline_reduced)
    
    Mainline capacity reduction:
        c_mainline_reduced = c_mainline × f_merge
    
    where f_merge depends on merge geometry and ramp flow
    
    For zipper merge:
        Alternating vehicles from each stream
        c_merge ≈ min(c_1, c_2) × 2 × efficiency
    """
    
    def __init__(self, 
                 mainline_capacity: float,
                 ramp_capacity: float,
                 merge_type: str = 'parallel'):  # 'parallel', 'taper', 'direct'
        self.mainline_capacity = mainline_capacity
        self.ramp_capacity = ramp_capacity
        self.merge_type = merge_type
    
    def calculate_merge_capacity(self, mainline_flow: float,
                                  ramp_flow: float) -> Tuple[float, float]:
        """
        Calculate capacity allocation between streams
        
        Returns:
            (mainline_throughput_capacity, ramp_throughput_capacity)
        """
        total_downstream_capacity = self.mainline_capacity
        
        if mainline_flow + ramp_flow <= total_downstream_capacity:
            # Undersaturated: both can pass
            return (mainline_flow, ramp_flow)
        
        # Oversaturated: proportional allocation
        total_demand = mainline_flow + ramp_flow
        mainline_share = mainline_flow / total_demand
        ramp_share = ramp_flow / total_demand
        
        return (total_downstream_capacity * mainline_share,
                total_downstream_capacity * ramp_share)
    
    def calculate_zipper_capacity(self, flow_1: float, flow_2: float,
                                   efficiency: float = 0.85) -> float:
        """
        Calculate zipper merge capacity
        
        Vehicles alternate between streams
        """
        # Ideal zipper: each stream contributes equally
        min_flow = min(flow_1, flow_2)
        max_flow = max(flow_1, flow_2)
        
        # Both streams contribute up to their minimum, then excess from larger
        zipper_throughput = 2 * min_flow * efficiency
        excess = max_flow - min_flow
        
        total = zipper_throughput + excess * (1 - efficiency)
        
        return min(total, self.mainline_capacity)


class DivergeModel:
    """
    Freeway/Highway Diverge Model
    
    Models capacity at diverge sections where one stream splits.
    
    Mathematical formulation:
    -------------------------
    
    Diverge capacity generally equals upstream capacity since
    vehicles don't compete (they separate).
    
    However, weaving before diverge can reduce effective capacity:
        c_diverge = c_upstream × f_weave
    
    Lane utilization affects throughput per exit.
    """
    
    def __init__(self,
                 upstream_capacity: float,
                 mainline_lanes: int,
                 exit_lanes: int):
        self.upstream_capacity = upstream_capacity
        self.mainline_lanes = mainline_lanes
        self.exit_lanes = exit_lanes
    
    def calculate_diverge_capacity(self, 
                                    mainline_demand: float,
                                    exit_demand: float) -> Tuple[float, float]:
        """
        Calculate capacity for each branch
        
        Returns:
            (mainline_capacity, exit_capacity)
        """
        # Total upstream capacity limits sum
        total_demand = mainline_demand + exit_demand
        
        if total_demand <= self.upstream_capacity:
            return (mainline_demand, exit_demand)
        
        # Proportional reduction
        factor = self.upstream_capacity / total_demand
        return (mainline_demand * factor, exit_demand * factor)


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_webster_optimal_cycle(total_lost_time: float,
                                     critical_flow_ratios: List[float]) -> float:
    """
    Webster's formula for optimal signal cycle length
    
    C_opt = (1.5L + 5) / (1 - Y)
    
    where:
        L = total lost time per cycle
        Y = sum of critical flow ratios (v/s for critical movements)
    
    Args:
        total_lost_time: Sum of lost times for all phases [s]
        critical_flow_ratios: List of v/s ratios for critical movements
        
    Returns:
        Optimal cycle length [s]
    """
    Y = sum(critical_flow_ratios)
    
    if Y >= 1.0:
        return float('inf')  # Oversaturated
    
    C_opt = (1.5 * total_lost_time + 5) / (1 - Y)
    
    # Practical limits
    return max(30, min(180, C_opt))


def estimate_queue_at_signal(arrival_rate: float,
                              capacity: float,
                              cycle_length: float,
                              green_ratio: float,
                              analysis_period: float = 900) -> Tuple[float, float]:
    """
    Estimate average and maximum queue at signalized intersection
    
    Returns:
        (average_queue, max_queue) in vehicles
    """
    # Average queue during red
    red_time = cycle_length * (1 - green_ratio)
    avg_queue_red = arrival_rate * red_time / 2
    
    x = arrival_rate / capacity if capacity > 0 else float('inf')
    
    if x < 1:
        # Undersaturated
        avg_queue = avg_queue_red / (1 - x) if x < 0.99 else avg_queue_red * 10
        max_queue = arrival_rate * red_time
    else:
        # Oversaturated - queue grows
        overflow_rate = arrival_rate - capacity
        max_queue = arrival_rate * red_time + overflow_rate * analysis_period
        avg_queue = max_queue / 2
    
    return (avg_queue, max_queue)

