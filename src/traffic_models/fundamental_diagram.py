"""
Fundamental Diagram Models for Mesoscopic Traffic Simulation

This module implements various speed-density-flow relationships used in traffic flow theory.
These relationships are the mathematical foundation for mesoscopic simulation accuracy.

Mathematical Background:
------------------------
The fundamental diagram describes the relationship between three macroscopic traffic variables:
- Density (ρ): vehicles per unit length [veh/km or veh/m]
- Flow (q): vehicles per unit time [veh/hr or veh/s]  
- Speed (v): distance per unit time [km/hr or m/s]

The fundamental relationship is:
    q = ρ × v

Key points on the fundamental diagram:
- Free-flow: ρ → 0, v → v_free, q → 0
- Capacity: ρ = ρ_c (critical), q = q_max
- Jam: ρ = ρ_jam, v → 0, q → 0

References:
-----------
[1] Greenshields, B.D. (1935). "A study of traffic capacity"
[2] Newell, G.F. (1993). "A simplified theory of kinematic waves"
[3] Daganzo, C.F. (1994). "The cell transmission model"
[4] Underwood, R.T. (1961). "Speed, volume, and density relationships"
[5] Drake, J.S. et al. (1967). "A statistical analysis of speed-density hypotheses"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, Dict, Any
import math


class TrafficState(Enum):
    """Traffic state classification based on density"""
    FREE_FLOW = "free_flow"       # ρ < ρ_c, uncongested
    SYNCHRONIZED = "synchronized"  # ρ ≈ ρ_c, capacity flow
    CONGESTED = "congested"       # ρ > ρ_c, queue forming
    JAMMED = "jammed"             # ρ → ρ_jam, near standstill


@dataclass
class TrafficConditions:
    """Container for traffic state at a point/segment"""
    density: float          # veh/km (or veh/m depending on units)
    speed: float            # km/h (or m/s)
    flow: float             # veh/hr (or veh/s)
    state: TrafficState
    occupancy: float        # fraction of capacity (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'density': self.density,
            'speed': self.speed,
            'flow': self.flow,
            'state': self.state.value,
            'occupancy': self.occupancy
        }


@dataclass
class FundamentalDiagramParameters:
    """
    Parameters defining a fundamental diagram
    
    All parameters use consistent units:
    - Speeds in m/s
    - Densities in veh/m (per lane)
    - Flows in veh/s (per lane)
    """
    v_free: float           # Free-flow speed [m/s]
    rho_jam: float          # Jam density [veh/m/lane]
    rho_crit: float         # Critical density [veh/m/lane]
    q_max: float            # Capacity flow [veh/s/lane]
    w: float                # Backward wave speed [m/s], typically 4-6 m/s
    num_lanes: int = 1      # Number of lanes
    
    def __post_init__(self):
        """Validate parameters and derive missing values"""
        assert self.v_free > 0, "Free-flow speed must be positive"
        assert self.rho_jam > 0, "Jam density must be positive"
        assert self.rho_crit > 0, "Critical density must be positive"
        assert self.rho_crit < self.rho_jam, "Critical density must be less than jam density"
        
        # If q_max not specified, derive from triangular relationship
        if self.q_max <= 0:
            # q_max = v_free × rho_crit (for triangular diagram)
            self.q_max = self.v_free * self.rho_crit
        
        # If wave speed not specified, derive from conservation
        if self.w <= 0:
            # From triangular: w = q_max / (rho_jam - rho_crit)
            self.w = self.q_max / (self.rho_jam - self.rho_crit)
    
    @classmethod
    def from_typical_urban(cls, speed_limit_kmh: float = 50, 
                           num_lanes: int = 1) -> 'FundamentalDiagramParameters':
        """
        Create parameters typical for urban arterial roads
        
        Based on HCM 2010 and empirical studies:
        - Jam density: ~150 veh/km/lane = 0.15 veh/m/lane
        - Capacity: ~1800 veh/hr/lane = 0.5 veh/s/lane
        - Critical density: ~30 veh/km/lane = 0.03 veh/m/lane
        - Wave speed: ~5 m/s (18 km/h)
        """
        v_free = speed_limit_kmh / 3.6  # Convert to m/s
        rho_jam = 0.15                   # veh/m/lane
        q_max = 0.5                      # veh/s/lane
        rho_crit = q_max / v_free        # Derived
        w = 5.0                          # m/s
        
        return cls(v_free=v_free, rho_jam=rho_jam, rho_crit=rho_crit,
                   q_max=q_max, w=w, num_lanes=num_lanes)
    
    @classmethod
    def from_typical_highway(cls, speed_limit_kmh: float = 120,
                             num_lanes: int = 2) -> 'FundamentalDiagramParameters':
        """
        Create parameters typical for highway/freeway
        
        Based on HCM 2010:
        - Jam density: ~180 veh/km/lane = 0.18 veh/m/lane
        - Capacity: ~2200 veh/hr/lane = 0.611 veh/s/lane
        - Wave speed: ~6 m/s (21.6 km/h)
        """
        v_free = speed_limit_kmh / 3.6
        rho_jam = 0.18
        q_max = 0.611
        rho_crit = q_max / v_free
        w = 6.0
        
        return cls(v_free=v_free, rho_jam=rho_jam, rho_crit=rho_crit,
                   q_max=q_max, w=w, num_lanes=num_lanes)
    
    @classmethod  
    def from_typical_residential(cls, speed_limit_kmh: float = 30,
                                  num_lanes: int = 1) -> 'FundamentalDiagramParameters':
        """
        Create parameters typical for residential streets
        
        Lower capacity due to pedestrians, parking, etc.
        """
        v_free = speed_limit_kmh / 3.6
        rho_jam = 0.12
        q_max = 0.33  # ~1200 veh/hr/lane
        rho_crit = q_max / v_free
        w = 4.0
        
        return cls(v_free=v_free, rho_jam=rho_jam, rho_crit=rho_crit,
                   q_max=q_max, w=w, num_lanes=num_lanes)


class FundamentalDiagram(ABC):
    """
    Abstract base class for fundamental diagram models
    
    A fundamental diagram defines the relationship v = f(ρ) between
    speed and density. From this, flow is derived as q = ρ × v.
    
    Subclasses implement different mathematical formulations.
    """
    
    def __init__(self, params: FundamentalDiagramParameters):
        self.params = params
    
    @abstractmethod
    def speed(self, density: float) -> float:
        """
        Calculate speed given density
        
        Args:
            density: Traffic density [veh/m/lane]
            
        Returns:
            Speed [m/s]
        """
        pass
    
    def flow(self, density: float) -> float:
        """
        Calculate flow given density (q = ρ × v)
        
        Args:
            density: Traffic density [veh/m/lane]
            
        Returns:
            Flow [veh/s/lane]
        """
        return density * self.speed(density)
    
    def travel_time(self, length: float, density: float) -> float:
        """
        Calculate travel time for a segment
        
        Args:
            length: Segment length [m]
            density: Traffic density [veh/m/lane]
            
        Returns:
            Travel time [s]
        """
        v = self.speed(density)
        if v <= 0:
            return float('inf')
        return length / v
    
    def free_flow_travel_time(self, length: float) -> float:
        """Calculate free-flow travel time for a segment"""
        return length / self.params.v_free
    
    def density_from_count(self, vehicle_count: int, length: float) -> float:
        """
        Convert vehicle count to density
        
        Args:
            vehicle_count: Number of vehicles on segment
            length: Segment length [m]
            
        Returns:
            Density [veh/m/lane]
        """
        return vehicle_count / (length * self.params.num_lanes)
    
    def capacity(self) -> float:
        """Return the capacity (max flow) [veh/s] for all lanes"""
        return self.params.q_max * self.params.num_lanes
    
    def jam_vehicle_count(self, length: float) -> int:
        """Maximum vehicles that fit on a segment"""
        return int(self.params.rho_jam * length * self.params.num_lanes)
    
    def classify_state(self, density: float) -> TrafficState:
        """
        Classify traffic state based on density
        
        Uses thresholds relative to critical and jam density
        """
        if density <= 0:
            return TrafficState.FREE_FLOW
        
        rho_crit = self.params.rho_crit
        rho_jam = self.params.rho_jam
        
        if density < 0.7 * rho_crit:
            return TrafficState.FREE_FLOW
        elif density < 1.3 * rho_crit:
            return TrafficState.SYNCHRONIZED
        elif density < 0.9 * rho_jam:
            return TrafficState.CONGESTED
        else:
            return TrafficState.JAMMED
    
    def get_conditions(self, density: float) -> TrafficConditions:
        """Get complete traffic conditions for a given density"""
        v = self.speed(density)
        q = self.flow(density)
        state = self.classify_state(density)
        occupancy = density / self.params.rho_jam
        
        return TrafficConditions(
            density=density,
            speed=v,
            flow=q,
            state=state,
            occupancy=occupancy
        )
    
    def wave_speed(self, density: float) -> float:
        """
        Calculate kinematic wave speed dq/dρ
        
        Positive for free-flow conditions, negative for congested
        """
        # Numerical differentiation
        eps = 1e-6
        q1 = self.flow(density - eps)
        q2 = self.flow(density + eps)
        return (q2 - q1) / (2 * eps)


class GreenshieldsModel(FundamentalDiagram):
    """
    Greenshields (1935) Linear Speed-Density Model
    
    The simplest and oldest fundamental diagram model.
    
    Mathematical formulation:
    -------------------------
    v(ρ) = v_free × (1 - ρ/ρ_jam)
    
    Resulting flow-density relationship:
    q(ρ) = v_free × ρ × (1 - ρ/ρ_jam)
    
    This is a parabola with:
    - Maximum flow at ρ_c = ρ_jam/2
    - q_max = v_free × ρ_jam / 4
    
    Advantages:
    - Simple, analytically tractable
    - Good for basic understanding
    
    Disadvantages:
    - Overestimates speed reduction at low densities
    - Critical density fixed at ρ_jam/2 (often too high)
    - Poor fit to real highway data
    
    Accuracy rating: ★★☆☆☆ (use for simple scenarios only)
    """
    
    def speed(self, density: float) -> float:
        """
        v(ρ) = v_free × (1 - ρ/ρ_jam)
        
        Clamped to [0, v_free]
        """
        if density <= 0:
            return self.params.v_free
        if density >= self.params.rho_jam:
            return 0.0
        
        v = self.params.v_free * (1 - density / self.params.rho_jam)
        return max(0.0, v)
    
    @property
    def theoretical_critical_density(self) -> float:
        """For Greenshields, ρ_c = ρ_jam / 2"""
        return self.params.rho_jam / 2
    
    @property
    def theoretical_max_flow(self) -> float:
        """For Greenshields, q_max = v_free × ρ_jam / 4"""
        return self.params.v_free * self.params.rho_jam / 4


class NewellDaganzoModel(FundamentalDiagram):
    """
    Newell-Daganzo Triangular Fundamental Diagram
    
    The most widely used model in mesoscopic simulation (including SUMO).
    Based on kinematic wave theory with two traffic regimes.
    
    Mathematical formulation:
    -------------------------
    Two-regime model:
    
    Free-flow regime (ρ ≤ ρ_c):
        v(ρ) = v_free
        q(ρ) = v_free × ρ
    
    Congested regime (ρ > ρ_c):
        v(ρ) = w × (ρ_jam - ρ) / ρ
        q(ρ) = w × (ρ_jam - ρ)
    
    where:
        w = backward wave speed (shock wave speed in congestion)
        ρ_c = q_max / v_free (critical density)
    
    The model creates a triangular shape in q-ρ space:
    - Left branch: q = v_free × ρ (free-flow)
    - Right branch: q = w × (ρ_jam - ρ) (congested)
    
    Conservation relationship:
        At ρ_c: v_free × ρ_c = w × (ρ_jam - ρ_c)
        Therefore: w = v_free × ρ_c / (ρ_jam - ρ_c)
    
    Kinematic wave speeds:
    - Free-flow: dq/dρ = v_free (positive, forward propagation)
    - Congested: dq/dρ = -w (negative, backward propagation)
    
    Advantages:
    - Physically motivated (kinematic wave theory)
    - Accurately models queue spillback
    - Matches observed capacity drop
    - Used in SUMO, VISSIM mesoscopic modes
    
    Accuracy rating: ★★★★★ (recommended for mesoscopic simulation)
    """
    
    def speed(self, density: float) -> float:
        """
        Triangular speed-density relationship
        
        Free-flow:  v = v_free (constant)
        Congested:  v = w × (ρ_jam - ρ) / ρ
        """
        if density <= 0:
            return self.params.v_free
        if density >= self.params.rho_jam:
            return 0.0
        
        rho_crit = self.params.rho_crit
        
        if density <= rho_crit:
            # Free-flow regime: constant speed
            return self.params.v_free
        else:
            # Congested regime: hyperbolic decrease
            # v = w × (ρ_jam - ρ) / ρ
            w = self.params.w
            rho_jam = self.params.rho_jam
            v = w * (rho_jam - density) / density
            return max(0.0, v)
    
    def flow(self, density: float) -> float:
        """
        Triangular flow-density relationship
        
        Provides cleaner analytical form than base class
        """
        if density <= 0:
            return 0.0
        if density >= self.params.rho_jam:
            return 0.0
        
        rho_crit = self.params.rho_crit
        
        if density <= rho_crit:
            # Free-flow: q = v_free × ρ
            return self.params.v_free * density
        else:
            # Congested: q = w × (ρ_jam - ρ)
            return self.params.w * (self.params.rho_jam - density)
    
    def wave_speed(self, density: float) -> float:
        """
        Kinematic wave speed (characteristic speed)
        
        Constant in each regime for triangular model
        """
        if density <= self.params.rho_crit:
            return self.params.v_free  # Forward wave
        else:
            return -self.params.w  # Backward wave (queue spillback)
    
    def shock_wave_speed(self, density_upstream: float, 
                         density_downstream: float) -> float:
        """
        Calculate shock wave speed between two traffic states
        
        Rankine-Hugoniot condition:
        σ = (q_downstream - q_upstream) / (ρ_downstream - ρ_upstream)
        
        This determines how fast a queue grows/shrinks
        """
        if abs(density_downstream - density_upstream) < 1e-10:
            return self.wave_speed(density_upstream)
        
        q_up = self.flow(density_upstream)
        q_down = self.flow(density_downstream)
        
        return (q_down - q_up) / (density_downstream - density_upstream)
    
    def queue_growth_rate(self, arrival_rate: float, 
                          service_rate: float) -> float:
        """
        Calculate rate at which queue length changes
        
        Args:
            arrival_rate: Flow arriving at queue tail [veh/s]
            service_rate: Flow departing queue head [veh/s]
            
        Returns:
            Queue growth rate [veh/s] (positive = growing)
        """
        return arrival_rate - service_rate
    
    def queue_spillback_time(self, queue_length: float, 
                             arrival_rate: float) -> float:
        """
        Time for queue to spill back a given distance
        
        Based on backward wave speed
        """
        if arrival_rate <= self.params.q_max:
            return float('inf')  # No spillback if under capacity
        
        # Queue growth rate in length units
        excess_rate = arrival_rate - self.params.q_max
        density_in_queue = self.params.rho_jam  # Approximate
        
        # Length growth = excess_vehicles / jam_density
        length_growth_rate = excess_rate / density_in_queue
        
        return queue_length / (self.params.w + length_growth_rate)


class UnderwoodModel(FundamentalDiagram):
    """
    Underwood (1961) Exponential Speed-Density Model
    
    Uses exponential decay of speed with increasing density.
    
    Mathematical formulation:
    -------------------------
    v(ρ) = v_free × exp(-ρ/ρ_c)
    
    where ρ_c is the critical density (density at maximum flow)
    
    Properties:
    - Speed never reaches zero (asymptotic)
    - Maximum flow at ρ = ρ_c
    - q_max = v_free × ρ_c × exp(-1) ≈ 0.368 × v_free × ρ_c
    
    Advantages:
    - Continuous and smooth
    - Good fit for uncongested conditions
    
    Disadvantages:
    - Speed doesn't reach zero at jam density
    - Can be unrealistic at very high densities
    
    Accuracy rating: ★★★☆☆ (good for moderate densities)
    """
    
    def speed(self, density: float) -> float:
        """
        v(ρ) = v_free × exp(-ρ/ρ_c)
        """
        if density <= 0:
            return self.params.v_free
        
        rho_crit = self.params.rho_crit
        v = self.params.v_free * math.exp(-density / rho_crit)
        
        # Enforce minimum near jam density
        if density >= 0.95 * self.params.rho_jam:
            return max(0.0, v * (self.params.rho_jam - density) / 
                       (0.05 * self.params.rho_jam))
        
        return v


class DrakeModel(FundamentalDiagram):
    """
    Drake (1967) Bell-Shaped Speed-Density Model
    
    Uses Gaussian-like decay for smoother transitions.
    
    Mathematical formulation:
    -------------------------
    v(ρ) = v_free × exp(-0.5 × (ρ/ρ_c)²)
    
    Properties:
    - Bell-shaped curve
    - Maximum flow at ρ = ρ_c / √2
    - Very smooth transitions
    
    Advantages:
    - Smooth second derivative
    - Good for stability in simulation
    
    Disadvantages:
    - May not capture sharp transitions at capacity
    
    Accuracy rating: ★★★☆☆ (good for smooth flow modeling)
    """
    
    def speed(self, density: float) -> float:
        """
        v(ρ) = v_free × exp(-0.5 × (ρ/ρ_c)²)
        """
        if density <= 0:
            return self.params.v_free
        
        rho_crit = self.params.rho_crit
        v = self.params.v_free * math.exp(-0.5 * (density / rho_crit) ** 2)
        
        # Enforce jam condition
        if density >= self.params.rho_jam:
            return 0.0
        
        return v


class ThreeParameterModel(FundamentalDiagram):
    """
    Three-Parameter Logistic Speed-Density Model
    
    A flexible model that can be calibrated to match real data closely.
    
    Mathematical formulation:
    -------------------------
    v(ρ) = v_free / (1 + exp(α × (ρ - ρ_c) / ρ_c))^β
    
    where:
        α = steepness parameter (typically 2-5)
        β = shape parameter (typically 0.5-2)
    
    Special cases:
        α → ∞: approaches step function (triangular-like)
        β = 1: standard logistic
    
    Advantages:
    - Highly flexible
    - Can match various observed patterns
    - Good for calibration
    
    Accuracy rating: ★★★★☆ (excellent with proper calibration)
    """
    
    def __init__(self, params: FundamentalDiagramParameters,
                 alpha: float = 3.0, beta: float = 1.0):
        super().__init__(params)
        self.alpha = alpha
        self.beta = beta
    
    def speed(self, density: float) -> float:
        """
        v(ρ) = v_free / (1 + exp(α × (ρ - ρ_c) / ρ_c))^β
        """
        if density <= 0:
            return self.params.v_free
        if density >= self.params.rho_jam:
            return 0.0
        
        rho_crit = self.params.rho_crit
        
        # Logistic component
        exp_term = math.exp(self.alpha * (density - rho_crit) / rho_crit)
        v = self.params.v_free / ((1 + exp_term) ** self.beta)
        
        return max(0.0, v)


# =============================================================================
# SUMO Compatibility Layer
# =============================================================================

class SUMOMesoSpeedModel:
    """
    SUMO Mesoscopic Speed Model Implementation
    
    SUMO meso uses a specific approach with TAU factors to adjust
    travel times based on edge state. This class replicates that behavior.
    
    SUMO Parameters:
    ----------------
    TAUFF: Travel time factor free-flow → free-flow (default: 1.4)
    TAUFJ: Travel time factor free-flow → jam (default: 1.4)  
    TAUJF: Travel time factor jam → free-flow (default: 2.0)
    TAUJJ: Travel time factor jam → jam (default: 1.4)
    
    The factor applied depends on:
    - Current edge state (free/jammed)
    - Next edge state (free/jammed)
    """
    
    # Default SUMO meso parameters
    DEFAULT_TAUFF = 1.4
    DEFAULT_TAUFJ = 1.4
    DEFAULT_TAUJF = 2.0
    DEFAULT_TAUJJ = 1.4
    DEFAULT_JAM_THRESHOLD = 0.8  # Occupancy threshold for "jammed"
    
    def __init__(self, 
                 tau_ff: float = DEFAULT_TAUFF,
                 tau_fj: float = DEFAULT_TAUFJ,
                 tau_jf: float = DEFAULT_TAUJF,
                 tau_jj: float = DEFAULT_TAUJJ,
                 jam_threshold: float = DEFAULT_JAM_THRESHOLD):
        self.tau_ff = tau_ff
        self.tau_fj = tau_fj
        self.tau_jf = tau_jf
        self.tau_jj = tau_jj
        self.jam_threshold = jam_threshold
    
    def is_jammed(self, occupancy: float) -> bool:
        """Check if edge is jammed based on occupancy"""
        return occupancy >= self.jam_threshold
    
    def get_tau_factor(self, curr_jammed: bool, next_jammed: bool) -> float:
        """
        Get appropriate TAU factor based on current and next edge states
        
        Args:
            curr_jammed: Is current edge jammed?
            next_jammed: Is next edge jammed?
            
        Returns:
            TAU factor to multiply base travel time by
        """
        if not curr_jammed and not next_jammed:
            return self.tau_ff
        elif not curr_jammed and next_jammed:
            return self.tau_fj
        elif curr_jammed and not next_jammed:
            return self.tau_jf
        else:  # both jammed
            return self.tau_jj
    
    def adjusted_travel_time(self, base_travel_time: float,
                             curr_occupancy: float,
                             next_occupancy: float) -> float:
        """
        Calculate SUMO-style adjusted travel time
        
        Args:
            base_travel_time: Free-flow travel time [s]
            curr_occupancy: Current edge occupancy (0-1)
            next_occupancy: Next edge occupancy (0-1)
            
        Returns:
            Adjusted travel time [s]
        """
        curr_jammed = self.is_jammed(curr_occupancy)
        next_jammed = self.is_jammed(next_occupancy)
        tau = self.get_tau_factor(curr_jammed, next_jammed)
        
        return base_travel_time * tau


# =============================================================================
# Factory Function
# =============================================================================

def create_fundamental_diagram(model_type: str, 
                               params: FundamentalDiagramParameters,
                               **kwargs) -> FundamentalDiagram:
    """
    Factory function to create fundamental diagram models
    
    Args:
        model_type: One of 'greenshields', 'newell_daganzo', 'underwood', 
                    'drake', 'three_parameter'
        params: Fundamental diagram parameters
        **kwargs: Additional model-specific parameters
        
    Returns:
        FundamentalDiagram instance
    """
    models = {
        'greenshields': GreenshieldsModel,
        'newell_daganzo': NewellDaganzoModel,
        'underwood': UnderwoodModel,
        'drake': DrakeModel,
        'three_parameter': ThreeParameterModel,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: {list(models.keys())}")
    
    return models[model_type](params, **kwargs)

