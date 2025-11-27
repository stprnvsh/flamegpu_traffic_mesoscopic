"""
Test Suite: Junction Models

This test bench validates junction capacity and delay models:
- Signalized intersections (HCM methodology)
- Unsignalized intersections (gap acceptance)
- Roundabouts
- Merge/diverge sections

Test Categories:
1. Capacity Calculations
   - Saturation flow adjustments
   - Green time allocation
   - Gap acceptance capacity
   
2. Delay Calculations
   - HCM control delay
   - Webster's formula
   - Queue estimation
   
3. Conflict Resolution
   - Signal phase conflicts
   - Priority rules
   - Merge allocation
   
4. Level of Service
   - LOS classification
   - Threshold validation
"""

import pytest
import math
from typing import List, Dict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from traffic_models.junction_models import (
    MovementType,
    SignalState,
    PriorityType,
    TurnMovement,
    SignalPhase,
    SignalTiming,
    JunctionCapacity,
    JunctionModel,
    SignalizedJunction,
    UnsignalizedJunction,
    RoundaboutJunction,
    MergeModel,
    DivergeModel,
    ConflictPair,
    calculate_webster_optimal_cycle,
    estimate_queue_at_signal,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basic_movements() -> List[TurnMovement]:
    """Four approach intersection movements"""
    return [
        TurnMovement(from_edge=0, to_edge=2, movement_type=MovementType.THROUGH, num_lanes=2),
        TurnMovement(from_edge=0, to_edge=3, movement_type=MovementType.LEFT, num_lanes=1),
        TurnMovement(from_edge=1, to_edge=3, movement_type=MovementType.THROUGH, num_lanes=2),
        TurnMovement(from_edge=1, to_edge=0, movement_type=MovementType.LEFT, num_lanes=1),
    ]


@pytest.fixture
def basic_signal_timing() -> SignalTiming:
    """Two-phase signal timing plan"""
    phases = [
        SignalPhase(phase_id=0, duration=30, green_movements=[0, 1], 
                   yellow_time=3, all_red_time=2),
        SignalPhase(phase_id=1, duration=25, green_movements=[2, 3],
                   yellow_time=3, all_red_time=2),
    ]
    return SignalTiming(phases=phases)


@pytest.fixture
def signalized_junction(basic_movements, basic_signal_timing) -> SignalizedJunction:
    """Standard signalized intersection"""
    return SignalizedJunction(
        junction_id=1,
        movements=basic_movements,
        signal_timing=basic_signal_timing
    )


@pytest.fixture
def unsignalized_movements() -> List[TurnMovement]:
    """T-intersection movements with priority"""
    return [
        TurnMovement(from_edge=0, to_edge=2, movement_type=MovementType.THROUGH,
                    priority=PriorityType.MAJOR, num_lanes=1),
        TurnMovement(from_edge=0, to_edge=1, movement_type=MovementType.LEFT,
                    priority=PriorityType.MAJOR, num_lanes=1),
        TurnMovement(from_edge=1, to_edge=0, movement_type=MovementType.LEFT,
                    priority=PriorityType.MINOR, num_lanes=1),
        TurnMovement(from_edge=1, to_edge=2, movement_type=MovementType.RIGHT,
                    priority=PriorityType.MINOR, num_lanes=1),
    ]


@pytest.fixture
def unsignalized_junction(unsignalized_movements) -> UnsignalizedJunction:
    """Standard unsignalized T-intersection"""
    return UnsignalizedJunction(
        junction_id=2,
        movements=unsignalized_movements,
        control_type='twsc'  # Two-way stop control
    )


# =============================================================================
# Test Class: Signal Timing
# =============================================================================

class TestSignalTiming:
    """Test signal timing plan functionality"""
    
    def test_cycle_length_calculation(self, basic_signal_timing):
        """Cycle length should sum all phase times"""
        # Phase 0: 30 + 3 + 2 = 35s
        # Phase 1: 25 + 3 + 2 = 30s
        # Total: 65s
        assert basic_signal_timing.cycle_length == 65
    
    def test_phase_at_time_first_phase(self, basic_signal_timing):
        """Should return first phase at start of cycle"""
        phase, time_into = basic_signal_timing.get_phase_at_time(0)
        
        assert phase.phase_id == 0
        assert time_into == 0
    
    def test_phase_at_time_second_phase(self, basic_signal_timing):
        """Should return second phase after first completes"""
        # First phase total time = 35s
        phase, time_into = basic_signal_timing.get_phase_at_time(40)
        
        assert phase.phase_id == 1
        assert time_into == pytest.approx(5, abs=0.1)  # 40 - 35 = 5s into phase 1
    
    def test_phase_at_time_wraps_around(self, basic_signal_timing):
        """Should wrap around at end of cycle"""
        cycle = basic_signal_timing.cycle_length
        
        phase, time_into = basic_signal_timing.get_phase_at_time(cycle + 10)
        
        assert phase.phase_id == 0
        assert time_into == pytest.approx(10, abs=0.1)
    
    def test_green_ratio_calculation(self, basic_signal_timing):
        """Green ratio should be effective_green / cycle_length"""
        # Movement 0 is green in phase 0
        # Effective green = 30 - 2 + 3*0.5 = 29.5s (approx)
        # g/C = 29.5 / 65 ≈ 0.45
        
        g_C = basic_signal_timing.get_green_ratio(0)
        
        assert 0.3 < g_C < 0.6
    
    def test_effective_green_calculation(self):
        """Effective green should account for startup loss"""
        phase = SignalPhase(
            phase_id=0,
            duration=30,
            green_movements=[0],
            yellow_time=3,
            all_red_time=2
        )
        
        # Effective = duration - startup_loss + yellow*0.5
        # = 30 - 2 + 1.5 = 29.5
        assert phase.effective_green == pytest.approx(29.5, rel=0.01)
    
    def test_offset_affects_phase_timing(self):
        """Offset should shift phase timing"""
        phases = [
            SignalPhase(phase_id=0, duration=30, green_movements=[0]),
            SignalPhase(phase_id=1, duration=30, green_movements=[1]),
        ]
        
        timing_no_offset = SignalTiming(phases=phases, offset=0)
        timing_with_offset = SignalTiming(phases=phases, offset=20)
        
        # At t=0, no offset should be in phase 0
        phase_no, _ = timing_no_offset.get_phase_at_time(0)
        assert phase_no.phase_id == 0
        
        # At t=0, offset=20 should be like t=-20 (mod cycle), so in phase 1
        phase_offset, _ = timing_with_offset.get_phase_at_time(0)
        # With offset=20, effective time = -20 mod cycle


# =============================================================================
# Test Class: Signalized Junction Capacity
# =============================================================================

class TestSignalizedJunctionCapacity:
    """Test signalized intersection capacity calculations"""
    
    def test_saturation_flow_base_value(self, signalized_junction):
        """Base saturation flow should be ~1900 veh/hr/lane"""
        movement = signalized_junction.movements[0]
        
        sat_flow = signalized_junction.calculate_saturation_flow(movement)
        
        # Should be close to base (1900) with minor adjustments
        assert 1600 < sat_flow < 2000
    
    def test_saturation_flow_heavy_vehicle_adjustment(self, signalized_junction):
        """Heavy vehicles should reduce saturation flow"""
        movement = signalized_junction.movements[0]
        
        sat_no_hv = signalized_junction.calculate_saturation_flow(
            movement, heavy_vehicle_pct=0)
        sat_with_hv = signalized_junction.calculate_saturation_flow(
            movement, heavy_vehicle_pct=0.10)
        
        assert sat_with_hv < sat_no_hv
    
    def test_saturation_flow_grade_adjustment(self, signalized_junction):
        """Uphill grade should reduce saturation flow"""
        movement = signalized_junction.movements[0]
        
        sat_flat = signalized_junction.calculate_saturation_flow(
            movement, grade_pct=0)
        sat_uphill = signalized_junction.calculate_saturation_flow(
            movement, grade_pct=5)
        
        assert sat_uphill < sat_flat
    
    def test_saturation_flow_turn_type_adjustment(self, signalized_junction):
        """Turn movements should have lower saturation flow"""
        through_movement = signalized_junction.movements[0]  # THROUGH
        left_movement = signalized_junction.movements[1]     # LEFT
        
        sat_through = signalized_junction.calculate_saturation_flow(through_movement)
        sat_left = signalized_junction.calculate_saturation_flow(left_movement)
        
        assert sat_left < sat_through
    
    def test_capacity_calculation(self, signalized_junction):
        """Capacity should be saturation_flow × g/C × lanes"""
        result = signalized_junction.calculate_capacity()
        
        # All movements should have positive capacity
        for idx, cap in result.movement_capacities.items():
            assert cap > 0
        
        assert result.total_capacity > 0
    
    def test_capacity_proportional_to_green_time(self):
        """More green time should give more capacity"""
        movements = [
            TurnMovement(from_edge=0, to_edge=1, movement_type=MovementType.THROUGH)
        ]
        
        # Short green
        short_phases = [SignalPhase(phase_id=0, duration=20, green_movements=[0])]
        short_timing = SignalTiming(phases=short_phases)
        short_junction = SignalizedJunction(1, movements.copy(), short_timing)
        
        # Long green
        long_phases = [SignalPhase(phase_id=0, duration=50, green_movements=[0])]
        long_timing = SignalTiming(phases=long_phases)
        long_junction = SignalizedJunction(2, movements.copy(), long_timing)
        
        short_cap = short_junction.calculate_capacity().movement_capacities[0]
        long_cap = long_junction.calculate_capacity().movement_capacities[0]
        
        assert long_cap > short_cap


# =============================================================================
# Test Class: Signalized Junction Delay
# =============================================================================

class TestSignalizedJunctionDelay:
    """Test signalized intersection delay calculations"""
    
    def test_delay_undersaturated(self, signalized_junction):
        """Delay should be finite when undersaturated"""
        capacity = signalized_junction.calculate_capacity()
        
        # Set volumes at 70% of capacity
        volumes = {idx: cap * 0.7 for idx, cap in 
                   capacity.movement_capacities.items()}
        
        delays = signalized_junction.calculate_delay(volumes)
        
        for idx, delay in delays.items():
            assert delay > 0
            assert math.isfinite(delay)
    
    def test_delay_increases_with_demand(self, signalized_junction):
        """Delay should increase as demand approaches capacity"""
        capacity = signalized_junction.calculate_capacity()
        
        delays_at_levels = []
        for x in [0.3, 0.5, 0.7, 0.9]:
            volumes = {idx: cap * x for idx, cap in 
                       capacity.movement_capacities.items()}
            delays = signalized_junction.calculate_delay(volumes)
            avg_delay = sum(delays.values()) / len(delays)
            delays_at_levels.append(avg_delay)
        
        # Should be monotonically increasing
        for i in range(len(delays_at_levels) - 1):
            assert delays_at_levels[i] < delays_at_levels[i + 1]
    
    def test_delay_approaches_infinity_at_saturation(self, signalized_junction):
        """Delay should become very large near saturation"""
        capacity = signalized_junction.calculate_capacity()
        
        # Set volumes at 99% of capacity
        volumes = {idx: cap * 0.99 for idx, cap in 
                   capacity.movement_capacities.items()}
        
        delays = signalized_junction.calculate_delay(volumes)
        
        # At least one delay should be high
        max_delay = max(delays.values())
        assert max_delay > 50  # More than 50 seconds
    
    def test_uniform_delay_component(self, signalized_junction):
        """Uniform delay should depend on green ratio and cycle"""
        # This is implicitly tested through overall delay
        capacity = signalized_junction.calculate_capacity()
        volumes = {idx: cap * 0.5 for idx, cap in 
                   capacity.movement_capacities.items()}
        
        delays = signalized_junction.calculate_delay(volumes)
        
        # Uniform delay typically 10-30s for 50% v/c
        for delay in delays.values():
            assert 5 < delay < 60


# =============================================================================
# Test Class: Signal State and Can Proceed
# =============================================================================

class TestSignalStateCanProceed:
    """Test signal state queries"""
    
    def test_can_proceed_during_green(self, signalized_junction):
        """Movement should proceed during its green phase"""
        # Movement 0 is green in phase 0 (first 30 seconds)
        can_proceed = signalized_junction.can_proceed(0, current_time=15)
        
        assert can_proceed == True
    
    def test_cannot_proceed_during_red(self, signalized_junction):
        """Movement should not proceed during red"""
        # Movement 0 is red in phase 1 (starts at 35s)
        can_proceed = signalized_junction.can_proceed(0, current_time=50)
        
        assert can_proceed == False
    
    def test_cannot_proceed_during_yellow(self, signalized_junction):
        """Movement should not proceed during yellow/all-red"""
        # Phase 0 duration is 30s, then yellow
        # At t=32, should be in yellow period
        can_proceed = signalized_junction.can_proceed(0, current_time=32)
        
        assert can_proceed == False
    
    def test_time_to_green_calculation(self, signalized_junction):
        """Time to green should be calculated correctly"""
        # Movement 0 is green in phase 0
        # If currently in phase 1, should calculate time until phase 0
        
        # At t=0, movement 0 is green, so time_to_green = 0
        time_to_green = signalized_junction.get_time_to_green(0, current_time=0)
        assert time_to_green == 0
        
        # At t=50 (in phase 1), should wait for next cycle
        time_to_green = signalized_junction.get_time_to_green(0, current_time=50)
        assert time_to_green > 0


# =============================================================================
# Test Class: Unsignalized Junction Capacity
# =============================================================================

class TestUnsignalizedJunctionCapacity:
    """Test unsignalized intersection capacity calculations"""
    
    def test_critical_gap_assignment(self, unsignalized_junction):
        """Critical gaps should be assigned by movement type and priority"""
        for movement in unsignalized_junction.movements:
            assert movement.critical_gap > 0
            
            # Left turns should have larger critical gaps
            if movement.movement_type == MovementType.LEFT:
                assert movement.critical_gap >= 4.0
    
    def test_potential_capacity_decreases_with_conflict(self, unsignalized_junction):
        """Capacity should decrease as conflicting flow increases"""
        movement = unsignalized_junction.movements[2]  # Minor left
        
        cap_low = unsignalized_junction.calculate_potential_capacity(movement, 100)
        cap_high = unsignalized_junction.calculate_potential_capacity(movement, 500)
        
        assert cap_high < cap_low
    
    def test_potential_capacity_high_with_no_conflict(self, unsignalized_junction):
        """Capacity should be high when no conflicting flow"""
        movement = unsignalized_junction.movements[0]  # Major through
        
        cap = unsignalized_junction.calculate_potential_capacity(movement, 0)
        
        # Should be limited only by follow-up time: 3600/t_f
        max_cap = 3600 / movement.follow_up_time
        assert cap == pytest.approx(max_cap, rel=0.1)
    
    def test_gap_acceptance_formula(self, unsignalized_junction):
        """Verify gap acceptance formula implementation"""
        movement = unsignalized_junction.movements[2]
        v_c = 400  # conflicting flow
        
        cap = unsignalized_junction.calculate_potential_capacity(movement, v_c)
        
        # Manual calculation: c = v_c × e^(-v_c×t_c/3600) / (1 - e^(-v_c×t_f/3600))
        t_c = movement.critical_gap
        t_f = movement.follow_up_time
        
        exp_tc = math.exp(-v_c * t_c / 3600)
        exp_tf = math.exp(-v_c * t_f / 3600)
        expected = v_c * exp_tc / (1 - exp_tf)
        
        assert cap == pytest.approx(expected, rel=0.01)
    
    def test_major_movement_higher_capacity(self, unsignalized_junction):
        """Major movements should have higher capacity than minor"""
        conflicting_flows = {
            0: 0,    # Major through - no conflict
            1: 200,  # Major left - conflicts with opposing
            2: 400,  # Minor left - conflicts with major
            3: 300,  # Minor right - conflicts with major
        }
        
        result = unsignalized_junction.calculate_capacity(conflicting_flows)
        
        # Major through should have highest capacity
        major_through_cap = result.movement_capacities[0]
        minor_left_cap = result.movement_capacities[2]
        
        assert major_through_cap > minor_left_cap


# =============================================================================
# Test Class: Unsignalized Junction Delay
# =============================================================================

class TestUnsignalizedJunctionDelay:
    """Test unsignalized intersection delay calculations"""
    
    def test_delay_positive_for_minor_movements(self, unsignalized_junction):
        """Minor movements should have positive delay"""
        volumes = {0: 200, 1: 50, 2: 100, 3: 150}
        conflicting_flows = {0: 0, 1: 200, 2: 400, 3: 300}
        
        delays = unsignalized_junction.calculate_delay(volumes, conflicting_flows)
        
        # Minor movements (2, 3) should have delay
        assert delays[2] > 0
        assert delays[3] > 0
    
    def test_delay_increases_with_conflicting_flow(self, unsignalized_junction):
        """Delay should increase with more conflicting traffic"""
        volumes = {0: 200, 1: 50, 2: 100, 3: 150}
        
        delays_low = unsignalized_junction.calculate_delay(
            volumes, {0: 0, 1: 100, 2: 200, 3: 200})
        delays_high = unsignalized_junction.calculate_delay(
            volumes, {0: 0, 1: 300, 2: 600, 3: 500})
        
        # Minor movement delay should be higher with more conflicts
        assert delays_high[2] > delays_low[2]


# =============================================================================
# Test Class: Roundabout
# =============================================================================

class TestRoundaboutJunction:
    """Test roundabout capacity model"""
    
    def test_entry_capacity_formula(self):
        """Verify entry capacity formula: c = A × e^(-B × v_c)"""
        movements = [TurnMovement(from_edge=0, to_edge=1, 
                                  movement_type=MovementType.THROUGH)]
        roundabout = RoundaboutJunction(1, movements, circulating_lanes=1)
        
        # Test with various circulating flows
        cap_low = roundabout.calculate_entry_capacity(200)
        cap_high = roundabout.calculate_entry_capacity(800)
        
        assert cap_low > cap_high
        
        # Manual verification for low flow
        expected_low = 1130 * math.exp(-0.001 * 200)
        assert cap_low == pytest.approx(expected_low, rel=0.01)
    
    def test_entry_capacity_scales_with_lanes(self):
        """Multi-lane entry should have higher capacity"""
        movements = [TurnMovement(from_edge=0, to_edge=1,
                                  movement_type=MovementType.THROUGH, num_lanes=1)]
        roundabout = RoundaboutJunction(1, movements)
        
        cap_1_lane = roundabout.calculate_entry_capacity(400, entry_lanes=1)
        cap_2_lane = roundabout.calculate_entry_capacity(400, entry_lanes=2)
        
        # Multi-lane entry should have more capacity
        assert cap_2_lane > cap_1_lane
        # Due to different coefficients for multi-lane (0.0007 vs 0.0010),
        # the ratio is > 2 (better efficiency per lane)
        assert cap_2_lane > cap_1_lane * 1.5
    
    def test_roundabout_delay_calculation(self):
        """Roundabout delay should be positive"""
        movements = [
            TurnMovement(from_edge=i, to_edge=(i+1)%4, 
                        movement_type=MovementType.THROUGH)
            for i in range(4)
        ]
        roundabout = RoundaboutJunction(1, movements)
        
        # Set capacities first
        roundabout.calculate_capacity({i: 400 for i in range(4)})
        
        delays = roundabout.calculate_delay({i: 300 for i in range(4)})
        
        for delay in delays.values():
            assert delay > 0


# =============================================================================
# Test Class: Merge Model
# =============================================================================

class TestMergeModel:
    """Test merge capacity allocation"""
    
    def test_undersaturated_merge(self):
        """Both streams should pass when under capacity"""
        merge = MergeModel(
            mainline_capacity=2000,
            ramp_capacity=800,
            merge_type='parallel'
        )
        
        mainline_cap, ramp_cap = merge.calculate_merge_capacity(
            mainline_flow=1200,
            ramp_flow=600
        )
        
        assert mainline_cap == 1200
        assert ramp_cap == 600
    
    def test_oversaturated_merge_proportional(self):
        """Oversaturated merge should allocate proportionally"""
        merge = MergeModel(
            mainline_capacity=2000,
            ramp_capacity=800,
            merge_type='parallel'
        )
        
        mainline_cap, ramp_cap = merge.calculate_merge_capacity(
            mainline_flow=1800,  # Together = 2400 > 2000
            ramp_flow=600
        )
        
        # Total should equal downstream capacity
        assert mainline_cap + ramp_cap == pytest.approx(2000, rel=0.01)
        
        # Should be proportional to demand
        ratio = mainline_cap / ramp_cap
        expected_ratio = 1800 / 600
        assert ratio == pytest.approx(expected_ratio, rel=0.01)
    
    def test_zipper_merge_efficiency(self):
        """Zipper merge should achieve efficiency factor"""
        merge = MergeModel(mainline_capacity=2000, ramp_capacity=800)
        
        throughput = merge.calculate_zipper_capacity(
            flow_1=800,
            flow_2=800,
            efficiency=0.85
        )
        
        # Equal flows should achieve max efficiency
        # 2 × 800 × 0.85 = 1360
        expected = 2 * 800 * 0.85
        assert throughput == pytest.approx(expected, rel=0.05)


# =============================================================================
# Test Class: Diverge Model
# =============================================================================

class TestDivergeModel:
    """Test diverge capacity allocation"""
    
    def test_undersaturated_diverge(self):
        """All traffic should pass when under capacity"""
        diverge = DivergeModel(
            upstream_capacity=2000,
            mainline_lanes=2,
            exit_lanes=1
        )
        
        mainline_cap, exit_cap = diverge.calculate_diverge_capacity(
            mainline_demand=1200,
            exit_demand=400
        )
        
        assert mainline_cap == 1200
        assert exit_cap == 400
    
    def test_diverge_limited_by_upstream(self):
        """Total diverge output limited by upstream capacity"""
        diverge = DivergeModel(
            upstream_capacity=2000,
            mainline_lanes=2,
            exit_lanes=1
        )
        
        mainline_cap, exit_cap = diverge.calculate_diverge_capacity(
            mainline_demand=1800,
            exit_demand=600  # Total = 2400 > 2000
        )
        
        total = mainline_cap + exit_cap
        assert total == pytest.approx(2000, rel=0.01)


# =============================================================================
# Test Class: Level of Service
# =============================================================================

class TestLevelOfService:
    """Test LOS classification"""
    
    def test_signalized_los_thresholds(self, signalized_junction):
        """Verify LOS thresholds for signalized intersections"""
        # LOS A: ≤10s, B: ≤20s, C: ≤35s, D: ≤55s, E: ≤80s, F: >80s
        
        los = signalized_junction._determine_los(8)
        assert los == 'A'
        
        los = signalized_junction._determine_los(15)
        assert los == 'B'
        
        los = signalized_junction._determine_los(30)
        assert los == 'C'
        
        los = signalized_junction._determine_los(50)
        assert los == 'D'
        
        los = signalized_junction._determine_los(70)
        assert los == 'E'
        
        los = signalized_junction._determine_los(100)
        assert los == 'F'
    
    def test_capacity_result_includes_los(self, signalized_junction):
        """Capacity result should include LOS"""
        result = signalized_junction.calculate_capacity()
        
        assert result.level_of_service in ['A', 'B', 'C', 'D', 'E', 'F']


# =============================================================================
# Test Class: Conflict Detection
# =============================================================================

class TestConflictDetection:
    """Test conflict matrix and detection"""
    
    def test_conflicting_movements_identified(self, signalized_junction):
        """Conflicting movements should be detected"""
        # Left turn conflicts with opposing through
        movement_left = signalized_junction.movements[1]   # LEFT
        movement_through = signalized_junction.movements[2]  # THROUGH (different approach)
        
        # Check if they conflict
        idx_left = signalized_junction.movements.index(movement_left)
        idx_through = signalized_junction.movements.index(movement_through)
        
        conflicts = signalized_junction.conflicts_with(idx_left, idx_through)
        # They should conflict based on crossing paths
        # (depends on geometry implementation)
    
    def test_same_approach_no_conflict(self, signalized_junction):
        """Movements from same approach should not conflict"""
        # Movements 0 and 1 are from same approach (from_edge=0)
        conflicts = signalized_junction.conflicts_with(0, 1)
        
        assert conflicts == False


# =============================================================================
# Test Class: Analytical Formulas
# =============================================================================

class TestAnalyticalFormulas:
    """Test standalone analytical formulas"""
    
    def test_webster_optimal_cycle(self):
        """Webster's optimal cycle formula"""
        lost_time = 12  # seconds
        flow_ratios = [0.3, 0.25]  # Two critical movements
        
        C_opt = calculate_webster_optimal_cycle(lost_time, flow_ratios)
        
        # C = (1.5L + 5) / (1 - Y)
        Y = sum(flow_ratios)
        expected = (1.5 * lost_time + 5) / (1 - Y)
        
        # Should be clamped to practical limits
        expected = max(30, min(180, expected))
        
        assert C_opt == pytest.approx(expected, rel=0.01)
    
    def test_webster_oversaturated_returns_inf(self):
        """Webster's formula should return inf when Y >= 1"""
        lost_time = 12
        flow_ratios = [0.6, 0.5]  # Sum = 1.1 > 1
        
        C_opt = calculate_webster_optimal_cycle(lost_time, flow_ratios)
        
        assert C_opt == float('inf')
    
    def test_queue_estimation_undersaturated(self):
        """Queue estimation for undersaturated conditions"""
        avg_q, max_q = estimate_queue_at_signal(
            arrival_rate=0.3,  # veh/s
            capacity=0.5,
            cycle_length=90,
            green_ratio=0.5
        )
        
        # Both should be positive
        assert avg_q >= 0
        assert max_q >= 0
        # Max queue is arrivals during red: arrival_rate × red_time
        # Red time = 90 × 0.5 = 45s, so max ≈ 0.3 × 45 = 13.5
        assert max_q < 50  # Reasonable for these parameters
    
    def test_queue_estimation_oversaturated(self):
        """Queue estimation for oversaturated conditions"""
        avg_q, max_q = estimate_queue_at_signal(
            arrival_rate=0.6,  # veh/s (above capacity)
            capacity=0.5,
            cycle_length=90,
            green_ratio=0.5,
            analysis_period=900  # 15 minutes
        )
        
        # Queue should grow continuously
        assert max_q > 50


# =============================================================================
# Test Class: Movement Lookup
# =============================================================================

class TestMovementLookup:
    """Test movement lookup functionality"""
    
    def test_get_movement_by_edges(self, signalized_junction):
        """Should find movement by from/to edges"""
        movement = signalized_junction.get_movement_by_edges(0, 2)
        
        assert movement is not None
        assert movement.from_edge == 0
        assert movement.to_edge == 2
    
    def test_get_movement_by_edges_not_found(self, signalized_junction):
        """Should return None for non-existent movement"""
        movement = signalized_junction.get_movement_by_edges(99, 98)
        
        assert movement is None


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestJunctionEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_movement_junction(self):
        """Junction with single movement should work"""
        movements = [
            TurnMovement(from_edge=0, to_edge=1, 
                        movement_type=MovementType.THROUGH)
        ]
        phases = [SignalPhase(phase_id=0, duration=60, green_movements=[0])]
        timing = SignalTiming(phases=phases)
        
        junction = SignalizedJunction(1, movements, timing)
        capacity = junction.calculate_capacity()
        
        assert 0 in capacity.movement_capacities
        assert capacity.movement_capacities[0] > 0
    
    def test_zero_green_time(self):
        """Movement with zero green should have zero capacity"""
        movements = [
            TurnMovement(from_edge=0, to_edge=1,
                        movement_type=MovementType.THROUGH)
        ]
        # Movement not included in any phase
        phases = [SignalPhase(phase_id=0, duration=60, green_movements=[])]
        timing = SignalTiming(phases=phases)
        
        junction = SignalizedJunction(1, movements, timing)
        capacity = junction.calculate_capacity()
        
        # Capacity should be zero or very small
        assert capacity.movement_capacities[0] < 100  # Less than 100 veh/hr
    
    def test_very_short_cycle(self):
        """Very short cycle should still work"""
        movements = [
            TurnMovement(from_edge=0, to_edge=1,
                        movement_type=MovementType.THROUGH)
        ]
        phases = [SignalPhase(phase_id=0, duration=5, green_movements=[0],
                             yellow_time=2, all_red_time=1)]
        timing = SignalTiming(phases=phases)
        
        junction = SignalizedJunction(1, movements, timing)
        capacity = junction.calculate_capacity()
        
        # Should have some capacity, just reduced
        assert capacity.movement_capacities[0] > 0


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

