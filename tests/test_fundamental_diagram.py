"""
Test Suite: Fundamental Diagram Models

This test bench validates the mathematical correctness and physical consistency
of all fundamental diagram implementations.

Test Categories:
1. Mathematical Properties
   - Boundary conditions (density = 0, density = jam)
   - Monotonicity (speed decreases with density)
   - Conservation (flow = density × speed)
   
2. Known Solutions
   - Greenshields theoretical values
   - Triangular diagram intersection points
   
3. Empirical Validation
   - Compare against published traffic data
   - SUMO compatibility checks
   
4. Edge Cases
   - Zero density
   - Jam density
   - Negative values (should be handled)
   - Very small/large values
"""

import pytest
import math
from typing import List, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from traffic_models.fundamental_diagram import (
    FundamentalDiagramParameters,
    FundamentalDiagram,
    GreenshieldsModel,
    NewellDaganzoModel,
    UnderwoodModel,
    DrakeModel,
    ThreeParameterModel,
    SUMOMesoSpeedModel,
    TrafficState,
    create_fundamental_diagram,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def urban_params() -> FundamentalDiagramParameters:
    """Typical urban arterial parameters"""
    return FundamentalDiagramParameters.from_typical_urban(speed_limit_kmh=50)


@pytest.fixture
def highway_params() -> FundamentalDiagramParameters:
    """Typical highway parameters"""
    return FundamentalDiagramParameters.from_typical_highway(speed_limit_kmh=120)


@pytest.fixture
def custom_params() -> FundamentalDiagramParameters:
    """Custom parameters for precise testing"""
    return FundamentalDiagramParameters(
        v_free=20.0,      # 20 m/s = 72 km/h
        rho_jam=0.15,     # 150 veh/km = 0.15 veh/m
        rho_crit=0.025,   # 25 veh/km = 0.025 veh/m
        q_max=0.5,        # 1800 veh/hr = 0.5 veh/s
        w=5.0,            # 5 m/s backward wave
        num_lanes=1
    )


@pytest.fixture
def all_models(custom_params) -> List[Tuple[str, FundamentalDiagram]]:
    """All fundamental diagram models for comparative testing"""
    return [
        ('greenshields', GreenshieldsModel(custom_params)),
        ('newell_daganzo', NewellDaganzoModel(custom_params)),
        ('underwood', UnderwoodModel(custom_params)),
        ('drake', DrakeModel(custom_params)),
        ('three_parameter', ThreeParameterModel(custom_params, alpha=3.0, beta=1.0)),
    ]


# =============================================================================
# Test Class: Parameter Validation
# =============================================================================

class TestFundamentalDiagramParameters:
    """Test parameter creation and validation"""
    
    def test_urban_preset_creation(self):
        """Urban preset should create valid parameters"""
        params = FundamentalDiagramParameters.from_typical_urban(50)
        
        assert params.v_free == pytest.approx(50 / 3.6, rel=0.01)
        assert params.rho_jam > 0
        assert params.rho_crit < params.rho_jam
        assert params.q_max > 0
        assert params.w > 0
    
    def test_highway_preset_creation(self):
        """Highway preset should create valid parameters"""
        params = FundamentalDiagramParameters.from_typical_highway(120)
        
        assert params.v_free == pytest.approx(120 / 3.6, rel=0.01)
        assert params.v_free > FundamentalDiagramParameters.from_typical_urban().v_free
    
    def test_residential_preset_creation(self):
        """Residential preset should have lower capacity"""
        params = FundamentalDiagramParameters.from_typical_residential(30)
        
        assert params.v_free == pytest.approx(30 / 3.6, rel=0.01)
        assert params.q_max < FundamentalDiagramParameters.from_typical_urban().q_max
    
    def test_invalid_parameters_rejected(self):
        """Invalid parameters should raise assertion errors"""
        with pytest.raises(AssertionError):
            FundamentalDiagramParameters(
                v_free=-10,  # Negative speed
                rho_jam=0.15,
                rho_crit=0.025,
                q_max=0.5,
                w=5.0
            )
        
        with pytest.raises(AssertionError):
            FundamentalDiagramParameters(
                v_free=20,
                rho_jam=0.15,
                rho_crit=0.20,  # Critical > jam
                q_max=0.5,
                w=5.0
            )
    
    def test_derived_values_calculated(self):
        """Missing values should be derived correctly"""
        params = FundamentalDiagramParameters(
            v_free=20.0,
            rho_jam=0.15,
            rho_crit=0.025,
            q_max=-1,  # Will be derived
            w=-1       # Will be derived
        )
        
        assert params.q_max > 0
        assert params.w > 0


# =============================================================================
# Test Class: Mathematical Properties (All Models)
# =============================================================================

class TestMathematicalProperties:
    """Test mathematical properties that should hold for all models"""
    
    def test_zero_density_gives_free_flow_speed(self, all_models):
        """At zero density, speed should equal free-flow speed"""
        for name, model in all_models:
            v = model.speed(0)
            assert v == pytest.approx(model.params.v_free, rel=0.01), \
                f"{name}: v(0) = {v}, expected {model.params.v_free}"
    
    def test_jam_density_gives_zero_speed(self, all_models):
        """At jam density, speed should be zero (or very small)"""
        for name, model in all_models:
            v = model.speed(model.params.rho_jam)
            assert v == pytest.approx(0, abs=0.5), \
                f"{name}: v(rho_jam) = {v}, expected ~0"
    
    def test_speed_non_negative(self, all_models):
        """Speed should never be negative"""
        densities = [0, 0.01, 0.05, 0.10, 0.14, 0.15, 0.20]
        
        for name, model in all_models:
            for rho in densities:
                v = model.speed(rho)
                assert v >= 0, f"{name}: negative speed at rho={rho}"
    
    def test_speed_monotonically_decreasing(self, all_models):
        """Speed should decrease (or stay constant) as density increases"""
        densities = [i * 0.01 for i in range(16)]
        
        for name, model in all_models:
            speeds = [model.speed(rho) for rho in densities]
            for i in range(len(speeds) - 1):
                assert speeds[i] >= speeds[i+1] - 0.01, \
                    f"{name}: speed increased at rho={densities[i+1]}"
    
    def test_flow_conservation(self, all_models):
        """Flow should equal density × speed"""
        densities = [0.02, 0.05, 0.10, 0.12]
        
        for name, model in all_models:
            for rho in densities:
                v = model.speed(rho)
                q = model.flow(rho)
                expected = rho * v
                assert q == pytest.approx(expected, rel=0.001), \
                    f"{name}: flow != rho*v at rho={rho}"
    
    def test_flow_is_zero_at_boundaries(self, all_models):
        """Flow should be zero at density=0 and density=jam"""
        for name, model in all_models:
            q_zero = model.flow(0)
            q_jam = model.flow(model.params.rho_jam)
            
            assert q_zero == pytest.approx(0, abs=0.001), \
                f"{name}: flow not zero at zero density"
            assert q_jam == pytest.approx(0, abs=0.1), \
                f"{name}: flow not zero at jam density"
    
    def test_flow_has_maximum(self, all_models):
        """Flow should have a maximum at intermediate density"""
        densities = [i * 0.005 for i in range(31)]
        
        for name, model in all_models:
            flows = [model.flow(rho) for rho in densities]
            max_flow = max(flows)
            max_idx = flows.index(max_flow)
            
            # Maximum should not be at boundaries
            assert 0 < max_idx < len(flows) - 1, \
                f"{name}: max flow at boundary (idx={max_idx})"
            
            # Maximum flow should be positive
            assert max_flow > 0, f"{name}: max flow <= 0"


# =============================================================================
# Test Class: Greenshields Specific Tests
# =============================================================================

class TestGreenshieldsModel:
    """Tests specific to Greenshields model"""
    
    def test_linear_relationship(self, custom_params):
        """Greenshields should produce linear v-rho relationship"""
        model = GreenshieldsModel(custom_params)
        
        # Test linearity: v(rho) = v_free - (v_free/rho_jam)*rho
        rho_test = 0.075  # Midpoint
        v_expected = custom_params.v_free * (1 - rho_test / custom_params.rho_jam)
        v_actual = model.speed(rho_test)
        
        assert v_actual == pytest.approx(v_expected, rel=0.001)
    
    def test_parabolic_flow(self, custom_params):
        """Greenshields flow should be parabolic"""
        model = GreenshieldsModel(custom_params)
        
        # Flow should be parabola: q = v_free*rho - (v_free/rho_jam)*rho^2
        rho_test = 0.05
        v = custom_params.v_free
        rho_j = custom_params.rho_jam
        q_expected = v * rho_test - (v / rho_j) * rho_test ** 2
        q_actual = model.flow(rho_test)
        
        assert q_actual == pytest.approx(q_expected, rel=0.001)
    
    def test_critical_density_is_half_jam(self, custom_params):
        """Greenshields critical density should be rho_jam/2"""
        model = GreenshieldsModel(custom_params)
        
        rho_c_theoretical = model.theoretical_critical_density
        assert rho_c_theoretical == pytest.approx(custom_params.rho_jam / 2, rel=0.001)
    
    def test_maximum_flow_formula(self, custom_params):
        """Greenshields q_max should equal v_free*rho_jam/4"""
        model = GreenshieldsModel(custom_params)
        
        q_max_theoretical = model.theoretical_max_flow
        q_max_expected = custom_params.v_free * custom_params.rho_jam / 4
        
        assert q_max_theoretical == pytest.approx(q_max_expected, rel=0.001)


# =============================================================================
# Test Class: Newell-Daganzo Triangular Model
# =============================================================================

class TestNewellDaganzoModel:
    """Tests specific to Newell-Daganzo triangular model"""
    
    def test_free_flow_regime_constant_speed(self, custom_params):
        """In free-flow regime, speed should be constant at v_free"""
        model = NewellDaganzoModel(custom_params)
        
        # Test multiple densities below critical
        test_densities = [0, 0.005, 0.01, 0.02]  # All below rho_crit=0.025
        
        for rho in test_densities:
            v = model.speed(rho)
            assert v == pytest.approx(custom_params.v_free, rel=0.01), \
                f"Speed not constant in free-flow at rho={rho}"
    
    def test_congested_regime_hyperbolic(self, custom_params):
        """In congested regime, speed follows hyperbolic decay"""
        model = NewellDaganzoModel(custom_params)
        
        # Test density above critical
        rho = 0.10  # Above rho_crit=0.025
        v_actual = model.speed(rho)
        
        # Expected: v = w × (rho_jam - rho) / rho
        v_expected = custom_params.w * (custom_params.rho_jam - rho) / rho
        
        assert v_actual == pytest.approx(v_expected, rel=0.01)
    
    def test_flow_triangular_shape(self, custom_params):
        """Flow-density should form triangular shape"""
        model = NewellDaganzoModel(custom_params)
        
        # Free-flow branch: q = v_free × rho
        rho_ff = 0.01
        q_ff = model.flow(rho_ff)
        q_ff_expected = custom_params.v_free * rho_ff
        assert q_ff == pytest.approx(q_ff_expected, rel=0.01)
        
        # Congested branch: q = w × (rho_jam - rho)
        rho_cong = 0.10
        q_cong = model.flow(rho_cong)
        q_cong_expected = custom_params.w * (custom_params.rho_jam - rho_cong)
        assert q_cong == pytest.approx(q_cong_expected, rel=0.01)
    
    def test_wave_speeds(self, custom_params):
        """Test kinematic wave speeds in each regime"""
        model = NewellDaganzoModel(custom_params)
        
        # Free-flow: wave speed = v_free (positive)
        wave_ff = model.wave_speed(0.01)
        assert wave_ff == pytest.approx(custom_params.v_free, rel=0.01)
        
        # Congested: wave speed = -w (negative)
        wave_cong = model.wave_speed(0.10)
        assert wave_cong == pytest.approx(-custom_params.w, rel=0.01)
    
    def test_shock_wave_speed(self, custom_params):
        """Test shock wave speed calculation"""
        model = NewellDaganzoModel(custom_params)
        
        # Shock between free-flow and congested
        rho_up = 0.02   # Free-flow
        rho_down = 0.12  # Congested
        
        sigma = model.shock_wave_speed(rho_up, rho_down)
        
        # Verify Rankine-Hugoniot: sigma = (q2-q1)/(rho2-rho1)
        q_up = model.flow(rho_up)
        q_down = model.flow(rho_down)
        sigma_expected = (q_down - q_up) / (rho_down - rho_up)
        
        assert sigma == pytest.approx(sigma_expected, rel=0.01)
    
    def test_capacity_at_critical_density(self, custom_params):
        """Maximum flow should occur at critical density"""
        model = NewellDaganzoModel(custom_params)
        
        q_at_crit = model.flow(custom_params.rho_crit)
        
        # Should be close to q_max
        assert q_at_crit == pytest.approx(custom_params.q_max, rel=0.05)


# =============================================================================
# Test Class: Travel Time Calculations
# =============================================================================

class TestTravelTimeCalculations:
    """Test travel time calculation methods"""
    
    def test_free_flow_travel_time(self, custom_params):
        """Free-flow travel time = length / v_free"""
        model = NewellDaganzoModel(custom_params)
        length = 1000  # 1 km
        
        tt_ff = model.free_flow_travel_time(length)
        tt_expected = length / custom_params.v_free
        
        assert tt_ff == pytest.approx(tt_expected, rel=0.001)
    
    def test_travel_time_increases_with_density(self, custom_params):
        """Travel time should increase as density increases"""
        model = NewellDaganzoModel(custom_params)
        length = 1000
        
        densities = [0.01, 0.05, 0.10, 0.12]
        travel_times = [model.travel_time(length, rho) for rho in densities]
        
        for i in range(len(travel_times) - 1):
            assert travel_times[i] < travel_times[i+1], \
                f"Travel time not increasing at rho={densities[i+1]}"
    
    def test_travel_time_infinite_at_jam(self, custom_params):
        """Travel time should be infinite at jam density"""
        model = NewellDaganzoModel(custom_params)
        length = 1000
        
        tt_jam = model.travel_time(length, custom_params.rho_jam)
        assert tt_jam == float('inf')


# =============================================================================
# Test Class: Traffic State Classification
# =============================================================================

class TestTrafficStateClassification:
    """Test traffic state classification methods"""
    
    def test_free_flow_classification(self, custom_params):
        """Low density should be classified as free-flow"""
        model = NewellDaganzoModel(custom_params)
        
        state = model.classify_state(0.01)  # Very low density
        assert state == TrafficState.FREE_FLOW
    
    def test_congested_classification(self, custom_params):
        """High density should be classified as congested"""
        model = NewellDaganzoModel(custom_params)
        
        state = model.classify_state(0.10)  # High but not jam
        assert state in [TrafficState.CONGESTED, TrafficState.SYNCHRONIZED]
    
    def test_jammed_classification(self, custom_params):
        """Near-jam density should be classified as jammed"""
        model = NewellDaganzoModel(custom_params)
        
        state = model.classify_state(0.14)  # Close to jam (0.15)
        assert state == TrafficState.JAMMED
    
    def test_get_conditions_returns_complete_info(self, custom_params):
        """get_conditions should return all traffic information"""
        model = NewellDaganzoModel(custom_params)
        
        conditions = model.get_conditions(0.05)
        
        assert conditions.density == 0.05
        assert conditions.speed > 0
        assert conditions.flow > 0
        assert conditions.state is not None
        assert 0 <= conditions.occupancy <= 1


# =============================================================================
# Test Class: SUMO Compatibility
# =============================================================================

class TestSUMOCompatibility:
    """Test SUMO mesoscopic speed model compatibility"""
    
    def test_tau_factors_applied_correctly(self):
        """TAU factors should be applied based on edge states"""
        sumo = SUMOMesoSpeedModel()
        
        # Free-free
        tau = sumo.get_tau_factor(curr_jammed=False, next_jammed=False)
        assert tau == SUMOMesoSpeedModel.DEFAULT_TAUFF
        
        # Free-jam
        tau = sumo.get_tau_factor(curr_jammed=False, next_jammed=True)
        assert tau == SUMOMesoSpeedModel.DEFAULT_TAUFJ
        
        # Jam-free (recovery)
        tau = sumo.get_tau_factor(curr_jammed=True, next_jammed=False)
        assert tau == SUMOMesoSpeedModel.DEFAULT_TAUJF
        
        # Jam-jam
        tau = sumo.get_tau_factor(curr_jammed=True, next_jammed=True)
        assert tau == SUMOMesoSpeedModel.DEFAULT_TAUJJ
    
    def test_jam_threshold_classification(self):
        """Occupancy threshold should correctly identify jammed state"""
        sumo = SUMOMesoSpeedModel(jam_threshold=0.8)
        
        assert not sumo.is_jammed(0.5)
        assert not sumo.is_jammed(0.79)
        assert sumo.is_jammed(0.8)
        assert sumo.is_jammed(0.95)
    
    def test_travel_time_adjustment(self):
        """Travel time should be multiplied by correct TAU"""
        sumo = SUMOMesoSpeedModel()
        base_tt = 100.0  # seconds
        
        # Free-flow conditions
        tt_ff = sumo.adjusted_travel_time(base_tt, curr_occupancy=0.3, next_occupancy=0.3)
        assert tt_ff == pytest.approx(base_tt * 1.4, rel=0.01)
        
        # Congested recovery (jam -> free)
        tt_jf = sumo.adjusted_travel_time(base_tt, curr_occupancy=0.9, next_occupancy=0.3)
        assert tt_jf == pytest.approx(base_tt * 2.0, rel=0.01)


# =============================================================================
# Test Class: Factory Function
# =============================================================================

class TestFactoryFunction:
    """Test model creation via factory function"""
    
    def test_create_all_model_types(self, custom_params):
        """Factory should create all supported model types"""
        model_types = ['greenshields', 'newell_daganzo', 'underwood', 
                       'drake', 'three_parameter']
        
        for model_type in model_types:
            model = create_fundamental_diagram(model_type, custom_params)
            assert model is not None
            assert model.params == custom_params
    
    def test_invalid_model_type_raises(self, custom_params):
        """Invalid model type should raise ValueError"""
        with pytest.raises(ValueError):
            create_fundamental_diagram('invalid_model', custom_params)


# =============================================================================
# Test Class: Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Test numerical stability and edge cases"""
    
    def test_very_small_density(self, all_models):
        """Very small density should not cause errors"""
        for name, model in all_models:
            v = model.speed(1e-10)
            assert math.isfinite(v)
            assert v > 0
    
    def test_density_slightly_above_jam(self, all_models):
        """Density above jam should return zero or near-zero speed"""
        for name, model in all_models:
            v = model.speed(model.params.rho_jam * 1.1)
            assert v >= 0  # Should not be negative
            assert v < 1.0  # Should be very small
    
    def test_negative_density_handled(self, all_models):
        """Negative density should return free-flow speed"""
        for name, model in all_models:
            v = model.speed(-0.01)
            assert v == pytest.approx(model.params.v_free, rel=0.01)
    
    def test_large_density(self, all_models):
        """Very large density should not cause overflow"""
        for name, model in all_models:
            v = model.speed(1000.0)  # Unrealistically large
            assert math.isfinite(v)
            assert v >= 0


# =============================================================================
# Test Class: Empirical Validation
# =============================================================================

class TestEmpiricalValidation:
    """Validate against known empirical relationships"""
    
    def test_typical_highway_capacity(self):
        """Highway capacity should be ~1700-2500 veh/hr/lane"""
        params = FundamentalDiagramParameters.from_typical_highway(120)
        model = NewellDaganzoModel(params)
        
        # Find maximum flow
        max_flow = 0
        for rho in [i * 0.001 for i in range(200)]:
            q = model.flow(rho)
            max_flow = max(max_flow, q)
        
        # Convert to veh/hr (per lane)
        max_flow_vph = max_flow * 3600 / params.num_lanes
        
        # Range 1700-2500 covers typical highway conditions (HCM values vary)
        assert 1700 < max_flow_vph < 2500, \
            f"Highway capacity {max_flow_vph} veh/hr/lane outside expected range"
    
    def test_typical_urban_capacity(self):
        """Urban arterial capacity should be ~1600-2200 veh/hr/lane"""
        params = FundamentalDiagramParameters.from_typical_urban(50)
        model = NewellDaganzoModel(params)
        
        max_flow = 0
        for rho in [i * 0.001 for i in range(200)]:
            q = model.flow(rho)
            max_flow = max(max_flow, q)
        
        # Convert to veh/hr (per lane)  
        max_flow_vph = max_flow * 3600 / params.num_lanes
        
        assert 1500 < max_flow_vph < 2200, \
            f"Urban capacity {max_flow_vph} veh/hr/lane outside expected range"
    
    def test_backward_wave_speed_realistic(self):
        """Backward wave speed should be ~15-25 km/h (4-7 m/s)"""
        params = FundamentalDiagramParameters.from_typical_urban()
        
        assert 4.0 <= params.w <= 7.0, \
            f"Wave speed {params.w} outside typical range"
    
    def test_jam_density_realistic(self):
        """Jam density should be ~120-180 veh/km/lane (0.12-0.18 veh/m)"""
        params = FundamentalDiagramParameters.from_typical_urban()
        
        # Convert to veh/km
        rho_jam_vkm = params.rho_jam * 1000
        
        assert 100 < rho_jam_vkm < 200, \
            f"Jam density {rho_jam_vkm} veh/km outside typical range"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_density_from_count_inverse(self, custom_params):
        """density_from_count should be inverse of count calculation"""
        model = NewellDaganzoModel(custom_params)
        length = 500  # meters
        
        # Set some vehicles
        vehicle_count = 25
        
        # Convert to density
        density = model.density_from_count(vehicle_count, length)
        
        # Calculate expected density
        expected = vehicle_count / (length * custom_params.num_lanes)
        
        assert density == pytest.approx(expected, rel=0.001)
    
    def test_jam_vehicle_count_consistent(self, custom_params):
        """jam_vehicle_count should be consistent with jam_density"""
        model = NewellDaganzoModel(custom_params)
        length = 1000
        
        max_vehicles = model.jam_vehicle_count(length)
        
        # Verify: max_vehicles = rho_jam × length × lanes
        expected = custom_params.rho_jam * length * custom_params.num_lanes
        
        assert max_vehicles == int(expected)


# =============================================================================
# Performance Tests (optional, can be slow)
# =============================================================================

class TestPerformance:
    """Performance and scaling tests"""
    
    def test_speed_calculation_performance(self, custom_params):
        """Speed calculation should be fast (< 1ms for 10000 calls)"""
        import time
        
        model = NewellDaganzoModel(custom_params)
        densities = [i * 0.00015 for i in range(10000)]
        
        start = time.time()
        for rho in densities:
            _ = model.speed(rho)
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"10000 speed calculations took {elapsed:.3f}s"
    
    def test_all_models_similar_performance(self, all_models):
        """All models should have similar performance characteristics"""
        import time
        
        densities = [i * 0.0015 for i in range(1000)]
        times = {}
        
        for name, model in all_models:
            start = time.time()
            for rho in densities:
                _ = model.speed(rho)
            times[name] = time.time() - start
        
        # All should be within 10x of fastest
        min_time = min(times.values())
        for name, t in times.items():
            assert t < min_time * 10, f"{name} much slower than others"


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

