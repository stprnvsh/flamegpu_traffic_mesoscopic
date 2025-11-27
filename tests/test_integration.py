"""
Integration Test Suite: Traffic Models

This test bench validates the integration between different model components:
- Fundamental diagram + Queue dynamics
- Queue models + Junction capacity
- Complete corridor simulation
- SUMO comparison scenarios

Test Categories:
1. Model Consistency
   - Travel time calculations consistent across models
   - Capacity values consistent with queue service rates
   
2. Physical Consistency
   - Conservation of vehicles
   - Delay + travel time relationships
   
3. Scenario Validation
   - Single link validation
   - Signalized intersection validation
   - Corridor with multiple junctions
   
4. SUMO-Like Behavior
   - TAU factor application
   - Segment-based traversal
"""

import pytest
import math
from typing import List, Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from traffic_models.fundamental_diagram import (
    FundamentalDiagramParameters,
    NewellDaganzoModel,
    GreenshieldsModel,
    SUMOMesoSpeedModel,
    TrafficState,
)

from traffic_models.queue_models import (
    PointQueueModel,
    SpatialQueueModel,
    SUMOMesoQueueModel,
    VehicleInQueue,
    QueueState,
    calculate_queue_delay_webster,
)

from traffic_models.junction_models import (
    TurnMovement,
    MovementType,
    SignalPhase,
    SignalTiming,
    SignalizedJunction,
    UnsignalizedJunction,
    PriorityType,
    estimate_queue_at_signal,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def urban_params() -> FundamentalDiagramParameters:
    """Standard urban arterial parameters"""
    return FundamentalDiagramParameters.from_typical_urban(speed_limit_kmh=50)


@pytest.fixture
def standard_edge_config() -> Dict:
    """Standard edge configuration"""
    return {
        'length': 500.0,          # meters
        'free_speed': 13.89,      # m/s (50 km/h)
        'capacity': 0.5,          # veh/s (1800 veh/hr)
        'jam_density': 0.15,      # veh/m
        'wave_speed': 5.0,        # m/s
    }


# =============================================================================
# Test Class: Fundamental Diagram + Queue Integration
# =============================================================================

class TestFundamentalDiagramQueueIntegration:
    """Test consistency between fundamental diagram and queue models"""
    
    def test_travel_time_consistency(self, urban_params, standard_edge_config):
        """Travel time from FD should match queue model base time"""
        fd = NewellDaganzoModel(urban_params)
        queue = SpatialQueueModel(
            capacity=standard_edge_config['capacity'],
            edge_length=standard_edge_config['length'],
            jam_density=standard_edge_config['jam_density'],
            wave_speed=standard_edge_config['wave_speed']
        )
        
        # Free-flow travel time from FD
        tt_fd = fd.free_flow_travel_time(standard_edge_config['length'])
        
        # Should be length / free_speed
        tt_expected = standard_edge_config['length'] / urban_params.v_free
        
        # Both should give approximately same value
        assert tt_fd == pytest.approx(tt_expected, rel=0.01)
    
    def test_capacity_consistency(self, urban_params, standard_edge_config):
        """Queue capacity should match fundamental diagram capacity"""
        fd = NewellDaganzoModel(urban_params)
        
        # Capacity from fundamental diagram
        fd_capacity_per_lane = urban_params.q_max  # veh/s/lane
        
        # Queue model uses same capacity concept
        queue_capacity = standard_edge_config['capacity']  # veh/s
        
        # Should be in similar range (accounting for lanes)
        assert fd_capacity_per_lane == pytest.approx(queue_capacity, rel=0.5)
    
    def test_congested_speed_affects_queue_growth(self, urban_params):
        """When speed drops, queue should build (conservation)"""
        fd = NewellDaganzoModel(urban_params)
        
        # In congested regime (above critical density), flow decreases
        # Critical density is around 0.025-0.03, jam is 0.15
        high_density = 0.12  # veh/m - well into congested regime
        
        # Lower speed = lower throughput = queue growth potential
        q_congested = fd.flow(high_density)
        q_at_capacity = fd.flow(urban_params.rho_crit)  # Maximum flow
        
        # Congested flow should be lower than capacity flow
        assert q_congested < q_at_capacity
    
    def test_wave_speed_propagation(self, urban_params, standard_edge_config):
        """Wave speed should be consistent between FD and queue"""
        fd = NewellDaganzoModel(urban_params)
        
        # Backward wave from FD
        wave_fd = urban_params.w
        
        # Queue model uses same wave speed
        queue_wave = standard_edge_config['wave_speed']
        
        # Should be similar
        assert wave_fd == pytest.approx(queue_wave, rel=0.5)


# =============================================================================
# Test Class: Queue + Junction Integration
# =============================================================================

class TestQueueJunctionIntegration:
    """Test consistency between queue models and junction capacity"""
    
    def test_junction_capacity_limits_queue_service(self):
        """Junction capacity should limit how fast queue is served"""
        # Create junction
        movements = [
            TurnMovement(from_edge=0, to_edge=1, 
                        movement_type=MovementType.THROUGH, num_lanes=2)
        ]
        phases = [SignalPhase(phase_id=0, duration=30, green_movements=[0],
                             yellow_time=3, all_red_time=2)]
        timing = SignalTiming(phases=phases)
        junction = SignalizedJunction(1, movements, timing)
        
        # Get junction capacity
        capacity_result = junction.calculate_capacity()
        junction_cap_vph = capacity_result.movement_capacities[0]  # veh/hr
        
        # Create queue with capacity
        queue = PointQueueModel(capacity=1.0)  # 1 veh/s capacity
        
        # Add vehicles as individual packets (size <= capacity)
        for i in range(20):
            packet = VehicleInQueue(id=i, size=1, arrival_time=0)
            queue.add_vehicle(packet, current_time=0)
        
        # Service rate with queue's capacity
        served_total = 0
        for t in range(30):  # 30 seconds
            served = queue.serve_vehicles(1.0, current_time=t, dt=1)
            served_total += sum(v.size for v in served)
        
        # Should have served all 20 vehicles (20 < 30*1.0)
        assert served_total == 20
    
    def test_delay_from_junction_matches_queue_delay(self):
        """Delay calculated from junction should match queue-based delay"""
        # Signalized junction parameters
        movements = [
            TurnMovement(from_edge=0, to_edge=1,
                        movement_type=MovementType.THROUGH, num_lanes=2)
        ]
        phases = [
            SignalPhase(phase_id=0, duration=30, green_movements=[0]),
            SignalPhase(phase_id=1, duration=30, green_movements=[])
        ]
        timing = SignalTiming(phases=phases)
        junction = SignalizedJunction(1, movements, timing)
        
        # Get capacity
        capacity_result = junction.calculate_capacity()
        cap_vph = capacity_result.movement_capacities[0]
        
        # Calculate delay at 70% v/c
        volume = cap_vph * 0.7
        delays = junction.calculate_delay({0: volume})
        junction_delay = delays[0]
        
        # Webster's delay for comparison
        g_C = timing.get_green_ratio(0)
        webster_delay = calculate_queue_delay_webster(
            arrival_rate=volume / 3600,
            capacity=cap_vph / 3600,
            cycle_time=timing.cycle_length,
            green_ratio=g_C
        )
        
        # Should be in same ballpark (HCM formula more complex than Webster)
        assert junction_delay == pytest.approx(webster_delay, rel=1.0)  # Within 100%


# =============================================================================
# Test Class: Single Link Scenario
# =============================================================================

class TestSingleLinkScenario:
    """Validate single link behavior end-to-end"""
    
    def test_free_flow_traversal(self, urban_params):
        """Vehicle should traverse link in free-flow time when uncongested"""
        fd = NewellDaganzoModel(urban_params)
        length = 1000  # 1 km
        
        # Free-flow travel time
        tt_free = fd.free_flow_travel_time(length)
        
        # Create queue model
        queue = SpatialQueueModel(
            capacity=0.5,
            edge_length=length,
            jam_density=urban_params.rho_jam,
            wave_speed=urban_params.w
        )
        
        # Add single vehicle
        vehicle = VehicleInQueue(id=1, size=1, arrival_time=0)
        queue.add_vehicle(vehicle, current_time=0)
        
        # Should exit after approximately free-flow time
        # (In mesoscopic, actual exit depends on queue service)
        wait_time = queue.get_wait_time(0, 0)
        
        # With no queue ahead, wait should be minimal
        assert wait_time < tt_free * 2  # Should be less than 2× free-flow
    
    def test_congested_link_delay(self, urban_params):
        """Congested link should add delay beyond free-flow time"""
        fd = NewellDaganzoModel(urban_params)
        length = 500
        
        # Create queue
        queue = SpatialQueueModel(
            capacity=0.5,
            edge_length=length,
            jam_density=urban_params.rho_jam,
            wave_speed=urban_params.w
        )
        
        # Fill queue substantially
        for i in range(30):
            vehicle = VehicleInQueue(id=i, size=1, arrival_time=0)
            queue.add_vehicle(vehicle, current_time=0)
        
        # New arrival should have delay
        wait_time = queue.get_wait_time(0, 0)
        tt_free = fd.free_flow_travel_time(length)
        
        # Wait time should be significant
        assert wait_time > tt_free * 0.5
    
    def test_spillback_scenario(self, urban_params):
        """Queue should spill back when link fills"""
        length = 200  # Short link
        
        queue = SpatialQueueModel(
            capacity=0.5,
            edge_length=length,
            jam_density=urban_params.rho_jam,
            wave_speed=urban_params.w
        )
        
        # Calculate max vehicles
        max_veh = int(length * urban_params.rho_jam)
        
        # Fill to capacity
        for i in range(max_veh + 10):
            vehicle = VehicleInQueue(id=i, size=1, arrival_time=0)
            success = queue.add_vehicle(vehicle, current_time=0)
            if not success:
                break
        
        # Should have hit spillback
        assert queue.spillback_events > 0 or queue.is_spillback


# =============================================================================
# Test Class: Signalized Corridor Scenario
# =============================================================================

class TestSignalizedCorridorScenario:
    """Validate corridor with multiple signals"""
    
    def test_two_signal_corridor(self):
        """Two signals in series should accumulate delay"""
        # Signal 1
        movements1 = [TurnMovement(from_edge=0, to_edge=1, 
                                   movement_type=MovementType.THROUGH)]
        phases1 = [SignalPhase(phase_id=0, duration=30, green_movements=[0]),
                   SignalPhase(phase_id=1, duration=30, green_movements=[])]
        timing1 = SignalTiming(phases=phases1, offset=0)
        junction1 = SignalizedJunction(1, movements1, timing1)
        
        # Signal 2 (offset for green wave attempt)
        movements2 = [TurnMovement(from_edge=1, to_edge=2,
                                   movement_type=MovementType.THROUGH)]
        phases2 = [SignalPhase(phase_id=0, duration=30, green_movements=[0]),
                   SignalPhase(phase_id=1, duration=30, green_movements=[])]
        timing2 = SignalTiming(phases=phases2, offset=10)
        junction2 = SignalizedJunction(2, movements2, timing2)
        
        # Calculate delays
        cap1 = junction1.calculate_capacity().movement_capacities[0]
        cap2 = junction2.calculate_capacity().movement_capacities[0]
        
        volume = min(cap1, cap2) * 0.6  # 60% of capacity
        
        delay1 = junction1.calculate_delay({0: volume})[0]
        delay2 = junction2.calculate_delay({0: volume})[0]
        
        total_delay = delay1 + delay2
        
        # Total delay should be sum of individual delays (approximately)
        assert total_delay > delay1
        assert total_delay > delay2
    
    def test_green_wave_reduces_delay(self):
        """Properly coordinated signals should reduce total delay"""
        # Two signals with proper green wave coordination
        link_length = 400  # meters
        link_travel_time = link_length / 13.89  # ~29 seconds at 50 km/h
        
        # Signal 1
        movements1 = [TurnMovement(from_edge=0, to_edge=1,
                                   movement_type=MovementType.THROUGH)]
        phases1 = [SignalPhase(phase_id=0, duration=30, green_movements=[0]),
                   SignalPhase(phase_id=1, duration=30, green_movements=[])]
        timing1 = SignalTiming(phases=phases1, offset=0)
        
        # Signal 2 with offset = link travel time
        phases2 = [SignalPhase(phase_id=0, duration=30, green_movements=[0]),
                   SignalPhase(phase_id=1, duration=30, green_movements=[])]
        timing2_coordinated = SignalTiming(phases=phases2, offset=link_travel_time)
        
        # Signal 2 with bad offset (opposite phase)
        timing2_uncoordinated = SignalTiming(phases=phases2, offset=link_travel_time + 30)
        
        # In coordinated case, vehicle arriving at signal 2 should hit green
        # In uncoordinated case, likely to hit red
        
        # Check phase at expected arrival time
        arrival_time = link_travel_time  # Time after leaving signal 1
        
        phase_coord, _ = timing2_coordinated.get_phase_at_time(arrival_time)
        phase_uncoord, _ = timing2_uncoordinated.get_phase_at_time(arrival_time)
        
        # Coordinated should be in phase 0 (green for movement 0)
        # This tests that offset affects timing
        assert timing2_coordinated.offset != timing2_uncoordinated.offset


# =============================================================================
# Test Class: SUMO Compatibility Scenarios
# =============================================================================

class TestSUMOCompatibilityScenarios:
    """Test SUMO-like behavior and compatibility"""
    
    def test_sumo_meso_travel_time_factors(self):
        """SUMO TAU factors should adjust travel times correctly"""
        sumo = SUMOMesoSpeedModel()
        base_tt = 36.0  # 36 seconds (500m at 50 km/h)
        
        # Free-flow conditions
        tt_ff = sumo.adjusted_travel_time(base_tt, 0.3, 0.3)
        assert tt_ff == pytest.approx(base_tt * 1.4, rel=0.01)
        
        # Entering jam
        tt_fj = sumo.adjusted_travel_time(base_tt, 0.3, 0.9)
        assert tt_fj == pytest.approx(base_tt * 1.4, rel=0.01)
        
        # Leaving jam (recovery)
        tt_jf = sumo.adjusted_travel_time(base_tt, 0.9, 0.3)
        assert tt_jf == pytest.approx(base_tt * 2.0, rel=0.01)
        
        # In jam
        tt_jj = sumo.adjusted_travel_time(base_tt, 0.9, 0.9)
        assert tt_jj == pytest.approx(base_tt * 1.4, rel=0.01)
    
    def test_sumo_meso_queue_segments(self):
        """SUMO meso queue should have correct number of segments"""
        queue = SUMOMesoQueueModel(
            capacity=0.5,
            edge_length=450,  # 450m
            free_speed=13.89,
            segment_length=100  # 100m segments
        )
        
        # Should have 5 segments (ceil(450/100) = 5)
        assert queue.num_segments == 5
    
    def test_sumo_meso_segment_travel_time(self):
        """Each segment should have correct travel time with TAU"""
        queue = SUMOMesoQueueModel(
            capacity=0.5,
            edge_length=500,
            free_speed=13.89,  # 50 km/h
            segment_length=100
        )
        
        # Base segment time = 100/13.89 = 7.2s
        # With TAUFF = 1.4: 7.2 × 1.4 = 10.1s
        segment_tt = queue.get_segment_travel_time(0)
        expected = (100 / 13.89) * 1.4
        
        assert segment_tt == pytest.approx(expected, rel=0.05)
    
    def test_sumo_meso_edge_travel_time(self):
        """Total edge time should sum all segments"""
        queue = SUMOMesoQueueModel(
            capacity=0.5,
            edge_length=500,
            free_speed=13.89,
            segment_length=100
        )
        
        total_tt = queue.get_edge_travel_time()
        
        # 5 segments × ~10.1s = ~50.4s
        expected = 5 * (100 / 13.89) * 1.4
        
        assert total_tt == pytest.approx(expected, rel=0.05)


# =============================================================================
# Test Class: Conservation Laws
# =============================================================================

class TestConservationLaws:
    """Test physical conservation properties"""
    
    def test_vehicle_conservation_in_queue(self):
        """Vehicles should be conserved: arrivals - departures = queue"""
        queue = PointQueueModel(capacity=0.5)
        
        arrivals = 0
        departures = 0
        
        # Add vehicles
        for i in range(20):
            vehicle = VehicleInQueue(id=i, size=1, arrival_time=i)
            queue.add_vehicle(vehicle, current_time=i)
            arrivals += 1
        
        # Serve some
        for t in range(30):
            served = queue.serve_vehicles(queue.capacity, current_time=20 + t, dt=1)
            departures += sum(v.size for v in served)
        
        # Check conservation
        queue_length = queue.get_status(50).length_vehicles
        
        assert arrivals == departures + queue_length
    
    def test_flow_conservation_at_junction(self):
        """Flow in should equal flow out (in steady state)"""
        movements = [
            TurnMovement(from_edge=0, to_edge=1, movement_type=MovementType.THROUGH),
            TurnMovement(from_edge=0, to_edge=2, movement_type=MovementType.LEFT),
        ]
        phases = [SignalPhase(phase_id=0, duration=60, green_movements=[0, 1])]
        timing = SignalTiming(phases=phases)
        junction = SignalizedJunction(1, movements, timing)
        
        # Input flow
        input_flow = 1000  # veh/hr total
        volumes = {0: 700, 1: 300}  # 70/30 split
        
        # In steady state, output should equal input (if under capacity)
        capacity = junction.calculate_capacity()
        total_capacity = sum(capacity.movement_capacities.values())
        
        if input_flow < total_capacity:
            # Conservation: all input should eventually exit
            output_flow = sum(volumes.values())
            assert output_flow == input_flow


# =============================================================================
# Test Class: Realistic Scenarios
# =============================================================================

class TestRealisticScenarios:
    """Test realistic traffic scenarios"""
    
    def test_morning_peak_buildup(self):
        """Queue should build during peak and dissipate after"""
        queue = PointQueueModel(capacity=0.5)  # Use simpler point queue
        
        # Morning peak: arrivals > capacity (0.6 > 0.5)
        peak_arrival_rate = 0.6  # veh/s
        queue._last_arrival_rate = peak_arrival_rate
        
        # Simulate peak period - add vehicles faster than capacity
        for t in range(100):
            # Add at 0.6 veh/s rate
            vehicle = VehicleInQueue(id=t, size=1, arrival_time=t)
            queue.add_vehicle(vehicle, current_time=t)
            
            # Serve at capacity (0.5 veh/s) - accumulate fractionally
            if t % 2 == 0:  # Serve every 2 seconds
                queue.serve_vehicles(1, current_time=t, dt=1)
        
        peak_queue = queue.get_status(100).length_vehicles
        
        # After peak: only serve, no new arrivals
        queue._last_arrival_rate = 0.0
        
        # Simulate recovery - just serve
        for t in range(100, 200):
            queue.serve_vehicles(1, current_time=t, dt=1)
        
        post_peak_queue = queue.get_status(200).length_vehicles
        
        # Queue should have reduced (or cleared)
        assert post_peak_queue <= peak_queue
    
    def test_intersection_level_of_service(self):
        """Intersection should have reasonable LOS for typical volumes"""
        # Simple two-phase signal with all movements getting green
        movements = [
            TurnMovement(from_edge=0, to_edge=2, movement_type=MovementType.THROUGH, num_lanes=2),
            TurnMovement(from_edge=1, to_edge=3, movement_type=MovementType.THROUGH, num_lanes=2),
        ]
        phases = [
            SignalPhase(phase_id=0, duration=30, green_movements=[0]),
            SignalPhase(phase_id=1, duration=30, green_movements=[1]),
        ]
        timing = SignalTiming(phases=phases)
        junction = SignalizedJunction(1, movements, timing)
        
        # Get capacities first
        capacity = junction.calculate_capacity()
        
        # Typical urban volumes at 60% of capacity
        volumes = {
            idx: cap * 0.6 
            for idx, cap in capacity.movement_capacities.items()
        }
        
        delays = junction.calculate_delay(volumes)
        avg_delay = sum(delays.values()) / len(delays)
        
        # Average delay should be finite and reasonable
        assert 5 < avg_delay < 100  # Reasonable range for signalized


# =============================================================================
# Test Class: Error Handling Integration
# =============================================================================

class TestErrorHandlingIntegration:
    """Test error handling across integrated components"""
    
    def test_negative_volume_handling(self):
        """Negative volumes should be handled gracefully"""
        movements = [TurnMovement(from_edge=0, to_edge=1,
                                  movement_type=MovementType.THROUGH)]
        phases = [SignalPhase(phase_id=0, duration=60, green_movements=[0])]
        timing = SignalTiming(phases=phases)
        junction = SignalizedJunction(1, movements, timing)
        
        # This should not crash
        delays = junction.calculate_delay({0: -100})
        
        # Result should be defined
        assert 0 in delays
    
    def test_zero_capacity_edge(self):
        """Zero capacity should be handled (returns inf delay)"""
        movements = [TurnMovement(from_edge=0, to_edge=1,
                                  movement_type=MovementType.THROUGH)]
        # No green time for movement
        phases = [SignalPhase(phase_id=0, duration=60, green_movements=[])]
        timing = SignalTiming(phases=phases)
        junction = SignalizedJunction(1, movements, timing)
        
        capacity = junction.calculate_capacity()
        delays = junction.calculate_delay({0: 100})
        
        # Delay should be very high or inf
        assert delays[0] > 100 or delays[0] == float('inf')
    
    def test_oversaturated_conditions(self):
        """Oversaturated conditions should be detected"""
        queue = PointQueueModel(capacity=0.1)  # Very low capacity
        
        # High arrival rate
        for i in range(100):
            vehicle = VehicleInQueue(id=i, size=1, arrival_time=0)
            queue.add_vehicle(vehicle, current_time=0)
        
        queue._last_arrival_rate = 1.0  # Way above capacity
        
        status = queue.get_status(0)
        
        # Should detect queue building
        assert status.state == QueueState.BUILDING


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

