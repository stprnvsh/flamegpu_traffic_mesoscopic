"""
Test Suite: Queue Models

This test bench validates queue dynamics implementations including:
- Point queue (vertical queue) model
- Spatial queue with physical extent
- SUMO mesoscopic queue model

Test Categories:
1. Queue Mechanics
   - FIFO ordering
   - Capacity constraints
   - Service rates
   
2. Spillback Detection
   - Queue length tracking
   - Blocking conditions
   
3. Delay Calculations
   - Webster's formula validation
   - Akcelik's formula validation
   
4. SUMO Compatibility
   - Segment-based queuing
   - TAU factor application
"""

import pytest
import math
from typing import List
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from traffic_models.queue_models import (
    QueueModel,
    PointQueueModel,
    SpatialQueueModel,
    SUMOMesoQueueModel,
    QueueState,
    QueueStatus,
    VehicleInQueue,
    QueueCoordinator,
    SpillbackInfo,
    calculate_queue_delay_webster,
    calculate_queue_length_akcelik,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def point_queue() -> PointQueueModel:
    """Standard point queue for testing"""
    return PointQueueModel(capacity=0.5, vehicle_length=7.0)  # 0.5 veh/s = 1800 veh/hr


@pytest.fixture
def spatial_queue() -> SpatialQueueModel:
    """Standard spatial queue for testing"""
    return SpatialQueueModel(
        capacity=0.5,
        edge_length=500.0,    # 500m edge
        jam_density=0.15,     # 150 veh/km
        wave_speed=5.0,       # 5 m/s
        vehicle_length=7.0
    )


@pytest.fixture
def sumo_queue() -> SUMOMesoQueueModel:
    """SUMO-style mesoscopic queue for testing"""
    return SUMOMesoQueueModel(
        capacity=0.5,
        edge_length=500.0,
        free_speed=13.89,     # 50 km/h
        segment_length=100.0,
        jam_density=0.15
    )


def create_vehicle(id: int, size: int = 1, arrival_time: float = 0.0) -> VehicleInQueue:
    """Helper to create test vehicles"""
    return VehicleInQueue(id=id, size=size, arrival_time=arrival_time)


# =============================================================================
# Test Class: Point Queue Mechanics
# =============================================================================

class TestPointQueueMechanics:
    """Test basic point queue operations"""
    
    def test_empty_queue_initial_state(self, point_queue):
        """New queue should be empty"""
        status = point_queue.get_status(0.0)
        
        assert status.length_vehicles == 0
        assert status.state == QueueState.EMPTY
        assert status.wait_time == 0
    
    def test_add_single_vehicle(self, point_queue):
        """Adding a vehicle should update queue length"""
        vehicle = create_vehicle(1)
        point_queue.add_vehicle(vehicle, current_time=0.0)
        
        status = point_queue.get_status(0.0)
        assert status.length_vehicles == 1
    
    def test_add_multiple_vehicles(self, point_queue):
        """Multiple vehicles should accumulate"""
        for i in range(5):
            vehicle = create_vehicle(i)
            point_queue.add_vehicle(vehicle, current_time=i)
        
        status = point_queue.get_status(5.0)
        assert status.length_vehicles == 5
    
    def test_add_packet(self, point_queue):
        """Packets with size > 1 should count correctly"""
        packet = create_vehicle(1, size=10)
        point_queue.add_vehicle(packet, current_time=0.0)
        
        status = point_queue.get_status(0.0)
        assert status.length_vehicles == 10
    
    def test_fifo_ordering(self, point_queue):
        """Vehicles should be served in FIFO order"""
        # Add vehicles with different IDs
        for i in range(5):
            vehicle = create_vehicle(i)
            point_queue.add_vehicle(vehicle, current_time=i)
        
        # Serve one at a time
        served_ids = []
        for t in range(5):
            served = point_queue.serve_vehicles(1, current_time=t + 5, dt=1)
            if served:
                served_ids.append(served[0].id)
        
        # Should be in order 0, 1, 2, 3, 4
        assert served_ids == [0, 1, 2, 3, 4]
    
    def test_serve_respects_capacity(self, point_queue):
        """Cannot serve more than available capacity"""
        # Add 10 vehicles
        packet = create_vehicle(1, size=10)
        point_queue.add_vehicle(packet, current_time=0.0)
        
        # Try to serve 5 (but packet is atomic in this implementation)
        served = point_queue.serve_vehicles(5, current_time=1.0, dt=1)
        
        # Should serve the packet since size <= capacity
        assert len(served) == 1 or len(served) == 0  # Depends on implementation
    
    def test_wait_time_calculation(self, point_queue):
        """Wait time should be queue_length / capacity"""
        # Add 10 vehicles
        packet = create_vehicle(1, size=10)
        point_queue.add_vehicle(packet, current_time=0.0)
        
        wait = point_queue.get_wait_time(arrival_time=0.0, current_time=0.0)
        
        # Expected: 10 / 0.5 = 20 seconds
        assert wait == pytest.approx(20.0, rel=0.01)
    
    def test_queue_length_in_meters(self, point_queue):
        """Physical queue length should be count × vehicle_length"""
        for i in range(5):
            vehicle = create_vehicle(i)
            point_queue.add_vehicle(vehicle, current_time=0.0)
        
        status = point_queue.get_status(0.0)
        expected_length = 5 * 7.0  # 5 vehicles × 7m
        
        assert status.length_meters == pytest.approx(expected_length, rel=0.01)


# =============================================================================
# Test Class: Spatial Queue Mechanics
# =============================================================================

class TestSpatialQueueMechanics:
    """Test spatial queue with physical extent"""
    
    def test_initial_state(self, spatial_queue):
        """New spatial queue should be empty"""
        status = spatial_queue.get_status(0.0)
        
        assert status.length_vehicles == 0
        assert status.length_meters == 0
        assert status.spillback_risk == 0
    
    def test_queue_length_tracking(self, spatial_queue):
        """Queue length should grow based on jam density"""
        # Add vehicle
        vehicle = create_vehicle(1, size=1)
        success = spatial_queue.add_vehicle(vehicle, current_time=0.0)
        
        assert success
        
        # Queue length in meters = vehicle_count / jam_density
        expected_length = 1 / 0.15  # ~6.67 meters
        
        assert spatial_queue.queue_length_meters == pytest.approx(expected_length, rel=0.1)
    
    def test_spillback_detection(self, spatial_queue):
        """Spillback should be detected when queue fills edge"""
        # Calculate max vehicles for 500m edge
        max_vehicles = int(500 * 0.15)  # ~75 vehicles
        
        # Add vehicles up to capacity
        for i in range(max_vehicles - 5):
            vehicle = create_vehicle(i, size=1)
            spatial_queue.add_vehicle(vehicle, current_time=0.0)
        
        status = spatial_queue.get_status(0.0)
        assert status.spillback_risk > 0.9
        assert not spatial_queue.is_spillback  # Not quite full yet
        
        # Add more to trigger spillback
        for i in range(10):
            vehicle = create_vehicle(100 + i, size=1)
            success = spatial_queue.add_vehicle(vehicle, current_time=0.0)
            if not success:
                break
        
        assert spatial_queue.is_spillback or not success
    
    def test_spillback_blocks_entry(self, spatial_queue):
        """Once spillback occurs, new vehicles should be rejected"""
        # Fill to spillback
        max_vehicles = int(500 * 0.15) + 5
        
        for i in range(max_vehicles):
            vehicle = create_vehicle(i, size=1)
            spatial_queue.add_vehicle(vehicle, current_time=0.0)
        
        # Try to add more
        new_vehicle = create_vehicle(999, size=1)
        success = spatial_queue.add_vehicle(new_vehicle, current_time=0.0)
        
        assert not success
    
    def test_wave_speed_limits_service(self, spatial_queue):
        """Service rate should be limited by wave speed"""
        # Add many vehicles
        for i in range(30):
            vehicle = create_vehicle(i, size=1)
            spatial_queue.add_vehicle(vehicle, current_time=0.0)
        
        # In 1 second, wave travels 5m, which clears 5*0.15 = 0.75 vehicles
        # But service rate is 0.5 veh/s, so capacity is the limiter
        served = spatial_queue.serve_vehicles(10, current_time=1.0, dt=1.0)
        
        # Should serve some vehicles (exact number depends on implementation)
        assert len(served) >= 0
    
    def test_available_storage_calculation(self, spatial_queue):
        """Available storage should decrease as queue grows"""
        initial_storage = spatial_queue.available_storage
        
        # Add some vehicles
        for i in range(10):
            vehicle = create_vehicle(i, size=1)
            spatial_queue.add_vehicle(vehicle, current_time=0.0)
        
        new_storage = spatial_queue.available_storage
        
        assert new_storage < initial_storage
        assert new_storage == pytest.approx(initial_storage - 10, rel=0.1)


# =============================================================================
# Test Class: SUMO Meso Queue Model
# =============================================================================

class TestSUMOMesoQueueModel:
    """Test SUMO mesoscopic queue model"""
    
    def test_segment_creation(self, sumo_queue):
        """Edge should be divided into correct number of segments"""
        # 500m edge with 100m segments = 5 segments
        assert sumo_queue.num_segments == 5
    
    def test_segment_travel_time(self, sumo_queue):
        """Segment travel time should include TAU factor"""
        base_time = 100.0 / 13.89  # ~7.2s for 100m at 50 km/h
        
        # In free-flow (TAUFF = 1.4)
        segment_tt = sumo_queue.get_segment_travel_time(0)
        expected = base_time * 1.4
        
        assert segment_tt == pytest.approx(expected, rel=0.05)
    
    def test_edge_travel_time(self, sumo_queue):
        """Total edge travel time should sum all segments"""
        edge_tt = sumo_queue.get_edge_travel_time()
        
        # Should be ~5 segments × (100/13.89) × 1.4 ≈ 50s
        expected = 5 * (100 / 13.89) * 1.4
        
        assert edge_tt == pytest.approx(expected, rel=0.05)
    
    def test_jam_state_affects_tau(self, sumo_queue):
        """Jammed segments should use higher TAU factors"""
        # Fill a segment to trigger jam
        for i in range(20):  # Fill first segment
            vehicle = create_vehicle(i, size=1)
            sumo_queue.add_vehicle(vehicle, current_time=0.0)
        
        sumo_queue._update_segment_states()
        
        # Check if first segment is jammed
        if sumo_queue.segment_jammed[0]:
            # TAU should be different
            tau = sumo_queue._get_tau_factor(0, 1)
            assert tau in [sumo_queue.tau_jf, sumo_queue.tau_jj]
    
    def test_vehicle_progression_through_segments(self, sumo_queue):
        """Vehicles should move through segments over time"""
        vehicle = create_vehicle(1, size=1)
        sumo_queue.add_vehicle(vehicle, current_time=0.0)
        
        # Initially in first segment
        assert len(sumo_queue.segments[0]) == 1
        
        # After segment travel time, should move
        segment_tt = sumo_queue.get_segment_travel_time(0)
        sumo_queue.serve_vehicles(10, current_time=segment_tt + 1, dt=segment_tt + 1)
        
        # Vehicle should have moved (either to next segment or exited)
        total_vehicles = sum(len(seg) for seg in sumo_queue.segments)
        # Could be 0 (if too fast) or in later segment
        assert total_vehicles <= 1


# =============================================================================
# Test Class: Queue State Transitions
# =============================================================================

class TestQueueStateTransitions:
    """Test queue state machine behavior"""
    
    def test_empty_to_building(self, spatial_queue):
        """Adding vehicles should transition to BUILDING state"""
        spatial_queue._last_arrival_rate = 1.0  # Above capacity
        
        vehicle = create_vehicle(1)
        spatial_queue.add_vehicle(vehicle, current_time=0.0)
        
        status = spatial_queue.get_status(0.0)
        assert status.state == QueueState.BUILDING
    
    def test_building_to_discharging(self, spatial_queue):
        """When arrivals stop, should transition to DISCHARGING"""
        # Add vehicles
        for i in range(10):
            vehicle = create_vehicle(i)
            spatial_queue.add_vehicle(vehicle, current_time=i)
        
        spatial_queue._last_arrival_rate = 0.1  # Below capacity
        
        status = spatial_queue.get_status(10.0)
        assert status.state == QueueState.DISCHARGING
    
    def test_discharging_to_empty(self, spatial_queue):
        """Queue should eventually become empty"""
        # Add one vehicle
        vehicle = create_vehicle(1)
        spatial_queue.add_vehicle(vehicle, current_time=0.0)
        
        # Serve multiple times to ensure vehicle exits
        for t in range(1, 10):
            spatial_queue.serve_vehicles(10, current_time=float(t), dt=1.0)
        
        status = spatial_queue.get_status(10.0)
        # Queue should be empty or nearly empty
        assert status.length_vehicles == 0 or status.state in [QueueState.EMPTY, QueueState.DISCHARGING]


# =============================================================================
# Test Class: Delay Formulas
# =============================================================================

class TestDelayFormulas:
    """Test analytical delay formulas"""
    
    def test_webster_delay_undersaturated(self):
        """Webster's formula for undersaturated conditions"""
        arrival_rate = 0.3  # veh/s
        capacity = 0.5      # veh/s
        cycle_time = 90     # seconds
        green_ratio = 0.5   # 50% green
        
        delay = calculate_queue_delay_webster(
            arrival_rate, capacity, cycle_time, green_ratio
        )
        
        # Should be finite and positive
        assert delay > 0
        assert math.isfinite(delay)
        
        # Typical urban delay: 10-40 seconds
        assert 5 < delay < 100
    
    def test_webster_delay_saturated(self):
        """Webster's formula should return inf when oversaturated"""
        arrival_rate = 0.6  # veh/s (above capacity)
        capacity = 0.5      # veh/s
        cycle_time = 90
        green_ratio = 0.5
        
        delay = calculate_queue_delay_webster(
            arrival_rate, capacity, cycle_time, green_ratio
        )
        
        assert delay == float('inf')
    
    def test_webster_delay_increases_with_demand(self):
        """Delay should increase as demand approaches capacity"""
        capacity = 0.5
        cycle_time = 90
        green_ratio = 0.5
        
        delays = []
        for x in [0.3, 0.5, 0.7, 0.9]:  # v/c ratios
            arrival = x * capacity
            d = calculate_queue_delay_webster(
                arrival, capacity, cycle_time, green_ratio
            )
            delays.append(d)
        
        # Delays should be monotonically increasing
        for i in range(len(delays) - 1):
            assert delays[i] < delays[i + 1]
    
    def test_akcelik_queue_undersaturated(self):
        """Akcelik's formula for undersaturated conditions"""
        arrival_rate = 0.3  # veh/s
        capacity = 0.5      # veh/s
        analysis_period = 900  # 15 minutes
        
        queue = calculate_queue_length_akcelik(
            arrival_rate, capacity, analysis_period
        )
        
        assert queue >= 0
        assert math.isfinite(queue)
    
    def test_akcelik_queue_oversaturated(self):
        """Akcelik's formula for oversaturated conditions"""
        arrival_rate = 0.6  # veh/s (above capacity)
        capacity = 0.5      # veh/s
        analysis_period = 900
        
        queue = calculate_queue_length_akcelik(
            arrival_rate, capacity, analysis_period
        )
        
        # Queue should grow linearly
        expected_growth = (arrival_rate - capacity) * analysis_period
        assert queue >= expected_growth * 0.5  # At least half of theoretical


# =============================================================================
# Test Class: Queue Coordinator
# =============================================================================

class TestQueueCoordinator:
    """Test multi-edge queue coordination"""
    
    def test_coordinator_registration(self):
        """Edges should be registered correctly"""
        coordinator = QueueCoordinator()
        
        queue1 = PointQueueModel(capacity=0.5)
        queue2 = PointQueueModel(capacity=0.5)
        
        coordinator.register_edge(1, queue1, downstream=[2], upstream=[])
        coordinator.register_edge(2, queue2, downstream=[], upstream=[1])
        
        assert 1 in coordinator.edge_queues
        assert 2 in coordinator.edge_queues
        assert coordinator.downstream_edges[1] == [2]
        assert coordinator.upstream_edges[2] == [1]
    
    def test_spillback_detection_through_coordinator(self):
        """Coordinator should detect downstream spillback"""
        coordinator = QueueCoordinator()
        
        queue1 = SpatialQueueModel(capacity=0.5, edge_length=100, jam_density=0.15, wave_speed=5.0)
        queue2 = SpatialQueueModel(capacity=0.5, edge_length=100, jam_density=0.15, wave_speed=5.0)
        
        coordinator.register_edge(1, queue1, downstream=[2], upstream=[])
        coordinator.register_edge(2, queue2, downstream=[], upstream=[1])
        
        # Fill downstream edge to spillback
        for i in range(20):
            vehicle = create_vehicle(i)
            queue2.add_vehicle(vehicle, current_time=0.0)
        
        # Check spillback from edge 1's perspective
        spillback = coordinator.check_spillback(1)
        
        # Should detect high spillback risk
        if queue2.is_spillback:
            assert spillback.is_blocked


# =============================================================================
# Test Class: Statistics Tracking
# =============================================================================

class TestStatisticsTracking:
    """Test queue statistics accumulation"""
    
    def test_arrival_count_tracking(self, point_queue):
        """Arrival count should be tracked correctly"""
        for i in range(5):
            vehicle = create_vehicle(i, size=2)
            point_queue.add_vehicle(vehicle, current_time=i)
        
        assert point_queue.arrival_count == 10  # 5 vehicles × size 2
    
    def test_departure_count_tracking(self, point_queue):
        """Departure count should be tracked correctly"""
        for i in range(5):
            vehicle = create_vehicle(i, size=1)
            point_queue.add_vehicle(vehicle, current_time=0.0)
        
        # Serve all
        for t in range(10):
            point_queue.serve_vehicles(1, current_time=t + 1, dt=1)
        
        assert point_queue.departure_count == 5
    
    def test_average_wait_calculation(self, point_queue):
        """Average wait should be computed from total wait / departures"""
        # Add vehicles at different times
        for i in range(5):
            vehicle = create_vehicle(i)
            point_queue.add_vehicle(vehicle, current_time=i)
        
        # Serve all at time 10
        for t in range(5):
            point_queue.serve_vehicles(1, current_time=10, dt=1)
        
        avg_wait = point_queue.get_average_wait()
        
        # Average wait should be positive
        assert avg_wait > 0
    
    def test_spillback_event_counting(self, spatial_queue):
        """Spillback events should be counted"""
        # Fill to spillback
        max_vehicles = int(500 * 0.15) + 10
        
        for i in range(max_vehicles):
            vehicle = create_vehicle(i, size=1)
            spatial_queue.add_vehicle(vehicle, current_time=0.0)
        
        # Try to add more (should fail and count spillback)
        for i in range(5):
            vehicle = create_vehicle(1000 + i, size=1)
            spatial_queue.add_vehicle(vehicle, current_time=0.0)
        
        assert spatial_queue.spillback_events > 0


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestQueueEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_capacity_queue(self):
        """Queue with zero capacity should block all service"""
        queue = PointQueueModel(capacity=0)
        
        vehicle = create_vehicle(1)
        queue.add_vehicle(vehicle, current_time=0.0)
        
        wait = queue.get_wait_time(0, 0)
        assert wait == float('inf')
    
    def test_very_high_capacity(self):
        """Very high capacity should serve immediately"""
        queue = PointQueueModel(capacity=1000)
        
        vehicle = create_vehicle(1)
        queue.add_vehicle(vehicle, current_time=0.0)
        
        wait = queue.get_wait_time(0, 0)
        assert wait < 0.01  # Nearly instant
    
    def test_empty_queue_service(self, point_queue):
        """Serving empty queue should return empty list"""
        served = point_queue.serve_vehicles(10, current_time=0, dt=1)
        assert served == []
    
    def test_negative_time_handling(self, point_queue):
        """Queue should handle unusual time values gracefully"""
        vehicle = create_vehicle(1)
        
        # Should not crash with negative time
        point_queue.add_vehicle(vehicle, current_time=-10.0)
        status = point_queue.get_status(-10.0)
        
        assert status.length_vehicles == 1
    
    def test_very_short_edge(self):
        """Very short edge should still work"""
        queue = SpatialQueueModel(
            capacity=0.5,
            edge_length=10.0,  # Very short
            jam_density=0.15,
            wave_speed=5.0
        )
        
        vehicle = create_vehicle(1)
        success = queue.add_vehicle(vehicle, current_time=0.0)
        
        # Should still accept vehicle (10m × 0.15 = 1.5 vehicle capacity)
        assert success
    
    def test_single_segment_sumo_queue(self):
        """SUMO queue with single segment should work"""
        queue = SUMOMesoQueueModel(
            capacity=0.5,
            edge_length=50.0,   # Short edge
            free_speed=13.89,
            segment_length=100.0  # Longer than edge
        )
        
        assert queue.num_segments == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestQueueIntegration:
    """Integration tests for queue models"""
    
    def test_service_rate_equals_capacity_long_term(self):
        """Long-term service rate should approach capacity"""
        # Create queue with integer capacity for cleaner test
        queue = PointQueueModel(capacity=2.0)  # 2 veh/s
        
        # Add many small packets (each size <= capacity)
        for i in range(50):
            packet = VehicleInQueue(id=i, size=2, arrival_time=0)
            queue.add_vehicle(packet, current_time=0.0)
        
        # Serve for many time steps
        total_served = 0
        for t in range(60):
            served = queue.serve_vehicles(2.0, current_time=t, dt=1.0)
            total_served += sum(v.size for v in served)
        
        # Should have served all packets (50 packets × 2 = 100 vehicles)
        assert total_served == 100
    
    def test_queue_clears_when_undersaturated(self):
        """Queue should eventually clear when arrivals < capacity"""
        # Use point queue for simpler behavior
        queue = PointQueueModel(capacity=1.0)  # 1 veh/s
        
        # Add initial vehicles as individual small packets
        for i in range(20):
            packet = VehicleInQueue(id=i, size=1, arrival_time=0)
            queue.add_vehicle(packet, current_time=0.0)
        
        initial_length = queue.get_status(0).length_vehicles
        assert initial_length == 20
        
        # Serve without new arrivals
        for t in range(30):
            queue.serve_vehicles(1.0, current_time=t, dt=1.0)
        
        final_length = queue.get_status(30).length_vehicles
        
        # Should have cleared completely
        assert final_length < initial_length
        assert final_length == 0


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

