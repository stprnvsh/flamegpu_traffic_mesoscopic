"""
Tests for FLAMEGPU2 Core Module

Tests cover:
- Agent configurations
- Message configurations
- Model building (without FLAMEGPU dependency)
- Network data structures
- Demand data structures
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# =============================================================================
# Test Class: Agent Configurations
# =============================================================================

class TestAgentConfigurations:
    """Test agent configuration dataclasses"""
    
    def test_edge_queue_config_defaults(self):
        """EdgeQueueConfig should have sensible defaults"""
        from core.agents import EdgeQueueConfig
        
        config = EdgeQueueConfig()
        
        assert config.max_edges == 1000  # Reduced for CUDA compatibility
        assert 'edge_id' in config.variables
        assert 'capacity' in config.variables
        assert 'curr_count' in config.variables
        assert 'length' in config.variables
        assert 'free_speed' in config.variables
        assert 'signal_id' in config.variables
        assert 'is_green' in config.variables
    
    def test_packet_config_defaults(self):
        """PacketConfig should have sensible defaults"""
        from core.agents import PacketConfig
        
        config = PacketConfig()
        
        assert config.max_route_length == 32
        assert 'size' in config.variables
        assert 'curr_edge' in config.variables
        assert 'next_edge' in config.variables
        assert 'remaining_time' in config.variables
        assert 'route' in config.array_variables
        assert config.array_variables['route'] == ('int', 32)
        assert 'traveling' in config.states
        assert 'waiting' in config.states
        assert config.initial_state == 'traveling'
    
    def test_signal_controller_config_defaults(self):
        """SignalControllerConfig should have sensible defaults"""
        from core.agents import SignalControllerConfig
        
        config = SignalControllerConfig()
        
        assert config.max_phases == 10
        assert config.max_edges_per_phase == 8
        assert 'node_id' in config.variables
        assert 'phase_index' in config.variables
        assert 'phase_count' in config.variables
        assert 'phase_durations' in config.array_variables
        assert 'phase_green_edges' in config.array_variables


# =============================================================================
# Test Class: Message Configurations
# =============================================================================

class TestMessageConfigurations:
    """Test message configuration"""
    
    def test_message_config_defaults(self):
        """MessageConfig should have sensible defaults"""
        from core.messages import MessageConfig
        
        config = MessageConfig()
        
        assert config.max_edges == 1000  # Reduced for CUDA compatibility


# =============================================================================
# Test Class: Model Configuration
# =============================================================================

class TestModelConfiguration:
    """Test model configuration"""
    
    def test_environment_config_defaults(self):
        """EnvironmentConfig should have sensible defaults"""
        from core.model import EnvironmentConfig
        
        config = EnvironmentConfig()
        
        assert config.time_step == 1.0
        assert config.max_edges == 1000  # Reduced for CUDA compatibility
        assert config.jam_density == 0.15
        assert config.wave_speed == 5.0
        assert config.fd_model in ['greenshields', 'newell_daganzo']
    
    def test_environment_config_tau_factors(self):
        """EnvironmentConfig should have SUMO-compatible TAU factors"""
        from core.model import EnvironmentConfig
        
        config = EnvironmentConfig()
        
        # SUMO default TAU factors
        assert config.tau_ff == pytest.approx(1.4, rel=0.1)
        assert config.tau_fj == pytest.approx(1.4, rel=0.1)
        assert config.tau_jf == pytest.approx(2.0, rel=0.1)
        assert config.tau_jj == pytest.approx(1.4, rel=0.1)
    
    def test_model_config_defaults(self):
        """ModelConfig should aggregate all component configs"""
        from core.model import ModelConfig
        
        config = ModelConfig()
        
        assert config.name == "MesoscopicTrafficModel"
        assert config.edge_config is not None
        assert config.packet_config is not None
        assert config.signal_config is not None
        assert config.message_config is not None
        assert config.environment_config is not None


# =============================================================================
# Test Class: Network Data
# =============================================================================

class TestNetworkData:
    """Test NetworkData structure"""
    
    def test_network_data_creation(self):
        """NetworkData should be created correctly"""
        from core.simulation import NetworkData
        
        network = NetworkData(
            edge_ids=['e1', 'e2'],
            edge_id_map={'e1': 0, 'e2': 1},
            edge_lengths=[100.0, 200.0],
            edge_speeds=[10.0, 20.0],
            edge_capacities=[15, 30],
            edge_lanes=[1, 2],
            edge_to_nodes=[0, 1],
            edge_signal_ids=[-1, 0],
            node_ids=['n1', 'n2'],
            node_id_map={'n1': 0, 'n2': 1},
            signals=[{'id': 's1', 'node_id': 0, 'phases': [], 'cycle_length': 60}],
        )
        
        assert network.num_edges == 2
        assert network.num_nodes == 2
        assert network.num_signals == 1
    
    def test_create_simple_network(self):
        """create_simple_network helper should work correctly"""
        from core.simulation import create_simple_network
        
        edges = [
            {'id': 'e1', 'length': 100.0, 'speed': 10.0, 'to_node': 'n2'},
            {'id': 'e2', 'length': 200.0, 'speed': 20.0, 'to_node': 'n3'},
        ]
        nodes = [
            {'id': 'n1'},
            {'id': 'n2'},
            {'id': 'n3'},
        ]
        
        network = create_simple_network(edges, nodes)
        
        assert network.num_edges == 2
        assert network.num_nodes == 3
        assert network.edge_lengths[0] == 100.0
        assert network.edge_speeds[1] == 20.0


# =============================================================================
# Test Class: Demand Data
# =============================================================================

class TestDemandData:
    """Test DemandData structure"""
    
    def test_demand_data_creation(self):
        """DemandData should be created correctly"""
        from core.simulation import DemandData
        
        demand = DemandData(
            departures=[
                (0.0, 'e1', ['e1', 'e2'], 10),
                (5.0, 'e1', ['e1', 'e2'], 5),
            ]
        )
        
        assert demand.num_departures == 2
        assert demand.total_vehicles == 15
    
    def test_create_simple_demand(self):
        """create_simple_demand helper should work correctly"""
        from core.simulation import create_simple_demand
        
        departures = [
            (0.0, 'e1', ['e1', 'e2'], 10),
            (10.0, 'e3', ['e3', 'e4'], 20),
        ]
        
        demand = create_simple_demand(departures)
        
        assert demand.num_departures == 2
        assert demand.total_vehicles == 30


# =============================================================================
# Test Class: Simulation Configuration
# =============================================================================

class TestSimulationConfiguration:
    """Test SimulationConfig"""
    
    def test_simulation_config_defaults(self):
        """SimulationConfig should have sensible defaults"""
        from core.simulation import SimulationConfig
        
        config = SimulationConfig()
        
        assert config.duration == 3600.0
        assert config.time_step == 1.0
        assert config.output_interval == 60.0
        assert config.verbose == True


# =============================================================================
# Test Class: Agent Function Code
# =============================================================================

class TestAgentFunctionCode:
    """Test that agent function code strings are valid"""
    
    def test_move_and_request_code_exists(self):
        """MOVE_AND_REQUEST_CODE should be defined"""
        from core.agents import MOVE_AND_REQUEST_CODE
        
        assert isinstance(MOVE_AND_REQUEST_CODE, str)
        assert 'FLAMEGPU_AGENT_FUNCTION' in MOVE_AND_REQUEST_CODE
        assert 'move_and_request' in MOVE_AND_REQUEST_CODE
    
    def test_wait_for_entry_code_exists(self):
        """WAIT_FOR_ENTRY_CODE should be defined"""
        from core.agents import WAIT_FOR_ENTRY_CODE
        
        assert isinstance(WAIT_FOR_ENTRY_CODE, str)
        assert 'FLAMEGPU_AGENT_FUNCTION' in WAIT_FOR_ENTRY_CODE
        assert 'wait_for_entry' in WAIT_FOR_ENTRY_CODE
    
    def test_process_edge_requests_code_exists(self):
        """PROCESS_EDGE_REQUESTS_CODE should be defined"""
        from core.agents import PROCESS_EDGE_REQUESTS_CODE
        
        assert isinstance(PROCESS_EDGE_REQUESTS_CODE, str)
        assert 'FLAMEGPU_AGENT_FUNCTION' in PROCESS_EDGE_REQUESTS_CODE
        assert 'process_edge_requests' in PROCESS_EDGE_REQUESTS_CODE
    
    def test_update_signal_code_exists(self):
        """UPDATE_SIGNAL_CODE should be defined"""
        from core.agents import UPDATE_SIGNAL_CODE
        
        assert isinstance(UPDATE_SIGNAL_CODE, str)
        assert 'FLAMEGPU_AGENT_FUNCTION' in UPDATE_SIGNAL_CODE
        assert 'update_signal' in UPDATE_SIGNAL_CODE
    
    def test_code_has_proper_return_statements(self):
        """All agent function code should return ALIVE or DEAD"""
        from core.agents import (
            MOVE_AND_REQUEST_CODE,
            WAIT_FOR_ENTRY_CODE,
            PROCESS_EDGE_REQUESTS_CODE,
            UPDATE_SIGNAL_CODE,
            SEND_DEPARTURE_CODE,
            PROCESS_DEPARTURES_CODE,
            UPDATE_GREEN_FLAG_CODE,
        )
        
        codes = [
            MOVE_AND_REQUEST_CODE,
            WAIT_FOR_ENTRY_CODE,
            PROCESS_EDGE_REQUESTS_CODE,
            UPDATE_SIGNAL_CODE,
            SEND_DEPARTURE_CODE,
            PROCESS_DEPARTURES_CODE,
            UPDATE_GREEN_FLAG_CODE,
        ]
        
        for code in codes:
            assert 'return flamegpu::ALIVE' in code or 'return flamegpu::DEAD' in code


# =============================================================================
# Test Class: Input Parsing
# =============================================================================

class TestInputParsing:
    """Test SUMO input parsing"""
    
    def test_sumo_edge_dataclass(self):
        """SUMOEdge should store edge properties"""
        from input.sumo_parser import SUMOEdge
        
        edge = SUMOEdge(
            id='e1',
            from_node='n1',
            to_node='n2',
            length=100.0,
            speed=13.89,
            lanes=2,
        )
        
        assert edge.id == 'e1'
        assert edge.length == 100.0
        assert edge.lanes == 2
    
    def test_sumo_node_dataclass(self):
        """SUMONode should store node properties"""
        from input.sumo_parser import SUMONode
        
        node = SUMONode(
            id='n1',
            x=100.0,
            y=200.0,
            type='traffic_light',
        )
        
        assert node.id == 'n1'
        assert node.type == 'traffic_light'
    
    def test_route_parser_grouping(self):
        """SUMORouteParser should group vehicles into packets"""
        from input.sumo_parser import SUMORouteParser
        
        parser = SUMORouteParser(grouping_window=5.0, max_packet_size=50)
        
        # Test with mixed routes - algorithm should handle interleaved departures
        parser.vehicles = [
            {'id': 'v1', 'depart': 0.0, 'route': ['e1', 'e2']},
            {'id': 'v2', 'depart': 1.0, 'route': ['e1', 'e2']},  # Same route, within window
            {'id': 'v3', 'depart': 2.0, 'route': ['e1', 'e2']},
            {'id': 'v4', 'depart': 10.0, 'route': ['e1', 'e2']},  # Different window
            {'id': 'v5', 'depart': 0.0, 'route': ['e3', 'e4']},  # Different route
        ]
        
        departures = parser._group_vehicles()
        
        # Should have 3 groups:
        # - (0s, e1-e2, 3 veh) - v1, v2, v3 grouped
        # - (0s, e3-e4, 1 veh) - v5 alone
        # - (10s, e1-e2, 1 veh) - v4 alone (outside 5s window)
        assert len(departures) == 3
        
        # Check each group (sorted by departure time)
        # Both routes start at t=0, so order between them may vary
        route_counts = {}
        for dep in departures:
            route_key = tuple(dep[2])
            if route_key not in route_counts:
                route_counts[route_key] = []
            route_counts[route_key].append((dep[0], dep[3]))
        
        # e1->e2 route should have 2 groups: (0s, 3 veh) and (10s, 1 veh)
        assert ('e1', 'e2') in route_counts
        e1_e2_groups = sorted(route_counts[('e1', 'e2')])
        assert e1_e2_groups == [(0.0, 3), (10.0, 1)]
        
        # e3->e4 route should have 1 group: (0s, 1 veh)
        assert ('e3', 'e4') in route_counts
        assert route_counts[('e3', 'e4')] == [(0.0, 1)]


# =============================================================================
# Test Class: Integration
# =============================================================================

class TestCoreIntegration:
    """Integration tests for core module"""
    
    def test_toy_network_creation(self):
        """Should be able to create the toy network from docs"""
        from core.simulation import create_simple_network, create_simple_demand
        
        # Toy network
        edges = [
            {'id': 'edge_0', 'length': 500.0, 'speed': 20.0, 'to_node': 'J1'},
            {'id': 'edge_1', 'length': 300.0, 'speed': 15.0, 'to_node': 'J1'},
            {'id': 'edge_2', 'length': 400.0, 'speed': 20.0, 'to_node': 'C'},
        ]
        nodes = [
            {'id': 'A'}, {'id': 'B'}, {'id': 'J1'}, {'id': 'C'}
        ]
        
        network = create_simple_network(edges, nodes)
        
        assert network.num_edges == 3
        assert network.num_nodes == 4
        
        # Check travel times
        tt_0 = network.edge_lengths[0] / network.edge_speeds[0]
        tt_1 = network.edge_lengths[1] / network.edge_speeds[1]
        tt_2 = network.edge_lengths[2] / network.edge_speeds[2]
        
        assert tt_0 == pytest.approx(25.0)  # 500/20
        assert tt_1 == pytest.approx(20.0)  # 300/15
        assert tt_2 == pytest.approx(20.0)  # 400/20
    
    def test_toy_demand_creation(self):
        """Should be able to create the toy demand from docs"""
        from core.simulation import create_simple_demand
        
        departures = [
            (0.0, 'edge_0', ['edge_0', 'edge_2'], 10),
            (0.0, 'edge_1', ['edge_1', 'edge_2'], 5),
        ]
        
        demand = create_simple_demand(departures)
        
        assert demand.num_departures == 2
        assert demand.total_vehicles == 15
    
    def test_model_config_composition(self):
        """Model config should properly compose all sub-configs"""
        from core.model import ModelConfig, EnvironmentConfig
        
        # Custom environment config
        env_config = EnvironmentConfig(
            time_step=0.5,
            jam_density=0.2,
        )
        
        config = ModelConfig(
            name="TestModel",
            environment_config=env_config,
        )
        
        assert config.name == "TestModel"
        assert config.environment_config.time_step == 0.5
        assert config.environment_config.jam_density == 0.2

