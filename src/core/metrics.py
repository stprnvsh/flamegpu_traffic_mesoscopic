"""
Metrics Collection for FLAMEGPU2 Mesoscopic Traffic Simulation

Collects SUMO-style edgeData metrics at intervals:
- Per-edge per-interval data (like SUMO's edgeData output)
- Network-level summaries

Outputs to Parquet format for analysis.
"""

from dataclasses import dataclass
from typing import List, Optional, Any

try:
    import pyflamegpu
    FLAMEGPU_AVAILABLE = True
except ImportError:
    FLAMEGPU_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    aggregation_interval: float = 900.0
    output_file: str = "edge_data.parquet"
    verbose: bool = True


class IntervalEdgeDataCollector:
    """
    Collects per-edge per-interval data similar to SUMO's edgeData output.
    
    Must be called manually after each interval since FLAMEGPU2 host functions
    cannot iterate over individual agents.
    """
    
    def __init__(self, 
                 simulation,
                 model,
                 network_data,
                 output_interval: float = 900.0,
                 output_file: str = "edge_data.parquet"):
        """
        Args:
            simulation: CUDASimulation object
            model: MesoscopicTrafficModel
            network_data: NetworkData with edge info
            output_interval: Collection interval [s]
            output_file: Output parquet file
        """
        self.simulation = simulation
        self.model = model
        self.network_data = network_data
        self.output_interval = output_interval
        self.output_file = output_file
        
        self.interval_data = []  # Per-edge per-interval records
        self.network_data_list = []  # Network summaries
        self.last_collection_time = 0.0
        
    def collect_if_needed(self, current_time: float):
        """Check if collection is needed and collect if so"""
        if current_time - self.last_collection_time >= self.output_interval:
            self.collect(current_time)
            self.last_collection_time = current_time
    
    def collect(self, current_time: float):
        """Collect per-edge data for current interval
        
        Note: EdgeQueue agents now represent SEGMENTS. We aggregate by parent edge.
        - Only count 'entered' for first segment of each edge
        - Only count 'left' for last segment of each edge
        - Sum sampledSeconds across all segments of each edge
        """
        if not FLAMEGPU_AVAILABLE:
            return
        
        interval_begin = self.last_collection_time
        interval_duration = current_time - interval_begin
        if interval_duration <= 0:
            interval_duration = self.output_interval
        
        # Get segment population
        edge_desc = self.model.model.getAgent("EdgeQueue")
        edge_pop = pyflamegpu.AgentVector(edge_desc)
        self.simulation.getPopulationData(edge_pop)
        
        # Aggregate by edge: edge_idx -> {sampled, entered, left, ...}
        edge_aggregates = {}
        
        # Debug: count next_segment values
        _debug_last_count = 0
        _debug_non_last_count = 0
        
        for i in range(edge_pop.size()):
            agent = edge_pop[i]
            
            # Get segment info
            try:
                segment_idx_in_edge = agent.getVariableInt("segment_idx")
                edge_idx = agent.getVariableInt("edge_idx")
                next_segment = agent.getVariableInt("next_segment")
            except Exception as e:
                # Fallback: non-segment mode
                edge_idx = agent.getVariableInt("edge_id")
                segment_idx_in_edge = 0
                next_segment = -1
                if i == 0:  # Debug first segment
                    print(f"[DEBUG] Exception reading segment vars: {e}")
            
            # Debug counting
            if next_segment == -1:
                _debug_last_count += 1
            else:
                _debug_non_last_count += 1
            
            curr_count = agent.getVariableInt("curr_count")
            capacity = agent.getVariableInt("capacity")
            travel_time = agent.getVariableFloat("travel_time")
            lane_count = agent.getVariableInt("lane_count")
            length = agent.getVariableFloat("length")
            free_speed = agent.getVariableFloat("free_speed")
            
            # Read interval metrics
            try:
                sampled_seconds = agent.getVariableFloat("interval_sampled_seconds")
                entered = agent.getVariableInt("interval_entered")
                left = agent.getVariableInt("interval_left")
            except:
                sampled_seconds = 0.0
                entered = 0
                left = 0
            
            # Initialize edge aggregate if not exists
            if edge_idx not in edge_aggregates:
                edge_aggregates[edge_idx] = {
                    'sampled': 0.0,
                    'entered': 0,  # Only first segment contributes
                    'left': 0,      # Only last segment contributes
                    'length': 0.0,  # Sum of segment lengths
                    'lane_count': lane_count,
                    'free_speed': free_speed,
                    'travel_time_sum': 0.0,
                    'segment_count': 0,
                    'curr_count': 0,
                    'capacity': 0,
                }
            
            agg = edge_aggregates[edge_idx]
            agg['sampled'] += sampled_seconds
            agg['length'] += length
            agg['travel_time_sum'] += travel_time
            agg['segment_count'] += 1
            agg['curr_count'] += curr_count
            agg['capacity'] += capacity
            
            # Only count entered for first segment (segment_idx == 0)
            if segment_idx_in_edge == 0:
                agg['entered'] += entered
            
            # Only count left for last segment (next_segment == -1)
            if next_segment == -1:
                agg['left'] += left
        
        # Add spawn counts to entered (spawns bypass the entry_request flow)
        try:
            from .model import SpawnTracker
            spawn_tracker = SpawnTracker.get_instance()
            for edge_idx, spawn_count in spawn_tracker.spawn_counts_by_edge.items():
                if edge_idx in edge_aggregates:
                    edge_aggregates[edge_idx]['entered'] += spawn_count
        except ImportError:
            pass
        
        # Debug: print segment counts (first collection only)
        if interval_begin == 0:
            print(f"[DEBUG] Segment counts: last={_debug_last_count}, non-last={_debug_non_last_count}")
        
        # Network totals for summary
        total_sampled = 0.0
        total_entered = 0
        total_left = 0
        
        # Generate per-edge records from aggregates
        for edge_idx, agg in edge_aggregates.items():
            sampled_seconds = agg['sampled']
            entered = agg['entered']
            left = agg['left']
            length = agg['length']
            lane_count = agg['lane_count']
            free_speed = agg['free_speed']
            
            # Average travel time across segments
            travel_time = agg['travel_time_sum'] / agg['segment_count'] if agg['segment_count'] > 0 else 1.0
            
            # Calculate SUMO-style metrics
            length_km = length / 1000.0 if length > 0 else 0.001
            
            # Density = sampledSeconds / (interval_duration * length_km)
            density = sampled_seconds / (interval_duration * length_km) if (interval_duration > 0 and length_km > 0) else 0
            lane_density = density / lane_count if lane_count > 0 else density
            
            # Occupancy
            jam_density = 150.0
            max_veh_seconds = interval_duration * length_km * lane_count * jam_density
            occupancy = sampled_seconds / max_veh_seconds if max_veh_seconds > 0 else 0
            
            # Speed
            speed = length / travel_time if travel_time > 0 else free_speed
            speed_relative = speed / free_speed if free_speed > 0 else 1.0
            
            # Flow = (entered / interval_duration) * 3600
            flow = (entered / interval_duration) * 3600 if interval_duration > 0 else 0
            
            # Get original edge ID
            original_id = self.network_data.edge_ids[edge_idx] if edge_idx < len(self.network_data.edge_ids) else str(edge_idx)
            
            # Store per-edge per-interval record
            self.interval_data.append({
                'interval_begin': interval_begin,
                'interval_end': current_time,
                'id': original_id,
                'sampledSeconds': round(sampled_seconds, 2),
                'traveltime': round(travel_time, 2),
                'density': round(density, 4),
                'laneDensity': round(lane_density, 4),
                'occupancy': round(occupancy, 4),
                'speed': round(speed, 2),
                'speedRelative': round(speed_relative, 3),
                'entered': entered,
                'left': left,
                'flow': round(flow, 1),
            })
            
            # Accumulate totals
            total_sampled += sampled_seconds
            total_entered += entered
            total_left += left
        
        # Store network summary
        total_length_km = sum(self.network_data.edge_lengths) / 1000.0
        network_density = total_sampled / (interval_duration * total_length_km) if total_length_km > 0 else 0
        network_flow = (total_entered / interval_duration) * 3600 if interval_duration > 0 else 0
        
        self.network_data_list.append({
            'interval_begin': interval_begin,
            'interval_end': current_time,
            'sampledSeconds': round(total_sampled, 2),
            'entered': total_entered,
            'left': total_left,
            'density': round(network_density, 4),
            'flow': round(network_flow, 1),
        })
        
        print(f"[t={current_time/3600:.2f}h] Interval {interval_begin:.0f}-{current_time:.0f}s: "
              f"entered={total_entered}, left={total_left}, "
              f"sampledSec={total_sampled:.0f}, density={network_density:.4f} veh/km")
    
    def trigger_reset(self):
        """Set environment property to trigger counter reset"""
        # This is called by the simulation loop after collection
        pass  # Reset is handled via environment property in step function
    
    def save(self, path: str = None):
        """Save all data to parquet files"""
        if not PANDAS_AVAILABLE:
            print("pandas not available for Parquet export")
            return
        
        base_path = path or self.output_file
        base_name = base_path.replace('.parquet', '')
        
        # Save per-edge per-interval data (main output like SUMO edgeData)
        if self.interval_data:
            df = pd.DataFrame(self.interval_data)
            edge_path = f"{base_name}_edges.parquet"
            df.to_parquet(edge_path, index=False)
            print(f"Edge interval data saved to: {edge_path} ({len(df)} records, {df['interval_end'].nunique()} intervals)")
        
        # Save network summary
        if self.network_data_list:
            df_net = pd.DataFrame(self.network_data_list)
            network_path = f"{base_name}_network.parquet"
            df_net.to_parquet(network_path, index=False)
            print(f"Network summary saved to: {network_path}")
    
    def get_edge_dataframe(self):
        """Get per-edge per-interval data as DataFrame"""
        if PANDAS_AVAILABLE and self.interval_data:
            return pd.DataFrame(self.interval_data)
        return None
    
    def get_network_dataframe(self):
        """Get network summary as DataFrame"""
        if PANDAS_AVAILABLE and self.network_data_list:
            return pd.DataFrame(self.network_data_list)
        return None


def create_edge_data_function(
    output_interval: float = 900.0,
    edge_ids: List[str] = None,
    edge_lengths: List[float] = None,
    edge_lanes: List[int] = None,
    edge_speeds: List[float] = None,
    output_file: str = "edge_data.parquet"
):
    """
    Create a simple host function for console logging only.
    Per-edge data collection is done via IntervalEdgeDataCollector.
    """
    if not FLAMEGPU_AVAILABLE:
        return None
    
    class SimpleLoggingFunction(pyflamegpu.HostFunction):
        def __init__(self):
            super().__init__()
            self.last_log_time = 0.0
            self.total_length_km = sum(edge_lengths) / 1000.0 if edge_lengths else 1.0
            
        def run(self, host_api):
            current_time = host_api.environment.getPropertyFloat("current_time")
            
            # Clear reset flag from previous step
            host_api.environment.setPropertyInt("reset_interval_counters", 0)
            
            if current_time - self.last_log_time >= output_interval:
                # Just set the reset flag - actual collection done elsewhere
                host_api.environment.setPropertyInt("reset_interval_counters", 1)
                self.last_log_time = current_time
    
    return SimpleLoggingFunction()


def collect_edge_data_from_simulation(simulation, model, network_data, sim_duration: float = 86400.0) -> 'pd.DataFrame':
    """
    Collect final per-edge metrics after simulation completes.
    """
    if not PANDAS_AVAILABLE:
        return None
    
    if not FLAMEGPU_AVAILABLE:
        return None
    
    # Get edge population
    edge_desc = model.model.getAgent("EdgeQueue")
    edge_pop = pyflamegpu.AgentVector(edge_desc)
    simulation.getPopulationData(edge_pop)
    
    edge_records = []
    for i in range(edge_pop.size()):
        agent = edge_pop[i]
        edge_idx = agent.getVariableInt("edge_id")
        curr_count = agent.getVariableInt("curr_count")
        capacity = agent.getVariableInt("capacity")
        travel_time = agent.getVariableFloat("travel_time")
        lane_count = agent.getVariableInt("lane_count")
        length = agent.getVariableFloat("length")
        free_speed = agent.getVariableFloat("free_speed")
        
        try:
            sampled_seconds = agent.getVariableFloat("interval_sampled_seconds")
            entered = agent.getVariableInt("interval_entered")
            left = agent.getVariableInt("interval_left")
        except:
            sampled_seconds = curr_count * travel_time
            entered = 0
            left = 0
        
        length_km = length / 1000.0 if length > 0 else 0.001
        density = sampled_seconds / (sim_duration * length_km) if (sim_duration > 0 and length_km > 0) else 0
        lane_density = density / lane_count if lane_count > 0 else density
        jam_density = 150.0
        max_veh_seconds = sim_duration * length_km * lane_count * jam_density
        occupancy = sampled_seconds / max_veh_seconds if max_veh_seconds > 0 else 0
        speed = length / travel_time if travel_time > 0 else free_speed
        speed_relative = speed / free_speed if free_speed > 0 else 1.0
        flow = (entered / sim_duration) * 3600 if sim_duration > 0 else 0
        
        original_id = network_data.edge_ids[edge_idx] if edge_idx < len(network_data.edge_ids) else str(edge_idx)
        
        edge_records.append({
            'id': original_id,
            'sampledSeconds': round(sampled_seconds, 2),
            'traveltime': round(travel_time, 2),
            'density': round(density, 4),
            'laneDensity': round(lane_density, 4),
            'occupancy': round(occupancy, 4),
            'speed': round(speed, 2),
            'speedRelative': round(speed_relative, 3),
            'entered': entered,
            'left': left,
            'flow': round(flow, 1),
            'vehicleCount': curr_count,
            'capacity': capacity,
            'length': round(length, 2),
            'lanes': lane_count,
            'freeSpeed': round(free_speed, 2),
        })
    
    return pd.DataFrame(edge_records)
