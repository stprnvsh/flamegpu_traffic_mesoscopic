# Output Design and Data Collection

The simulation should produce outputs comparable to SUMO's edge-based statistics.

## Edge-based Metrics

For each edge and for each time interval (e.g., every 60 seconds of simulation time), we want to output:

- **Flow**: number of vehicles (or packets' total vehicles) that passed the edge (either entered or exited) in that interval
- **Average speed or travel time**: average travel time of vehicles through that edge in that interval, or average speed (which is length/travel_time)
- **Density/occupancy**: maybe average number of vehicles on the edge during the interval, or end-of-interval queue length
- **Queue length**: maximum queue length (vehicles waiting) at the end of the edge, if applicable
- **Emissions**: total emissions (per pollutant) produced on that edge in the interval

## Data Collection Approach

We can gather these from our simulation by leveraging environment macro properties or logging:

### Accumulation Arrays

We can accumulate data continuously and then output every interval from a host function:

- **Travel times**: Have `edge_time_sum[edge]` and `edge_exit_count[edge]` that agent functions add to whenever a packet exits an edge. Every 60s, the host function divides to get average travel time = time_sum/count (if count>0). Then resets these counters to 0 for next interval.

- **Flow**: flow (veh per interval) is just `edge_exit_count` in that interval

- **Occupancy**: we could average `curr_count` over the interval. A simpler measure: just take `curr_count` at end of interval as a representative

- **Emissions**: maintain `edge_co2_sum[edge]` similar to travel time sum, add to it whenever vehicles traverse, then output per interval and reset

## Output Format

We format the output in CSV or XML similar to SUMO's `<edgeData>` output:

### CSV Format

```
time, edge_id, flow_veh, avg_speed_m_s, avg_travel_time_s, mean_density, CO2_grams
0, 0, 10, 13.9, 36, 0.3, 5000
0, 1, 5, 12.5, 24, 0.2, 3000
...
```

### XML Format

```xml
<edgeData interval="0-60">
  <edge id="edge0" flow="10" speed="13.9" traveltime="36" occupancy="0.3" CO2="5000"/>
  <edge id="edge1" flow="5" speed="12.5" traveltime="24" occupancy="0.2" CO2="3000"/>
  ...
</edgeData>
```

## Logging Implementation

We can use a host step function at multiples of the interval:

```python
interval = 60  # seconds

@pyflamegpu.host_function
def log_metrics(sim):
    current_time = sim.getStepCounter() * sim.getSimulationConfig().time_step
    
    if current_time % interval < 1e-6:  # if exactly a multiple of 60
        # Gather data
        edge_time = sim.getEnvironmentPropertyArrayFloat("edge_time_sum")
        edge_count = sim.getEnvironmentPropertyArrayFloat("edge_exit_count")
        edge_co2 = sim.getEnvironmentPropertyArrayFloat("edge_co2_sum")
        
        for e in range(num_edges):
            if edge_count[e] > 0:
                avg_time = edge_time[e] / edge_count[e]
                avg_speed = edge_length[e] / avg_time  # in m/s if length in m, time in s
            else:
                avg_time = 0
                avg_speed = edge_speed[e]  # or 0 if no data
            
            flow = edge_count[e]  # vehicles in this interval
            co2 = edge_co2[e]
            
            # Write to file or console
            print(f"{int(current_time)},{e},{flow},{avg_time:.2f},{avg_speed:.2f},{co2:.1f}")
        
        # Reset counters for next interval
        for e in range(num_edges):
            edge_time[e] = 0
            edge_count[e] = 0
            edge_co2[e] = 0
```

## Alternative Approaches

- **FLAMEGPU2 Logging API**: FLAMEGPU2 has a logging API where you can specify agent variables to log at given steps. However, that might be a lot of data for a city-scale model (thousands of edges * steps). More efficient is to aggregate within the simulation and output summary.

- **Post-processing**: We might simply let simulation run and at the end iterate through logged data. But doing it in real-time allows writing incremental output.

- **Agent logging**: FLAMEGPU2 can log agent variables each step or at end via LoggingConfig. But since we want per-edge values, not aggregated, we might skip built-in logging and do custom as above.

