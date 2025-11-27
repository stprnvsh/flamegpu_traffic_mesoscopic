# Scaling and Performance Considerations

When scaling this mesoscopic simulation up to a full city with potentially thousands of intersections and edges and heavy traffic demand, here are important considerations and how our implementation addresses them.

## Scalability of Agents

The number of EdgeQueue agents equals the number of edges in the network. A large city might have on the order of 10^4 – 10^5 edges (for every road segment). 10^5 agents is within the capability of modern GPUs (for reference, FLAMEGPU can handle millions of agents).

The number of Packet agents varies with demand; at peak rush hour, if e.g. 50,000 vehicles on the network and we group them into ~10,000 packets, that's also manageable. The simulation workload scales roughly with total packets processed over time. If 1 million vehicle trips occur in a day, and we group 5 per packet, that's 200k packets to simulate through the network. The GPU can likely handle this, especially since not all exist at once but spawn and die over time.

## Time Step and Real-Time Simulation

We used a fixed 1-second step. This yields fine resolution for urban traffic signals (which usually have cycle times of ~30-120s). If needed, we could reduce to 0.5s for better resolution of queue formation (but doubling steps doubles computation). In mesoscopic models, 1s is typically acceptable. With GPU, even 0.5s might be fine if needed (just more steps).

## Accuracy vs Aggregation

Grouping vehicles sacrifices some detail: e.g., within a packet, vehicles are assumed to behave uniformly. If a packet is too large spanning a long time gap, it might not capture variation (like the first vehicles might clear a signal while last ones get stuck at red – our model would treat them all as one unit that either all clear or all wait).

**Mitigation:** Grouping should consider traffic signals: typically vehicles arriving in one green phase form a packet, and those arriving in the next green form another. This way, the packet doesn't straddle a red light. We can refine grouping rules accordingly using the schedule of signals or maximum packet length (time-wise).

## Memory Limitations

GPU memory could be a limiting factor if we naively store too much. But as analyzed, our memory usage is modest. We should ensure not to store large unused arrays. For example, our route array of length 32 might be mostly empty for short trips – but it still occupies memory. If that's concern, we could compress route storage by storing route in a global array and just indices.

Another memory hog could be messaging overhead if not careful, but FLAMEGPU2 manages message memory efficiently internally.

## Load Balancing

In a city, some areas may have many agents (vehicles in dense downtown), others sparse. But since GPU executes in parallel across all, it's fine. However, if a large fraction of Packet agents are waiting (doing minimal work) while others are traveling (doing more work), warps could see some underutilization. But typically, waiting state work is lighter, not heavier, so it's okay.

If needed, one can separate Packet agent into two agent types (TravelingPacket vs WaitingPacket) to schedule differently. But since state accomplishes similar, and overhead of a few idle threads is negligible, we likely don't need that.

## Multi-GPU and Distributed Simulation

If one GPU isn't enough (for extremely large networks or higher fidelity requiring smaller packets and steps), one could consider splitting the network into regions each on a GPU. FLAMEGPU2 doesn't natively simulate one model over multiple GPUs except via ensembles (which are independent). However, one could run two coupled simulations with minimal interaction if the city can be partitioned (with vehicles seldom crossing partitions). That's complex and beyond scope.

Realistically, one modern GPU (e.g., NVIDIA RTX/Quadro) can handle a big chunk. For research, scaling beyond one GPU might not be necessary unless simulating entire countries or so.

## Performance Metrics

We expect significant speedup over microscopic simulation:

SUMO meso is reported up to 100x faster than micro. Our GPU-based approach could further speed up because of parallelism. Possibly a city simulation that might run at, say, 10x real-time on CPU could run 100x or more on GPU. The actual factor depends on GPU utilization and overhead.

The advantage is especially high when many agents can be processed in parallel (peak traffic). In low traffic (few agents), GPU overhead might dominate, making it less efficient than a simple CPU loop. But for city-scale, we target heavy load scenarios where GPU shines.

## Validation

As we scale, we should validate that macroscopic characteristics (like fundamental diagram outputs) match expectations. We could simulate a single road with increasing traffic to reproduce a fundamental diagram curve, adjusting our model parameters (jam threshold, etc.) to calibrate. This ensures the model yields realistic speeds and flows at various densities.

Because we used a simple linear speed-density, it might not perfectly match empirical data, but it can be tuned or replaced with a better formula if needed.

## Edge Cases

### Extremely Short Edges

Extremely short edges (few meters) might cause our jam threshold logic to classify them incorrectly (e.g., an edge of 5m with 1 vehicle is 100% occupied). SUMO meso had an issue with short edges always appearing jammed because occupancy is binary. We need to be cautious: maybe handle edges shorter than say 50m differently (e.g., treat them as part of junction). Or ensure jam threshold accounts for length.

### Traffic Lights with Very Short Cycle Times

If step=1s, we should be fine, but if some phase is 2s, that is still okay.

### Agent Deletion and Creation Overhead

If we spawn many small packets rapidly, the overhead of constantly creating/destroying might impact performance slightly. FLAMEGPU2 is designed for dynamic agents but it's something to monitor. If it's an issue, one might pool agents (recycle them). However, likely fine given their approach.

## Extendability

Our design can incorporate more complexities if needed:

- **Multi-lane dynamics or partial overtaking**: Some meso models allow faster vehicles to bypass slower ones within the queue if space. We could allow overtaking by e.g. splitting queues per lane or by adding a probability that a packet can pass another if gap.

- **Integration with route choice**: If one wanted to do dynamic traffic assignment, one could feed back edge travel times to a route choice module outside simulation or use FLAMEGPU ensemble to evaluate scenarios.

- **Mixed traffic (trucks vs cars)**: Just incorporate vehicle type into Packet (maybe split into separate pack types or include an average factor).

All these would increase agent count or complexity, but the GPU likely can handle moderate increases.

## Conclusion

By using FLAMEGPU2 and carefully structuring the simulation, we achieve a highly parallel mesoscopic traffic model. It captures the essential dynamics (queues, delays, flows) while remaining computationally efficient. The agent-based approach is flexible to add features or adjust granularity. As tested on the small example and reasoned through performance analysis, the model should scale to large networks and high traffic volumes, enabling city planners or researchers to simulate scenarios (like city-wide signal timing changes or congestion pricing effects) faster than real time for decision support.

