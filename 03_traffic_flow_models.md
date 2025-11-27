# Governing Equations and Traffic Flow Models

Our mesoscopic model relies on macroscopic traffic flow relationships (the fundamental diagram) to determine speeds, flows, and travel times as a function of vehicle density.

## Speed–Density Relationship

### Greenshields Linear Model

We assume that as density (vehicles per km) on a link increases, the average speed declines. A simple choice is the Greenshields linear model:

\[
v(\rho) = v_{\text{free}} \left(1 - \frac{\rho}{\rho_{\text{jam}}}\right)
\]

Where:
- \(v(\rho)\): speed at density \(\rho\)
- \(v_{\text{free}}\): free-flow speed (e.g., 50 km/h)
- \(\rho_{\text{jam}}\): jam density (e.g., 150 vehicles/km)

Valid for:

\[
0 \leq \rho \leq \rho_{\text{jam}}
\]

### Triangular Fundamental Diagram (Newell-Daganzo Model)

Alternatively, one can use a piecewise-defined triangular fundamental diagram (Newell-Daganzo model), characterized by free-flow speed \(v_0\), wave (congestion back-propagation) speed \(w\), and capacity flow \(q_{\text{max}}\). This yields two linear branches:

\[
v(\rho) = \begin{cases}
v_0, & \text{if } \rho \leq \rho_c \\
w \left(\frac{\rho_{\text{jam}} - \rho}{\rho} \right), & \text{if } \rho > \rho_c
\end{cases}
\]

Where:
- \(v_0\): free-flow speed
- \(w\): congestion wave speed (negative)
- \(\rho_c = \frac{q_{\text{max}}}{v_0}\): critical density
- \(q_{\text{max}}\): max flow capacity (vehicles/sec)

## Capacity and Flow

### Edge Capacity

Each EdgeQueue agent has a capacity \(N_{\text{max}}\) (number of vehicles it can hold). This can be derived from jam density:

\[
N_{\text{max}} = \rho_{\text{jam}} \cdot (L \cdot \text{lanes})
\]

Where:
- \(L\): length of the edge in km
- \(\text{lanes}\): number of lanes on the edge

If \(N_{\text{max}}\) is exceeded, it means the edge is fully jammed – in simulation, we will prevent further entry of vehicles until space frees up.

### Flow Capacity

The capacity flow (vehicles per time step that can exit the edge) can be taken as:

\[
q_{\text{max}} = \rho_c \cdot v_0
\]

Typical example:

\[
q_{\text{max}} = 0.5 \text{ veh/sec per lane (or 1800 veh/hr)}
\]

## Queue Dynamics and Edge Travel Time

We conceptually split a road segment's behavior into free-travel and queuing portions. If an edge is uncongested (density below threshold, e.g. < 80% of capacity), vehicles traverse at roughly free-flow speed and do not experience delays. If an edge (or its downstream node) is congested/jammed, vehicles may have to queue – wait some time before they can enter the next segment.

### Free-Flow Travel Time

In our model, a packet's `remaining_time` initially is set to the free-flow travel time:

\[
T_{\text{free}} = \frac{L}{v_{\text{free}}}
\]

### Congestion Check

We classify an edge as "jammed" if its occupancy exceeds a threshold (similar to SUMO's mesoscopic model). Define congestion threshold:

\[
\rho_{\text{threshold}} = \alpha \cdot \rho_{\text{jam}}, \quad \alpha \in [0.7, 0.9]
\]

If:

\[
\rho > \rho_{\text{threshold}} \Rightarrow \text{edge is congested}
\]

For example, we might say if \(N_{\text{curr}}/N_{\text{max}} > 0.8\) (80% full), treat it as congested.

### Total Travel Time

If congestion occurs, the packet may end up waiting extra time at the end of the edge. We account for this by transitioning the packet to a WAITING state upon reaching the end, and it will accumulate waiting time until it can proceed:

\[
T_{\text{total}} = T_{\text{free}} + T_{\text{wait}}
\]

Where:
- \(T_{\text{wait}}\): waiting delay due to downstream blockage or signals

## Junction Delay and Conflict Resolution

Intersections are where conflicts occur between flows. Two or more incoming edges may compete to send packets into the same outgoing edge or across each other. We incorporate two primary mechanisms: traffic signals and unsignalized priority rules.

### Traffic Signals

If a junction is signal-controlled, vehicles on an approach can only proceed during green phases for that approach. In our model, a SignalController agent cycles through phases. When an incoming Packet reaches a red light, it will enter WAITING state at that junction until the light turns green.

Signal program can be defined as:

\[
\text{Phase}_i = (\text{duration}_i, \text{green edges}_i)
\]

- During red: packets wait at the junction
- During green: packets move, subject to capacity

### Priority Rules (Unsignalized)

Without signals, we need rules to decide which packet goes first when two packets arrive at the same time from different roads. A simple assumption: if two packets reach an unsignalized merge, they will alternate (first come first serve). We can implement this by using an ordering based on arrival time at the junction.

- FIFO rule at merging points
- Priority index assigned per incoming edge

### Merging Rules

At merges (multiple upstream edges converging to one downstream edge), capacity must be shared.

#### Proportional Merging

If two edges (1 and 2) feed into edge 3 with capacity \(C\), and \(N_1, N_2\) are waiting:

\[
C_1 = C \cdot \frac{N_1}{N_1 + N_2}, \quad C_2 = C \cdot \frac{N_2}{N_1 + N_2}
\]

#### Zipper Merge

Alternate between upstream edges:
- Odd steps: allow one packet from Edge 1
- Even steps: allow one packet from Edge 2

## Travel Time Updates

Each step, an EdgeQueue can update its `travel_time` estimate based on current occupancy. We can recompute:

\[
\rho = \frac{N_{\text{curr}}}{L \cdot \text{lanes}} \quad \Rightarrow \quad v(\rho), T = \frac{L}{v(\rho)}
\]

This updated \(T\) can be used for feedback routing or edge statistics. However, since in our simulation packets explicitly experience delays, the `travel_time` variable is more for output.

## Emissions Model (Optional)

We can estimate emissions (like CO₂, NOx, fuel consumption) based on the packet's speed or delays. One common approach is the average speed model: emissions are a function of average speed of a vehicle. For instance, CO₂ emission (g/km) is typically higher at very low speeds (due to idling) and at very high speeds (due to engine load), and lowest at moderate speeds.

### Speed-Based Approximation

\[
E_{\text{CO}_2}(v) = \begin{cases}
120 + (v - 30) \cdot 2, & v > 30 \\
120 + (30 - v) \cdot 4, & v < 30
\end{cases}
\]

Units: grams CO₂ per km

These numbers are just illustrative. The idea is emissions rise when speed deviates from optimal (30 km/h in this example). This can be extended to fuel consumption or NOx using calibrated coefficients.

## Additional Considerations

- **Stochastic delays**: Can add random perturbation to queue times or signal delays
- **Route choice feedback**: Use current edge travel times to reroute future packets (like SUMO DUA)
- **Multi-class traffic**: Trucks vs cars with different speeds and emission factors

