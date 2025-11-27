# End-to-End Example: Toy Network

Let's walk through a simple scenario to illustrate the simulation operation.

## Network Setup

Consider a T-shaped network: two incoming roads merging into one outgoing road at a junction J.

- **Edge 0**: Road from A -> J, length 500 m, capacity 40 vehicles. Free speed 20 m/s (~72 km/h)
- **Edge 1**: Road from B -> J, length 300 m, capacity 24 vehicles. Free speed 15 m/s (~54 km/h)
- **Edge 2**: Road from J -> C, length 400 m, capacity 32 vehicles. Free speed 20 m/s

Junction J is a merge (unsignalized for now).

## Demand

- At time 0, 10 vehicles depart from A towards C (via J). Group into 1 packet of size 10
- At time 0, also 5 vehicles depart from B towards C. Group into 1 packet of size 5

So two packets are released at t=0: Packet P_A of 10 veh on Edge0, Packet P_B of 5 veh on Edge1.

## Step-by-step Simulation

### Initialization

- Edge0.curr_count=10, Edge1.curr_count=5, Edge2.curr_count=0
- Two packets created:
  - Packet P_A on Edge0 (curr_edge=0, next_edge=2, remaining_time = 500/20 = 25s)
  - Packet P_B on Edge1 (curr_edge=1, next_edge=2, remaining_time = 300/15 = 20s)
- Both are in traveling state

### Time 0 to 1s

- **Layer1 (move_and_request)**: Both P_A and P_B decrement their remaining_time by 1. Now P_A.rem=24s, P_B.rem=19s. Neither finished edge, so they don't send any request. They remain traveling.
- **Layer2 (signals)**: No signal in this scenario, skip
- **Layer3 (edge processing)**: Edges receive no entry_request messages, so they do nothing
- **Layer4 (waiting packets)**: None in waiting state yet, nothing to do

### Time 1 to 19s

Each second, P_A and P_B continue moving, decrementing rem. By t=19s:
- P_A.rem = 25 - 19 = 6s left
- P_B.rem = 20 - 19 = 1s left

They haven't finished yet, no requests, edges still receiving none.

### Time 20s

- **Layer1**: P_A.rem goes from 6 to 5s. P_B.rem goes from 1 to 0s. Packet P_B now finishes Edge1 this step.
  - P_B sees next_edge = 2. It sends an entry_request with key=2, size=5, agent_id=P_B.id. Then P_B transitions to waiting state.
  - P_A is still traveling (5s remaining)

- **Layer3b (process_edge_requests)**: Edge2 agent processes messages for key=2. It finds one request from P_B of size=5.
  - Check Edge2's available_space: capacity 32 - curr_count 0 = 32 available. total_in=5 <= 32, so all can be accepted.
  - Edge2.curr_count becomes 5
  - Edge2 sends an entry_accept message with agent_id = P_B.id

- **Layer4 (waiting packets)**: P_B in waiting state reads the entry_accept message. It finds a match, so accepted=True.
  - P_B transitions to traveling state:
    - Set curr_edge = 2
    - next_edge = -1 (route complete)
    - remaining_time for Edge2 = length400 / speed20 = 20s
    - entry_time = current time (20s)
  - Edge1.curr_count set to 0 (vehicles departed)

**At end of t=20**: Edge0.curr_count=10, Edge1.curr_count=0, Edge2.curr_count=5. P_A (10 vehicles) on Edge0, P_B (5 vehicles) now on Edge2.

### Time 21 to 24s

- P_A continues traveling from 5s rem down to 1s rem by t=24
- P_B on Edge2 travels from 20s rem down to 16s rem by t=24
- Edge2.curr_count stays 5, Edge0 still 10

### Time 25s

- **Layer1**: P_A.rem goes from 1 to 0 – P_A finishes Edge0 this step. It sends entry_request (key=2, size=10, id=P_A.id) for Edge2 and enters waiting state. P_B.rem goes 16 -> 15 (still traveling on Edge2)

- **Layer3 (edge process)**: Edge2 now receives a request from P_A of size=10. Edge2.curr_count is currently 5. capacity 32, available 27. total_in=10 <= 27, so accepted fully.
  - Edge2.curr_count becomes 5+10 = 15
  - Edge2 sends entry_accept for P_A
  - Edge0.curr_count set to 0 (vehicles departed)

- **Layer4**: P_A in waiting reads acceptance, gets accepted.
  - P_A now travels on Edge2: curr_edge=2, next_edge=-1, remaining_time = 20s, entry_time=25
  - P_A state -> traveling

**End of t=25**: Edge0.curr_count=0, Edge1.curr_count=0, Edge2.curr_count=15 (5 from B + 10 from A). Packets: P_A (10 veh) and P_B (5 veh) both on Edge2 in traveling state.

### Time 26 to ~45s

Now on Edge2, we have two packets:
- P_B entered Edge2 at t=20, will finish at t=20+20=40
- P_A entered Edge2 at t=25, will finish at t=25+20=45

- **At t=40**: P_B finishes, sends no request (next_edge=-1, so it will mark DEAD). Edge2 reduces curr_count by 5. P_B is removed (end of route).

- **At t=45**: P_A finishes, also next_edge=-1, so P_A terminates (DEAD). Edge2 reduces curr_count by 10.

After t=45, simulation could end (no more agents).

## Outputs in this Example

### Travel Times

- **Edge0**: Packet P_A entered at 0, left at 25s, travel time 25s (free-flow was 25s, no delay)
- **Edge1**: Packet P_B entered at 0, left at 20s, travel time 20s (free-flow was 20s, no delay)
- **Edge2**: Packet P_B entered at 20, left at 40, travel time 20s. P_A entered at 25, left at 45, travel time 20s

### Queuing

There was essentially no wait because we allowed both flows through without conflict (we didn't enforce a merge delay, we just added them concurrently since capacity allowed it).

### Flows

Edge2 saw 5 vehicles enter at t=20 and 10 at t=25. We could output flows per minute or so.

### Emissions

If we compute with our approximate model:
- All were free-flow ~ 72 km/h on edge0 and 54 km/h on edge1, ~72 km/h on edge2
- Edge0: 0.5 km, 120 g/km, 10 veh = 600 g CO₂
- Edge1: 0.3 km, ~130 g/km, 5 veh ≈ 195 g
- Edge2: 0.4 km, 120 g/km, 15 veh = 720 g
- Total ~1515 g

This toy example shows how vehicles (packets) move through, queue at the node, and how the model outputs travel times and edge counts.

