# Emission Modeling (Optional)

Although not the core of traffic dynamics, we can extend the model to estimate emissions or fuel consumption on the fly, using the mesoscopic information.

## Average Speed Method

For each packet on an edge, determine its average speed = edge length / travel time. Then use a predetermined function to compute emissions.

### Piecewise Emission Values

- If average speed > 50 km/h (free-flow), assume moderate fuel consumption (e.g., 6 L/100km for a car, which corresponds to ~14000 g CO₂ per 100 km, or 140 g/km)
- If speed is very low (congested, < 10 km/h), consumption might be higher (say 20 L/100km equivalent due to idling, which is ~320 g/km)
- At intermediate speeds, it might be lowest (say 5 L/100km at ~30 km/h for some vehicles, ~116 g/km)

### Simple Curve

We could define a simple curve: \(CO2(g/km) = a + b/v + cv\) to capture higher emissions at both low and high speeds.

Alternatively, use a linear piecewise approximation:

- For \(v > 30\) km/h: CO₂ = 120 g/km + (v-30)*2 g/km (small increase with speed)
- For \(v < 30\) km/h: CO₂ = 120 g/km + (30-v)*4 g/km (higher increase as speed drops)

These numbers are just illustrative. The idea is emissions rise when speed deviates from optimal (30 km/h in this example).

## Integration in Simulation

We have two options:

### Calculate per step

Continuously update emission based on instantaneous speed. Might not be accurate to accumulate per step because packets either move at free flow or wait. We can assume when waiting, vehicle is idling (emitting some fixed rate).

### Calculate on edge exit

When a packet finishes an edge, we know how long it took (including any waiting). We can compute average speed and then multiply by distance to get g CO₂ for that packet on that edge. Summing those for all packets yields total emissions.

### Implementation Example

When Packet leaves an edge (in `move_and_request` function where `remaining_time <= 0`), before sending request or dying, compute emission:

```python
travel_time = pyflamegpu.environment.getPropertyFloat("current_time") - pyflamegpu.getVariableFloat("entry_time")
length = pyflamegpu.getVariableFloat("edge_length")  # if stored or get via env by curr_edge
avg_speed = (length / 1000.0) / (travel_time/3600.0)  # km/h

co2_per_km = computeCO2(avg_speed)
co2_emitted = co2_per_km * (length/1000.0) * pyflamegpu.getVariableInt("size")

# Add to environment counter for that edge:
env_co2 = pyflamegpu.environment.getMacroPropertyArrayFloat("edge_co2")
atomicAdd(env_co2[curr_edge], co2_emitted)
```

### Device Function for CO₂ Calculation

```python
@pyflamegpu.device_function
def computeCO2(speed_kmh: float) -> float:
    # return g/km as described
    if speed_kmh < 1: 
        return 300.0  # if effectively zero, idle high number
    if speed_kmh < 30:
        return 120 + (30 - speed_kmh) * 4
    else:
        return 120 + (speed_kmh - 30) * 2
```

## Summary

By tracking each packet's travel time on edges, we can estimate emissions per edge or per trip. This can be done either during the sim via additional agent logic or after by analyzing output travel times. For now, we include emissions as an optional metric that could be computed similarly to travel times.

Given that emission modeling can be quite complex, we treat it as an approximate extension. The key point is that with known vehicle count and speeds, computing emissions is straightforward in a post-processing sense. Our simulation provides average speeds and delays, which can feed standard emission models. For example, SUMO's emissions are based on the HBEFA model which essentially looks up emission rates by average speed for each vehicle type.

