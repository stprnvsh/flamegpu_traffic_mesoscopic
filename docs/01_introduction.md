# GPU-Accelerated Mesoscopic Traffic Simulation with FLAMEGPU2

## Introduction

This document outlines how to implement a city-scale mesoscopic traffic simulation using the FLAMEGPU2 Python API. Our goal is to create a simulation comparable to SUMO's mesoscopic mode, where vehicles are grouped into packets and road links are treated as queues with capacity, enabling much faster-than-microscopic simulation speeds.

In mesoscopic modeling, vehicle groups ("packets") travel as single entities whose speed on each road (link) is governed by a speed–density (fundamental diagram) relationship. We describe an agent-based design for such a model, leveraging GPU parallelism to handle city-scale networks.

## Key Features

- **Packet agents** representing groups of vehicles (variable size based on demand), moving along routes
- **EdgeQueue agents** representing road segments (edges) as queues with finite capacity and fundamental diagram-based travel times
- **Junction/Signal agents** handling intersections, traffic signal phases, and conflict resolution
- **Discrete-time simulation loop** with message passing between agents for vehicle movements and queue updates
- **Data input** from SUMO `.net.xml` (network) and `.rou.xml` (routes/flows) files, converting them to initial agent states and vehicle injection schedules
- **Optional emission model** using average speeds of packets to estimate pollutants (e.g. CO₂)
- **Implementation details** for FLAMEGPU2's Python API: agent and message definitions (ModelDescription, AgentDescription, etc.), agent functions (with GPU device logic), and simulation execution
- **Toy example** demonstrating a simple network (3 edges, 2 junctions) from input to output
- **Output metrics** analogous to SUMO's: per-edge travel times, densities, flows, queue lengths, and optional emissions logged over time
- **GPU optimization techniques** (structure-of-arrays memory, parallel updates, minimizing divergence) and discussion of scalability to large networks

With these components, we can simulate urban traffic with high performance and reasonable fidelity. The following sections delve into each aspect in detail.

