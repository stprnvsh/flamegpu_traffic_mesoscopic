"""
FLAMEGPU2 Core Module for Mesoscopic Traffic Simulation

This module provides the GPU-accelerated simulation infrastructure using FLAMEGPU2.

Components:
- Agent definitions (EdgeQueue, Packet, SignalController)
- Message definitions (entry_request, entry_accept, departure_notice, green_signal)
- Agent functions (GPU kernels for traffic dynamics)
- Simulation runner and configuration

References:
- FLAMEGPU2 Documentation: https://docs.flamegpu.com/
- FLAMEGPU2 Python API: https://docs.flamegpu.com/api/
"""

from .model import (
    MesoscopicTrafficModel,
    create_model,
)

from .agents import (
    define_edge_queue_agent,
    define_packet_agent,
    define_signal_controller_agent,
)

from .messages import (
    define_messages,
)

from .simulation import (
    MesoscopicSimulation,
    SimulationConfig,
)

__all__ = [
    'MesoscopicTrafficModel',
    'create_model',
    'define_edge_queue_agent',
    'define_packet_agent', 
    'define_signal_controller_agent',
    'define_messages',
    'MesoscopicSimulation',
    'SimulationConfig',
]
