"""
torch_buildings: A PyTorch-based library for differentiable building simulation.

This library provides physics-based, differentiable models for building systems
including HVAC components, building envelopes, and complete building systems.
All models are compatible with PyTorch's automatic differentiation for
optimization, control, and machine learning applications.
"""

# Core base classes
from .building_components.base import BuildingComponent

# Main building components
from .building_components.envelope import Envelope
from .building_components.vav_box import VAVBox

# Actuators
from .actuators.damper import Damper
from .actuators.electric_reheat_coil import ElectricReheatCoil
from .actuators.actuator import Actuator

# Simulation inputs
from .simulation_inputs.schedules import (
    binary_schedule,
    multi_level_schedule,
    sinusoidal_temperature,
    seasonal_temperature
)

