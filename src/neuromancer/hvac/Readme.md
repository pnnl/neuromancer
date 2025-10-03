# neuromancer.hvac

A PyTorch-based neuromancer subpackage for differentiable building simulation and HVAC system modeling.

## Overview

`neuromancer.hvac` provides physics-based, differentiable models for HVAC systems that are compatible with PyTorch's automatic differentiation. This enables optimization, control design, and machine learning applications for building energy systems.

## Features

- **Differentiable building physics**: All models support gradient computation
- **Zone vectorization**: Efficient simulation of multi-zone buildings
- **Realistic HVAC components**: Physics-based models with actuator dynamics
- **Autonomous simulation**: Built-in scheduling and input generation
- **ODE integration**: Compatible with `torchdiffeq` for time-domain simulation

## Components

### Building Components
- **`Envelope`**: Multi-zone building thermal envelope with RC network modeling
- **`VAVBox`**: Variable Air Volume terminal unit with damper and electric reheat
- **`BuildingComponent`**: Base class for all building system components

### Actuators
- **`Actuator`**: Base class for HVAC actuators with realistic dynamics
- **`Damper`**: Air damper with flow characteristics and pressure effects
- **`ElectricReheatCoil`**: Electric heating coil with efficiency modeling

### Simulation Inputs
- **Scheduling functions**: `binary_schedule`, `multi_level_schedule`, `sinusoidal_temperature`
- **Weather patterns**: `seasonal_temperature` for realistic outdoor conditions
- **Stochastic inputs**: Support for noise and persistent excitation


### Prerequisites
- Python 3.8+
- Git

### Dependencies
- **torch** (>= 1.9.0) - PyTorch for tensor operations and automatic differentiation
- **torchdiffeq** (>= 0.2.0) - Differential equation solvers for PyTorch
- **numpy** (>= 1.21.0) - Numerical computing
- **matplotlib** (>= 3.5.0) - Plotting and visualization

## Quick Start

```python
from torch_buildings import VAVBox, Envelope

# Create a 3-zone VAV system
vav = VAVBox(
    n_zones=3,
    control_gain=[2.0, 2.5, 3.0],      # Zone-specific control gains
    min_airflow=[0.05, 0.08, 0.06],    # Minimum airflow per zone [kg/s]
    max_airflow=[0.3, 0.4, 0.35]       # Maximum airflow per zone [kg/s]
)

# Create building envelope
envelope = Envelope(
    n_zones=3,
    R_env=[0.1, 0.12, 0.09],           # Envelope resistance per zone [K/W]
    C_env=[1e6, 1.2e6, 0.9e6]          # Thermal capacitance per zone [J/K]
)

# Run autonomous simulation
results = vav.simulate(
    n_steps=24,        # 24 hours
    dt=3600.0,         # 1-hour time steps
    batch_size=1,
    return_full_trajectory=True
)

print("Simulation completed!")
print(f"Final airflows: {results['airflow'][-1, 0, :]} kg/s")
print(f"Final power consumption: {results['total_power'][-1, 0, :]} W")
```

## Subpackage Structure

```
├── hvac/
│   ├── __init__.py                    # Main package imports
│   ├── building_components/
│   │   ├── __init__.py
│   │   ├── base.py                    # BuildingComponent base class
│   │   ├── envelope.py                # Building envelope model
│   │   └── vav_box.py                 # VAV terminal unit
│   ├── actuators/
│   │   ├── __init__.py
│   │   ├── actuator.py                # Actuator base class
│   │   ├── damper.py                  # Air damper model
│   │   └── electric_reheat_coil.py    # Electric heating coil
│   └── simulation_inputs/
│       ├── __init__.py
│       └── schedules.py               # Time-varying input functions
└── tests/
    ├── test_building_components/      # Building component tests
    └── test_actuators/                # Actuator tests
```
