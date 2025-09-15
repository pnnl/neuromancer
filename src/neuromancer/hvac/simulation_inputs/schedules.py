
"""
simulation_inputs.py

Synthetic input generation for transforming non-autonomous HVAC models into autonomous systems.

PURPOSE:
========
This module provides functions for generating synthetic time-series inputs that enable simulation
of HVAC equipment models without requiring external data sources. It transforms non-autonomous
systems (which depend on external inputs like weather, occupancy, and control signals) into
autonomous systems suitable for testing, demonstration, and educational purposes.

WORKFLOW:
=========
HVAC equipment models in torch_buildings are designed as non-autonomous systems that require
external inputs such as:
- Environmental conditions (outdoor temperature, humidity, solar irradiance)
- Building conditions (return air temperature, zone temperatures)
- Control signals (setpoints, operating modes, flow rates)
- Occupancy patterns (internal loads, schedules)

To simulate these models standalone, we need to provide realistic synthetic inputs that capture
the essential dynamics and patterns of real building operation.

TYPICAL USAGE PATTERN:
======================

1. **Model Instantiation:**
   ```python
   from torch_buildings.air_handlers import RTU
   rtu = RTU()
   ```

2. **Generate Input Functions:**
   ```python
   from torch_buildings.simulation_inputs import sinusoidal_temperature, binary_schedule

   # Create realistic input profiles
   T_oa_fn = lambda t: sinusoidal_temperature(t, base_temp=293.15, amplitude=15.0)
   T_ra_fn = lambda t: sinusoidal_temperature(t, base_temp=295.15, amplitude=4.0)
   setpoint_fn = lambda t: binary_schedule(t, 287.15, 289.15)
   ```

3. **Autonomous Simulation:**
   ```python
   def rhs(t, x):
       return rtu(
           t=t, T_supply=x,
           T_oa=T_oa_fn(t),
           T_ra=T_ra_fn(t),
           supply_temp_setpoint=setpoint_fn(t),
           # ... other synthetic inputs
       )

   from torchdiffeq import odeint
   trajectory = odeint(rhs, initial_state, time_vector)
   ```

4. **Default Input Functions (Class Method Pattern):**
   ```python
   # Each equipment class provides complete default scenarios
   input_funcs = RTU.default_input_functions()

   def rhs(t, x):
       return rtu(
           t=t, T_supply=x,
           T_oa=input_funcs["T_oa_fn"](t),
           T_ra=input_funcs["T_ra_fn"](t),
           supply_temp_setpoint=input_funcs["supply_temp_setpoint_fn"](t),
           # ... etc
       )
   ```

INPUT CATEGORIES:
=================

1. **Environmental Inputs:**
   - Outdoor air temperature (diurnal and seasonal cycles)
   - Solar irradiance (time of day and weather dependent)
   - Humidity and wet-bulb temperature

2. **Building Inputs:**
   - Return air temperature (influenced by internal loads)
   - Zone temperatures (with realistic drift and variation)
   - Internal heat gains (occupancy, equipment, lighting)

3. **Control Inputs:**
   - Temperature setpoints (occupied/unoccupied schedules)
   - Airflow rates (ventilation and comfort requirements)
   - Operating modes (heating, cooling, economizer operation)

4. **Schedule Patterns:**
   - Binary on/off schedules (simple day/night operation)
   - Multi-level schedules (morning startup, day, evening, night)
   - Sinusoidal variations (natural temperature cycles)
   - Stochastic variations (realistic operational noise)

UNITS CONVENTION:
=================
All temperature inputs and outputs use Kelvin [K] for consistency with torch_buildings.
Time inputs are in seconds [s] from midnight.
Other units follow torch_buildings conventions (kg/s for airflow, W for power, etc.).

EXTENSIBILITY:
==============
This module can be extended with:
- Weather data integration (reading actual weather files)
- Building-specific profiles (different building types and uses)
- Stochastic input generation (Monte Carlo simulation support)
- Seasonal and regional variations (different climates and locations)
- Advanced control strategies (optimal control, predictive control)
"""

import torch
import numpy as np

################################
## Cyclical time-based schedules
################################

def sinusoidal_temperature(t: float, base_temp: float = 293.15, amplitude: float = 10.0,
                          peak_hour: float = 14.0, shape=(1,1)) -> torch.Tensor:
    """
    Generate sinusoidal temperature profile for natural daily variation.

    Args:
        t (float): Time [s] from midnight.
        base_temp (float, optional): Base (average) temperature [K]. Default 293.15 K (20°C).
        amplitude (float, optional): Temperature swing amplitude [K] (peak = base + amplitude). Default 10.0 K.
        peak_hour (float, optional): Hour of peak temperature [0-24], default 14.0 (2 PM).
        shape (tuple, optional): Output tensor shape. Default (1,1).

    Returns:
        torch.Tensor: Temperature [K] at time t, shape `shape`.

    See file docstring for physical interpretation and usage.
    """
    t_hr = t / 3600.0
    temp_k = base_temp + amplitude * np.sin(2 * np.pi * (t_hr - peak_hour) / 24)
    return torch.tensor([temp_k]).repeat(shape)


def binary_schedule(t: float, primary_value: float = 1., secondary_value: float = 0.,
                   start_hour: float = 7.0, end_hour: float = 19.0, shape=(1,1)) -> torch.Tensor:
    """
    Generate binary schedule, e.g., simple occupied/unoccupied operation patterns.

    Args:
        t (float): Time [s] from midnight.
        primary_value (float, optional): Value during primary period (occupied/day/peak). Default 1.0.
        secondary_value (float, optional): Value during secondary period (unoccupied/night/base). Default 0.0.
        start_hour (float, optional): Start of primary period [0-24]. Default 7.0 (7 AM).
        end_hour (float, optional): End of primary period [0-24]. Default 19.0 (7 PM).
        shape (tuple, optional): Output tensor shape. Default (1,1).

    Returns:
        torch.Tensor: Scheduled value at time t.

    See file docstring for physical interpretation and usage.
    """
    t_hr = t / 3600.0
    value = primary_value if start_hour <= t_hr < end_hour else secondary_value
    return torch.tensor([value]).repeat(shape)


def multi_level_schedule(t: float, schedule_dict: dict, shape=(1,1)) -> torch.Tensor:
    """
    Generate multi-level schedule with different values for different time periods.

    Args:
        t (float): Time [s] from midnight.
        schedule_dict (dict): Mapping of (start_hour, end_hour) tuples to scheduled value.
            Example: {(0, 6): 298.0, (6, 8): 296.0, ...}
        shape (tuple, optional): Output tensor shape. Default (1,1).

    Returns:
        torch.Tensor: Scheduled value at time t.

    See file docstring for physical interpretation and usage.
    """
    t_hr = t / 3600.0
    for (start_hour, end_hour), value in schedule_dict.items():
        if start_hour <= t_hr < end_hour:
            return torch.tensor([value]).repeat(shape)
    # Default fallback (should not occur with complete schedule)
    return torch.tensor([list(schedule_dict.values())[0]]).repeat(shape)


def ramp_schedule(t: float, start_hour: float = 6.0,
                  start_value: float = 298.0, end_value: float = 295.0,
                  ramp_duration: float = 1.0, shape=(1,1)) -> torch.Tensor:
    """
    Generate ramped transition schedule for smooth equipment startup/shutdown.

    Args:
        t (float): Time [s] from midnight.
        start_hour (float, optional): Hour to begin transition [0-24]. Default 6.0.
        end_hour (float, optional): Hour to complete transition [0-24]. Default 8.0.
        start_value (float, optional): Value before/during ramp start. Default 298.0.
        end_value (float, optional): Value after ramp completion. Default 295.0.
        ramp_duration (float, optional): Duration of transition [hours]. Default 1.0.
        shape (tuple, optional): Output tensor shape. Default (1,1).

    Returns:
        torch.Tensor: Ramped value at time t.

    See file docstring for physical interpretation and usage.
    """
    t_hr = t / 3600.0
    if t_hr < start_hour:
        return torch.tensor([start_value]).repeat(shape)
    elif t_hr > start_hour + ramp_duration:
        return torch.tensor([end_value]).repeat(shape)
    else:
        # Linear ramp between start and end values
        progress = (t_hr - start_hour) / ramp_duration
        value = start_value + progress * (end_value - start_value)
        return torch.tensor([value]).repeat(shape)


def seasonal_temperature(t: float, base_temp: float = 288.15, daily_amplitude: float = 10.0, peak_hour=14.,
                        seasonal_amplitude: float = 20.0, day_of_year: int = None, shape=(1,1)) -> torch.Tensor:
    """
    Generate temperature profile with both daily and seasonal variation.

    Args:
        t (float): Time [s] from midnight.
        base_temp (float, optional): Annual average temperature [K]. Default 288.15 K.
        daily_amplitude (float, optional): Daily temperature swing [K]. Default 10.0 K.
        peak_hour (float, optional): Hour of peak temperature [0-24], default 14.0 (2 PM).
        seasonal_amplitude (float, optional): Seasonal temperature swing [K]. Default 20.0 K.
        day_of_year (int, optional): Day of year [1-365]. If None, calculated from t. Default None.
        nzones (int, optional): Output batch size. Default 1.

    Returns:
        torch.Tensor: Temperature [K] with daily and seasonal variation (shape: [nzones]).

    See file docstring for physical interpretation and usage.
    """
    if isinstance(t, torch.Tensor):
        print('tttttttt\n', t.shape)
        exit()
    t_hr = t / 3600.0
    if day_of_year is None:
        day_of_year = int((t / 86400) % 365) + 1
    daily_temp = daily_amplitude * np.sin(2 * np.pi * (t_hr - peak_hour) / 24)
    seasonal_temp = seasonal_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    total_temp = base_temp + daily_temp + seasonal_temp
    total_temp = torch.full(shape, total_temp)
    print("temp", total_temp.shape)
    return total_temp
######################################
### Adding stochasticity to schedules
######################################

def stochastic_variation(*, base_function: callable, noise_amplitude: float = 1.0,
                        correlation_time: float = 3600.0) -> callable:
    """
    Add realistic stochastic variation to deterministic input functions.

    Args:
        base_function (callable): Deterministic function to add noise to.
        noise_amplitude (float, optional): Standard deviation of noise [same units as base function]. Default 1.0.
        correlation_time (float, optional): Time correlation of noise [s]. Default 3600.0 (1 hour).

    Returns:
        callable: New function with stochastic variation (same signature as base_function).

    See file docstring for physical interpretation and usage.
    """
    def noisy_function(t: float, batch_size: int) -> torch.Tensor:
        base_value = base_function(t, batch_size)
        # Simple correlated noise using time-based seed
        torch.manual_seed(int(t / correlation_time))
        noise = torch.normal(0.0, noise_amplitude, size=base_value.shape)
        return base_value + noise
    return noisy_function


def persistent_excitation(*, base_function: callable, amp: float = 1.0) -> callable:
    """
    Wrap input function for persistent excitation (multi-sine + noise).

    Args:
        base_function (callable): The deterministic input function to modulate.
        amp (float, optional): Amplitude of the persistent excitation added. Default 1.0.

    Returns:
        callable: New function with persistent excitation (multi-sine + noise).

    Usage:
        Used for system identification, robust control testing, or to stress test models with diverse input frequencies.
    """
    def excited(t: float, batch_size: int):
        base_value = base_function(t, batch_size)
        t_ = torch.as_tensor(t, dtype=torch.float32)
        n_freqs = 4
        freq_base = 0.0001  # cycles/sec (~3 cycles/day)
        excitation = torch.zeros(*base_value.shape, device=base_value.device, dtype=base_value.dtype)
        for i in range(n_freqs):
            freq = freq_base * (i+1)
            phase = torch.rand(*base_value.shape, device=base_value.device) * 2 * torch.pi
            excitation += amp * torch.sin(2 * torch.pi * freq * t_ + phase)
        return base_value + excitation
    return excited
