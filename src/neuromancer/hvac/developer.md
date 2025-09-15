# Variable Naming Conventions

## Physical Quantity Prefixes

Use unit-based prefixes for physical quantities to enable immediate unit recognition:

- **`T_*`** → Temperature in Kelvin [K]  
  Examples: `T_supply`, `T_zone`, `T_setpoint`, `T_outdoor`

- **`Q_*`** → Thermal power in Watts [W]  
  Examples: `Q_cooling_load`, `Q_solar`, `Q_coil_actual`

- **`P_*`** → Pressure in Pascals [Pa]  
  Examples: `P_supply`, `P_duct`, `P_static`

## Physical Quantity Suffixes

- **`*_power`** → Electrical power in Watts [W]  
  Examples: `fan_power`, `total_power`, `heating_power`

## Limit Suffixes

- **`*_min`** → Hard minimum limits (used in clamp operations)  
  Examples: `airflow_min`, `T_supply_min`

- **`*_max`** → Hard maximum limits (used in clamp operations)  
  Examples: `airflow_max`, `Q_coil_max`

## Control Parameters

- **`ctrl_*`** → Control-related parameters  
  Examples: `ctrl_deadband`, `ctrl_integral_time`, `ctrl_Kp`

## Time Constants

- **`tau_*`** → Time constants in seconds [s]  
  Examples: `tau_damper`, `tau_thermal`

## Descriptive Names

Use descriptive names without prefixes for:
- **Airflow variables** in kg/s: `airflow_actual`, `supply_airflow`
- **Dimensionless variables**: `damper_position`, `efficiency`, `weather_factor`
- **Coefficients with units**: `cp_air` [J/kg/K], `fan_power_per_flow` [W/(kg/s)]

## Benefits

1. **Immediate unit recognition**: `T_zone` = "temperature in Kelvin"
2. **Physical meaning clarity**: `Q_reheat` vs `reheat_power` (thermal vs electrical)
3. **Limit identification**: `*_min`/`*_max` = "hard constraint used in clamp"
4. **System coordination**: Clear which variables need coordination across components

## Examples

```python
# ✅ Correct naming
T_supply = 286.15          # [K] Temperature
Q_cooling_load = 5000.0    # [W] Thermal power
P_supply = 102000.0        # [Pa] Pressure
fan_power = 1200.0         # [W] Electrical power
airflow_max = 3.0          # [kg/s] Maximum airflow limit
ctrl_deadband = 2.0        # [K] Control parameter
tau_damper = 5.0           # [s] Time constant
supply_airflow = 2.5       # [kg/s] Descriptive airflow

# ❌ Incorrect naming
supply_temp = 286.15       # Missing T_ prefix
cooling_load = 5000.0      # Missing Q_ prefix
supply_pressure = 102000.0 # Missing P_ prefix
electrical_power = 1200.0  # Use *_power suffix
max_airflow = 3.0          # Use airflow_max
deadband = 2.0             # Missing ctrl_ prefix
```

# Context-Driven Initialization Guide for BuildingComponent

## Overview

All BuildingComponent classes should implement context-aware `initial_state_functions` and `input_functions` to ensure coordinated system startup and eliminate unrealistic transients.

## Key Principles

1. **Use `self.context` directly** - The base class automatically provides a default context (MILD_COOLING_CONTEXT)
2. **Provide sensible fallbacks** - Use `self.context.get(key, default)` for robustness
3. **Maintain time continuity** - Use context `time_of_day` as simulation baseline
4. **Physical consistency** - Ensure all values match the intended building scenario

## Context Parameters Available

```python
# Environmental conditions
T_outdoor: float          # [K] Outdoor air temperature
time_of_day: float        # [hr] Hour of day (0-24)
day_of_year: int          # [day] Day of year (1-365)
weather_factor: float     # [-] Sky clarity (0=overcast, 1=clear)

# Building operating state
occupancy_state: str      # "occupied", "unoccupied", "transition"
system_mode: str          # "cooling", "heating", "setback", "economizer"
T_setpoint_base: float    # [K] Zone temperature setpoint
T_supply_base: float      # [K] Supply air temperature

# System operating points
supply_airflow_fraction: float  # [-] Fraction of maximum airflow
P_duct_base: float        # [Pa] Duct pressure
```

## Implementation Pattern

### 1. Initial State Functions

```python
def initial_state_functions(self, mode="steady_state"):
    """Return context-aware initial state sampling functions."""
    return {
        "my_state": lambda bs: self._sample_my_state(bs, mode),
    }

def _sample_my_state(self, batch_size, mode):
    """Sample initial state based on context."""
    # Get context value with fallback
    context_value = self.context.get("relevant_context_param", default_value)
    base_state = torch.full((batch_size, self.n_zones), context_value, 
                           device=self.device, dtype=self.dtype)
    
    if mode == "steady_state":
        return base_state
    elif mode == "realistic":
        # Add small realistic variations
        noise = torch.normal(0.0, variation_std, (batch_size, self.n_zones),
                           device=self.device, dtype=self.dtype)
        return torch.clamp(base_state + noise, min_val, max_val)
```

### 2. Input Functions

```python
@property
def input_functions(self):
    """Context-aware input functions."""
    if not hasattr(self, '_input_functions'):
        # Extract context values with fallbacks
        T_outdoor_base = self.context.get("T_outdoor", 288.15)
        time_of_day = self.context.get("time_of_day", 12.0)
        day_of_year = self.context.get("day_of_year", 100)
        occupancy_state = self.context.get("occupancy_state", "occupied")
        
        def my_input_fn(t, batch_size=1):
            # Calculate current time from context baseline
            context_hour = time_of_day
            sim_hours = t / 3600.0
            current_hour = (context_hour + sim_hours) % 24
            
            # Apply context-based logic
            if occupancy_state == "occupied":
                # Occupied logic
                pass
            elif occupancy_state == "unoccupied":
                # Unoccupied logic
                pass
            
            # Include daily/seasonal variations as appropriate
            daily_variation = amplitude * torch.sin(torch.tensor(2 * torch.pi * (current_hour - peak_hour) / 24))
            seasonal_variation = amplitude * torch.sin(torch.tensor(2 * torch.pi * (day_of_year - 80) / 365))
            
            result = base_value + daily_variation + seasonal_variation
            return torch.full((batch_size, self.n_zones), result, 
                            device=self.device, dtype=self.dtype)
        
        self._input_functions = {
            "my_input": my_input_fn,
        }
    
    return self._input_functions
```

## Time Handling Standards

**Always use this pattern for time progression:**
```python
context_hour = time_of_day
sim_hours = t / 3600.0
current_hour = (context_hour + sim_hours) % 24
```

**For seasonal patterns:**
```python
seasonal_effect = amplitude * torch.sin(torch.tensor(2 * torch.pi * (day_of_year - 80) / 365))
# Day 80 ≈ March 21 (spring equinox reference point)
```

## Common Context Mappings

| Component Type | Key Context Parameters |
|----------------|----------------------|
| **Envelope** | `T_outdoor`, `T_setpoint_base`, `occupancy_state` |
| **RTU/HVAC** | `T_supply_base`, `system_mode`, `supply_airflow_fraction` |
| **VAV Box** | `T_setpoint_base`, `supply_airflow_fraction` |
| **Solar Gains** | `weather_factor`, `time_of_day`, `day_of_year` |

## Validation Checklist

✅ **Uses `self.context.get()` with sensible fallbacks**  
✅ **Proper time progression from context baseline**  
✅ **Physical values match intended building scenario**  
✅ **Occupancy state affects relevant parameters**  
✅ **System mode influences equipment operation**  
✅ **Daily and seasonal patterns are realistic**  
✅ **Initial states are consistent with input functions**

## Example Usage

```python
from torch_buildings.contexts import PEAK_COOLING_CONTEXT, WINTER_HEATING_CONTEXT

# Components automatically coordinate through shared context
envelope = Envelope(n_zones=3, context=PEAK_COOLING_CONTEXT)
rtu = RTU(context=PEAK_COOLING_CONTEXT)

# Both components start with consistent:
# - Hot outdoor temperatures (35°C)
# - Cooling system mode
# - High airflow fraction (80%)
# - Cold supply air (12°C)
```

## Benefits

- **Eliminates startup transients** - Components start in physically consistent states
- **Scenario flexibility** - Easy switching between operating conditions
- **Realistic behavior** - All variables reflect actual building operation patterns  
- **System coordination** - Multiple components automatically align their assumptions