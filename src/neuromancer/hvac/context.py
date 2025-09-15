"""
context.py - Building System Context Definitions

This module provides predefined building operating contexts that promote consistent
steady-state conditions across all building components during simulation initialization.

WHAT ARE BUILDING CONTEXTS?
Building contexts represent realistic snapshots of building conditions at specific
times and operating modes. They define the environmental conditions, occupancy state,
and equipment operating points that components should assume at t=0, ensuring that
all parts of the building system start from physically consistent conditions.

Think of a context as answering: "What kind of building day are we simulating?"
- Is it a hot summer afternoon with full cooling load?
- A mild spring morning with mixed heating/cooling needs?
- A winter night with the building in energy-saving setback mode?

WHY USE CONTEXTS?
Without coordinated contexts, individual building components make independent
assumptions about operating conditions, leading to:
- Temperature mismatches (hot zones with cold supply air)
- Airflow inconsistencies (high pressure with low flows)
- Control logic conflicts (heating and cooling simultaneously)
- Unrealistic transient behavior during simulation startup

CONTEXT PARAMETERS EXPLAINED:

Environmental Conditions:
- T_outdoor [K]: Outdoor air temperature - drives envelope heat transfer and economizer operation
- day_of_year [day]: Day of year (1-365) - determines seasonal solar patterns and weather expectations
- weather_factor [-]: Sky clarity (0=overcast, 1=clear) - modulates solar radiation and outdoor temperature patterns

Building Operating State:
- occupancy_state [str]: "occupied", "unoccupied", "transition" - affects internal heat gains and equipment schedules
- system_mode [str]: "cooling", "heating", "setback", "economizer" - primary HVAC operating mode
- T_setpoint_base [K]: Baseline zone temperature setpoint - target temperature for HVAC control systems
- T_supply_base [K]: Baseline supply air temperature - conditioned air delivered to zones

Units and Typical Ranges:
- Temperature [K]: 268.15-313.15 (−5°C to 40°C)
- Time [hr]: 0.0-24.0 (midnight to midnight)
- Day [day]: 1-365 (January 1 to December 31)
- Weather factor [-]: 0.0-1.0 (completely overcast to perfectly clear)
- Pressure [Pa]: 200-1500 (low to high duct pressure)
- Power [W]: -3000 to +3000 per zone (cooling negative, heating positive)
- Airflow fraction [-]: 0.1-1.0 (minimum ventilation to maximum capacity)

USAGE:
Import and apply contexts to building components before simulation:

    from torch_buildings.contexts import MILD_COOLING_CONTEXT

    envelope.context.update(MILD_COOLING_CONTEXT)
    rtu.context.update(MILD_COOLING_CONTEXT)
    vav_box.context.update(MILD_COOLING_CONTEXT)

    results = building_system.simulate(duration_hours=24.0)

This ensures all components generate input functions and initial states that are
physically consistent and representative of the chosen building operating scenario.
"""

# =============================================================================
# DEFAULT CONTEXT - MILD COOLING CONDITIONS
# =============================================================================

MILD_COOLING_CONTEXT = {
    # Environmental conditions - Pleasant spring/fall day
    "T_outdoor": 288.15,  # [K] 15°C - Mild outdoor temperature
    "day_of_year": 100,  # [day] Early April - Shoulder season
    "weather_factor": 0.7,  # [-] Partly cloudy conditions

    # Building operating state - Normal occupied operation
    "occupancy_state": "occupied",  # [str] Normal business hours
    "system_mode": "cooling",  # [str] Light cooling mode
    "T_setpoint_base": 293.15,  # [K] 20°C - Cooling setpoint
    "T_supply_base": 286.15,  # [K] 13°C - Cool supply air

    # System operating points - Moderate load conditions
    "supply_airflow_fraction": 0.5,  # [-] 50% of maximum airflow
    "P_duct_base": 600.0,  # [Pa] Normal occupied duct pressure
}

# =============================================================================
# NIGHT SETBACK CONTEXT - UNOCCUPIED ENERGY SAVING MODE
# =============================================================================

NIGHT_SETBACK_CONTEXT = {
    # Environmental conditions - Cool night
    "T_outdoor": 283.15,  # [K] 10°C - Cool nighttime temperature
    "day_of_year": 100,  # [day] Early April
    "weather_factor": 0.0,  # [-] No solar radiation

    # Building operating state - Energy-saving setback
    "occupancy_state": "unoccupied",  # [str] Building unoccupied
    "system_mode": "setback",  # [str] Energy-saving mode
    "T_setpoint_base": 296.15,  # [K] 23°C - Setback setpoint
    "T_supply_base": 294.15,  # [K] 21°C - Warmer supply air

    # System operating points - Minimal operation
    "supply_airflow_fraction": 0.2,  # [-] Minimum ventilation airflow
    "P_duct_base": 400.0,  # [Pa] Reduced nighttime pressure
}

# =============================================================================
# PEAK COOLING CONTEXT - HOT SUMMER DAY WITH HIGH COOLING LOADS
# =============================================================================

PEAK_COOLING_CONTEXT = {
    # Environmental conditions - Hot summer afternoon
    "T_outdoor": 308.15,  # [K] 35°C - Hot outdoor temperature
    "day_of_year": 200,  # [day] Mid July - Peak summer
    "weather_factor": 1.0,  # [-] Clear sky, maximum solar

    # Building operating state - Peak cooling demand
    "occupancy_state": "occupied",  # [str] Full occupancy during peak
    "system_mode": "cooling",  # [str] Maximum cooling mode
    "T_setpoint_base": 293.15,  # [K] 20°C - Cooling setpoint
    "T_supply_base": 285.15,  # [K] 12°C - Very cold supply air

    # System operating points - High load conditions
    "supply_airflow_fraction": 0.8,  # [-] High airflow for cooling
    "P_duct_base": 800.0,  # [Pa] High pressure for maximum flow
}

# =============================================================================
# WINTER HEATING CONTEXT - COLD DAY WITH HEATING LOADS
# =============================================================================

WINTER_HEATING_CONTEXT = {
    # Environmental conditions - Cold winter day
    "T_outdoor": 268.15,  # [K] -5°C - Cold outdoor temperature
    "day_of_year": 15,  # [day] January - Peak winter
    "weather_factor": 0.3,  # [-] Overcast winter conditions

    # Building operating state - Heating mode
    "occupancy_state": "occupied",  # [str] Normal occupied hours
    "system_mode": "heating",  # [str] Primary heating mode
    "T_setpoint_base": 293.15,  # [K] 20°C - Heating setpoint
    "T_supply_base": 298.15,  # [K] 25°C - Warm supply air for heating

    # System operating points - Heating load conditions
    "supply_airflow_fraction": 0.4,  # [-] Moderate airflow for heating
    "P_duct_base": 550.0,  # [Pa] Moderate pressure
}

# =============================================================================
# ECONOMIZER CONTEXT - MILD CONDITIONS FAVORING FREE COOLING
# =============================================================================

ECONOMIZER_CONTEXT = {
    # Environmental conditions - Perfect for economizer operation
    "T_outdoor": 285.15,  # [K] 12°C - Cool outdoor air
    "day_of_year": 80,  # [day] Late March - Spring conditions
    "weather_factor": 0.8,  # [-] Mostly clear for solar gains

    # Building operating state - Mixed mode operation
    "occupancy_state": "occupied",  # [str] Normal occupied operation
    "system_mode": "economizer",  # [str] Free cooling mode
    "T_setpoint_base": 293.15,  # [K] 20°C - Cooling setpoint
    "T_supply_base": 287.15,  # [K] 14°C - Mixed outdoor/return air

    # System operating points - Economizer-friendly conditions
    "supply_airflow_fraction": 0.6,  # [-] Higher airflow for free cooling
    "P_duct_base": 650.0,  # [Pa] Moderate-high pressure
}

# =============================================================================
# TRANSITION CONTEXT - SHOULDER SEASON WITH MINIMAL HVAC LOADS
# =============================================================================

TRANSITION_CONTEXT = {
    # Environmental conditions - Perfect weather
    "T_outdoor": 291.15,  # [K] 18°C - Very mild outdoor temperature
    "day_of_year": 120,  # [day] Late April - Ideal spring
    "weather_factor": 0.6,  # [-] Partly cloudy

    # Building operating state - Minimal conditioning needed
    "occupancy_state": "transition",  # [str] Building warming up for occupancy
    "system_mode": "minimal",  # [str] Minimal HVAC operation
    "T_setpoint_base": 293.15,  # [K] 20°C - Setpoint
    "T_supply_base": 291.15,  # [K] 18°C - Minimal conditioning

    # System operating points - Very light loads
    "supply_airflow_fraction": 0.3,  # [-] Ventilation plus light conditioning
    "P_duct_base": 500.0,  # [Pa] Moderate pressure
}

# =============================================================================
# CONTEXT DICTIONARY FOR EASY ACCESS
# =============================================================================

BUILDING_CONTEXTS = {
    "mild_cooling": MILD_COOLING_CONTEXT,
    "night_setback": NIGHT_SETBACK_CONTEXT,
    "peak_cooling": PEAK_COOLING_CONTEXT,
    "winter_heating": WINTER_HEATING_CONTEXT,
    "economizer": ECONOMIZER_CONTEXT,
    "transition": TRANSITION_CONTEXT,
}