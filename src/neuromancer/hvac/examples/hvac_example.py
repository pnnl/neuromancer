"""
Building System Simulation Example

This example demonstrates a complete building HVAC system simulation using:
- RTU (Rooftop Unit) as central air handler
- VAVBox terminal units for 2 zones
- 2-zone Envelope for thermal dynamics
- SolarGains for external heat gains

The components are automatically wired together based on matching variable names.

System Architecture:
SolarGains → Envelope ← VAVBox ← RTU
               ↓         ↑       ↑
           T_zones → setpoint   return_airflow_zones
"""

import torch
import matplotlib.pyplot as plt

# Import building components
from torch_buildings.building_components import RTU, VAVBox, Envelope, SolarGains
from torch_buildings.building import BuildingNode, BuildingSystem
from torch_buildings.plot import simplot

print("Building System Simulation Example")
print("="*60)
print("Components: RTU + VAVBox + Envelope + SolarGains")
print("Configuration: 2 zones, automatic wiring")
print("="*60)

# System configuration
n_zones = 2

# =============================================================================
# CREATE BUILDING COMPONENTS
# =============================================================================

print("Creating building components...")

# 1. Solar gains for external heat input
solar = SolarGains(
    n_zones=n_zones,
    window_area=25.0,  # m² per zone
    window_orientation=[0.0, 90.0],  # South, West facing windows
    window_shgc=0.6,   # Solar heat gain coefficient
    latitude_deg=40.0, # Building latitude
    max_solar_irradiance=800.0  # W/m²
)

# 2. Building envelope for thermal dynamics
envelope = Envelope(
    n_zones=n_zones,
    R_env=[0.1, 0.12],    # Zone-specific thermal resistance [K/W]
    C_env=[1.2e6, 1.0e6], # Zone-specific thermal mass [J/K]
    R_internal=0.05,      # Inter-zone resistance [K/W]
)

# 3. RTU central air handler
rtu = RTU(
    n_zones=n_zones,
    airflow_max=4.0,      # Total system capacity [kg/s]
    airflow_oa_min=0.4,      # Minimum outdoor air [kg/s]
    Q_coil_max=20000,     # Heating/cooling capacity [W]
    fan_power_per_flow=800,  # Fan efficiency [W/(kg/s)]
    cooling_COP=3.2,      # Cooling efficiency
    heating_efficiency=0.88  # Heating efficiency
)

# 4. VAV boxes for zone control
vav = VAVBox(
    n_zones=n_zones,
    airflow_min=[0.1, 0.08],     # Zone minimums [kg/s]
    airflow_max=[0.8, 0.6],      # Zone maximums [kg/s]
    control_gain=[2.5, 2.0],     # Zone control sensitivity
    Q_reheat_max=[3000, 2500], # Zone reheat capacity [W]
    reheat_efficiency=0.95       # Electric reheat efficiency
)

# =============================================================================
# CREATE BUILDING SYSTEM NODES
# =============================================================================

print("Creating BuildingNode wrappers...")

# Wrap components as nodes
envelope_inputs = {
    "envelope.T_zones": "T_zones",
    "T_outdoor": "T_outdoor",
    "solar.Q_solar": "Q_solar",
    "Q_internal": "Q_internal",
    "vav.Q_supply_flow": "Q_hvac"
}

rtu_inputs = {
    "T_outdoor": "T_outdoor",
    "envelope.T_zones": "T_return_zones",
    "vav.supply_airflow": "return_airflow_zones",
    "rtu_T_supply_setpoint": "T_supply_setpoint",
    "rtu_supply_airflow_setpoint": "supply_airflow_setpoint",
    "rtu.damper_position": "damper_position",
    "rtu.valve_position": "valve_position",
    "rtu.T_supply": "T_supply",
    "rtu.integral_accumulator": "integral_accumulator",
}

vav_inputs = {
    "envelope.T_zones": "T_zone",
    "vav_T_setpoint": "T_setpoint",
    "rtu.T_supply": "T_supply_upstream",
    "rtu.P_supply": "P_duct",
    "vav.damper_position": "damper_position",
    "vav.reheat_position": "reheat_position",
}

solar_inputs = {
    "T_outdoor": "T_outdoor",
    "weather_factor": "weather_factor",
    "day_of_year": "day_of_year"
}

solar_node = BuildingNode(solar, input_map=solar_inputs, name="solar")
envelope_node = BuildingNode(envelope, input_map=envelope_inputs, name="envelope")
rtu_node = BuildingNode(rtu, input_map=rtu_inputs, name="rtu")
vav_node = BuildingNode(vav, input_map=vav_inputs, name="vav")

data = {}
# Weather and occupancy disturbance variables
data["Q_internal"] = torch.stack([envelope.input_functions["Q_internal"](t, batch_size=1) for t in range(288)], dim=1)
data["T_outdoor"] = torch.stack([solar.input_functions["T_outdoor"](t, batch_size=1) for t in range(288)], dim=1)
data["weather_factor"] = torch.stack([solar.input_functions["weather_factor"](t, batch_size=1) for t in range(288)], dim=1)
data["day_of_year"] = torch.stack([solar.input_functions["day_of_year"](t, batch_size=1) for t in range(288)], dim=1)

# Control variables
data["rtu_T_supply_setpoint"] = torch.stack([rtu.input_functions["T_supply_setpoint"](t, batch_size=1) for t in range(288)], dim=1)
data["rtu_supply_airflow_setpoint"] = torch.stack([rtu.input_functions["supply_airflow_setpoint"](t, batch_size=1) for t in range(288)], dim=1)
data["vav_T_setpoint"] = torch.stack([vav.input_functions["T_setpoint"](t, batch_size=1) for t in range(288)], dim=1)


# 1. SolarGains - generates solar_gains (external input)
# 2. RTU - processes return air, generates supply conditions
# 3. VAVBox - modulates supply air, generates zone loads
# 4. Envelope - integrates all heat sources, updates zone temperatures
system = BuildingSystem([
    solar_node,    # External gains first
    rtu_node,      # Central equipment processes return air
    vav_node,      # Terminal units modulate supply
    envelope_node  # Thermal dynamics last (integrates all loads)
], name="TwoZoneBuilding")

fig, _ = simplot(
    system,
    results=data,
    variables=[k for k in data],
    time_range=None,  # Full day
    figsize=(14, 10),
    title="Initial data",
    filename='plots/initial_data.png'
)

# Print system summary
print("\n" + "="*60)
system.show(figname='example_hvac.png')
print("="*60)

# =============================================================================
# RUN BUILDING SIMULATION
# =============================================================================

print("\nRunning 24-hour simulation...")
results = system.simulate(
    duration_hours=24.0,    # Full day
    dt_minutes=5.0,         # 5-minute time steps
    t_start=6.0,       # Start at 6 AM
    batch_size=1,
    external_inputs=data,# Single scenario
)
print(f"Simulation complete!")
print(f"Results contain {len(results)} variables")
print(f"Time steps: {results['t'].shape[1]}")
print(f"Variables: {list(results.keys())}")

# =============================================================================
# ANALYZE AND PLOT RESULTS USING SIMPLOT
# =============================================================================

print("\nAnalyzing results...")

# Key variables to analyze
key_variables = [
    'envelope.T_zones',      # Zone temperatures
    'rtu.T_supply',          # Supply air temperature
    'rtu.total_power',       # RTU power consumption
    'vav.damper_position',   # VAV damper positions
    'vav.reheat_position',   # VAV reheat positions
    'vav.Q_supply_flow',
    'solar.Q_solar',     # Solar heat gains
]

# Check which variables are available
available_vars = [var for var in key_variables if var in results]
missing_vars = [var for var in key_variables if var not in results]

if missing_vars:
    print(f"Warning: Missing variables: {missing_vars}")

print(f"Plotting {len(available_vars)} key variables...")

fig, _ = simplot(
    system,
    results=results,
    variables=available_vars,
    time_range=None,  # Full day
    figsize=(14, 10),
    title="Two-Zone Building System - 24 Hour Simulation",
    filename='plots/24hr.png'
)
#
# # Create zoomed morning startup plot
# print("Creating morning startup analysis...")
# fig_morning, _ = simplot(
#     system,
#     results=results,
#     variables=['envelope.T_zones', 'rtu.T_supply', 'rtu.total_power'],
#     time_range=(0, 4),  # First 4 hours after start (6 AM to 10 AM)
#     figsize=(12, 8),
#     title="Morning Startup Analysis (6 AM - 10 AM)",
#     filename='iniital_example.png'
# )
#
# # =============================================================================
# # PERFORMANCE SUMMARY
# # =============================================================================
#
# print("\n" + "="*60)
# print("PERFORMANCE SUMMARY")
# print("="*60)
#
# # Extract final values (last time step, first batch)
# final_step = -1
# batch_idx = 0
#
# # Zone temperatures
# if 'envelope.T_zones' in results:
#     final_temps = results['envelope.T_zones'][batch_idx, final_step, :]
#     print(f"Final Zone Temperatures:")
#     for i, temp in enumerate(final_temps):
#         temp_c = temp.item() - 273.15  # Convert K to °C
#         print(f"  Zone {i+1}: {temp_c:.1f}°C")
#
# # Total energy consumption
# if 'rtu.total_power' in results:
#     power_time_series = results['rtu.total_power'][batch_idx, :, 0]  # [time]
#     avg_power = power_time_series.mean().item()
#     max_power = power_time_series.max().item()
#     total_energy = power_time_series.sum().item() * (5 * 60) / 3600  # 5-min steps to kWh
#     print(f"\nRTU Energy Performance:")
#     print(f"  Average Power: {avg_power:.0f} W")
#     print(f"  Peak Power: {max_power:.0f} W")
#     print(f"  Total Energy: {total_energy/1000:.1f} kWh")
#
# # Solar gains
# if 'solar.solar_gains' in results:
#     solar_time_series = results['solar.solar_gains'][batch_idx, :, :]  # [time, zones]
#     total_solar = solar_time_series.sum().item() * (5 * 60) / 3600  # kWh
#     peak_solar = solar_time_series.max().item()
#     print(f"\nSolar Heat Gains:")
#     print(f"  Peak Solar: {peak_solar:.0f} W")
#     print(f"  Total Solar: {total_solar/1000:.1f} kWh")
#
# print("="*60)
#
# # =============================================================================
# # DISPLAY RESULTS
# # =============================================================================
#
#
# print(f"\nSimulation completed successfully!")
#
# # =============================================================================
# # INTERACTIVE ANALYSIS SUGGESTIONS
# # =============================================================================
#
# print("\nFor interactive analysis, try:")
# print("  results['envelope.T_zones'][0, :, 0]  # Zone 1 temperature over time")
# print("  results['rtu.total_power'][0, :, 0]   # RTU power over time")
# print("  system.simulate(duration_hours=12, dt_minutes=1)  # High-res simulation")
#
# # Variable connections for reference:
# print("\nAutomatic Variable Connections:")
# print("  SolarGains.solar_gains → Envelope.Q_solar")
# print("  Envelope.T_zones → VAVBox.T_zone")
# print("  Envelope.T_zones → RTU.T_return_zones")
# print("  RTU.T_supply → VAVBox.supply_temp")
# print("  RTU.supply_airflow → VAVBox.supply_airflow")
# print("  RTU.supply_pressure → VAVBox.duct_pressure")
# print("  VAVBox.supply_heat_flow → Envelope.Q_hvac")
# print("  VAVBox.supply_airflow → RTU.return_airflow_zones")
#
# # Optional: Save results for further analysis
# # torch.save(results, 'building_simulation_results.pt')
# print("\nExample completed. Results available in 'system' and 'results' variables.")