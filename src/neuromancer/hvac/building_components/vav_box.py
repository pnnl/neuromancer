"""
vav_box.py

Overview
--------
This module implements a physics-based, differentiable model for a Variable Air Volume (VAV) terminal unit,
the most common type of terminal in modern commercial buildings. VAV boxes deliver conditioned air from
central air handling units to individual building zones, adjusting airflow and temperature to maintain
zone comfort with energy efficiency.

VAV systems offer precise temperature control by modulating both the supply air volume and (optionally)
electric reheat energy. This model supports learning and optimization scenarios in scientific machine
learning and control policy research.

Modeling Approach
-----------------
- Represents the VAV box as a static algebraic model with actuator dynamics handled by sub-components.
- Airflow control uses proportional feedback on zone temperature error, with higher airflow when zones
  are warmer than setpoint (cooling demand), constrained by minimum/maximum flow limits.
- Damper airflow calculation is pressure-dependent with actuator dynamics handled by the Damper component.
- Optional electric reheat coil maintains minimum supply air temperature to prevent zone overcooling,
  with actuator dynamics handled by the ElectricReheatCoil component.
- All component parameters can be made learnable for data-driven or adaptive control applications.
- Modular design: Delegates actuator physics to specialized Damper and ElectricReheatCoil components.

Assumptions and Simplifications
-------------------------------
- Static terminal model: No thermal mass or temperature dynamics in the VAV box itself.
- Zone thermal dynamics are handled by separate zone models - this component only calculates supply conditions.
- Instantaneous mixing: Supply air temperature changes immediately with reheat coil operation.
- Sensible heat only (no humidity or latent effects).
- No explicit modeling of duct losses, air leakage, or transport delays from box to zone.
- Actuator dynamics are first-order and handled by sub-components (Damper, ElectricReheatCoil).
- Electric reheat coil has constant efficiency (no part-load losses).
- Simplified pressure drop modeling across VAV box components.
- No sound, vibration, or mechanical wear effects.

Control Logic
-------------
- Primary control: Modulates airflow based on zone temperature error (more flow for warmer zones).
- Secondary control: Electric reheat maintains minimum supply temperature when primary air is too cold.
- Both controls use proportional feedback with actuator dynamics providing realistic response delays.

Units
-----
- Temperatures: Kelvin [K]
- Airflow: kg/s
- Power, energy, heat: Watts [W]
- Pressure: Pascals [Pa]
- All time: seconds [s]

This module is intended for research, simulation, and educational purposes and is compatible with batched
and differentiable scientific computing in PyTorch.

Author: Aaron Tuor, Claude, Chat-GPT
Date: [YYYY-MM-DD]
"""

import torch
from .base import BuildingComponent
from ..actuators.damper import Damper
from ..actuators.electric_reheat_coil import ElectricReheatCoil


def calculate_reheat_load(airflow: torch.Tensor, cp_air: torch.Tensor,
                          T_current: torch.Tensor, T_min: torch.Tensor) -> torch.Tensor:
    """
    Calculate required reheat coil load to raise supply air temperature to minimum required level.

    This function determines the thermal energy needed to prevent zone overcooling by ensuring
    supply air temperature meets the minimum threshold (typically 55-60°F in real systems).

    Args:
        airflow (torch.Tensor): Air mass flow rate [kg/s], shape (...).
        cp_air (torch.Tensor): Specific heat of air [J/kg/K], shape (...).
        T_current (torch.Tensor): Supply air temp before reheat [K], shape (...).
        T_min (torch.Tensor): Minimum required supply temp after reheat [K], shape (...).

    Returns:
        torch.Tensor: Required reheat load [W] (always >= 0). Zero when T_current >= T_min.
    """
    delta_T = torch.clamp(T_min - T_current, min=0.0)
    return airflow * cp_air * delta_T


class VAVBox(BuildingComponent):
    """
    Physics-based differentiable model of a Variable Air Volume (VAV) terminal unit with optional electric reheat coil.

    This component models the steady-state behavior of a VAV terminal, calculating supply air conditions
    (temperature, flow rate, pressure) delivered to building zones based on zone temperature feedback and
    upstream supply conditions. Actuator dynamics are handled by sub-components.

    Actuator States (maintained by sub-components):
        - damper_position: Damper actuator position [0-1] (managed by Damper component).
        - reheat_position: Electric reheat coil actuator position [0-1] (managed by ElectricReheatCoil component).

    External Inputs (shape [batch_size, n_zones]):
        - T_zone: Current zone air temperature [K].
        - T_setpoint: Zone temperature setpoint [K].
        - T_supply: Primary supply air temperature from central system [K].
        - P_duct: Upstream duct static pressure [Pa].

    Zone-Specific Parameters (expandable to [n_zones] vectors):
        - airflow_min: Minimum supply airflow [kg/s].
        - airflow_max: Maximum supply airflow [kg/s].
        - control_gain: Proportional gain for airflow control [-].
        - tau_damper: Damper actuator time constant [s].
        - Q_reheat_max: Maximum reheat coil thermal output [W].
        - reheat_efficiency: Electric reheat coil electrical efficiency [0-1].
        - tau_reheat: Electric reheat coil actuator time constant [s].

    Shared Parameters (scalars):
        - cp_air: Specific heat of air [J/kg/K].
        - P_nominal: Reference duct static pressure for damper sizing [Pa].

    Primary Outputs (FluidState - shape [batch_size, n_zones]):
        - T_supply: Final supply air temperature delivered to zones [K].
        - Q_supply_flow: Thermal energy content of supply air stream [W].
        - P_supply: Supply air pressure at zone inlet [Pa].
        - supply_airflow: Actual delivered airflow [kg/s].

    Diagnostic Outputs (available via .diagnostics property):
        - damper_position: Current damper actuator position [0-1].
        - reheat_position: Current reheat coil actuator position [0-1].
        - total_power: Total electrical power consumption [W].

    Control Behavior:
        - Mode Detection: Automatically detects cooling vs heating mode based on supply air temperature.
        - Cooling Mode: Increases airflow when zone temperature exceeds setpoint (cooling demand).
        - Heating Mode: Increases airflow when zone temperature is below setpoint (heating demand).
        - Reheat Control: Prevents overcooling (cooling mode) or provides temperature boost (heating mode).
        - Both controls include realistic actuator response delays via sub-component dynamics.
    """

    # --- Class variable metadata for simulation and introspection ---
    _state_ranges = {
        "damper_position": (0.0, 1.0),  # [0-1]
        "reheat_position": (0.0, 1.0),  # [0-1]
    }
    _external_ranges = {
        "T_zone": (283.15, 308.15),  # [K] zone temp (10-35°C)
        "T_setpoint": (291.15, 300.15),  # [K] setpoint (18-27°C)
        "T_supply_upstream": (280.15, 291.15),  # [K] supply air from central air handler (7-18°C)
        "P_duct": (200.0, 1500.0),  # [Pa] upstream pressure
    }

    _zone_param_ranges = {
        # Zone-specific parameters (expanded to [n_zones] vectors)
        "airflow_min": (0.01, 0.5),  # [kg/s]
        "airflow_max": (0.1, 2.0),  # [kg/s]
        "control_gain": (1.0, 5.0),  # [-] proportional gain
        "tau_damper": (1.0, 60.0),  # [s] damper time constant
        "Q_reheat_max": (0.0, 5000.0),  # [W] max reheat output
        "reheat_efficiency": (0.7, 1.0),  # [-] electrical efficiency
        "tau_reheat": (5.0, 120.0),  # [s] reheat time constant
    }
    _param_ranges = {
        # Shared parameters (scalars)
        "cp_air": (990.0, 1020.0),  # [J/kg/K] specific heat of air
        "P_nominal": (200.0, 1500.0),  # [Pa] reference pressure
        "mode_deadband": (2.0, 8.0),  # [K] deadband for mode switching
        "T_supply_cooling_min": (285.15, 291.15),  # [K] min supply temp in cooling mode (12-18°C)
    }

    _output_ranges = {
        # Primary FluidState outputs
        "T_supply": (283.15, 323.15),  # [K] Supply air temperature to zones
        "Q_supply_flow": (-10e3, 50e3),  # [W] Thermal energy in supply air stream
        "P_supply": (200.0, 2000.0),  # [Pa] Supply air pressure
        "supply_airflow": (0.0, 2.0),  # [kg/s] Actual delivered airflow

        # Diagnostic outputs
        "total_power": (0.0, 5500.0),  # [W] Total power consumption
    }

    def __init__(
            self,
            n_zones: int = 1,
            # Zone-specific parameters (can be scalar or list[n_zones])
            control_gain=2.0,
            airflow_min=0.05,
            airflow_max=0.3,
            tau_damper=5.0,
            Q_reheat_max=3000.0,
            reheat_efficiency=0.98,
            tau_reheat=10.0,
            # Shared parameters (scalars only)
            cp_air=1005.0,
            P_nominal=500,
            mode_deadband=5.0,
            T_supply_cooling_min=288.15,  # 15°C (60°F)
            # String parameters (not expanded)
            heating_characteristic='linear',
            actuator_model='smooth_approximation',
            # Standard parameters
            learnable: dict = None,
            device=None,
            dtype=torch.float32,
    ):
        """
        Initialize VAVBox component with zone vectorization support.

        The VAV box delegates actuator dynamics to specialized sub-components (Damper and ElectricReheatCoil)
        while implementing the overall control logic and supply air conditioning calculations.

        ZONE PARAMETER HANDLING:
        - Scalar parameters are automatically expanded to all zones by the base class
        - List parameters must have length n_zones for zone-specific values
        - Sub-components receive properly shaped parameter tensors

        Args:
            n_zones (int): Number of zones served by this VAV system.
            control_gain (float or list): Proportional gain for airflow control [-].
                Controls sensitivity of airflow response to temperature error.
            airflow_min (float or list): Minimum supply airflow [kg/s].
                Maintains minimum ventilation even when no cooling is needed.
            airflow_max (float or list): Maximum supply airflow [kg/s].
                Limits maximum cooling capacity and fan energy.
            tau_damper (float or list): Damper actuator time constant [s].
                Controls speed of airflow response - passed to Damper component.
            Q_reheat_max (float or list): Max electric reheat coil thermal output [W].
                Maximum heating available to prevent overcooling.
            reheat_efficiency (float or list): Electric reheat coil electrical efficiency [0-1].
                Conversion efficiency from electrical to thermal power.
            tau_reheat (float or list): Electric reheat coil actuator time constant [s].
                Controls speed of reheat response - passed to ElectricReheatCoil component.
            cp_air (float): Specific heat of air [J/kg/K] - shared across all zones.
            P_nominal (float): Reference duct static pressure [Pa] for damper sizing.
            mode_deadband (float): Temperature deadband [K] for mode detection to prevent hunting.
            T_supply_cooling_min (float): Minimum supply temperature [K] in cooling mode for reheat control.
            heating_characteristic (str): Electric reheat coil heating curve - passed to sub-component.
            actuator_model (str): Actuator model type - passed to both sub-components.
            learnable (dict): Parameters to make learnable for optimization/learning applications.
            device (torch.device): Device for tensor computations.
            dtype (torch.dtype): Tensor data type for computation.
        """
        super().__init__(params=locals(), learnable=learnable, device=device, dtype=dtype)

        # Initialize sub-components with zone-specific parameters
        self.damper = Damper(
            max_airflow=self.airflow_max,         # [n_zones] tensor from base class
            tau=self.tau_damper,                  # [n_zones] tensor from base class
            flow_characteristic='linear',
            nominal_pressure=self.P_nominal,
            actuator_model=actuator_model
        )

        self.electric_reheat_coil = ElectricReheatCoil(
            max_thermal_output=self.Q_reheat_max,      # [n_zones] tensor from base class
            heating_characteristic=heating_characteristic,
            electrical_efficiency=self.reheat_efficiency, # [n_zones] tensor from base class
            tau=self.tau_reheat,                          # [n_zones] tensor from base class
            actuator_model=actuator_model
        )

    def forward(
            self, *,
            t: float,
            T_zone: torch.Tensor,            # [batch_size, n_zones]
            T_setpoint: torch.Tensor,        # [batch_size, n_zones]
            T_supply_upstream: torch.Tensor,          # [batch_size, 1]
            P_duct: torch.Tensor,            # [batch_size, 1]
            damper_position: torch.Tensor,   # [batch_size, n_zones]
            reheat_position: torch.Tensor,   # [batch_size, n_zones]
            dt: float = 1.0,
    ) -> dict:
        """
        Calculate VAV box supply air conditions for one simulation time step.

        Implements mode detection and control logic, delegating actuator dynamics to sub-components.
        Returns complete FluidState for downstream zone models or other building components.

        CONTROL SEQUENCE:
        1. Detect operating mode (cooling vs heating) based on supply air temperature
        2. Calculate airflow demand using mode-appropriate control logic
        3. Update damper position and calculate actual airflow (via Damper component)
        4. Calculate reheat demand based on operating mode
        5. Update reheat position and calculate thermal output (via ElectricReheatCoil component)
        6. Calculate final supply air conditions (temperature, pressure, heat content)

        MODE DETECTION:
        - Cooling mode: T_supply_upstream < (T_setpoint - mode_deadband) → cold supply air from central AHU
        - Heating mode: T_supply_upstream >= (T_setpoint - mode_deadband) → warm supply air from central AHU

        CONTROL LOGIC BY MODE:
        - Cooling mode: More airflow when zone is warmer than setpoint (cooling demand)
        - Heating mode: More airflow when zone is colder than setpoint (heating demand)
        - Reheat: Prevents overcooling (cooling mode) or provides temperature boost (heating mode)

        TENSOR SHAPES:
        - Zone-specific inputs: [batch_size, n_zones]
        - Supply-side inputs: [batch_size, n_zones] or [batch_size, 1] (auto-broadcast)
        - All outputs: [batch_size, n_zones]

        Args:
            t (float): Current simulation time [s] (passed to sub-components for time-based logic).
            T_zone (torch.Tensor): [batch_size, n_zones] Zone air temperature [K].
            T_setpoint (torch.Tensor): [batch_size, n_zones] Zone temperature setpoint [K].
            T_supply_upstream (torch.Tensor): [batch_size, n_zones or 1] Primary supply air temperature [K].
            P_duct (torch.Tensor): [batch_size, n_zones or 1] Upstream duct static pressure [Pa].
            dt (float): Time step [s] (passed to sub-components for actuator dynamics).

        Returns:
            dict: Complete FluidState and diagnostics. All tensors shape [batch_size, n_zones].
                Primary outputs: T_supply, Q_supply_flow, P_supply, supply_airflow
                Diagnostic outputs available via .diagnostics property after calling forward().
        """
        # --- MODE DETECTION ---
        # Detect cooling vs heating mode based on supply air temperature relative to setpoint
        cooling_mode = T_supply_upstream < (T_setpoint - self.mode_deadband)  # [batch_size, n_zones]

        # --- MODE-DEPENDENT AIRFLOW CONTROL ---
        T_error = T_setpoint - T_zone  # [batch_size, n_zones] - positive when zone is cold

        # Cooling mode: More airflow when zone is warm (negative T_error)
        # Heating mode: More airflow when zone is cold (positive T_error)
        control_error = torch.where(
            cooling_mode,
            -T_error,  # Cooling: airflow increases when zone is warmer than setpoint
            T_error    # Heating: airflow increases when zone is colder than setpoint
        )

        # Apply proportional control with zone-specific gains
        normalized_demand = torch.clamp(control_error * self.control_gain, 0.0, 1.0)
        airflow_target = self.airflow_min + (self.airflow_max - self.airflow_min) * normalized_demand

        # --- DAMPER ACTUATOR UPDATE ---
        damper_result = self.damper.forward(
            t=t,
            target_airflow=airflow_target,
            current_position=damper_position,
            dt=dt,
            duct_pressure=P_duct
        )

        airflow_actual = damper_result["airflow"]
        damper_position = damper_result["position"]

        # --- MODE-DEPENDENT REHEAT CONTROL ---
        Q_reheat_needed = torch.where(
            cooling_mode,
            # Cooling mode: Prevent overcooling by maintaining minimum supply temperature
            calculate_reheat_load(
                airflow=airflow_actual,
                cp_air=self.cp_air,
                T_current=T_supply_upstream,
                T_min=torch.full_like(T_setpoint, self.T_supply_cooling_min)
            ),
            # Heating mode: Boost heating when central supply isn't sufficient
            calculate_reheat_load(
                airflow=airflow_actual,
                cp_air=self.cp_air,
                T_current=T_supply_upstream,
                T_min=T_setpoint + 2.0  # Heat supply air slightly above setpoint for heating
            )
        )

        # --- ELECTRIC REHEAT COIL ACTUATOR UPDATE ---
        reheat_result = self.electric_reheat_coil.forward(
            t=t,
            target_heating=Q_reheat_needed,
            current_position=reheat_position,
            dt=dt
        )

        reheat_position = reheat_result["position"]
        Q_reheat_actual = reheat_result["thermal_output"]
        reheat_electrical_power = reheat_result["electrical_power"]

        # --- CALCULATE FINAL SUPPLY CONDITIONS (FLUIDSTATE) ---
        T_supply_final = torch.where(
            airflow_actual > 1e-6,
            torch.clamp(
                T_supply_upstream + Q_reheat_actual / (airflow_actual * self.cp_air),
                min=T_supply_upstream,  # Never below incoming supply temp
                max=T_supply_upstream + 25.0  # Reasonable upper limit (25K rise)
            ),
            T_supply_upstream
        )

        # Calculate supply air heat flow (thermal energy content)
        Q_supply_flow = airflow_actual * self.cp_air * T_supply_final

        # Calculate supply air pressure (airflow-dependent pressure drop model)
        # Pressure drop increases with airflow - typical VAV box pressure drop curve
        pressure_drop_coeff = 100.0  # Base pressure drop coefficient [Pa/(kg/s)]
        P_supply = P_duct - pressure_drop_coeff * airflow_actual

        return {
            # ================================================================
            # FLUIDSTATE OUTPUTS - Primary interface for downstream equipment
            # ================================================================
            "T_supply": T_supply_final,         # [K] Final supply air temperature
            "Q_supply_flow": Q_supply_flow,     # [W] Thermal energy in supply air stream
            "P_supply": P_supply,               # [Pa] Supply air pressure
            "supply_airflow": airflow_actual,   # [kg/s] Actual delivered airflow
            # ====================
            # Diagnostic outputs
            # ====================
            "damper_position": damper_position,
            "reheat_position": reheat_position,
            "total_power": reheat_electrical_power.mean(dim=-1, keepdim=True),
        }

    def initial_state_functions(self, mode="realistic"):
        """
        Return functions for sampling intelligent initial states using context.

        Args:
            mode: Initialization strategy
                - "realistic": Realistic operating conditions with small variations
                - "steady_state": Ideal steady-state conditions
                - "random": Uniform random from state ranges (fallback to base class)
        """
        return {
                "damper_position": lambda bs: self._sample_damper_position(bs, mode),
                "reheat_position": lambda bs: self._sample_reheat_position(bs, mode),
        }

    def _sample_damper_position(self, batch_size, mode):
        """Sample initial damper positions based on context."""
        # Use context airflow fraction if available
        airflow_fraction = self.context["supply_airflow_fraction"]
        base_position = torch.full((batch_size, self.n_zones), airflow_fraction,
                                   device=self.device, dtype=self.dtype)
        if mode == "steady_state":
            return torch.clamp(base_position, 0.0, 1.0)
        elif mode == "realistic":
            noise = torch.normal(0.0, 0.05, (batch_size, self.n_zones),
                                 device=self.device, dtype=self.dtype)
            return torch.clamp(base_position + noise, 0.1, 1.0)

    def _sample_reheat_position(self, batch_size, mode):
        """Sample initial reheat coil positions based on context system mode."""
        system_mode = self.context.get("system_mode", "cooling")

        if system_mode == "heating":
            # Heating mode: moderate reheat usage
            base_position = 0.4
        elif system_mode == "setback":
            # Setback mode: minimal reheat
            base_position = 0.1
        elif system_mode == "economizer":
            # Economizer mode: light reheat to prevent overcooling
            base_position = 0.2
        else:  # "cooling" or "minimal"
            # Cooling mode: minimal reheat
            base_position = 0.05

        base_tensor = torch.full((batch_size, self.n_zones), base_position,
                                 device=self.device, dtype=self.dtype)

        if mode == "steady_state":
            return torch.clamp(base_tensor, 0.0, 1.0)
        elif mode == "realistic":
            noise = torch.normal(0.0, 0.1, (batch_size, self.n_zones),
                                 device=self.device, dtype=self.dtype)
            return torch.clamp(base_tensor + noise, 0.0, 1.0)

    @property
    def input_functions(self):
        """
        Context-aware input functions for VAVBox component.
        Returns functions that generate tensors with proper shapes for VAV box operation.

        Returns:
            dict: Mapping from input variable names to callables (t, batch_size) -> torch.Tensor.
                  All functions return [batch_size, n_zones] for consistent logging in simulation.
        """
        if not hasattr(self, "_input_functions"):
            # Get context values with fallbacks
            T_setpoint_base = self.context.get("T_setpoint_base", 293.15)  # Default: 20°C
            T_supply_base = self.context.get("T_supply_base", 286.15)  # Default: 13°C
            day_of_year = self.context.get("day_of_year", 100)
            occupancy_state = self.context.get("occupancy_state", "occupied")
            system_mode = self.context.get("system_mode", "cooling")
            P_duct_base = self.context.get("P_duct_base", 600.0)  # Default: 600 Pa

            def day_of_year_fn(t):
                """Context-aware day of year with progression."""
                # Start from context day and progress with simulation time
                sim_days = t / 86400.0
                current_day = (day_of_year + sim_days) % 365
                if current_day == 0:
                    current_day = 365
                return current_day

            def T_zone_fn(t, batch_size=1):
                current_hour = (t / 3600.0) % 24
                day_of_year = day_of_year_fn(t)
                # Zone temperature varies around setpoint based on occupancy and time
                if occupancy_state == "occupied":
                    if 8 <= current_hour <= 17:  # Business hours
                        temp_rise = 1.5  # Zones run slightly warm during peak occupancy
                    elif 7 <= current_hour < 8 or 17 < current_hour <= 19:  # Transition
                        temp_rise = 0.5  # Slight temperature rise
                    else:
                        temp_rise = -0.5  # Cooler at night
                elif occupancy_state == "unoccupied":
                    temp_rise = -1.0  # Zones drift from setpoint when unoccupied
                else:  # "transition"
                    temp_rise = 0.0  # Near setpoint during startup

                # Add small daily variation and solar effects
                daily_variation = 0.5 * torch.sin(torch.tensor(2 * torch.pi * (current_hour - 14) / 24))
                seasonal_variation = 0.3 * torch.sin(torch.tensor(2 * torch.pi * (day_of_year - 80) / 365))

                T_zone = T_setpoint_base + temp_rise + daily_variation + seasonal_variation
                return torch.full((batch_size, self.n_zones), T_zone,
                                  device=self.device, dtype=self.dtype)

            def T_setpoint_fn(t, batch_size=1):
                """Context-aware zone setpoints with schedule adjustments."""
                current_hour = (t / 3600.0) % 24

                if occupancy_state == "occupied":
                    if 8 <= current_hour <= 17:  # Business hours
                        setpoint = T_setpoint_base  # Normal setpoint
                    else:
                        setpoint = T_setpoint_base + 1.0  # Slightly relaxed after hours
                elif occupancy_state == "unoccupied":
                    # Setback: warmer in summer (less cooling), cooler in winter (less heating)
                    if system_mode in ["cooling", "economizer"]:
                        setpoint = T_setpoint_base + 3.0  # Warmer setback for cooling
                    else:  # heating
                        setpoint = T_setpoint_base - 2.0  # Cooler setback for heating
                else:  # "transition"
                    setpoint = T_setpoint_base + 0.5  # Moderate setback during startup

                return torch.full((batch_size, self.n_zones), setpoint,
                                  device=self.device, dtype=self.dtype)

            def T_supply_upstream_fn(t, batch_size=1):
                """Context-aware upstream supply temperature from RTU."""
                current_hour = (t / 3600.0) % 24

                # Supply temperature varies with load and system mode
                if occupancy_state == "occupied":
                    if 8 <= current_hour <= 17:  # Peak hours
                        supply_temp = T_supply_base  # Design supply temperature
                    else:
                        # Warmer supply air after hours (less aggressive conditioning)
                        supply_temp = T_supply_base + 2.0
                elif occupancy_state == "unoccupied":
                    # Much warmer supply during unoccupied (setback mode)
                    supply_temp = T_supply_base + 5.0
                else:  # "transition"
                    # Moderate supply temperature during startup
                    supply_temp = T_supply_base + 1.0

                # Seasonal adjustment for realistic operation
                seasonal_adjustment = 1.0 * torch.sin(torch.tensor(2 * torch.pi * (day_of_year - 80) / 365))
                supply_temp = supply_temp + seasonal_adjustment

                return torch.full((batch_size, 1), supply_temp,
                                  device=self.device, dtype=self.dtype)

            def P_duct_fn(t, batch_size=1):
                """Context-aware duct pressure based on system operation."""
                current_hour = (t / 3600.0) % 24

                # Duct pressure varies with system load
                if occupancy_state == "occupied":
                    if 8 <= current_hour <= 17:  # Business hours
                        duct_pressure = P_duct_base  # Design pressure
                    else:
                        duct_pressure = P_duct_base * 0.7  # Reduced pressure after hours
                elif occupancy_state == "unoccupied":
                    duct_pressure = P_duct_base * 0.4  # Low pressure during unoccupied
                else:  # "transition"
                    duct_pressure = P_duct_base * 0.6  # Moderate pressure during startup

                return torch.full((batch_size, 1), duct_pressure,
                                  device=self.device, dtype=self.dtype)

            self._input_functions = {
                "T_zone": T_zone_fn,
                "T_setpoint": T_setpoint_fn,
                "T_supply_upstream": T_supply_upstream_fn,
                "P_duct": P_duct_fn,
            }

        return self._input_functions

    # @input_functions.setter
    # def input_functions(self, value):
    #     """Allow custom input functions to be set."""
    #     if not hasattr(self, '_input_functions'):
    #         self._input_functions = {}
    #     self._input_functions.update(value)