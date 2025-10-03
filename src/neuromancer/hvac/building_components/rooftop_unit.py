"""
rooftop_unit.py - RTU (Rooftop Unit) Model with Context-Driven Initialization

This module implements a physics-based, differentiable model for packaged rooftop units (RTUs),
with context-driven initialization for coordinated building system startup.

CONTEXT-DRIVEN INITIALIZATION:
The RTU now accepts a "context" parameter in its __init__ method that provides:
- Environmental conditions (outdoor temperature, time of day, weather)
- Building operating state (occupancy, system mode, setpoints)
- System operating points (airflows, pressures)

This ensures the RTU starts with initial conditions and input functions that are
physically consistent with other building components and the intended simulation scenario.

SINGLE-UNIT ARCHITECTURE:
- RTU produces one supply air stream for distribution to multiple zones
- Zone-specific control is handled downstream by terminal units (VAV boxes, dampers)
- All RTU parameters are shared (no zone-specific RTU parameters)
- Return air mixing calculated from zone temperatures and return airflows
- Output tensors shaped for broadcast to downstream zone equipment

Overview:
    Rooftop units integrate all HVAC components (compressor, coils, fan, controls) in a
    weatherproof package for outdoor installation. Unlike central air handling units (AHUs)
    that use chilled/hot water from central plants, RTUs typically use direct expansion (DX)
    cooling and gas-fired heating. This makes them popular for smaller buildings due to
    lower installation cost and system complexity.

Physical System Description:
    - **Air Mixing Section**: Combines outdoor air (for ventilation) with return air
      (mixed from zones based on return airflows) in proportions controlled by mixing dampers
    - **Heating Coil**: Gas-fired furnace or electric resistance heating elements
      that warm mixed air during heating season
    - **Cooling Coil**: Direct expansion (DX) evaporator coil with refrigerant
      that cools and dehumidifies air during cooling season
    - **Supply Fan**: Centrifugal or axial fan that moves conditioned air through
      ductwork to building zones
    - **Controls**: Integrated controller managing dampers, valves, and fan operation
      based on zone temperature and supply air setpoints

System Architecture:
    RTU → Ductwork → Terminal Units (VAV boxes) → Zones
    Zones → Terminal Units → Return Ductwork → RTU

    The RTU provides conditioned supply air that gets distributed to zones.
    Zone-specific control happens at the terminal units, not at the RTU level.
    Return air from zones is mixed based on actual return airflows.

Units:
    - Temperature: Kelvin [K]
    - Airflow: kg/s (mass flow rate)
    - Power, thermal loads: Watts [W]
    - Pressure: Pascals [Pa]
    - Time: seconds [s]
    - Actuator positions: 0-1 (normalized)
    - Efficiency/COP: dimensionless ratios
"""

import torch
from .base import BuildingComponent
from ..actuators.actuator import Actuator
from ..simulation_inputs.schedules import seasonal_temperature

# =============================================================================
# POWER CALCULATION FUNCTIONS
# =============================================================================

def calculate_fan_power(airflow: torch.Tensor, fan_power_per_flow: torch.Tensor) -> torch.Tensor:
    """Calculate electrical power consumption for air moving equipment."""
    return airflow * fan_power_per_flow


def calculate_cooling_power(Q_cooling_load: torch.Tensor, cooling_COP: torch.Tensor) -> torch.Tensor:
    """Calculate electrical power input for mechanical cooling equipment."""
    return Q_cooling_load / cooling_COP


def calculate_heating_power(Q_heating_load: torch.Tensor, heating_efficiency: torch.Tensor) -> torch.Tensor:
    """Calculate fuel or electrical power input for heating equipment."""
    return Q_heating_load / heating_efficiency

def calculate_hvac_power(actual_airflow: torch.Tensor, Q_cooling_load: torch.Tensor,
                         Q_heating_load: torch.Tensor, fan_power_per_flow: torch.Tensor,
                         cooling_COP: torch.Tensor, heating_efficiency: torch.Tensor) -> dict:
    """Calculate comprehensive power breakdown for central HVAC equipment."""
    fan_power = calculate_fan_power(actual_airflow, fan_power_per_flow)
    cooling_power = calculate_cooling_power(Q_cooling_load, cooling_COP)
    heating_power = calculate_heating_power(Q_heating_load, heating_efficiency)
    total_power = fan_power + cooling_power + heating_power

    return {
        "fan_power": fan_power,
        "cooling_power": cooling_power,
        "heating_power": heating_power,
        "total_power": total_power
    }

# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================


def calculate_air_mixing(airflow: torch.Tensor, airflow_oa_min: torch.Tensor,
                         T_oa: torch.Tensor, T_ra: torch.Tensor) -> tuple:
    """Calculate air mixing for HVAC components that combine outdoor and return air."""
    # Handle zero airflow case
    zero_flow_mask = airflow < 1e-6

    # Calculate outdoor air fraction (clamped to valid range)
    oa_fraction = torch.clamp(airflow_oa_min / (airflow + 1e-8), 0.0, 1.0)

    # For zero airflow, use 100% outdoor air
    oa_fraction = torch.where(zero_flow_mask, torch.ones_like(oa_fraction), oa_fraction)

    # Mixed temperature = weighted average
    T_mixed_air = oa_fraction * T_oa + (1 - oa_fraction) * T_ra

    return oa_fraction, T_mixed_air


def calculate_thermal_load(airflow: torch.Tensor, cp_air: torch.Tensor,
                           setpoint_temp: torch.Tensor, current_temp: torch.Tensor) -> torch.Tensor:
    """Calculate thermal load required to change air temperature to setpoint."""
    return airflow * cp_air * (setpoint_temp - current_temp)


def apply_mode_limits(thermal_load: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply HVAC operating mode restrictions to thermal load."""
    if mode == "cool":
        return torch.clamp(thermal_load, max=0.0)  # Only cooling
    elif mode == "heat":
        return torch.clamp(thermal_load, min=0.0)  # Only heating
    else:  # mode == "auto"
        return thermal_load  # No restrictions


def calculate_economizer_fraction(T_oa, T_ra, min_oa_fraction, T_economizer_max=297.15, ctrl_economizer_deadband=2.0):
    """
    Calculate outdoor air fraction for economizer operation.

    Args:
        T_oa: Outdoor air temperature [K]
        T_ra: Return air temperature [K]
        min_oa_fraction: Minimum outdoor air fraction for ventilation
        T_economizer_max: High temperature limit for economizer lockout [K]
        ctrl_economizer_deadband: Temperature deadband to prevent cycling [K]

    Returns:
        Enhanced outdoor air fraction for economizer operation
    """
    # Check economizer conditions
    too_hot_outside = T_oa > T_economizer_max
    insufficient_delta_t = T_oa >= (T_ra - ctrl_economizer_deadband)

    # If economizer conditions not met, use minimum OA
    if too_hot_outside.any() or insufficient_delta_t.any():
        return min_oa_fraction

    # Calculate enhanced fraction based on temperature difference
    delta_T = T_ra - T_oa
    enhancement = torch.clamp(delta_T * 0.1, max=0.7)  # Up to 70% enhancement
    enhanced_fraction = min_oa_fraction + enhancement

    return torch.clamp(enhanced_fraction, max=1.0)


class RTU(BuildingComponent):
    """
    Rooftop Unit (RTU): Single packaged air handler serving multiple zones.

    SINGLE-UNIT ARCHITECTURE:
    - RTU produces one supply air stream for distribution to multiple zones
    - Zone-specific control is handled downstream by terminal units (VAV boxes, dampers)
    - All RTU parameters are shared (no zone-specific RTU parameters)
    - Return air mixing calculated from zone temperatures and return airflows
    - Output tensors shaped for broadcast to downstream zone equipment

    Physical System:
        Models a self-contained rooftop HVAC unit with integrated heating,
        cooling, and air handling. RTUs combine outdoor and return air,
        condition it through coils, and deliver it to a supply duct system
        for distribution to multiple building zones.

    States:
        - T_supply (computed state): Supply air temperature [K]
        - damper_position (computed state): Airflow damper position [0-1]
        - valve_position (computed state): Coil valve position [0-1]
        - integral_accumulator (computed state): PI controller integral state [-]

    External Inputs:
        - T_outdoor: Outdoor air temperature [K], shape [batch_size, 1]
        - T_return_zones: Zone air temperatures from envelope [K], shape [batch_size, n_zones]
        - return_airflow_zones: Return airflow from each zone [kg/s], shape [batch_size, n_zones]
        - T_supply_setpoint: Desired supply air temperature [K], shape [batch_size, 1]
        - supply_airflow_setpoint: Desired total airflow rate [kg/s], shape [batch_size, 1]

    Parameters (all shared, no zone-specific parameters):
        - airflow_max: Maximum total airflow capacity [kg/s]
        - airflow_oa_min: Minimum outdoor air for ventilation [kg/s]
        - Q_coil_max: Maximum coil thermal capacity [W]
        - C_air: Thermal mass of air in unit [J/K]
        - R_env: Thermal resistance to outdoor environment [K/W]
        - fan_power_per_flow: Specific fan power [W/(kg/s)]
        - cooling_COP: Cooling coefficient of performance [-]
        - heating_efficiency: Heating efficiency [-]
        - tau_airflow: Damper actuator time constant [s]
        - tau_coil: Valve actuator time constant [s]
        - cp_air: Specific heat capacity of air [J/kg/K]
        - coil_eff: Coil effectiveness [-]

    Outputs (scalars or [batch_size, 1] for broadcast to zones):
        - T_supply: Final supply air temperature [K]
        - supply_heat_flow: Thermal energy in supply air stream [W]
        - P_supply: Supply duct pressure [Pa]
        - supply_airflow: Actual delivered airflow [kg/s]
    """

    _computed_state_vars = ["damper_position", "valve_position", "T_supply", "integral_accumulator"]
    _state_shapes = {
        "damper_position": ("batch", 1),
        "valve_position": ("batch", 1),
        "T_supply": ("batch", 1),
        "integral_accumulator": ("batch", 1),
    }

    # All parameters are shared (no zone-specific parameters)
    _param_ranges = {
        "airflow_max": (0.5, 5.0),  # [kg/s] Maximum total airflow capacity
        "airflow_oa_min": (0.0, 2.0),  # [kg/s] Minimum outdoor air for ventilation
        "Q_coil_max": (5e3, 25e3),  # [W] Maximum coil capacity
        "C_air": (5e3, 5e4),  # [J/K] Thermal mass
        "R_env": (0.01, 0.5),  # [K/W] Thermal resistance to outdoor
        "fan_power_per_flow": (500.0, 2000.0),  # [W/(kg/s)] Fan efficiency
        "cooling_COP": (2.5, 4.5),  # [–] Cooling COP
        "heating_efficiency": (0.7, 0.95),  # [–] Heating efficiency
        "tau_airflow": (5.0, 30.0),  # [s] Damper time constant
        "tau_coil": (10.0, 60.0),  # [s] Valve time constant
        "cp_air": (990.0, 1020.0),  # [J/kg/K] Specific heat of air
        "coil_eff": (0.7, 1.0),  # [–] Coil efficiency
        "ctrl_proportional_band": (5.0, 20.0),  # [K] PI controller proportional band
        "ctrl_integral_time": (60.0, 600.0),  # [s] PI controller integral time
        "ctrl_deadband": (0.5, 3.0),  # [K] Control deadband
        "ctrl_economizer_deadband": (1.0, 5.0),  # [K] Economizer deadband
        "T_economizer_max": (290.15, 305.15),  # [K] Economizer high limit
    }

    # Variable ranges
    _state_ranges = {
        "T_supply": (273.15 + 5, 273.15 + 50),  # [K] Supply air temperature
        "damper_position": (0.0, 1.0),  # [0-1] Damper position
        "valve_position": (0.0, 1.0),   # [0-1] Valve position
        "integral_accumulator": (-1.0, 1.0),  # [-] PI controller integral state
    }

    _external_ranges = {
        "T_outdoor": (273.15 - 10, 273.15 + 45),  # [K] Outdoor air temperature
        "T_return_zones": (273.15 + 10, 273.15 + 35),  # [K] Zone air temperatures from envelope
        "return_airflow_zones": (0.0, 2.0),  # [kg/s] Return airflow from each zone
        "T_supply_setpoint": (273.15 + 10, 273.15 + 25),  # [K] Desired supply temp
        "supply_airflow_setpoint": (0.0, 5.0),  # [kg/s] Desired total airflow
    }

    _output_ranges = {
        "fan_power": (0.0, 10e3),  # Supply fan electrical power consumption [W]
        "cooling_power": (0.0, 10e3),  # Cooling compressor electrical power consumption [W]
        "heating_power": (0.0, 30e3),  # Heating fuel/electrical power consumption [W]
        "total_power": (0.0, 50e3),  # Total RTU power consumption [W]
        # =========================================================================
        # RETURN VALUES - Primary outputs for downstream equipment
        # =========================================================================
        "supply_airflow": (0.0, 5.0),  # Actual total supply airflow delivered [kg/s]
        "supply_heat_flow": (-50e3, 100e3),  # Thermal energy in supply air stream [W]
        "P_supply": (100e3, 105e3),  # Supply duct pressure [Pa]
    }

    def __init__(
            self,
            # RTU parameters
            airflow_max: float = 3.0,  # [kg/s] Maximum total airflow capacity
            airflow_oa_min: float = 0.2,  # [kg/s] Minimum outdoor air for ventilation
            Q_coil_max: float = 15e3,  # [W] Maximum coil thermal capacity
            C_air: float = 2e4,  # [J/K] Thermal mass of air volume within RTU
            R_env: float = 0.1,  # [K/W] Thermal resistance to outdoor environment
            fan_power_per_flow: float = 1000.0,  # [W/(kg/s)] Specific fan power
            cooling_COP: float = 3.0,  # [-] Cooling coefficient of performance
            heating_efficiency: float = 0.85,  # [-] Heating efficiency
            tau_airflow: float = 10.0,  # [s] Damper actuator time constant
            tau_coil: float = 15.0,  # [s] Valve actuator time constant
            cp_air: float = 1005.0,  # [J/kg/K] Specific heat capacity of air
            coil_eff: float = 0.9,  # [-] Coil effectiveness

            # Control parameters
            ctrl_proportional_band: float = 20.,  # [K] PI controller proportional band
            ctrl_integral_time: float = 600.0,  # [s] PI controller integral time
            ctrl_deadband: float = 3.0,  # [K] Control deadband
            ctrl_economizer_deadband: float = 2.0,  # [K] Economizer deadband
            T_economizer_max: float = 297.15,  # [K] Economizer high limit

            # System parameters
            n_zones=1,  # [-] Number of zones served (for input function generation)
            actuator_model: str = "smooth_approximation",  # [-] Actuator dynamics model type
            learnable: dict = None,  # [-] Parameters to make learnable
            device=None,  # [-] Device for tensor computations
            dtype=torch.float32,  # [-] Tensor type for computation
            initial_states=None,

            # Context-driven initialization
            context: dict = None,  # Building operating context for coordinated initialization
        ):

        super().__init__(params=locals(), learnable=learnable, device=device, dtype=dtype)

        # Calculate control gains
        self.ctrl_Kp = 1.0 / self.ctrl_proportional_band  # [-] Proportional gain
        self.ctrl_Ki = self.ctrl_Kp / self.ctrl_integral_time  # [1/s] Integral gain

        # Create single actuators (no zone vectorization)
        self.airflow_damper = Actuator(
            tau=self.tau_airflow,  # [s] Scalar value
            model=actuator_model,
            name="airflow_damper"
        )

        self.coil_valve = Actuator(
            tau=self.tau_coil,  # [s] Scalar value
            model=actuator_model,
            name="coil_valve"
        )

    def forward(
            self, *,
            t: float = 0.,  # [s] Current simulation time
            T_outdoor: torch.Tensor,  # [K] Outdoor air temperature, shape [batch_size, 1]
            T_return_zones: torch.Tensor,  # [K] Zone air temperatures, shape [batch_size, n_zones]
            return_airflow_zones: torch.Tensor,  # [kg/s] Return airflow from zones, shape [batch_size, n_zones]
            T_supply_setpoint: torch.Tensor,  # [K] Desired supply temperature, shape [batch_size, 1]
            supply_airflow_setpoint: torch.Tensor,  # [kg/s] Desired total airflow, shape [batch_size, 1]
            damper_position: torch.Tensor,  # [0-1],
            valve_position: torch.Tensor,  # [0-1],
            T_supply: torch.Tensor, # [K]
            integral_accumulator: torch.Tensor,
            dt: float = 1.0,  # [s] Time step
    ) -> dict:
        """
        RTU simulation step with integrated economizer and air mixing control.

        Control Sequence:
        1. Calculate mixed return air temperature from zones
        2. Determine optimal economizer operation (OA fraction)
        3. Calculate resulting mixed air temperature
        4. Use hierarchical control: economizer first, then mechanical cooling/heating
        5. Update actuator positions and calculate final supply conditions
        """
        # =====================================================================
        # STEP 1: CALCULATE MIXED RETURN AIR FROM ZONES
        # =====================================================================

        total_return_airflow = return_airflow_zones.sum(dim=-1, keepdim=True)  # [kg/s]
        epsilon = 1e-8  # [kg/s] Small value to prevent division by zero
        safe_total_return = total_return_airflow + epsilon  # [kg/s]
        weighted_temps = T_return_zones * return_airflow_zones  # [K*kg/s]
        total_weighted_temp = weighted_temps.sum(dim=-1, keepdim=True)  # [K*kg/s]
        T_return = total_weighted_temp / safe_total_return  # [K]

        # =====================================================================
        # STEP 2: AIRFLOW CONTROL (INDEPENDENT OF TEMPERATURE CONTROL)
        # =====================================================================

        # Update damper position for airflow control
        airflow_setpoint_norm = torch.clamp(supply_airflow_setpoint / self.airflow_max, 0.0, 1.0)  # [-]

        damper_position = self.airflow_damper.forward(
            t=t,  # [s]
            setpoint=airflow_setpoint_norm,  # [-]
            position=damper_position,  # [-]
            dt=dt  # [s]
        )

        supply_airflow = damper_position * self.airflow_max  # [kg/s]
        self.supply_airflow = supply_airflow
        # =====================================================================
        # STEP 3: ECONOMIZER CONTROL LOGIC
        # =====================================================================

        # Basic outdoor air fraction for ventilation
        min_oa_fraction = torch.clamp(self.airflow_oa_min / (supply_airflow + epsilon), 0.0, 1.0)  # [-]

        # Determine if economizer operation is beneficial
        # Economizer is beneficial when outdoor air is cooler than return air
        economizer_beneficial = (
                (T_outdoor < T_return - self.ctrl_economizer_deadband) &  # [K] < [K] OA significantly cooler
                (T_outdoor < self.T_economizer_max)  # [K] < [K] OA not too hot
        )

        # Calculate optimal OA fraction for economizer operation
        # If economizer is beneficial, increase OA fraction
        if economizer_beneficial.any():
            # Calculate how much free cooling we can get
            supply_temp_error = T_supply_setpoint - T_supply  # [K]
            cooling_needed = supply_temp_error < -self.ctrl_deadband  # [bool]

            # For zones needing cooling, calculate optimal OA fraction
            optimal_oa_fraction = torch.where(
                cooling_needed & economizer_beneficial,
                # Calculate OA fraction needed to meet setpoint (if possible)
                torch.clamp(
                    (T_return - T_supply_setpoint) / (T_return - T_outdoor + epsilon),  # [-]
                    min_oa_fraction, torch.tensor(1.0)  # [-]
                ),
                min_oa_fraction  # [-] Just use minimum ventilation
            )
        else:
            optimal_oa_fraction = min_oa_fraction  # [-]

        # =====================================================================
        # STEP 4: CALCULATE MIXED AIR TEMPERATURE WITH ECONOMIZER
        # =====================================================================

        T_mixed_air = optimal_oa_fraction * T_outdoor + (1 - optimal_oa_fraction) * T_return  # [K]

        # =====================================================================
        # STEP 5: HIERARCHICAL TEMPERATURE CONTROL
        # =====================================================================

        # Calculate temperature error after economizer operation
        temp_error_after_economizer = T_supply_setpoint - T_mixed_air  # [K]

        # Deadband logic
        deadband_active = torch.abs(temp_error_after_economizer) > self.ctrl_deadband  # [bool]

        # PI Controller for mechanical heating/cooling
        # Only acts on remaining error after economizer
        P_term = self.ctrl_Kp * temp_error_after_economizer  # [-]

        # Integral term update (only when outside deadband)
        integral_update = torch.where(
            deadband_active,
            temp_error_after_economizer * dt * self.ctrl_Ki,  # [K] * [s] * [1/s] = [-]
            torch.zeros_like(temp_error_after_economizer)  # [-]
        )

        integral_accumulator = torch.clamp(
            integral_accumulator + integral_update, -1.0, 1.0  # [-]
        )

        I_term = integral_accumulator  # [-]
        control_output = P_term + I_term  # [-]

        # Apply deadband
        control_output = torch.where(deadband_active, control_output, torch.zeros_like(control_output))  # [-]

        # Mode determination
        cooling_mode = temp_error_after_economizer < -self.ctrl_deadband  # [bool]
        heating_mode = temp_error_after_economizer > self.ctrl_deadband  # [bool]

        # Convert to valve position
        coil_setpoint_norm = torch.where(
            cooling_mode,
            torch.clamp(-control_output, 0.0, 1.0),  # Cooling: negate negative error
            torch.where(
                heating_mode,
                torch.clamp(control_output, 0.0, 1.0),  # Heating: use positive error
                torch.zeros_like(control_output)  # No action in deadband
            )
        )


        # =====================================================================
        # STEP 6: UPDATE COIL VALVE POSITION
        # =====================================================================

        valve_position = self.coil_valve.forward(
            t=t,  # [s]
            setpoint=coil_setpoint_norm,  # [-]
            position=valve_position,  # [-]
            dt=dt  # [s]
        )

        # =====================================================================
        # STEP 7: CALCULATE FINAL SUPPLY CONDITIONS
        # =====================================================================

        # Coil heat transfer based on mode and valve position
        Q_coil_available = self.Q_coil_max * valve_position  # [W]
        Q_coil_actual = torch.zeros_like(Q_coil_available)  # [W]

        # Apply coil capacity based on operating mode
        cooling_active = cooling_mode & (valve_position > 0.01)  # [bool]
        Q_coil_actual = torch.where(
            cooling_active,
            -Q_coil_available * self.coil_eff,  # [W] Negative for cooling
            Q_coil_actual  # [W]
        )

        heating_active = heating_mode & (valve_position > 0.01)  # [bool]
        Q_coil_actual = torch.where(
            heating_active,
            Q_coil_available * self.coil_eff,  # [W] Positive for heating
            Q_coil_actual  # [W]
        )

        # Final supply air temperature
        safe_airflow = torch.clamp(supply_airflow, min=0.01)  # [kg/s]
        delta_T_coil = Q_coil_actual / (safe_airflow * self.cp_air)  # [W] / ([kg/s] * [J/kg/K]) = [K]
        T_supply_final = T_mixed_air + delta_T_coil  # [K]
        T_supply_final = torch.clamp(T_supply_final, 250.0, 350.0)  # [K]
        self.T_supply = T_supply_final
        # =====================================================================
        # STEP 8: POWER CALCULATIONS
        # =====================================================================

        Q_cooling_load = torch.clamp(-Q_coil_actual, min=0.0)  # [W]
        Q_heating_load = torch.clamp(Q_coil_actual, min=0.0)  # [W]

        power = calculate_hvac_power(
            actual_airflow=supply_airflow,  # [kg/s]
            Q_cooling_load=Q_cooling_load,  # [W]
            Q_heating_load=Q_heating_load,  # [W]
            fan_power_per_flow=self.fan_power_per_flow,  # [W/(kg/s)]
            cooling_COP=self.cooling_COP,  # [-]
            heating_efficiency=self.heating_efficiency  # [-]
        )

        # =====================================================================
        # STEP 10: SUPPLY CONDITIONS FOR DOWNSTREAM EQUIPMENT
        # =====================================================================

        # Supply air pressure
        normalized_flow = supply_airflow / self.airflow_max  # [-]
        P_fan_rise = 800.0 * normalized_flow ** 2  # [Pa]
        P_supply = torch.full_like(supply_airflow, 101325.0) + P_fan_rise  # [Pa]

        # Supply air heat flow
        supply_heat_flow = supply_airflow * self.cp_air * T_supply_final  # [kg/s] * [J/kg/K] * [K] = [W]

        return {
            # Outputs
            "T_supply": T_supply_final,  # [K]
            "supply_heat_flow": supply_heat_flow,  # [W]
            "P_supply": P_supply,  # [Pa]
            "supply_airflow": supply_airflow,  # [kg/s]

            # Equipment states
            "damper_position": damper_position,  # [-]
            "valve_position": valve_position,  # [-]
            "integral_accumulator": integral_accumulator,  # [-]

            # Diagnostics
            **power  # [W] for all power variables
        }

    def initial_state_functions(self, mode="realistic"):
        """
        Return functions for sampling intelligent initial states using context.

        Args:
            mode: "realistic", "steady_state", or "random"
        """
        if mode == "random":
            return super().initial_state_functions(mode)
        else:
            return {
                "integral_accumulator": lambda bs: self._sample_integral_accumulator(bs, mode),
                "T_supply": lambda bs: self._sample_T_supply(bs, mode),
                "damper_position": lambda bs: self._sample_damper_position(bs, mode),
                "valve_position": lambda bs: self._sample_valve_position(bs, mode),
            }

    def _sample_integral_accumulator(self, batch_size, mode):
        """Initialize integral accumulator based on context and mode."""
        if mode == "steady_state":
            return torch.zeros((batch_size, 1), device=self.device, dtype=self.dtype)
        else:  # realistic
            return torch.normal(0.0, 0.1, (batch_size, 1), device=self.device, dtype=self.dtype).clamp(-0.2, 0.2)

    def _sample_T_supply(self, batch_size, mode):
        """Initialize supply air temperature based on context."""
        # Get context-based supply temperature, fallback to setpoint-based approach
        T_supply_context = self.context["T_supply_base"]
        base_temp = torch.full((batch_size, 1), T_supply_context, device=self.device, dtype=self.dtype)

        if mode == "steady_state":
            return base_temp
        elif mode == 'realistic':
            # Add small variation around context-based temperature
            variation = torch.normal(0.0, 1.0, (batch_size, 1), device=self.device, dtype=self.dtype).clamp(-2.0, 2.0)
            return base_temp + variation

    def _sample_damper_position(self, batch_size, mode):
        """Initialize damper position based on context airflow fraction."""
        airflow_fraction = self.context["supply_airflow_fraction"]
        base_position = torch.full((batch_size, 1), airflow_fraction, device=self.device, dtype=self.dtype)

        if mode == "steady_state":
            return base_position.clamp(0.0, 1.0)
        elif mode == 'realistic':
            variation = torch.normal(0.0, 0.05, (batch_size, 1), device=self.device, dtype=self.dtype)
            return (base_position + variation).clamp(0.0, 1.0)

    def _sample_valve_position(self, batch_size, mode):
        """Initialize valve position based on context system mode."""
        system_mode = self.context.get("system_mode", "cooling")

        if system_mode == "cooling":
            # Moderate cooling valve position
            base_position = 0.3
        elif system_mode == "heating":
            # Moderate heating valve position
            base_position = 0.4
        elif system_mode == "setback":
            # Minimal valve position for setback
            base_position = 0.1
        elif system_mode == "economizer":
            # Low mechanical cooling, relying on free cooling
            base_position = 0.15
        else:  # "minimal" or unknown
            # Very low valve position
            base_position = 0.05

        base_tensor = torch.full((batch_size, 1), base_position, device=self.device, dtype=self.dtype)

        if mode == "steady_state":
            return base_tensor.clamp(0.0, 1.0)
        elif mode == 'realistic':
            variation = torch.normal(0.0, 0.1, (batch_size, 1), device=self.device, dtype=self.dtype)
            return (base_tensor + variation).clamp(0.0, 1.0)

    @property
    def input_functions(self):
        """
        Context-aware input functions for RTU component.

        Returns:
            dict: Mapping from input variable names to callables (t, batch_size) -> torch.Tensor.
        """
        if not hasattr(self, '_input_functions'):
            # Get context values with fallbacks
            T_outdoor_base = self.context.get("T_outdoor", 288.15)  # Default: 15°C
            day_of_year = self.context.get("day_of_year", 100)
            T_setpoint_base = self.context.get("T_setpoint_base", 293.15)  # Default: 20°C
            T_supply_base = self.context.get("T_supply_base", 286.15)  # Default: 13°C
            supply_airflow_fraction = self.context.get("supply_airflow_fraction", 0.5)
            occupancy_state = self.context.get("occupancy_state", "occupied")

            def day_of_year_fn(t):
                """Context-aware day of year with progression."""
                # Start from context day and progress with simulation time
                sim_days = t / 86400.0
                current_day = (day_of_year + sim_days) % 365
                if current_day == 0:
                    current_day = 365
                return current_day

            def T_zones_fn(t, batch_size=1):
                """Context-aware return temperatures based on supply air and building load."""
                # Get current supply temperature from RTU or use context default
                if hasattr(self, 'T_supply') and self.T_supply is not None:
                    T_supply = self.T_supply.expand(batch_size, self.n_zones)
                else:
                    T_supply = torch.full((batch_size, self.n_zones), T_supply_base,
                                          device=self.device, dtype=self.dtype)

                # Zone temperature rise depends on occupancy and time of day
                hour_of_day = (t / 3600.0) % 24

                if occupancy_state == "occupied":
                    # Higher internal gains during occupied hours
                    if 7 <= hour_of_day <= 19:  # Business hours
                        zone_temp_rise = 6.0  # Higher rise due to people, equipment, lights
                    else:
                        zone_temp_rise = 3.0  # Lower rise after hours
                elif occupancy_state == "unoccupied":
                    zone_temp_rise = 2.0  # Minimal internal gains
                else:  # transition
                    zone_temp_rise = 4.0  # Moderate gains during startup

                # Add small solar and time-of-day effects
                solar_effect = 1.0 * torch.sin(torch.tensor(2 * torch.pi * (hour_of_day - 12) / 24))

                T_return = T_supply + zone_temp_rise + solar_effect
                return T_return.to(device=self.device, dtype=self.dtype)

            def zone_return_airflows_fn(t, batch_size=1):
                """Context-aware return airflows based on supply airflow and system mode."""
                # Calculate expected supply airflow based on context
                supply_flow_expected = supply_airflow_fraction * self.airflow_max

                # Get current supply airflow from RTU or use expected
                if hasattr(self, 'supply_airflow') and self.supply_airflow is not None:
                    total_supply = self.supply_airflow
                else:
                    total_supply = torch.full((batch_size, 1), supply_flow_expected,
                                              device=self.device, dtype=self.dtype)

                # Return air is typically 90-95% of supply (some leakage)
                return_efficiency = 0.92
                return_per_zone = return_efficiency * total_supply.expand(-1, self.n_zones)
                return torch.clamp(return_per_zone, min=0.05)

            def T_supply_setpoint_fn(t, batch_size=1):
                """Context-aware supply air setpoint based on system mode and schedule."""
                day_of_year = day_of_year_fn(t)
                if occupancy_state == "occupied":
                    # Normal occupied setpoint
                    setpoint = T_setpoint_base - 7.0  # Typically 7K below zone setpoint
                elif occupancy_state == "unoccupied":
                    # Setback mode: less aggressive conditioning
                    setpoint = T_setpoint_base - 3.0  # Warmer supply air for setback
                else:  # transition
                    # Startup mode: moderate conditioning
                    setpoint = T_setpoint_base - 5.0

                # Seasonal adjustment: slightly warmer supply in winter, cooler in summer
                seasonal_adjustment = 1.0 * torch.sin(torch.tensor(2 * torch.pi * (day_of_year - 80) / 365))
                setpoint = setpoint + seasonal_adjustment

                return torch.full((batch_size, 1), setpoint, device=self.device, dtype=self.dtype)

            def supply_airflow_setpoint_fn(t, batch_size=1):
                """Context-aware airflow setpoint based on occupancy and context."""
                expected_airflow = supply_airflow_fraction * self.airflow_max

                current_hour = (t / 3600.0) % 24

                # Modulate based on occupancy schedule and time
                if occupancy_state == "occupied":
                    if 7 <= current_hour <= 19:  # Business hours
                        airflow_multiplier = 1.0
                    else:
                        airflow_multiplier = 0.7  # Reduced after hours
                elif occupancy_state == "unoccupied":
                    airflow_multiplier = 0.3  # Minimum ventilation only
                else:  # transition
                    airflow_multiplier = 0.6  # Building startup

                airflow_setpoint = expected_airflow * airflow_multiplier
                setpoint = torch.full((batch_size, 1), airflow_setpoint, device=self.device, dtype=self.dtype)
                return torch.full((batch_size, 1), airflow_setpoint, device=self.device, dtype=self.dtype)

            self._input_functions = {
                "T_outdoor": lambda t, batch_size=1: seasonal_temperature(t, base_temp=T_outdoor_base,
                                                                          day_of_year=day_of_year_fn(t),
                                                                          shape=(batch_size,1)),
                "T_return_zones": T_zones_fn,
                "return_airflow_zones": zone_return_airflows_fn,
                "T_supply_setpoint": T_supply_setpoint_fn,
                "supply_airflow_setpoint": supply_airflow_setpoint_fn,
            }

        return self._input_functions

    # @input_functions.setter
    # def input_functions(self, value):
    #     """Allow custom input functions to be set."""
    #     if not hasattr(self, '_input_functions'):
    #         self._input_functions = {}
    #     self._input_functions.update(value)