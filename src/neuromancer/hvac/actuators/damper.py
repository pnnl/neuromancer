"""
HVAC Air Damper Model with Actuator Dynamics and Zone Vectorization Support

This module implements a physics-based model of Variable Air Volume (VAV) dampers
commonly used in commercial HVAC systems. Combines realistic actuator positioning
dynamics with airflow physics based on damper characteristics and duct pressure.

ZONE VECTORIZATION SUPPORT:
- Supports 1 to n_zones with zone-specific or shared parameters
- Input tensors expected as [batch_size, n_zones]
- Output tensors produced as [batch_size, n_zones]
- Zone-specific parameters automatically expanded from scalars if needed
"""

import torch
from .actuator import Actuator
from typing import Literal, Union, List


class Damper(Actuator):
    """
    HVAC Air Damper Model with Actuator Dynamics and Zone Vectorization

    Physical System:
        Models a variable air volume (VAV) damper - a motorized blade assembly that
        controls airflow in HVAC ductwork. The damper consists of one or more pivoting
        blades connected to an electric or pneumatic actuator. As the actuator rotates
        the blades from closed (0°) to open (90°), airflow increases according to the
        damper's flow characteristic curve.

    Zone Vectorization:
        - Handles multiple zones simultaneously with zone-specific parameters
        - All tensor inputs/outputs have shape [batch_size, n_zones]
        - Parameters can be scalars (shared) or vectors (zone-specific)
        - Automatic broadcasting ensures compatibility between parameters and inputs

    Physics Model:
        - Airflow follows standard HVAC damper characteristics based on blade geometry
        - Pressure correction accounts for variable duct static pressure using
          theoretical square-root pressure-flow relationship (Q ∝ √ΔP)
        - Position dynamics include realistic actuator lag with configurable time constant
        - Inherits actuator dynamics (first-order lag) from base Actuator class

    Flow Characteristics:
        - linear: Q = position × Q_max (simple proportional, rare in practice)
        - sqrt: Q = √position × Q_max (most common, accounts for pressure drop)
        - equal_percent: Q = position³ × Q_max (fine control at low flows)

    Typical Applications:
        - VAV terminal unit dampers (zone airflow control)
        - Supply air dampers (air handler units)
        - Return/exhaust air dampers
        - Outside air economizer dampers

    Typical Parameter Ranges:
        max_airflow: 0.05-2.0 kg/s per zone
            Small VAV box: 0.05-0.3 kg/s (50-300 CFM)
            Large VAV box: 0.3-1.0 kg/s (300-1000 CFM)
            Supply air damper: 1.0-2.0+ kg/s (1000+ CFM)

        nominal_pressure: 200-1500 Pa (typically shared across zones)
            Low pressure systems: 200-500 Pa (residential, small commercial)
            Medium pressure: 500-1000 Pa (typical commercial VAV)
            High pressure: 1000-1500 Pa (high-rise, industrial)

        tau: 3-15 s per zone (actuator response time constant)
            Fast electric actuators: 3-5 s
            Standard electric: 5-10 s
            Pneumatic actuators: 8-15 s

    Units:
        airflow: kg/s (mass flow rate of air)
        pressure: Pa (gauge pressure relative to atmospheric)
        position: 0-1 (normalized damper position, 0=closed, 1=fully open)
        time: s (seconds)

    Tensor Shapes:
        Input tensors: [batch_size, n_zones]
        Output tensors: [batch_size, n_zones]
        Parameters: [n_zones] for zone-specific, scalar for shared
    """

    def __init__(
            self,
            max_airflow: Union[float, List[float], torch.Tensor] = 1.0,
            # [kg/s] Maximum airflow capacity when fully open
            # Can be:
            #   - Scalar: Same max airflow for all zones
            #   - List[n_zones]: Zone-specific max airflows
            #   - Tensor[n_zones]: Zone-specific max airflows
            # Design airflow at nominal pressure with damper 100% open
            # Typical: 0.05 kg/s (small zone) to 2.0 kg/s (supply air)

            flow_characteristic: Literal["linear", "sqrt", "equal_percent"] = "sqrt",
            # Damper blade/linkage geometry determines flow curve
            # "linear": Q ∝ position (rare, simple proportional)
            # "sqrt": Q ∝ √position (most common for VAV dampers)
            # "equal_percent": Q ∝ position³ (fine control at low flows)

            nominal_pressure: Union[float, torch.Tensor] = 500.0,
            # [Pa] Design duct static pressure for max_airflow calculation
            # Reference pressure used for flow calculations and sizing
            # Typically shared across zones, but can be zone-specific
            # Typical: 200-1500 Pa depending on system design

            tau: Union[float, List[float], torch.Tensor] = 5.0,
            # [s] Actuator time constant per zone (63% response time)
            # Can be:
            #   - Scalar: Same response time for all zones
            #   - List[n_zones]: Zone-specific time constants
            #   - Tensor[n_zones]: Zone-specific time constants
            # Time for actuator to reach 63% of final position after step input
            # Electric actuators: 3-10 s, Pneumatic: 8-15 s

            actuator_model: str = "smooth_approximation",
            # Actuator dynamics model type
            # Options: "instantaneous", "analytic", "smooth_approximation"
            # Inherited from Actuator base class
    ):
        """
        Initialize damper with specified physical and control parameters.

        Zone Vectorization:
            Parameters can be provided as scalars (shared across zones) or as
            lists/tensors (zone-specific values). The BuildingComponent base class
            handles automatic expansion of scalar parameters to zone vectors.

        Args:
            max_airflow: Maximum airflow when fully open at nominal pressure [kg/s]
                        Scalar (shared) or vector (zone-specific)
            flow_characteristic: Damper flow curve type based on blade geometry
            nominal_pressure: Reference duct pressure for flow calculations [Pa]
                             Scalar (shared) or vector (zone-specific)
            tau: Actuator response time constant [s]
                 Scalar (shared) or vector (zone-specific)
            actuator_model: Dynamics model for actuator positioning
        """
        super().__init__(tau=tau, model=actuator_model, name="damper")

        self.max_airflow = max_airflow
        self.flow_characteristic = flow_characteristic
        self.nominal_pressure = nominal_pressure

    def airflow_to_position(self, target_airflow: torch.Tensor,
                            duct_pressure: torch.Tensor = None) -> torch.Tensor:
        """
        Convert target airflow to damper position setpoint (inverse flow calculation).

        Solves the inverse of the damper flow characteristic to determine what
        position is needed to achieve the desired airflow rate.

        Args:
            target_airflow: Desired airflow rate [kg/s], shape [batch_size, n_zones]
            duct_pressure: Current duct static pressure [Pa], shape [batch_size, n_zones] or [batch_size, 1]
                          If None, uses nominal_pressure for calculation

        Returns:
            torch.Tensor: Required damper position [0-1], shape [batch_size, n_zones]
        """
        # Normalize airflow to fraction of maximum capacity
        # Broadcasting: [batch_size, n_zones] / [n_zones] -> [batch_size, n_zones]
        flow_fraction = target_airflow / self.max_airflow

        # Apply inverse flow characteristic to get required position
        if self.flow_characteristic == "linear":
            # Linear: airflow = position × max_airflow
            # Inverse: position = airflow / max_airflow
            position = flow_fraction

        elif self.flow_characteristic == "sqrt":
            # Square root: airflow = √position × max_airflow
            # Inverse: position = (airflow / max_airflow)²
            position = flow_fraction ** 2

        elif self.flow_characteristic == "equal_percent":
            # Equal percentage: airflow = position³ × max_airflow
            # Inverse: position = ∛(airflow / max_airflow)
            position = torch.pow(torch.clamp(flow_fraction, min=0.0), 1.0 / 3.0)

        else:
            raise ValueError(f"Unknown flow characteristic: {self.flow_characteristic}")

        return torch.clamp(position, 0.0, 1.0)

    def position_to_airflow(self, position: torch.Tensor,
                            duct_pressure: torch.Tensor = None) -> torch.Tensor:
        """
        Convert damper position to actual airflow (forward flow calculation).

        Calculates airflow based on damper position using the specified flow
        characteristic, with optional pressure correction for varying duct conditions.

        Args:
            position: Current damper position [0-1], shape [batch_size, n_zones]
            duct_pressure: Current duct static pressure [Pa], shape [batch_size, n_zones] or [batch_size, 1]
                          If None, uses nominal pressure (no correction applied)

        Returns:
            torch.Tensor: Actual airflow rate [kg/s], shape [batch_size, n_zones]
        """
        position = torch.clamp(position, 0.0, 1.0)

        # Apply flow characteristic curve based on damper blade geometry
        if self.flow_characteristic == "linear":
            # Linear: direct proportional relationship (rare in practice)
            flow_fraction = position

        elif self.flow_characteristic == "sqrt":
            # Square root: most common for VAV dampers
            # Accounts for pressure drop characteristics across damper blades
            flow_fraction = torch.sqrt(position)

        elif self.flow_characteristic == "equal_percent":
            # Equal percentage: cubic relationship for fine control at low flows
            # Each equal increment in position gives equal percentage change in flow
            flow_fraction = position ** 3

        else:
            raise ValueError(f"Unknown flow characteristic: {self.flow_characteristic}")

        # Calculate base airflow at nominal conditions
        # Broadcasting: [batch_size, n_zones] * [n_zones] -> [batch_size, n_zones]
        base_airflow = flow_fraction * self.max_airflow

        # Apply pressure correction if duct pressure varies from nominal
        if duct_pressure is not None:
            # Theoretical relationship: Q ∝ √ΔP (orifice flow equation)
            # Broadcasting: [batch_size, n_zones or 1] / [n_zones or scalar] -> [batch_size, n_zones]
            pressure_factor = torch.sqrt(duct_pressure / self.nominal_pressure)
            # Clamp to reasonable range to prevent unrealistic flows
            pressure_factor = torch.clamp(pressure_factor, 0.5, 2.0)
            airflow = base_airflow * pressure_factor
        else:
            airflow = base_airflow

        return airflow

    def forward(self, t: float, target_airflow: torch.Tensor = None,
                current_position: torch.Tensor = None, dt: float = 1.0,
                duct_pressure: torch.Tensor = None) -> dict:
        """
        Complete damper simulation step: airflow control → actuator dynamics → actual flow.

        This is the main method called by terminal units and other HVAC components.
        Handles the complete control loop from airflow setpoint to actual delivered airflow.

        Control sequence:
        1. Convert target airflow to position setpoint (flow characteristic inverse)
        2. Apply actuator dynamics to update actual position (first-order lag)
        3. Calculate actual airflow from new position (flow characteristic + pressure correction)

        Args:
            t: Current simulation time [s]
            target_airflow: Desired airflow rate [kg/s], shape [batch_size, n_zones]
            current_position: Current actuator position [0-1], shape [batch_size, n_zones]
            dt: Simulation time step [s]
            duct_pressure: Current duct static pressure [Pa], shape [batch_size, n_zones] or [batch_size, 1]

        Returns:
            dict: Complete damper state and outputs, all tensors shape [batch_size, n_zones]
                position: New actuator position after dynamics [0-1]
                airflow: Actual delivered airflow [kg/s]
                position_setpoint: Position command from flow controller [0-1]
        """
        # Step 1: Convert airflow target to position setpoint
        position_setpoint = self.airflow_to_position(target_airflow, duct_pressure)

        # Step 2: Update actuator position with realistic dynamics (inherited from Actuator)
        # NOTE: Requires Actuator base class to handle [batch_size, n_zones] tensors
        new_position = super().forward(
            t=t,
            setpoint=position_setpoint,
            position=current_position,
            dt=dt
        )

        # Step 3: Calculate actual airflow from new position with pressure effects
        actual_airflow = self.position_to_airflow(new_position, duct_pressure)

        return {
            "position": new_position,
            "airflow": actual_airflow,
            "position_setpoint": position_setpoint
        }