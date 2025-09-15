"""
Electric Reheat Coil Model with Actuator Dynamics and Zone Vectorization Support

This module implements a physics-based model of electric reheat coils commonly
used in Variable Air Volume (VAV) terminal units. Combines realistic actuator
positioning dynamics with electrical heating physics.

ZONE VECTORIZATION SUPPORT:
- Supports 1 to n_zones with zone-specific or shared parameters
- Input tensors expected as [batch_size, n_zones]
- Output tensors produced as [batch_size, n_zones]
- Zone-specific parameters automatically expanded from scalars if needed
"""

import torch
from .actuator import Actuator
from typing import Literal, Union, List


class ElectricReheatCoil(Actuator):
    """
    Electric Reheat Coil Model with Actuator Dynamics and Zone Vectorization

    Physical System:
        Models an electric reheat coil assembly commonly found in VAV terminal units.
        The system consists of electric heating elements (resistance coils or finned
        tube heaters) controlled by an actuator that modulates electrical power input.
        The actuator typically controls electrical contactors, SCR thyristors, or
        modulating transformers to vary the electrical power delivered to the heating elements.

    Zone Vectorization:
        - Handles multiple zones simultaneously with zone-specific parameters
        - All tensor inputs/outputs have shape [batch_size, n_zones]
        - Parameters can be scalars (shared) or vectors (zone-specific)
        - Automatic broadcasting ensures compatibility between parameters and inputs

    Physics Model:
        - Position directly controls electrical power input to heating elements
        - Thermal output follows standard electric coil characteristics based on control strategy
        - Electrical efficiency accounts for resistive losses and control losses
        - Position dynamics include realistic actuator lag with configurable time constant
        - Inherits actuator positioning dynamics from base Actuator class

    Heating Characteristics:
        - linear: P = position × P_max (simple proportional control, common)
        - equal_percent: P = position³ × P_max (fine control at low loads)
        - quick_opening: P = √position × P_max (rapid heating at low positions)

    Typical Applications:
        - VAV terminal unit reheat (perimeter zones)
        - Zone heating in all-air systems
        - Frost protection heating
        - Makeup air heating
        - Space heating in heat pump systems

    Typical Parameter Ranges:
        max_thermal_output: 1000-6000 W per zone
            Small VAV box: 1000-2000 W (3,400-6,800 BTU/hr)
            Medium VAV box: 2000-4000 W (6,800-13,600 BTU/hr)
            Large VAV box: 4000-6000 W (13,600-20,500 BTU/hr)

        electrical_efficiency: 0.95-1.0 per zone
            Direct resistance heating: 0.98-1.0 (very efficient)
            Modulated control systems: 0.95-0.98 (small control losses)
            Finned tube heaters: 0.96-0.99 (convective efficiency)

        tau: 5-20 s per zone (actuator response time constant)
            Electric contactors: 5-10 s (fast switching)
            SCR/thyristor control: 8-15 s (electronic modulation)
            Modulating transformers: 10-20 s (mechanical/magnetic)

    Units:
        thermal_output: W (thermal power delivered to air)
        electrical_power: W (electrical power consumed from grid)
        position: 0-1 (normalized control position, 0=off, 1=full power)
        time: s (seconds)
        efficiency: 0-1 (fraction, thermal output / electrical input)

    Tensor Shapes:
        Input tensors: [batch_size, n_zones]
        Output tensors: [batch_size, n_zones]
        Parameters: [n_zones] for zone-specific, scalar for shared
    """

    def __init__(
            self,
            max_thermal_output: Union[float, List[float], torch.Tensor] = 3000.0,
            # [W] Maximum thermal output at full power
            # Can be:
            #   - Scalar: Same max thermal output for all zones
            #   - List[n_zones]: Zone-specific max thermal outputs
            #   - Tensor[n_zones]: Zone-specific max thermal outputs
            # Design heating capacity of electric coil elements
            # Typical: 1000 W (small zone) to 6000 W (large zone)

            heating_characteristic: Literal["linear", "equal_percent", "quick_opening"] = "linear",
            # Control characteristic for electrical power modulation
            # "linear": P ∝ position (most common for electric coils)
            # "equal_percent": P ∝ position³ (fine control at low loads)
            # "quick_opening": P ∝ √position (rapid heating response)

            electrical_efficiency: Union[float, List[float], torch.Tensor] = 0.98,
            # [0-1] Electrical to thermal conversion efficiency
            # Can be:
            #   - Scalar: Same efficiency for all zones
            #   - List[n_zones]: Zone-specific efficiencies
            #   - Tensor[n_zones]: Zone-specific efficiencies
            # Accounts for resistive losses, control losses, convective efficiency
            # Electric resistance: 0.98-1.0, Control systems: 0.95-0.98

            tau: Union[float, List[float], torch.Tensor] = 10.0,
            # [s] Actuator time constant per zone (63% response time)
            # Can be:
            #   - Scalar: Same response time for all zones
            #   - List[n_zones]: Zone-specific time constants
            #   - Tensor[n_zones]: Zone-specific time constants
            # Time for actuator to reach 63% of final position after step input
            # Electric contactors: 5-10 s, SCR control: 8-15 s, Transformers: 10-20 s

            actuator_model: str = "smooth_approximation",
            # Actuator dynamics model type
            # Options: "instantaneous", "odesolve", "analytic", "smooth_approximation"
            # Inherited from Actuator base class
    ):
        """
        Initialize electric reheat coil with specified physical and control parameters.

        Zone Vectorization:
            Parameters can be provided as scalars (shared across zones) or as
            lists/tensors (zone-specific values). The BuildingComponent base class
            handles automatic expansion of scalar parameters to zone vectors.

        Args:
            max_thermal_output: Maximum thermal output at full electrical power [W]
                               Scalar (shared) or vector (zone-specific)
            heating_characteristic: Control curve type for power modulation
            electrical_efficiency: Electrical to thermal conversion efficiency [0-1]
                                   Scalar (shared) or vector (zone-specific)
            tau: Actuator response time constant [s]
                 Scalar (shared) or vector (zone-specific)
            actuator_model: Dynamics model for actuator positioning
        """
        super().__init__(tau=tau, model=actuator_model, name="electric_reheat_coil")

        self.max_thermal_output = max_thermal_output
        self.heating_characteristic = heating_characteristic
        self.electrical_efficiency = electrical_efficiency

    def target_heating_to_position(self, target_heating: torch.Tensor) -> torch.Tensor:
        """
        Convert target thermal output to actuator position setpoint (inverse heating calculation).

        Solves the inverse of the electric coil heating characteristic to determine what
        position is needed to achieve the desired thermal output.

        Args:
            target_heating: Desired thermal output [W], shape [batch_size, n_zones]

        Returns:
            torch.Tensor: Required actuator position [0-1], shape [batch_size, n_zones]
        """
        # Normalize heating to fraction of maximum capacity
        # Broadcasting: [batch_size, n_zones] / [n_zones] -> [batch_size, n_zones]
        heating_fraction = target_heating / self.max_thermal_output

        # Apply inverse heating characteristic to get required position
        if self.heating_characteristic == "linear":
            # Linear: thermal_output = position × max_thermal_output
            # Inverse: position = thermal_output / max_thermal_output
            position = heating_fraction

        elif self.heating_characteristic == "equal_percent":
            # Equal percentage: thermal_output = position³ × max_thermal_output
            # Inverse: position = ∛(thermal_output / max_thermal_output)
            position = torch.pow(torch.clamp(heating_fraction, min=0.0), 1.0/3.0)

        elif self.heating_characteristic == "quick_opening":
            # Quick opening: thermal_output = √position × max_thermal_output
            # Inverse: position = (thermal_output / max_thermal_output)²
            position = heating_fraction ** 2

        else:
            raise ValueError(f"Unknown heating characteristic: {self.heating_characteristic}")

        return torch.clamp(position, 0.0, 1.0)

    def position_to_thermal_output(self, position: torch.Tensor) -> torch.Tensor:
        """
        Convert actuator position to actual thermal output (forward heating calculation).

        Calculates thermal power delivered to air based on actuator position using the
        specified heating characteristic curve.

        Args:
            position: Current actuator position [0-1], shape [batch_size, n_zones]

        Returns:
            torch.Tensor: Actual thermal output [W], shape [batch_size, n_zones]
        """
        position = torch.clamp(position, 0.0, 1.0)

        # Apply heating characteristic curve based on control strategy
        if self.heating_characteristic == "linear":
            # Linear: direct proportional relationship (most common for electric coils)
            heating_fraction = position

        elif self.heating_characteristic == "equal_percent":
            # Equal percentage: cubic relationship for fine control at low loads
            # Each equal increment in position gives equal percentage change in heating
            heating_fraction = position ** 3

        elif self.heating_characteristic == "quick_opening":
            # Quick opening: square root relationship for rapid heating response
            # Large heating increase at low positions, useful for fast warm-up
            heating_fraction = torch.sqrt(position)

        else:
            raise ValueError(f"Unknown heating characteristic: {self.heating_characteristic}")

        # Calculate thermal output delivered to air
        # Broadcasting: [batch_size, n_zones] * [n_zones] -> [batch_size, n_zones]
        thermal_output = heating_fraction * self.max_thermal_output
        return thermal_output

    def position_to_electrical_power(self, position: torch.Tensor) -> torch.Tensor:
        """
        Convert actuator position to electrical power consumption.

        Calculates electrical power drawn from the grid based on thermal output
        and electrical efficiency of the heating system.

        Args:
            position: Current actuator position [0-1], shape [batch_size, n_zones]

        Returns:
            torch.Tensor: Electrical power consumption [W], shape [batch_size, n_zones]
        """
        thermal_output = self.position_to_thermal_output(position)
        # Broadcasting: [batch_size, n_zones] / [n_zones] -> [batch_size, n_zones]
        electrical_power = thermal_output / self.electrical_efficiency
        return electrical_power

    def forward(self, t: float, target_heating: torch.Tensor = None,
                current_position: torch.Tensor = None, dt: float = 1.0) -> dict:
        """
        Complete electric reheat coil simulation step: heating control → actuator dynamics → actual heating.

        This is the main method called by VAV terminal units and other HVAC components.
        Handles the complete control loop from heating setpoint to actual delivered thermal output.

        Control sequence:
        1. Convert target heating to position setpoint (heating characteristic inverse)
        2. Apply actuator dynamics to update actual position (first-order lag)
        3. Calculate actual thermal output and electrical power from new position

        Args:
            t: Current simulation time [s]
            target_heating: Desired thermal output [W], shape [batch_size, n_zones]
            current_position: Current actuator position [0-1], shape [batch_size, n_zones]
            dt: Simulation time step [s]

        Returns:
            dict: Complete electric reheat coil state and outputs, all tensors shape [batch_size, n_zones]
                position: New actuator position after dynamics [0-1]
                thermal_output: Actual thermal power delivered to air [W]
                electrical_power: Actual electrical power consumed from grid [W]
                position_setpoint: Position command from heating controller [0-1]
        """
        # Step 1: Convert heating target to position setpoint
        position_setpoint = self.target_heating_to_position(target_heating)

        # Step 2: Update actuator position with realistic dynamics (inherited from Actuator)
        # NOTE: Requires Actuator base class to handle [batch_size, n_zones] tensors
        new_position = super().forward(
            t=t,
            setpoint=position_setpoint,
            position=current_position,
            dt=dt
        )

        # Step 3: Calculate actual thermal output and electrical power consumption
        actual_thermal_output = self.position_to_thermal_output(new_position)
        actual_electrical_power = self.position_to_electrical_power(new_position)

        return {
            "position": new_position,
            "thermal_output": actual_thermal_output,
            "electrical_power": actual_electrical_power,
            "position_setpoint": position_setpoint
        }