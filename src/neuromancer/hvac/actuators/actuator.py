"""
actuator.py

General Actuator class for HVAC components with Zone Vectorization Support.
Supports different levels of fidelity for different use cases while maintaining
a consistent interface.

ZONE VECTORIZATION SUPPORT:
- Supports 1 to n_zones with zone-specific or shared parameters
- Input tensors expected as [batch_size, n_zones]
- Output tensors produced as [batch_size, n_zones]
- Zone-specific parameters automatically expanded from scalars if needed
"""

import torch
from typing import Literal, Union, List


class Actuator(torch.nn.Module):
    """
    General actuator class with pluggable dynamics models and zone vectorization.

    Supports multiple modeling approaches:
    - "instantaneous": No lag, output = setpoint immediately
    - "analytic": Analytic solution to first-order lag (exact, fast)
    - "smooth_approximation": Numerically stable approximation of the analytic solution
                              Provides nicely behaved gradients for learning tau

    Zone Vectorization:
        - Handles multiple zones simultaneously with zone-specific parameters
        - All tensor inputs/outputs have shape [batch_size, n_zones]
        - Parameters can be scalars (shared) or vectors (zone-specific)
        - Automatic broadcasting ensures compatibility between parameters and inputs

    Numerical Considerations:
        The analytic method provides exact solutions to the first-order lag equation.
        For gradient-based learning of tau parameters, use smooth_approximation which
        provides numerically stable gradients. The instantaneous method has no dynamics.

    Removed Methods:
        - "odesolve": Removed due to vectorization complexity and no accuracy advantage
                     over the exact analytic solution. Use "analytic" instead.

    Units:
        position: [0-1] Normalized actuator position
        setpoint: [0-1] Normalized actuator setpoint
        tau: [s] Time constant (per zone)
        t: [s] Time

    Tensor Shapes:
        Input tensors: [batch_size, n_zones]
        Output tensors: [batch_size, n_zones]
        Parameters: [n_zones] for zone-specific, scalar for shared
    """
# TODO: Offset in addition to time lag
    def __init__(
            self,
            tau: Union[float, List[float], torch.Tensor] = 15.0,
            # [s] Time constant per zone
            # Can be:
            #   - Scalar: Same time constant for all zones
            #   - List[n_zones]: Zone-specific time constants
            #   - Tensor[n_zones]: Zone-specific time constants
            # Typical: 5-15 s for electric actuators, 10-30 s for pneumatic
            model: Literal["instantaneous", "analytic", "smooth_approximation"] = "instantaneous",
            # Dynamics model type
            # "instantaneous": No lag (immediate response)
            # "analytic": Exact first-order lag solution
            # "smooth_approximation": Gradient-friendly approximation for learning

            name: str = "actuator",
            # Actuator name for identification and debugging
    ):
        """
        Initialize actuator with specified dynamics model.

        Zone Vectorization:
            Parameters can be provided as scalars (shared across zones) or as
            lists/tensors (zone-specific values). The BuildingComponent base class
            handles automatic expansion of scalar parameters to zone vectors.

        Args:
            tau: Time constant [s] - scalar (shared) or vector (zone-specific)
            model: Dynamics model type
            name: Actuator name for identification
        """
        super().__init__()
        self.model = model
        self.name = name
        # tau will be expanded to [n_zones] tensor by BuildingComponent base class
        self.tau = tau
        # Validate model type
        valid_models = ["instantaneous", "analytic", "smooth_approximation"]
        if model not in valid_models:
            raise ValueError(f"model must be one of {valid_models}, got {model}")

    def forward(
            self,
            t: float = 0.,  # [s] Current time
            setpoint: torch.Tensor = None,  # [0-1] Desired actuator position, shape [batch_size, n_zones]
            position: torch.Tensor = None,  # [0-1] Current position, shape [batch_size, n_zones]
            dt: float = 1.,  # [s] Time step (for analytic/smooth models)
    ) -> torch.Tensor:
        """
        Compute actuator response with zone vectorization support.

        Args:
            t (float): Current simulation time [s]
            setpoint (Tensor): Desired actuator position [0-1], shape [batch_size, n_zones]
            position (Tensor): Current actuator position [0-1], shape [batch_size, n_zones]
                              Required for non-instantaneous models.
            dt (float): Time step [s]. Used for analytic/smooth models.

        Returns:
            Tensor: New actuator position [0-1], shape [batch_size, n_zones]
        """
        if self.model == "instantaneous":
            return self._forward_instantaneous(setpoint)

        elif self.model == "analytic":
            return self._forward_analytic(position, setpoint, dt)

        elif self.model == "smooth_approximation":
            return self._forward_smooth_approximation(position, setpoint, dt)

        else:
            raise ValueError(f"Unknown model type: {self.model}")

    def _forward_instantaneous(self, setpoint: torch.Tensor) -> torch.Tensor:
        """
        Instantaneous response - no lag.

        Args:
            setpoint: Desired position [0-1], shape [batch_size, n_zones]

        Returns:
            Tensor: New position [0-1], shape [batch_size, n_zones]
        """
        return setpoint

    def _forward_analytic(self, position: torch.Tensor, setpoint: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Analytic solution to first-order lag (exact solution).

        Uses the exact mathematical solution: x(t+dt) = setpoint + (x₀ - setpoint) * exp(-dt/tau)

        Args:
            position: Current position [0-1], shape [batch_size, n_zones]
            setpoint: Desired position [0-1], shape [batch_size, n_zones]
            dt: Time step [s]

        Returns:
            Tensor: New position [0-1], shape [batch_size, n_zones]
        """
        if position is None:
            raise ValueError("position required for analytic model")

        # Analytic solution: x(t+dt) = setpoint + (x_current - setpoint) * exp(-dt/tau)
        # Broadcasting: scalar / [n_zones] -> [n_zones]
        decay_factor = torch.exp(-dt / self.tau)
        # Broadcasting: [batch_size, n_zones] + ([batch_size, n_zones] - [batch_size, n_zones]) * [n_zones]
        position_new = setpoint + (position - setpoint) * decay_factor

        return position_new

    def _forward_smooth_approximation(self, position: torch.Tensor, setpoint: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Smooth approximation for stable gradients with adaptive approximation selection.

        This method implements multiple approximations to the exact first-order lag response
        (1 - exp(-dt/tau)) and selects the most appropriate one based on the rate (dt/tau)
        and whether the model is in training or inference mode.

        Approximation Methods:
        - Taylor (1st order): rate ≈ (1 - exp(-rate)) for small rates
        - Pade [1,1]: 2*rate/(2+rate) - rational approximation with good stability
        - Tanh: tanh(rate) - bounded output, prevents gradient explosion
        - Exact: 1 - exp(-rate) - mathematically exact solution

        Selection Strategy:
        Training mode (prioritizes gradient stability):
        - Small rates (< 0.1): Taylor approximation for smooth gradients
        - Medium rates (0.1-1.0): Pade approximation for stable gradients
        - Large rates (> 1.0): Tanh approximation for bounded gradients

        Inference mode (prioritizes accuracy):
        - Most rates (< 2.0): Exact solution for maximum accuracy
        - Very large rates (> 2.0): Tanh to avoid numerical overflow

        This adaptive approach enables stable gradient-based learning of tau while
        maintaining accuracy during inference.

        Args:
            position: Current position [0-1], shape [batch_size, n_zones]
            setpoint: Desired position [0-1], shape [batch_size, n_zones]
            dt: Time step [s]

        Returns:
            Tensor: New position [0-1], shape [batch_size, n_zones]
        """
        if position is None:
            raise ValueError("position required for smooth_approximation model")

        # Broadcasting: scalar / [n_zones] -> [n_zones]
        rate = dt / self.tau

        # Select and calculate the needed step factor (element-wise operations)
        if self.training:
            # During training: prioritize gradient stability
            # All conditions operate element-wise on rate tensor
            small_rate = rate < 0.1
            medium_rate = (rate >= 0.1) & (rate < 1.0)
            large_rate = rate >= 1.0

            step_factor = torch.zeros_like(rate)
            step_factor[small_rate] = rate[small_rate]  # Taylor
            step_factor[medium_rate] = (2 * rate[medium_rate]) / (2 + rate[medium_rate])  # Pade
            step_factor[large_rate] = torch.tanh(rate[large_rate])  # Tanh
        else:
            # During inference: prioritize accuracy
            normal_rate = rate < 2.0
            large_rate = rate >= 2.0

            step_factor = torch.zeros_like(rate)
            step_factor[normal_rate] = 1 - torch.exp(-rate[normal_rate])  # Exact
            step_factor[large_rate] = torch.tanh(rate[large_rate])  # Tanh for numerical stability

        # Broadcasting: [batch_size, n_zones] + ([batch_size, n_zones] - [batch_size, n_zones]) * [n_zones]
        position_new = position + (setpoint - position) * step_factor
        return position_new