"""
BuildingComponent Base Class

This module defines `BuildingComponent`, an abstract base class for differentiable
building simulation components in torch_buildings.

Key features:
- Standardizes handling of learnable and fixed parameters using a unified pattern
  compatible with PyTorch modules.
- Supports specification and per-instance override of physically reasonable
  variable ranges for states, inputs, parameters, and outputs.
- Facilitates code reuse and consistent interface across all building system
  components (e.g., envelope, HVAC, loads).

Usage:
    class MyComponent(BuildingComponent):
        _variable_ranges = {
            "my_state": (min, max),       # [unit] brief description
            "my_input": (min, max),       # [unit] brief description
            # ...
        }

        def __init__(self, param1=..., param2=..., learnable=None, ...):
            # Construct parameter tensors as needed
            param_init = {...}
            super().__init__(learnable=learnable, param_init=param_init)
            # Additional setup ...

        # Implement required model methods
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Tuple
import torch
import torch.nn as nn
import copy
from torchdiffeq import odeint

from ..simulation_inputs.schedules import stochastic_variation, persistent_excitation
from ..context import MILD_COOLING_CONTEXT


class BuildingComponent(nn.Module, ABC):
    """
    Base class for all building system components in torch_buildings.

    Handles:
      - Partitioned variable range dictionaries (with per-instance override)
      - Learnable/fixed parameter setup

    Partitioned variable range dictionaries (to be defined in each subclass):
        - _state_ranges:       System state variables.
        - _external_ranges:    Inputs required by forward() not wired internally (exogenous or control).
        - _output_ranges:      System output variables.
        - _param_ranges:       Model parameters (fixed or learnable).
        - _zone_param_ranges:   Model parameters (fixed or learnable).

    These dictionaries serve two primary purposes:
        1. Validation: Physical/engineering range checking for simulation health and diagnostics.
        2. Automated wiring: Used by factories or system composition tools to:
            - Identify all variables that must be externally supplied versus internally managed.

    """
    _state_ranges = {}
    _external_ranges = {}
    _param_ranges = {}
    _zone_param_ranges = {}
    _output_ranges = {}

    def __init__(self, params: dict = None, learnable: set = None, device=None, dtype=torch.float32):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        params = params or {}
        params.pop("self", 0)

        # Every component starts with a sensible default context
        self.context = params.pop("context", MILD_COOLING_CONTEXT)
        if not self.context:
            self.context = MILD_COOLING_CONTEXT
        learnable = learnable or set()

        # User-level overrides (simple assignment)
        self._user_variable_ranges = {}
        self.n_zones = params.pop("n_zones", 1)


        # Parameter setup
        for name, value in params.items():
            if isinstance(value, (int, float, torch.Tensor, list, tuple)):
                if name in self._zone_param_ranges:
                    tensor = expand_parameter(value=value, n_zones=self.n_zones, name=name, device=self.device,
                                              dtype=self.dtype)
                else:
                    tensor = torch.as_tensor(value, device=self.device)
                    if torch.is_floating_point(tensor):
                        tensor = tensor.to(dtype=self.dtype)
                if name in learnable:
                    setattr(self, name, nn.Parameter(tensor))
                else:
                    self.register_buffer(name, tensor)
            else:
                setattr(self, name, value)


    @property
    def variable_ranges(self):
        if not hasattr(self, '_user_variable_ranges') or not self._user_variable_ranges:
            # No user overrides - return default structure directly
            return {
                "state": self._state_ranges,
                "external": self._external_ranges,
                "param": self._param_ranges,
                "zone_param": self._zone_param_ranges,
                "output": self._output_ranges,
            }

        merged = copy.deepcopy(self._variable_ranges)
        for key, override in self._user_variable_ranges.items():
            if key in merged:
                merged[key].update(override)
            else:
                merged[key] = override
        return merged

    @variable_ranges.setter
    def variable_ranges(self, new_ranges):
        """
        Per-instance override for any of the partitioned dicts.
        E.g., variable_ranges = {'state': {...}, 'external': {...}}
        """
        self._user_variable_ranges = copy.deepcopy(new_ranges)

    @property
    @abstractmethod
    def input_functions(self):
        """
        Default input functions for testing, validation, and simulation in isolation of other building components.
        :return: dictionary key with input arguments names value a function which takes time and batchsize as arguments
                 and returns batchsize X dim shaped tensor for that input.
        """
        pass

    @input_functions.setter
    def input_functions(self, value):
        """Allow custom input functions to be set."""
        self._input_functions.update(value)

    def initial_state_functions(self):
        return {}

    def simulate(
            self,
            # Time and simulation control
            duration_hours=24.0,  # Simulation duration in hours
            dt_minutes=5.0,  # Time step in minutes
            t_start_hour=5.0,  # Start time in hours (5 = 5 AM)
            batch_size=1,  # Number of parallel simulations

            # State and input control
            initial_state=None,  # Override initial states
            input_functions=None,  # Custom input functions

            # Stochastic options
            pe=False,  # Persistent excitation
            input_noise_amplitude=0.0,  # Input noise level
    ):
        """
        Unified simulation with intuitive time parameters.

        Args:
            duration_hours (float): Total simulation duration in hours (default: 24.0).
            dt_minutes (float): Time step in minutes (default: 5.0).
            t_start_hour (float): Start time in hours (default: 5.0 for 5 AM).
            batch_size (int): Number of parallel simulations (default: 1).

            initial_state (dict, optional): Override initial states.
            input_functions (dict, optional): Custom input functions.

            pe (bool): Enable persistent excitation for inputs (default: False).
            input_noise_amplitude (float): Add Gaussian noise to inputs (default: 0.0).

        Returns:
            dict: Simulation results with tensors of shape [batch_size, time_steps, dim]

        Examples:
            # Quick 1-hour simulation with 1-minute resolution
            results = component.simulate(duration_hours=1.0, dt_minutes=1.0)

            # Full day simulation with default 5-minute resolution
            results = component.simulate(duration_hours=24.0)

            # High-resolution step response (10-second steps for 30 minutes)
            results = component.simulate(duration_hours=0.5, dt_minutes=0.167)

            # Multiple scenarios in parallel
            results = component.simulate(duration_hours=12.0, batch_size=10)
        """
        # Convert intuitive parameters to internal format
        self.dt = dt_minutes * 60.0  # Convert minutes to seconds
        self.batch_size = batch_size
        n_steps = int((duration_hours * 3600.0) / self.dt)

        # self.build_input_functions(
        #     input_functions=input_functions,
        #     pe=pe,
        #     input_noise_amplitude=input_noise_amplitude,
        # )

        log = defaultdict(list)
        current_time = t_start_hour * 3600.0

        # Generate initial states and inputs
        states = {k: f(batch_size) for k, f in self.initial_state_functions().items()}
        inputs = {k: fn(current_time, batch_size) for k, fn in self.input_functions.items()}

        # Log initial states, disturbances, controls
        for k, v in {**inputs, **states}.items():
            log[k].append(v)

        for step in range(n_steps):
            # Forward pass
            inputs = {**inputs, **states}
            outputs = self.forward(t=current_time, dt=self.dt, **inputs)

            # Update for next step
            current_time += self.dt
            states = {k: v for k, v in outputs.items() if k in self._state_ranges}
            inputs = {k: fn(current_time, batch_size) for k, fn in self.input_functions.items()}

            # Log
            for k, v in {**inputs, **outputs}.items():
                log[k].append(v)

        # Convert to [batch_size, time_steps, dim] format for consistency with BuildingSystem
        results = {}
        for k, v_list in log.items():
            print(k)
            stacked = torch.stack(v_list, dim=0)  # [time_steps, batch_size, dim]
            results[k] = stacked.transpose(0, 1)  # [batch_size, time_steps, dim]
        return results

    def build_input_functions(
        self,
        input_functions=None,
        pe=False, # persistent excitation
        input_noise_amplitude=0.0,
    ):
        """
        Constructs a dictionary of input functions for simulation, handling:
            - Use of component's default input functions if input_functions is None or partial
            - Wrapping input functions with stochastic noise, or persistent excitation as requested

        Args:
            input_functions: Optional dict mapping variable names to callables (t, batch_size) -> tensor.
                If any required input is missing, defaults are filled in.
            pe: If True, wraps each input function in a persistent excitation generator.
            input_noise_amplitude: If > 0, adds i.i.d. Gaussian noise to each input at each time step.

        Returns:
            input_functions: Complete dict of callables (t, batch_size) -> tensor for each required exogenous input.

        Raises:
            NotImplementedError: If the component does not implement input_functions().
        """
        input_functions = input_functions or {}

        for key, fn in self.input_functions.items():
            # User-supplied functions override defaults
            if key in input_functions:
                continue
            # Wrap with stochastic/persistent excitation as needed
            if pe:
                input_functions[key] = persistent_excitation(base_function=fn, amp=1.0)
            elif input_noise_amplitude > 0.0:
                input_functions[key] = stochastic_variation(base_function=fn, noise_amplitude=input_noise_amplitude)
            else:
                input_functions[key] = fn
        self.input_functions = input_functions


def expand_parameter(
        value: Union[float, int, List, Tuple, torch.Tensor],
        n_zones: int,
        name: str = "parameter",
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Expand a parameter value to a zone vector.

    Handles expansion from scalar values to zone-specific vectors, validates
    list/tuple lengths, and ensures proper tensor format.

    Args:
        value: Parameter value to expand. Can be:
            - Scalar (int/float): Expanded to all zones
            - List/Tuple: Must have length n_zones, converted to tensor
            - Tensor: Validated and returned (with proper device/dtype)
        n_zones (int): Number of zones to expand to
        name (str): Parameter name for error messages
        device (torch.device, optional): Target device for tensor
        dtype (torch.dtype): Target dtype for tensor

    Returns:
        torch.Tensor: Parameter tensor of shape (n_zones,)

    Raises:
        ValueError: If list/tuple length doesn't match n_zones
        TypeError: If value type is not supported

    Examples:
        >>> expand_parameter(2.0, 3)  # Scalar to vector
        tensor([2., 2., 2.])

        >>> expand_parameter([1.0, 2.0, 3.0], 3)  # List to tensor
        tensor([1., 2., 3.])

        >>> expand_parameter([1.0, 2.0], 3)  # Wrong length
        ValueError: Parameter parameter list length 2 != n_zones 3
    """
    device = device or torch.device("cpu")

    if isinstance(value, (int, float)):
        # Scalar: expand to all zones
        return torch.full((n_zones,), float(value), device=device, dtype=dtype)

    elif isinstance(value, (list, tuple)):
        # List/tuple: validate length and convert
        if len(value) != n_zones:
            raise ValueError(
                f"Parameter {name} list length {len(value)} != n_zones {n_zones}"
            )
        return torch.tensor(value, device=device, dtype=dtype)

    elif isinstance(value, torch.Tensor):
        # Tensor: validate shape and ensure correct device/dtype
        if value.numel() == 1:
            # Single-element tensor: expand like scalar
            return torch.full((n_zones,), float(value.item()), device=device, dtype=dtype)
        elif value.numel() == n_zones:
            # Multi-element tensor: ensure correct shape and move to device/dtype
            return value.flatten().to(device=device, dtype=dtype)
        else:
            raise ValueError(
                f"Parameter {name} tensor has {value.numel()} elements, "
                f"expected 1 or {n_zones}"
            )
    else:
        raise TypeError(
            f"Parameter {name} must be scalar, list, tuple, or tensor, "
            f"got {type(value)}"
        )