"""
solar_gains.py

Models solar heat gains to building zones from readily available inputs:
outdoor temperature, weather conditions, and basic building specifications.

This model focuses specifically on solar irradiance and heat gains through windows,
avoiding overlap with envelope models that handle conduction and infiltration
through thermal resistance parameters.

Key Features:
- Estimates solar irradiance from outdoor temperature and weather patterns
- Accounts for building orientation and window specifications
- Supports zone vectorization for multi-zone buildings
- Uses empirical correlations based on typical weather relationships
- Compatible with envelope models (no double-counting of envelope heat transfer)
- Inherits from BuildingComponent for simulation and parameter management

Units:
- Temperature: Kelvin [K]
- Power: Watts [W]
- Area: Square meters [m²]
- Irradiance: Watts per square meter [W/m²]
"""
import numpy as np
import torch
import math
from typing import Union, List
from .base import BuildingComponent
from ..simulation_inputs.schedules import seasonal_temperature, binary_schedule


class SolarGains(BuildingComponent):
    """
    Models solar heat gains to building zones using outdoor temperature,
    weather conditions, and window specifications.

    This model estimates solar irradiance from readily available weather data
    and calculates resulting heat gains through windows. It's designed to work
    alongside envelope models that handle conduction and infiltration separately.

    Solar Heat Gain Process:
    1. Estimate solar irradiance from outdoor temperature and weather patterns
    2. Account for window orientation relative to sun position
    3. Apply solar heat gain coefficient (SHGC) for window properties
    4. Calculate final solar heat gains delivered to zones

    The model uses empirical relationships to estimate solar irradiance from
    outdoor temperature patterns and weather conditions, making it suitable
    for applications where detailed solar measurement data is unavailable.
    """

    # Variable ranges for BuildingComponent base class
    _external_ranges = {
        "T_outdoor": (253.15, 318.15),  # [K] Outdoor temperature (-20°C to 45°C)
        "weather_factor": (0.0, 1.0),  # [-] Weather clarity (0=overcast, 1=clear)
        "day_of_year": (1.0, 365.0),  # [-] Day of year
    }

    _zone_param_ranges = {
        # Zone-specific parameters (expanded to [n_zones] vectors)
        "window_area": (1.0, 100.0),  # [m²] Window area per zone
        "window_orientation": (-180.0, 180.0),  # [deg] Window orientation
        "window_shgc": (0.1, 0.9),  # [-] Solar heat gain coefficient
    }

    _param_ranges = {
        # Shared parameters (scalars)
        "latitude_deg": (-90.0, 90.0),  # [deg] Building latitude
        "max_solar_irradiance": (400.0, 1200.0),  # [W/m²] Peak solar irradiance
    }

    _output_ranges = {
        "Q_solar": (0.0, 10000.0),  # [W] Solar gains per zone
    }

    def __init__(
            self,
            # Zone specifications
            n_zones: int = 1,

            # Window parameters (can be scalar or per-zone)
            window_area: Union[float, List[float]] = 10.0,  # [m²] Window area per zone
            window_orientation: Union[float, List[float]] = 0.0,  # [deg] 0=south, 90=west, etc.
            window_shgc: Union[float, List[float]] = 0.6,  # [-] Solar heat gain coefficient

            # Location and solar estimation parameters
            latitude_deg: float = 40.0,  # [deg] Building latitude
            max_solar_irradiance: float = 800.0,  # [W/m²] Peak solar irradiance

            # Standard BuildingComponent parameters
            learnable: dict = None,
            device=None,
            dtype=torch.float32
    ):
        """
        Initialize solar gains model using BuildingComponent infrastructure.

        Args:
            n_zones: Number of building zones
            window_area: Window area per zone [m²] (scalar or list[n_zones])
            window_orientation: Window orientation [deg] (scalar or list[n_zones])
            window_shgc: Solar heat gain coefficient [0-1] (scalar or list[n_zones])
            latitude_deg: Building latitude [deg]
            max_solar_irradiance: Peak solar irradiance [W/m²]
            learnable: Parameters to make learnable for optimization
            device: Device for tensor computations
            dtype: Tensor data type
        """
        # BuildingComponent handles parameter expansion and device/dtype setup
        super().__init__(params=locals(), learnable=learnable, device=device, dtype=dtype)

        # Convert latitude to radians for calculations
        self.latitude_rad = math.radians(self.latitude_deg)

        # Zone-specific parameters are automatically expanded by base class
        # self.window_area: [n_zones] tensor
        # self.window_orientation: [n_zones] tensor
        # self.window_shgc: [n_zones] tensor

    def estimate_solar_irradiance(
            self,
            t: torch.Tensor,
            T_outdoor: torch.Tensor,
            weather_factor: torch.Tensor,
            day_of_year: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate solar irradiance from outdoor temperature and weather patterns.

        Uses empirical relationships based on the observation that:
        - Peak outdoor temperatures correlate with solar intensity
        - Daily temperature patterns follow solar patterns with thermal lag
        - Weather factors modulate both temperature and solar gains

        Args:
            t: Time of day [s] since midnight, shape [batch_size, 1]
            T_outdoor: Outdoor air temperature [K], shape [batch_size, 1]
            weather_factor: Weather clarity [0-1], shape [batch_size, 1]
            day_of_year: Day of year [1-365], shape [batch_size, 1]

        Returns:
            torch.Tensor: Estimated solar irradiance [W/m²], shape [batch_size, 1]
        """
        # Convert time to hours
        hour_of_day = t / 3600.0  # [batch_size, 1]

        # Solar position approximation (simplified)
        # Declination angle (seasonal variation)
        day_angle = 2 * math.pi * (day_of_year - 81) / 365  # [batch_size, 1]
        declination = 23.45 * math.pi / 180 * torch.sin(day_angle)  # [batch_size, 1]

        # Hour angle
        hour_angle = (hour_of_day - 12) * 15 * math.pi / 180  # [batch_size, 1]

        # Solar elevation (simplified)
        lat = torch.full_like(declination, self.latitude_rad)
        solar_elevation = torch.asin(
            torch.clamp(
                torch.sin(lat) * torch.sin(declination) +
                torch.cos(lat) * torch.cos(declination) * torch.cos(hour_angle),
                -1.0, 1.0
            )
        )  # [batch_size, 1]

        # Base solar irradiance from geometry
        base_irradiance = self.max_solar_irradiance * torch.clamp(torch.sin(solar_elevation), 0.0, 1.0)

        # Temperature-based adjustment
        # Higher outdoor temperatures suggest stronger solar conditions
        # Use a reference temperature for normalization
        T_ref = 293.15  # [K] Reference temperature (20°C)
        temp_factor = 1.0 + 0.02 * (T_outdoor - T_ref)  # 2% increase per degree above reference
        temp_factor = torch.clamp(temp_factor, 0.5, 1.5)  # Reasonable bounds

        # Apply weather factor and temperature correlation
        estimated_irradiance = base_irradiance * weather_factor * temp_factor

        return torch.clamp(estimated_irradiance, 0.0, self.max_solar_irradiance)

    def calculate_solar_gains(
            self,
            t: torch.Tensor,
            T_outdoor: torch.Tensor,
            weather_factor: torch.Tensor,
            day_of_year: torch.Tensor,

    ) -> torch.Tensor:
        """
        Calculate solar heat gains through windows for all zones.

        Args:
            t: Time of day [s], shape [batch_size, 1]
            T_outdoor: Outdoor temperature [K], shape [batch_size, 1]
            weather_factor: Weather clarity [0-1], shape [batch_size, 1]
            day_of_year: Day of year [1-365], shape [batch_size, 1]

        Returns:
            torch.Tensor: Solar heat gains [W], shape [batch_size, n_zones]
        """
        # Estimate solar irradiance
        irradiance = self.estimate_solar_irradiance(t, T_outdoor, weather_factor, day_of_year)
        # [batch_size, 1]

        # Calculate solar gains for each zone
        # Simple orientation factor (peak at south, reduced for other orientations)
        orientation_rad = self.window_orientation * math.pi / 180  # [n_zones]
        orientation_factor = torch.clamp(torch.cos(orientation_rad), 0.3, 1.0)  # [n_zones]

        # Solar gains = irradiance × window_area × SHGC × orientation_factor
        Q_solar = (
                irradiance *  # [batch_size, 1]
                self.window_area.unsqueeze(0) *  # [1, n_zones]
                self.window_shgc.unsqueeze(0) *  # [1, n_zones]
                orientation_factor.unsqueeze(0)  # [1, n_zones]
        )  # [batch_size, n_zones]

        return Q_solar

    def forward(
            self, *,
            t: float,  # [s] Time of day
            T_outdoor: torch.Tensor,  # [K] Outdoor temperature, shape [batch_size, 1]
            weather_factor: torch.Tensor,  # [-] Weather clarity, shape [batch_size, 1]
            day_of_year: torch.Tensor,  # [-] Day of year, shape [batch_size, 1]
            dt: float = 1.0,  # [s] Time step (unused but required by interface)
    ) -> dict:
        """
        Calculate solar heat gains to building zones.

        Args:
            t: Time of day [s] since midnight
            T_outdoor: Outdoor air temperature [K], shape [batch_size, 1]
            weather_factor: Weather clarity factor [0-1], shape [batch_size, 1]
            day_of_year: Day of year [1-365], shape [batch_size, 1]
            dt: Time step [s] (unused but required by BuildingComponent interface)

        Returns:
            dict: Solar gains and diagnostics, all tensors shape [batch_size, n_zones]
                Q_solar: Solar heat gains through windows [W]
        """
        # Convert time to tensor for calculations
        t_tensor = torch.full_like(T_outdoor, float(t))

        # Calculate solar gains
        Q_solar = self.calculate_solar_gains(t_tensor, T_outdoor, weather_factor, day_of_year)

        self.diagnostics = {}

        return {
            'Q_solar': Q_solar,  # [W] Solar gains through windows
        }

    @property
    def input_functions(self):
        """
        Context-aware input functions for SolarGains component.

        Returns functions that generate realistic solar input patterns based on
        building context for coordinated simulation.

        Returns:
            dict: Mapping from input variable names to callables (t, batch_size) -> torch.Tensor
        """
        if not hasattr(self, '_input_functions'):
            # Get context values with fallbacks
            T_outdoor_base = self.context.get("T_outdoor", 293.15)  # Default: 20°C
            day_of_year = self.context.get("day_of_year", 100)
            weather_factor_base = self.context.get("weather_factor", 0.7)  # Default: partly cloudy

            def weather_factor_fn(t, batch_size=1):
                """Context-aware weather factor with day/night cycle."""
                # Calculate current hour using context time as baseline
                current_hour = (t / 3600.0) % 24

                # Weather factor follows solar availability (zero at night)
                if 6 <= current_hour <= 18:  # Daylight hours
                    # Use context weather factor during daylight
                    weather_factor = weather_factor_base

                    # Add realistic cloud variation during the day
                    cloud_variation = 0.1 * torch.sin(torch.tensor(2 * torch.pi * (current_hour - 10) / 8))
                    weather_factor = torch.clamp(
                        torch.tensor(weather_factor + cloud_variation), 0.0, 1.0
                    )
                else:
                    # No solar radiation at night
                    weather_factor = torch.tensor(0.0)

                return torch.full((batch_size, 1), weather_factor,
                                  device=self.device, dtype=self.dtype)

            def day_of_year_fn(t, batch_size=1):
                """Context-aware day of year with progression."""
                # Start from context day and progress with simulation time
                sim_days = t / 86400.0
                current_day = (day_of_year + sim_days) % 365
                if current_day == 0:
                    current_day = 365

                return torch.full((batch_size, 1), current_day,
                                  device=self.device, dtype=self.dtype)

            def T_outdoor_fn(t, batch_size=1):
                daily_amplitude = 10.0
                peak_hour = 14.
                seasonal_amplitude = 20.0
                base_temp = T_outdoor_base
                day_of_year = day_of_year_fn(t).item()
                t_hr = t / 3600.0
                if day_of_year is None:
                    day_of_year = int((t / 86400) % 365) + 1
                daily_temp = daily_amplitude * np.sin(2 * np.pi * (t_hr - peak_hour) / 24)
                seasonal_temp = seasonal_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                total_temp = base_temp + daily_temp + seasonal_temp
                total_temp = torch.full((batch_size, 1), total_temp)
                return total_temp

            self._input_functions = {
                "T_outdoor": T_outdoor_fn,
                "weather_factor": weather_factor_fn,
                "day_of_year": day_of_year_fn,
            }
        return self._input_functions

    # @input_functions.setter
    # def input_functions(self, value):
    #     """Allow custom input functions to be set."""
    #     if not hasattr(self, '_input_functions'):
    #         self._input_functions = {}
    #     self._input_functions.update(value)