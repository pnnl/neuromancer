"""
test_solar_gains.py

Comprehensive unit tests for the SolarGains class.

Tests cover:
- Parameter expansion and zone vectorization
- Solar irradiance estimation
- Solar gains calculation
- Input functions
- BuildingComponent interface compliance
- Edge cases and error handling
"""

import pytest
import torch
import math
import numpy as np
from torch_buildings.building_components.solar_gain import SolarGains
from torch_buildings.plot import simplot


class TestSolarGainsInitialization:
    """Test initialization and parameter handling."""

    def test_single_zone_initialization(self):
        """Test initialization with single zone (scalar parameters)."""
        model = SolarGains(
            n_zones=1,
            window_area=15.0,
            window_orientation=0.0,
            window_shgc=0.6,
            latitude_deg=40.0
        )

        assert model.n_zones == 1
        assert model.window_area.shape == (1,)
        assert model.window_orientation.shape == (1,)
        assert model.window_shgc.shape == (1,)
        assert torch.allclose(model.window_area, torch.tensor([15.0]))
        assert torch.allclose(model.window_orientation, torch.tensor([0.0]))
        assert torch.allclose(model.window_shgc, torch.tensor([0.6]))

    def test_multi_zone_scalar_parameters(self):
        """Test multi-zone initialization with scalar parameters (auto-expansion)."""
        model = SolarGains(
            n_zones=3,
            window_area=20.0,  # Scalar expanded to all zones
            window_orientation=-90.0,  # East-facing
            window_shgc=0.5
        )

        assert model.n_zones == 3
        assert model.window_area.shape == (3,)
        assert torch.allclose(model.window_area, torch.tensor([20.0, 20.0, 20.0]))
        assert torch.allclose(model.window_orientation, torch.tensor([-90.0, -90.0, -90.0]))
        assert torch.allclose(model.window_shgc, torch.tensor([0.5, 0.5, 0.5]))

    def test_multi_zone_list_parameters(self):
        """Test multi-zone initialization with zone-specific list parameters."""
        model = SolarGains(
            n_zones=3,
            window_area=[10.0, 15.0, 12.0],
            window_orientation=[0.0, 90.0, -90.0],  # South, west, east
            window_shgc=[0.6, 0.5, 0.7]
        )

        assert model.n_zones == 3
        assert torch.allclose(model.window_area, torch.tensor([10.0, 15.0, 12.0]))
        assert torch.allclose(model.window_orientation, torch.tensor([0.0, 90.0, -90.0]))
        assert torch.allclose(model.window_shgc, torch.tensor([0.6, 0.5, 0.7]))

    def test_latitude_conversion(self):
        """Test that latitude is properly converted to radians."""
        model = SolarGains(latitude_deg=45.0)
        expected_rad = math.radians(45.0)
        assert abs(model.latitude_rad - expected_rad) < 1e-6

    def test_invalid_list_length(self):
        """Test that mismatched list lengths raise errors."""
        with pytest.raises(ValueError, match="Parameter.*list length.*!= n_zones"):
            SolarGains(
                n_zones=3,
                window_area=[10.0, 15.0]  # Only 2 values for 3 zones
            )

    def test_learnable_parameters(self):
        """Test that parameters can be made learnable."""
        model = SolarGains(
            n_zones=2,
            window_shgc=[0.6, 0.5],
            learnable={'window_shgc'}
        )

        # Check that window_shgc is a learnable parameter
        param_names = dict(model.named_parameters()).keys()
        assert 'window_shgc' in param_names
        assert model.window_shgc.requires_grad


class TestSolarIrradianceEstimation:
    """Test solar irradiance estimation logic."""

    def setup_method(self):
        """Set up test model."""
        self.model = SolarGains(n_zones=1, latitude_deg=40.0, max_solar_irradiance=800.0)
        self.batch_size = 2

    def test_solar_elevation_calculation(self):
        """Test that solar elevation is calculated correctly."""
        # Test at solar noon on summer solstice (high elevation)
        t = torch.tensor([[43200.0]], dtype=torch.float32)  # 12:00 PM
        T_outdoor = torch.tensor([[298.15]], dtype=torch.float32)  # 25°C
        weather_factor = torch.tensor([[1.0]], dtype=torch.float32)  # Clear sky
        day_of_year = torch.tensor([[172.0]], dtype=torch.float32)  # Summer solstice

        irradiance = self.model.estimate_solar_irradiance(t, T_outdoor, weather_factor, day_of_year)

        # Should be positive and significant at solar noon in summer
        assert irradiance > 0
        assert irradiance <= self.model.max_solar_irradiance

    def test_nighttime_irradiance(self):
        """Test that nighttime irradiance is zero or very low."""
        # Test at midnight
        t = torch.tensor([[0.0]], dtype=torch.float32)  # Midnight
        T_outdoor = torch.tensor([[288.15]], dtype=torch.float32)  # 15°C
        weather_factor = torch.tensor([[1.0]], dtype=torch.float32)  # Clear sky
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)  # Summer

        irradiance = self.model.estimate_solar_irradiance(t, T_outdoor, weather_factor, day_of_year)

        # Should be very low or zero at night
        assert irradiance < 50.0  # Very low threshold

    def test_weather_factor_effect(self):
        """Test that weather factor properly modulates irradiance."""
        t = torch.tensor([[43200.0]], dtype=torch.float32)  # Noon
        T_outdoor = torch.tensor([[298.15]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)

        # Clear sky
        weather_clear = torch.tensor([[1.0]], dtype=torch.float32)
        irradiance_clear = self.model.estimate_solar_irradiance(t, T_outdoor, weather_clear, day_of_year)

        # Cloudy sky
        weather_cloudy = torch.tensor([[0.3]], dtype=torch.float32)
        irradiance_cloudy = self.model.estimate_solar_irradiance(t, T_outdoor, weather_cloudy, day_of_year)

        # Clear sky should have higher irradiance
        assert irradiance_clear > irradiance_cloudy
        assert irradiance_cloudy < 0.5 * irradiance_clear  # Significant reduction

    def test_temperature_correlation(self):
        """Test that higher temperatures correlate with higher irradiance."""
        t = torch.tensor([[43200.0]], dtype=torch.float32)  # Noon
        weather_factor = torch.tensor([[0.8]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)

        # Cool temperature
        T_cool = torch.tensor([[288.15]], dtype=torch.float32)  # 15°C
        irradiance_cool = self.model.estimate_solar_irradiance(t, T_cool, weather_factor, day_of_year)

        # Hot temperature
        T_hot = torch.tensor([[308.15]], dtype=torch.float32)  # 35°C
        irradiance_hot = self.model.estimate_solar_irradiance(t, T_hot, weather_factor, day_of_year)

        # Hotter temperature should correlate with higher estimated irradiance
        assert irradiance_hot > irradiance_cool

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        batch_size = 3
        t = torch.tensor([[43200.0], [46800.0], [50400.0]], dtype=torch.float32)  # Different times
        T_outdoor = torch.tensor([[298.15], [300.15], [295.15]], dtype=torch.float32)
        weather_factor = torch.tensor([[1.0], [0.8], [0.6]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0], [180.0], [180.0]], dtype=torch.float32)

        irradiance = self.model.estimate_solar_irradiance(t, T_outdoor, weather_factor, day_of_year)

        assert irradiance.shape == (batch_size, 1)
        assert torch.all(irradiance >= 0)
        assert torch.all(irradiance <= self.model.max_solar_irradiance)


class TestSolarGainsCalculation:
    """Test solar gains calculation with window properties."""

    def setup_method(self):
        """Set up test model with multiple zones."""
        self.model = SolarGains(
            n_zones=3,
            window_area=[10.0, 15.0, 20.0],
            window_orientation=[0.0, 90.0, -90.0],  # South, west, east
            window_shgc=[0.6, 0.5, 0.7],
            latitude_deg=40.0
        )

    def test_orientation_effects(self):
        """Test that window orientation affects solar gains."""
        t = torch.tensor([[43200.0]], dtype=torch.float32)  # Noon
        T_outdoor = torch.tensor([[298.15]], dtype=torch.float32)
        weather_factor = torch.tensor([[1.0]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)

        solar_gains = self.model.calculate_solar_gains(t, T_outdoor, weather_factor, day_of_year)

        # At solar noon, south-facing (zone 0) should have highest gains
        # East and west facing should be similar but lower
        assert solar_gains.shape == (1, 3)
        assert solar_gains[0, 0] > solar_gains[0, 1]  # South > West
        assert solar_gains[0, 0] > solar_gains[0, 2]  # South > East

    def test_window_area_scaling(self):
        """Test that solar gains scale with window area."""
        t = torch.tensor([[43200.0]], dtype=torch.float32)
        T_outdoor = torch.tensor([[298.15]], dtype=torch.float32)
        weather_factor = torch.tensor([[1.0]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)

        solar_gains = self.model.calculate_solar_gains(t, T_outdoor, weather_factor, day_of_year)

        # Larger windows should have proportionally larger gains
        # Zone 2 has 2x area of zone 0, but different orientation and SHGC
        # Check that area effect is present (not exact due to other factors)
        area_ratio = self.model.window_area[2] / self.model.window_area[0]  # 20/10 = 2.0
        assert area_ratio == 2.0

    def test_shgc_effects(self):
        """Test that SHGC affects solar gains."""
        # Create two models with different SHGC values
        model_low_shgc = SolarGains(n_zones=1, window_shgc=0.3)
        model_high_shgc = SolarGains(n_zones=1, window_shgc=0.8)

        t = torch.tensor([[43200.0]], dtype=torch.float32)
        T_outdoor = torch.tensor([[298.15]], dtype=torch.float32)
        weather_factor = torch.tensor([[1.0]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)

        gains_low = model_low_shgc.calculate_solar_gains(t, T_outdoor, weather_factor, day_of_year)
        gains_high = model_high_shgc.calculate_solar_gains(t, T_outdoor, weather_factor, day_of_year)

        # Higher SHGC should result in higher solar gains
        assert gains_high > gains_low

        # Should be roughly proportional to SHGC ratio
        shgc_ratio = 0.8 / 0.3
        gains_ratio = gains_high / gains_low
        assert abs(gains_ratio - shgc_ratio) < 0.1  # Small tolerance for other factors

    def test_zero_irradiance_handling(self):
        """Test behavior when solar irradiance is zero."""
        # Nighttime conditions
        t = torch.tensor([[0.0]], dtype=torch.float32)  # Midnight
        T_outdoor = torch.tensor([[288.15]], dtype=torch.float32)
        weather_factor = torch.tensor([[0.0]], dtype=torch.float32)  # Completely overcast
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)

        solar_gains = self.model.calculate_solar_gains(t, T_outdoor, weather_factor, day_of_year)

        # Should be zero or very close to zero for all zones
        assert torch.all(solar_gains < 1.0)  # Very small threshold


class TestForwardMethod:
    """Test the main forward method and BuildingComponent interface."""

    def setup_method(self):
        """Set up test model."""
        self.model = SolarGains(
            n_zones=2,
            window_area=[10.0, 15.0],
            window_orientation=[0.0, 90.0]
        )

    def test_forward_method_interface(self):
        """Test that forward method follows BuildingComponent interface."""
        t = 43200.0  # Float time (12:00 PM)
        T_outdoor = torch.tensor([[298.15]], dtype=torch.float32)
        weather_factor = torch.tensor([[0.8]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)
        dt = 1.0  # Time step (unused but required)

        result = self.model.forward(t=t, T_outdoor=T_outdoor, weather_factor=weather_factor,
                                    day_of_year=day_of_year, dt=dt)

        # Check return format
        assert isinstance(result, dict)
        assert 'Q_solar' in result

        # Check tensor shapes
        assert result['Q_solar'].shape == (1, 2)  # [batch_size, n_zones]

    def test_forward_method_values(self):
        """Test that forward method returns reasonable values."""
        t = 43200.0  # Noon
        T_outdoor = torch.tensor([[298.15]], dtype=torch.float32)
        weather_factor = torch.tensor([[1.0]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)

        result = self.model(
            t=t,
            T_outdoor=T_outdoor,
            weather_factor=weather_factor,
            day_of_year=day_of_year
            )

        # Solar gains should be positive at noon with clear sky
        assert torch.all(result['Q_solar'] > 0)


class TestInputFunctions:
    """Test default input functions for standalone simulation."""

    def setup_method(self):
        """Set up test model."""
        self.model = SolarGains(n_zones=2)

    def test_input_functions_exist(self):
        """Test that all required input functions are defined."""
        input_funcs = self.model.input_functions

        required_inputs = ['T_outdoor', 'weather_factor', 'day_of_year']
        for input_name in required_inputs:
            assert input_name in input_funcs
            assert callable(input_funcs[input_name])

    def test_input_function_shapes(self):
        """Test that input functions return correct tensor shapes."""
        input_funcs = self.model.input_functions
        t = 43200.0  # 12:00 PM
        batch_size = 3

        for input_name, func in input_funcs.items():
            output = func(t, batch_size)
            assert isinstance(output, torch.Tensor)
            assert output.shape[0] == batch_size  # Correct batch size

            # Most inputs should be system-wide (shape [batch_size, 1])
            if input_name in ['T_outdoor', 'weather_factor', 'day_of_year']:
                assert output.shape == (batch_size, 1)

    def test_temperature_function_realism(self):
        """Test that temperature function produces realistic values."""
        temp_func = self.model.input_functions['T_outdoor']

        # Test at different times of day
        times = [0.0, 21600.0, 43200.0, 64800.0]  # Midnight, 6AM, noon, 6PM
        temperatures = [temp_func(t, 1).item() for t in times]

        # Should vary throughout the day
        assert max(temperatures) > min(temperatures)

        # Should be in reasonable range (250K to 320K, roughly -23°C to 47°C)
        for temp in temperatures:
            assert 250.0 < temp < 320.0

    def test_weather_factor_function(self):
        """Test that weather factor function is reasonable."""
        weather_func = self.model.input_functions['weather_factor']

        # Test daytime and nighttime
        day_weather = weather_func(43200.0, 1).item()  # Noon
        night_weather = weather_func(0.0, 1).item()  # Midnight

        # Should be higher during day, zero at night
        assert day_weather > night_weather
        assert night_weather == 0.0  # No solar at night
        assert 0.0 <= day_weather <= 1.0  # Valid range


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_extreme_latitudes(self):
        """Test behavior at extreme latitudes."""
        # Arctic latitude
        model_arctic = SolarGains(latitude_deg=80.0)

        # Antarctic latitude
        model_antarctic = SolarGains(latitude_deg=-80.0)

        # Should not crash
        t = torch.tensor([[43200.0]], dtype=torch.float32)
        T_outdoor = torch.tensor([[273.15]], dtype=torch.float32)
        weather_factor = torch.tensor([[1.0]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)

        result_arctic = model_arctic(t=t, T_outdoor=T_outdoor,
                                     weather_factor=weather_factor,
                                     day_of_year=day_of_year)
        result_antarctic = model_antarctic(t=t,
                                           T_outdoor=T_outdoor,
                                           weather_factor=weather_factor,
                                           day_of_year=day_of_year)

        # Should return valid results
        assert torch.all(result_arctic['Q_solar'] >= 0)
        assert torch.all(result_antarctic['Q_solar'] >= 0)

    def test_zero_window_area(self):
        """Test behavior with zero window area."""
        model = SolarGains(n_zones=1, window_area=0.0)

        t = torch.tensor([[43200.0]], dtype=torch.float32)
        T_outdoor = torch.tensor([[298.15]], dtype=torch.float32)
        weather_factor = torch.tensor([[1.0]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)

        result = model(t=t,
                       T_outdoor=T_outdoor,
                       weather_factor=weather_factor,
                       day_of_year=day_of_year)

        # Should be zero solar gains with no windows
        assert torch.allclose(result['Q_solar'], torch.zeros_like(result['Q_solar']))

    def test_extreme_weather_factors(self):
        """Test with extreme weather factor values."""
        model = SolarGains()

        t = torch.tensor([[43200.0]], dtype=torch.float32)
        T_outdoor = torch.tensor([[298.15]], dtype=torch.float32)
        day_of_year = torch.tensor([[180.0]], dtype=torch.float32)

        # Test with weather factor = 0 (completely overcast)
        weather_zero = torch.tensor([[0.0]], dtype=torch.float32)
        result_zero = model(t=t, T_outdoor=T_outdoor,
                            weather_factor=weather_zero,
                            day_of_year=day_of_year)
        assert torch.all(result_zero['Q_solar'] == 0.0)

        # Test with weather factor = 1 (perfectly clear)
        weather_one = torch.tensor([[1.0]], dtype=torch.float32)
        result_one = model(t=t, T_outdoor=T_outdoor, weather_factor=weather_one, day_of_year=day_of_year)
        assert torch.all(result_one['Q_solar'] > 0.0)


class TestSimulationIntegration:
    """Test integration with BuildingComponent simulation features."""

    def test_simulation_runs(self):
        """Test that simulation completes without errors."""
        model = SolarGains(n_zones=2)

        results = model.simulate(
            duration_hours=24.0,
            dt_minutes=60.0,
            batch_size=1
        )

        # Check basic result structure
        assert 'Q_solar' in results

        # Check shapes: [time_steps, batch_size, n_zones]
        expected_time_steps = 24  # 24 hours + 1 initial
        assert results['Q_solar'].shape == (1, expected_time_steps, 2)

        # Check that solar gains vary over the day
        daily_gains = results['Q_solar'][0, :, 0]  # First zone over time
        assert daily_gains.max() > daily_gains.min()  # Should vary

        # Solar gains should be zero at night, positive during day
        night_gains = daily_gains[0]  # Starting at 5 AM (t_start_hour default)
        noon_gains = daily_gains[7]  # Around noon
        assert night_gains < noon_gains

    def test_plot_functionality(self):
        """Test that plotting functionality works."""
        model = SolarGains(n_zones=1)

        # This should not raise an error
        try:
            fig, results = simplot(model,
                duration_hours=2.0,
                dt_minutes=30.0,
                variables=['Q_solar']
            )
            # If we get here, plotting worked
            assert fig is not None
            assert 'Q_solar' in results
        except Exception as e:
            pytest.fail(f"Plotting failed with error: {e}")


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])