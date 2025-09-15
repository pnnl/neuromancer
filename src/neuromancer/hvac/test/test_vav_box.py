"""
test_vav_box.py - Updated for new base class architecture

Streamlined test suite for the VAVBox class, updated for the new stateless component design:
1. No more reset() method - using initial_state_functions()
2. Forward pass now includes state inputs (damper_position, reheat_position)
3. Simulation framework handles state management externally
4. Updated method names and signatures

Run with: python -m pytest test_vav_box.py -v
"""

import pytest
import torch
import numpy as np
from torch_buildings.building_components.vav_box import VAVBox, calculate_reheat_load


class TestVAVBoxInitialization:
    """Test VAVBox initialization and parameter handling."""

    def test_default_single_zone_initialization(self):
        """Test VAVBox with default parameters for single zone."""
        vav = VAVBox()

        # Check basic attributes
        assert hasattr(vav, 'n_zones')
        assert vav.n_zones == 1
        assert hasattr(vav, 'control_gain')
        assert hasattr(vav, 'airflow_min')
        assert hasattr(vav, 'airflow_max')
        assert hasattr(vav, 'Q_reheat_max')
        assert hasattr(vav, 'cp_air')

        # Check sub-components
        assert hasattr(vav, 'damper')
        assert hasattr(vav, 'electric_reheat_coil')
        assert vav.damper is not None
        assert vav.electric_reheat_coil is not None

        # Check parameter values and types
        assert isinstance(vav.control_gain, torch.Tensor)
        assert isinstance(vav.airflow_min, torch.Tensor)
        assert isinstance(vav.airflow_max, torch.Tensor)
        assert vav.control_gain.item() == pytest.approx(2.0, abs=1e-6)
        assert vav.airflow_min.item() == pytest.approx(0.05, abs=1e-6)
        assert vav.airflow_max.item() == pytest.approx(0.3, abs=1e-6)

    def test_multi_zone_initialization(self):
        """Test VAVBox initialization with multiple zones."""
        n_zones = 5
        vav = VAVBox(n_zones=n_zones)

        # Check zone count
        assert vav.n_zones == n_zones

        # Check that zone-specific parameters are expanded correctly
        assert vav.control_gain.shape == (n_zones,)
        assert vav.airflow_min.shape == (n_zones,)
        assert vav.airflow_max.shape == (n_zones,)
        assert vav.tau_damper.shape == (n_zones,)
        assert vav.Q_reheat_max.shape == (n_zones,)

    def test_custom_zone_specific_parameters(self):
        """Test VAVBox with zone-specific parameter lists."""
        n_zones = 3
        control_gains = [1.5, 2.0, 2.5]
        airflow_mins = [0.03, 0.05, 0.04]
        airflow_maxs = [0.2, 0.3, 0.25]
        Q_reheat_maxs = [2000.0, 3000.0, 2500.0]

        vav = VAVBox(
            n_zones=n_zones,
            control_gain=control_gains,
            airflow_min=airflow_mins,
            airflow_max=airflow_maxs,
            Q_reheat_max=Q_reheat_maxs,
            tau_damper=10.0,  # Scalar - should be expanded
            reheat_efficiency=0.95,  # Scalar - should be expanded
        )

        # Check that list parameters are converted to tensors correctly
        torch.testing.assert_close(vav.control_gain, torch.tensor(control_gains))
        torch.testing.assert_close(vav.airflow_min, torch.tensor(airflow_mins))
        torch.testing.assert_close(vav.airflow_max, torch.tensor(airflow_maxs))
        torch.testing.assert_close(vav.Q_reheat_max, torch.tensor(Q_reheat_maxs))

        # Check that scalar parameters are expanded
        assert vav.tau_damper.shape == (n_zones,)
        assert vav.reheat_efficiency.shape == (n_zones,)
        assert torch.allclose(vav.tau_damper, torch.tensor([10.0, 10.0, 10.0]))

    def test_learnable_parameters(self):
        """Test making parameters learnable."""
        learnable = {'control_gain', 'Q_reheat_max', 'cp_air', 'airflow_min'}
        vav = VAVBox(n_zones=2, learnable=learnable)

        # Check that learnable parameters are nn.Parameters
        assert isinstance(vav.control_gain, torch.nn.Parameter)
        assert isinstance(vav.Q_reheat_max, torch.nn.Parameter)
        assert isinstance(vav.cp_air, torch.nn.Parameter)
        assert isinstance(vav.airflow_min, torch.nn.Parameter)

        # Check that non-learnable parameters are buffers
        assert not isinstance(vav.airflow_max, torch.nn.Parameter)
        assert not isinstance(vav.P_nominal, torch.nn.Parameter)

        # Check gradient requirements
        assert vav.control_gain.requires_grad
        assert vav.Q_reheat_max.requires_grad

    def test_initial_state_functions(self):
        """Test initial state functions."""
        vav = VAVBox(n_zones=3)

        # Test that initial state functions exist
        state_funcs = vav.initial_state_functions()
        assert 'damper_position' in state_funcs
        assert 'reheat_position' in state_funcs

        # Test calling state functions
        batch_size = 2
        damper_pos = state_funcs['damper_position'](batch_size)
        reheat_pos = state_funcs['reheat_position'](batch_size)

        assert isinstance(damper_pos, torch.Tensor)
        assert isinstance(reheat_pos, torch.Tensor)
        assert damper_pos.shape == (batch_size, 3)
        assert reheat_pos.shape == (batch_size, 3)

        # Check reasonable values
        assert torch.all(damper_pos >= 0.0) and torch.all(damper_pos <= 1.0)
        assert torch.all(reheat_pos >= 0.0) and torch.all(reheat_pos <= 1.0)


class TestVAVBoxPhysics:
    """Test individual physics calculation functions."""

    def test_calculate_reheat_load_basic(self):
        """Test basic reheat load calculation."""
        airflow = torch.tensor([1.0, 2.0, 0.5])
        cp_air = torch.tensor([1005.0])
        T_current = torch.tensor([285.0, 280.0, 290.0])  # Current supply temps
        T_min = torch.tensor([288.0, 288.0, 288.0])  # Minimum required temps

        reheat_load = calculate_reheat_load(airflow, cp_air, T_current, T_min)

        # Manual calculation:
        # Zone 1: 1.0 * 1005 * max(0, 288-285) = 1.0 * 1005 * 3 = 3015 W
        # Zone 2: 2.0 * 1005 * max(0, 288-280) = 2.0 * 1005 * 8 = 16080 W
        # Zone 3: 0.5 * 1005 * max(0, 288-290) = 0.5 * 1005 * 0 = 0 W
        expected = torch.tensor([3015.0, 16080.0, 0.0])
        torch.testing.assert_close(reheat_load, expected)

    def test_calculate_reheat_load_no_reheat_needed(self):
        """Test reheat load when current temp exceeds minimum."""
        airflow = torch.tensor([1.0, 1.0])
        cp_air = torch.tensor([1005.0])
        T_current = torch.tensor([290.0, 295.0])  # Already above minimum
        T_min = torch.tensor([288.0, 288.0])

        reheat_load = calculate_reheat_load(airflow, cp_air, T_current, T_min)

        # No reheat needed - should be zero
        expected = torch.tensor([0.0, 0.0])
        torch.testing.assert_close(reheat_load, expected)

    def test_calculate_reheat_load_zero_airflow(self):
        """Test reheat load with zero airflow."""
        airflow = torch.tensor([0.0])
        cp_air = torch.tensor([1005.0])
        T_current = torch.tensor([280.0])
        T_min = torch.tensor([290.0])

        reheat_load = calculate_reheat_load(airflow, cp_air, T_current, T_min)

        # Zero airflow means zero reheat load regardless of temperature difference
        expected = torch.tensor([0.0])
        torch.testing.assert_close(reheat_load, expected)


class TestVAVBoxForward:
    """Test VAVBox forward pass behavior and complete physics."""

    def setup_method(self):
        """Set up test VAVBox for forward pass tests."""
        self.vav = VAVBox(
            n_zones=3,
            control_gain=2.0,
            airflow_min=[0.04, 0.05, 0.03],
            airflow_max=[0.25, 0.30, 0.20],
            Q_reheat_max=[2500.0, 3000.0, 2000.0],
            actuator_model="instantaneous"  # For predictable testing
        )

        # Create initial states manually
        batch_size = 2
        state_funcs = self.vav.initial_state_functions()
        self.states = {name: func(batch_size) for name, func in state_funcs.items()}

        # Standard test inputs
        self.batch_size = 2
        self.n_zones = 3

        # Zone inputs
        self.T_zone = torch.tensor([
            [295.0, 297.0, 293.0],  # Batch 0: varied zone temperatures
            [294.0, 295.0, 296.0]   # Batch 1: different zone pattern
        ])
        self.T_setpoint = torch.tensor([
            [293.0, 293.0, 293.0],  # Batch 0: uniform setpoints
            [292.0, 294.0, 295.0]   # Batch 1: varied setpoints
        ])

        # Supply-side inputs (can be broadcast)
        self.T_supply = torch.tensor([[285.0], [287.0]])  # Different per batch
        self.P_duct = torch.tensor([[500.0], [600.0]])

    def test_forward_pass_basic(self):
        """Test basic forward pass functionality."""
        # Combine inputs and states
        forward_inputs = {
            'T_zone': self.T_zone,
            'T_setpoint': self.T_setpoint,
            'T_supply_upstream': self.T_supply,
            'P_duct': self.P_duct,
            **self.states
        }

        result = self.vav.forward(t=0.0, dt=1.0, **forward_inputs)

        # Check required FluidState outputs exist
        required_outputs = [
            'T_supply', 'Q_supply_flow', 'P_supply', 'supply_airflow'
        ]

        for output in required_outputs:
            assert output in result, f"Missing output: {output}"

        # Check output shapes (all should be [batch_size, n_zones])
        for output in required_outputs:
            expected_shape = (self.batch_size, self.n_zones)
            assert result[output].shape == expected_shape, \
                f"Wrong shape for {output}: {result[output].shape} != {expected_shape}"

        # Check that values are reasonable
        assert (result['T_supply'] > 273.15).all(), "Supply temperature should be above freezing"
        assert (result['T_supply'] < 323.15).all(), "Supply temperature should be below 50°C"
        assert (result['supply_airflow'] >= 0.0).all(), "Supply airflow should be non-negative"
        assert (result['P_supply'] > 0.0).all(), "Supply pressure should be positive"

    def test_state_updates(self):
        """Test that state outputs are updated properly."""
        forward_inputs = {
            'T_zone': self.T_zone,
            'T_setpoint': self.T_setpoint,
            'T_supply_upstream': self.T_supply,
            'P_duct': self.P_duct,
            **self.states
        }

        result = self.vav.forward(t=0.0, dt=1.0, **forward_inputs)

        # Check that states were returned
        assert 'damper_position' in result
        assert 'reheat_position' in result

        # Check shapes
        assert result['damper_position'].shape == (self.batch_size, self.n_zones)
        assert result['reheat_position'].shape == (self.batch_size, self.n_zones)


class TestVAVBoxMultiZone:
    """Test multi-zone operation and vectorization."""

    def test_independent_zone_control(self):
        """Test that zones can be controlled independently."""
        n_zones = 4
        vav = VAVBox(
            n_zones=n_zones,
            control_gain=2.0,
            airflow_min=0.05,
            airflow_max=0.3,
            actuator_model="instantaneous"
        )

        # Create initial states
        state_funcs = vav.initial_state_functions()
        states = {name: func(1) for name, func in state_funcs.items()}

        # Set up zones with different conditions
        inputs = {
            'T_zone': torch.tensor([[293.0, 297.0, 291.0, 299.0]]),  # Varied temperatures
            'T_setpoint': torch.tensor([[293.0, 293.0, 293.0, 293.0]]),  # Same setpoint
            'T_supply_upstream': torch.tensor([[285.0]]),
            'P_duct': torch.tensor([[500.0]]),
            **states
        }

        result = vav.forward(t=0.0, dt=1.0, **inputs)

        # All zones should have independent damper positions
        damper_positions = result['damper_position'][0]
        # Check that not all positions are the same (zones respond independently)
        position_range = torch.max(damper_positions) - torch.min(damper_positions)
        assert position_range > 0.01, "Zones should have different damper positions"


class TestVAVBoxSimulation:
    """Test VAVBox integration with simulation framework."""

    def test_input_functions(self):
        """Test default input functions."""
        vav = VAVBox(n_zones=2)
        input_funcs = vav.input_functions

        required_inputs = ['T_zone', 'T_setpoint', 'T_supply_upstream', 'P_duct']
        for inp in required_inputs:
            assert inp in input_funcs

        # Test calling functions
        t_test = 3600.0
        batch_size = 2

        for input_name, func in input_funcs.items():
            result = func(t_test, batch_size)
            assert isinstance(result, torch.Tensor)

            if input_name in ['T_zone', 'T_setpoint']:
                assert result.shape == (batch_size, 2)
            else:  # T_supply, P_duct
                assert result.shape == (batch_size, 1)

    def test_short_simulation(self):
        """Test short simulation run."""
        vav = VAVBox(n_zones=2, actuator_model="instantaneous")

        try:
            results = vav.simulate(
                duration_hours=0.5,
                dt_minutes=15.0,
                batch_size=1,
            )

            # Check that results contain expected variables
            expected_vars = ['T_supply', 'supply_airflow', 'P_supply', 'Q_supply_flow']
            for var in expected_vars:
                assert var in results, f"Missing variable: {var}"

            # Check trajectory shape
            n_steps = int(0.5 * 60 / 15) + 1  # 3 points
            for var in expected_vars:
                assert results[var].shape[0] == 1, f"Wrong batch dimension for {var}"
                assert results[var].shape[2] == 2, f"Wrong zone dimension for {var}"

            # Check that values are reasonable
            assert (results['T_supply'] > 273.15).all(), "Supply temperature should be above freezing"
            assert (results['T_supply'] < 323.15).all(), "Supply temperature should be below 50°C"
            assert (results['supply_airflow'] >= 0.0).all(), "Supply airflow should be non-negative"

        except Exception as e:
            pytest.fail(f"Simulation failed: {e}")

    def test_simulation_with_custom_inputs(self):
        """Test simulation with custom input functions."""
        vav = VAVBox(n_zones=1)

        def constant_zone_temp(t, batch_size):
            return torch.full((batch_size, 1), 295.0)

        def constant_setpoint(t, batch_size):
            return torch.full((batch_size, 1), 293.0)

        def constant_supply_temp(t, batch_size):
            return torch.full((batch_size, 1), 285.0)

        def constant_pressure(t, batch_size):
            return torch.full((batch_size, 1), 500.0)

        custom_inputs = {
            'T_zone': constant_zone_temp,
            'T_setpoint': constant_setpoint,
            'T_supply_upstream': constant_supply_temp,
            'P_duct': constant_pressure,
        }

        results = vav.simulate(
            duration_hours=0.5,
            dt_minutes=15.0,
            batch_size=1,
            input_functions=custom_inputs,
        )

        assert 'T_supply' in results
        assert 'supply_airflow' in results

class TestVAVBoxEdgeCases:
    """Test VAVBox behavior in edge cases and error conditions."""

    def test_zero_airflow_demand(self):
        """Test behavior when zones are at setpoint (no airflow demand)."""
        vav = VAVBox(n_zones=2)

        # Create initial states
        state_funcs = vav.initial_state_functions()
        states = {name: func(1) for name, func in state_funcs.items()}

        inputs = {
            'T_zone': torch.tensor([[293.0, 293.0]]),  # At setpoint
            'T_setpoint': torch.tensor([[293.0, 293.0]]),  # Same as zone temp
            'T_supply_upstream': torch.tensor([[285.0]]),
            'P_duct': torch.tensor([[500.0]]),
            **states
        }

        result = vav.forward(t=0.0, dt=1.0, **inputs)

        # Should handle zero demand gracefully
        assert not torch.isnan(result['supply_airflow']).any()
        assert not torch.isnan(result['T_supply']).any()
        assert (result['supply_airflow'] >= 0.0).all()

    def test_extreme_temperature_differences(self):
        """Test behavior with extreme temperature differences."""
        vav = VAVBox(n_zones=2)

        # Create initial states
        state_funcs = vav.initial_state_functions()
        states = {name: func(1) for name, func in state_funcs.items()}

        # Very large cooling demand
        inputs = {
            'T_zone': torch.tensor([[310.0, 315.0]]),  # Very hot zones
            'T_setpoint': torch.tensor([[293.0, 293.0]]),  # 20°C setpoint
            'T_supply_upstream': torch.tensor([[275.0]]),  # Very cold supply
            'P_duct': torch.tensor([[500.0]]),
            **states
        }

        result = vav.forward(t=0.0, dt=1.0, **inputs)

        # Should handle extreme conditions without NaN/inf
        assert not torch.isnan(result['T_supply']).any()
        assert not torch.isinf(result['T_supply']).any()
        assert (result['supply_airflow'] >= 0.0).all()


# Test utilities and helpers
class VAVBoxTestUtils:
    """Utility functions for VAV box testing and debugging."""

    @staticmethod
    def create_standard_vav(n_zones=2):
        """Create VAV box with standard test parameters."""
        return VAVBox(
            n_zones=n_zones,
            control_gain=2.0,
            airflow_min=0.05,
            airflow_max=0.30,
            Q_reheat_max=3000.0,
            actuator_model="instantaneous"
        )

    @staticmethod
    def create_standard_inputs(batch_size=1, n_zones=2):
        """Create standard test inputs for VAV box forward pass."""
        # Create VAV box to get initial states
        vav = VAVBoxTestUtils.create_standard_vav(n_zones)
        state_funcs = vav.initial_state_functions()
        states = {name: func(batch_size) for name, func in state_funcs.items()}

        inputs = {
            'T_zone': torch.full((batch_size, n_zones), 295.0),  # 22°C zones
            'T_setpoint': torch.full((batch_size, n_zones), 293.0),  # 20°C setpoint
            'T_supply_upstream': torch.full((batch_size, 1), 285.0),  # 12°C supply air
            'P_duct': torch.full((batch_size, 1), 500.0),  # 500 Pa pressure
        }

        return {**inputs, **states}

    @staticmethod
    def run_single_step_test(vav=None, inputs=None, dt=1.0):
        """Run a single forward step for debugging."""
        if vav is None:
            vav = VAVBoxTestUtils.create_standard_vav()

        if inputs is None:
            inputs = VAVBoxTestUtils.create_standard_inputs()

        result = vav.forward(t=0.0, dt=dt, **inputs)

        print("VAV Box Single Step Test Results:")
        print("-" * 40)

        # Print main outputs
        main_outputs = ['T_supply', 'supply_airflow', 'P_supply', 'Q_supply_flow']
        for output in main_outputs:
            if output in result:
                values = result[output][0]  # First batch
                print(f"{output}: {values}")

        return result


# Helper functions for running specific test groups
def run_initialization_tests():
    """Run only initialization tests."""
    pytest.main(['-v', __file__ + '::TestVAVBoxInitialization'])

def run_physics_tests():
    """Run only physics calculation tests."""
    pytest.main(['-v', __file__ + '::TestVAVBoxPhysics'])

def run_forward_tests():
    """Run only forward pass tests."""
    pytest.main(['-v', __file__ + '::TestVAVBoxForward'])

def run_multizone_tests():
    """Run only multi-zone tests."""
    pytest.main(['-v', __file__ + '::TestVAVBoxMultiZone'])

def run_simulation_tests():
    """Run only simulation integration tests."""
    pytest.main(['-v', __file__ + '::TestVAVBoxSimulation'])

def run_edge_case_tests():
    """Run only edge case tests."""
    pytest.main(['-v', __file__ + '::TestVAVBoxEdgeCases'])

def run_all_tests():
    """Run all VAV box tests."""
    pytest.main(['-v', __file__])

def run_quick_tests():
    """Run a subset of tests for quick validation."""
    pytest.main(['-v',
                __file__ + '::TestVAVBoxInitialization::test_default_single_zone_initialization',
                __file__ + '::TestVAVBoxPhysics::test_calculate_reheat_load_basic',
                __file__ + '::TestVAVBoxForward::test_forward_pass_basic',
                __file__ + '::TestVAVBoxSimulation::test_short_simulation'])


if __name__ == "__main__":
    # Run quick tests when script is executed directly
    print("Running VAV Box test suite...")
    run_quick_tests()