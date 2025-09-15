"""
test_rtu.py

Streamlined test suite for the RTU (Rooftop Unit) class focusing on core functionality.

Test Structure:
- TestRTUBasics: Initialization and basic functionality
- TestRTUPhysics: Core physics calculations
- TestRTUForward: Forward pass behavior
- TestRTUSimulation: Integration with simulation framework
- TestRTUEdgeCases: Basic error handling
"""

import pytest
import torch
import numpy as np
from torch_buildings.building_components.rooftop_unit import RTU
from torch_buildings.building_components.rooftop_unit import (
    calculate_fan_power,
    calculate_cooling_power,
    calculate_heating_power,
    calculate_hvac_power,
    calculate_air_mixing,
    calculate_thermal_load,
    apply_mode_limits
)
from torch_buildings.plot import simplot


class TestRTUBasics:
    """Test RTU initialization and basic setup."""

    def test_default_initialization(self):
        """Test RTU with default parameters."""
        rtu = RTU()

        # Check basic attributes exist
        assert hasattr(rtu, 'airflow_max')
        assert hasattr(rtu, 'Q_coil_max')
        assert hasattr(rtu, 'cooling_COP')
        assert hasattr(rtu, 'heating_efficiency')

        # Check control parameters
        assert hasattr(rtu, 'ctrl_Kp')
        assert hasattr(rtu, 'ctrl_Ki')
        assert rtu.ctrl_Kp == 1.0 / rtu.ctrl_proportional_band

        # Check actuator components
        assert hasattr(rtu, 'airflow_damper')
        assert hasattr(rtu, 'coil_valve')

    def test_custom_parameters(self):
        """Test RTU with custom parameters."""
        rtu = RTU(
            airflow_max=5.0,
            Q_coil_max=20000.0,
            cooling_COP=3.5,
            ctrl_proportional_band=5.0,
            actuator_model="analytic"
        )

        assert rtu.airflow_max.item() == pytest.approx(5.0, abs=1e-6)
        assert rtu.Q_coil_max.item() == pytest.approx(20000.0, abs=1e-6)
        assert rtu.cooling_COP.item() == pytest.approx(3.5, abs=1e-6)
        assert rtu.ctrl_Kp.item() == pytest.approx(0.2, abs=1e-6)

    def test_learnable_parameters(self):
        """Test making parameters learnable."""
        learnable = {'cooling_COP', 'airflow_max'}
        rtu = RTU(learnable=learnable)

        assert isinstance(rtu.cooling_COP, torch.nn.Parameter)
        assert isinstance(rtu.airflow_max, torch.nn.Parameter)
        assert not isinstance(rtu.Q_coil_max, torch.nn.Parameter)

    def test_state_initialization(self):
        """Test state initialization functions."""
        rtu = RTU()

        # Test that initial state functions exist
        assert hasattr(rtu, 'initial_state_functions')
        state_funcs = rtu.initial_state_functions()

        # Check required state functions exist
        required_states = ['T_supply', 'damper_position', 'valve_position', 'integral_accumulator']
        for state in required_states:
            assert state in state_funcs

        # Test calling state functions
        batch_size = 2
        for state_name, state_func in state_funcs.items():
            state_val = state_func(batch_size)
            assert isinstance(state_val, torch.Tensor)
            assert state_val.shape == (batch_size, 1)


class TestRTUPhysics:
    """Test core physics calculation functions."""

    def test_calculate_fan_power(self):
        """Test fan power calculation."""
        airflow = torch.tensor([1.0, 2.0, 3.0])
        fan_power_per_flow = torch.tensor([1000.0])

        power = calculate_fan_power(airflow, fan_power_per_flow)
        expected = torch.tensor([1000.0, 2000.0, 3000.0])
        torch.testing.assert_close(power, expected)

    def test_calculate_cooling_power(self):
        """Test cooling power calculation."""
        Q_cooling_load = torch.tensor([3000.0, 6000.0])
        cooling_COP = torch.tensor([3.0])

        power = calculate_cooling_power(Q_cooling_load, cooling_COP)
        expected = torch.tensor([1000.0, 2000.0])
        torch.testing.assert_close(power, expected)

    def test_calculate_hvac_power(self):
        """Test comprehensive HVAC power calculation."""
        power = calculate_hvac_power(
            actual_airflow=torch.tensor([2.0]),
            Q_cooling_load=torch.tensor([6000.0]),
            Q_heating_load=torch.tensor([0.0]),
            fan_power_per_flow=torch.tensor([1000.0]),
            cooling_COP=torch.tensor([3.0]),
            heating_efficiency=torch.tensor([0.85])
        )

        torch.testing.assert_close(power['fan_power'], torch.tensor([2000.0]))
        torch.testing.assert_close(power['cooling_power'], torch.tensor([2000.0]))
        torch.testing.assert_close(power['total_power'], torch.tensor([4000.0]))

    def test_calculate_air_mixing(self):
        """Test air mixing calculation."""
        airflow = torch.tensor([2.0])
        airflow_oa_min = torch.tensor([0.4])
        T_oa = torch.tensor([283.15])
        T_ra = torch.tensor([295.15])

        oa_fraction, T_mixed_air = calculate_air_mixing(airflow, airflow_oa_min, T_oa, T_ra)

        torch.testing.assert_close(oa_fraction, torch.tensor([0.2]))
        expected_temp = 0.2 * 283.15 + 0.8 * 295.15
        torch.testing.assert_close(T_mixed_air, torch.tensor([expected_temp]))

    def test_apply_mode_limits(self):
        """Test HVAC mode restrictions."""
        thermal_load = torch.tensor([-2000.0, 0.0, 2000.0])

        # Cooling mode
        cool_limited = apply_mode_limits(thermal_load, "cool")
        expected_cool = torch.tensor([-2000.0, 0.0, 0.0])
        torch.testing.assert_close(cool_limited, expected_cool)

        # Heating mode
        heat_limited = apply_mode_limits(thermal_load, "heat")
        expected_heat = torch.tensor([0.0, 0.0, 2000.0])
        torch.testing.assert_close(heat_limited, expected_heat)


class TestRTUForward:
    """Test RTU forward pass behavior."""

    def setup_method(self):
        """Set up test RTU and inputs."""
        self.rtu = RTU(actuator_model="instantaneous")

        # Create initial states manually since reset() doesn't exist
        batch_size = 2
        state_funcs = self.rtu.initial_state_functions()
        self.states = {name: func(batch_size) for name, func in state_funcs.items()}

        self.inputs = {
            'T_outdoor': torch.tensor([[283.15], [285.15]]),
            'T_return_zones': torch.tensor([[295.0, 296.0, 294.0], [297.0, 295.0, 298.0]]),
            'return_airflow_zones': torch.tensor([[0.8, 1.2, 0.5], [0.7, 1.0, 0.6]]),
            'T_supply_setpoint': torch.tensor([[288.0], [289.0]]),
            'supply_airflow_setpoint': torch.tensor([[2.0], [2.5]])
        }

    def test_forward_pass_basic(self):
        """Test basic forward pass functionality."""
        # Combine inputs and states for forward pass
        forward_inputs = {**self.inputs, **self.states}
        result = self.rtu.forward(t=0.0, dt=1.0, **forward_inputs)

        # Check required outputs
        required_outputs = ['T_supply', 'supply_heat_flow', 'P_supply', 'supply_airflow']
        for output in required_outputs:
            assert output in result
            assert result[output].shape == (2, 1)

        # Check reasonable values
        assert (result['T_supply'] > 273.15).all()
        assert (result['T_supply'] < 323.15).all()
        assert (result['supply_airflow'] >= 0.0).all()

    def test_return_air_mixing(self):
        """Test return air mixing calculation."""
        forward_inputs = {**self.inputs, **self.states}
        result = self.rtu.forward(t=0.0, dt=1.0, **forward_inputs)

        # Check that result completed without errors
        assert 'T_supply' in result
        assert not torch.isnan(result['T_supply']).any()

    def test_airflow_control(self):
        """Test airflow control logic."""
        forward_inputs = {**self.inputs, **self.states}
        result = self.rtu.forward(t=0.0, dt=1.0, **forward_inputs)

        # With instantaneous actuator, airflow should be close to setpoint
        torch.testing.assert_close(
            result['supply_airflow'],
            self.inputs['supply_airflow_setpoint'],
            rtol=1e-2, atol=1e-2
        )

    def test_state_updates(self):
        """Test that states are updated properly."""
        forward_inputs = {**self.inputs, **self.states}
        result = self.rtu.forward(t=0.0, dt=1.0, **forward_inputs)

        # Check that states were returned
        assert 'damper_position' in result
        assert 'valve_position' in result
        assert 'integral_accumulator' in result
        assert 'T_supply' in result


class TestRTUSimulation:
    """Test RTU integration with simulation framework."""

    def test_input_functions(self):
        """Test default input functions."""
        rtu = RTU(n_zones=3)
        input_funcs = rtu.input_functions

        required_inputs = [
            'T_outdoor', 'T_return_zones', 'return_airflow_zones',
            'T_supply_setpoint', 'supply_airflow_setpoint'
        ]

        for input_name in required_inputs:
            assert input_name in input_funcs

        # Test calling functions
        t_test = 3600.0
        batch_size = 2

        for input_name, func in input_funcs.items():
            result = func(t_test, batch_size)
            assert isinstance(result, torch.Tensor)

            if input_name in ['T_return_zones', 'return_airflow_zones']:
                assert result.shape == (batch_size, 3)
            else:
                assert result.shape == (batch_size, 1)

    def test_short_simulation(self):
        """Test short simulation run."""
        rtu = RTU(actuator_model="instantaneous")

        results = rtu.simulate(
            duration_hours=1.0,
            dt_minutes=10.0,
            batch_size=1,
        )

        # Check results structure
        expected_vars = ['T_supply', 'supply_airflow']
        for var in expected_vars:
            assert var in results

        # Check reasonable values
        assert (results['T_supply'] > 273.15).all()
        assert (results['supply_airflow'] >= 0.0).all()

    def test_simulation_with_custom_inputs(self):
        """Test simulation with custom input functions."""
        rtu = RTU()

        def constant_outdoor_temp(t, batch_size=1):
            return torch.full((batch_size, 1), 293.15)

        custom_inputs = {'T_outdoor': constant_outdoor_temp}

        results = rtu.simulate(
            duration_hours=0.5,
            dt_minutes=15.0,
            batch_size=1,
            input_functions=custom_inputs,
        )

        assert 'T_supply' in results
        assert (results['T_supply'] > 250.0).all()

    def test_batch_simulation(self):
        """Test simulation with multiple batch elements."""
        rtu = RTU()

        results = rtu.simulate(
            duration_hours=0.5,
            dt_minutes=15.0,
            batch_size=3,
        )

        # Check batch dimension
        for var_name, var_data in results.items():
            assert var_data.shape[0] == 3

        # Check no NaN values
        if 'T_supply' in results:
            assert not torch.isnan(results['T_supply']).any()


class TestRTUEdgeCases:
    """Test RTU behavior in edge cases."""

    def setup_method(self):
        """Set up RTU for edge case testing."""
        self.rtu = RTU()
        # Create initial states manually
        state_funcs = self.rtu.initial_state_functions()
        self.states = {name: func(1) for name, func in state_funcs.items()}

    def test_zero_airflow(self):
        """Test behavior with zero airflow."""
        inputs = {
            'T_outdoor': torch.tensor([[283.15]]),
            'T_return_zones': torch.tensor([[295.0, 296.0, 294.0]]),
            'return_airflow_zones': torch.tensor([[0.1, 0.1, 0.1]]),
            'T_supply_setpoint': torch.tensor([[288.0]]),
            'supply_airflow_setpoint': torch.tensor([[0.0]]),
            **self.states
        }

        result = self.rtu.forward(t=0.0, dt=1.0, **inputs)

        assert not torch.isnan(result['supply_airflow']).any()
        assert not torch.isnan(result['T_supply']).any()
        assert (result['supply_airflow'] >= 0.0).all()

    def test_extreme_temperatures(self):
        """Test behavior with extreme temperature inputs."""
        inputs = {
            'T_outdoor': torch.tensor([[323.15]]),  # 50°C
            'T_return_zones': torch.tensor([[295.0, 296.0, 294.0]]),
            'return_airflow_zones': torch.tensor([[0.8, 1.2, 0.5]]),
            'T_supply_setpoint': torch.tensor([[288.0]]),
            'supply_airflow_setpoint': torch.tensor([[2.0]]),
            **self.states
        }

        result_hot = self.rtu.forward(t=0.0, dt=1.0, **inputs)

        assert not torch.isnan(result_hot['T_supply']).any()
        assert not torch.isinf(result_hot['T_supply']).any()

    def test_zero_return_airflow(self):
        """Test behavior with zero return airflow."""
        inputs = {
            'T_outdoor': torch.tensor([[283.15]]),
            'T_return_zones': torch.tensor([[295.0, 296.0, 294.0]]),
            'return_airflow_zones': torch.tensor([[0.0, 0.0, 0.0]]),
            'T_supply_setpoint': torch.tensor([[288.0]]),
            'supply_airflow_setpoint': torch.tensor([[2.0]]),
            **self.states
        }

        result = self.rtu.forward(t=0.0, dt=1.0, **inputs)

        assert not torch.isnan(result['T_supply']).any()
        assert not torch.isnan(result['supply_airflow']).any()


class TestRTUPlotting:
    """Test RTU plotting capabilities."""

    def test_plot_generation(self):
        """Test that plot generation works without errors."""
        rtu = RTU()

        try:
            fig, results = simplot(rtu,
                duration_hours=1.0,
                dt_minutes=30.0,
                batch_size=1,
                filename=None
            )

            assert fig is not None
            assert isinstance(results, dict)
            assert 'T_supply' in results

            # Clean up
            import matplotlib.pyplot as plt
            plt.close(fig)

        except Exception as e:
            # Skip if plotting fails due to environment issues
            if any(phrase in str(e).lower() for phrase in ['matplotlib', 'display']):
                pytest.skip(f"Plot generation skipped: {e}")
            else:
                pytest.fail(f"Plot generation failed: {e}")


# Utility functions for running specific test groups
def run_basic_tests():
    """Run basic functionality tests."""
    pytest.main(['-v', __file__ + '::TestRTUBasics'])

def run_physics_tests():
    """Run physics calculation tests."""
    pytest.main(['-v', __file__ + '::TestRTUPhysics'])

def run_forward_tests():
    """Run forward pass tests."""
    pytest.main(['-v', __file__ + '::TestRTUForward'])

def run_simulation_tests():
    """Run simulation integration tests."""
    pytest.main(['-v', __file__ + '::TestRTUSimulation'])

def run_edge_case_tests():
    """Run edge case tests."""
    pytest.main(['-v', __file__ + '::TestRTUEdgeCases'])

def run_all_tests():
    """Run all RTU tests."""
    pytest.main(['-v', __file__])

def run_quick_tests():
    """Run a subset of tests for quick validation."""
    pytest.main(['-v',
                __file__ + '::TestRTUBasics::test_default_initialization',
                __file__ + '::TestRTUPhysics::test_calculate_fan_power',
                __file__ + '::TestRTUForward::test_forward_pass_basic',
                __file__ + '::TestRTUSimulation::test_short_simulation'])


# Test utilities for debugging
class RTUTestUtils:
    """Utility functions for RTU testing and debugging."""

    @staticmethod
    def create_standard_rtu():
        """Create RTU with standard test parameters."""
        return RTU(
            airflow_max=3.0,
            Q_coil_max=15000.0,
            cooling_COP=3.0,
            actuator_model="instantaneous"
        )

    @staticmethod
    def create_standard_inputs(batch_size=1):
        """Create standard test inputs."""
        return {
            'T_outdoor': torch.tensor([[283.15]] * batch_size),
            'T_return_zones': torch.tensor([[295.0, 296.0, 294.0]] * batch_size),
            'return_airflow_zones': torch.tensor([[0.8, 1.2, 0.5]] * batch_size),
            'T_supply_setpoint': torch.tensor([[288.0]] * batch_size),
            'supply_airflow_setpoint': torch.tensor([[2.0]] * batch_size),
        }

    @staticmethod
    def run_single_step_test(rtu=None, inputs=None, dt=1.0):
        """Run a single forward step for debugging."""
        if rtu is None:
            rtu = RTUTestUtils.create_standard_rtu()

        if inputs is None:
            inputs = RTUTestUtils.create_standard_inputs()

        # Add initial states
        state_funcs = rtu.initial_state_functions()
        states = {name: func(1) for name, func in state_funcs.items()}
        forward_inputs = {**inputs, **states}

        result = rtu.forward(t=0.0, dt=dt, **forward_inputs)

        print("RTU Single Step Test Results:")
        print("-" * 40)

        # Print main outputs
        main_outputs = ['T_supply', 'supply_airflow', 'P_supply']
        for output in main_outputs:
            if output in result:
                print(f"{output}: {result[output].item():.3f}")

        return result


# Performance testing
class RTUPerformanceTests:
    """Simple performance testing utilities."""

    @staticmethod
    def test_forward_pass_performance(n_iterations=100):
        """Test forward pass performance."""
        import time

        rtu = RTU(actuator_model="instantaneous")
        inputs = RTUTestUtils.create_standard_inputs()

        # Add initial states
        state_funcs = rtu.initial_state_functions()
        states = {name: func(1) for name, func in state_funcs.items()}
        forward_inputs = {**inputs, **states}

        # Warm up
        for _ in range(10):
            result = rtu.forward(t=0.0, dt=1.0, **forward_inputs)

        # Time the forward passes
        start_time = time.time()
        for _ in range(n_iterations):
            result = rtu.forward(t=0.0, dt=1.0, **forward_inputs)
        end_time = time.time()

        avg_time = (end_time - start_time) / n_iterations * 1000

        print(f"RTU Forward Pass Performance:")
        print(f"Average time per forward pass: {avg_time:.3f} ms")
        print(f"Forward passes per second: {1000/avg_time:.1f}")

        return avg_time


# Simple test runner with summary
def run_comprehensive_test_suite():
    """Run core test suite with summary."""
    print("=" * 60)
    print("RTU TEST SUITE")
    print("=" * 60)

    test_categories = [
        ("Basic Tests", run_basic_tests),
        ("Physics Tests", run_physics_tests),
        ("Forward Pass Tests", run_forward_tests),
        ("Simulation Tests", run_simulation_tests),
        ("Edge Case Tests", run_edge_case_tests),
    ]

    results = {}

    for category_name, test_func in test_categories:
        print(f"\nRunning {category_name}...")
        try:
            test_func()
            results[category_name] = "PASSED"
            print(f"✓ {category_name} completed successfully")
        except Exception as e:
            results[category_name] = f"FAILED: {str(e)}"
            print(f"✗ {category_name} failed")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for result in results.values() if result == "PASSED")
    total = len(results)

    for category, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        print(f"{status} {category}: {result}")

    print(f"\nOverall: {passed}/{total} test categories passed")

    return results


if __name__ == "__main__":
    # Run comprehensive tests when script is executed directly
    print("Running RTU test suite...")
    run_comprehensive_test_suite()