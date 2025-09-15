"""
test_zone_vectorization_batching.py

Comprehensive test suite for zone vectorization and batch processing functionality
across Actuator, Damper, and ElectricReheatCoil components.

Tests both orthogonal capabilities:
1. Zone vectorization: Multiple zones with zone-specific parameters
2. Batch processing: Multiple samples/scenarios processed simultaneously

Coverage includes:
- Parameter expansion (scalar -> zone vector)
- Tensor shape consistency 
- Broadcasting behavior
- Zone-specific vs shared parameters
- Batch + zone combinations
- Edge cases and error conditions
"""

import pytest
import torch
import numpy as np
from torch_buildings import Actuator
from torch_buildings import Damper
from torch_buildings import ElectricReheatCoil


class TestActuatorVectorization:
    """Test suite for Actuator zone vectorization and batching."""

    @pytest.fixture
    def batch_zone_shapes(self):
        """Standard test shapes for batch and zone dimensions."""
        return {
            'batch_size': 3,
            'n_zones': 4,
            'single_batch': 1,
            'single_zone': 1
        }

    def test_scalar_tau_expansion(self, batch_zone_shapes):
        """Test that scalar tau parameters work with zone vectorization."""
        batch_size, n_zones = batch_zone_shapes['batch_size'], batch_zone_shapes['n_zones']

        # Create actuator with scalar tau (should work for all zones)
        actuator = Actuator(tau=10.0, model="analytic")

        # Mock parameter expansion (normally done by BuildingComponent)
        actuator.tau = torch.full((n_zones,), 10.0)

        # Test with vectorized inputs
        setpoint = torch.rand(batch_size, n_zones)
        position = torch.rand(batch_size, n_zones)

        result = actuator.forward(setpoint=setpoint, position=position, dt=1.0)

        assert result.shape == (batch_size, n_zones)
        assert torch.all(torch.isfinite(result))
        assert torch.all((result >= 0) & (result <= 1))

    def test_zone_specific_tau(self, batch_zone_shapes):
        """Test zone-specific tau parameters."""
        batch_size, n_zones = batch_zone_shapes['batch_size'], batch_zone_shapes['n_zones']

        # Create actuator with zone-specific tau values
        actuator = Actuator(tau=[5.0, 10.0, 15.0, 20.0], model="analytic")

        # Mock parameter expansion
        actuator.tau = torch.tensor([5.0, 10.0, 15.0, 20.0])

        setpoint = torch.ones(batch_size, n_zones)  # Step input
        position = torch.zeros(batch_size, n_zones)  # Starting from zero
        dt = 5.0  # One time constant for zone 0

        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Zone 0 (tau=5) should respond more than zone 3 (tau=20)
        # After dt=5s: zone0 responds ~63%, zone3 responds ~22%
        assert torch.all(result[:, 0] > result[:, 3])
        assert result.shape == (batch_size, n_zones)

    @pytest.mark.parametrize("model_type", ["instantaneous", "analytic", "smooth_approximation"])
    def test_all_actuator_models_vectorized(self, model_type, batch_zone_shapes):
        """Test all actuator models work with zone vectorization."""
        batch_size, n_zones = batch_zone_shapes['batch_size'], batch_zone_shapes['n_zones']

        actuator = Actuator(tau=10.0, model=model_type)
        actuator.tau = torch.full((n_zones,), 10.0)

        setpoint = torch.rand(batch_size, n_zones)
        position = torch.rand(batch_size, n_zones) if model_type != "instantaneous" else None

        result = actuator.forward(setpoint=setpoint, position=position, dt=1.0)

        assert result.shape == (batch_size, n_zones)
        if model_type == "instantaneous":
            torch.testing.assert_close(result, setpoint)

    def test_smooth_approximation_training_vs_inference(self, batch_zone_shapes):
        """Test smooth_approximation behaves differently in training vs inference."""
        batch_size, n_zones = batch_zone_shapes['batch_size'], batch_zone_shapes['n_zones']

        actuator = Actuator(tau=1.0, model="smooth_approximation")  # Small tau for large rates
        actuator.tau = torch.full((n_zones,), 1.0)

        setpoint = torch.ones(batch_size, n_zones)
        position = torch.zeros(batch_size, n_zones)
        dt = 0.5  # Medium rate = dt/tau = 0.5 (should trigger different behavior)

        # Training mode
        actuator.train()
        result_train = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Inference mode
        actuator.eval()
        result_eval = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Results should be different (training uses Pade, inference uses exact)
        assert not torch.allclose(result_train, result_eval, atol=1e-6)
        assert result_train.shape == result_eval.shape == (batch_size, n_zones)


class TestDamperVectorization:
    """Test suite for Damper zone vectorization and batching."""

    @pytest.fixture
    def damper_params(self):
        """Standard damper parameters for testing."""
        return {
            'max_airflow': [0.1, 0.2, 0.3, 0.4],  # Zone-specific
            'tau': 5.0,  # Shared
            'nominal_pressure': 500.0,  # Shared
            'flow_characteristic': 'sqrt'
        }

    def test_zone_specific_max_airflow(self, damper_params):
        """Test dampers with zone-specific maximum airflow."""
        damper = Damper(**damper_params, actuator_model="instantaneous")

        # Mock parameter expansion
        damper.max_airflow = torch.tensor([0.1, 0.2, 0.3, 0.4])
        damper.tau = torch.full((4,), 5.0)

        batch_size, n_zones = 2, 4
        target_airflow = torch.tensor([[0.05, 0.1, 0.15, 0.2],    # 50% of max for each zone
                                      [0.1, 0.2, 0.3, 0.4]])      # 100% of max for each zone
        current_position = torch.zeros(batch_size, n_zones)

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflow,
            current_position=current_position,
            dt=1.0
        )

        # Check output shapes
        assert result['airflow'].shape == (batch_size, n_zones)
        assert result['position'].shape == (batch_size, n_zones)

        # Check airflow matches target (instantaneous actuator)
        torch.testing.assert_close(result['airflow'], target_airflow, rtol=1e-4, atol=1e-6)

        # Check position setpoints are correct for sqrt characteristic
        expected_positions = (target_airflow / damper.max_airflow) ** 2  # Inverse of sqrt
        torch.testing.assert_close(result['position_setpoint'], expected_positions, rtol=1e-4, atol=1e-6)

    def test_pressure_correction_vectorized(self, damper_params):
        """Test pressure correction works with vectorized inputs."""
        damper = Damper(**damper_params, actuator_model="instantaneous")
        damper.max_airflow = torch.tensor([0.1, 0.2, 0.3, 0.4])
        damper.tau = torch.full((4,), 5.0)

        batch_size, n_zones = 3, 4
        target_airflow = torch.full((batch_size, n_zones), 0.1)
        current_position = torch.zeros(batch_size, n_zones)

        # Different pressures per batch
        duct_pressure = torch.tensor([[400.0], [500.0], [600.0]])  # Broadcast to all zones

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflow,
            current_position=current_position,
            duct_pressure=duct_pressure,
            dt=1.0
        )

        # Higher pressure should give higher airflow
        assert torch.all(result['airflow'][2] > result['airflow'][1])  # 600Pa > 500Pa
        assert torch.all(result['airflow'][1] > result['airflow'][0])  # 500Pa > 400Pa

    @pytest.mark.parametrize("flow_char", ["linear", "sqrt", "equal_percent"])
    def test_flow_characteristics_vectorized(self, flow_char):
        """Test all flow characteristics work with vectorization."""
        damper = Damper(
            max_airflow=[0.2, 0.2, 0.2],
            flow_characteristic=flow_char,
            actuator_model="instantaneous"
        )
        damper.max_airflow = torch.full((3,), 0.2)
        damper.tau = torch.full((3,), 5.0)

        # Test position -> airflow -> position round trip
        batch_size = 2
        original_positions = torch.tensor([[0.25, 0.5, 0.75],
                                          [0.1, 0.6, 0.9]])

        # Forward: position -> airflow
        airflows = damper.position_to_airflow(original_positions)

        # Inverse: airflow -> position
        recovered_positions = damper.airflow_to_position(airflows)

        # Should recover original positions (within numerical precision)
        torch.testing.assert_close(recovered_positions, original_positions, rtol=1e-5, atol=1e-7)


class TestElectricReheatCoilVectorization:
    """Test suite for ElectricReheatCoil zone vectorization and batching."""

    def test_zone_specific_capacity_and_efficiency(self):
        """Test electric reheat coil with zone-specific parameters."""
        coil = ElectricReheatCoil(
            max_thermal_output=[1000, 2000, 3000],  # Different capacity per zone
            electrical_efficiency=[0.95, 0.98, 1.0],  # Different efficiency per zone
            actuator_model="instantaneous"
        )

        # Mock parameter expansion
        coil.max_thermal_output = torch.tensor([1000.0, 2000.0, 3000.0])
        coil.electrical_efficiency = torch.tensor([0.95, 0.98, 1.0])
        coil.tau = torch.full((3,), 10.0)

        batch_size, n_zones = 2, 3
        target_heating = torch.tensor([[500.0, 1000.0, 1500.0],   # 50% of max for each zone
                                      [1000.0, 2000.0, 3000.0]])  # 100% of max for each zone
        current_position = torch.zeros(batch_size, n_zones)

        result = coil.forward(
            t=0.0,
            target_heating=target_heating,
            current_position=current_position,
            dt=1.0
        )

        # Check shapes
        assert result['thermal_output'].shape == (batch_size, n_zones)
        assert result['electrical_power'].shape == (batch_size, n_zones)

        # Check thermal output matches target (instantaneous actuator)
        torch.testing.assert_close(result['thermal_output'], target_heating, rtol=1e-4, atol=1e-6)

        # Check electrical power accounts for different efficiencies
        expected_electrical = target_heating / coil.electrical_efficiency
        torch.testing.assert_close(result['electrical_power'], expected_electrical, rtol=1e-4, atol=1e-6)

        # Zone 2 (100% efficient) should have electrical = thermal
        torch.testing.assert_close(
            result['electrical_power'][:, 2],
            result['thermal_output'][:, 2],
            rtol=1e-6, atol=1e-8
        )


class TestCombinedBatchingAndZoneVectorization:
    """Test complex scenarios combining batching and zone vectorization."""

    def test_large_batch_many_zones(self):
        """Test performance and correctness with large batch and zone counts."""
        batch_size, n_zones = 50, 20

        # Create damper with zone-specific parameters
        damper = Damper(actuator_model="analytic")
        damper.max_airflow = torch.linspace(0.1, 2.0, n_zones)  # Varying capacity
        damper.tau = torch.linspace(3.0, 15.0, n_zones)  # Varying response time

        # Random inputs
        torch.manual_seed(42)  # Reproducible
        target_airflow = torch.rand(batch_size, n_zones) * damper.max_airflow
        current_position = torch.rand(batch_size, n_zones)

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflow,
            current_position=current_position,
            dt=1.0
        )

        # Basic sanity checks
        assert result['airflow'].shape == (batch_size, n_zones)
        assert torch.all(torch.isfinite(result['airflow']))
        assert torch.all(result['airflow'] >= 0)
        assert torch.all(result['position'] >= 0) and torch.all(result['position'] <= 1)

    def test_broadcasting_edge_cases(self):
        """Test edge cases in broadcasting between batch and zone dimensions."""
        batch_size, n_zones = 3, 4

        damper = Damper(actuator_model="instantaneous")
        damper.max_airflow = torch.tensor([0.1, 0.2, 0.3, 0.4])
        damper.tau = torch.full((n_zones,), 5.0)

        # Test different broadcasting scenarios
        scenarios = [
            {
                'target_airflow': torch.full((batch_size, n_zones), 0.1),  # Same for all
                'duct_pressure': torch.tensor([[500.0], [600.0], [700.0]]),  # Different per batch
                'description': 'Batch-varying pressure, zone-constant airflow'
            },
            {
                'target_airflow': torch.tensor([[0.05, 0.1, 0.15, 0.2]]).expand(batch_size, -1),  # Zone-varying
                'duct_pressure': torch.tensor([[500.0, 500.0, 600.0, 600.0]]).expand(batch_size, -1),  # Zone-varying
                'description': 'Both batch and zone varying'
            }
        ]

        for scenario in scenarios:
            result = damper.forward(
                t=0.0,
                target_airflow=scenario['target_airflow'],
                current_position=torch.zeros(batch_size, n_zones),
                duct_pressure=scenario.get('duct_pressure'),
                dt=1.0
            )

            assert result['airflow'].shape == (batch_size, n_zones), \
                f"Failed for: {scenario['description']}"

    def test_basic_error_conditions(self):
        """Test basic error conditions that are already implemented."""
        actuator = Actuator(model="analytic")
        actuator.tau = torch.full((3,), 10.0)  # 3 zones

        # Missing required position for non-instantaneous model (this error IS implemented)
        with pytest.raises(ValueError, match="position required"):
            actuator.forward(
                setpoint=torch.rand(2, 3),
                position=None,  # Required but missing
                dt=1.0
            )


class TestParameterExpansionMocking:
    """Test parameter expansion behavior (mocked since BuildingComponent handles this)."""

    def test_scalar_to_vector_expansion_mock(self):
        """Test that scalar parameters can be expanded to zone vectors."""
        from torch_buildings.building_components.base import expand_parameter

        # Test scalar expansion
        result = expand_parameter(5.0, n_zones=4, name="tau")
        expected = torch.tensor([5.0, 5.0, 5.0, 5.0])
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-8)

        # Test list input
        result = expand_parameter([1.0, 2.0, 3.0], n_zones=3, name="tau")
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-8)

        # Test wrong list length
        with pytest.raises(ValueError, match="list length 2 != n_zones 3"):
            expand_parameter([1.0, 2.0], n_zones=3, name="tau")


if __name__ == "__main__":
    # Run specific test for development
    pytest.main([__file__ + "::TestActuatorVectorization::test_zone_specific_tau", "-v"])