"""
Comprehensive unit tests for the Damper class with zone vectorization support.

Tests cover:
- Initialization and parameter validation
- Flow characteristic curves (linear, sqrt, equal_percent)
- Pressure correction physics
- Actuator dynamics integration
- Zone vectorization and batch processing
- Edge cases and numerical stability
- Physical realism and bounds checking
"""

import pytest
import torch
import numpy as np

from torch_buildings import Damper


class TestDamperInitialization:
    """Test damper initialization and parameter handling with vectorization support."""

    def test_default_initialization(self):
        """Test damper with default parameters."""
        damper = Damper()

        assert damper.max_airflow == 1.0
        assert damper.flow_characteristic == "sqrt"
        assert damper.nominal_pressure == 500.0
        assert damper.tau == 5.0
        assert damper.name == "damper"

    def test_custom_initialization(self):
        """Test damper with custom parameters."""
        damper = Damper(
            max_airflow=2.5,
            flow_characteristic="linear",
            nominal_pressure=750.0,
            tau=8.0,
            actuator_model="analytic"
        )

        assert damper.max_airflow == 2.5
        assert damper.flow_characteristic == "linear"
        assert damper.nominal_pressure == 750.0
        assert damper.tau == 8.0

    def test_zone_specific_initialization(self):
        """Test damper with zone-specific parameters."""
        damper = Damper(
            max_airflow=[0.5, 1.0, 1.5],  # Zone-specific
            tau=[3.0, 5.0, 8.0],          # Zone-specific
            nominal_pressure=500.0,        # Shared
            flow_characteristic="sqrt"
        )

        # Parameters stored as provided (expansion happens in BuildingComponent)
        assert damper.max_airflow == [0.5, 1.0, 1.5]
        assert damper.tau == [3.0, 5.0, 8.0]
        assert damper.nominal_pressure == 500.0

    def test_invalid_flow_characteristic(self):
        """Test that invalid flow characteristics raise errors in methods."""
        damper = Damper()
        damper.flow_characteristic = "invalid"

        with pytest.raises(ValueError, match="Unknown flow characteristic"):
            damper.position_to_airflow(torch.tensor(0.5))

        with pytest.raises(ValueError, match="Unknown flow characteristic"):
            damper.airflow_to_position(torch.tensor(0.5))


class TestFlowCharacteristics:
    """Test the three flow characteristic curves with vectorization support."""

    @pytest.fixture
    def positions_scalar(self):
        """Standard test positions from 0 to 1 (scalar)."""
        return torch.linspace(0, 1, 11)  # [0, 0.1, 0.2, ..., 1.0]

    @pytest.fixture
    def positions_vectorized(self):
        """Standard test positions for vectorized testing."""
        return torch.linspace(0, 1, 11).unsqueeze(0).expand(2, -1)  # [2, 11] batch x positions

    def test_linear_characteristic_scalar(self, positions_scalar):
        """Test linear flow characteristic with scalar parameters."""
        damper = Damper(max_airflow=1.0, flow_characteristic="linear")
        # Mock expanded parameter
        damper.max_airflow = torch.tensor(1.0)

        # Forward: position → airflow
        calculated_airflows = damper.position_to_airflow(positions_scalar)
        expected_airflows = positions_scalar * damper.max_airflow
        torch.testing.assert_close(calculated_airflows, expected_airflows, rtol=1e-6, atol=1e-10)

        # Inverse: airflow → position
        airflows = torch.linspace(0, 1.0, 11)
        calculated_positions = damper.airflow_to_position(airflows)
        expected_positions = airflows / damper.max_airflow
        torch.testing.assert_close(calculated_positions, expected_positions, rtol=1e-6, atol=1e-10)

    def test_linear_characteristic_vectorized(self):
        """Test linear flow characteristic with zone vectorization."""
        damper = Damper(max_airflow=[0.5, 1.0, 1.5], flow_characteristic="linear")
        # Mock expanded parameters
        damper.max_airflow = torch.tensor([0.5, 1.0, 1.5])

        # Test with [batch_size, n_zones] input
        positions = torch.tensor([[0.0, 0.5, 1.0],
                                 [0.2, 0.6, 0.8]])  # 2 batches, 3 zones

        # Forward: position → airflow
        calculated_airflows = damper.position_to_airflow(positions)
        expected_airflows = positions * damper.max_airflow  # Broadcasting
        torch.testing.assert_close(calculated_airflows, expected_airflows, rtol=1e-6, atol=1e-10)

        # Inverse: airflow → position
        calculated_positions = damper.airflow_to_position(calculated_airflows)
        torch.testing.assert_close(calculated_positions, positions, rtol=1e-6, atol=1e-10)

    def test_sqrt_characteristic_vectorized(self):
        """Test square root flow characteristic with vectorization."""
        damper = Damper(max_airflow=[1.0, 2.0], flow_characteristic="sqrt")
        damper.max_airflow = torch.tensor([1.0, 2.0])

        positions = torch.tensor([[0.25, 0.64],
                                 [0.81, 0.16]])  # 2 batches, 2 zones

        # Forward: position → airflow
        calculated_airflows = damper.position_to_airflow(positions)
        expected_airflows = torch.sqrt(positions) * damper.max_airflow
        torch.testing.assert_close(calculated_airflows, expected_airflows, rtol=1e-6, atol=1e-10)

        # Test round-trip accuracy
        reconstructed_positions = damper.airflow_to_position(calculated_airflows)
        torch.testing.assert_close(reconstructed_positions, positions, rtol=1e-5, atol=1e-8)

    def test_equal_percent_characteristic_vectorized(self):
        """Test equal percentage flow characteristic with vectorization."""
        damper = Damper(max_airflow=[0.8, 1.2, 1.6], flow_characteristic="equal_percent")
        damper.max_airflow = torch.tensor([0.8, 1.2, 1.6])

        positions = torch.tensor([[0.125, 0.343, 0.729]])  # Cube roots: 0.5, 0.7, 0.9

        # Forward: position → airflow
        calculated_airflows = damper.position_to_airflow(positions)
        expected_airflows = (positions ** 3) * damper.max_airflow
        torch.testing.assert_close(calculated_airflows, expected_airflows, rtol=1e-6, atol=1e-10)

        # Test round-trip accuracy
        reconstructed_positions = damper.airflow_to_position(calculated_airflows)
        torch.testing.assert_close(reconstructed_positions, positions, rtol=1e-4, atol=1e-8)

    def test_flow_characteristic_monotonicity_vectorized(self):
        """Test that all flow characteristics are monotonically increasing with vectorization."""
        positions = torch.linspace(0, 1, 100).unsqueeze(0)  # [1, 100]

        for characteristic in ["linear", "sqrt", "equal_percent"]:
            damper = Damper(flow_characteristic=characteristic)
            damper.max_airflow = torch.tensor([1.0])  # Mock expanded parameter

            airflows = damper.position_to_airflow(positions)

            # Check monotonicity: each airflow should be >= previous
            diffs = airflows[:, 1:] - airflows[:, :-1]
            assert torch.all(diffs >= -1e-10), f"{characteristic} not monotonic"

    def test_flow_characteristic_bounds_vectorized(self):
        """Test that flow characteristics respect physical bounds with vectorization."""
        damper = Damper(max_airflow=[1.0, 2.0, 3.0])
        damper.max_airflow = torch.tensor([1.0, 2.0, 3.0])

        for characteristic in ["linear", "sqrt", "equal_percent"]:
            damper.flow_characteristic = characteristic

            # Position 0 should give airflow 0
            airflow_zero = damper.position_to_airflow(torch.zeros(1, 3))
            torch.testing.assert_close(airflow_zero, torch.zeros(1, 3), rtol=1e-10, atol=1e-10)

            # Position 1 should give max_airflow
            airflow_max = damper.position_to_airflow(torch.ones(1, 3))
            expected = damper.max_airflow.unsqueeze(0)
            torch.testing.assert_close(airflow_max, expected, rtol=1e-6, atol=1e-10)


class TestPressureCorrection:
    """Test pressure correction physics with vectorization support."""

    def test_no_pressure_correction_vectorized(self):
        """Test behavior when no duct pressure is provided."""
        damper = Damper(max_airflow=[1.0, 1.5], nominal_pressure=500.0)
        damper.max_airflow = torch.tensor([1.0, 1.5])
        damper.nominal_pressure = torch.tensor(500.0)

        position = torch.tensor([[0.5, 0.7]])

        # Should get base airflow without pressure correction
        airflow_no_pressure = damper.position_to_airflow(position)
        airflow_nominal_pressure = damper.position_to_airflow(position, torch.tensor([[500.0, 500.0]]))

        torch.testing.assert_close(airflow_no_pressure, airflow_nominal_pressure, rtol=1e-6, atol=1e-10)

    def test_pressure_correction_scaling_vectorized(self):
        """Test pressure correction follows Q ∝ √P relationship with vectorization."""
        damper = Damper(max_airflow=[1.0, 2.0], nominal_pressure=500.0)
        damper.max_airflow = torch.tensor([1.0, 2.0])
        damper.nominal_pressure = torch.tensor(500.0)

        position = torch.tensor([[1.0, 1.0]])  # Full open

        # Test with different pressures per zone
        pressures = torch.tensor([[250.0, 1000.0]])  # 0.5x and 2x nominal
        expected_factors = torch.sqrt(pressures / 500.0)

        base_airflow = damper.position_to_airflow(position)
        corrected_airflow = damper.position_to_airflow(position, pressures)

        expected_airflow = base_airflow * expected_factors.clamp(0.5, 2.0)
        torch.testing.assert_close(corrected_airflow, expected_airflow, rtol=1e-6, atol=1e-10)

    def test_pressure_correction_clamping_vectorized(self):
        """Test pressure correction clamping with vectorization."""
        damper = Damper(max_airflow=[1.0, 1.0])
        damper.max_airflow = torch.tensor([1.0, 1.0])
        damper.nominal_pressure = torch.tensor(500.0)

        position = torch.tensor([[1.0, 1.0]])

        # Very low and very high pressures
        extreme_pressures = torch.tensor([[50.0, 5000.0]])  # 10x lower, 10x higher
        airflows = damper.position_to_airflow(position, extreme_pressures)

        base_airflow = damper.position_to_airflow(position)
        expected = base_airflow * torch.tensor([[0.5, 2.0]])  # Clamped factors
        torch.testing.assert_close(airflows, expected, rtol=1e-6, atol=1e-10)

    def test_pressure_correction_broadcasting(self):
        """Test pressure correction with different broadcasting scenarios."""
        damper = Damper(max_airflow=[1.0, 1.5, 2.0])
        damper.max_airflow = torch.tensor([1.0, 1.5, 2.0])
        damper.nominal_pressure = torch.tensor(500.0)

        position = torch.tensor([[0.5, 0.7, 0.9],
                                [0.3, 0.6, 0.8]])  # 2 batches, 3 zones

        # Scenario 1: Single pressure for all zones
        pressure_single = torch.tensor([[800.0]])  # Broadcasts to all zones
        airflow1 = damper.position_to_airflow(position, pressure_single)
        assert airflow1.shape == (2, 3)

        # Scenario 2: Different pressure per zone
        pressure_zones = torch.tensor([[400.0, 500.0, 600.0]])  # Different per zone
        airflow2 = damper.position_to_airflow(position, pressure_zones)
        assert airflow2.shape == (2, 3)

        # All results should be finite and positive
        assert torch.all(torch.isfinite(airflow1))
        assert torch.all(torch.isfinite(airflow2))
        assert torch.all(airflow1 >= 0)
        assert torch.all(airflow2 >= 0)


class TestActuatorIntegration:
    """Test integration with actuator dynamics with vectorization support."""

    def test_instantaneous_actuator_vectorized(self):
        """Test with instantaneous actuator and zone vectorization."""
        damper = Damper(actuator_model="instantaneous", max_airflow=[1.0, 2.0])
        damper.max_airflow = torch.tensor([1.0, 2.0])
        damper.tau = torch.tensor([5.0, 5.0])  # Not used by instantaneous

        target_airflow = torch.tensor([[0.6, 1.2]])
        current_position = torch.tensor([[0.2, 0.3]])

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflow,
            current_position=current_position,
            dt=1.0
        )

        # With instantaneous actuator, position should equal setpoint immediately
        expected_position = damper.airflow_to_position(target_airflow)
        torch.testing.assert_close(result["position"], expected_position, rtol=1e-6, atol=1e-10)
        torch.testing.assert_close(result["position_setpoint"], expected_position, rtol=1e-6, atol=1e-10)

    def test_dynamic_actuator_lag_vectorized(self):
        """Test that dynamic actuator shows lag behavior with vectorization."""
        damper = Damper(actuator_model="analytic", tau=[5.0, 15.0], max_airflow=[1.0, 1.0])
        # Mock expanded parameters
        damper.max_airflow = torch.tensor([1.0, 1.0])
        damper.tau = torch.tensor([5.0, 15.0])

        # Step input from 0 to full airflow for both zones
        target_airflow = torch.tensor([[1.0, 1.0]])
        current_position = torch.tensor([[0.0, 0.0]])

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflow,
            current_position=current_position,
            dt=1.0
        )

        # Both zones should move toward setpoint but not reach it immediately
        setpoint = result["position_setpoint"]
        position = result["position"]

        assert torch.all(position > current_position)  # Should move toward setpoint
        assert torch.all(position < setpoint)  # But not reach it due to lag

        # Zone 0 (faster tau) should respond more than zone 1 (slower tau)
        assert position[0, 0] > position[0, 1], "Faster zone should respond more"

    def test_forward_complete_workflow_vectorized(self):
        """Test complete forward pass workflow with vectorization."""
        damper = Damper(max_airflow=[1.5, 2.0, 2.5], flow_characteristic="sqrt")
        # Mock expanded parameters
        damper.max_airflow = torch.tensor([1.5, 2.0, 2.5])
        damper.tau = torch.tensor([5.0, 5.0, 5.0])

        result = damper.forward(
            t=5.0,
            target_airflow=torch.tensor([[0.8, 1.0, 1.2],
                                        [1.0, 1.5, 2.0]]),
            current_position=torch.tensor([[0.3, 0.4, 0.5],
                                          [0.2, 0.3, 0.4]]),
            dt=1.0,
            duct_pressure=torch.tensor([[600.0, 500.0, 400.0]])  # Different per zone
        )

        # Check all expected outputs are present
        assert "position" in result
        assert "airflow" in result
        assert "position_setpoint" in result

        # Check output shapes
        assert result["position"].shape == (2, 3)
        assert result["airflow"].shape == (2, 3)
        assert result["position_setpoint"].shape == (2, 3)

        # Check output ranges
        assert torch.all((result["position"] >= 0.0) & (result["position"] <= 1.0))
        assert torch.all(result["airflow"] >= 0.0)
        assert torch.all((result["position_setpoint"] >= 0.0) & (result["position_setpoint"] <= 1.0))


class TestEdgeCasesAndNumericalStability:
    """Test edge cases and numerical stability with vectorization support."""

    def test_zero_values_vectorized(self):
        """Test behavior at zero values with vectorization."""
        damper = Damper(max_airflow=[1.0, 2.0])
        damper.max_airflow = torch.tensor([1.0, 2.0])

        for characteristic in ["linear", "sqrt", "equal_percent"]:
            damper.flow_characteristic = characteristic

            # Zero position
            airflow = damper.position_to_airflow(torch.zeros(1, 2))
            torch.testing.assert_close(airflow, torch.zeros(1, 2), rtol=1e-10, atol=1e-10)

            # Zero airflow
            position = damper.airflow_to_position(torch.zeros(1, 2))
            torch.testing.assert_close(position, torch.zeros(1, 2), rtol=1e-10, atol=1e-10)

    def test_position_clamping_vectorized(self):
        """Test that positions are clamped to [0, 1] range with vectorization."""
        damper = Damper(max_airflow=[1.0, 1.5])
        damper.max_airflow = torch.tensor([1.0, 1.5])

        # Test airflow_to_position clamping
        excessive_airflow = torch.tensor([[5.0, 8.0]])  # Much larger than max_airflow
        position = damper.airflow_to_position(excessive_airflow)
        assert torch.all(position <= 1.0)

        # Test position_to_airflow clamping
        positions = torch.tensor([[-0.5, 1.5]])  # Outside [0, 1] range
        airflows = damper.position_to_airflow(positions)
        assert torch.all(airflows >= 0.0)

    def test_large_batch_zone_operations(self):
        """Test damper with large batch and zone dimensions."""
        batch_size, n_zones = 10, 5
        damper = Damper()
        # Mock expanded parameters
        damper.max_airflow = torch.linspace(0.5, 2.5, n_zones)
        damper.tau = torch.full((n_zones,), 5.0)

        # Test large batch operations
        positions = torch.rand(batch_size, n_zones)
        airflows = damper.position_to_airflow(positions)
        assert airflows.shape == (batch_size, n_zones)

        target_airflows = torch.rand(batch_size, n_zones) * damper.max_airflow
        result = damper.forward(
            t=0.0,
            target_airflow=target_airflows,
            current_position=positions,
            dt=1.0
        )

        # All outputs should have correct shape
        assert result["position"].shape == (batch_size, n_zones)
        assert result["airflow"].shape == (batch_size, n_zones)

    def test_equal_percent_cube_root_stability_vectorized(self):
        """Test numerical stability of cube root with vectorization."""
        damper = Damper(flow_characteristic="equal_percent", max_airflow=[1.0, 1.5])
        damper.max_airflow = torch.tensor([1.0, 1.5])

        # Test very small airflows
        small_airflows = torch.tensor([[1e-10, 1e-8],
                                      [1e-6, 1e-4]])
        positions = damper.airflow_to_position(small_airflows)

        # Should not produce NaN or infinite values
        assert torch.all(torch.isfinite(positions))
        assert torch.all(positions >= 0.0)

        # Test round-trip accuracy
        reconstructed_airflows = damper.position_to_airflow(positions)
        torch.testing.assert_close(reconstructed_airflows, small_airflows, rtol=1e-4, atol=1e-10)


class TestPhysicalRealism:
    """Test that outputs are physically realistic with vectorization support."""

    def test_realistic_parameter_ranges_vectorized(self):
        """Test damper with realistic HVAC parameter ranges and zone vectorization."""
        # Different zone types: small VAV, medium VAV, large supply
        damper = Damper(
            max_airflow=[0.1, 0.5, 2.5],        # Different capacity per zone
            nominal_pressure=500.0,               # Shared
            tau=[5.0, 8.0, 12.0],                # Different response times
            flow_characteristic="sqrt"
        )

        # Mock expanded parameters
        damper.max_airflow = torch.tensor([0.1, 0.5, 2.5])
        damper.tau = torch.tensor([5.0, 8.0, 12.0])

        target_airflows = damper.max_airflow * 0.7  # 70% of max for each zone
        current_positions = torch.tensor([[0.5, 0.5, 0.5]])

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflows.unsqueeze(0),
            current_position=current_positions,
            dt=2.0
        )

        # All zones should produce realistic outputs
        assert torch.all((result["position"] >= 0.0) & (result["position"] <= 1.0))
        assert torch.all(result["airflow"] >= 0.0)

        # Each zone should respect its max airflow limit (with small tolerance for pressure effects)
        max_expected = damper.max_airflow * 1.1
        assert torch.all(result["airflow"] <= max_expected)

    def test_actuator_time_constants_vectorized(self):
        """Test realistic actuator response times with vectorization."""
        damper = Damper(tau=[3.0, 15.0], max_airflow=[1.0, 1.0])  # Fast vs slow
        damper.max_airflow = torch.tensor([1.0, 1.0])
        damper.tau = torch.tensor([3.0, 15.0])

        target_airflow = torch.tensor([[0.8, 0.8]])
        initial_position = torch.tensor([[0.2, 0.2]])

        result = damper.forward(
            t=0.0, target_airflow=target_airflow,
            current_position=initial_position, dt=1.0
        )

        # Fast actuator (zone 0) should respond quicker than slow actuator (zone 1)
        movements = result["position"] - initial_position
        assert movements[0, 0] > movements[0, 1], "Fast actuator should respond quicker"

    def test_zone_independence(self):
        """Test that zones operate independently with different parameters."""
        damper = Damper(
            max_airflow=[0.5, 1.0, 1.5],
            tau=[3.0, 6.0, 9.0],
            flow_characteristic="sqrt"
        )

        # Mock expanded parameters
        damper.max_airflow = torch.tensor([0.5, 1.0, 1.5])
        damper.tau = torch.tensor([3.0, 6.0, 9.0])

        # Same relative target (50% of max) for all zones
        target_airflows = damper.max_airflow * 0.5
        current_positions = torch.zeros(1, 3)

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflows.unsqueeze(0),
            current_position=current_positions,
            dt=1.0
        )

        # All zones should aim for same relative position setpoint (sqrt characteristic)
        expected_setpoint = 0.5 ** 2  # For sqrt: position = (airflow/max)^2
        torch.testing.assert_close(
            result["position_setpoint"],
            torch.full((1, 3), expected_setpoint),
            rtol=1e-5, atol=1e-8
        )

        # But actual positions should differ due to different time constants
        positions = result["position"][0]
        assert positions[0] > positions[1] > positions[2], "Faster zones should respond more"


class TestRoundTripAccuracy:
    """Test round-trip accuracy of airflow ↔ position conversions with vectorization."""

    @pytest.mark.parametrize("characteristic", ["linear", "sqrt", "equal_percent"])
    def test_position_airflow_roundtrip_vectorized(self, characteristic):
        """Test position → airflow → position round-trip with vectorization."""
        damper = Damper(flow_characteristic=characteristic, max_airflow=[1.0, 1.5, 2.0])
        damper.max_airflow = torch.tensor([1.0, 1.5, 2.0])

        original_positions = torch.tensor([[0.1, 0.3, 0.5],
                                          [0.2, 0.6, 0.9],
                                          [0.05, 0.4, 0.8]])  # 3 batches, 3 zones

        # Forward: position → airflow
        airflows = damper.position_to_airflow(original_positions)

        # Inverse: airflow → position
        reconstructed_positions = damper.airflow_to_position(airflows)

        # Should recover original positions (within numerical precision)
        torch.testing.assert_close(reconstructed_positions, original_positions, rtol=1e-5, atol=1e-9)

    @pytest.mark.parametrize("characteristic", ["linear", "sqrt", "equal_percent"])
    def test_airflow_position_roundtrip_vectorized(self, characteristic):
        """Test airflow → position → airflow round-trip with vectorization."""
        damper = Damper(flow_characteristic=characteristic, max_airflow=[0.8, 1.2, 1.6])
        damper.max_airflow = torch.tensor([0.8, 1.2, 1.6])

        original_airflows = torch.tensor([[0.1, 0.3, 0.5],
                                         [0.2, 0.8, 1.2]])  # 2 batches, 3 zones

        # Forward: airflow → position
        positions = damper.airflow_to_position(original_airflows)

        # Inverse: position → airflow
        reconstructed_airflows = damper.position_to_airflow(positions)

        # Should recover original airflows (within numerical precision)
        torch.testing.assert_close(reconstructed_airflows, original_airflows, rtol=1e-5, atol=1e-9)


"""
Batch + Zone vectorization tests for Damper and ElectricReheatCoil components.
These tests isolate potential tensor shape issues when batch_size > 1 and n_zones > 1.

Add these to test_damper.py and test_electric_reheat_coil.py respectively.
"""


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])