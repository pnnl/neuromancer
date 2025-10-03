"""
Comprehensive unit tests for the ElectricReheatCoil class with zone vectorization support.

Tests cover:
- Initialization and parameter validation
- Heating characteristic curves (linear, equal_percent, quick_opening)
- Electrical efficiency and power consumption
- Actuator dynamics integration
- Zone vectorization and batch processing
- Edge cases and numerical stability
- Physical realism and bounds checking
- Energy conservation principles
"""

import pytest
import torch
import numpy as np

from torch_buildings import ElectricReheatCoil


class TestElectricReheatCoilInitialization:
    """Test electric reheat coil initialization and parameter handling with vectorization support."""

    def test_default_initialization(self):
        """Test reheat coil with default parameters."""
        coil = ElectricReheatCoil()

        assert coil.max_thermal_output == 3000.0
        assert coil.heating_characteristic == "linear"
        assert coil.electrical_efficiency == 0.98
        assert coil.tau == 10.0
        assert coil.name == "electric_reheat_coil"

    def test_custom_initialization(self):
        """Test reheat coil with custom parameters."""
        coil = ElectricReheatCoil(
            max_thermal_output=5000.0,
            heating_characteristic="equal_percent",
            electrical_efficiency=0.95,
            tau=15.0,
            actuator_model="analytic"
        )

        assert coil.max_thermal_output == 5000.0
        assert coil.heating_characteristic == "equal_percent"
        assert coil.electrical_efficiency == 0.95
        assert coil.tau == 15.0

    def test_zone_specific_initialization(self):
        """Test reheat coil with zone-specific parameters."""
        coil = ElectricReheatCoil(
            max_thermal_output=[1500.0, 3000.0, 4500.0],  # Zone-specific
            electrical_efficiency=[0.95, 0.98, 1.0],      # Zone-specific
            tau=[8.0, 10.0, 12.0],                        # Zone-specific
            heating_characteristic="linear"                # Shared
        )

        # Parameters stored as provided (expansion happens in BuildingComponent)
        assert coil.max_thermal_output == [1500.0, 3000.0, 4500.0]
        assert coil.electrical_efficiency == [0.95, 0.98, 1.0]
        assert coil.tau == [8.0, 10.0, 12.0]
        assert coil.heating_characteristic == "linear"

    def test_invalid_heating_characteristic(self):
        """Test that invalid heating characteristics raise errors in methods."""
        coil = ElectricReheatCoil()
        coil.heating_characteristic = "invalid"

        with pytest.raises(ValueError, match="Unknown heating characteristic"):
            coil.position_to_thermal_output(torch.tensor(0.5))

        with pytest.raises(ValueError, match="Unknown heating characteristic"):
            coil.target_heating_to_position(torch.tensor(1500.0))


class TestHeatingCharacteristics:
    """Test the three heating characteristic curves with vectorization support."""

    @pytest.fixture
    def positions_scalar(self):
        """Standard test positions from 0 to 1 (scalar)."""
        return torch.linspace(0, 1, 11)  # [0, 0.1, 0.2, ..., 1.0]

    def test_linear_characteristic_scalar(self, positions_scalar):
        """Test linear heating characteristic with scalar parameters."""
        coil = ElectricReheatCoil(max_thermal_output=3000.0, heating_characteristic="linear")
        # Mock expanded parameter
        coil.max_thermal_output = torch.tensor(3000.0)

        # Forward: position → thermal output
        calculated_outputs = coil.position_to_thermal_output(positions_scalar)
        expected_outputs = positions_scalar * coil.max_thermal_output
        torch.testing.assert_close(calculated_outputs, expected_outputs, rtol=1e-6, atol=1e-10)

        # Inverse: thermal output → position
        thermal_outputs = torch.linspace(0, 3000.0, 11)
        calculated_positions = coil.target_heating_to_position(thermal_outputs)
        expected_positions = thermal_outputs / coil.max_thermal_output
        torch.testing.assert_close(calculated_positions, expected_positions, rtol=1e-6, atol=1e-10)

    def test_linear_characteristic_vectorized(self):
        """Test linear heating characteristic with zone vectorization."""
        coil = ElectricReheatCoil(max_thermal_output=[2000.0, 3000.0, 4000.0], heating_characteristic="linear")
        # Mock expanded parameters
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0, 4000.0])

        # Test with [batch_size, n_zones] input
        positions = torch.tensor([[0.0, 0.5, 1.0],
                                 [0.25, 0.75, 0.5]])  # 2 batches, 3 zones

        # Forward: position → thermal output
        calculated_outputs = coil.position_to_thermal_output(positions)
        expected_outputs = positions * coil.max_thermal_output  # Broadcasting
        torch.testing.assert_close(calculated_outputs, expected_outputs, rtol=1e-6, atol=1e-10)

        # Inverse: thermal output → position
        calculated_positions = coil.target_heating_to_position(calculated_outputs)
        torch.testing.assert_close(calculated_positions, positions, rtol=1e-6, atol=1e-10)

    def test_equal_percent_characteristic_vectorized(self):
        """Test equal percentage heating characteristic with vectorization."""
        coil = ElectricReheatCoil(max_thermal_output=[2000.0, 3000.0], heating_characteristic="equal_percent")
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0])

        # Use cube values for easy testing: 0.125 = 0.5³, 0.343 = 0.7³
        positions = torch.tensor([[0.125, 0.343],
                                 [0.729, 0.216]])  # 2 batches, 2 zones

        # Forward: position → thermal output
        calculated_outputs = coil.position_to_thermal_output(positions)
        expected_outputs = (positions ** 3) * coil.max_thermal_output
        torch.testing.assert_close(calculated_outputs, expected_outputs, rtol=1e-6, atol=1e-10)

        # Test round-trip accuracy
        reconstructed_positions = coil.target_heating_to_position(calculated_outputs)
        torch.testing.assert_close(reconstructed_positions, positions, rtol=1e-4, atol=1e-8)

    def test_quick_opening_characteristic_vectorized(self):
        """Test quick opening heating characteristic with vectorization."""
        coil = ElectricReheatCoil(max_thermal_output=[1500.0, 2500.0, 3500.0], heating_characteristic="quick_opening")
        coil.max_thermal_output = torch.tensor([1500.0, 2500.0, 3500.0])

        # Use square values for easy testing: 0.25 = 0.5², 0.64 = 0.8²
        positions = torch.tensor([[0.25, 0.64, 0.81]])  # sqrt gives 0.5, 0.8, 0.9

        # Forward: position → thermal output
        calculated_outputs = coil.position_to_thermal_output(positions)
        expected_outputs = torch.sqrt(positions) * coil.max_thermal_output
        torch.testing.assert_close(calculated_outputs, expected_outputs, rtol=1e-6, atol=1e-10)

        # Test round-trip accuracy
        reconstructed_positions = coil.target_heating_to_position(calculated_outputs)
        torch.testing.assert_close(reconstructed_positions, positions, rtol=1e-5, atol=1e-8)

    def test_heating_characteristic_monotonicity_vectorized(self):
        """Test that all heating characteristics are monotonically increasing with vectorization."""
        positions = torch.linspace(0, 1, 100).unsqueeze(0)  # [1, 100]

        for characteristic in ["linear", "equal_percent", "quick_opening"]:
            coil = ElectricReheatCoil(heating_characteristic=characteristic)
            coil.max_thermal_output = torch.tensor([3000.0])  # Mock expanded parameter

            thermal_outputs = coil.position_to_thermal_output(positions)

            # Check monotonicity: each output should be >= previous
            diffs = thermal_outputs[:, 1:] - thermal_outputs[:, :-1]
            assert torch.all(diffs >= -1e-10), f"{characteristic} not monotonic"

    def test_heating_characteristic_bounds_vectorized(self):
        """Test that heating characteristics respect physical bounds with vectorization."""
        coil = ElectricReheatCoil(max_thermal_output=[2000.0, 3000.0, 4000.0])
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0, 4000.0])

        for characteristic in ["linear", "equal_percent", "quick_opening"]:
            coil.heating_characteristic = characteristic

            # Position 0 should give thermal output 0
            output_zero = coil.position_to_thermal_output(torch.zeros(1, 3))
            torch.testing.assert_close(output_zero, torch.zeros(1, 3), rtol=1e-10, atol=1e-10)

            # Position 1 should give max_thermal_output
            output_max = coil.position_to_thermal_output(torch.ones(1, 3))
            expected = coil.max_thermal_output.unsqueeze(0)
            torch.testing.assert_close(output_max, expected, rtol=1e-6, atol=1e-10)


class TestElectricalEfficiencyAndPower:
    """Test electrical efficiency and power consumption calculations with vectorization support."""

    def test_electrical_power_calculation_vectorized(self):
        """Test electrical power calculation with zone-specific efficiencies."""
        coil = ElectricReheatCoil(
            max_thermal_output=[2000.0, 3000.0],
            electrical_efficiency=[0.95, 0.98]
        )
        # Mock expanded parameters
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0])
        coil.electrical_efficiency = torch.tensor([0.95, 0.98])

        # Test various positions
        positions = torch.tensor([[0.25, 0.5],
                                 [0.75, 1.0]])  # 2 batches, 2 zones

        thermal_outputs = coil.position_to_thermal_output(positions)
        electrical_powers = coil.position_to_electrical_power(positions)

        # Verify electrical power = thermal output / efficiency for each zone
        expected_powers = thermal_outputs / coil.electrical_efficiency
        torch.testing.assert_close(electrical_powers, expected_powers, rtol=1e-6, atol=1e-10)

    def test_efficiency_values_vectorized(self):
        """Test realistic electrical efficiency values with vectorization."""
        # Different efficiency types per zone
        coil = ElectricReheatCoil(
            electrical_efficiency=[0.99, 0.95, 0.98],  # High, low, medium efficiency
            max_thermal_output=[2000.0, 2000.0, 2000.0]
        )
        coil.max_thermal_output = torch.tensor([2000.0, 2000.0, 2000.0])
        coil.electrical_efficiency = torch.tensor([0.99, 0.95, 0.98])

        # Full power for all zones
        thermal_out = coil.position_to_thermal_output(torch.ones(1, 3))
        electrical_in = coil.position_to_electrical_power(torch.ones(1, 3))

        # Power input should be higher than thermal output for all zones
        assert torch.all(electrical_in > thermal_out)

        # Verify efficiency calculations per zone
        efficiency_check = thermal_out / electrical_in
        expected_efficiency = coil.electrical_efficiency.unsqueeze(0)
        torch.testing.assert_close(efficiency_check, expected_efficiency, rtol=1e-6, atol=1e-10)

        # Zone 1 (lowest efficiency) should require most electrical power
        assert electrical_in[0, 1] > electrical_in[0, 0]  # 0.95 > 0.99 efficiency
        assert electrical_in[0, 1] > electrical_in[0, 2]  # 0.95 > 0.98 efficiency

    def test_power_consumption_significance_vectorized(self):
        """Test electrical power consumption across different zone sizes."""
        # Different capacity zones: small, medium, large VAV boxes
        coil = ElectricReheatCoil(
            max_thermal_output=[1500.0, 3000.0, 6000.0],
            electrical_efficiency=[0.98, 0.97, 0.96]
        )
        coil.max_thermal_output = torch.tensor([1500.0, 3000.0, 6000.0])
        coil.electrical_efficiency = torch.tensor([0.98, 0.97, 0.96])

        # At full power
        max_electrical_powers = coil.position_to_electrical_power(torch.ones(1, 3))
        expected_powers = coil.max_thermal_output / coil.electrical_efficiency

        torch.testing.assert_close(max_electrical_powers[0], expected_powers, rtol=1e-6, atol=1e-10)

        # Verify all are in significant ranges (kW)
        assert torch.all(max_electrical_powers > 1000.0)  # > 1 kW
        assert max_electrical_powers[0, 2] > 6000.0  # Large zone > 6 kW


class TestActuatorIntegration:
    """Test integration with actuator dynamics with vectorization support."""

    def test_instantaneous_actuator_vectorized(self):
        """Test with instantaneous actuator and zone vectorization."""
        coil = ElectricReheatCoil(
            actuator_model="instantaneous",
            max_thermal_output=[2000.0, 3000.0]
        )
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0])
        coil.tau = torch.tensor([10.0, 10.0])  # Not used by instantaneous

        target_heating = torch.tensor([[1200.0, 1800.0]])  # 60% of max for each zone
        current_position = torch.tensor([[0.1, 0.2]])

        result = coil.forward(
            t=0.0,
            target_heating=target_heating,
            current_position=current_position,
            dt=1.0
        )

        # With instantaneous actuator, position should equal setpoint immediately
        expected_position = coil.target_heating_to_position(target_heating)
        torch.testing.assert_close(result["position"], expected_position, rtol=1e-6, atol=1e-10)
        torch.testing.assert_close(result["position_setpoint"], expected_position, rtol=1e-6, atol=1e-10)

    def test_dynamic_actuator_lag_vectorized(self):
        """Test that dynamic actuator shows lag behavior with vectorization."""
        coil = ElectricReheatCoil(
            actuator_model="analytic",
            tau=[8.0, 20.0],  # Fast vs slow actuators
            max_thermal_output=[3000.0, 3000.0]
        )
        # Mock expanded parameters
        coil.max_thermal_output = torch.tensor([3000.0, 3000.0])
        coil.tau = torch.tensor([8.0, 20.0])

        # Step input from 0 to full heating for both zones
        target_heating = torch.tensor([[3000.0, 3000.0]])
        current_position = torch.tensor([[0.0, 0.0]])

        result = coil.forward(
            t=0.0,
            target_heating=target_heating,
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
        coil = ElectricReheatCoil(
            max_thermal_output=[2000.0, 3000.0, 4000.0],
            heating_characteristic="equal_percent",
            electrical_efficiency=[0.95, 0.98, 1.0]
        )
        # Mock expanded parameters
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0, 4000.0])
        coil.electrical_efficiency = torch.tensor([0.95, 0.98, 1.0])
        coil.tau = torch.tensor([10.0, 10.0, 10.0])

        result = coil.forward(
            t=5.0,
            target_heating=torch.tensor([[1000.0, 1500.0, 2000.0],
                                        [1500.0, 2000.0, 3000.0]]),
            current_position=torch.tensor([[0.3, 0.4, 0.5],
                                          [0.2, 0.3, 0.4]]),
            dt=1.0
        )

        # Check all expected outputs are present
        assert "position" in result
        assert "thermal_output" in result
        assert "electrical_power" in result
        assert "position_setpoint" in result

        # Check output shapes
        assert result["position"].shape == (2, 3)
        assert result["thermal_output"].shape == (2, 3)
        assert result["electrical_power"].shape == (2, 3)
        assert result["position_setpoint"].shape == (2, 3)

        # Check output ranges
        assert torch.all((result["position"] >= 0.0) & (result["position"] <= 1.0))
        assert torch.all(result["thermal_output"] >= 0.0)
        assert torch.all(result["electrical_power"] >= result["thermal_output"])  # Power >= thermal
        assert torch.all((result["position_setpoint"] >= 0.0) & (result["position_setpoint"] <= 1.0))


class TestEdgeCasesAndNumericalStability:
    """Test edge cases and numerical stability with vectorization support."""

    def test_zero_values_vectorized(self):
        """Test behavior at zero values with vectorization."""
        coil = ElectricReheatCoil(max_thermal_output=[2000.0, 3000.0])
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0])
        coil.electrical_efficiency = torch.tensor([0.98, 0.95])

        for characteristic in ["linear", "equal_percent", "quick_opening"]:
            coil.heating_characteristic = characteristic

            # Zero position
            thermal_output = coil.position_to_thermal_output(torch.zeros(1, 2))
            electrical_power = coil.position_to_electrical_power(torch.zeros(1, 2))

            torch.testing.assert_close(thermal_output, torch.zeros(1, 2), rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(electrical_power, torch.zeros(1, 2), rtol=1e-10, atol=1e-10)

            # Zero target heating
            position = coil.target_heating_to_position(torch.zeros(1, 2))
            torch.testing.assert_close(position, torch.zeros(1, 2), rtol=1e-10, atol=1e-10)

    def test_position_clamping_vectorized(self):
        """Test that positions are clamped to [0, 1] range with vectorization."""
        coil = ElectricReheatCoil(max_thermal_output=[2000.0, 3000.0])
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0])

        # Test target_heating_to_position clamping
        excessive_heating = torch.tensor([[10000.0, 15000.0]])  # Much larger than max
        position = coil.target_heating_to_position(excessive_heating)
        assert torch.all(position <= 1.0)

        # Test position_to_thermal_output clamping
        positions = torch.tensor([[-0.5, 1.5]])  # Outside [0, 1] range
        thermal_outputs = coil.position_to_thermal_output(positions)
        assert torch.all(thermal_outputs >= 0.0)

    def test_large_batch_zone_operations(self):
        """Test coil with large batch and zone dimensions."""
        batch_size, n_zones = 8, 6
        coil = ElectricReheatCoil()
        # Mock expanded parameters
        coil.max_thermal_output = torch.linspace(1500.0, 4500.0, n_zones)
        coil.electrical_efficiency = torch.linspace(0.95, 0.99, n_zones)
        coil.tau = torch.full((n_zones,), 10.0)

        # Test large batch operations
        positions = torch.rand(batch_size, n_zones)
        thermal_outputs = coil.position_to_thermal_output(positions)
        electrical_powers = coil.position_to_electrical_power(positions)

        assert thermal_outputs.shape == (batch_size, n_zones)
        assert electrical_powers.shape == (batch_size, n_zones)

        target_heatings = torch.rand(batch_size, n_zones) * coil.max_thermal_output
        result = coil.forward(
            t=0.0,
            target_heating=target_heatings,
            current_position=positions,
            dt=1.0
        )

        # All outputs should have correct shape
        assert result["position"].shape == (batch_size, n_zones)
        assert result["thermal_output"].shape == (batch_size, n_zones)
        assert result["electrical_power"].shape == (batch_size, n_zones)

    def test_equal_percent_cube_root_stability_vectorized(self):
        """Test numerical stability of cube root with vectorization."""
        coil = ElectricReheatCoil(
            heating_characteristic="equal_percent",
            max_thermal_output=[2000.0, 3000.0]
        )
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0])

        # Test very small thermal outputs
        small_outputs = torch.tensor([[1e-10, 1e-8],
                                     [1e-6, 1e-4]])
        positions = coil.target_heating_to_position(small_outputs)

        # Should not produce NaN or infinite values
        assert torch.all(torch.isfinite(positions))
        assert torch.all(positions >= 0.0)

        # Test round-trip accuracy
        reconstructed_outputs = coil.position_to_thermal_output(positions)
        torch.testing.assert_close(reconstructed_outputs, small_outputs, rtol=1e-4, atol=1e-10)


class TestPhysicalRealism:
    """Test that outputs are physically realistic with vectorization support."""

    def test_energy_conservation_vectorized(self):
        """Test energy conservation across multiple zones with different efficiencies."""
        coil = ElectricReheatCoil(
            max_thermal_output=[2000.0, 3000.0, 4000.0],
            electrical_efficiency=[0.94, 0.97, 1.0]
        )
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0, 4000.0])
        coil.electrical_efficiency = torch.tensor([0.94, 0.97, 1.0])

        positions = torch.linspace(0, 1, 20).unsqueeze(1).expand(-1, 3)  # [20, 3]

        thermal_outputs = coil.position_to_thermal_output(positions)
        electrical_powers = coil.position_to_electrical_power(positions)

        # Electrical power should always be >= thermal output (due to losses)
        assert torch.all(electrical_powers >= thermal_outputs - 1e-10)

        # Efficiency check for non-zero values
        non_zero_mask = electrical_powers > 1e-10
        calculated_efficiencies = torch.where(
            non_zero_mask,
            thermal_outputs / electrical_powers,
            torch.zeros_like(thermal_outputs)
        )

        expected_efficiencies = coil.electrical_efficiency.unsqueeze(0).expand(20, -1)
        torch.testing.assert_close(
            calculated_efficiencies[non_zero_mask],
            expected_efficiencies[non_zero_mask],
            rtol=1e-6, atol=1e-10
        )

    def test_realistic_parameter_ranges_vectorized(self):
        """Test electric reheat coil with realistic parameter ranges and zone vectorization."""
        # Different zone types: small, medium, large VAV boxes
        coil = ElectricReheatCoil(
            max_thermal_output=[1500.0, 3000.0, 5000.0],  # Different capacity per zone
            electrical_efficiency=[0.98, 0.97, 0.96],     # Different efficiency per zone
            tau=[8.0, 10.0, 12.0]                         # Different response times
        )

        # Mock expanded parameters
        coil.max_thermal_output = torch.tensor([1500.0, 3000.0, 5000.0])
        coil.electrical_efficiency = torch.tensor([0.98, 0.97, 0.96])
        coil.tau = torch.tensor([8.0, 10.0, 12.0])

        target_heatings = coil.max_thermal_output * 0.7  # 70% of max for each zone
        current_positions = torch.tensor([[0.3, 0.3, 0.3]])

        result = coil.forward(
            t=0.0,
            target_heating=target_heatings.unsqueeze(0),
            current_position=current_positions,
            dt=2.0
        )

        # All zones should produce realistic outputs
        assert torch.all((result["position"] >= 0.0) & (result["position"] <= 1.0))
        assert torch.all(result["thermal_output"] >= 0.0)
        assert torch.all(result["thermal_output"] <= coil.max_thermal_output)
        assert torch.all(result["electrical_power"] >= result["thermal_output"])

    def test_actuator_time_constants_vectorized(self):
        """Test realistic actuator response times with vectorization."""
        coil = ElectricReheatCoil(
            tau=[5.0, 20.0],  # Fast contactors vs slow transformers
            max_thermal_output=[3000.0, 3000.0]
        )
        coil.max_thermal_output = torch.tensor([3000.0, 3000.0])
        coil.tau = torch.tensor([5.0, 20.0])

        target_heating = torch.tensor([[2000.0, 2000.0]])
        initial_position = torch.tensor([[0.1, 0.1]])

        result = coil.forward(
            t=0.0, target_heating=target_heating,
            current_position=initial_position, dt=1.0
        )

        # Fast actuator (zone 0) should respond quicker than slow actuator (zone 1)
        movements = result["position"] - initial_position
        assert movements[0, 0] > movements[0, 1], "Fast actuator should respond quicker"

    def test_zone_independence(self):
        """Test that zones operate independently with different parameters."""
        coil = ElectricReheatCoil(
            max_thermal_output=[1000.0, 2000.0, 3000.0],
            electrical_efficiency=[0.95, 0.98, 1.0],
            tau=[8.0, 10.0, 12.0],
            heating_characteristic="linear"
        )

        # Mock expanded parameters
        coil.max_thermal_output = torch.tensor([1000.0, 2000.0, 3000.0])
        coil.electrical_efficiency = torch.tensor([0.95, 0.98, 1.0])
        coil.tau = torch.tensor([8.0, 10.0, 12.0])

        # Same relative target (50% of max) for all zones
        target_heatings = coil.max_thermal_output * 0.5
        current_positions = torch.zeros(1, 3)

        result = coil.forward(
            t=0.0,
            target_heating=target_heatings.unsqueeze(0),
            current_position=current_positions,
            dt=1.0
        )

        # All zones should aim for same relative position setpoint (linear characteristic)
        expected_setpoint = 0.5  # For linear: position = heating/max_heating
        torch.testing.assert_close(
            result["position_setpoint"],
            torch.full((1, 3), expected_setpoint),
            rtol=1e-5, atol=1e-8
        )

        # But actual positions should differ due to different time constants
        positions = result["position"][0]
        assert positions[0] > positions[1] > positions[2], "Faster zones should respond more"

        # Electrical power should vary due to different efficiencies
        electrical_powers = result["electrical_power"][0]
        thermal_outputs = result["thermal_output"][0]

        # Zone 0 (lowest efficiency) should use most electrical power for same thermal output
        efficiency_ratios = thermal_outputs / electrical_powers
        expected_efficiencies = coil.electrical_efficiency
        torch.testing.assert_close(efficiency_ratios, expected_efficiencies, rtol=1e-5, atol=1e-8)


class TestRoundTripAccuracy:
    """Test round-trip accuracy of heating ↔ position conversions with vectorization."""

    @pytest.mark.parametrize("characteristic", ["linear", "equal_percent", "quick_opening"])
    def test_position_heating_roundtrip_vectorized(self, characteristic):
        """Test position → heating → position round-trip with vectorization."""
        coil = ElectricReheatCoil(
            heating_characteristic=characteristic,
            max_thermal_output=[2000.0, 3000.0, 4000.0]
        )
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0, 4000.0])

        original_positions = torch.tensor([[0.1, 0.3, 0.5],
                                          [0.2, 0.6, 0.9],
                                          [0.05, 0.4, 0.8]])  # 3 batches, 3 zones

        # Forward: position → thermal output
        thermal_outputs = coil.position_to_thermal_output(original_positions)

        # Inverse: thermal output → position
        reconstructed_positions = coil.target_heating_to_position(thermal_outputs)

        # Should recover original positions (within numerical precision)
        torch.testing.assert_close(reconstructed_positions, original_positions, rtol=1e-5, atol=1e-9)

    @pytest.mark.parametrize("characteristic", ["linear", "equal_percent", "quick_opening"])
    def test_heating_position_roundtrip_vectorized(self, characteristic):
        """Test heating → position → heating round-trip with vectorization."""
        coil = ElectricReheatCoil(
            heating_characteristic=characteristic,
            max_thermal_output=[2500.0, 3500.0, 4500.0]
        )
        coil.max_thermal_output = torch.tensor([2500.0, 3500.0, 4500.0])

        original_heatings = torch.tensor([[250.0, 700.0, 1350.0],
                                         [500.0, 2100.0, 3600.0]])  # 2 batches, 3 zones

        # Forward: thermal output → position
        positions = coil.target_heating_to_position(original_heatings)

        # Inverse: position → thermal output
        reconstructed_heatings = coil.position_to_thermal_output(positions)

        # Should recover original thermal outputs (within numerical precision)
        torch.testing.assert_close(reconstructed_heatings, original_heatings, rtol=1e-5, atol=1e-9)


class TestEnergyAnalysis:
    """Test energy-related calculations important for building simulation with vectorization."""

    def test_part_load_efficiency_vectorized(self):
        """Test that efficiency remains constant at all load levels across zones."""
        coil = ElectricReheatCoil(electrical_efficiency=[0.94, 0.97, 0.99])
        coil.max_thermal_output = torch.tensor([3000.0, 3000.0, 3000.0])
        coil.electrical_efficiency = torch.tensor([0.94, 0.97, 0.99])

        positions = torch.linspace(0.1, 1.0, 10).unsqueeze(1).expand(-1, 3)  # [10, 3]

        thermal_outputs = coil.position_to_thermal_output(positions)
        electrical_powers = coil.position_to_electrical_power(positions)

        efficiencies = thermal_outputs / electrical_powers
        expected_efficiencies = coil.electrical_efficiency.unsqueeze(0).expand(10, -1)

        torch.testing.assert_close(efficiencies, expected_efficiencies, rtol=1e-6, atol=1e-10)

    def test_annual_energy_calculation_vectorized(self):
        """Test calculations relevant for annual energy analysis across zones."""
        coil = ElectricReheatCoil(
            max_thermal_output=[2000.0, 3000.0, 4000.0],  # Small, medium, large zones
            electrical_efficiency=[0.98, 0.97, 0.96]
        )
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0, 4000.0])
        coil.electrical_efficiency = torch.tensor([0.98, 0.97, 0.96])

        # Simulate typical winter heating scenario
        # Different heating loads per zone: 40%, 60%, 80% of max
        avg_positions = torch.tensor([[0.4, 0.6, 0.8]])
        hours_per_day = 8.0
        days_per_winter = 120.0  # 4 months

        thermal_outputs = coil.position_to_thermal_output(avg_positions)
        electrical_powers = coil.position_to_electrical_power(avg_positions)

        # Daily energy consumption per zone
        daily_thermal_kwh = thermal_outputs * hours_per_day / 1000.0  # Convert W·h to kWh
        daily_electrical_kwh = electrical_powers * hours_per_day / 1000.0

        # Winter seasonal consumption per zone
        winter_thermal_kwh = daily_thermal_kwh * days_per_winter
        winter_electrical_kwh = daily_electrical_kwh * days_per_winter

        # Verify reasonable values for different zone sizes
        # Small zone (2kW max): expect 0.5-2 MWh per season
        # Large zone (4kW max): expect 2-6 MWh per season
        assert torch.all(winter_electrical_kwh >= 500.0)   # At least 0.5 MWh
        assert torch.all(winter_electrical_kwh <= 6000.0)  # At most 6 MWh
        assert torch.all(winter_thermal_kwh < winter_electrical_kwh)  # Electrical > thermal

    def test_zone_energy_comparison(self):
        """Test energy consumption comparisons between different zone types."""
        coil = ElectricReheatCoil(
            max_thermal_output=[1500.0, 3000.0, 6000.0],  # Small, medium, large
            electrical_efficiency=[0.98, 0.97, 0.95]      # High, medium, low efficiency
        )
        coil.max_thermal_output = torch.tensor([1500.0, 3000.0, 6000.0])
        coil.electrical_efficiency = torch.tensor([0.98, 0.97, 0.95])

        # Same relative heating load (50% of capacity) for all zones
        positions = torch.tensor([[0.5, 0.5, 0.5]])

        thermal_outputs = coil.position_to_thermal_output(positions)
        electrical_powers = coil.position_to_electrical_power(positions)

        # Thermal outputs should scale with capacity
        expected_thermal = coil.max_thermal_output * 0.5
        torch.testing.assert_close(thermal_outputs[0], expected_thermal, rtol=1e-6, atol=1e-10)

        # Electrical powers should account for different efficiencies
        expected_electrical = expected_thermal / coil.electrical_efficiency
        torch.testing.assert_close(electrical_powers[0], expected_electrical, rtol=1e-6, atol=1e-10)

        # Large zone with low efficiency should use most power
        assert electrical_powers[0, 2] > electrical_powers[0, 1] > electrical_powers[0, 0]


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])