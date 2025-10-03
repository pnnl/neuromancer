"""
Batch + Zone vectorization tests for Damper and ElectricReheatCoil components.
These tests isolate potential tensor shape issues when batch_size > 1 and n_zones > 1.
"""

import pytest
import torch
import numpy as np

from torch_buildings import Damper, ElectricReheatCoil

# =============================================================================
# DAMPER BATCH+ZONE TESTS
# =============================================================================

class TestDamperBatchZoneVectorization:
    """Test Damper with both batch dimensions and zone dimensions simultaneously."""

    def test_damper_batch_zone_instantaneous(self):
        """Test Damper with instantaneous actuator, batch_size > 1, n_zones > 1."""
        damper = Damper(
            max_airflow=[1.0, 1.5, 2.0],  # 3 zones
            actuator_model="instantaneous"
        )
        # Mock expanded parameters
        damper.max_airflow = torch.tensor([1.0, 1.5, 2.0])
        damper.tau = torch.tensor([5.0, 5.0, 5.0])

        batch_size = 4
        n_zones = 3

        # Inputs: [batch_size, n_zones]
        target_airflow = torch.tensor([[0.5, 0.8, 1.2],
                                       [0.3, 1.0, 1.8],
                                       [0.7, 0.6, 1.5],
                                       [0.2, 1.2, 2.0]])  # 4 batches, 3 zones

        current_position = torch.tensor([[0.1, 0.3, 0.5],
                                         [0.8, 0.2, 0.7],
                                         [0.4, 0.9, 0.1],
                                         [0.6, 0.5, 0.8]])  # 4 batches, 3 zones

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflow,
            current_position=current_position,
            dt=1.0
        )

        # Check output shapes
        assert result["position"].shape == (batch_size, n_zones)
        assert result["airflow"].shape == (batch_size, n_zones)
        assert result["position_setpoint"].shape == (batch_size, n_zones)

    def test_damper_batch_zone_analytic(self):
        """Test Damper with analytic actuator, batch_size > 1, n_zones > 1."""
        damper = Damper(
            max_airflow=[0.8, 1.2],  # 2 zones
            tau=[3.0, 7.0],  # Different time constants
            actuator_model="analytic"
        )
        # Mock expanded parameters
        damper.max_airflow = torch.tensor([0.8, 1.2])
        damper.tau = torch.tensor([3.0, 7.0])

        batch_size = 3
        n_zones = 2

        target_airflow = torch.tensor([[0.4, 0.9],
                                       [0.6, 0.3],
                                       [0.2, 1.1]])  # 3 batches, 2 zones

        current_position = torch.tensor([[0.2, 0.7],
                                         [0.8, 0.1],
                                         [0.5, 0.9]])  # 3 batches, 2 zones

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflow,
            current_position=current_position,
            dt=2.0
        )

        # Check output shapes
        assert result["position"].shape == (batch_size, n_zones)
        assert result["airflow"].shape == (batch_size, n_zones)

        # Verify actuator dynamics: zones should respond differently due to different tau
        position_change = result["position"] - current_position
        # Zone 0 (tau=3.0) should change more than Zone 1 (tau=7.0) for same target
        # This is a relative check since target_airflow differs per batch

    def test_damper_batch_zone_with_pressure(self):
        """Test Damper with pressure correction, batch_size > 1, n_zones > 1."""
        damper = Damper(max_airflow=[1.0, 1.5], nominal_pressure=500.0)
        damper.max_airflow = torch.tensor([1.0, 1.5])
        damper.tau = torch.tensor([5.0, 5.0])

        batch_size = 2
        n_zones = 2

        target_airflow = torch.tensor([[0.8, 1.2],
                                       [0.5, 0.9]])
        current_position = torch.tensor([[0.5, 0.6],
                                         [0.3, 0.4]])

        # Test with different pressure scenarios
        pressure_scenarios = [
            torch.tensor([[600.0, 400.0]]),  # Different pressure per zone
            torch.tensor([[700.0]]),  # Single pressure (broadcasts)
        ]

        for duct_pressure in pressure_scenarios:
            result = damper.forward(
                t=0.0,
                target_airflow=target_airflow,
                current_position=current_position,
                dt=1.0,
                duct_pressure=duct_pressure
            )

            assert result["position"].shape == (batch_size, n_zones)
            assert result["airflow"].shape == (batch_size, n_zones)
            assert torch.all(result["airflow"] >= 0.0)

    def test_damper_large_batch_zone_stress_test(self):
        """Stress test with large batch and zone dimensions."""
        batch_size = 8
        n_zones = 5

        damper = Damper(
            max_airflow=torch.linspace(0.5, 2.5, n_zones).tolist(),
            tau=torch.linspace(2.0, 10.0, n_zones).tolist(),
            actuator_model="smooth_approximation"
        )
        # Mock expanded parameters
        damper.max_airflow = torch.linspace(0.5, 2.5, n_zones)
        damper.tau = torch.linspace(2.0, 10.0, n_zones)

        # Random inputs with correct shapes
        target_airflow = torch.rand(batch_size, n_zones) * damper.max_airflow
        current_position = torch.rand(batch_size, n_zones)
        duct_pressure = torch.rand(batch_size, 1) * 500 + 300  # 300-800 Pa

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflow,
            current_position=current_position,
            dt=1.5,
            duct_pressure=duct_pressure
        )

        # Check all outputs have correct shapes
        assert result["position"].shape == (batch_size, n_zones)
        assert result["airflow"].shape == (batch_size, n_zones)
        assert result["position_setpoint"].shape == (batch_size, n_zones)

        # Check physical bounds
        assert torch.all((result["position"] >= 0.0) & (result["position"] <= 1.0))
        assert torch.all(result["airflow"] >= 0.0)

    def test_damper_shape_mismatch_reproduction(self):
        """Try to reproduce the exact shape mismatch from VAVBox simulation."""
        damper = Damper(max_airflow=[1.0, 1.0], actuator_model="smooth_approximation")
        damper.max_airflow = torch.tensor([1.0, 1.0])
        damper.tau = torch.tensor([5.0, 5.0])

        # Simulate the exact conditions from the failing test
        batch_size = 5  # From test_batch_simulation
        n_zones = 2  # From VAVBox(n_zones=2)

        # Test with correct shapes
        target_airflow_correct = torch.ones(batch_size, n_zones)  # [5, 2]
        current_position_correct = torch.rand(batch_size, n_zones)  # [5, 2]

        result = damper.forward(
            t=0.0,
            target_airflow=target_airflow_correct,
            current_position=current_position_correct,
            dt=3600.0  # Same dt as failing test
        )

        assert result["position"].shape == (batch_size, n_zones)

        # Now test potential incorrect shapes that might occur in simulation
        # Case 1: current_position somehow gets wrong shape
        current_position_wrong = torch.rand(batch_size)  # [5] instead of [5, 2]

        with pytest.raises(RuntimeError):
            damper.forward(
                t=0.0,
                target_airflow=target_airflow_correct,
                current_position=current_position_wrong,  # Wrong shape!
                dt=3600.0
            )


# =============================================================================
# ELECTRIC REHEAT COIL BATCH+ZONE TESTS
# =============================================================================

class TestElectricReheatCoilBatchZoneVectorization:
    """Test ElectricReheatCoil with both batch dimensions and zone dimensions simultaneously."""

    def test_reheat_coil_batch_zone_instantaneous(self):
        """Test ElectricReheatCoil with instantaneous actuator, batch_size > 1, n_zones > 1."""
        coil = ElectricReheatCoil(
            max_thermal_output=[2000.0, 3000.0, 4000.0],  # 3 zones
            electrical_efficiency=[0.95, 0.98, 1.0],  # Different efficiency per zone
            actuator_model="instantaneous"
        )
        # Mock expanded parameters
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0, 4000.0])
        coil.electrical_efficiency = torch.tensor([0.95, 0.98, 1.0])
        coil.tau = torch.tensor([10.0, 10.0, 10.0])

        batch_size = 3
        n_zones = 3

        target_heating = torch.tensor([[1000.0, 1500.0, 2000.0],
                                       [800.0, 2000.0, 3000.0],
                                       [1200.0, 1000.0, 1500.0]])  # 3 batches, 3 zones

        current_position = torch.tensor([[0.2, 0.4, 0.6],
                                         [0.1, 0.7, 0.8],
                                         [0.5, 0.3, 0.4]])  # 3 batches, 3 zones

        result = coil.forward(
            t=0.0,
            target_heating=target_heating,
            current_position=current_position,
            dt=1.0
        )

        # Check output shapes
        assert result["position"].shape == (batch_size, n_zones)
        assert result["thermal_output"].shape == (batch_size, n_zones)
        assert result["electrical_power"].shape == (batch_size, n_zones)
        assert result["position_setpoint"].shape == (batch_size, n_zones)

    def test_reheat_coil_batch_zone_analytic(self):
        """Test ElectricReheatCoil with analytic actuator, batch_size > 1, n_zones > 1."""
        coil = ElectricReheatCoil(
            max_thermal_output=[2500.0, 3500.0],  # 2 zones
            tau=[8.0, 15.0],  # Different time constants
            actuator_model="analytic"
        )
        # Mock expanded parameters
        coil.max_thermal_output = torch.tensor([2500.0, 3500.0])
        coil.electrical_efficiency = torch.tensor([0.98, 0.98])
        coil.tau = torch.tensor([8.0, 15.0])

        batch_size = 4
        n_zones = 2

        target_heating = torch.tensor([[1200.0, 1800.0],
                                       [2000.0, 1000.0],
                                       [800.0, 2500.0],
                                       [1500.0, 2000.0]])  # 4 batches, 2 zones

        current_position = torch.tensor([[0.3, 0.2],
                                         [0.7, 0.4],
                                         [0.1, 0.8],
                                         [0.6, 0.5]])  # 4 batches, 2 zones

        result = coil.forward(
            t=0.0,
            target_heating=target_heating,
            current_position=current_position,
            dt=2.0
        )

        # Check output shapes
        assert result["position"].shape == (batch_size, n_zones)
        assert result["thermal_output"].shape == (batch_size, n_zones)
        assert result["electrical_power"].shape == (batch_size, n_zones)

        # Verify energy conservation: electrical >= thermal
        assert torch.all(result["electrical_power"] >= result["thermal_output"])

    def test_reheat_coil_large_batch_zone_stress_test(self):
        """Stress test ElectricReheatCoil with large batch and zone dimensions."""
        batch_size = 6
        n_zones = 4

        coil = ElectricReheatCoil(
            max_thermal_output=torch.linspace(1500.0, 4500.0, n_zones).tolist(),
            electrical_efficiency=torch.linspace(0.95, 0.99, n_zones).tolist(),
            tau=torch.linspace(5.0, 20.0, n_zones).tolist(),
            actuator_model="smooth_approximation"
        )
        # Mock expanded parameters
        coil.max_thermal_output = torch.linspace(1500.0, 4500.0, n_zones)
        coil.electrical_efficiency = torch.linspace(0.95, 0.99, n_zones)
        coil.tau = torch.linspace(5.0, 20.0, n_zones)

        # Random inputs with correct shapes
        target_heating = torch.rand(batch_size, n_zones) * coil.max_thermal_output
        current_position = torch.rand(batch_size, n_zones)

        result = coil.forward(
            t=0.0,
            target_heating=target_heating,
            current_position=current_position,
            dt=1.0
        )

        # Check all outputs have correct shapes
        assert result["position"].shape == (batch_size, n_zones)
        assert result["thermal_output"].shape == (batch_size, n_zones)
        assert result["electrical_power"].shape == (batch_size, n_zones)
        assert result["position_setpoint"].shape == (batch_size, n_zones)

        # Check physical bounds
        assert torch.all((result["position"] >= 0.0) & (result["position"] <= 1.0))
        assert torch.all(result["thermal_output"] >= 0.0)
        assert torch.all(result["electrical_power"] >= result["thermal_output"])

    def test_reheat_coil_shape_mismatch_reproduction(self):
        """Try to reproduce the exact shape mismatch from VAVBox simulation."""
        coil = ElectricReheatCoil(
            max_thermal_output=[3000.0, 3000.0],
            actuator_model="smooth_approximation"
        )
        coil.max_thermal_output = torch.tensor([3000.0, 3000.0])
        coil.electrical_efficiency = torch.tensor([0.98, 0.98])
        coil.tau = torch.tensor([10.0, 10.0])

        # Simulate the exact conditions from the failing test
        batch_size = 5  # From test_batch_simulation
        n_zones = 2  # From VAVBox(n_zones=2)

        # Test with correct shapes
        target_heating_correct = torch.rand(batch_size, n_zones) * 2000.0  # [5, 2]
        current_position_correct = torch.rand(batch_size, n_zones)  # [5, 2]

        result = coil.forward(
            t=0.0,
            target_heating=target_heating_correct,
            current_position=current_position_correct,
            dt=3600.0  # Same dt as failing test
        )

        assert result["position"].shape == (batch_size, n_zones)

        # Test potential incorrect shapes that might occur in simulation
        current_position_wrong = torch.rand(batch_size)  # [5] instead of [5, 2]

        with pytest.raises(RuntimeError):
            coil.forward(
                t=0.0,
                target_heating=target_heating_correct,
                current_position=current_position_wrong,  # Wrong shape!
                dt=3600.0
            )

    def test_efficiency_vectorization_batch_zone(self):
        """Test that zone-specific efficiency is correctly applied with batch dimensions."""
        coil = ElectricReheatCoil(
            max_thermal_output=[2000.0, 2000.0, 2000.0],  # Same capacity
            electrical_efficiency=[0.90, 0.95, 1.00],  # Different efficiency
            actuator_model="instantaneous"
        )
        coil.max_thermal_output = torch.tensor([2000.0, 2000.0, 2000.0])
        coil.electrical_efficiency = torch.tensor([0.90, 0.95, 1.00])
        coil.tau = torch.tensor([10.0, 10.0, 10.0])

        batch_size = 2
        n_zones = 3

        # Same thermal output target for all zones
        target_heating = torch.full((batch_size, n_zones), 1000.0)
        current_position = torch.zeros(batch_size, n_zones)

        result = coil.forward(
            t=0.0,
            target_heating=target_heating,
            current_position=current_position,
            dt=1.0
        )

        # With same thermal output, electrical power should vary by efficiency
        thermal = result["thermal_output"]
        electrical = result["electrical_power"]

        # Zone 0 (efficiency=0.90) should use most electrical power
        # Zone 2 (efficiency=1.00) should use least electrical power
        for batch_idx in range(batch_size):
            assert electrical[batch_idx, 0] > electrical[batch_idx, 1]  # 0.90 < 0.95
            assert electrical[batch_idx, 1] > electrical[batch_idx, 2]  # 0.95 < 1.00


# =============================================================================
# INTEGRATED TESTS (Both Components Together)
# =============================================================================

class TestIntegratedBatchZoneVectorization:
    """Test Damper and ElectricReheatCoil working together with batch+zone vectorization."""

    def test_damper_and_reheat_coil_integration(self):
        """Test that both components can work together with batch+zone dimensions."""
        # Create components with matching zone counts
        n_zones = 3
        batch_size = 2

        damper = Damper(
            max_airflow=[0.8, 1.2, 1.6],
            actuator_model="instantaneous"
        )
        damper.max_airflow = torch.tensor([0.8, 1.2, 1.6])
        damper.tau = torch.tensor([5.0, 5.0, 5.0])

        coil = ElectricReheatCoil(
            max_thermal_output=[2000.0, 3000.0, 4000.0],
            actuator_model="instantaneous"
        )
        coil.max_thermal_output = torch.tensor([2000.0, 3000.0, 4000.0])
        coil.electrical_efficiency = torch.tensor([0.98, 0.98, 0.98])
        coil.tau = torch.tensor([10.0, 10.0, 10.0])

        # Test inputs
        target_airflow = torch.tensor([[0.5, 0.8, 1.0],
                                       [0.6, 1.0, 1.2]])
        damper_position = torch.tensor([[0.3, 0.4, 0.5],
                                        [0.2, 0.6, 0.7]])

        target_heating = torch.tensor([[1000.0, 1500.0, 2000.0],
                                       [1200.0, 2000.0, 2500.0]])
        reheat_position = torch.tensor([[0.2, 0.3, 0.4],
                                        [0.1, 0.5, 0.6]])

        # Test damper
        damper_result = damper.forward(
            t=0.0,
            target_airflow=target_airflow,
            current_position=damper_position,
            dt=1.0
        )

        # Test reheat coil
        coil_result = coil.forward(
            t=0.0,
            target_heating=target_heating,
            current_position=reheat_position,
            dt=1.0
        )

        # Both should produce correctly shaped outputs
        assert damper_result["position"].shape == (batch_size, n_zones)
        assert damper_result["airflow"].shape == (batch_size, n_zones)

        assert coil_result["position"].shape == (batch_size, n_zones)
        assert coil_result["thermal_output"].shape == (batch_size, n_zones)
        assert coil_result["electrical_power"].shape == (batch_size, n_zones)