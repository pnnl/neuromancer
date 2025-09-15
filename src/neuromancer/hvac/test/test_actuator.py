"""
test_actuator.py

Comprehensive pytest test suite for the Actuator class with zone vectorization support.
Tests all methods, edge cases, parameter types, and integration scenarios.

Run with: pytest test_actuator.py -v
"""

import pytest
import torch
import numpy as np

from torch_buildings import Actuator


class TestActuatorInitialization:
    """Test actuator initialization and parameter handling."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        actuator = Actuator()
        assert actuator.model == "instantaneous"
        assert actuator.name == "actuator"
        # tau should be stored as provided (will be expanded by BuildingComponent)
        assert actuator.tau == 15.0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        actuator = Actuator(tau=10.0, model="analytic", name="test_actuator")
        assert actuator.model == "analytic"
        assert actuator.name == "test_actuator"
        assert actuator.tau == 10.0

    def test_tensor_tau_initialization(self):
        """Test initialization with tensor tau."""
        tau_tensor = torch.tensor([20.0, 25.0, 30.0])  # Zone-specific tau
        actuator = Actuator(tau=tau_tensor)
        assert torch.allclose(actuator.tau, tau_tensor)
        assert isinstance(actuator.tau, torch.Tensor)

    def test_list_tau_initialization(self):
        """Test initialization with list tau (zone-specific)."""
        tau_list = [10.0, 15.0, 20.0]
        actuator = Actuator(tau=tau_list)
        # tau should be stored as provided (expansion happens in BuildingComponent)
        assert actuator.tau == tau_list

    def test_invalid_model_raises_error(self):
        """Test that invalid model names raise ValueError."""
        with pytest.raises(ValueError, match="model must be one of"):
            Actuator(model="invalid_model")

    def test_valid_models(self):
        """Test all valid model types can be initialized."""
        # Updated: removed "odesolve"
        valid_models = ["instantaneous", "analytic", "smooth_approximation"]
        for model in valid_models:
            actuator = Actuator(model=model)
            assert actuator.model == model

    def test_odesolve_model_removed(self):
        """Test that odesolve model is no longer supported."""
        with pytest.raises(ValueError, match="model must be one of"):
            Actuator(model="odesolve")


class TestInstantaneousMethod:
    """Test the instantaneous method with vectorization support."""

    def test_instantaneous_scalar(self):
        """Test instantaneous response with scalar input."""
        actuator = Actuator(model="instantaneous")
        setpoint = torch.tensor(0.7)
        result = actuator.forward(setpoint=setpoint)
        assert torch.allclose(result, setpoint)

    def test_instantaneous_batch(self):
        """Test instantaneous response with batch input."""
        actuator = Actuator(model="instantaneous")
        setpoint = torch.tensor([0.2, 0.5, 0.8])
        result = actuator.forward(setpoint=setpoint)
        assert torch.allclose(result, setpoint)

    def test_instantaneous_zone_vectorized(self):
        """Test instantaneous response with zone vectorization."""
        actuator = Actuator(model="instantaneous")
        # Mock zone-specific tau (not used by instantaneous but should not cause issues)
        actuator.tau = torch.tensor([5.0, 10.0, 15.0])

        # Test with [batch_size, n_zones] input
        setpoint = torch.tensor([[0.2, 0.5, 0.8],
                                [0.1, 0.6, 0.9]])  # 2 batches, 3 zones
        result = actuator.forward(setpoint=setpoint)
        assert torch.allclose(result, setpoint)
        assert result.shape == (2, 3)

    def test_instantaneous_ignores_position(self):
        """Test that instantaneous method ignores position input."""
        actuator = Actuator(model="instantaneous")
        setpoint = torch.tensor(0.5)
        position = torch.tensor(0.9)  # Should be ignored
        result = actuator.forward(setpoint=setpoint, position=position)
        assert torch.allclose(result, setpoint)


class TestAnalyticMethod:
    """Test the analytic method with vectorization support."""

    def test_analytic_requires_position(self):
        """Test that analytic method requires position input."""
        actuator = Actuator(model="analytic")
        actuator.tau = torch.tensor(10.0)  # Mock expanded parameter
        setpoint = torch.tensor(0.5)
        with pytest.raises(ValueError, match="position required"):
            actuator.forward(setpoint=setpoint, position=None)

    def test_analytic_step_response(self):
        """Test analytic method step response behavior."""
        actuator = Actuator(tau=10.0, model="analytic")
        actuator.tau = torch.tensor(10.0)  # Mock expanded parameter
        position = torch.tensor(0.0)
        setpoint = torch.tensor(1.0)
        dt = 1.0

        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Should follow first-order lag: x(t+dt) = sp + (x0-sp)*exp(-dt/tau)
        expected = setpoint + (position - setpoint) * torch.exp(-dt / actuator.tau)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_analytic_zone_vectorized(self):
        """Test analytic method with zone vectorization."""
        actuator = Actuator(tau=[5.0, 10.0, 15.0], model="analytic")
        # Mock expanded zone-specific tau
        actuator.tau = torch.tensor([5.0, 10.0, 15.0])

        position = torch.tensor([[0.0, 0.5, 1.0],
                                [0.2, 0.3, 0.8]])  # 2 batches, 3 zones
        setpoint = torch.tensor([[1.0, 0.0, 0.5],
                                [0.8, 0.7, 0.2]])
        dt = 2.0

        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Check shape
        assert result.shape == (2, 3)

        # Verify each zone uses its own tau
        expected = setpoint + (position - setpoint) * torch.exp(-dt / actuator.tau)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_analytic_steady_state(self):
        """Test that analytic method reaches steady state."""
        actuator = Actuator(tau=1.0, model="analytic")
        actuator.tau = torch.tensor(1.0)  # Mock expanded parameter
        position = torch.tensor(0.0)
        setpoint = torch.tensor(0.8)

        # Simulate many time steps
        for _ in range(100):
            position = actuator.forward(setpoint=setpoint, position=position, dt=1.0)

        # Should be very close to setpoint
        assert torch.allclose(position, setpoint, atol=1e-3)


class TestSmoothApproximationMethod:
    """Test the smooth approximation method with vectorization support."""

    def test_smooth_requires_position(self):
        """Test that smooth approximation requires position input."""
        actuator = Actuator(model="smooth_approximation")
        actuator.tau = torch.tensor(10.0)  # Mock expanded parameter
        setpoint = torch.tensor(0.5)
        with pytest.raises(ValueError, match="position required"):
            actuator.forward(setpoint=setpoint, position=None)

    def test_smooth_training_vs_eval_mode(self):
        """Test different behavior in training vs evaluation mode."""
        actuator = Actuator(tau=10.0, model="smooth_approximation")
        actuator.tau = torch.tensor(10.0)  # Mock expanded parameter
        position = torch.tensor(0.0)
        setpoint = torch.tensor(1.0)
        dt = 5.0  # Medium rate = dt/tau = 0.5

        # Training mode (uses Pade for medium rates)
        actuator.train()
        result_train = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Evaluation mode (uses exact for rates < 2.0)
        actuator.eval()
        result_eval = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Results should be different due to different approximation methods
        assert not torch.allclose(result_train, result_eval, atol=1e-4), \
            f"Train: {result_train.item():.6f}, Eval: {result_eval.item():.6f}"

    def test_smooth_zone_vectorized_training_modes(self):
        """Test smooth approximation with vectorized zones in different training modes."""
        actuator = Actuator(model="smooth_approximation")
        # Zone-specific tau values creating different rate regimes
        actuator.tau = torch.tensor([10.0, 2.0, 0.5])  # rates: 0.5, 2.5, 10.0 with dt=5.0

        position = torch.tensor([[0.0, 0.0, 0.0]])  # 1 batch, 3 zones
        setpoint = torch.tensor([[1.0, 1.0, 1.0]])
        dt = 5.0

        # Training mode: different approximations for different zones
        actuator.train()
        result_train = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Evaluation mode
        actuator.eval()
        result_eval = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        assert result_train.shape == (1, 3)
        assert result_eval.shape == (1, 3)

        # At least some zones should have different results
        differences = torch.abs(result_train - result_eval)
        assert torch.any(differences > 1e-4), "Training and eval should differ for some zones"

    def test_smooth_approximation_regions(self):
        """Test different approximation regions based on dt/tau ratio with vectorization."""
        actuator = Actuator(model="smooth_approximation")
        # Set up different tau values to test different rate regions
        actuator.tau = torch.tensor([100.0, 10.0, 1.0])  # Small, medium, large rates

        position = torch.tensor([[0.0, 0.0, 0.0]])
        setpoint = torch.tensor([[1.0, 1.0, 1.0]])
        dt = 5.0  # rates: 0.05, 0.5, 5.0

        actuator.train()
        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # All should give reasonable results between 0 and 1
        assert torch.all((result >= 0) & (result <= 1))
        assert result.shape == (1, 3)


# Remove TestODESolveMethod class entirely since odesolve is no longer supported


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_forward_model(self):
        """Test error handling for invalid model in forward pass."""
        actuator = Actuator()
        actuator.model = "invalid_model"  # Manually set invalid model

        with pytest.raises(ValueError, match="Unknown model type"):
            actuator.forward(setpoint=torch.tensor(0.5))

    def test_zero_tau_handling(self):
        """Test behavior with very small tau values."""
        actuator = Actuator(tau=1e-6, model="analytic")
        actuator.tau = torch.tensor(1e-6)  # Mock expanded parameter
        position = torch.tensor(0.0)
        setpoint = torch.tensor(1.0)
        dt = 1.0

        # Should not crash and should be close to setpoint
        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)
        assert torch.isfinite(result).all()
        assert torch.allclose(result, setpoint, atol=1e-3)

    def test_large_tau_handling(self):
        """Test behavior with very large tau values."""
        actuator = Actuator(tau=1e6, model="analytic")
        actuator.tau = torch.tensor(1e6)  # Mock expanded parameter
        position = torch.tensor(0.0)
        setpoint = torch.tensor(1.0)
        dt = 1.0

        # Should barely move from initial position
        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)
        assert torch.isfinite(result).all()
        assert torch.allclose(result, position, atol=1e-3)


class TestGradientComputation:
    """Test gradient computation for learnable parameters with vectorization."""

    def test_learnable_tau_gradients_scalar(self):
        """Test that gradients flow through learnable scalar tau."""
        tau = torch.nn.Parameter(torch.tensor(10.0, requires_grad=True))
        actuator = Actuator(tau=tau, model="analytic")
        actuator.tau = tau  # Mock: in real usage, BuildingComponent would expand this

        position = torch.tensor(0.0, requires_grad=False)
        setpoint = torch.tensor(1.0, requires_grad=False)
        dt = 1.0

        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)
        loss = torch.sum(result)
        loss.backward()

        # Tau should have gradients
        assert tau.grad is not None
        assert torch.isfinite(tau.grad).all()

    def test_learnable_tau_gradients_vectorized(self):
        """Test that gradients flow through learnable zone-specific tau."""
        tau = torch.nn.Parameter(torch.tensor([5.0, 10.0, 15.0], requires_grad=True))
        actuator = Actuator(tau=tau, model="analytic")
        actuator.tau = tau

        position = torch.tensor([[0.0, 0.2, 0.5]])  # 1 batch, 3 zones
        setpoint = torch.tensor([[1.0, 0.8, 0.3]])
        dt = 1.0

        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)
        loss = torch.sum(result)
        loss.backward()

        # All tau values should have gradients
        assert tau.grad is not None
        assert tau.grad.shape == (3,)
        assert torch.isfinite(tau.grad).all()

    def test_gradient_stability_smooth_approximation(self):
        """Test gradient stability in smooth approximation method."""
        tau = torch.nn.Parameter(torch.tensor([1.0, 5.0, 10.0], requires_grad=True))
        actuator = Actuator(tau=tau, model="smooth_approximation")
        actuator.tau = tau

        position = torch.tensor([[0.0, 0.0, 0.0]])
        setpoint = torch.tensor([[1.0, 1.0, 1.0]])

        # Test across different dt values (creating different rates for each zone)
        for dt in [0.1, 1.0, 5.0, 10.0]:
            tau.grad = None  # Reset gradients

            result = actuator.forward(setpoint=setpoint, position=position, dt=dt)
            loss = torch.sum(result ** 2)
            loss.backward()

            # Gradients should be finite for all zones
            assert torch.isfinite(tau.grad).all()
            assert not torch.isnan(tau.grad).any()


class TestDeviceAndDtypeHandling:
    """Test device and dtype compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test that actuator works with CUDA tensors."""
        device = torch.device('cuda')
        actuator = Actuator(model="analytic")
        actuator.tau = torch.tensor([10.0, 15.0], device=device)  # Mock expanded parameter

        position = torch.tensor([[0.0, 0.5]], device=device)
        setpoint = torch.tensor([[1.0, 0.3]], device=device)

        result = actuator.forward(setpoint=setpoint, position=position, dt=1.0)

        assert result.device == device
        assert torch.isfinite(result).all()

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float64]:
            actuator = Actuator(model="analytic")
            actuator.tau = torch.tensor([10.0, 15.0], dtype=dtype)

            position = torch.tensor([[0.0, 0.5]], dtype=dtype)
            setpoint = torch.tensor([[1.0, 0.3]], dtype=dtype)

            result = actuator.forward(setpoint=setpoint, position=position, dt=1.0)

            assert result.dtype == dtype


class TestIntegrationScenarios:
    """Test realistic integration scenarios with vectorization."""

    def test_control_loop_simulation_vectorized(self):
        """Test actuator in a vectorized control loop simulation."""
        actuator = Actuator(model="analytic")
        # Different time constants for different zones
        actuator.tau = torch.tensor([2.0, 5.0, 10.0])

        # Use same setpoint for fair comparison of convergence rates
        position = torch.tensor([[0.0, 0.0, 0.0]])  # 1 batch, 3 zones
        setpoint = torch.tensor([[0.8, 0.8, 0.8]])  # Same setpoint for all zones
        dt = 1.0

        positions_history = []
        for i in range(30):
            position = actuator.forward(setpoint=setpoint, position=position, dt=dt)
            positions_history.append(position.clone())

        positions_history = torch.stack(positions_history, dim=0)  # [time, batch, zones]

        # Compare convergence after fewer steps to see the difference
        early_positions = positions_history[4, 0, :]  # After 5 time steps
        final_positions = positions_history[-1, 0, :]  # Last time step, first batch

        # Zone 0 (tau=2) should converge faster than zone 2 (tau=10)
        # Check that faster zone is closer to setpoint after early steps
        error_zone0_early = abs(early_positions[0] - setpoint[0, 0])
        error_zone2_early = abs(early_positions[2] - setpoint[0, 2])
        assert error_zone0_early < error_zone2_early, \
            f"Zone 0 (tau=2) should converge faster. Errors: {error_zone0_early:.4f} vs {error_zone2_early:.4f}"

        # All should eventually reach setpoint (with reasonable tolerance for slow systems)
        final_errors = torch.abs(final_positions - setpoint[0])
        assert torch.all(final_errors < 0.05), f"Final errors: {final_errors}"

        # Verify the convergence trend - zone 0 should have smallest error, zone 2 largest
        assert final_errors[0] < final_errors[1] < final_errors[2], \
            f"Errors should increase with tau: {final_errors}"

    def test_batch_and_zone_simulation(self):
        """Test simulation with both batch and zone dimensions."""
        n_batches, n_zones = 3, 4
        actuator = Actuator(model="analytic")
        actuator.tau = torch.linspace(3.0, 12.0, n_zones)  # Different tau per zone

        positions = torch.zeros(n_batches, n_zones)
        # Different setpoints for each batch and zone
        setpoints = torch.rand(n_batches, n_zones) * 0.8 + 0.1  # Between 0.1 and 0.9
        dt = 2.0

        # Simulate multiple time steps
        for _ in range(15):
            positions = actuator.forward(setpoint=setpoints, position=positions, dt=dt)

        # Each actuator should approach its setpoint
        errors = torch.abs(positions - setpoints)
        assert torch.all(errors < 0.15)  # Reasonable convergence

    def test_parameter_optimization_scenario_vectorized(self):
        """Test scenario where zone-specific tau values are optimized."""
        # Target: different zones should reach different response levels
        target_responses = torch.tensor([0.6, 0.8, 0.9])  # Different targets per zone
        target_time = 8.0
        dt = 1.0
        n_steps = int(target_time / dt)

        # Initialize learnable zone-specific tau
        tau = torch.nn.Parameter(torch.tensor([5.0, 5.0, 5.0], requires_grad=True))
        actuator = Actuator(tau=tau, model="smooth_approximation")
        actuator.tau = tau

        optimizer = torch.optim.Adam([tau], lr=0.05)

        # Optimize tau to achieve target responses
        for epoch in range(50):  # Reduced iterations for testing
            optimizer.zero_grad()

            position = torch.tensor([[0.0, 0.0, 0.0]])
            setpoint = torch.tensor([[1.0, 1.0, 1.0]])

            # Simulate step response
            for _ in range(n_steps):
                position = actuator.forward(setpoint=setpoint, position=position, dt=dt)

            # Loss: want each zone to reach its target response
            loss = torch.sum((position[0] - target_responses) ** 2)
            loss.backward()
            optimizer.step()

            # Constrain tau to be positive
            with torch.no_grad():
                tau.clamp_(min=0.5)

        # Should have learned reasonable tau values
        assert torch.all((tau >= 0.5) & (tau <= 30.0))
        final_errors = torch.abs(position[0] - target_responses)
        assert torch.all(final_errors < 0.2)  # Reasonable convergence


class TestNumericalStability:
    """Test numerical stability across different parameter ranges with vectorization."""

    def test_extreme_dt_values_vectorized(self):
        """Test with very small and very large dt values across multiple zones."""
        actuator = Actuator(model="smooth_approximation")
        actuator.tau = torch.tensor([1.0, 10.0, 100.0])  # Different time scales

        position = torch.tensor([[0.0, 0.5, 1.0]])
        setpoint = torch.tensor([[1.0, 0.0, 0.5]])

        # Very small dt
        result_small = actuator.forward(setpoint=setpoint, position=position, dt=1e-6)
        assert torch.isfinite(result_small).all()
        assert torch.allclose(result_small, position, atol=1e-3)

        # Very large dt
        result_large = actuator.forward(setpoint=setpoint, position=position, dt=1e6)
        assert torch.isfinite(result_large).all()
        # With very large dt, should be close to setpoint
        assert torch.allclose(result_large, setpoint, atol=1e-2)

    def test_extreme_setpoint_values_vectorized(self):
        """Test with setpoint values at boundaries across multiple zones."""
        actuator = Actuator(model="analytic")
        actuator.tau = torch.tensor([5.0, 10.0])

        position = torch.tensor([[0.5, 0.5]])
        dt = 1.0

        # Test boundary values for different zones
        boundary_setpoints = torch.tensor([[0.0, 1.0],    # Min/max
                                          [-0.1, 1.1],    # Beyond boundaries
                                          [0.5, 0.5]])    # Middle values

        for setpoint in boundary_setpoints:
            result = actuator.forward(setpoint=setpoint.unsqueeze(0), position=position, dt=dt)
            assert torch.isfinite(result).all()


"""
Additional unit tests for Actuator to specifically test batch_size > 1 with n_zones > 1.
These tests isolate the tensor shape issues seen in VAVBox simulation.

Add these to the existing test_actuator.py file.
"""


class TestBatchZoneVectorization:
    """Test actuator with both batch dimensions and zone dimensions simultaneously."""

    def test_batch_zone_instantaneous(self):
        """Test instantaneous actuator with batch_size > 1 and n_zones > 1."""
        actuator = Actuator(model="instantaneous")
        # Zone-specific tau (not used by instantaneous but should not interfere)
        actuator.tau = torch.tensor([5.0, 10.0])  # 2 zones

        batch_size = 3
        n_zones = 2

        # Input: [batch_size, n_zones]
        setpoint = torch.tensor([[0.2, 0.5],
                                 [0.3, 0.7],
                                 [0.1, 0.9]])  # 3 batches, 2 zones

        result = actuator.forward(setpoint=setpoint)

        # Should return exact setpoint for instantaneous
        assert torch.allclose(result, setpoint)
        assert result.shape == (batch_size, n_zones)

    def test_batch_zone_analytic(self):
        """Test analytic actuator with batch_size > 1 and n_zones > 1."""
        actuator = Actuator(model="analytic")
        actuator.tau = torch.tensor([5.0, 10.0])  # 2 zones

        batch_size = 4
        n_zones = 2

        # Inputs: [batch_size, n_zones]
        position = torch.tensor([[0.1, 0.2],
                                 [0.0, 0.5],
                                 [0.8, 0.3],
                                 [0.4, 0.7]])  # 4 batches, 2 zones

        setpoint = torch.tensor([[0.9, 0.1],
                                 [0.7, 0.2],
                                 [0.3, 0.8],
                                 [0.2, 0.9]])  # 4 batches, 2 zones
        dt = 2.0

        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Check output shape
        assert result.shape == (batch_size, n_zones)

        # Verify physics: each zone should use its own tau
        expected = setpoint + (position - setpoint) * torch.exp(-dt / actuator.tau)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_batch_zone_smooth_approximation(self):
        """Test smooth approximation actuator with batch_size > 1 and n_zones > 1."""
        actuator = Actuator(model="smooth_approximation")
        actuator.tau = torch.tensor([2.0, 8.0, 15.0])  # 3 zones

        batch_size = 5
        n_zones = 3

        # Inputs: [batch_size, n_zones]
        position = torch.zeros(batch_size, n_zones)
        setpoint = torch.ones(batch_size, n_zones)
        dt = 1.0

        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Check output shape
        assert result.shape == (batch_size, n_zones)

        # Should be between position and setpoint
        assert torch.all((result >= position) & (result <= setpoint))

    def test_large_batch_zone_dimensions(self):
        """Test with large batch and zone dimensions to stress-test broadcasting."""
        actuator = Actuator(model="analytic")

        batch_size = 10
        n_zones = 6
        actuator.tau = torch.linspace(1.0, 20.0, n_zones)  # Different tau per zone

        # Random inputs with correct shapes
        position = torch.rand(batch_size, n_zones) * 0.5  # [0, 0.5]
        setpoint = torch.rand(batch_size, n_zones) * 0.5 + 0.5  # [0.5, 1.0]
        dt = 1.5

        result = actuator.forward(setpoint=setpoint, position=position, dt=dt)

        # Check output shape
        assert result.shape == (batch_size, n_zones)

        # Should be finite and bounded
        assert torch.isfinite(result).all()
        assert torch.all((result >= 0.0) & (result <= 1.0))

    def test_broadcasting_edge_cases(self):
        """Test edge cases in tensor broadcasting between batch and zone dimensions."""
        actuator = Actuator(model="analytic")
        actuator.tau = torch.tensor([3.0, 6.0])  # 2 zones

        # Test Case 1: Single batch, multiple zones
        position1 = torch.tensor([[0.2, 0.8]])  # [1, 2]
        setpoint1 = torch.tensor([[0.7, 0.3]])  # [1, 2]

        result1 = actuator.forward(setpoint=setpoint1, position=position1, dt=1.0)
        assert result1.shape == (1, 2)

        # Test Case 2: Multiple batches, multiple zones
        position2 = torch.tensor([[0.1, 0.9],
                                  [0.4, 0.6]])  # [2, 2]
        setpoint2 = torch.tensor([[0.8, 0.2],
                                  [0.3, 0.7]])  # [2, 2]

        result2 = actuator.forward(setpoint=setpoint2, position=position2, dt=1.0)
        assert result2.shape == (2, 2)

    def test_tau_broadcasting_with_batch_zone(self):
        """Test that zone-specific tau correctly broadcasts with batch dimensions."""
        actuator = Actuator(model="analytic")

        # Different scenarios for tau broadcasting
        test_cases = [
            {
                "tau": torch.tensor([5.0]),  # Single zone
                "position_shape": (3, 1),  # 3 batches, 1 zone
                "description": "Single zone, multiple batches"
            },
            {
                "tau": torch.tensor([2.0, 8.0]),  # 2 zones
                "position_shape": (1, 2),  # 1 batch, 2 zones
                "description": "Multiple zones, single batch"
            },
            {
                "tau": torch.tensor([1.0, 4.0, 10.0]),  # 3 zones
                "position_shape": (4, 3),  # 4 batches, 3 zones
                "description": "Multiple zones, multiple batches"
            }
        ]

        for case in test_cases:
            actuator.tau = case["tau"]
            n_zones = len(case["tau"])
            batch_size, zones = case["position_shape"]

            position = torch.zeros(batch_size, zones)
            setpoint = torch.ones(batch_size, zones)
            dt = 2.0

            result = actuator.forward(setpoint=setpoint, position=position, dt=dt)

            # Check output shape matches input
            assert result.shape == (batch_size, zones), \
                f"Failed for {case['description']}: expected {(batch_size, zones)}, got {result.shape}"

            # Verify broadcasting worked correctly by checking physics
            if zones == n_zones:  # Only when zones match tau length
                expected = setpoint + (position - setpoint) * torch.exp(-dt / case["tau"])
                assert torch.allclose(result, expected, atol=1e-6), \
                    f"Physics check failed for {case['description']}"

    def test_mismatched_dimensions_error_handling(self):
        """Test that mismatched tensor dimensions raise clear errors."""
        actuator = Actuator(model="analytic")
        actuator.tau = torch.tensor([5.0, 10.0])  # 2 zones

        # Case 1: Position and setpoint have different zone dimensions
        position = torch.tensor([[0.1, 0.2]])  # [1, 2]
        setpoint = torch.tensor([[0.7, 0.3, 0.9]])  # [1, 3] - MISMATCH!

        with pytest.raises(RuntimeError, match="size"):
            actuator.forward(setpoint=setpoint, position=position, dt=1.0)

        # Case 2: tau doesn't match zone dimension
        actuator.tau = torch.tensor([2.0, 5.0, 8.0])  # 3 zones
        position = torch.tensor([[0.1, 0.2]])  # [1, 2] - MISMATCH!
        setpoint = torch.tensor([[0.7, 0.3]])  # [1, 2] - MISMATCH!

        # This should fail due to broadcasting issues
        with pytest.raises(RuntimeError):
            actuator.forward(setpoint=setpoint, position=position, dt=1.0)

    def test_dtype_consistency_batch_zone(self):
        """Test dtype consistency across batch and zone dimensions."""
        for dtype in [torch.float32, torch.float64]:
            actuator = Actuator(model="analytic")
            actuator.tau = torch.tensor([3.0, 7.0], dtype=dtype)

            position = torch.tensor([[0.1, 0.8], [0.4, 0.2]], dtype=dtype)
            setpoint = torch.tensor([[0.9, 0.1], [0.6, 0.7]], dtype=dtype)

            result = actuator.forward(setpoint=setpoint, position=position, dt=1.0)

            assert result.dtype == dtype
            assert result.shape == (2, 2)

    def test_gradient_flow_batch_zone(self):
        """Test gradient flow through learnable tau with batch and zone dimensions."""
        tau = torch.nn.Parameter(torch.tensor([4.0, 12.0], requires_grad=True))
        actuator = Actuator(tau=tau, model="analytic")
        actuator.tau = tau

        batch_size = 3
        n_zones = 2

        position = torch.zeros(batch_size, n_zones, requires_grad=False)
        setpoint = torch.ones(batch_size, n_zones, requires_grad=False)

        result = actuator.forward(setpoint=setpoint, position=position, dt=1.0)
        loss = torch.sum(result)
        loss.backward()

        # Both tau values should have gradients
        assert tau.grad is not None
        assert tau.grad.shape == (2,)
        assert torch.isfinite(tau.grad).all()

    def test_simulation_stress_test(self):
        """Stress test simulating the conditions from VAVBox simulation failures."""
        # This test replicates the exact conditions that cause the VAVBox simulation to fail
        actuator = Actuator(model="smooth_approximation")
        actuator.tau = torch.tensor([5.0, 5.0])  # 2 zones, same as VAVBox default

        # Simulate the exact error case from test_batch_simulation
        batch_size = 5  # Same as failing test
        n_zones = 2  # Same as VAVBox(n_zones=2)

        # The error showed: tensor a (2) vs tensor b (5)
        # This suggests position had shape [5] while setpoint had shape [batch_size, 2]

        # Correct shapes
        position_correct = torch.rand(batch_size, n_zones)
        setpoint_correct = torch.rand(batch_size, n_zones)

        result_correct = actuator.forward(setpoint=setpoint_correct, position=position_correct, dt=3600.0)
        assert result_correct.shape == (batch_size, n_zones)

        # Now test the problematic case that might occur in real simulation
        # If somehow position gets flattened to [batch_size * n_zones] = [10]
        # but setpoint remains [batch_size, n_zones] = [5, 2]
        position_flattened = torch.rand(batch_size * n_zones)  # [10] - this is wrong!
        setpoint_matrix = torch.rand(batch_size, n_zones)  # [5, 2] - this is correct

        # This should fail - exactly like the VAVBox simulation
        with pytest.raises(RuntimeError, match="size"):
            actuator.forward(setpoint=setpoint_matrix, position=position_flattened, dt=3600.0)

        print("Stress test confirmed: Shape mismatch occurs when position is flattened but setpoint is not")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])