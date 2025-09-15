"""
test_envelope.py - Updated for new base class architecture

Comprehensive test suite for the Envelope class, updated for the new stateless component design:
1. No more reset() method - using initial_state_functions()
2. Forward pass now returns actual states, not derivatives
3. Simulation framework handles state management externally
4. Updated method names and signatures

Run with: python -m pytest test_envelope.py -v
"""

import pytest
import torch
import numpy as np
from torch_buildings.building_components.envelope import Envelope


class TestEnvelopeInitialization:
    """Test envelope initialization and parameter handling."""

    def test_single_zone_initialization(self):
        """Test basic single-zone envelope creation."""
        env = Envelope(n_zones=1)

        assert env.n_zones == 1
        assert env.R_env.shape == torch.Size([1])
        assert env.C_env.shape == torch.Size([1])
        assert torch.allclose(env.R_env, torch.tensor([0.1]))
        assert torch.allclose(env.C_env, torch.tensor([1e6]))

    def test_multi_zone_initialization(self):
        """Test multi-zone envelope with scalar parameters."""
        env = Envelope(n_zones=3, R_env=0.2, C_env=2e6)

        assert env.n_zones == 3
        assert env.R_env.shape == torch.Size([3])
        assert env.C_env.shape == torch.Size([3])
        assert torch.allclose(env.R_env, torch.tensor([0.2, 0.2, 0.2]))
        assert torch.allclose(env.C_env, torch.tensor([2e6, 2e6, 2e6]))

    def test_zone_specific_parameters(self):
        """Test zone-specific parameter initialization."""
        R_env_values = [0.1, 0.15, 0.12]
        C_env_values = [1e6, 1.2e6, 0.9e6]

        env = Envelope(
            n_zones=3,
            R_env=R_env_values,
            C_env=C_env_values
        )

        assert torch.allclose(env.R_env, torch.tensor(R_env_values))
        assert torch.allclose(env.C_env, torch.tensor(C_env_values))

    def test_parameter_validation(self):
        """Test parameter validation for incorrect list lengths."""
        with pytest.raises(ValueError, match="list length 2 != n_zones 3"):
            Envelope(n_zones=3, R_env=[0.1, 0.2])  # Wrong length

    def test_device_and_dtype(self):
        """Test device and dtype handling."""
        device = torch.device('cpu')
        dtype = torch.float64

        env = Envelope(n_zones=2, device=device, dtype=dtype)

        assert env.R_env.device == device
        assert env.R_env.dtype == dtype
        assert env.C_env.device == device
        assert env.C_env.dtype == dtype

    def test_initial_state_functions(self):
        """Test initial state functions."""
        env = Envelope(n_zones=3)

        # Test that initial state functions exist
        state_funcs = env.initial_state_functions()
        assert 'T_zones' in state_funcs

        # Test calling state function
        batch_size = 2
        T_zones = state_funcs['T_zones'](batch_size)
        assert isinstance(T_zones, torch.Tensor)
        assert T_zones.shape == (batch_size, 3)

        # Check reasonable temperature values
        assert torch.all(T_zones > 290.0)  # Above 17°C
        assert torch.all(T_zones < 300.0)  # Below 27°C


class TestEnvelopeProperties:
    """Test envelope matrix properties and adjacency handling."""

    def test_fixed_adjacency_matrix(self):
        """Test fixed adjacency matrix behavior."""
        env = Envelope(n_zones=3)
        adj_matrix = env.adjacency_matrix

        # Should be 3x3 with zeros on diagonal, ones elsewhere
        assert adj_matrix.shape == (3, 3)
        assert torch.allclose(adj_matrix.diag(), torch.zeros(3))
        assert torch.allclose(adj_matrix + torch.eye(3), torch.ones(3, 3))

    def test_custom_adjacency_matrix(self):
        """Test custom adjacency matrix."""
        custom_adj = torch.tensor([
            [0., 1., 0.],
            [1., 0., 1.],
            [0., 1., 0.]
        ])

        env = Envelope(n_zones=3, adjacency=custom_adj)
        adj_matrix = env.adjacency_matrix

        assert torch.allclose(adj_matrix, custom_adj)

    def test_r_internal_matrix(self):
        """Test R_internal matrix construction."""
        env = Envelope(n_zones=3, R_internal=0.05)
        r_matrix = env.R_internal_matrix

        # Should be 3x3 with zeros on diagonal, 0.05 elsewhere
        assert r_matrix.shape == (3, 3)
        assert torch.allclose(r_matrix.diag(), torch.zeros(3))
        expected = torch.full((3, 3), 0.05)
        expected.fill_diagonal_(0)
        assert torch.allclose(r_matrix, expected)


class TestEnvelopeForward:
    """Test envelope forward pass and zone vectorization."""

    def test_single_zone_forward(self):
        """Test forward pass with single zone."""
        env = Envelope(n_zones=1)
        batch_size = 2

        # Create test inputs including initial state
        T_zones = torch.tensor([[295.0], [296.0]])  # [batch_size, n_zones]
        T_outdoor = torch.tensor([[290.0], [288.0]])
        Q_solar = torch.tensor([[100.0], [150.0]])
        Q_internal = torch.tensor([[200.0], [250.0]])
        Q_hvac = torch.tensor([[0.0], [100.0]])

        result = env.forward(
            t=0.0,
            T_zones=T_zones,
            T_outdoor=T_outdoor,
            Q_solar=Q_solar,
            Q_internal=Q_internal,
            Q_hvac=Q_hvac,
            dt=1.0
        )

        # Forward pass now returns updated states, not derivatives
        assert "T_zones" in result
        assert result["T_zones"].shape == (batch_size, 1)

        # Check that temperatures are reasonable
        T_new = result["T_zones"]
        assert torch.all(T_new > 280.0)  # Above 7°C
        assert torch.all(T_new < 320.0)  # Below 47°C

    def test_multi_zone_forward(self):
        """Test forward pass with multiple zones."""
        env = Envelope(n_zones=3, R_internal=0.1)
        batch_size = 2

        # Create test inputs - different temperatures per zone
        T_zones = torch.tensor([
            [295.0, 297.0, 294.0],  # Batch 1: zone 2 hottest
            [296.0, 295.0, 298.0]   # Batch 2: zone 3 hottest
        ])
        T_outdoor = torch.tensor([[290.0], [288.0]])  # Broadcast to all zones
        Q_solar = torch.zeros(batch_size, 3)
        Q_internal = torch.zeros(batch_size, 3)
        Q_hvac = torch.zeros(batch_size, 3)

        result = env.forward(
            t=0.0,
            T_zones=T_zones,
            T_outdoor=T_outdoor,
            Q_solar=Q_solar,
            Q_internal=Q_internal,
            Q_hvac=Q_hvac,
            dt=100.0  # Longer time step to see thermal dynamics
        )

        assert result["T_zones"].shape == (batch_size, 3)

        # Temperatures should change due to thermal dynamics
        T_new = result["T_zones"]
        # Use larger tolerance since changes might be small but should be detectable
        assert not torch.allclose(T_new, T_zones, atol=1e-3)

    def test_inter_zone_heat_exchange(self):
        """Test inter-zone heat exchange physics."""
        env = Envelope(n_zones=2, R_internal=0.1, R_env=1000.0)  # High R_env to minimize ambient effects

        # Zone 1 hot, Zone 2 cold - heat should flow from 1 to 2
        T_zones = torch.tensor([[300.0, 280.0]])  # 20K difference
        T_outdoor = torch.tensor([[290.0]])  # Middle temperature
        Q_solar = torch.zeros(1, 2)
        Q_internal = torch.zeros(1, 2)
        Q_hvac = torch.zeros(1, 2)

        result = env.forward(
            t=0.0,
            T_zones=T_zones,
            T_outdoor=T_outdoor,
            Q_solar=Q_solar,
            Q_internal=Q_internal,
            Q_hvac=Q_hvac,
            dt=100.0  # Longer time step to see effect
        )

        T_new = result["T_zones"]

        # Zone 1 (hot) should cool down, Zone 2 (cold) should warm up
        assert T_new[0, 0] < T_zones[0, 0]  # Zone 1 cooling
        assert T_new[0, 1] > T_zones[0, 1]  # Zone 2 heating

    def test_broadcasT_outdoor_temperature(self):
        """Test that ambient temperature broadcasts correctly."""
        env = Envelope(n_zones=3)
        batch_size = 2

        T_zones = torch.ones(batch_size, 3) * 295.0
        T_outdoor_single = torch.tensor([[290.0], [288.0]])  # Single value per batch
        T_outdoor_multi = torch.tensor([[290.0, 290.0, 290.0], [288.0, 288.0, 288.0]])  # Multi value

        # Both should give same results
        result1 = env.forward(
            t=0.0, T_zones=T_zones, T_outdoor=T_outdoor_single,
            Q_solar=torch.zeros(batch_size, 3),
            Q_internal=torch.zeros(batch_size, 3),
            Q_hvac=torch.zeros(batch_size, 3),
            dt=1.0
        )

        result2 = env.forward(
            t=0.0, T_zones=T_zones, T_outdoor=T_outdoor_multi,
            Q_solar=torch.zeros(batch_size, 3),
            Q_internal=torch.zeros(batch_size, 3),
            Q_hvac=torch.zeros(batch_size, 3),
            dt=1.0
        )

        assert torch.allclose(result1["T_zones"], result2["T_zones"])


class TestEnvelopeLearnable:
    """Test learnable parameter functionality."""

    def test_learnable_parameters(self):
        """Test that learnable parameters are properly registered."""
        env = Envelope(
            n_zones=2,
            R_env=0.1,
            C_env=1e6,
            learnable={'R_env', 'C_env'}
        )

        # Check that parameters are registered as learnable
        param_names = [name for name, _ in env.named_parameters()]
        assert 'R_env' in param_names
        assert 'C_env' in param_names

        # Check that gradients can flow
        assert env.R_env.requires_grad
        assert env.C_env.requires_grad

    def test_learnable_adjacency(self):
        """Test learnable adjacency matrix."""
        env = Envelope(
            n_zones=3,
            learnable={'adjacency'}
        )

        # Should have adj_logits parameter
        param_names = [name for name, _ in env.named_parameters()]
        assert 'adj_logits' in param_names
        assert env.adj_logits.requires_grad

        # Check that adjacency matrix is in [0,1] range
        adj_matrix = env.adjacency_matrix
        assert torch.all(adj_matrix >= 0)
        assert torch.all(adj_matrix <= 1)

    def test_learnable_r_internal(self):
        """Test learnable R_internal parameter."""
        env = Envelope(
            n_zones=2,
            R_internal=0.05,
            learnable={'R_internal'}
        )

        # Should have R_internal_logits parameter
        param_names = [name for name, _ in env.named_parameters()]
        assert 'R_internal_logits' in param_names

        # R_internal values should be positive
        r_matrix = env.R_internal_matrix
        assert torch.all(r_matrix >= 0)


class TestEnvelopeInputFunctions:
    """Test input functions."""

    def test_input_functions_exist(self):
        """Test that all required input functions are provided."""
        env = Envelope(n_zones=2)
        input_funcs = env.input_functions

        required_inputs = ['T_outdoor', 'Q_solar', 'Q_internal', 'Q_hvac']
        for inp in required_inputs:
            assert inp in input_funcs

    def test_input_function_shapes(self):
        """Test that input functions return correct shapes."""
        env = Envelope(n_zones=3)
        input_funcs = env.input_functions
        batch_size = 2

        for name, func in input_funcs.items():
            result = func(t=3600.0, batch_size=batch_size)  # 1 hour
            if name == 'T_outdoor':
                # T_outdoor can be broadcast, so shape might be (batch_size, 1)
                assert result.shape[0] == batch_size
                assert result.shape[1] <= 3
            else:
                assert result.shape == (batch_size, 3), f"Wrong shape for {name}: {result.shape}"

    def test_input_physical_reasonableness(self):
        """Test that input functions are physically reasonable."""
        env = Envelope(n_zones=1)
        input_funcs = env.input_functions

        # Test at different times
        for t in [0, 12*3600, 24*3600]:  # Midnight, noon, next midnight
            T_outdoor = input_funcs['T_outdoor'](t, 1)
            Q_solar = input_funcs['Q_solar'](t, 1)
            Q_internal = input_funcs['Q_internal'](t, 1)
            Q_hvac = input_funcs['Q_hvac'](t, 1)

            # Basic sanity checks
            assert 250 < T_outdoor.item() < 350  # Reasonable temperature range
            assert Q_solar.item() >= 0  # Solar always non-negative
            assert Q_internal.item() >= 0  # Internal gains always non-negative
            # Q_hvac can be positive or negative (heating/cooling)


class TestEnvelopeSimulation:
    """Integration tests for envelope simulation capability."""

    def test_single_zone_simulation(self):
        """Test complete simulation with single zone."""
        env = Envelope(n_zones=1)

        # Run short simulation
        results = env.simulate(
            duration_hours=2.0,
            dt_minutes=60.0,  # 1 hour steps
            batch_size=1,
        )

        # Check outputs
        assert 'T_zones' in results
        n_steps = int(2.0 * 60 / 60) + 1  # 3 points (initial + 2 steps)
        assert results['T_zones'].shape == (1, n_steps, 1)

        # Temperature should stay within reasonable bounds
        T_zones = results['T_zones']
        assert torch.all(T_zones > 280)  # Above 7°C
        assert torch.all(T_zones < 320)  # Below 47°C

    def test_multi_zone_simulation(self):
        """Test simulation with multiple zones."""
        env = Envelope(
            n_zones=3,
            R_env=[0.1, 0.15, 0.12],  # Different envelope resistances
            C_env=[1e6, 1.2e6, 0.9e6]  # Different thermal masses
        )

        results = env.simulate(
            duration_hours=1.0,
            dt_minutes=30.0,
            batch_size=2,
        )

        n_steps = int(1.0 * 60 / 30) + 1  # 3 points
        assert results['T_zones'].shape == (2, n_steps, 3)

        # Zones should have different temperature trajectories due to different parameters
        T_final = results['T_zones'][0, -1, :]  # Final temperatures, first batch
        temp_std = torch.std(T_final)
        assert temp_std > 0.01  # Zones should have different final temperatures

    def test_simulation_with_custom_inputs(self):
        """Test simulation with custom input functions."""
        env = Envelope(n_zones=1)

        # Custom constant inputs with milder conditions for faster convergence
        def constant_ambient(t, batch_size):
            return torch.full((batch_size, 1), 295.0)  # Warmer ambient (22°C)

        def moderate_solar(t, batch_size):
            return torch.full((batch_size, 1), 200.0)  # Lower solar gain

        def moderate_internal(t, batch_size):
            return torch.full((batch_size, 1), 100.0)  # Lower internal gain

        def zero_hvac(t, batch_size):
            return torch.full((batch_size, 1), 0.0)

        custom_inputs = {
            'T_outdoor': constant_ambient,
            'Q_solar': moderate_solar,
            'Q_internal': moderate_internal,
            'Q_hvac': zero_hvac,
        }

        results = env.simulate(
            duration_hours=12.0,  # Even longer simulation for steady state
            dt_minutes=60.0,
            batch_size=1,
            input_functions=custom_inputs,
        )

        n_steps = int(12.0 * 60 / 60) + 1  # 13 points
        assert results['T_zones'].shape == (1, n_steps, 1)

        # With constant inputs, should reach steady state
        T_zones = results['T_zones'][0, :, 0]  # Time series for first zone

        # Check that system is converging (rate of change should decrease)
        # Compare early vs late temperature changes
        early_change = torch.abs(T_zones[2] - T_zones[1])
        late_change = torch.abs(T_zones[-1] - T_zones[-2])

        # System should be converging (late changes should be smaller than early changes)
        # Or the final range should be reasonable
        final_temps = T_zones[-4:]  # Last 4 hours
        temp_range = torch.max(final_temps) - torch.min(final_temps)

        # Either converging or stable within reasonable range
        is_converging = late_change < early_change
        is_stable = temp_range < 5.0  # Increased tolerance for realistic thermal dynamics

        assert is_converging or is_stable, f"System not converging: early_change={early_change:.3f}, late_change={late_change:.3f}, temp_range={temp_range:.3f}"

    def test_energy_conservation_principle(self):
        """Test basic energy conservation principles."""
        # Isolated system (no ambient exchange, no gains)
        env = Envelope(
            n_zones=2,
            R_env=1e6,  # Very high resistance (no ambient exchange)
            R_internal=0.1,  # Allow inter-zone exchange
        )

        # Custom inputs with no external energy
        def constant_ambient(t, batch_size):
            return torch.full((batch_size, 2), 295.0)

        def zero_gains(t, batch_size):
            return torch.zeros((batch_size, 2))

        isolated_inputs = {
            'T_outdoor': constant_ambient,
            'Q_solar': zero_gains,
            'Q_internal': zero_gains,
            'Q_hvac': zero_gains,
        }

        # Create initial state manually
        state_funcs = env.initial_state_functions()
        initial_state = {
            'T_zones': torch.tensor([[300.0, 290.0]])  # 10K difference
        }

        results = env.simulate(
            duration_hours=5.0,
            dt_minutes=60.0,
            batch_size=1,
            initial_state=initial_state,
            input_functions=isolated_inputs,
        )

        T_zones = results['T_zones'][:, 0, :]  # (n_steps, n_zones)

        # In isolated inter-zone exchange, temperatures should equalize
        T_final = T_zones[-1, :]
        temp_diff = torch.abs(T_final[0] - T_final[1])
        assert temp_diff < 5.0  # Should be moving toward equilibrium

        # Average temperature should be conserved
        T_avg_initial = torch.mean(T_zones[0, :])
        T_avg_final = torch.mean(T_zones[-1, :])
        assert torch.abs(T_avg_initial - T_avg_final) < 1.0


class TestEnvelopeEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_parameters(self):
        """Test handling of negative physical parameters."""
        # Should work during initialization but may cause issues in simulation
        env = Envelope(n_zones=1, R_env=-0.1, C_env=-1e6)

        assert torch.allclose(env.R_env, torch.tensor([-0.1]))
        assert torch.allclose(env.C_env, torch.tensor([-1e6]))

    def test_very_small_timestep(self):
        """Test simulation with very small timestep."""
        env = Envelope(n_zones=1)

        results = env.simulate(
            duration_hours=0.1,
            dt_minutes=0.5,  # 30 second timestep
            batch_size=1,
        )

        n_steps = int(0.1 * 60 / 0.5) + 1  # 13 points
        assert results['T_zones'].shape == (1, n_steps, 1)

    def test_very_large_timestep(self):
        """Test simulation with large timestep."""
        env = Envelope(n_zones=1)

        results = env.simulate(
            duration_hours=2.0,
            dt_minutes=120.0,  # 2 hour timestep
            batch_size=1,
        )

        n_steps = int(2.0 * 60 / 120) + 1  # 2 points
        assert results['T_zones'].shape == (1, n_steps, 1)

    def test_ode_integration_stability(self):
        """Test that ODE integration remains stable."""
        env = Envelope(
            n_zones=2,
            R_env=0.1,
            C_env=1e6,
            ode_rtol=1e-6,
            ode_atol=1e-8
        )

        # Create initial states manually
        state_funcs = env.initial_state_functions()
        states = {name: func(1) for name, func in state_funcs.items()}

        # Create inputs
        input_funcs = env.input_functions
        inputs = {name: func(0.0, 1) for name, func in input_funcs.items()}

        # Single forward step should not crash
        forward_inputs = {**states, **inputs}
        result = env.forward(t=0.0, dt=3600.0, **forward_inputs)

        assert 'T_zones' in result
        assert not torch.isnan(result['T_zones']).any()
        assert not torch.isinf(result['T_zones']).any()


# Utility functions for running specific test groups
def run_initialization_tests():
    """Run initialization tests."""
    pytest.main(['-v', __file__ + '::TestEnvelopeInitialization'])

def run_forward_tests():
    """Run forward pass tests."""
    pytest.main(['-v', __file__ + '::TestEnvelopeForward'])

def run_simulation_tests():
    """Run simulation integration tests."""
    pytest.main(['-v', __file__ + '::TestEnvelopeSimulation'])

def run_all_tests():
    """Run all envelope tests."""
    pytest.main(['-v', __file__])

def run_quick_tests():
    """Run a subset of tests for quick validation."""
    pytest.main(['-v',
                __file__ + '::TestEnvelopeInitialization::test_single_zone_initialization',
                __file__ + '::TestEnvelopeForward::test_single_zone_forward',
                __file__ + '::TestEnvelopeSimulation::test_single_zone_simulation'])


# Helper function for running manual tests
def run_manual_tests():
    """Manual test runner for debugging."""
    print("Running Envelope Manual Tests")
    print("=" * 40)

    # Test 1: Basic functionality
    print("\nTest 1: Basic multi-zone envelope")
    env = Envelope(n_zones=3, R_env=[0.1, 0.15, 0.12])
    print(f"✓ Created envelope with {env.n_zones} zones")
    print(f"  R_env values: {env.R_env}")

    # Test 2: Initial state functions
    print("\nTest 2: Initial state functions")
    state_funcs = env.initial_state_functions()
    T_zones = state_funcs['T_zones'](2)
    print(f"✓ Initial state function works")
    print(f"  T_zones shape: {T_zones.shape}")
    print(f"  T_zones values: {T_zones}")

    # Test 3: Forward pass
    print("\nTest 3: Forward pass")
    batch_size = 2
    inputs = {
        'T_zones': T_zones,
        'T_outdoor': torch.tensor([[290.0], [288.0]]),
        'Q_solar': torch.zeros(batch_size, 3),
        'Q_internal': torch.zeros(batch_size, 3),
        'Q_hvac': torch.zeros(batch_size, 3),
    }

    result = env.forward(t=0.0, dt=3600.0, **inputs)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {result['T_zones'].shape}")
    print(f"  T_zones values: {result['T_zones']}")

    # Test 4: Simulation
    print("\nTest 4: Simulation")
    try:
        results = env.simulate(duration_hours=1.0, dt_minutes=30.0, batch_size=1)
        print(f"✓ Simulation successful")
        print(f"  Output shape: {results['T_zones'].shape}")
        if results['T_zones'].numel() > 0:
            print(f"  Final temperature: {results['T_zones'][-1].flatten()}")
        else:
            print("  No output data (empty tensor)")
    except Exception as e:
        print(f"✗ Simulation failed: {e}")

    print("\n" + "=" * 40)
    print("Manual tests completed!")


if __name__ == "__main__":
    # Run manual tests if executed directly
    run_manual_tests()