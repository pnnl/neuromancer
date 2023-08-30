from hypothesis import given, settings, strategies as st
import pytest
from itertools import product
import torch, numpy as np
from neuromancer.psl.base import Backend
from neuromancer.psl.autonomous import systems as autosys
from neuromancer.psl.nonautonomous import systems as nonautosys
from neuromancer.psl.building_envelope import systems as buildsys
from neuromancer.psl.coupled_systems import systems as coupsys

SEED = 0

autosys = [v for v in autosys.values()]
nonautosys = [v for v in nonautosys.values()]
buildsys = [v for v in buildsys.values()]
coupsys = [v for v in coupsys.values()]

all_systems = autosys + nonautosys + buildsys + coupsys
all_systems_minus = autosys + nonautosys + buildsys

backends = list(Backend.backends)
backend_base_type = {
    "torch": torch.Tensor,
    "numpy": np.ndarray,
}


@pytest.mark.parametrize("emulator,backend", product(all_systems_minus, backends))
def test_normalize(emulator, backend):
    """
    Normalize(x) should not equal x
    Normalize from dict should produce same result as normalize from key
    Denormalize should invert normalize
    Output should be same type as input regardless of backend
    """
    if backend == 'torch':
        eq = torch.allclose
    else:
        eq = np.allclose
    system = emulator(backend=backend, set_stats=False, seed=SEED)
    if hasattr(system, 'U'):
        data = system.simulate(nsim=50, x0=system.x0, U=system.U[:51])
    else:
        data = system.simulate(nsim=50, x0=system.x0)
    system.set_stats(sim=data)
    for k in system.stats:
        system.stats[k]['std'] += 1e-5
    expected_type = backend_base_type[backend]
    x = system.simulate(5)['X']

    x_norm = system.normalize(x, key="X")
    assert isinstance(x_norm, expected_type)


    x_norm_dict = system.normalize({"X": x})
    assert isinstance(x_norm_dict["X"], expected_type)

    assert eq(x_norm, x_norm_dict["X"])
    assert not eq(x, x_norm)

    x_denorm = system.denormalize(x_norm, key="X")
    assert isinstance(x_denorm, expected_type)
    assert eq(x, x_denorm, rtol=1e-5, atol=1e-4)

    x_denorm_dict = system.denormalize(x_norm_dict)
    assert isinstance(x_denorm_dict["X"], expected_type)
    assert eq(x, x_denorm_dict["X"], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("emulator,backend", product(all_systems_minus, backends))
def test_get_x0_initial_value(emulator, backend):
    """
    When an object is instantiated first call of getters should always give same value
    """
    emulator_a = emulator(backend=backend, set_stats=False, seed=SEED)
    if hasattr(emulator_a, 'U'):
        data = emulator_a.simulate(nsim=5, U=emulator_a.U[:6], x0=emulator_a.x0)
    else:
        data = emulator_a.simulate(nsim=5, x0=emulator_a.x0)
    emulator_a.set_stats(sim=data)
    x0_a = emulator_a.get_x0()

    emulator_b = emulator(backend=backend, set_stats=False, seed=SEED)
    if hasattr(emulator_b, 'U'):
        data = emulator_b.simulate(nsim=5, U=emulator_b.U[:6], x0=emulator_b.x0)
    else:
        data = emulator_b.simulate(nsim=5, x0=emulator_b.x0)
    emulator_b.set_stats(sim=data)
    x0_b = emulator_b.get_x0()

    core = Backend.backends[backend]["core"]
    assert core.allclose(x0_a, x0_b)


@pytest.mark.parametrize("emulator,backend", product(all_systems_minus, backends))
def test_get_x0_backend_type(emulator, backend):
    """
    All getters should return datatype of backend
    """
    emulator = emulator(backend=backend, set_stats=False)
    if hasattr(emulator, 'U'):
        data = emulator.simulate(nsim=5, x0=emulator.x0, U=emulator.U[:6])
    else:
        data = emulator.simulate(nsim=5, x0=emulator.x0)
    emulator.set_stats(sim=data)
    x0 = emulator.get_x0()

    expected_type = backend_base_type[backend]
    assert isinstance(x0, expected_type)


@pytest.mark.parametrize("emulator,backend", product(all_systems_minus, backends))
def test_get_x0_shape(emulator, backend):
    """
    All get_x0 returns should be 1d arrays or tensors
    Shape of output = emulator.nx or emulator.nx0 attribute (if the class has one)
    """
    emulator = emulator(backend=backend, set_stats=False)
    if hasattr(emulator, 'U'):
        data = emulator.simulate(nsim=5, x0=emulator.x0, U=emulator.U[:6])
    else:
        data = emulator.simulate(nsim=5, x0=emulator.x0)
    emulator.set_stats(sim=data)
    x0 = emulator.get_x0()
    assert len(x0.shape) == 1
    assert x0.shape[0] == emulator.nx0


@pytest.mark.parametrize("emulator,backend", product(nonautosys + buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_U_initial_value(emulator, backend, nsim):
    """
    When an object is instantiated first call of getters should always give same value
    """
    emulator_a = emulator(backend=backend, set_stats=False, seed=SEED)
    data = emulator_a.simulate(nsim=nsim, U=emulator_a.U[:nsim + 1], x0=emulator_a.x0)
    emulator_a.set_stats(sim=data)
    U_a = emulator_a.get_U(nsim)

    emulator_b = emulator(backend=backend, set_stats=False, seed=SEED)
    data = emulator_b.simulate(nsim=nsim, U=emulator_b.U[:nsim + 1], x0=emulator_b.x0)
    emulator_b.set_stats(sim=data)
    U_b = emulator_b.get_U(nsim)

    core = Backend.backends[backend]["core"]
    assert core.allclose(U_a, U_b)


@pytest.mark.parametrize("emulator,backend", product(nonautosys + buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_U_backend_type(emulator, backend, nsim):
    """
    All getters should return datatype of backend
    """
    emulator = emulator(backend=backend, set_stats=False)
    data = emulator.simulate(nsim=nsim, U=emulator.U[:nsim + 1], x0=emulator.x0)
    emulator.set_stats(sim=data)
    U = emulator.get_U(nsim)

    expected_type = backend_base_type[backend]
    assert isinstance(U, expected_type)


@pytest.mark.parametrize("emulator,backend", product(nonautosys + buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_U_shape(emulator, backend, nsim):
    """
    For: NonAutonomous, BuildingEnvelope
    All get_U returns should be 2d arrays or tensors
    Shape of 2nd dim of output = emulator.nu or emulator.nU (if the class has one)
    """
    emulator = emulator(backend=backend, set_stats=False)
    data = emulator.simulate(nsim=nsim, U=emulator.U[:nsim + 1], x0=emulator.x0)
    emulator.set_stats(sim=data)
    U = emulator.get_U(nsim)
    assert len(U.shape) == 2
    U.shape[1] == emulator.nu


@pytest.mark.parametrize("emulator,backend", product(nonautosys + buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_R_initial_value(emulator, backend, nsim):
    """
    When an object is instantiated first call of getters should always give same value
    """
    emulator_a = emulator(backend=backend, set_stats=False, seed=SEED)
    data = emulator_a.simulate(nsim=nsim, U=emulator_a.U[:nsim + 1], x0=emulator_a.x0)
    emulator_a.set_stats(sim=data)
    R_a = emulator_a.get_R(nsim)

    emulator_b = emulator(backend=backend, set_stats=False, seed=SEED)
    data = emulator_b.simulate(nsim=nsim, U=emulator_b.U[:nsim + 1], x0=emulator_b.x0)
    emulator_b.set_stats(sim=data)
    R_b = emulator_b.get_R(nsim)

    core = Backend.backends[backend]["core"]
    assert core.allclose(R_a, R_b)


@pytest.mark.parametrize("emulator,backend", product(nonautosys + buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_R_backend_type(emulator, backend, nsim):
    """
    All getters should return datatype of backend
    """
    emulator = emulator(backend=backend, set_stats=False)
    data = emulator.simulate(nsim=nsim, U=emulator.U[:nsim + 1], x0=emulator.x0)
    emulator.set_stats(sim=data)
    R = emulator.get_R(nsim)
    expected_type = backend_base_type[backend]
    assert isinstance(R, expected_type)


@pytest.mark.parametrize("emulator,backend", product(nonautosys + buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_R_shape(emulator, backend, nsim):
    """
    For: NonAutonomous, BuildingEnvelope
    All get_R returns should be 2d arrays or tensors
    Shape of 2nd dim of output = emulator.nr or emulator.nR (if the class has one)
    """
    emulator = emulator(backend=backend, set_stats=False)
    data = emulator.simulate(nsim=nsim, U=emulator.U[:nsim+1], x0=emulator.x0)
    emulator.set_stats(sim=data)
    R = emulator.get_R(nsim)
    assert len(R.shape) == 2
    assert R.shape[1] == emulator.ny


@pytest.mark.parametrize("emulator,backend", product(buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_D_initial_value(emulator, backend, nsim):
    """
    When an object is instantiated first call of getters should always give same value
    """
    emulator_a = emulator(backend=backend, set_stats=False, seed=SEED)
    D_a = emulator_a.get_D(nsim)

    emulator_b = emulator(backend=backend, set_stats=False, seed=SEED)
    D_b = emulator_b.get_D(nsim)

    core = Backend.backends[backend]["core"]
    assert core.allclose(D_a, D_b)


@pytest.mark.parametrize("emulator,backend", product(buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_D_backend_type(emulator, backend, nsim):
    """
    All getters should return datatype of backend
    """
    emulator = emulator(backend=backend, set_stats=False)
    D = emulator.get_D(nsim)

    expected_type = backend_base_type[backend]
    assert isinstance(D, expected_type)


@pytest.mark.parametrize("emulator,backend", product(buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_D_obs_initial_value(emulator, backend, nsim):
    """
    When an object is instantiated first call of getters should always give same value
    """
    emulator_a = emulator(backend=backend, seed=SEED)
    D_obs_a = emulator_a.get_D_obs(nsim)

    emulator_b = emulator(backend=backend, seed=SEED)
    D_obs_b = emulator_b.get_D_obs(nsim)

    core = Backend.backends[backend]["core"]
    assert core.allclose(D_obs_a, D_obs_b)


@pytest.mark.parametrize("emulator,backend", product(buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_D_obs_backend_type(emulator, backend, nsim):
    """
    All getters should return datatype of backend
    """
    emulator = emulator(backend=backend, set_stats=False)
    D_obs = emulator.get_D_obs(nsim)

    expected_type = backend_base_type[backend]
    assert isinstance(D_obs, expected_type)


@pytest.mark.parametrize("emulator,backend", product(buildsys, backends))
@given(nsim=st.integers(3, 5))
@settings(max_examples=1, deadline=None)
def test_get_D_obs_shape(emulator, backend, nsim):
    """
    For: BuildingEnvelope
    1 feature (2d object with a 1 dim)
    """
    emulator = emulator(backend=backend, set_stats=False)
    D_obs = emulator.get_D_obs(nsim)

    assert len(D_obs.shape) == 2
    assert len(D_obs.squeeze().shape) == 1

