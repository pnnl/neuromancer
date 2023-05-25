from neuromancer.psl.norms import StandardScaler, normalize, denormalize
import torch
import numpy as np
from hypothesis import given, settings, strategies as st
from neuromancer.psl.nonautonomous import systems
import neuromancer.psl as pkg
import pytest

rand = [lambda shape: torch.rand(*shape), lambda shape: np.random.uniform(size=shape)]
systems = [v for v in systems.values()]
building_systems = [v for v in pkg.building_envelope.systems.values()]
auto_systems = [v for v in pkg.autonomous.systems.values()]


def get_stats(data):
    if len(data.shape) > 2:
        data = data.reshape(-1, data.shape[-1])
    stats = {'mean': data.mean(axis=0, keepdims=True), 'std': data.std(axis=0, keepdims=True)}
    print(stats)
    return stats


@given(st.lists(st.integers(1, 7), min_size=1, max_size=4),
       st.sampled_from(rand))
@settings(max_examples=100, deadline=None)
def test_data_shape(shape, rand):
    shape = [2] + list(shape)
    X = rand(shape)
    norm = StandardScaler(get_stats(X))
    Z = normalize(X, norm)
    Xprime = denormalize(Z, norm)
    assert Xprime.shape == X.shape


@given(st.lists(st.integers(1, 7), min_size=1, max_size=4),
       st.sampled_from(rand))
@settings(max_examples=1000, deadline=None)
def test_data_values(shape, rand):
    shape = [2] + list(shape)
    X = rand(shape)
    norm = StandardScaler(get_stats(X))
    Z = normalize(X, norm)
    Xprime = denormalize(Z, norm)
    if isinstance(Xprime, torch.Tensor):
        assert torch.allclose(Xprime, X, atol=1e-7, equal_nan=True), f'X: {X.mean()}, Xprime: {Xprime.mean()}'
    else:
        assert np.allclose(Xprime, X, atol=1e-7, equal_nan=True), f'X: {X.mean()}, Xprime: {Xprime.mean()}'


@given(st.lists(st.integers(1, 7), min_size=1, max_size=4),
       st.sampled_from(rand))
@settings(max_examples=1000, deadline=None)
def test_data_values2(shape, rand):
    shape = [2] + list(shape)
    X = rand(shape)
    norm = StandardScaler(get_stats(X))
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    else:
        X = torch.tensor(X)
    Z = normalize(X, norm)
    Xprime = denormalize(Z, norm)
    if isinstance(Xprime, torch.Tensor):
        assert torch.allclose(Xprime, X, atol=1e-7, equal_nan=True), f'X: {X.mean()}, Xprime: {Xprime.mean()}'
    else:
        assert np.allclose(Xprime, X, atol=1e-7, equal_nan=True), f'X: {X.mean()}, Xprime: {Xprime.mean()}'


@given(st.lists(st.integers(1, 7), min_size=1, max_size=4),
       st.sampled_from(rand))
@settings(max_examples=1000, deadline=None)
def test_data_type(shape, rand):
    shape = [2] + list(shape)
    X = rand(shape)
    norm = StandardScaler(get_stats(X))
    Z = normalize(X, norm)
    Xprime = denormalize(Z, norm)
    assert type(Z) is type(X)
    assert type(Xprime) is type(X)


@given(st.lists(st.integers(1, 7), min_size=1, max_size=4),
       st.sampled_from(rand))
@settings(max_examples=1000, deadline=None)
def test_data_dtype(shape, rand):
    shape = [2] + list(shape)
    X = rand(shape)
    norm = StandardScaler(get_stats(X))
    Z = normalize(X, norm)
    Xprime = denormalize(Z, norm)
    assert Z.dtype is X.dtype
    assert Xprime.dtype is X.dtype


@given(st.lists(st.integers(1, 7), min_size=1, max_size=4), st.sampled_from(rand))
@settings(max_examples=100, deadline=None)
def test_datadict_shape(shape, rand):
    shape = [2] + list(shape)
    d = {'X': rand(shape), 'Y': rand(shape), 'U': rand(shape), 'D': rand(shape), 'R': rand(shape)}
    norms = {k: StandardScaler(get_stats(v)) for k, v in d.items()}
    zd = normalize(d, norms)
    dprime = denormalize(zd, norms)
    assert all([xp.shape == x.shape for xp, x in zip(list(d.values()), list(dprime.values()))])
    assert all([xp.shape == x.shape for xp, x in zip(list(zd.values()), list(d.values()))])


@pytest.mark.parametrize("system", systems)
def test_emulator_integration(system):
    sys = system()
    data = sys.simulate()
    dataz = sys.normalize(data)
    dataprime = sys.denormalize(dataz)
    assert all([xp.shape == x.shape for xp, x in zip(list(data.values()), list(dataz.values()))])
    assert all([xp.shape == x.shape for xp, x in zip(list(data.values()), list(dataprime.values()))])
    for k in data:
        if isinstance(data[k], torch.Tensor):
            assert torch.allclose(data[k], dataprime[k], atol=1e-7)
    else:
        if isinstance(data[k], np.ndarray):
            assert np.allclose(data[k], dataprime[k], atol=1e-7)


@pytest.mark.parametrize("system", building_systems)
def test_emulator_integration2(system):
    sys = system()
    data = sys.simulate()
    dataz = sys.normalize(data)
    dataprime = sys.denormalize(dataz)
    assert all([xp.shape == x.shape for xp, x in zip(list(data.values()), list(dataz.values()))])
    assert all([xp.shape == x.shape for xp, x in zip(list(data.values()), list(dataprime.values()))])
    for k in data:
        if isinstance(data[k], torch.Tensor):
            assert torch.allclose(data[k], dataprime[k], atol=1e-7)
    else:
        if isinstance(data[k], np.ndarray):
            assert np.allclose(data[k], dataprime[k], atol=1e-7)


@pytest.mark.parametrize("system", auto_systems)
def test_emulator_integration3(system):
    sys = system()
    data = sys.simulate()
    dataz = sys.normalize(data)
    dataprime = sys.denormalize(dataz)
    assert all([xp.shape == x.shape for xp, x in zip(list(data.values()), list(dataz.values()))])
    assert all([xp.shape == x.shape for xp, x in zip(list(data.values()), list(dataprime.values()))])
    for k in data:
        if isinstance(data[k], torch.Tensor):
            assert torch.allclose(data[k], dataprime[k], atol=1e-7)
    else:
        if isinstance(data[k], np.ndarray):
            assert np.allclose(data[k], dataprime[k], atol=1e-7)


@pytest.mark.parametrize("system", auto_systems)
def test_emulator_integration4(system):
    sys = system(backend='torch')
    data = sys.simulate(nsim=5)
    data = {k: sys.B.core.stack([v]*3) for k, v in data.items()}
    dataz = sys.normalize(data)
    dataprime = sys.denormalize(dataz)
    assert all([xp.shape == x.shape for xp, x in zip(list(data.values()), list(dataz.values()))])
    assert all([xp.shape == x.shape for xp, x in zip(list(data.values()), list(dataprime.values()))])
    for k in data:
        if isinstance(data[k], torch.Tensor):
            assert torch.allclose(data[k], dataprime[k], atol=1e-7)
    else:
        if isinstance(data[k], np.ndarray):
            assert np.allclose(data[k], dataprime[k], atol=1e-7)

