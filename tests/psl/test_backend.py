from neuromancer.psl import nonautonomous, autonomous, building_envelope
import numpy as np
import pytest

auto_systems = list(autonomous.systems.values())
nauto_systems = list(nonautonomous.systems.values())
building_systems = list(building_envelope.systems.values())


@pytest.mark.parametrize("system", nauto_systems)
def test_non_autonomous(system):
    sys2 = system(set_stats=False)
    sys3 = system(backend='torch', set_stats=False)
    data2 = sys2.simulate(nsim=2, x0=sys2.x0, U=sys2.U[:3])
    data3 = sys3.simulate(nsim=2, x0=sys3.x0, U=sys3.U[:3])
    assert np.isclose(data2['X'], data3['X'], rtol=1e-04, atol=1e-05).all(), f'{system} failed'


@pytest.mark.parametrize("system", auto_systems)
def test_autonomous(system):
    sys2 = system(set_stats=False)
    sys3 = system(backend='torch', set_stats=False)
    data2 = sys2.simulate(nsim=2, x0=sys2.x0)
    data3 = sys3.simulate(nsim=2, x0=sys3.x0)
    assert np.isclose(data2['X'], data3['X'], rtol=1e-04, atol=1e-05).all(), f'{system} failed'


@pytest.mark.parametrize("system", building_systems)
def test_building(system):
    sys2 = system(set_stats=False)
    sys3 = system(backend='torch', set_stats=False)
    data2 = sys2.simulate(nsim=2, x0=sys2.x0, U=sys2.U[:3], D=sys2._D[:2])
    data3 = sys3.simulate(nsim=2, x0=sys3.x0, U=sys3.U[:3], D=sys3._D[:2])
    assert np.isclose(data2['X'], data3['X'], rtol=1e-04, atol=1e-05).all(), f'{system} failed'